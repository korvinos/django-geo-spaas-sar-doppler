import os, warnings
from math import sin, pi, cos, acos, copysign
import numpy as np
from scipy.ndimage.filters import median_filter

from dateutil.parser import parse
from datetime import timedelta

from django.conf import settings
from django.utils import timezone
from django.db import models
from django.contrib.gis.geos import WKTReader

from geospaas.utils import nansat_filename, media_path, product_path
from geospaas.vocabularies.models import Parameter
from geospaas.catalog.models import DatasetParameter, GeographicLocation
from geospaas.catalog.models import Dataset, DatasetURI
from geospaas.viewer.models import Visualization
from geospaas.viewer.models import VisualizationParameter
from geospaas.nansat_ingestor.managers import DatasetManager as DM

from nansat.nsr import NSR
from nansat.domain import Domain
from nansat.figure import Figure
from sardoppler.sardoppler import Doppler
from sar_doppler.errors import AlreadyExists


class DatasetManager(DM):

    def get_or_create(self, uri, reprocess=False, *args, **kwargs):
        # ingest file to db

        if DatasetURI.objects.filter(uri=uri):
            raise AlreadyExists

        ds, created = super(DatasetManager, self).get_or_create(uri, *args, **kwargs)

        if not type(ds) == Dataset:
            return ds, False

        # set Dataset entry_title
        ds.entry_title = 'SAR Doppler'
        ds.save()

        fn = nansat_filename(uri)
        n = Doppler(fn, subswath=0)
        gg = WKTReader().read(n.get_border_wkt())

        if ds.geographic_location.geometry.area>gg.area and not reprocess:
            return ds, False

        # Update dataset border geometry
        # This must be done every time a Doppler file is processed. It is time
        # consuming but apparently the only way to do it. Could be checked
        # though...

        n_subswaths = 5
        swath_data = {}
        lon = {}
        lat = {}
        astep = {}
        rstep = {}
        az_left_lon = {}
        ra_upper_lon = {}
        az_right_lon = {}
        ra_lower_lon = {}
        az_left_lat = {}
        ra_upper_lat = {}
        az_right_lat = {}
        ra_lower_lat = {}
        num_border_points = 10
        border = 'POLYGON(('

        for i in range(n_subswaths):
            # Read subswaths 
            swath_data[i] = Doppler(fn, subswath=i)

            # Should use nansat.domain.get_border - see nansat issue #166
            # (https://github.com/nansencenter/nansat/issues/166)
            lon[i], lat[i] = swath_data[i].get_geolocation_grids()

            astep[i] = max(1, (lon[i].shape[0] / 2 * 2 - 1) / num_border_points)
            rstep[i] = max(1, (lon[i].shape[1] / 2 * 2 - 1) / num_border_points)

            az_left_lon[i] = lon[i][0:-1:astep[i], 0]
            az_left_lat[i] = lat[i][0:-1:astep[i], 0]

            az_right_lon[i] = lon[i][0:-1:astep[i], -1]
            az_right_lat[i] = lat[i][0:-1:astep[i], -1]

            ra_upper_lon[i] = lon[i][-1, 0:-1:rstep[i]]
            ra_upper_lat[i] = lat[i][-1, 0:-1:rstep[i]]

            ra_lower_lon[i] = lon[i][0, 0:-1:rstep[i]]
            ra_lower_lat[i] = lat[i][0, 0:-1:rstep[i]]

        lons = np.concatenate((az_left_lon[0],  ra_upper_lon[0],
                               ra_upper_lon[1], ra_upper_lon[2],
                               ra_upper_lon[3], ra_upper_lon[4],
                               np.flipud(az_right_lon[4]), np.flipud(ra_lower_lon[4]),
                               np.flipud(ra_lower_lon[3]), np.flipud(ra_lower_lon[2]),
                               np.flipud(ra_lower_lon[1]), np.flipud(ra_lower_lon[0])))

        # apply 180 degree correction to longitude - code copied from
        # get_border_wkt...

        for ilon, llo in enumerate(lons):
            lons[ilon] = copysign(acos(cos(llo * pi / 180.)) / pi * 180,
                                  sin(llo * pi / 180.))

        lats = np.concatenate((az_left_lat[0], ra_upper_lat[0],
                               ra_upper_lat[1], ra_upper_lat[2],
                               ra_upper_lat[3], ra_upper_lat[4],
                               np.flipud(az_right_lat[4]), np.flipud(ra_lower_lat[4]),
                               np.flipud(ra_lower_lat[3]), np.flipud(ra_lower_lat[2]),
                               np.flipud(ra_lower_lat[1]), np.flipud(ra_lower_lat[0])))

        poly_border = ','.join(str(llo) + ' ' + str(lla) for llo, lla in zip(lons, lats))
        wkt = 'POLYGON((%s))' % poly_border
        new_geometry = WKTReader().read(wkt)

        # Get geolocation of dataset - this must be updated
        geoloc = ds.geographic_location
        # Check geometry, return if it is the same as the stored one
        if geoloc.geometry == new_geometry and not reprocess:
            return ds, False

        if geoloc.geometry != new_geometry:
            # Change the dataset geolocation to cover all subswaths
            geoloc.geometry = new_geometry
            geoloc.save()

        # Create data products
        mm = self.__module__.split('.')
        module = '%s.%s' % (mm[0], mm[1])
        mp = media_path(module, swath_data[i].fileName)
        ppath = product_path(module, swath_data[i].fileName)

        for i in range(n_subswaths):
            is_corrupted = False
            # Check if the file is corrupted
            try:
                inci = swath_data[i]['incidence_angle']
            #  TODO: What kind of exception ?
            except:
                is_corrupted = True
                continue

            # Add Doppler anomaly
            swath_data[i].add_band(array=swath_data[i].anomaly(), parameters={
                'wkv':
                'anomaly_of_surface_backwards_doppler_centroid_frequency_shift_of_radar_wave'
            })

            # Find matching NCEP forecast wind field
            wind = Dataset.objects.filter(
                    source__platform__short_name='NCEP-GFS',
                    time_coverage_start__range=[
                        parse(swath_data[i].get_metadata()['time_coverage_start'])
                        - timedelta(hours=3),
                        parse(swath_data[i].get_metadata()['time_coverage_start'])
                        + timedelta(hours=3)
                    ]
                )
            band_number = swath_data[i]._get_band_number({
                'standard_name': 'surface_backwards_doppler_centroid_frequency_shift_of_radar_wave',
                })

            pol = swath_data[i].get_metadata(bandID=band_number, key='polarization')

            if wind:
                dates = [w.time_coverage_start for w in wind]

                # TODO: Come back later (!!!)
                nearest_date = min(dates, key=lambda d:
                        abs(d-parse(swath_data[i].get_metadata()['time_coverage_start']).replace(tzinfo=timezone.utc)))

                fww = swath_data[i].wind_waves_doppler(
                        nansat_filename(wind[dates.index(nearest_date)].dataseturi_set.all()[0].uri),
                        pol
                    )

                swath_data[i].add_band(array=fww, parameters={
                    'wkv':
                    'surface_backwards_doppler_frequency_shift_of_radar_wave_due_to_wind_waves'
                })

                fdg, land_corr = swath_data[i].geophysical_doppler_shift(
                    wind=nansat_filename(wind[dates.index(nearest_date)].dataseturi_set.all()[0].uri)
                )

                # Estimate current by subtracting wind-waves Doppler
                theta = swath_data[i]['incidence_angle'] * np.pi / 180.
                vcurrent = -np.pi * (fdg - fww) / (112. * np.sin(theta))

                # Smooth...
                # vcurrent = median_filter(vcurrent, size=(3,3))
                swath_data[i].add_band(
                    array=vcurrent,
                    parameters={
                        'wkv': 'surface_radial_doppler_sea_water_velocity'
                    })
            else:
                fww = None
                fdg, land_corr = swath_data[i].geophysical_doppler_shift()

            swath_data[i].add_band(
                array=fdg,
                parameters={
                    'wkv': 'surface_backwards_doppler_frequency_shift_of_radar_wave_due_to_surface_velocity'
                })

            # Export data to netcdf
            print('Exporting %s (subswath %d)' % (swath_data[i].fileName, i))

            fn = os.path.join(ppath,
                              os.path.basename(swath_data[i].fileName).split('.')[0]
                              + 'subswath%d.nc' % i)

            origFile = swath_data[i].fileName

            try:
                swath_data[i].set_metadata(key='Originating file',
                                           value=origFile)
            except Exception as e:
                # TODO: Should it be here?
                warnings.warn('%s: BUG IN GDAL(?) - SHOULD BE CHECKED..' % e.message)

            swath_data[i].export(fileName=fn)
            ncuri = os.path.join('file://localhost', fn)
            new_uri, created = DatasetURI.objects.get_or_create(uri=ncuri,
                                                                dataset=ds)

            # Maybe add figures in satellite projection...
            # filename = 'satproj_fdg_subswath_%d.png'%i
            # swath_data[i].write_figure(os.path.join(mp, filename),
            #        bands='fdg', clim=[-60,60], cmapName='jet')

            #  Add figure to db
            
            # Reproject to leaflet projection
            xlon, xlat = swath_data[i].get_corners()
            d = Domain(NSR(3857),
                       '-lle %f %f %f %f -tr 1000 1000'
                       % (xlon.min(), xlat.min(), xlon.max(), xlat.max()))

            swath_data[i].reproject(d, eResampleAlg=1, tps=True)

            # Check if the reprojection failed
            try:
                inci = swath_data[i]['incidence_angle']

            # TODO: What kind of exception?
            except:
                is_corrupted = True
                warnings.warn('Could not read incidence angles - reprojection failed')
                continue

            # Visualizations of the following bands (short_names) are created
            # when ingesting data:
            ingest_creates = ['valid_doppler',
                              'valid_land_doppler',
                              'valid_sea_doppler',
                              'dca',
                              'fdg']
            if wind:
                ingest_creates.extend(['fww', 'Ur'])

            # (the geophysical doppler shift must later be added in a separate
            # manager method in order to estimate the range bias after
            # processing multiple files)
            for band in ingest_creates:
                filename = '%s_subswath_%d.png' % (band, i)
                # check uniqueness of parameter
                param = Parameter.objects.get(short_name=band)
                fig = swath_data[i].write_figure(
                    os.path.join(mp, filename),
                    bands=band,
                    mask_array=swath_data[i]['swathmask'],
                    mask_lut={0: [128, 128, 128]},
                    transparency=[128, 128, 128])

                if type(fig) == Figure:
                    print 'Created figure of subswath %d, band %s' % (i, band)
                else:
                    warnings.warn('Figure NOT CREATED')

                # Get DatasetParameter
                dsp, created = DatasetParameter.objects.get_or_create(dataset=ds,
                                                                      parameter=param)

                # Create Visualization
                try:
                    geom, created = GeographicLocation.objects.get_or_create(
                        geometry=WKTReader().read(swath_data[i].get_border_wkt()))
                except Exception as inst:
                    print(type(inst))
                    import ipdb
                    ipdb.set_trace()
                    raise

                vv, created = Visualization.objects.get_or_create(
                    uri='file://localhost%s/%s' % (mp, filename),
                    title='%s (swath %d)' % (param.standard_name, i + 1),
                    geographic_location=geom
                )

                # Create VisualizationParameter
                vp, created = VisualizationParameter.objects.get_or_create(
                    visualization=vv,
                    ds_parameter=dsp
                )

        return ds, not is_corrupted
