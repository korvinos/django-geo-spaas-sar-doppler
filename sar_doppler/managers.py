import os, warnings
from math import sin, pi, cos, acos, copysign
import numpy as np
from scipy.ndimage.filters import median_filter

from dateutil.parser import parse
from datetime import timedelta

from django.conf import settings
from django.utils import timezone
from django.db import models
from django.contrib.gis.geos import WKTReader, Polygon

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

    DOMAIN = 'file://localhost'
    NUM_SUBSWATS = 5
    NUM_BORDER_POINTS = 10
    WKV_NAME = {
        'dc_anomaly': 'anomaly_of_surface_backwards_doppler_centroid_frequency_shift_of_radar_wave',
        'dc_wind': 'surface_backwards_doppler_frequency_shift_of_radar_wave_due_to_wind_waves',
        'radial_velocity': 'surface_radial_doppler_sea_water_velocity',
        'dc_velocity': 'surface_backwards_doppler_frequency_shift_of_radar_wave_due_to_surface_velocity'
    }

    def list_of_coordinates(self, left, right, upper, lower, axis):
        coord_list = np.concatenate(
            (left[axis][0], upper[axis][0], upper[axis][1], upper[axis][2],
             upper[axis][3], upper[axis][4], np.flipud(right[axis][4]),
             np.flipud(lower[axis][4]), np.flipud(lower[axis][3]),
             np.flipud(lower[axis][2]), np.flipud(lower[axis][1]),
             np.flipud(lower[axis][0])))

        return coord_list

    def find_wind(self, swath):
        """
        Find matching NCEP forecast wind field
        :param swath:
        :return:
        """
        wind = Dataset.objects.filter(
            source__platform__short_name='NCEP-GFS',
            time_coverage_start__range=[
                parse(swath.get_metadata()['time_coverage_start']) - timedelta(hours=3),
                parse(swath.get_metadata()['time_coverage_start']) + timedelta(hours=3)
            ]
        )
        return wind

    def get_wind(self, swath, wind):
        # Find band number of surface_backwards_doppler_centroid_frequency_shift_of_radar_wave
        band_number = swath._get_band_number({'short_name': 'dc'})
        # Get information about polarization from doppler centroid band
        polarization = swath.get_metadata(bandID=band_number, key='polarization')

        dates = [w.time_coverage_start for w in wind]
        # TODO: Come back later (!!!)
        nearest_date = min(dates, key=lambda d:
            abs(d - parse(swath.get_metadata()['time_coverage_start']).replace(tzinfo=timezone.utc)))

        wind_uri = nansat_filename(wind[dates.index(nearest_date)].dataseturi_set.all()[0].uri)
        fww = swath.wind_waves_doppler(wind_uri, polarization)
        fdg, land_corr = swath.geophysical_doppler_shift(wind=wind_uri)
        # Estimate current by subtracting wind-waves Doppler
        theta = swath['incidence_angle'] * np.pi / 180.
        current_velocity = -np.pi * (fdg - fww) / (112. * np.sin(theta))

        return fww, fdg, current_velocity

    def create_visualization(self, swath_data, swath_num, band, mp, ds):
        filename = '%s_subswath_%d.png' % (band, swath_num)
        # check uniqueness of parameter
        param = Parameter.objects.get(short_name=band)
        fig = swath_data.write_figure(
            os.path.join(mp, filename),
            bands=band,
            mask_array=swath_data['swathmask'],
            mask_lut={0: [128, 128, 128]},
            transparency=[128, 128, 128]
        )

        # TODO: Is it really possible?
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
                geometry=WKTReader().read(swath_data.get_border_wkt()))
        except Exception as inst:
            print(type(inst))

        vv, created = Visualization.objects.get_or_create(
            uri='file://localhost%s/%s' % (mp, filename),
            title='%s (swath %d)' % (param.standard_name, swath_num + 1),
            geographic_location=geom
        )

        # Create VisualizationParameter
        vp, created = VisualizationParameter.objects.get_or_create(
            visualization=vv,
            ds_parameter=dsp
        )

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
        # TODO: It will be ever swath #1 so should we specify?
        n = Doppler(fn, subswath=0)
        gg = WKTReader().read(n.get_border_wkt())

        if ds.geographic_location.geometry.area > gg.area and not reprocess:
            return ds, False

        # Update dataset border geometry
        # This must be done every time a Doppler file is processed. It is time
        # consuming but apparently the only way to do it. Could be checked
        # though...

        swath_data = {}
        # Dictionaries for accumulation of lat/lon for each subswat
        # <subswat #> : <array>
        lon = {}
        lat = {}

        astep = {}
        rstep = {}

        # Dictionaries for accumulation of image border coordinates
        left_border = {'lat': {}, 'lon': {}}    # Azimuthal direction
        right_border = {'lat': {}, 'lon': {}}   # Azimuthal direction
        upper_border = {'lat': {}, 'lon': {}}   # Range direction
        lower_border = {'lat': {}, 'lon': {}}   # Range direction

        # Flag for detection of corruption in img
        not_corrupted = True

        for i in range(self.NUM_SUBSWATS):
            # Read subswaths
            # TODO: Should we really keep it in memory?
            swath_data[i] = Doppler(fn, subswath=i)

            # Should use nansat.domain.get_border - see nansat issue #166
            # (https://github.com/nansencenter/nansat/issues/166)
            lon[i], lat[i] = swath_data[i].get_geolocation_grids()

            astep[i] = max(1, (lon[i].shape[0] / 2 * 2 - 1) / self.NUM_BORDER_POINTS)
            rstep[i] = max(1, (lon[i].shape[1] / 2 * 2 - 1) / self.NUM_BORDER_POINTS)

            left_border['lon'][i] = lon[i][0:-1:astep[i], 0]
            left_border['lat'][i] = lat[i][0:-1:astep[i], 0]

            right_border['lon'][i] = lon[i][0:-1:astep[i], -1]
            right_border['lat'][i] = lat[i][0:-1:astep[i], -1]

            upper_border['lon'][i] = lon[i][-1, 0:-1:rstep[i]]
            upper_border['lat'][i] = lat[i][-1, 0:-1:rstep[i]]

            lower_border['lon'][i] = lon[i][0, 0:-1:rstep[i]]
            lower_border['lat'][i] = lat[i][0, 0:-1:rstep[i]]

        lons = self.list_of_coordinates(left_border, right_border, upper_border, lower_border, 'lon')
        # apply 180 degree correction to longitude - code copied from
        # get_border_wkt...

        # TODO: This loop returns exactly the same list of lons
        for ilon, llo in enumerate(lons):
            lons[ilon] = copysign(acos(cos(llo * pi / 180.)) / pi * 180, sin(llo * pi / 180.))

        lats = self.list_of_coordinates(left_border, right_border, upper_border, lower_border, 'lat')

        # Create a polygon form lats and lons
        new_geometry = Polygon(zip(lons, lats))

        # Get geolocation of dataset - this must be updated
        geoloc = ds.geographic_location
        # Check geometry, return if it is the same as the stored one
        if geoloc.geometry == new_geometry and not reprocess:
            return ds, True

        if geoloc.geometry != new_geometry:
            # Change the dataset geolocation to cover all subswaths
            geoloc.geometry = new_geometry
            geoloc.save()

        # Create data products
        mm = self.__module__.split('.')
        module = '%s.%s' % (mm[0], mm[1])
        mp = media_path(module, swath_data[i].fileName)
        ppath = product_path(module, swath_data[i].fileName)

        # TODO: Go to the one loop with several methods
        for i in range(self.NUM_SUBSWATS):
            # Check if the file is corrupted
            try:
                inci = swath_data[i]['incidence_angle']
            #  TODO: What kind of exception ? KeyError
            except:
                not_corrupted = False
                continue

            # Add Doppler anomaly
            swath_data[i].add_band(array=swath_data[i].anomaly(),
                                   parameters={'wkv': self.WKV_NAME['dc_anomaly']})

            # Find matching NCEP forecast wind field
            wind = self.find_wind(swath_data[i])
            if wind:
                fww, fdg, current_velocity = self.get_wind(swath_data[i])
                swath_data[i].add_band(array=fww,
                                       parameters={'wkv': self.WKV_NAME['dc_wind']})

                swath_data[i].add_band(array=current_velocity,
                                       parameters={'wkv': self.WKV_NAME['radial_velocity']})
            else:
                fdg, land_corr = swath_data[i].geophysical_doppler_shift()

            swath_data[i].add_band(array=fdg,
                                   parameters={'wkv': self.WKV_NAME['dc_velocity']})

            # Export data to netcdf
            self.export(swath_data[i], i, ppath, ds)

            #  Add figure to db
            
            # Reproject to leaflet projection
            xlon, xlat = swath_data[i].get_corners()
            d = Domain(NSR(3857), '-lle %f %f %f %f -tr 1000 1000'
                       % (xlon.min(), xlat.min(), xlon.max(), xlat.max()))

            swath_data[i].reproject(d, eResampleAlg=1, tps=True)

            # Check if the reprojection failed
            try:
                inci = swath_data[i]['incidence_angle']

            # TODO: What kind of exception?
            except:
                not_corrupted = False
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
                self.create_visualization(swath_data[i], i, band, mp, ds)

        return ds, not_corrupted

    def export(self, swath_data, swath_num, ppath, dataset):
        """
        Export data to netcdf
        :param swath_data:
        :param swath_num:
        :param ppath:
        :param dataset:
        :return: None
        """
        orig_file = swath_data.fileName
        print('Exporting %s (subswath %d)' % (orig_file, swath_num))
        # TODO: Correct a file name pateern
        file_name = os.path.join(ppath, os.path.basename(orig_file).split('.')[0] + 'subswath%d.nc' % swath_num)

        try:
            swath_data.set_metadata(key='Originating file', value=orig_file)
        except Exception as e:
            # TODO: Should it be here?
            warnings.warn('%s: BUG IN GDAL(?) - SHOULD BE CHECKED..' % e.message)

        swath_data.export(fileName=file_name)
        ncuri = os.path.join(self.DOMAIN, file_name)
        new_uri, created = DatasetURI.objects.get_or_create(uri=ncuri, dataset=dataset)

