import io
import time
import logging
from datetime import datetime
from typing import List

import cachey
import cf_xarray  # noqa
import xarray as xr
import pandas as pd
import numpy as np
from fastapi.responses import StreamingResponse
from rasterio.enums import Resampling
from rasterio.transform import from_bounds
from PIL import Image
from matplotlib import cm
from pykdtree.kdtree import KDTree

from xpublish_wms.utils import to_lnglat

logger = logging.getLogger(__name__)


class OgcWmsGetMap:
    TIME_CF_NAME: str = "time"
    DEFAULT_CRS: str = "4326"

    cache: cachey.Cache

    # Data selection
    parameter : str
    time: datetime = None

    # Grid
    crs: str
    bbox = List[float]
    width: int
    height: int

    # Output style
    style: str
    colorscalerange: List[float]
    colorbaronly: bool
    autoscale: bool

    def get_map(self, ds: xr.Dataset, query: dict):
        """
        Return the WMS map for the dataset and given parameters
        """
        # Decode request params
        self.ensure_query_types(query)

        # Select data according to request
        da = self.select_layer(ds)
        da = self.select_time(da)
        da = self.select_custom_dim(da)
        da_bbox = self.select_grid(da)

        # Generate output
        image_bytes = self.draw(da, da_bbox)

        return StreamingResponse(image_bytes, media_type="image/png")

    def ensure_query_types(self, query: dict):
        """
        Decode request params

        :param query:
        :return:
        """
        # Data selection
        self.parameter = query['layers']
        time_str = query.get('time', None)
        if time_str:
            self.time = pd.to_datetime(time_str).tz_localize(None)

        # Grid
        self.crs = query.get('crs', None) or query.get('srs')
        self.bbox = [float(x) for x in query['bbox'].split(',')]
        self.width = int(query['width'])
        self.height = int(query['height'])

        # Output style
        self.style = query.get('styles', "default")
        self.colorscalerange = [float(x) for x in query.get('colorscalerange', 'nan,nan').split(',')]
        self.autoscale = query.get('autoscale', "true") == "true"

    def select_layer(self, ds: xr.Dataset) -> xr.DataArray:
        """
        Select Dataset variable, according to WMS layers request
        :param ds:
        :return:
        """
        return ds[self.parameter]

    def select_time(self, da: xr.DataArray) -> xr.DataArray:
        """
        Ensure time selection

        If time is provided :
            - use cf_xarray to access time
            - by default use TIME_CF_NAME
            - method nearest to ensure at least one result

        Otherwise:
            - Get latest one

        :param da:
        :return:
        """
        if self.time is not None:
            da = da.cf.sel(
                {self.TIME_CF_NAME: self.time},
                method="nearest"
            )
        else:
            da = da.cf.isel({
                self.TIME_CF_NAME: -1
            })

        return da

    def select_custom_dim(self, da: xr.DataArray) -> xr.DataArray:
        """
        Select other dimension, ensuring a 2D array
        :param da:
        :return:
        """
        # Squeeze single value dimensions
        da = da.squeeze()

        # TODO: Filter dimension from custom query, if any

        # Squeeze multiple values dimensions, by selecting the last value
        for coord_name in da.cf.coords:
            if coord_name in ("latitude", "longitude"):
                continue
            coord = da.cf.coords[coord_name]
            if coord.size > 1:
                da = da.cf.isel({coord_name: -1})

        return da

    def select_grid(self, da: xr.DataArray) -> xr.DataArray:
        """
        Select grid and reproject if needed
        :param da:
        :return:
        """
        # Some basic check for dataset
        if not da.rio.crs:
            da = da.rio.write_crs(self.DEFAULT_CRS)

        if self.grid_is_regular(da):
            da = self.select_regular_grid(da)
        else:
            da = self.select_irregular_grid(da)
        return da

    def grid_is_regular(self, da: xr.DataArray) -> bool:
        return da.cf.coords['longitude'].dims[0] == da.cf.coords['longitude'].name

    def select_regular_grid(self, da: xr.DataArray) -> xr.DataArray:
        """
        Filter regular grid according to WMS request :
            - bbox
            - width and height
        :param da:
        :return:
        """
        # Regular grid
        miny, minx, maxy, maxx = self.bbox

        # TBD:
        #   got issues on my dataset using rioxarray :
        #       The WKT could not be parsed. OGR Error code 5
        #   use xr.sel as backup

        #clipped = da.rio.clip_box(*self.bbox, crs=self.crs)
        #resampled_data = clipped.rio.reproject(
        #    dst_crs=self.crs,
        #    shape=(self.width, self.height),
        #    resampling=Resampling.nearest,
        #    transform=from_bounds(*self.bbox, width=self.width, height=self.height),
        #)
        #

        longitude = np.linspace(minx, maxx, self.width)
        latitude = np.linspace(miny, maxy, self.height)

        if da.longitude[0] > da.longitude[-1]:
            longitude = np.flip(longitude)
        if da.latitude[0] > da.latitude[-1]:
            latitude = np.flip(latitude)

        resampled_data = da.cf.interp(
            latitude=latitude, longitude=longitude,
            method="nearest"
        )

        return resampled_data

    def select_irregular_grid(self, da: xr.DataArray) -> xr.DataArray:
        """
        Filter irregular grid according to WMS request :
            - bbox
            - width and height
        :param da:
        :return:
        """
        bbox = self.bbox

        start = time.time()

        # irregular grid
        t_lat, t_lng = to_lnglat.transform([bbox[0], bbox[2]], [bbox[1], bbox[3]])
        lats = np.linspace(t_lng[0], t_lng[1], self.width)
        lngs = np.linspace(t_lat[0], t_lat[1], self.height)
        grid_lngs, grid_lats = np.meshgrid(lngs, lats)
        pts = np.column_stack((grid_lngs.ravel(), grid_lats.ravel()))
        kd = get_spatial_kdtree(da, self.cache)
        _, n = kd.query(pts)
        ni = n.argsort()
        pp = n[ni]

        index_time = time.time()
        logger.info(f'index and kdtree irregular: {index_time - start}')

        # This is slow because it has to pull into numpy array, can we do better?
        z = da.zeta[0][pp].values
        z = z[ni.argsort()].reshape((self.height, self.width))

        extraction_time = time.time()
        logger.info(f'extract data irregular: {extraction_time - index_time}')

        rds = xr.Dataset(
            data_vars=dict(
                z=(["y", "x"], z),
            ),
            coords=dict(
                x=(["x"], lngs),
                y=(["y"], lats),
            )
        )
        rds.rio.write_crs(4326, inplace=True)
        resampled_data = rds.z.rio.reproject(
            dst_crs=self.crs,
            shape=(self.width, self.height),
            resampling=Resampling.nearest,
            transform=from_bounds(*bbox, width=self.width, height=self.height),
        )

        reproject_time = time.time()
        logger.info(f'clip and reproject irregular: {reproject_time - extraction_time}')
        return resampled_data

    def draw(self, da: xr.DataArray, da_bbox: xr.DataArray) -> io.BytesIO:
        """
        Generate drawing, could be easily overriden

        :param da:
        :param da_bbox:
        :return:
        """
        da_scaled = self.draw_pil_get_colormap_scaled_data(da, da_bbox)
        return self.draw_pil_generate_map(da_scaled)

    def draw_pil_get_colormap_scaled_data(self, da: xr.DataArray, da_bbox: xr.DataArray) -> xr.DataArray:
        """
        Generate numpy array from our datasset, ensuring colormap is computed from the
        non-clipped data
        :param da:
        :param da_bbox:
        :return:
        """
        if self.autoscale:
            min_value = float(da.min())
            max_value = float(da.max())
        else:
            min_value = self.colorscalerange[0]
            max_value = self.colorscalerange[1]

        return (da_bbox - min_value) / (max_value - min_value)

    def draw_pil_generate_map(self, da: xr.DataArray) -> io.BytesIO:
        """
        Draw as PIL.Image
        :param da:
        :return:
        """
        stylename, palettename = self.style.split('/')

        # Let user pick cm from here https://predictablynoisy.com/matplotlib/gallery/color/colormap_reference.html#sphx-glr-gallery-color-colormap-reference-py
        # Otherwise default to rainbow
        if palettename == "default":
            palettename = "rainbow"
        im = Image.fromarray(np.uint8(cm.get_cmap(palettename)(da) * 255))

        image_bytes = io.BytesIO()
        im.save(image_bytes, format='PNG')
        image_bytes.seek(0)

        return image_bytes


def get_spatial_kdtree(ds: xr.Dataset, cache: cachey.Cache) -> KDTree:
    cache_key = f"dataset-kdtree-{ds.attrs['title']}"
    kd = cache.get(cache_key)
    if kd:
        return kd

    lng = ds.cf['longitude']
    lat = ds.cf['latitude']

    verts = np.column_stack((lng, lat))
    kd = KDTree(verts)

    cache.put(cache_key, kd, 5)

    return kd
