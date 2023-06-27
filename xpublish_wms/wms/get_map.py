import io
import logging
from datetime import datetime
from typing import List, Union

import cachey
import cf_xarray  # noqa
import datashader as dsh
import datashader.transfer_functions as tf
import datashader.utils as dshu
import matplotlib.cm as cm
import numpy as np
import pandas as pd
import xarray as xr
from fastapi.responses import StreamingResponse
from PIL import Image
from rasterio.enums import Resampling
from rasterio.transform import from_bounds
from scipy.spatial import Delaunay

from xpublish_wms.grid import GridType
from xpublish_wms.utils import to_lnglat, to_mercator

logger = logging.getLogger("uvicorn")


class GetMap:
    """
    TODO - Add docstring
    """

    TIME_CF_NAME: str = "time"
    ELEVATION_CF_NAME: str = "vertical"
    DEFAULT_CRS: str = "EPSG:3857"
    DEFAULT_STYLE: str = "raster/default"
    DEFAULT_PALETTE: str = "turbo"

    cache: cachey.Cache

    # Data selection
    parameter: str
    time: datetime = None
    has_time: bool
    elevation: float = None
    has_elevation: bool

    # Grid
    grid_type: GridType
    crs: str
    bbox = List[float]
    width: int
    height: int

    # Output style
    style: str
    colorscalerange: List[float]
    autoscale: bool

    def __init__(self, cache: cachey.Cache):
        self.cache = cache

    def get_map(self, ds: xr.Dataset, query: dict) -> StreamingResponse:
        """
        Return the WMS map for the dataset and given parameters
        """
        # Decode request params
        self.ensure_query_types(ds, query)

        # Select data according to request
        da = self.select_layer(ds)
        da = self.select_time(da)
        da = self.select_elevation(da)
        da = self.select_custom_dim(da)

        # Render the data using the render that matches the dataset type
        # The data selection and render are coupled because they are both driven by
        # The grid type for now. This can be revisited if we choose to interpolate or
        # use the contoured renderer for regular grid datasets
        image_buffer = io.BytesIO()
        render_result = self.render(da, image_buffer, False)
        if render_result:
            image_buffer.seek(0)

        return StreamingResponse(image_buffer, media_type="image/png")

    def get_minmax(self, ds: xr.Dataset, query: dict) -> dict:
        """
        Return the range of values for the dataset and given parameters
        """
        entire_layer = False
        if "bbox" not in query:
            # When BBOX is not specified, we are just going to slice the layer in time and elevation
            # and return the min and max values for the entire layer so bbox can just be the whoel world
            entire_layer = True
            query["bbox"] = "-180,-90,180,90"
            query["width"] = 1
            query["height"] = 1

        # Decode request params
        self.ensure_query_types(ds, query)

        # Select data according to request
        da = self.select_layer(ds)
        da = self.select_time(da)
        da = self.select_elevation(da)

        # Prepare the data as if we are going to render it, but instead grab the min and max
        # values from the data to represent the range of values in the given area
        if entire_layer:
            return {"min": float(da.min()), "max": float(da.max())}
        else:
            return self.render(da, None, minmax_only=True)

    def ensure_query_types(self, ds: xr.Dataset, query: dict):
        """
        Decode request params

        :param query:
        :return:
        """
        self.grid_type = GridType.from_ds(ds)

        # Data selection
        self.parameter = query["layers"]
        self.time_str = query.get("time", None)
        if self.time_str:
            self.time = pd.to_datetime(self.time_str).tz_localize(None)
        else:
            self.time = None
        self.has_time = "time" in ds[self.parameter].cf.coordinates

        self.elevation_str = query.get("elevation", None)
        if self.elevation_str:
            self.elevation = float(self.elevation_str)
        else:
            self.elevation = None
        self.has_elevation = "vertical" in ds[self.parameter].cf.coordinates

        # Grid
        self.crs = query.get("crs", None) or query.get("srs")
        self.bbox = [float(x) for x in query["bbox"].split(",")]
        self.width = int(query["width"])
        self.height = int(query["height"])

        # Output style
        self.style = query.get("styles", self.DEFAULT_STYLE)
        # Let user pick cm from here https://predictablynoisy.com/matplotlib/gallery/color/colormap_reference.html#sphx-glr-gallery-color-colormap-reference-py
        # Otherwise default to rainbow
        try:
            self.stylename, self.palettename = self.style.split("/")
        except Exception:
            self.stylename = "raster"
            self.palettename = "default"
        finally:
            if self.palettename == "default":
                self.palettename = self.DEFAULT_PALETTE

        self.colorscalerange = [
            float(x) for x in query.get("colorscalerange", "nan,nan").split(",")
        ]
        self.autoscale = query.get("autoscale", "false") == "true"

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
            da = da.cf.sel({self.TIME_CF_NAME: self.time}, method="nearest")
        else:
            da = da.cf.isel({self.TIME_CF_NAME: -1})

        return da

    def select_elevation(self, da: xr.DataArray) -> xr.DataArray:
        """
        Ensure elevation selection

        If elevation is provided :
            - use cf_xarray to access vertical coord
            - by default use ELEVATION_CF_NAME
            - method nearest to ensure at least one result

        Otherwise:
            - Get latest one

        :param da:
        :return:
        """
        if self.elevation is not None and self.has_elevation:
            da = da.cf.sel({self.ELEVATION_CF_NAME: self.elevation}, method="nearest")
        elif self.has_elevation:
            da = da.cf.isel({self.ELEVATION_CF_NAME: 0})

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
        for key in da.cf.coordinates.keys():
            if key in ("latitude", "longitude", "X", "Y"):
                continue

            coord = da.cf.coords[key]
            if coord.size > 1:
                da = da.cf.isel({key: -1})

        return da

    def render(
        self,
        da: xr.DataArray,
        buffer: io.BytesIO,
        minmax_only: bool,
    ) -> Union[bool, dict]:
        """
        Render the data array into an image buffer
        :param da:
        :return:
        """
        # For now, try to render everything as a quad grid
        # TODO: FVCOM and other grids
        return self.render_quad_grid(da, buffer, minmax_only)

    def render_quad_grid(
        self,
        da: xr.DataArray,
        buffer: io.BytesIO,
        minmax_only: bool,
    ) -> Union[bool, dict]:
        """
        Render the data array into an image buffer when the dataset is using a
        regular or staggered (ala ROMS) grid
        :param da:
        :return:
        """
        if self.crs == "EPSG:3857":
            bbox_lng, bbox_lat = to_lnglat.transform(
                [self.bbox[0], self.bbox[2]],
                [self.bbox[1], self.bbox[3]],
            )
            bbox_ll = [*bbox_lng, *bbox_lat]
        else:
            bbox_ll = [self.bbox[0], self.bbox[2], self.bbox[1], self.bbox[3]]

        if minmax_only:
            x = np.array(da.cf["longitude"].values)
            y = np.array(da.cf["latitude"].values)
            data = np.array(da.values)
            inds = np.where(
                (x >= (bbox_ll[0] - 0.18))
                & (x <= (bbox_ll[1] + 0.18))
                & (y >= (bbox_ll[2] - 0.18))
                & (y <= (bbox_ll[3] + 0.18)),
            )
            # x_sel = x[inds].flatten()
            # y_sel = y[inds].flatten()
            data_sel = data[inds].flatten()
            return {
                "min": float(np.nanmin(data_sel)),
                "max": float(np.nanmax(data_sel)),
            }

        cvs = dsh.Canvas(
            plot_height=self.height,
            plot_width=self.width,
            x_range=(bbox_ll[0], bbox_ll[1]),
            y_range=(bbox_ll[2], bbox_ll[3]),
        )

        if not self.autoscale:
            vmin, vmax = self.colorscalerange
        else:
            vmin, vmax = [None, None]

        im = tf.shade(
            cvs.quadmesh(da, x=da.cf.coords['longitude'].name, y=da.cf.coords['latitude'].name),
            cmap=cm.get_cmap(self.palettename),
            how="linear",
            span=(vmin, vmax),
        ).to_pil()
        im.save(buffer, format="PNG")

        return True
