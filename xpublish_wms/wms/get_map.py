import io
import logging
import time
from datetime import datetime
from typing import List, Union

import cachey
import cf_xarray  # noqa
import datashader as dsh
import datashader.transfer_functions as tf
import matplotlib.cm as cm
import numpy as np
import pandas as pd
import rioxarray  # noqa
import xarray as xr
from fastapi.responses import StreamingResponse

from xpublish_wms.grid import RenderMethod
from xpublish_wms.utils import parse_float

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
        da = self.select_elevation(ds, da)
        # da = self.select_custom_dim(da)

        # Render the data using the render that matches the dataset type
        # The data selection and render are coupled because they are both driven by
        # The grid type for now. This can be revisited if we choose to interpolate or
        # use the contoured renderer for regular grid datasets
        image_buffer = io.BytesIO()
        render_result = self.render(ds, da, image_buffer, False)
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
        da = self.select_elevation(ds, da)

        # Prepare the data as if we are going to render it, but instead grab the min and max
        # values from the data to represent the range of values in the given area
        if entire_layer:
            return {"min": float(da.min()), "max": float(da.max())}
        else:
            return self.render(ds, da, None, minmax_only=True)

    def ensure_query_types(self, ds: xr.Dataset, query: dict):
        """
        Decode request params

        :param query:
        :return:
        """
        # Data selection
        self.parameter = query["layers"]
        self.time_str = query.get("time", None)

        if self.time_str:
            self.time = pd.to_datetime(self.time_str).tz_localize(None)
        else:
            self.time = None
        self.has_time = self.TIME_CF_NAME in ds[self.parameter].cf.coords

        self.elevation_str = query.get("elevation", None)
        if self.elevation_str:
            self.elevation = float(self.elevation_str)
        else:
            self.elevation = None
        self.has_elevation = self.ELEVATION_CF_NAME in ds[self.parameter].cf.coords

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
            parse_float(x) for x in query.get("colorscalerange", "nan,nan").split(",")
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
        time_dim = da.cf.coordinates.get(self.TIME_CF_NAME, None)
        if time_dim is not None and len(time_dim):
            time_dim = time_dim[0]

        if not time_dim or time_dim not in list(da.dims):
            return da

        if self.time is not None:
            da = da.cf.sel({self.TIME_CF_NAME: self.time}, method="nearest")
        elif self.TIME_CF_NAME in da.cf.coords:
            da = da.cf.isel({self.TIME_CF_NAME: -1})

        return da

    def select_elevation(self, ds: xr.Dataset, da: xr.DataArray) -> xr.DataArray:
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
        da = ds.gridded.select_by_elevation(da, [self.elevation])

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
        ds: xr.Dataset,
        da: xr.DataArray,
        buffer: io.BytesIO,
        minmax_only: bool,
    ) -> Union[bool, dict]:
        """
        Render the data array into an image buffer
        """
        # For now, try to render everything as a quad grid
        # TODO: FVCOM and other grids
        # return self.render_quad_grid(da, buffer, minmax_only)
        projection_start = time.time()
        da = ds.gridded.project(da, self.crs)
        logger.debug(f"Projection time: {time.time() - projection_start}")

        if minmax_only:
            da = da.persist()
            x = np.array(da.x.values)
            y = np.array(da.y.values)
            data = np.array(da.values)
            inds = np.where(
                (x >= (self.bbox[0] - 0.18))
                & (x <= (self.bbox[2] + 0.18))
                & (y >= (self.bbox[1] - 0.18))
                & (y <= (self.bbox[3] + 0.18)),
            )
            # x_sel = x[inds].flatten()
            # y_sel = y[inds].flatten()
            data_sel = data[inds].flatten()
            return {
                "min": float(np.nanmin(data_sel)),
                "max": float(np.nanmax(data_sel)),
            }

        if not self.autoscale:
            vmin, vmax = self.colorscalerange
        else:
            vmin, vmax = [None, None]

        start_dask = time.time()
        da.persist()
        da.y.persist()
        da.x.persist()
        logger.debug(f"dask compute: {time.time() - start_dask}")

        start_shade = time.time()
        cvs = dsh.Canvas(
            plot_height=self.height,
            plot_width=self.width,
            x_range=(self.bbox[0], self.bbox[2]),
            y_range=(self.bbox[1], self.bbox[3]),
        )

        # Squeeze single value dimensions
        da = da.squeeze()

        if ds.gridded.render_method == RenderMethod.Quad:
            mesh = cvs.quadmesh(
                da,
                x="x",
                y="y",
            )
        elif ds.gridded.render_method == RenderMethod.Triangle:
            triangles = ds.gridded.tessellate(da)
            verts = pd.DataFrame({"x": da.x, "y": da.y, "z": da})
            tris = pd.DataFrame(triangles.astype(int), columns=["v0", "v1", "v2"])

            mesh = cvs.trimesh(
                verts,
                tris,
            )

        shaded = tf.shade(
            mesh,
            cmap=cm.get_cmap(self.palettename),
            how="linear",
            span=(vmin, vmax),
        )
        logger.debug(f"Shade time: {time.time() - start_shade}")

        im = shaded.to_pil()
        im.save(buffer, format="PNG")
        return True

    # def render_quad_grid(
    #     self,
    #     da: xr.DataArray,
    #     buffer: io.BytesIO,
    #     minmax_only: bool,
    # ) -> Union[bool, dict]:
    #     """
    #     Render the data array into an image buffer when the dataset is using a
    #     2d grid
    #     :param da:
    #     :return:
    #     """
    #     projection_start = time.time()
    #     if self.crs == "EPSG:3857":
    #         if (
    #             self.grid_type == GridType.NON_DIMENSIONAL
    #             or self.grid_type == GridType.SGRID
    #         ):
    #             x, y = to_mercator.transform(da.cf["longitude"], da.cf["latitude"])
    #             x_chunks = (
    #                 da.cf["longitude"].chunks if da.cf["longitude"].chunks else x.shape
    #             )
    #             y_chunks = (
    #                 da.cf["latitude"].chunks if da.cf["latitude"].chunks else y.shape
    #             )

    #             da = da.assign_coords(
    #                 {
    #                     "x": (
    #                         da.cf["longitude"].dims,
    #                         dask_array.from_array(x, chunks=x_chunks),
    #                     ),
    #                     "y": (
    #                         da.cf["latitude"].dims,
    #                         dask_array.from_array(y, chunks=y_chunks),
    #                     ),
    #                 },
    #             )

    #             da = da.unify_chunks()
    #         elif self.grid_type == GridType.REGULAR:
    #             da = da.rio.reproject("EPSG:3857")
    #     else:
    #         da = da.assign_coords({"x": da.cf["longitude"], "y": da.cf["latitude"]})

    #     logger.debug(f"Projection time: {time.time() - projection_start}")
