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
import mercantile
import numpy as np
import pandas as pd
import xarray as xr
from fastapi.responses import StreamingResponse

from xpublish_wms.grids import RenderMethod
from xpublish_wms.query import WMSGetMapQuery
from xpublish_wms.utils import filter_data_within_bbox, parse_float

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

    BBOX_BUFFER = 0.18

    cache: cachey.Cache

    # Data selection
    parameter: str
    time: datetime = None
    has_time: bool
    elevation: float = None
    has_elevation: bool
    dim_selectors: dict

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

    def get_map(
        self,
        ds: xr.Dataset,
        query: WMSGetMapQuery,
        query_params: dict,
    ) -> StreamingResponse:
        """
        Return the WMS map for the dataset and given parameters
        """
        # Decode request params
        self.ensure_query_types(ds, query, query_params)

        # Select data according to request
        da = self.select_layer(ds)
        da = self.select_time(da)
        da = self.select_elevation(ds, da)
        da = self.select_custom_dim(da)

        # Render the data using the render that matches the dataset type
        # The data selection and render are coupled because they are both driven by
        # The grid type for now. This can be revisited if we choose to interpolate or
        # use the contoured renderer for regular grid datasets
        image_buffer = io.BytesIO()
        render_result = self.render(ds, da, image_buffer, False)
        if render_result:
            image_buffer.seek(0)

        return StreamingResponse(image_buffer, media_type="image/png")

    def get_minmax(
        self,
        ds: xr.Dataset,
        query: WMSGetMapQuery,
        query_params: dict,
        entire_layer: bool,
    ) -> dict:
        """
        Return the range of values for the dataset and given parameters
        """
        # Decode request params
        self.ensure_query_types(ds, query, query_params)

        # Select data according to request
        da = self.select_layer(ds)
        da = self.select_time(da)
        da = self.select_elevation(ds, da)
        da = self.select_custom_dim(da)

        # Prepare the data as if we are going to render it, but instead grab the min and max
        # values from the data to represent the range of values in the given area
        if entire_layer:
            return {"min": float(da.min()), "max": float(da.max())}
        else:
            return self.render(ds, da, None, minmax_only=True)

    def ensure_query_types(
        self,
        ds: xr.Dataset,
        query: WMSGetMapQuery,
        query_params: dict,
    ):
        """
        Decode request params

        :param query:
        :return:
        """
        # Data selection
        self.parameter = query.layers
        self.time_str = query.time

        if self.time_str:
            self.time = pd.to_datetime(self.time_str).tz_localize(None)
        else:
            self.time = None
        self.has_time = self.TIME_CF_NAME in ds[self.parameter].cf.coords

        self.elevation_str = query.elevation
        if self.elevation_str:
            self.elevation = float(self.elevation_str)
        else:
            self.elevation = None
        self.has_elevation = self.ELEVATION_CF_NAME in ds[self.parameter].cf.coords

        # Grid
        self.crs = query.crs
        tile = query.tile
        if tile is not None:
            tile = [float(x) for x in query.tile.split(",")]
            self.bbox = mercantile.xy_bounds(*tile)
            self.crs = "EPSG:3857"  # tiles are always mercator
        else:
            self.bbox = [float(x) for x in query.bbox.split(",")]
        self.width = query.width
        self.height = query.height

        # Output style
        self.style = query.styles
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
            parse_float(x) for x in query.colorscalerange.split(",")
        ]
        self.autoscale = query.autoscale

        available_selectors = ds.gridded.additional_coords(ds[self.parameter])
        self.dim_selectors = {k: query_params[k] for k in available_selectors}

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
        # Filter dimension from custom query, if any
        for dim, value in self.dim_selectors.items():
            if dim in da.coords:
                dtype = da[dim].dtype
                if "timedelta" in str(dtype):
                    value = pd.to_timedelta(value)
                elif np.issubdtype(dtype, np.integer):
                    value = int(value)
                elif np.issubdtype(dtype, np.floating):
                    value = float(value)
                da = da.sel({dim: value}, method="nearest")

        # Squeeze single value dimensions
        da = da.squeeze()

        # Squeeze multiple values dimensions, by selecting the last value
        # for key in da.cf.coordinates.keys():
        #     if key in ("latitude", "longitude", "X", "Y"):
        #         continue

        #     coord = da.cf.coords[key]
        #     if coord.size > 1:
        #         da = da.cf.isel({key: -1})

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

        x = None
        y = None
        try:
            da, x, y = ds.gridded.project(da, self.crs)
        except Exception as e:
            logger.warning(f"Projection failed: {e}")
            if minmax_only:
                logger.warning("Falling back to default minmax")
                return {"min": float(da.min()), "max": float(da.max())}

        # x and y are only set for triangle grids, we dont subset the data for triangle grids
        # at this time.
        if x is None:
            try:
                # Grab a buffer around the bbox to ensure we have enough data to render
                diff = (da.x[1] - da.x[0]).values
                diff = diff * 1.05

                # Filter the data to only include the data within the bbox + buffer so
                # we don't have to render a ton of empty space or pull down more chunks
                # than we need
                da = filter_data_within_bbox(da, self.bbox, diff)
            except Exception as e:
                logger.error(f"Error filtering data within bbox: {e}")
                logger.warning("Falling back to full layer")

        # Squeeze single value dimensions
        da = da.squeeze()

        logger.info(f"WMS GetMap Projection time: {time.time() - projection_start}")

        start_dask = time.time()

        da = da.compute()
        if x is not None and y is not None:
            x = x.compute()
            y = y.compute()

        logger.info(f"WMS GetMap dask compute: {time.time() - start_dask}")

        if minmax_only:
            try:
                return {
                    "min": float(np.nanmin(da)),
                    "max": float(np.nanmax(da)),
                }
            except Exception as e:
                logger.error(
                    f"Error computing minmax: {e}, falling back to full layer minmax",
                )
                return {"min": float(da.min()), "max": float(da.max())}

        if not self.autoscale:
            vmin, vmax = self.colorscalerange
        else:
            vmin, vmax = [None, None]

        start_shade = time.time()
        cvs = dsh.Canvas(
            plot_height=self.height,
            plot_width=self.width,
            x_range=(self.bbox[0], self.bbox[2]),
            y_range=(self.bbox[1], self.bbox[3]),
        )

        print(da)

        if ds.gridded.render_method == RenderMethod.Quad:
            mesh = cvs.quadmesh(
                da,
                x="x",
                y="y",
            )
        elif ds.gridded.render_method == RenderMethod.Triangle:
            triangles = ds.gridded.tessellate(da)
            if x is not None:
                # We are coloring the triangles by the data values
                verts = pd.DataFrame({"x": x, "y": y})
                tris = pd.DataFrame(triangles.astype(int), columns=["v0", "v1", "v2"])
                tris = tris.assign(z=da.values)
            else:
                # We are coloring the vertices by the data values
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
        logger.info(f"WMS GetMap Shade time: {time.time() - start_shade}")

        im = shaded.to_pil()
        im.save(buffer, format="PNG")
        return True
