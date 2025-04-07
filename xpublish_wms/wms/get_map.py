import io
import time
from datetime import datetime
from typing import List, Union

import cachey
import cf_xarray  # noqa
import datashader as dsh
import datashader.transfer_functions as tf
import matplotlib
import mercantile
import numpy as np
import pandas as pd
import xarray as xr
from fastapi import HTTPException
from fastapi.responses import StreamingResponse

from xpublish_wms.grids import RenderMethod
from xpublish_wms.logger import logger
from xpublish_wms.query import WMSGetMapQuery


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
    array_render_threshold_bytes: int

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

    def __init__(
        self,
        array_render_threshold_bytes: int,
        cache: cachey.Cache | None = None,
    ):
        self.cache = cache
        self.array_render_threshold_bytes = array_render_threshold_bytes

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
        try:
            self.ensure_query_types(ds, query, query_params)
        except Exception as e:
            logger.error(f"Error decoding request params: {e}")
            raise HTTPException(
                422,
                "Error decoding request params, please check the request is valid. See the logs for more details.",
            )

        # Select data according to request
        try:
            da = self.select_layer(ds)
        except Exception as e:
            logger.error(f"Error selecting layer: {e}")
            raise HTTPException(
                422,
                "Error selecting layer, please check the layer name is correct and the dataset has a variable with that name. See the logs for more details.",
            )

        try:
            da = self.select_time(da)
        except Exception as e:
            logger.error(f"Error selecting time: {e}")
            raise HTTPException(
                422,
                "Error selecting time, please check the time format is correct and the time dimension exists in the dataset. See the logs for more details.",
            )

        try:
            da = self.select_elevation(ds, da)
        except Exception as e:
            logger.error(f"Error selecting elevation: {e}")
            raise HTTPException(
                422,
                "Error selecting elevation, please check the elevation format is correct and the vertical dimension exists in the dataset. See the logs for more details.",
            )

        try:
            da = self.select_custom_dim(da)
        except Exception as e:
            logger.error(f"Error selecting custom dimensions: {e}")
            raise HTTPException(
                422,
                "Error selecting custom dimensions, please check all custom selectors are valid and the dimensions exists in the dataset. See the logs for more details.",
            )

        # Render the data using the render that matches the dataset type
        # The data selection and render are coupled because they are both driven by
        # The grid type for now. This can be revisited if we choose to interpolate or
        # use the contoured renderer for regular grid datasets
        image_buffer = io.BytesIO()
        try:
            render_result = self.render(ds, da, image_buffer, False)
        except HTTPException as e:
            raise e
        except Exception as e:
            logger.error(f"Error rendering data: {e}")
            raise HTTPException(
                422,
                "Error rendering data, please check the data is valid and the render method is supported for the dataset type. See the logs for more details.",
            )

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
            self.bbox = mercantile.xy_bounds(*tile)
            self.crs = "EPSG:3857"  # tiles are always mercator
        else:
            self.bbox = query.bbox
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

        self.colorscalerange = query.colorscalerange
        self.autoscale = query.autoscale

        available_selectors = ds.gridded.additional_coords(ds[self.parameter])
        self.dim_selectors = {
            k: query_params[k] if k in query_params else None
            for k in available_selectors
        }

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
        # by using coords, we will fallback to ds.coords[self.TIME_CF_NAME]
        # if cf-xarray can't identify self.TIME_CF_NAME using attributes
        # This is a nice fallback for datasets with `"time"`
        if not self.has_time:
            return da
        if self.time is None:
            return da.cf.isel({self.TIME_CF_NAME: -1})
        else:
            return da.cf.sel({self.TIME_CF_NAME: self.time}, method="nearest")

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

        If dimension is provided :
            - use xarray to access custom coord
            - method nearest to ensure at least one result

        Otherwise:
            - Get first value of coord

        :param da:
        :return:
        """
        # Filter dimension from custom query, if any
        for dim, value in self.dim_selectors.items():
            if dim in da.coords:
                if value is None:
                    da = da.isel({dim: 0})
                else:
                    dtype = da[dim].dtype
                    method = None
                    if "timedelta" in str(dtype):
                        value = pd.to_timedelta(value)
                    elif np.issubdtype(dtype, np.integer):
                        value = int(value)
                        method = "nearest"
                    elif np.issubdtype(dtype, np.floating):
                        value = float(value)
                        method = "nearest"
                    da = da.sel({dim: value}, method=method)

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

        # default context object to pass around between grid functions
        render_context = dict()

        filter_start = time.time()
        try:
            # Grab a buffer around the bbox to ensure we have enough data to render
            x_buffer = (
                abs(max(self.bbox[0], self.bbox[2]) - min(self.bbox[0], self.bbox[2]))
                * 0.15
            )
            y_buffer = (
                abs(max(self.bbox[1], self.bbox[3]) - min(self.bbox[1], self.bbox[3]))
                * 0.15
            )
            bbox = [
                self.bbox[0] - x_buffer,
                self.bbox[1] - y_buffer,
                self.bbox[2] + x_buffer,
                self.bbox[3] + y_buffer,
            ]

            # Filter the data to only include the data within the bbox + buffer so
            # we don't have to render a ton of empty space or pull down more chunks
            # than we need
            da, render_context = ds.gridded.filter_by_bbox(da, bbox, self.crs, render_context=render_context)
        except Exception as e:
            logger.error(f"Error filtering data within bbox: {e}")
            logger.warning("Falling back to full layer")
        logger.debug(f"WMS GetMap BBOX filter time: {time.time() - filter_start}")

        projection_start = time.time()
        try:
            da, render_context = ds.gridded.project(da, self.crs, render_context=render_context)
        except Exception as e:
            logger.warning(f"Projection failed: {e}")
            if minmax_only:
                logger.warning("Falling back to default minmax")
                return {"min": float(da.min()), "max": float(da.max())}

        # Squeeze single value dimensions
        da = da.squeeze()
        logger.debug(f"WMS GetMap Projection time: {time.time() - projection_start}")

        # Print the size of the da in megabytes
        da_size = da.nbytes
        if da_size > self.array_render_threshold_bytes:
            logger.error(
                f"DataArray size is {da_size:.2f} bytes, which is larger than the "
                f"threshold of {self.array_render_threshold_bytes} bytes. "
                f"Consider increasing the threshold in the plugin configuration.",
            )
            raise HTTPException(
                413,
                f"DataArray too large to render: threshold is {self.array_render_threshold_bytes} bytes, data is {da_size:.2f} bytes",
            )
        logger.debug(f"WMS GetMap loading DataArray size: {da_size:.2f} bytes")

        start_dask = time.time()
        da = da.load()
        logger.debug(f"WMS GetMap load data: {time.time() - start_dask}")

        if da.size == 0:
            logger.warning("No data to render")
            return False

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
            span = (self.colorscalerange[0], self.colorscalerange[1])
        else:
            span = None

        start_mesh = time.time()
        cvs = dsh.Canvas(
            plot_height=self.height,
            plot_width=self.width,
            x_range=(self.bbox[0], self.bbox[2]),
            y_range=(self.bbox[1], self.bbox[3]),
        )

        # numba only supports float32 and float64. Cast everything else
        if da.dtype.kind == "f" and da.dtype.itemsize != 4 and da.dtype.itemsize != 8:
            logger.warning(
                f"DataArray dtype is {da.dtype}, which is not a floating point type "
                f"of size 32 or 64. This will result in a slow render.",
            )
            if da.dtype.itemsize < 4:
                logger.warning(
                    "DataArray dtype is 16-bit. This must be converted to 32-bit before rendering.",
                )
                da = da.astype(np.float32)
            elif da.dtype.itemsize < 8:
                logger.warning(
                    "DataArray dtype is 32-bit. This must be converted to 64-bit before rendering.",
                )
                da = da.astype(np.float64)
            else:
                raise ValueError(
                    f"DataArray dtype is {da.dtype}, which is not a floating point type "
                    f"greater than 64-bit. This is not currently supported.",
                )

        if ds.gridded.render_method == RenderMethod.Raster:
            mesh = cvs.raster(
                da,
            )
        elif ds.gridded.render_method == RenderMethod.Quad:
            try:
                mesh = cvs.quadmesh(
                    da,
                    x="x",
                    y="y",
                )
            except Exception as e:
                logger.warning(f"Error rendering quadmesh: {e}, falling back to raster")
                mesh = cvs.raster(
                    da,
                )
        elif ds.gridded.render_method == RenderMethod.Triangle:
            triangles, render_context = ds.gridded.tessellate(da, render_context=render_context)

            # TODO - maybe this discrepancy between coloring by verts v tris should be part of the grid?
            if "tri_x" in render_context and "tri_y" in render_context:
                # We are coloring the triangles by the data values
                verts = pd.DataFrame({"x": render_context["tri_x"], "y": render_context["tri_y"]})
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
        logger.debug(f"WMS GetMap Mesh time: {time.time() - start_mesh}")

        start_shade = time.time()
        shaded = tf.shade(
            mesh,
            cmap=matplotlib.colormaps.get_cmap(self.palettename),
            how="linear",
            span=span,
        )
        logger.debug(f"WMS GetMap Shade time: {time.time() - start_shade}")

        im = shaded.to_pil()
        im.save(buffer, format="PNG")
        return True
