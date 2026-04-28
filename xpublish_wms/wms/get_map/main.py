import io
import time
from datetime import datetime
from functools import partial
from typing import Iterable, List, Sequence, Union

import cachey
import cf_xarray  # noqa
import datashader as dsh
import datashader.transfer_functions as tf
import matplotlib
import mercantile
import numpy as np
from numpy.typing import NDArray
import pandas as pd
import xarray as xr
from fastapi import HTTPException
from fastapi.responses import Response
from PIL.Image import Image

from xpublish_wms.grids import RenderMethod
from xpublish_wms.logger import logger
from xpublish_wms.query import WMSGetMapQuery
from xpublish_wms.wms.get_map.style_types import (
    ColormapStyleParams,
    ShadingStyleParams,
    VectorStyleParams,
)
from xpublish_wms.wms.get_map.vector_styles import visualize_vectors, get_cell_center_indices


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
    parameters: list[str]
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
    styles: ShadingStyleParams

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
    ) -> Response:
        """
        Return the WMS map for the dataset and given parameters
        """
        start_get_map = time.time()
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
            das = self.select_layers(ds)
        except Exception as e:
            logger.error(f"Error selecting layer: {e}")
            raise HTTPException(
                422,
                "Error selecting layer, please check the layer name is correct and the dataset has a variable with that name. See the logs for more details.",
            )

        try:
            das = [self.select_time(da) for da in das]
        except Exception as e:
            logger.error(f"Error selecting time: {e}")
            raise HTTPException(
                422,
                "Error selecting time, please check the time format is correct and the time dimension exists in the dataset. See the logs for more details.",
            )

        try:
            das = [self.select_elevation(ds, da) for da in das]
        except Exception as e:
            logger.error(f"Error selecting elevation: {e}")
            raise HTTPException(
                422,
                "Error selecting elevation, please check the elevation format is correct and the vertical dimension exists in the dataset. See the logs for more details.",
            )

        try:
            das = [self.select_custom_dim(da) for da in das]
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
            render_result = self.render(ds, das, image_buffer, False)
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

        logger.debug(f"WMS GetMap TOTAL time: {time.time() - start_get_map}")
        return Response(image_buffer.getbuffer(), media_type="image/png")

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
        das = self.select_layers(ds)
        selectors = [
            self.select_time,
            partial(self.select_elevation, ds),
            self.select_custom_dim,
        ]
        for selector in selectors:
            das = [selector(da) for da in das]

        # Prepare the data as if we are going to render it, but instead grab the min and max
        # values from the data to represent the range of values in the given area
        filtered_das = (
            das if entire_layer else self.render(ds, das, None, minmax_only=True)
        )
        if isinstance(filtered_das, bool):
            # render method returned False because the filtered DataArray was empty
            return {"min": 0, "max": 0}

        # Get one DataArray from one or more
        da = das_to_scalar(filtered_das)

        try:
            # `da` seems to be lazy and when accessed here for a computed magnitude
            # of vector component layers, we can run into allocation errors.
            return {"min": float(da.min()), "max": float(da.max())}
        except MemoryError as err:
            logger.error(
                f"Failed to allocate enough memory to calculate min/max: {err}",
            )
            if len(filtered_das) == 2:
                # for vector layers, try to calculate a ceiling of the magnitude as a fallback
                max_x, max_y = (np.abs(da).max() for da in filtered_das)
                ceiling = float(np.sqrt(max_x**2 + max_y**2))
                # magnitude cannot be negative so 0 is the floor
                return {"min": 0, "max": ceiling}

            raise err

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
        self.parameters = query.layers
        self.time_str = query.time

        if self.time_str:
            self.time = pd.to_datetime(self.time_str).tz_localize(None)
        else:
            self.time = None
        self.has_time = all(
            self.TIME_CF_NAME in ds[parameter].cf.coords
            for parameter in self.parameters
        )

        self.elevation_str = query.elevation
        if self.elevation_str:
            self.elevation = float(self.elevation_str)
        else:
            self.elevation = None
        self.has_elevation = all(
            self.ELEVATION_CF_NAME in ds[parameter].cf.coords
            for parameter in self.parameters
        )

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
        self.decode_query_styles(query)

        available_selectors = list(
            set.intersection(
                *(
                    set(ds.gridded.additional_coords(ds[parameter]))
                    for parameter in self.parameters
                ),
            ),
        )
        self.dim_selectors = {
            k: query_params[k] if k in query_params else None
            for k in available_selectors
        }

    def decode_query_styles(self, query: WMSGetMapQuery):
        """
        Decode request parameters related to render style
        """
        style_type, colormap = query.styles

        # Let user pick colormap from here https://predictablynoisy.com/matplotlib/gallery/color/colormap_reference.html#sphx-glr-gallery-color-colormap-reference-py
        # Otherwise default
        palette_name = (
            self.DEFAULT_PALETTE if colormap in ["colormap", "default"] else colormap
        )

        if style_type.startswith("vector"):
            self.styles = VectorStyleParams(
                type="vector",
                color=query.color,
                density=query.density or 2,
                scaling=VectorStyleParams.GlyphScaling.CONSTANT,
                colorscale_range=query.colorscalerange,
                colormap=None if palette_name == "none" else palette_name,
                draw_backing=palette_name != "none" and style_type in ("vector-arrow", "vector-cells-arrow"),
                arrow_mag_color=style_type in ("vector-arrow-color", "vector-cells-arrow-color"),
                use_cell_centers=style_type.startswith("vector-cells-"),
            )
            return

        # else: `style_type` is "raster"
        self.styles = ColormapStyleParams(
            type="colormap",
            palettename=palette_name,
            colorscale_range=query.colorscalerange,
            autoscale=query.autoscale,
        )

    def select_layers(self, ds: xr.Dataset) -> list[xr.DataArray]:
        """
        Select Dataset variables, according to WMS layers request
        :param ds:
        :return:
        """
        return [ds[parameter] for parameter in self.parameters]

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
        das: Iterable[xr.DataArray],
        buffer: io.BytesIO | None,
        minmax_only: bool,
    ) -> Union[bool, List[xr.DataArray]]:
        """
        Render the data array into an image buffer.

        If `minmax_only` is True, return the DataArrays before rendering them into
        the image buffer instead.
        """

        # default context object to pass around between grid functions
        # this is useful for each gridded function involved in the render process to set flags or parse values
        # and then send those values/flags to the next gridded function involved in the render process.
        #
        # ex. if ds.gridded.filter_by_bbox applies the grid mask to da, ds.gridded.project can avoid re-masking by checking the context
        render_contexts = [dict() for _ in das]
        das_with_contexts = list(zip(das, render_contexts))

        filter_start = time.time()
        filter_success = False
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
            das_with_contexts = [
                ds.gridded.filter_by_bbox(
                    da,
                    bbox,
                    self.crs,
                    render_context=context,
                )
                for da, context in das_with_contexts
            ]

            filter_success = True
        except Exception as e:
            logger.error(f"Error filtering data within bbox: {e}")
            logger.warning("Falling back to full layer")

            filter_success = False
        logger.debug(f"WMS GetMap BBOX filter time: {time.time() - filter_start}")

        # if filter_by_bbox was successful, preload data for projection
        if filter_success:
            filter_load_time = time.time()
            das_with_contexts = [
                (da.load(), context) for da, context in das_with_contexts
            ]
            logger.debug(
                f"WMS GetMap load filtered data: {time.time() - filter_load_time}",
            )

        projection_start = time.time()
        try:
            das_with_contexts = [
                ds.gridded.project(
                    da,
                    self.crs,
                    render_context=context,
                )
                for da, context in das_with_contexts
            ]
        except Exception as e:
            logger.warning(f"Projection failed: {e}")
            if minmax_only:
                logger.warning("Falling back to default minmax")
                return [da for da, _ in das_with_contexts]

        das = [da for da, _ in das_with_contexts]
        render_contexts = [context for _, context in das_with_contexts]

        # Squeeze single value dimensions
        das = [da.squeeze() for da in das]
        logger.debug(f"WMS GetMap Projection time: {time.time() - projection_start}")

        # Print the size of the das in megabytes
        das_size = sum(da.nbytes for da in das)
        if das_size > self.array_render_threshold_bytes:
            logger.error(
                f"DataArrays size is {das_size:.2f} bytes, which is larger than the "
                f"threshold of {self.array_render_threshold_bytes} bytes. "
                f"Consider increasing the threshold in the plugin configuration.",
            )
            raise HTTPException(
                413,
                f"DataArrays too large to render: threshold is {self.array_render_threshold_bytes} bytes, data is {das_size:.2f} bytes",
            )
        logger.debug(f"WMS GetMap loading DataArray size: {das_size:.2f} bytes")

        start_dask = time.time()
        das = [da.load() for da in das]
        logger.debug(f"wms getmap load full data: {time.time() - start_dask}")

        if sum(da.size for da in das) == 0:
            logger.warning("no data to render")
            return False

        if minmax_only:
            return das

        start_mesh = time.time()
        meshes = [
            self.create_mesh(ds, da, render_context=context)
            for da, context in zip(das, render_contexts)
        ]
        cell_center_indices = (
            get_cell_center_indices(das, self.bbox, self.width, self.height)
            if self.styles.type == "vector" and self.styles.use_cell_centers
            else None
        )
        logger.debug(f"WMS GetMap Mesh time: {time.time() - start_mesh}")

        start_shade = time.time()
        im = self.shade_mesh(meshes, cell_center_indices)
        logger.debug(f"WMS GetMap Shade time: {time.time() - start_shade}")

        # NOTE: remember `assert` can be disabled with `python -O`
        # This should never fail, caller should always pass `buffer` unless `minax=True`
        assert buffer is not None
        im.save(buffer, format="PNG")
        return True

    def create_mesh(
        self,
        ds: xr.Dataset,
        da: xr.DataArray,
        render_context: dict,
    ) -> xr.DataArray:
        """
        Create map mesh
        """
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

        cvs = dsh.Canvas(
            plot_height=self.height,
            plot_width=self.width,
            x_range=(self.bbox[0], self.bbox[2]),
            y_range=(self.bbox[1], self.bbox[3]),
        )

        if ds.gridded.render_method == RenderMethod.Raster:
            return cvs.raster(
                da,
            )

        if ds.gridded.render_method == RenderMethod.Quad:
            try:
                return cvs.quadmesh(
                    da,
                    x="x",
                    y="y",
                )
            except Exception as e:
                logger.warning(f"Error rendering quadmesh: {e}, falling back to raster")
                return cvs.raster(
                    da,
                )

        if ds.gridded.render_method == RenderMethod.Triangle:
            triangles, render_context = ds.gridded.tessellate(
                da,
                render_context=render_context,
            )

            # TODO - maybe this discrepancy between coloring by verts v tris should be part of the grid?
            if "tri_x" in render_context and "tri_y" in render_context:
                # We are coloring the triangles by the data values
                verts = pd.DataFrame(
                    {"x": render_context["tri_x"], "y": render_context["tri_y"]},
                )
                tris = pd.DataFrame(triangles.astype(int), columns=["v0", "v1", "v2"])
                tris = tris.assign(z=da.values)
            else:
                # We are coloring the vertices by the data values
                verts = pd.DataFrame({"x": da.x, "y": da.y, "z": da})
                tris = pd.DataFrame(triangles.astype(int), columns=["v0", "v1", "v2"])

            return cvs.trimesh(
                verts,
                tris,
            )

        raise ValueError(
            f"Unexpected gridded dataset render method {ds.gridded.render_method}",
        )

    def shade_mesh(
        self,
        meshes: Sequence[xr.DataArray],
        cell_center_indices: tuple[NDArray[np.intp], NDArray[np.intp]] | None,
    ) -> Image:
        if self.styles.type == "vector":
            style_kwargs = self.styles.model_dump()
            style_kwargs.pop("type")
            use_cell_centers = style_kwargs.pop("use_cell_centers")
            return visualize_vectors(
                meshes,
                cell_center_indices=cell_center_indices if use_cell_centers else None,
                **style_kwargs,
            )

        # else -> self.style_params.type is "colormap"
        span = (
            None
            if self.styles.autoscale or self.styles.colorscale_range is None
            else tuple(self.styles.colorscale_range[0:2])
        )
        return tf.shade(
            meshes[0],
            cmap=matplotlib.colormaps.get_cmap(self.styles.palettename),
            how="linear",
            span=span,
        ).to_pil()


def das_to_scalar(das: List[xr.DataArray]) -> xr.DataArray | np.ndarray:
    """Get a scalar data array from one or two data arrays.

    We assume that the input `das` is length one or at least two.

    If length is one, we simply return that single array. If length is more,
    we assume there are two arrays and they are a pair of vector components.
    In that case we return the vector magnitude.
    """
    if len(das) == 1:
        return das[0]
    return np.sqrt(das[0] ** 2 + das[1] ** 2)
