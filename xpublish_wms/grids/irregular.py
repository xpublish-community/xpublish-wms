from typing import Optional

import dask.array as dask_array
import numpy as np
import xarray as xr

from xpublish_wms.grids.grid import Grid, RenderMethod
from xpublish_wms.utils import (
    bilinear_interp,
    lat_lng_find_quad,
    lat_lng_quad_percentage,
    strip_float,
    to_mercator,
)


class IrregularGrid(Grid):
    def __init__(self, ds: xr.Dataset):
        self.ds = ds

    @staticmethod
    def recognize(ds: xr.Dataset) -> bool:
        try:
            return len(ds.cf["latitude"].dims) == 2
        except Exception:
            return False

    @property
    def name(self) -> str:
        return "nondimensional"

    @property
    def render_method(self) -> RenderMethod:
        return RenderMethod.Quad

    @property
    def crs(self) -> str:
        return "EPSG:4326"

    def project(
        self,
        da: xr.DataArray,
        crs: str,
    ) -> tuple[xr.DataArray, Optional[xr.DataArray], Optional[xr.DataArray]]:
        da = self.mask(da)

        if crs == "EPSG:4326":
            da = da.assign_coords({"x": da.cf["longitude"], "y": da.cf["latitude"]})
        elif crs == "EPSG:3857":
            x, y = to_mercator.transform(da.cf["longitude"], da.cf["latitude"])
            x_chunks = (
                da.cf["longitude"].chunks if da.cf["longitude"].chunks else x.shape
            )
            y_chunks = da.cf["latitude"].chunks if da.cf["latitude"].chunks else y.shape

            if da.chunks:
                x_coord_array = dask_array.from_array(x, chunks=x_chunks)
                y_coord_array = dask_array.from_array(y, chunks=y_chunks)
            else:
                x_coord_array = x
                y_coord_array = y

            da = da.assign_coords(
                {
                    "x": (
                        da.cf["longitude"].dims,
                        x_coord_array,
                    ),
                    "y": (
                        da.cf["latitude"].dims,
                        y_coord_array,
                    ),
                },
            )

            da = da.unify_chunks()
        return da, None, None

    def sel_lat_lng(
        self,
        subset: xr.Dataset,
        lng,
        lat,
        parameters,
    ) -> tuple[xr.Dataset, list, list]:
        """Select the given dataset by the given lon/lat and optional elevation"""

        subset = self.mask(subset)

        # cut the dataset down to 1 point, the values are adjusted anyhow so doesn't matter the point
        subset_keys = [key for key in subset.dims]
        ret_subset = subset.isel({subset_keys[-2]: 0, subset_keys[-1]: 0})

        lng_name = ret_subset.cf["longitude"].name
        lat_name = ret_subset.cf["latitude"].name

        # adjust the lng/lat to the requested point
        ret_subset.__setitem__(
            lng_name,
            (
                ret_subset[lng_name].dims,
                np.full(ret_subset[lng_name].shape, lng),
                ret_subset[lng_name].attrs,
            ),
        )
        ret_subset.__setitem__(
            lat_name,
            (
                ret_subset[lat_name].dims,
                np.full(ret_subset[lat_name].shape, lat),
                ret_subset[lat_name].attrs,
            ),
        )

        lng_values = subset.cf["longitude"].values
        lat_values = subset.cf["latitude"].values

        # find if the selected lng/lat is within a quad
        valid_quad = lat_lng_find_quad(lng, lat, lng_values, lat_values)

        # if no -> set all values to nan
        if valid_quad is None:
            for parameter in parameters:
                ret_subset.__setitem__(
                    parameter,
                    (
                        ret_subset[parameter].dims,
                        np.full(ret_subset[parameter].shape, np.nan),
                        ret_subset[parameter].attrs,
                    ),
                )
        # if yes -> interpolate the values using bilinear interpolation
        else:
            percent_quad, percent_point = lat_lng_quad_percentage(
                lng,
                lat,
                lng_values,
                lat_values,
                valid_quad,
            )

            for parameter in parameters:
                values = subset[parameter].values[
                    ...,
                    valid_quad[0][0] : (valid_quad[1][0] + 1),
                    valid_quad[0][1] : (valid_quad[1][1] + 1),
                ]

                new_value = bilinear_interp(percent_point, percent_quad, values)
                ret_subset.__setitem__(
                    parameter,
                    (
                        ret_subset[parameter].dims,
                        np.full(ret_subset[parameter].shape, new_value),
                        ret_subset[parameter].attrs,
                    ),
                )

        x_axis = [strip_float(ret_subset.cf["longitude"])]
        y_axis = [strip_float(ret_subset.cf["latitude"])]
        return ret_subset, x_axis, y_axis
