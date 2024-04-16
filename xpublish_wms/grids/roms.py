from typing import Optional, Union

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


class ROMSGrid(Grid):
    def __init__(self, ds: xr.Dataset):
        self.ds = ds

    @staticmethod
    def recognize(ds: xr.Dataset) -> bool:
        return "grid_topology" in ds.cf.cf_roles

    @property
    def name(self) -> str:
        return "roms"

    @property
    def render_method(self) -> RenderMethod:
        return RenderMethod.Quad

    @property
    def crs(self) -> str:
        return "EPSG:4326"

    def mask(
        self,
        da: Union[xr.DataArray, xr.Dataset],
    ) -> Union[xr.DataArray, xr.Dataset]:
        mask = self.ds[f'mask_{da.cf["latitude"].name.split("_")[1]}']
        if "time" in mask.cf.coords:
            mask = mask.cf.isel(time=0).squeeze(drop=True).cf.drop_vars("time")
        else:
            mask = mask.cf.squeeze(drop=True).copy(deep=True)

        mask[:-1, :] = mask[:-1, :].where(mask[1:, :] == 1, 0)
        mask[:, :-1] = mask[:, :-1].where(mask[:, 1:] == 1, 0)
        mask[1:, :] = mask[1:, :].where(mask[:-1, :] == 1, 0)
        mask[:, 1:] = mask[:, 1:].where(mask[:, :-1] == 1, 0)

        return da.where(mask == 1)

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

            da = da.assign_coords(
                {
                    "x": (
                        da.cf["longitude"].dims,
                        dask_array.from_array(x, chunks=x_chunks),
                    ),
                    "y": (
                        da.cf["latitude"].dims,
                        dask_array.from_array(y, chunks=y_chunks),
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
        unique_dims = dict()
        for parameter in parameters:
            # using a custom mask for now because mask() can cause nan values in quads where there should be 4 corners of data
            mask = self.ds[
                f'mask_{subset[parameter].cf["latitude"].name.split("_")[1]}'
            ]
            if "time" in mask.cf.coords:
                mask = mask.cf.isel(time=0).squeeze(drop=True).cf.drop_vars("time")
            else:
                # We apparently need to deep copy because for some reason
                # if we dont this function will overwrite the mask in the dataset
                # I'm guessing that squeeze is a no-op if there are no length 1
                # dimensions
                mask = mask.cf.squeeze(drop=True).copy(deep=True)

            subset[parameter] = subset[parameter].where(mask == 1)

            # copy unique dims from each parameter
            for dim in subset[parameter].dims[-2:]:
                if dim not in unique_dims:
                    unique_dims[dim] = 0

        # cut the dataset down to 1 point, the values are adjusted anyhow so doesn't matter the point
        ret_subset = subset.isel(unique_dims)

        # ROMs can use different lng/lat arrays for different variables, so all variables need to be updated
        lng_variables = list(ret_subset.cf[["longitude"]].coords)
        # adjust all lng variables to the requested point
        for lng_name in lng_variables:
            ret_subset.__setitem__(
                lng_name,
                (
                    ret_subset[lng_name].dims,
                    np.full(ret_subset[lng_name].shape, lng),
                    ret_subset[lng_name].attrs,
                ),
            )
        lat_variables = list(ret_subset.cf[["latitude"]].coords)
        # adjust all lat variables to the requested point
        for lat_name in lat_variables:
            ret_subset.__setitem__(
                lat_name,
                (
                    ret_subset[lat_name].dims,
                    np.full(ret_subset[lat_name].shape, lat),
                    ret_subset[lat_name].attrs,
                ),
            )

        for parameter in parameters:
            lng_values = subset[parameter].cf["longitude"].values
            lat_values = subset[parameter].cf["latitude"].values

            # find if the selected lng/lat is within a quad
            valid_quad = lat_lng_find_quad(lng, lat, lng_values, lat_values)

            # if no -> set all values to nan
            if valid_quad is None:
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

        x_axis = [strip_float(ret_subset.cf[["longitude"]][lng_variables[0]])]
        y_axis = [strip_float(ret_subset.cf[["latitude"]][lat_variables[0]])]
        return ret_subset, x_axis, y_axis
