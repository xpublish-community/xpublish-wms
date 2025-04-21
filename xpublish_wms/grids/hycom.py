from typing import Optional, Union

import numpy as np
import xarray as xr

from xpublish_wms.grids.grid import Grid, RenderMethod
from xpublish_wms.utils import (
    bilinear_interp,
    lat_lng_find_quad,
    lat_lng_quad_percentage,
    lnglat_to_mercator,
    strip_float,
    to_lnglat_allow_over
)


class HYCOMGrid(Grid):
    def __init__(self, ds: xr.Dataset):
        self.ds = ds

    @staticmethod
    def recognize(ds: xr.Dataset) -> bool:
        return ds.attrs.get("title", "").lower().startswith("hycom")

    @property
    def name(self) -> str:
        return "hycom"

    @property
    def render_method(self) -> RenderMethod:
        return RenderMethod.Quad

    @property
    def crs(self) -> str:
        return "EPSG:4326"

    def bbox(self, da: xr.DataArray) -> tuple[float, float, float, float]:
        # HYCOM global grid (RTOFS) has invalid longitude values
        # over 500 that need to masked. Then the coords need to be
        # normalized between -180 and 180
        lng = da.cf["longitude"]
        lng = lng.where(lng < 500) % 360
        lng = xr.where(lng > 180, lng - 360, lng)

        return (
            float(lng.min()),
            float(da.cf["latitude"].min()),
            float(lng.max()),
            float(da.cf["latitude"].max()),
        )

    def mask(
        self,
        da: Union[xr.DataArray, xr.Dataset],
    ) -> Union[xr.DataArray, xr.Dataset]:
        # mask values where the longitude is >500 bc of RTOFS-Global https://oceanpython.org/2012/11/29/global-rtofs-real-time-ocean-forecast-system/
        lng = da.cf["longitude"]

        # add missing dimensions to lng mask (ie. elevation & time)
        if len(da.dims) > len(lng.dims):
            missing_dims = list(set(da.dims).symmetric_difference(lng.dims))
            missing_dims.reverse()

            for dim in missing_dims:
                lng = lng.expand_dims({dim: da[dim]})

        mask = lng > 500

        np_mask = np.ma.masked_where(mask == 1, da.values)[:]
        np_mask[:-1, :] = np.ma.masked_where(mask[1:, :] == 1, np_mask[:-1, :])[:]
        np_mask[:, :-1] = np.ma.masked_where(mask[:, 1:] == 1, np_mask[:, :-1])[:]
        np_mask[1:, :] = np.ma.masked_where(mask[:-1, :] == 1, np_mask[1:, :])[:]
        np_mask[:, 1:] = np.ma.masked_where(mask[:, :-1] == 1, np_mask[:, 1:])[:]

        return da.where(np_mask.mask == 0)

    def project(
        self,
        da: xr.DataArray,
        crs: str,
        render_context: Optional[dict] = dict(),
    ) -> tuple[xr.DataArray, Optional[xr.DataArray], Optional[xr.DataArray]]:
        if not render_context.get("masked", False):
            da = self.mask(da)

        if not render_context.get("lng_adjusted", False):
            da = self._adjust_lng(da)
        
        if crs == "EPSG:4326":
            da = da.assign_coords({"x": da.cf["longitude"], "y": da.cf["latitude"]})
        elif crs == "EPSG:3857":
            lng, lat = lnglat_to_mercator(da.cf["longitude"], da.cf["latitude"])

            da = da.assign_coords({"x": lng, "y": lat})
            da = da.unify_chunks()

        return da, render_context

    def filter_by_bbox(self, da, bbox, crs, render_context: Optional[dict] = dict()):
        da = self.mask(da)
        render_context["masked"] = True

        if crs == "EPSG:3857":
            bbox = to_lnglat_allow_over.transform(
                [bbox[0], bbox[2]],
                [bbox[1], bbox[3]],
            )
            bbox = [bbox[0][0], bbox[1][0], bbox[0][1], bbox[1][1]]

        da = self._adjust_lng(da)
        render_context["lng_adjusted"] = True

        # Get the x and y values
        x = da.cf["longitude"]
        y = da.cf["latitude"]

       # Find the indices of the data within the bounding box
        x_inds = np.where((x >= bbox[0]) & (x <= bbox[2]))
        y_inds = np.where((y >= bbox[1]) & (y <= bbox[3]))

        if len(x.dims) != len(y.dims) or len(x_inds) != len(y_inds):
            raise Exception("Mismatched number of dims for filter_by_bbox")

        # Select and return the data within the bounding box
        sel_dims = dict()
        for i in range(len(x.dims)):
            sel_dims[x.dims[i]] = np.intersect1d(x_inds[i], y_inds[i])

        da = da.isel(sel_dims)
        return da, render_context

    def sel_lat_lng(
        self,
        subset: xr.Dataset,
        lng,
        lat,
        parameters,
    ) -> tuple[xr.Dataset, list, list]:
        """Select the given dataset by the given lon/lat and optional elevation"""

        for parameter in parameters:
            subset[parameter] = self.mask(subset[parameter])

        subset.__setitem__(
            subset.cf["longitude"].name,
            xr.where(
                subset.cf["longitude"] > 180,
                subset.cf["longitude"] - 360,
                subset.cf["longitude"],
            ),
        )

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


    def _adjust_lng(self, da: xr.DataArray):
        # create 2 separate DataArrays where points lng>180 are put at the beginning of the array
        mask = xr.where(da.cf["longitude"] <= 180, 1, 0).compute()
        temp_da_0 = da.where(mask == 1, drop=True)
        da_0 = xr.DataArray(
            data=temp_da_0,
            dims=temp_da_0.dims,
            name=temp_da_0.name,
            coords=temp_da_0.coords,
            attrs=temp_da_0.attrs,
        )

        temp_da_1 = da.where(mask == 0, drop=True)
        temp_da_1[temp_da_1.cf["longitude"].name] -= 360
        da_1 = xr.DataArray(
            data=temp_da_1,
            dims=temp_da_1.dims,
            name=temp_da_1.name,
            coords=temp_da_1.coords,
            attrs=temp_da_1.attrs,
        )

        # put the 2 DataArrays back together in the proper order
        return xr.concat([da_1, da_0], dim="X")