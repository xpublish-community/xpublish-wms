from typing import Optional, Sequence, Union

import dask.array as dask_array
import matplotlib.tri as tri
import numpy as np
import xarray as xr

from xpublish_wms.grids.grid import Grid, RenderMethod
from xpublish_wms.utils import (
    barycentric_weights,
    lat_lng_find_tri,
    strip_float,
    to_mercator,
)


class SELFEGrid(Grid):
    def __init__(self, ds: xr.Dataset):
        self.ds = ds

    @staticmethod
    def recognize(ds: xr.Dataset) -> bool:
        return ds.attrs.get("source", "").lower().startswith("selfe")

    @property
    def name(self) -> str:
        return "selfe"

    @property
    def render_method(self) -> RenderMethod:
        return RenderMethod.Triangle

    @property
    def crs(self) -> str:
        return "EPSG:4326"

    def has_elevation(self, da: xr.DataArray) -> bool:
        return "nv" in da.dims

    def elevation_units(self, da: xr.DataArray) -> Optional[str]:
        if self.has_elevation(da):
            return "sigma"
        else:
            return None

    def elevation_positive_direction(self, da: xr.DataArray) -> Optional[str]:
        if self.has_elevation(da):
            return self.ds.cf["vertical"].attrs.get("positive", "up")
        else:
            return None

    def elevations(self, da: xr.DataArray) -> Optional[xr.DataArray]:
        if self.has_elevation(da):
            # clean up elevation values using nv as index array
            vertical = self.ds.cf["vertical"].values
            elevations = []
            for index in da.nv.values:
                if index < len(vertical):
                    elevations.append(vertical[index])

            return xr.DataArray(
                data=elevations,
                dims=da.nv.dims,
                coords=da.nv.coords,
                name=self.ds.cf["vertical"].name,
                attrs=self.ds.cf["vertical"].attrs,
            )

        return None

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
        if "nele" in subset.dims:
            ret_subset = subset.isel(nele=0)
        else:
            ret_subset = subset.isel(node=0)

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

        # find if the selected lng/lat is within a triangle
        valid_tri = lat_lng_find_tri(
            lng,
            lat,
            lng_values,
            lat_values,
            self.tessellate(subset),
        )
        # if no -> set all values to nan
        if valid_tri is None:
            for parameter in parameters:
                ret_subset.__setitem__(
                    parameter,
                    (
                        ret_subset[parameter].dims,
                        np.full(ret_subset[parameter].shape, np.nan),
                        ret_subset[parameter].attrs,
                    ),
                )
        # if yes -> interpolate the values based on barycentric weights
        else:
            p1 = [lng_values[valid_tri[0]], lat_values[valid_tri[0]]]
            p2 = [lng_values[valid_tri[1]], lat_values[valid_tri[1]]]
            p3 = [lng_values[valid_tri[2]], lat_values[valid_tri[2]]]
            w1, w2, w3 = barycentric_weights([lng, lat], p1, p2, p3)

            for parameter in parameters:
                values = subset[parameter].values
                v1 = values[..., valid_tri[0]]
                v2 = values[..., valid_tri[1]]
                v3 = values[..., valid_tri[2]]

                new_value = (v1 * w1) + (v2 * w2) + (v3 * w3)
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

    def select_by_elevation(
        self,
        da: xr.DataArray,
        elevations: Optional[Sequence[float]],
    ) -> xr.DataArray:
        """Select the given data array by elevation"""
        if not self.has_elevation(da):
            return da

        if (
            elevations is None
            or len(elevations) == 0
            or all(v is None for v in elevations)
        ):
            elevations = [0.0]

        da_elevations = self.elevations(da)
        elevation_index = [
            int(np.absolute(da_elevations - elevation).argmin().values)
            for elevation in elevations
        ]
        if len(elevation_index) == 1:
            elevation_index = elevation_index[0]

        if "vertical" not in da.cf:
            if da.nv.shape[0] > da_elevations.shape[0]:
                # need to fill the nv array w/ nan to match dimensions of the var's nv
                new_nv_data = da_elevations.values.tolist()
                for i in range(da.nv.shape[0] - da_elevations.shape[0]):
                    new_nv_data.append(np.nan)

                da_elevations = xr.DataArray(
                    data=new_nv_data,
                    dims=da_elevations.dims,
                    coords=da_elevations.coords,
                    name=da_elevations.name,
                    attrs=da_elevations.attrs,
                )

            da.__setitem__("nv", da_elevations)

        if "vertical" in da.cf:
            da = da.cf.isel(vertical=elevation_index)

        return da

    def project(self, da: xr.DataArray, crs: str) -> any:
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
        return da

    def tessellate(self, da: Union[xr.DataArray, xr.Dataset]) -> np.ndarray:
        ele = self.ds.ele
        if len(ele.shape) > 2:
            for i in range(len(ele.shape) - 2):
                ele = ele[0]

        return tri.Triangulation(
            da.cf["longitude"],
            da.cf["latitude"],
            ele.T - 1,
        ).triangles
