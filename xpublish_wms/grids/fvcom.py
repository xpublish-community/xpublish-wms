from typing import Optional, Sequence, Union

import dask.array as dask_array
import matplotlib.tri as tri
import numpy as np
import xarray as xr

from xpublish_wms.grids.grid import Grid, RenderMethod
from xpublish_wms.utils import (
    barycentric_weights,
    lat_lng_find_tri,
    lat_lng_find_tri_ind,
    strip_float,
    to_mercator,
)


class FVCOMGrid(Grid):
    def __init__(self, ds: xr.Dataset):
        self.ds = ds

    @staticmethod
    def recognize(ds: xr.Dataset) -> bool:
        return ds.attrs.get("source", "").lower().startswith("fvcom")

    @property
    def name(self) -> str:
        return "fvcom"

    @property
    def render_method(self) -> RenderMethod:
        return RenderMethod.Triangle

    @property
    def crs(self) -> str:
        return "EPSG:4326"

    def has_elevation(self, da: xr.DataArray) -> bool:
        return "vertical" in da.cf or "siglay" in da.dims or "siglev" in da.dims

    def elevation_units(self, da: xr.DataArray) -> Optional[str]:
        if "vertical" in da.cf:
            return da.cf["vertical"].attrs.get("units", "sigma")
        elif "siglay" in da.dims:
            # Sometimes fvcom variables dont have coordinates assigned correctly, so brute force it
            return "sigma"
        elif "siglev" in da.dims:
            # Sometimes fvcom variables dont have coordinates assigned correctly, so brute force it
            return "sigma"
        else:
            return None

    def elevation_positive_direction(self, da: xr.DataArray) -> Optional[str]:
        if "vertical" in da.cf:
            return da.cf["vertical"].attrs.get("positive", "up")
        elif "siglay" in da.dims:
            # Sometimes fvcom variables dont have coordinates assigned correctly, so brute force it
            return self.ds.siglay.attrs.get("positive", "up")
        elif "siglev" in da.dims:
            # Sometimes fvcom variables dont have coordinates assigned correctly, so brute force it
            return self.ds.siglev.attrs.get("positive", "up")
        else:
            return None

    def elevations(self, da: xr.DataArray) -> Optional[xr.DataArray]:
        if "vertical" in da.cf:
            return da.cf["vertical"][:, 0]
        else:
            # Sometimes fvcom variables dont have coordinates assigned correctly, so brute force it
            vertical_var = None
            if "siglay" in da.dims:
                vertical_var = "siglay"
            elif "siglev" in da.dims:
                vertical_var = "siglev"

            if vertical_var is not None:
                return xr.DataArray(
                    data=self.ds[vertical_var][:, 0].values,
                    dims=vertical_var,
                    name=self.ds[vertical_var].name,
                    attrs=self.ds[vertical_var].attrs,
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

        if "siglay" in subset.dims:
            subset = subset.rename_dims({"siglay": "sigma"})
        elif "siglev" in subset.dims:
            subset = subset.rename_dims({"siglev": "sigmaa"})

        if "nele" in subset.dims:
            return self.sel_lat_lng_nele(subset, lng, lat, parameters)
        else:
            return self.sel_lat_lng_node(subset, lng, lat, parameters)

    def sel_lat_lng_node(
        self,
        subset: xr.Dataset,
        lng,
        lat,
        parameters,
    ) -> tuple[xr.Dataset, list, list]:
        temp_arrays = dict()
        # create new dataarrays so that nele variables can be adjusted appropriately
        for parameter in parameters:
            # copy existing dataarray into temp_arrays
            if "nele" not in subset.dims:
                temp_arrays[subset[parameter].name] = subset[parameter]
            # create new data by getting values from the surrounding edges
            else:
                if "time" in self.ds.ntve.coords:
                    elem_count = self.ds.ntve.isel(time=0).values
                else:
                    elem_count = self.ds.ntve.values

                if "time" in self.ds.nbve.coords:
                    neighbors = self.ds.nbve.isel(time=0).values
                else:
                    neighbors = self.ds.nbve.values

                mask = neighbors[:, :] > 0
                data = (
                    np.sum(
                        subset[parameter].values[..., neighbors[:, :] - 1],
                        axis=-2,
                        where=mask,
                    )
                    / elem_count
                )
                temp_arrays[subset[parameter].name] = (
                    subset[parameter].dims,
                    data,
                    subset[parameter].attrs,
                )

        lng_name = subset.cf["longitude"].name
        lat_name = subset.cf["latitude"].name

        coords = dict()
        # need to create new lng & lat coordinates with dataset values while dropping the old ones
        # can't keep the original values or else subset will have 2 lng/lat arrays
        for coord in subset.coords:
            if coord != lng_name and coord != lat_name:
                coords[coord] = subset.coords[coord]

        # new lng array
        coords[lng_name] = (
            subset[lng_name].dims,
            self.ds.lon.values,
            subset[lng_name].attrs,
        )

        # new lat array
        coords[lat_name] = (
            subset[lat_name].dims,
            self.ds.lat.values,
            subset[lat_name].attrs,
        )

        # new dataset
        subset = xr.Dataset(data_vars=temp_arrays, coords=coords, attrs=subset.attrs)

        # cut the dataset down to 1 point, the values are adjusted anyhow so doesn't matter the point
        if "nele" in subset.dims:
            ret_subset = subset.isel(nele=0)
        else:
            ret_subset = subset.isel(node=0)

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

        if lng < 0 and np.min(lng_values) > 0:
            lng += 360
        elif lng > 180 and np.min(lng_values) < 0:
            lng -= 360

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

    def sel_lat_lng_nele(
        self,
        subset: xr.Dataset,
        lng,
        lat,
        parameters,
    ) -> tuple[xr.Dataset, list, list]:
        lng_values = self.ds.lon.values
        lat_values = self.ds.lat.values

        if lng < 0 and np.max(lng_values) > 180:
            lng += 360
        elif lng > 180 and np.min(lng_values) < 0:
            lng -= 360

        # find if the selected lng/lat is within a triangle
        valid_tri = lat_lng_find_tri_ind(
            lng,
            lat,
            lng_values,
            lat_values,
            self.tessellate(subset),
        )

        ret_subset = subset.isel(nele=0)

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
        else:
            ret_subset = subset.isel(nele=valid_tri[0])

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
            if "siglay" in da.dims:
                da.__setitem__("siglay", da_elevations)
            elif "siglev" in da.dims:
                da.__setitem__("siglev", da_elevations)

        if "vertical" in da.cf:
            da = da.cf.isel(vertical=elevation_index)

        return da

    def project(
        self,
        da: xr.DataArray,
        crs: str,
    ) -> tuple[xr.DataArray, Optional[xr.DataArray], Optional[xr.DataArray]]:
        da = self.mask(da)

        data = da.values

        coords = dict()
        # need to create new x & y coordinates with dataset values while dropping the old ones
        # can't keep the original values or else da.cf will have 2 lng/lat arrays
        for coord in da.coords:
            if coord != da.cf["longitude"].name and coord != da.cf["latitude"].name:
                coords[coord] = da.coords[coord]

        # create new data by getting values from the surrounding edges if the metadata is available
        # and the variable is zonal
        if "nele" in da.dims and "ntve" in self.ds:
            if "time" in self.ds.ntve.coords:
                elem_count = self.ds.ntve.isel(time=0).values
            else:
                elem_count = self.ds.ntve.values

            if "time" in self.ds.nbve.coords:
                neighbors = self.ds.nbve.isel(time=0).values
            else:
                neighbors = self.ds.nbve.values

            mask = neighbors[:, :] > 0
            data = np.sum(data[neighbors[:, :] - 1], axis=0, where=mask) / elem_count

            coords["x"] = (
                da.cf["longitude"].dims,
                self.ds.lon.values,
                da.cf["longitude"].attrs,
            )
            coords["y"] = (
                da.cf["latitude"].dims,
                self.ds.lat.values,
                da.cf["latitude"].attrs,
            )
            tri_x = None
            tri_y = None
        else:
            # build new x coordinate
            coords["x"] = da.cf["longitude"]
            # build new y coordinate
            coords["y"] = da.cf["latitude"]

            if "nele" in da.dims:
                tri_x = self.ds.lon
                tri_y = self.ds.lat
            else:
                tri_x = None
                tri_y = None

        # build new data array
        da = xr.DataArray(
            data=data,
            dims=da.dims,
            name=da.name,
            attrs=da.attrs,
            coords=coords,
        )

        if crs == "EPSG:4326":
            da.__setitem__(da.cf["longitude"].name, da.cf["longitude"] - 360)
            if tri_x is not None:
                tri_x -= 360
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
                        da.cf["longitude"].attrs,
                    ),
                    "y": (
                        da.cf["latitude"].dims,
                        dask_array.from_array(y, chunks=y_chunks),
                        da.cf["latitude"].attrs,
                    ),
                },
            )

            da = da.unify_chunks()

            if tri_x is not None:
                x, y = to_mercator.transform(tri_x, tri_y)
                tri_x = dask_array.from_array(x)
                tri_y = dask_array.from_array(y)

                return da, tri_x, tri_y

        return da, None, None

    def tessellate(self, da: Union[xr.DataArray, xr.Dataset]) -> np.ndarray:
        nv = self.ds.nv
        if len(nv.shape) > 2:
            for i in range(len(nv.shape) - 2):
                nv = nv[0]

        if "nele" in da.dims:
            return nv.T - 1
        else:
            return tri.Triangulation(
                da.cf["longitude"],
                da.cf["latitude"],
                nv.T - 1,
            ).triangles
