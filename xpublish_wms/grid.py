from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, Optional, Sequence, Tuple, Union

import cartopy.geodesic
import cf_xarray  # noqa
import dask.array as dask_array
import matplotlib.tri as tri
import numpy as np
import rioxarray  # noqa
import xarray as xr

from xpublish_wms.utils import lnglat_to_mercator, strip_float, to_mercator


class RenderMethod(Enum):
    Quad = "quad"
    Triangle = "triangle"


class Grid(ABC):
    @staticmethod
    @abstractmethod
    def recognize(ds: xr.Dataset) -> bool:
        """Recognize whether the given dataset is of this grid type"""
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """Name of the grid type"""
        pass

    @property
    @abstractmethod
    def render_method(self) -> RenderMethod:
        """Name of the render method"""
        pass

    @property
    @abstractmethod
    def crs(self) -> str:
        """CRS of the grid"""
        pass

    def bbox(self, da: xr.DataArray) -> Tuple[float, float, float, float]:
        """Bounding box of the grid for the given data array in the form (minx, miny, maxx, maxy)"""
        lng = da.cf["longitude"]
        lng = xr.where(lng > 180, lng - 360, lng)
        return (
            float(lng.min()),
            float(da.cf["latitude"].min()),
            float(lng.max()),
            float(da.cf["latitude"].max()),
        )

    def has_elevation(self, da: xr.DataArray) -> bool:
        """Whether the given data array has elevation"""
        return "vertical" in da.cf

    def elevation_units(self, da: xr.DataArray) -> Optional[str]:
        """Return the elevation inits for the given data array"""
        coord = da.cf.coords.get("vertical", None)
        if coord is not None:
            return coord.attrs.get("units", "sigma")
        else:
            return None

    def elevation_positive_direction(self, da: xr.DataArray) -> Optional[str]:
        """Return the elevation positive direction for the given data array"""
        coord = da.cf.coords.get("vertical", None)
        if coord is not None:
            return coord.attrs.get("positive", "up")
        else:
            return None

    def elevations(self, da: xr.DataArray) -> Optional[xr.DataArray]:
        """Return the elevations for the given data array"""
        if "vertical" in da.cf:
            return da.cf["vertical"]
        else:
            return None

    def select_by_elevation(
        self,
        da: xr.DataArray,
        elevations: Sequence[float],
    ) -> xr.DataArray:
        """Select the given data array by elevation"""

        if (
            elevations is None
            or len(elevations) == 0
            or all(v is None for v in elevations)
        ):
            elevations = [0.0]

        if "vertical" in da.cf:
            if len(elevations) == 1:
                return da.cf.sel(vertical=elevations[0], method="nearest")
            elif len(elevations) > 1:
                return da.cf.sel(vertical=elevations)
            else:
                return da.cf.sel(vertical=0, method="nearest")

        return da

    def mask(
        self,
        da: Union[xr.DataArray, xr.Dataset],
    ) -> Union[xr.DataArray, xr.Dataset]:
        """Mask the given data array"""
        return da

    @abstractmethod
    def project(self, da: xr.DataArray, crs: str) -> Any:
        """Project the given data array from this dataset and grid to the given crs"""
        pass

    def tessellate(self, da: Union[xr.DataArray, xr.Dataset]) -> np.ndarray:
        """Tessellate the given data array into triangles. Only required for RenderingMode.Triangle"""
        pass

    def sel_lat_lng(
        self,
        subset: xr.Dataset,
        lng,
        lat,
        parameters,
    ) -> Tuple[xr.Dataset, list, list]:
        """Select the given dataset by the given lon/lat and optional elevation"""

        subset = self.mask(subset)
        subset = subset.cf.interp(longitude=lng, latitude=lat)

        x_axis = [strip_float(subset.cf["longitude"])]
        y_axis = [strip_float(subset.cf["latitude"])]
        return subset, x_axis, y_axis


class RegularGrid(Grid):
    def __init__(self, ds: xr.Dataset):
        self.ds = ds

    @staticmethod
    def recognize(ds: xr.Dataset) -> bool:
        return True

    @property
    def name(self) -> str:
        return "regular"

    @property
    def render_method(self) -> RenderMethod:
        return RenderMethod.Quad

    @property
    def crs(self) -> str:
        return "EPSG:4326"

    def project(self, da: xr.DataArray, crs: str) -> xr.DataArray:
        da = self.mask(da)

        coords = dict()
        # need to convert longitude and latitude to x and y for the mesh to work properly
        # regular grid doesn't have x & y as dimensions so have to remake the whole data array
        for coord in da.coords:
            if coord != da.cf["longitude"].name and coord != da.cf["latitude"].name:
                coords[coord] = da.coords[coord]

        # build new x coordinate
        coords["x"] = ("x", da.cf["longitude"].values, da.cf["longitude"].attrs)
        # build new y coordinate
        coords["y"] = ("y", da.cf["latitude"].values, da.cf["latitude"].attrs)
        # build new data array
        da = xr.DataArray(
            data=da,
            dims=("y", "x"),
            coords=coords,
            name=da.name,
            attrs=da.attrs,
        )

        # convert to mercator
        if crs == "EPSG:3857":
            lng, lat = lnglat_to_mercator(da.cf["longitude"], da.cf["latitude"])

            da = da.assign_coords({"x": lng, "y": lat})
            da = da.unify_chunks()

        return da


class NonDimensionalGrid(Grid):
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

    def project(self, da: xr.DataArray, crs: str) -> xr.DataArray:
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

    def sel_lat_lng(
        self,
        subset: xr.Dataset,
        lng,
        lat,
        parameters,
    ) -> Tuple[xr.Dataset, list, list]:
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

    def project(self, da: xr.DataArray, crs: str) -> xr.DataArray:
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

    def sel_lat_lng(
        self,
        subset: xr.Dataset,
        lng,
        lat,
        parameters,
    ) -> Tuple[xr.Dataset, list, list]:
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

    def bbox(self, da: xr.DataArray) -> Tuple[float, float, float, float]:
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

    def project(self, da: xr.DataArray, crs: str) -> Any:
        da = self.mask(da)

        # create 2 separate DataArrays where points lng>180 are put at the beginning of the array
        mask_0 = xr.where(da.cf["longitude"] <= 180, 1, 0)
        temp_da_0 = da.where(mask_0.compute() == 1, drop=True)
        da_0 = xr.DataArray(
            data=temp_da_0,
            dims=temp_da_0.dims,
            name=temp_da_0.name,
            coords=temp_da_0.coords,
            attrs=temp_da_0.attrs,
        )

        mask_1 = xr.where(da.cf["longitude"] > 180, 1, 0)
        temp_da_1 = da.where(mask_1.compute() == 1, drop=True)
        temp_da_1.cf["longitude"][:] = temp_da_1.cf["longitude"][:] - 360
        da_1 = xr.DataArray(
            data=temp_da_1,
            dims=temp_da_1.dims,
            name=temp_da_1.name,
            coords=temp_da_1.coords,
            attrs=temp_da_1.attrs,
        )

        # put the 2 DataArrays back together in the proper order
        da = xr.concat([da_1, da_0], dim="X")

        if crs == "EPSG:4326":
            da = da.assign_coords({"x": da.cf["longitude"], "y": da.cf["latitude"]})
        elif crs == "EPSG:3857":
            lng, lat = lnglat_to_mercator(da.cf["longitude"], da.cf["latitude"])

            da = da.assign_coords({"x": lng, "y": lat})
            da = da.unify_chunks()

        return da

    def sel_lat_lng(
        self,
        subset: xr.Dataset,
        lng,
        lat,
        parameters,
    ) -> Tuple[xr.Dataset, list, list]:
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
    ) -> Tuple[xr.Dataset, list, list]:
        """Select the given dataset by the given lon/lat and optional elevation"""

        subset = self.mask(subset)

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

        # find if the selected lng/lat is within a triangle
        valid_tri = lat_lng_find_tri(
            lng + 360,
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
            w1, w2, w3 = barycentric_weights([lng + 360, lat], p1, p2, p3)

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
            if "siglay" in da.dims:
                da.__setitem__("siglay", da_elevations)
            elif "siglev" in da.dims:
                da.__setitem__("siglev", da_elevations)

        if "vertical" in da.cf:
            da = da.cf.isel(vertical=elevation_index)

        return da

    def project(self, da: xr.DataArray, crs: str) -> Any:
        da = self.mask(da)

        data = da.values
        # create new data by getting values from the surrounding edges
        if "nele" in da.dims:
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

        coords = dict()
        # need to create new x & y coordinates with dataset values while dropping the old ones
        # can't keep the original values or else da.cf will have 2 lng/lat arrays
        for coord in da.coords:
            if coord != da.cf["longitude"].name and coord != da.cf["latitude"].name:
                coords[coord] = da.coords[coord]

        # build new x coordinate
        coords["x"] = (
            da.cf["longitude"].dims,
            self.ds.lon.values,
            da.cf["longitude"].attrs,
        )
        # build new y coordinate
        coords["y"] = (
            da.cf["latitude"].dims,
            self.ds.lat.values,
            da.cf["latitude"].attrs,
        )

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
        return da

    def tessellate(self, da: Union[xr.DataArray, xr.Dataset]) -> np.ndarray:
        nv = self.ds.nv
        if len(nv.shape) > 2:
            for i in range(len(nv.shape) - 2):
                nv = nv[0]

        return tri.Triangulation(
            da.cf["longitude"],
            da.cf["latitude"],
            nv.T - 1,
        ).triangles


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
    ) -> Tuple[xr.Dataset, list, list]:
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

    def project(self, da: xr.DataArray, crs: str) -> Any:
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


_grid_impls = [
    HYCOMGrid,
    FVCOMGrid,
    SELFEGrid,
    ROMSGrid,
    NonDimensionalGrid,
    RegularGrid,
]


def register_grid_impl(grid_impl: Grid, priority: int = 0):
    """
    Register a new grid implementation.
    :param grid_impl: The grid implementation to register
    :param priority: The priority of the implementation. Highest priority is 0. Default is 0.
    """
    _grid_impls.insert(grid_impl, priority)


def grid_factory(ds: xr.Dataset) -> Optional[Grid]:
    for grid_impl in _grid_impls:
        if grid_impl.recognize(ds):
            return grid_impl(ds)

    return None


@xr.register_dataset_accessor("gridded")
class GridDatasetAccessor:
    _ds: xr.Dataset
    _grid: Optional[Grid]

    def __init__(self, ds: xr.Dataset):
        self._ds = ds
        self._grid = grid_factory(ds)

    @property
    def grid(self) -> Grid:
        if self._grid is None:
            return None
        else:
            return self._grid

    @property
    def name(self) -> str:
        if self._grid is None:
            return "unsupported"
        else:
            return self.grid.name

    @property
    def render_method(self) -> RenderMethod:
        if self._grid is None:
            return RenderMethod.Quad
        else:
            return self._grid.render_method

    @property
    def crs(self) -> str:
        if self._grid is None:
            return None
        else:
            return self.grid.crs

    def bbox(self, var) -> Tuple[float, float, float, float]:
        if self._grid is None:
            return None
        else:
            return self._grid.bbox(var)

    def has_elevation(self, da: xr.DataArray) -> bool:
        if self._grid is None:
            return False
        else:
            return self._grid.has_elevation(da)

    def elevation_units(self, da: xr.DataArray) -> Optional[str]:
        if self._grid is None:
            return None
        else:
            return self._grid.elevation_units(da)

    def elevation_positive_direction(self, da: xr.DataArray) -> Optional[str]:
        if self._grid is None:
            return None
        else:
            return self._grid.elevation_positive_direction(da)

    def elevations(self, da: xr.DataArray) -> Optional[xr.DataArray]:
        if self._grid is None:
            return None
        else:
            return self._grid.elevations(da)

    def select_by_elevation(
        self,
        da: xr.DataArray,
        elevations: Optional[Sequence[float]],
    ) -> xr.DataArray:
        if self._grid is None:
            return None
        else:
            return self._grid.select_by_elevation(da, elevations)

    def mask(
        self,
        da: Union[xr.DataArray, xr.Dataset],
    ) -> Union[xr.DataArray, xr.Dataset]:
        if self._grid is None:
            return None
        else:
            return self._grid.mask(da)

    def project(self, da: xr.DataArray, crs: str) -> xr.DataArray:
        if self._grid is None:
            return None
        else:
            return self._grid.project(da, crs)

    def tessellate(self, da: Union[xr.DataArray, xr.Dataset]) -> np.ndarray:
        if self._grid is None:
            return None
        else:
            return self._grid.tessellate(da)

    def sel_lat_lng(
        self,
        subset: xr.Dataset,
        lng,
        lat,
        parameters,
    ) -> Tuple[xr.Dataset, list, list]:
        if self._grid is None:
            return None
        else:
            return self._grid.sel_lat_lng(subset, lng, lat, parameters)


def argsel2d(lons, lats, lon0, lat0):
    """Find the indices of coordinate pair closest to another point.
    Adapted from https://github.com/xoceanmodel/xroms/blob/main/xroms/utilities.py which is failing to run for some reason

    Inputs
    ------
    lons: DataArray, ndarray, list
        Longitudes of points to search through for closest point.
    lats: DataArray, ndarray, list
        Latitudes of points to search through for closest point.
    lon0: float, int
        Longitude of comparison point.
    lat0: float, int
        Latitude of comparison point.
    Returns
    -------
    Index or indices of location in coordinate pairs made up of lons, lats
    that is closest to location lon0, lat0. Number of dimensions of
    returned indices will correspond to the shape of input lons.
    Notes
    -----
    This function uses Great Circle distance to calculate distances assuming
    longitudes and latitudes as point coordinates. Uses cartopy function
    `Geodesic`: https://scitools.org.uk/cartopy/docs/latest/cartopy/geodesic.html
    If searching for the closest grid node to a lon/lat location, be sure to
    use the correct horizontal grid (rho, u, v, or psi). This is accounted for
    if this function is used through the accessor.
    Example usage
    -------------
    >>> xroms.argsel2d(ds.lon_rho, ds.lat_rho, -96, 27)
    """

    # input lons and lats can be multidimensional and might be DataArrays or lists
    pts = list(zip(*(np.asarray(lons).flatten(), np.asarray(lats).flatten())))
    endpts = list(zip(*(np.asarray(lon0).flatten(), np.asarray(lat0).flatten())))

    G = cartopy.geodesic.Geodesic()  # set up class
    dist = np.asarray(G.inverse(pts, endpts)[:, 0])  # select distances specifically
    iclosest = abs(np.asarray(dist)).argmin()  # find indices of closest point
    # return index or indices in input array shape
    inds = np.unravel_index(iclosest, np.asarray(lons).shape)

    return inds


def sel2d(ds, lons, lats, lon0, lat0):
    """Find the value of the var at closest location to lon0,lat0.
    Adapted from https://github.com/xoceanmodel/xroms/blob/main/xroms/utilities.py which is failing to run for some reason

    Inputs
    ------
    ds: DataArray, ndarray, or DataSet
        Dataset to operate of
    lons: DataArray, ndarray, list
        Longitudes of points to search through for closest point.
    lats: DataArray, ndarray, list
        Latitudes of points to search through for closest point.
    lon0: float, int
        Longitude of comparison point.
    lat0: float, int
        Latitude of comparison point.
    Returns
    -------
    Value in var of location in coordinate pairs made up of lons, lats
    that is closest to location lon0, lat0. If var has other
    dimensions, they are brought along.
    Notes
    -----
    This function uses Great Circle distance to calculate distances assuming
    longitudes and latitudes as point coordinates. Uses cartopy function
    `Geodesic`: https://scitools.org.uk/cartopy/docs/latest/cartopy/geodesic.html
    If searching for the closest grid node to a lon/lat location, be sure to
    use the correct horizontal grid (rho, u, v, or psi). This is accounted for
    if this function is used through the accessor.
    This is meant to be used by the accessor to conveniently wrap
    `argsel2d`.
    Example usage
    -------------
    >>> xroms.sel2d(ds.temp, ds.lon_rho, ds.lat_rho, -96, 27)
    """
    inds = argsel2d(lons, lats, lon0, lat0)
    return ds.isel({lats.dims[0]: inds[0], lats.dims[1]: inds[1]})


def barycentric_weights(point, v1, v2, v3):
    """
    calculate the barycentric weight for each of the triangle vertices

    Inputs
    ------
    point: [float, float]
        [Longitude, Latitude] of comparison point.
    v1: [float, float]
        Vertex 1
    v2: [float, float]
        Vertex 1
    v3: [float, float]
        Vertex 1
    Returns
    -------
    3 weights relative to each of the 3 vertices. Then the interpolated value can be calculated using
    the following formula: (w1 * value1) + (w2 * value2) + (w3 * value3)
    """

    denominator = ((v2[1] - v3[1]) * (v1[0] - v3[0])) + (
        (v3[0] - v2[0]) * (v1[1] - v3[1])
    )

    w1 = (
        ((v2[1] - v3[1]) * (point[0] - v3[0])) + ((v3[0] - v2[0]) * (point[1] - v3[1]))
    ) / denominator
    w2 = (
        ((v3[1] - v1[1]) * (point[0] - v3[0])) + ((v1[0] - v3[0]) * (point[1] - v3[1]))
    ) / denominator
    w3 = 1 - w1 - w2

    return w1, w2, w3


def lat_lng_find_tri(lng, lat, lng_values, lat_values, triangles):
    """
    Find the triangle that the inputted lng/lat is within

    Inputs
    ------
    lng: float, int
        Longitude of comparison point.
    lat: float, int
        Latitude of comparison point.
    lng_values: xr.DataArray
        Longitudes of points corresponding to the indices in triangles
    lat_values: xr.DataArray
        Latitudes of points corresponding to the indices in triangles
    triangles: ndarray of shape (X, 3)
        Triangle mesh of indices as generated by tri.Triangulation
    Returns
    -------
    Triangle of indices that the lng/lat is within, which can be used with barycentric_weights
    to interpolate the value accurate to the lng/lat requested, or None if the point is outside
    the triangular mesh
    """

    lnglat_data = np.stack((lng_values[triangles], lat_values[triangles]), axis=2)

    d1 = (
        (lng - lnglat_data[:, 1, 0]) * (lnglat_data[:, 0, 1] - lnglat_data[:, 1, 1])
    ) - ((lnglat_data[:, 0, 0] - lnglat_data[:, 1, 0]) * (lat - lnglat_data[:, 1, 1]))
    d2 = (
        (lng - lnglat_data[:, 2, 0]) * (lnglat_data[:, 1, 1] - lnglat_data[:, 2, 1])
    ) - ((lnglat_data[:, 1, 0] - lnglat_data[:, 2, 0]) * (lat - lnglat_data[:, 2, 1]))
    d3 = (
        (lng - lnglat_data[:, 0, 0]) * (lnglat_data[:, 2, 1] - lnglat_data[:, 0, 1])
    ) - ((lnglat_data[:, 2, 0] - lnglat_data[:, 0, 0]) * (lat - lnglat_data[:, 0, 1]))

    has_neg = np.logical_or(np.logical_or(d1 < 0, d2 < 0), d3 < 0)
    has_pos = np.logical_or(np.logical_or(d1 > 0, d2 > 0), d3 > 0)

    not_in_tri = np.logical_and(has_neg, has_pos)
    tri_index = np.where(not_in_tri == 0)

    if len(tri_index) == 0 or len(tri_index[0]) == 0:
        return None
    else:
        return triangles[tri_index[0]][0]


def bilinear_interp(percent_point, percent_quad, value_quad):
    """
    Calculates the bilinear interpolation of values provided by the value_quad, where the percent_quad and
    percent_point variables determine where the point to be interpolated is

    Inputs
    ------
    percent_point: [..., float, float]
        [Longitude, Latitude] vertex normalized based on lat_lng_quad_percentage
    percent_quad: [..., [float, float], [float, float], [float, float], [float, float]]
        [Longitude, Latitude] vertices representing each corner of the quad normalized based on lat_lng_quad_percentage
    value_quad: [..., float, float, float, float]
        Data values at each corner represented by the percent_quad
    Returns
    -------
    The interpolated value at the percent_point specified
    """

    a = -percent_quad[0][0][0] + percent_quad[0][1][0]
    b = -percent_quad[0][0][0] + percent_quad[1][0][0]
    c = (
        percent_quad[0][0][0]
        - percent_quad[1][0][0]
        - percent_quad[0][1][0]
        + percent_quad[1][1][0]
    )
    d = percent_point[0] - percent_quad[0][0][0]
    e = -percent_quad[0][0][1] + percent_quad[0][1][1]
    f = -percent_quad[0][0][1] + percent_quad[1][0][1]
    g = (
        percent_quad[0][0][1]
        - percent_quad[1][0][1]
        - percent_quad[0][1][1]
        + percent_quad[1][1][1]
    )
    h = percent_point[1] - percent_quad[0][0][1]

    alpha_denominator = 2 * c * e - 2 * a * g
    beta_denominator = 2 * c * f - 2 * b * g

    # for regular grids, just use x/y percents as alpha/beta
    if alpha_denominator == 0 or beta_denominator == 0:
        alpha = percent_point[0]
        beta = percent_point[1]
    else:
        alpha = (
            -(
                b * e
                - a * f
                + d * g
                - c * h
                + np.sqrt(
                    -4 * (c * e - a * g) * (d * f - b * h)
                    + np.power((b * e - a * f + d * g - c * h), 2),
                )
            )
            / alpha_denominator
        )
        beta = (
            b * e
            - a * f
            - d * g
            + c * h
            + np.sqrt(
                -4 * (c * e - a * g) * (d * f - b * h)
                + np.power((b * e - a * f + d * g - c * h), 2),
            )
        ) / beta_denominator

    return (1 - alpha) * (
        (1 - beta) * value_quad[..., 0, 0] + beta * value_quad[..., 1, 0]
    ) + alpha * ((1 - beta) * value_quad[..., 0, 1] + beta * value_quad[..., 1, 1])


def lat_lng_quad_percentage(lng, lat, lng_values, lat_values, quad):
    """
    Calculates the percentage of each point in the lng_values & lat_values list, where the min & max are 0 & 1
    respectively. Also calculates the percentage for the lng/lat point

    Inputs
    ------
    lng: float, int
        Longitude of comparison point.
    lat: float, int
        Latitude of comparison point.
    lng_values: [float, float, float, float]
        Longitudes of points corresponding to each corner of the quad
    lat_values: [float, float, float, float]
        Latitudes of points corresponding to each corner of the quad
    Returns
    -------
    Quad of percentages based on the input lng_values & lat_values. Also returns a lng/lat vertex as a percentage
    within the percent quad
    """

    lngs = lng_values[quad[0][0] : (quad[1][0] + 1), quad[0][1] : (quad[1][1] + 1)]
    lats = lat_values[quad[0][0] : (quad[1][0] + 1), quad[0][1] : (quad[1][1] + 1)]

    lng_min = np.min(lngs)
    lng_max = np.max(lngs)
    lat_min = np.min(lats)
    lat_max = np.max(lats)

    lng_denominator = lng_max - lng_min
    lat_denominator = lat_max - lat_min

    percent_quad = np.zeros((2, 2, 2))
    percent_quad[:, :, 0] = (lngs - lng_min) / lng_denominator
    percent_quad[:, :, 1] = (lats - lat_min) / lat_denominator

    percent_lng = (lng - lng_min) / lng_denominator
    percent_lat = (lat - lat_min) / lat_denominator

    return percent_quad, [percent_lng, percent_lat]


def lat_lng_find_quad(lng, lat, lng_values, lat_values):
    """
    Find the quad that the inputted lng/lat is within

    Inputs
    ------
    lng: float, int
        Longitude of comparison point.
    lat: float, int
        Latitude of comparison point.
    lng_values: xr.DataArray
        Longitudes of points corresponding to the indices in triangles
    lat_values: xr.DataArray
        Latitudes of points corresponding to the indices in triangles
    Returns
    -------
    Quad of indices that the lng/lat is within, which can be used with lat_lng_quad_percentage and bilinear_interp
    to interpolate the value accurate to the lng/lat requested, or None if the point is outside the mesh
    """

    lnglat_data = np.stack((lng_values, lat_values), axis=2)

    x0y0tox0y1 = np.where(
        (
            (lnglat_data[1:, :-1, 0] - lnglat_data[:-1, :-1, 0])
            * (lat - lnglat_data[:-1, :-1, 1])
            - (lng - lnglat_data[:-1, :-1, 0])
            * (lnglat_data[1:, :-1, 1] - lnglat_data[:-1, :-1, 1])
        )
        <= 0,
        1,
        0,
    )
    x0y1tox1y1 = np.where(
        (
            (lnglat_data[1:, 1:, 0] - lnglat_data[1:, :-1, 0])
            * (lat - lnglat_data[1:, :-1, 1])
            - (lng - lnglat_data[1:, :-1, 0])
            * (lnglat_data[1:, 1:, 1] - lnglat_data[1:, :-1, 1])
        )
        <= 0,
        1,
        0,
    )
    x1y1tox1y0 = np.where(
        (
            (lnglat_data[:-1, 1:, 0] - lnglat_data[1:, 1:, 0])
            * (lat - lnglat_data[1:, 1:, 1])
            - (lng - lnglat_data[1:, 1:, 0])
            * (lnglat_data[:-1, 1:, 1] - lnglat_data[1:, 1:, 1])
        )
        <= 0,
        1,
        0,
    )
    x1y0tox0y0 = np.where(
        (
            (lnglat_data[:-1, :-1, 0] - lnglat_data[:-1, 1:, 0])
            * (lat - lnglat_data[:-1, 1:, 1])
            - (lng - lnglat_data[:-1, 1:, 0])
            * (lnglat_data[:-1, :-1, 1] - lnglat_data[:-1, 1:, 1])
            <= 0
        ),
        1,
        0,
    )

    top_left_index = np.where(
        np.logical_and(
            np.logical_and(x0y0tox0y1, x0y1tox1y1),
            np.logical_and(x1y1tox1y0, x1y0tox0y0),
        ),
    )

    if len(top_left_index) == 0 or len(top_left_index[0]) == 0:
        return None
    else:
        return [
            [top_left_index[0][0], top_left_index[1][0]],
            [top_left_index[0][0] + 1, top_left_index[1][0] + 1],
        ]
