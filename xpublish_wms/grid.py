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
from sklearn.neighbors import BallTree

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

    def tessellate(self, da: xr.DataArray) -> np.ndarray:
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

        subset = sel2d(
            subset[parameters],
            lons=subset.cf["longitude"],
            lats=subset.cf["latitude"],
            lon0=lng,
            lat0=lat,
        )
        x_axis = [strip_float(subset.cf["longitude"])]
        y_axis = [strip_float(subset.cf["latitude"])]
        return subset, x_axis, y_axis


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
            mask = mask.cf.squeeze(drop=True).cf

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
        topology = self.ds.grid

        merged_ds = None
        x_axis = None
        y_axis = None

        for parameter in parameters:
            grid_location = subset[parameter].attrs["location"]
            lng_coord, lat_coord = topology.attrs[f"{grid_location}_coordinates"].split(
                " ",
            )

            new_selected_ds = self.mask(subset[[parameter]])
            new_selected_ds = sel2d(
                new_selected_ds,
                lons=new_selected_ds.cf[lng_coord],
                lats=new_selected_ds.cf[lat_coord],
                lon0=lng,
                lat0=lat,
            )

            if merged_ds is None:
                merged_ds = new_selected_ds[[parameter, lat_coord, lng_coord]]
            else:
                merged_ds = new_selected_ds[[parameter, lat_coord, lng_coord]].merge(
                    merged_ds,
                    compat="override",
                )

            if x_axis is None:
                x_axis = [strip_float(new_selected_ds.cf[lng_coord])]
            if y_axis is None:
                y_axis = [strip_float(new_selected_ds.cf[lat_coord])]

        subset = merged_ds
        return subset, x_axis, y_axis


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
            for dim in set(da.dims).symmetric_difference(lng.dims):
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

        subset = sel2d(
            subset[parameters],
            lons=subset.cf["longitude"],
            lats=subset.cf["latitude"],
            lon0=lng,
            lat0=lat,
        )

        x_axis = [strip_float(subset.cf["longitude"])]
        y_axis = [strip_float(subset.cf["latitude"])]
        return subset, x_axis, y_axis


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
                temp_elevations = self.ds[vertical_var].values[:, 0]
                return xr.DataArray(
                    data=[temp_elevations[i] for i in da[vertical_var]],
                    dims=da[vertical_var].dims,
                    coords=da[vertical_var].coords,
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

        lng_rad = np.deg2rad(subset.cf["longitude"])
        lat_rad = np.deg2rad(subset.cf["latitude"])

        stacked = np.stack([lng_rad, lat_rad], axis=-1)
        tree = BallTree(stacked, leaf_size=2, metric="haversine")

        idx = tree.query(
            [[np.deg2rad(360 + lng), np.deg2rad(lat)]],
            return_distance=False,
        )

        if "nele" in subset.dims:
            subset = subset.isel(nele=idx[0][0])
        else:
            subset = subset.isel(node=idx[0][0])

        x_axis = [strip_float(subset.cf["longitude"])]
        y_axis = [strip_float(subset.cf["latitude"])]
        return subset, x_axis, y_axis

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

    def tessellate(self, da: xr.DataArray) -> np.ndarray:
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

        lng_rad = np.deg2rad(subset.cf["longitude"])
        lat_rad = np.deg2rad(subset.cf["latitude"])

        stacked = np.stack([lng_rad, lat_rad], axis=-1)
        tree = BallTree(stacked, leaf_size=2, metric="haversine")

        idx = tree.query(
            [[np.deg2rad((360 + lng) if lng < 0 else lng), np.deg2rad(lat)]],
            return_distance=False,
        )

        if "nele" in subset.dims:
            subset = subset.isel(nele=idx[0][0])
        else:
            subset = subset.isel(node=idx[0][0])

        x_axis = [strip_float(subset.cf["longitude"])]
        y_axis = [strip_float(subset.cf["latitude"])]
        return subset, x_axis, y_axis

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

    def tessellate(self, da: xr.DataArray) -> np.ndarray:
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

    def tessellate(self, da: xr.DataArray) -> np.ndarray:
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
