from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, Optional, Tuple

import cartopy.geodesic
import cf_xarray  # noqa
import dask.array as dask_array
import matplotlib.tri as tri
import numpy as np
import rioxarray  # noqa
import xarray as xr

from xpublish_wms.utils import strip_float, to_mercator


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
        elevation: float = 0.0,
    ) -> xr.DataArray:
        """Select the given data array by elevation"""
        if "vertical" in da.cf:
            da = da.cf.sel({"vertical": elevation}, method="nearest")
        return da

    def mask(self, da: xr.DataArray) -> xr.DataArray:
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
        if crs == "EPSG:4326":
            da = da.assign_coords({"x": da.cf["longitude"], "y": da.cf["latitude"]})
        elif crs == "EPSG:3857":
            da = da.rio.reproject("EPSG:3857")
        return da


class ROMSGrid(Grid):
    def __init__(self, ds: xr.Dataset):
        self.ds = ds

    @staticmethod
    def recognize(ds: xr.Dataset) -> bool:
        print(ds.cf.cf_roles)
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

    def mask(self, da: xr.DataArray) -> xr.DataArray:
        mask = self.ds[f'mask_{da.cf["latitude"].name.split("_")[1]}'].cf.isel(time=0)
        mask[:-1,:] = mask[:-1,:].where(mask[1:,:] == 1, 0)
        mask[:,:-1] = mask[:,:-1].where(mask[:,1:] == 1, 0)
        mask[1:,:] = mask[1:,:].where(mask[:-1,:] == 1, 0)
        mask[:,1:] = mask[:,1:].where(mask[:,:-1] == 1, 0)

        return da.where(mask == 1)

    def project(self, da: xr.DataArray, crs: str) -> xr.DataArray:
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
        topology = self.ds.cf["grid_topology"]

        merged_ds = None
        x_axis = None
        y_axis = None

        for parameter in parameters:
            grid_location = subset[parameter].attrs["location"]
            lng_coord, lat_coord = topology.attrs[f"{grid_location}_coordinates"].split(
                " ",
            )
            new_selected_ds = sel2d(
                subset,
                lons=subset.cf[lng_coord],
                lats=subset.cf[lat_coord],
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
        return ds.attrs.get("title", "").startswith("HYCOM")

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

    def project(self, da: xr.DataArray, crs: str) -> Any:
        # TODO: Figure out global coords
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


class FVCOMGrid(Grid):
    def __init__(self, ds: xr.Dataset):
        self.ds = ds

    @staticmethod
    def recognize(ds: xr.Dataset) -> bool:
        return ds.attrs.get("source", "").startswith("FVCOM")

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
        elif "siglay" in da.dims:
            # Sometimes fvcom variables dont have coordinates assigned correctly, so brute force it
            return self.ds.siglay[:, 0]
        elif "siglev" in da.dims:
            # Sometimes fvcom variables dont have coordinates assigned correctly, so brute force it
            return self.ds.siglev[:, 0]
        else:
            return None

    def select_by_elevation(
        self,
        da: xr.DataArray,
        elevation: Optional[float],
    ) -> xr.DataArray:
        """Select the given data array by elevation"""
        print(da.coords)
        print(da.cf)
        if not self.has_elevation(da):
            return da

        if elevation is None:
            elevation = 0.0

        elevations = self.elevations(da)
        diff = np.absolute(elevations - elevation)
        elevation_index = int(diff.argmin().values)
        if "vertical" in da.cf:
            da = da.cf.isel(vertical=elevation_index)
        elif "siglay" in da.dims:
            print(elevation_index)
            da = da.isel(siglay=elevation_index)
        elif "siglev" in da.dims:
            da = da.isel(siglev=elevation_index)

        print(da.coords)
        print(da.cf)

        return da

    def project(self, da: xr.DataArray, crs: str) -> Any:
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
        return tri.Triangulation(
            da.cf["longitude"],
            da.cf["latitude"],
            self.ds.nv[0].T - 1,
        ).triangles


_grid_impls = [HYCOMGrid, FVCOMGrid, ROMSGrid, RegularGrid]


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


@xr.register_dataset_accessor("grid")
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
        elevation: Optional[float],
    ) -> xr.DataArray:
        if self._grid is None:
            return None
        else:
            return self._grid.select_by_elevation(da, elevation)

    def mask(self, da: xr.DataArray) -> xr.DataArray:
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


class GridType(Enum):
    REGULAR = 1
    NON_DIMENSIONAL = 2
    SGRID = 3
    UNSUPPORTED = 255

    @classmethod
    def from_ds(cls, ds: xr.Dataset):
        if "grid_topology" in ds.cf.cf_roles:
            return cls.SGRID

        try:
            if len(ds.cf["latitude"].dims) > 1:
                return cls.NON_DIMENSIONAL
            elif "latitude" in ds.cf["latitude"].dims:
                return cls.REGULAR
        except Exception:
            return cls.UNSUPPORTED

        return cls.UNSUPPORTED


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
