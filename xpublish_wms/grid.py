from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, Optional, Tuple

import cartopy.geodesic
import numpy as np
import xarray as xr


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

    @abstractmethod
    def crs(self) -> str:
        """CRS of the grid"""
        pass

    @abstractmethod
    def bbox(self, var: str) -> Tuple[float, float, float, float]:
        """Bounding box of the grid for the given variable in the form (minx, miny, maxx, maxy)"""
        pass

    @abstractmethod
    def render(self, var: str, bbox: Tuple[float, float, float, float], width: int, height: int) -> Any:
        """Render the given variable for the given bounding box with the given width and height"""
        pass


class RegularGrid(Grid):
    def __init__(self, ds: xr.Dataset):
        self.ds = ds

    @staticmethod
    def recognize(ds: xr.Dataset) -> bool:
        return True

    def name(self) -> str:
        return "regular"

    def crs(self) -> str:
        return 'EPSG:4326'

    def bbox(self, var: str) -> Tuple[float, float, float, float]:
        da = self.ds[var]
        return (
            float(da.cf["longitude"].min()),
            float(da.cf["latitude"].min()),
            float(da.cf["longitude"].max()),
            float(da.cf["latitude"].max()),
        )

    def render(self, var: str, crs: str, bbox: Tuple[float, float, float, float], width: int, height: int) -> Any:
        pass

class ROMSGrid(Grid):
    def __init__(self, ds: xr.Dataset):
        self.ds = ds

    @staticmethod
    def recognize(ds: xr.Dataset) -> bool:
        return "grid_topology" in ds.cf.cf_roles
    
    def name(self) -> str:
        return "roms"

    def crs(self) -> str:
        return 'EPSG:4326'

    def bbox(self, var: str) -> Tuple[float, float, float, float]:
        da = self.ds[var]
        return (
            float(da.cf["longitude"].min()),
            float(da.cf["latitude"].min()),
            float(da.cf["longitude"].max()),
            float(da.cf["latitude"].max()),
        )


def grid_factory(ds: xr.Dataset) -> Optional[Grid]:
    grid_impls = [ROMSGrid, RegularGrid]
    for grid_impl in grid_impls:
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
    def name(self) -> str:
        if self._grid is None:
            return "unsupported"
        else:
            return self.grid.name
        
    @property
    def bbox(self, var) -> Tuple[float, float, float, float]:
        if self._grid is None:
            return None
        else:
            return self.grid.bbox(var)
        
    @property
    def crs(self) -> str:
        if self._grid is None:
            return None
        else:
            return self.grid.crs


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
