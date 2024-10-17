from abc import ABC, abstractmethod
from enum import Enum
from typing import Optional, Sequence, Union

import numpy as np
import xarray as xr

from xpublish_wms.utils import strip_float


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

    def bbox(self, da: xr.DataArray) -> tuple[float, float, float, float]:
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

    def additional_coords(self, da: xr.DataArray) -> list[str]:
        """Return the additional coordinate dimensions for the given data array

        Given a data array with a shape of (time, latitude, longitude),
        this function would return [].

        Given a data array with a shape of (time, latitude, longitude, vertical),
        this function would return [].

        Given a data array with a shape of (band, latitude, longitude),
        this function would return ["band"].
        """
        lat_dim = da.cf["latitude"].name
        lng_dim = da.cf["longitude"].name
        filter_dims = [lat_dim, lng_dim]

        time_dim = da.cf.coords.get("time", None)
        if time_dim is not None:
            filter_dims.append(time_dim.name)

        elevation_dim = da.cf.coords.get("vertical", None)
        if elevation_dim is not None:
            filter_dims.append(elevation_dim.name)

        return [dim for dim in da.dims if dim not in filter_dims]

    def mask(
        self,
        da: Union[xr.DataArray, xr.Dataset],
    ) -> Union[xr.DataArray, xr.Dataset]:
        """Mask the given data array"""
        return da

    @abstractmethod
    def project(
        self,
        da: xr.DataArray,
        crs: str,
    ) -> tuple[xr.DataArray, Optional[xr.DataArray], Optional[xr.DataArray]]:
        """Project the given data array from this dataset and grid to the given crs

        returns the projected data array and optionally the x and y coordinates if they are required for triangulation
        TODO: This is a leaky abstraction and should be refactored. It was added specifically for odd FVCOM datasets
        that do not have the neighbor connectivity information in the dataset.
        """
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
    ) -> tuple[xr.Dataset, list, list]:
        """Select the given dataset by the given lon/lat and optional elevation"""
        subset = self.mask(subset[parameters])

        subset = subset.cf.interp(
            longitude=lng,
            latitude=lat,
        )

        x_axis = [strip_float(subset.cf["longitude"])]
        y_axis = [strip_float(subset.cf["latitude"])]
        return subset, x_axis, y_axis
