from typing import Optional, Sequence, Tuple, Union

import cf_xarray  # noqa
import numpy as np
import xarray as xr

from xpublish_wms.grids.fvcom import FVCOMGrid
from xpublish_wms.grids.grid import Grid, RenderMethod
from xpublish_wms.grids.hycom import HYCOMGrid
from xpublish_wms.grids.irregular import IrregularGrid
from xpublish_wms.grids.regular import RegularGrid
from xpublish_wms.grids.roms import ROMSGrid
from xpublish_wms.grids.selfe import SELFEGrid

_grid_impls = [
    HYCOMGrid,
    FVCOMGrid,
    SELFEGrid,
    ROMSGrid,
    IrregularGrid,
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

    def additional_coords(self, da: xr.DataArray) -> list[str]:
        if self._grid is None:
            return None
        else:
            return self._grid.additional_coords(da)

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
