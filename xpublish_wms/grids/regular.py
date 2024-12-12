from typing import Optional

import xarray as xr

from xpublish_wms.grids.grid import Grid, RenderMethod
from xpublish_wms.utils import lnglat_to_mercator


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

    def project(
        self,
        da: xr.DataArray,
        crs: str,
    ) -> tuple[xr.DataArray, Optional[xr.DataArray], Optional[xr.DataArray]]:
        da = self.mask(da)

        coords = dict()
        # need to convert longitude and latitude to x and y for the mesh to work properly
        # regular grid doesn't have x & y as dimensions so have to remake the whole data array
        for coord in da.coords:
            if coord != da.cf["longitude"].name and coord != da.cf["latitude"].name:
                coords[coord] = da.coords[coord]

        # normalize longitude to be between -180 and 180
        da = da.cf.assign_coords(
            longitude=(((da.cf["longitude"] + 180) % 360) - 180),
        ).sortby(da.cf["longitude"].name)

        # build new x coordinate
        coords["x"] = ("x", da.cf["longitude"].values, da.cf["longitude"].attrs)
        # build new y coordinate
        coords["y"] = ("y", da.cf["latitude"].values, da.cf["latitude"].attrs)
        # build new data array
        shape = da.shape
        if shape == (len(da.cf["latitude"]), len(da.cf["longitude"])):
            dims = ("y", "x")
        else:
            dims = ("x", "y")

        da = xr.DataArray(
            data=da,
            dims=dims,
            coords=coords,
            name=da.name,
            attrs=da.attrs,
        )

        # convert to mercator
        if crs == "EPSG:3857":
            lng, lat = lnglat_to_mercator(da.cf["longitude"], da.cf["latitude"])

            da = da.assign_coords({"x": lng, "y": lat})
            da = da.unify_chunks()

        return da, None, None

    def sel_lat_lng(
        self,
        subset: xr.Dataset,
        lng,
        lat,
        parameters,
    ) -> tuple[xr.Dataset, list, list]:
        # normalize longitude to be between -180 and 180
        subset = subset.cf.assign_coords(
            longitude=(((subset.cf["longitude"] + 180) % 360) - 180),
        ).sortby(subset.cf["longitude"].name)

        if lng > 180:
            lng = (lng + 180) % 360 - 180

        return super().sel_lat_lng(subset, lng, lat, parameters)
