from typing import Optional

import numpy as np
import xarray as xr

from xpublish_wms.grids.grid import Grid, RenderMethod
from xpublish_wms.utils import lnglat_to_mercator, to_lnglat_allow_over


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
        render_context: Optional[dict] = dict(),
    ) -> tuple[xr.DataArray, Optional[xr.DataArray], Optional[xr.DataArray]]:
        if not render_context.get("masked", False):
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

        return da, render_context

    def filter_by_bbox(self, da, bbox, crs, render_context: Optional[dict] = dict()):
        """Subset the data array by the given bounding box.
        Also normalizes the longitude to be between -180 and 180.
        """
        
        da = self.mask(da)
        render_context["masked"] = True

        if crs == "EPSG:3857":
            bbox = to_lnglat_allow_over.transform(
                [bbox[0], bbox[2]],
                [bbox[1], bbox[3]],
            )
            bbox = [bbox[0][0], bbox[1][0], bbox[0][1], bbox[1][1]]

        # normalize longitude to be between -180 and 180
        da = da.cf.assign_coords(
            longitude=(((da.cf["longitude"] + 180) % 360) - 180),
        ).sortby(da.cf["longitude"].name)

        # Get the x and y values
        x = da.cf["longitude"]
        y = da.cf["latitude"]

        # Find the indices of the data within the bounding box
        x_inds = np.where((x >= bbox[0]) & (x <= bbox[2]))[0]
        y_inds = np.where((y >= bbox[1]) & (y <= bbox[3]))[0]

        # Select and return the data within the bounding box
        da = da.isel(
            {da.cf["longitude"].dims[0]: x_inds, da.cf["latitude"].dims[0]: y_inds},
        )
        return da, render_context

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
