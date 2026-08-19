from typing import Optional, Union

import dask.array as dask_array
import numpy as np
import xarray as xr

from xpublish_wms.grids.grid import Grid, RenderMethod
from xpublish_wms.utils import (
    bilinear_interp,
    lat_lng_find_quad,
    lat_lng_quad_percentage,
    strip_float,
    to_lnglat_allow_over,
    to_mercator,
)


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
        return RenderMethod.Triangle

    @property
    def crs(self) -> str:
        return "EPSG:4326"

    def grid_mask(self, grid_point: str) -> Optional[xr.DataArray]:
        """The land/sea mask for one of the staggered grids, eg 'rho' or 'u'."""

        # use the wet/dry mask when available, otherwise use the standard mask.
        mask = self.ds.get(f"wetdry_mask_{grid_point}")
        if mask is None:
            mask = self.ds.get(f"mask_{grid_point}")
        if mask is None:
            return None

        if "time" in mask.cf.coords:
            mask = mask.cf.isel(time=0).squeeze(drop=True).cf.drop_vars("time")
        else:
            mask = mask.cf.squeeze(drop=True)

        return mask

    def mask(
        self,
        da: Union[xr.DataArray, xr.Dataset],
    ) -> Union[xr.DataArray, xr.Dataset]:
        mask = self.grid_mask(da.cf["latitude"].name.split("_")[-1])
        if mask is None:
            return da

        return da.where(mask == 1)

    def project(
        self,
        da: xr.DataArray,
        crs: str,
        render_context: Optional[dict] = dict(),
    ) -> tuple[xr.DataArray, Optional[xr.DataArray], Optional[xr.DataArray]]:
        if not render_context.get("masked", False):
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

        return da, render_context

    def tessellate(
        self,
        da: xr.DataArray,
        render_context: Optional[dict] = {},
    ) -> tuple[np.ndarray, dict]:
        """
        Builds the ROMS cells in a way that exactly matches the native grids for tested ROMS models (CBOFS, DBOFS, TBOFS, etc.)

        There are two main ways to interpret the rho points of a ROMS grid as quads.
        The old way in this plugin (using quadmesh) treated rho points as the cell centers,
        and expanded the grid by one cell in each direction to get the corners of each cell.
        However, the output grid does not match the real ROMS native grid when using this approach.

        We now interpret rho points is as the corners of the cells, which exactly matches
        the native ROMS grid. The complication is that the ROMS grid has masked cells,
        and the approach for dealing with these masked cells varies from model to model.

        On most models, the masked cells have valid lat/lng even in the mask, but the values
        of masked cells are set to -9999. To render these models, we simply look at the
        lower left corner `pts[i,j]` of each potential cell, and if that corner has valid data,
        we render a cell `pts[i,j] / [i,j+1] / [i+1,j+1] / [i+1,j]` as two triangles and fill
        the cell with the value of the lower-left corner `data[i,j]`.

        However, certain ROMS models (CBOFS, DBOFS, NYOFS are known examples)
        have masked cells that do not have valid lat/lng, and instead have wildly varying
        lat/lng values. If we render these models using the same approach as above,
        we will end up with many broken / smeared cells. For these models specifically,
        we need to check if ALL FOUR corners of a potential cell have valid data, instead of
        just the lower-left corner.

        Even if there technically are better ways of rendering this grid, the above approach is
        the only way to exactly match the native ROMS grid for all tested models. Another benefit
        is that directly rendering the quads as triangles is considerably faster than using quadmesh,
        which has to infer the corners of each cell.
        """
        x = np.asarray(da.x.values)
        y = np.asarray(da.y.values)
        z = np.asarray(da.values)

        n_eta, n_xi = z.shape

        # Get the corners of each cell
        corner_ll = z[:-1, :-1]
        corner_lr = z[:-1, 1:]
        corner_ur = z[1:, 1:]
        corner_ul = z[1:, :-1]

        # check all corners for the "special case" ROMS models that have invalid lat/lng for masked cells (see docstring)
        # this is the worst hack I have ever written in my career, please forgive me
        # TODO figure out if there's a better way to detect these models than checking the dataset name
        if any(
            sub in self.ds.attrs.get("title", "").lower()
            for sub in ["cbofs", "dbofs", "nyofs"]
        ):
            valid = (
                np.isfinite(corner_ll)
                & np.isfinite(corner_lr)
                & np.isfinite(corner_ur)
                & np.isfinite(corner_ul)
            )
        else:
            valid = np.isfinite(corner_ll)

        eta, xi = np.nonzero(valid)

        # Flattened (row-major) index of each rho corner into the vertex table
        v_ll = eta * n_xi + xi
        v_lr = eta * n_xi + (xi + 1)
        v_ur = (eta + 1) * n_xi + (xi + 1)
        v_ul = (eta + 1) * n_xi + xi

        # Split each quad along one diagonal into two triangles
        triangles = np.empty((2 * eta.size, 3), dtype=np.int64)
        triangles[0::2] = np.stack([v_ll, v_lr, v_ur], axis=1)
        triangles[1::2] = np.stack([v_ll, v_ur, v_ul], axis=1)

        # Both triangles of a quad take the lower-left corner's value
        # TODO do we want to instead take an average of the valid corner values?
        # more accurate, but slower
        cell_z = corner_ll[eta, xi]
        tri_z = np.repeat(cell_z, 2)

        render_context["tri_x"] = x.reshape(-1)
        render_context["tri_y"] = y.reshape(-1)
        render_context["tri_z"] = tri_z

        return triangles, render_context

    def filter_by_bbox(self, da, bbox, crs, render_context: Optional[dict] = {}):
        da = self.mask(da)
        render_context["masked"] = True

        if crs == "EPSG:3857":
            bbox = to_lnglat_allow_over.transform(
                [bbox[0], bbox[2]],
                [bbox[1], bbox[3]],
            )
            bbox = [bbox[0][0], bbox[1][0], bbox[0][1], bbox[1][1]]

        # Get the x and y values
        x = da.cf["longitude"]
        y = da.cf["latitude"]

        if x.dims != y.dims or x.ndim != 2:
            raise Exception("Mismatched dims for filter_by_bbox")

        # masked/land cells can carry filler positions (eg. in cbofs, dbofs), so they cannot be trusted to say
        # where the grid is or whether they are in view
        mask = self.grid_mask(x.name.split("_")[-1])
        known = None
        if mask is not None and mask.shape == x.shape:
            known = np.asarray(mask.values) == 1

        lng = np.asarray(x.values)
        lat = np.asarray(y.values)

        adjust_lng = 0
        in_grid = lng if known is None else lng[known]
        if in_grid.size:
            if np.min(in_grid) < -180:
                adjust_lng = 360
            elif np.max(in_grid) > 180:
                adjust_lng = -360

        lng = lng + adjust_lng

        # check which cells are in view using a joint lng/lat test to prevent mixing up rows and columns
        hits = (
            (lng >= bbox[0] - 0.0)
            & (lng <= bbox[2] + 0.0)
            & (lat >= bbox[1] - 0.0)
            & (lat <= bbox[3] + 0.0)
        )

        inside = hits if known is None else hits & known

        if not inside.any():
            # A bbox smaller than a grid cell can sit entirely between the corners of a single cell.
            # I have a working approach for dealing with this edge case, but it adds a lot of complexity for a case we
            # are extremely unlikely to ever actually see
            # this also catches tiles that are completely outside the grid, which is a more common case
            raise Exception("No fully visible cells in bbox (skipping)")

        # Take a contiguous window around the visible cells rather than the
        # individual indices: dropping interior rows/columns would stitch cells
        # together that are nowhere near each other on the grid
        sel_dims = {}
        for axis, dim in enumerate(x.dims):
            hits = np.where(inside.any(axis=1 - axis))[0]
            if hits.size == 0:
                sel_dims = {dim: slice(0, 0) for dim in x.dims}
                break

            low = max(int(hits[0]) - 2, 0)
            high = min(int(hits[-1]) + 3, inside.shape[axis])

            # Keep at least one full cell in each direction, otherwise the data
            # squeezes down to a line and can no longer be rendered as quads
            if high - low < 2:
                low = max(high - 2, 0)
                high = min(low + 2, inside.shape[axis])

            sel_dims[dim] = slice(low, high)

        da = da.isel(sel_dims)
        return da, render_context

    def sel_lat_lng(
        self,
        subset: xr.Dataset,
        lng,
        lat,
        parameters,
    ) -> tuple[xr.Dataset, list, list]:
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
