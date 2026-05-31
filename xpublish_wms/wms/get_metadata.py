import datetime as dt
from typing import List

import cachey
import cf_xarray  # noqa
import xarray as xr
from fastapi import HTTPException, Response
from fastapi.responses import JSONResponse

from xpublish_wms.logger import logger
from xpublish_wms.query import (
    GET_MAP_STYLE_METHODS,
    WMSGetMapQuery,
    WMSGetMetadataQuery,
)
from xpublish_wms.utils import format_timestamp

from .get_map import GetMap


def get_metadata(
    ds: xr.Dataset,
    cache: cachey.Cache,
    query: WMSGetMetadataQuery,
    query_params: dict,
    array_get_map_render_threshold_bytes: int,
) -> Response:
    """
    Return the WMS metadata for the dataset

    This is compliant subset of ncwms2's GetMetadata handler. Specifically, layerdetails, timesteps and minmax are supported.
    """
    metadata_type = query.item.lower()

    if metadata_type in ["minmax", "menu"]:
        pass  # minmax and menu do not require extra layers validation here
    elif not query.layers:
        logger.error("layerName must be specified for GetMetadata requests")
        raise HTTPException(
            422,
            detail="layerName must be specified",
        )
    elif any(layer not in ds for layer in query.layers):
        not_found = [layer for layer in query.layers if layer not in ds]
        logger.error(f"layers {not_found} were not found in dataset")
        raise HTTPException(
            422,
            detail=f"layer name(s) {', '.join(not_found)} not found in dataset",
        )

    if metadata_type == "menu":
        payload = get_menu(ds)
    elif metadata_type == "layerdetails":
        payload = get_layer_details(ds, query.layers)
    elif metadata_type == "timesteps":
        # If there are multiple layers, we assume they are vector components,
        # and therefore should have the same timesteps, so we use the first layer name
        first_layer = query.layers[0]
        payload = get_timesteps(ds[first_layer], query)
    elif metadata_type == "minmax":
        payload = get_minmax(
            ds,
            cache,
            query,
            query_params,
            array_get_map_render_threshold_bytes,
        )
    else:
        logger.error(f"item {metadata_type} not supported for GetMetadata requests")
        raise HTTPException(
            422,
            detail=f"item {metadata_type} not supported",
        )

    return JSONResponse(content=payload)


def get_timesteps(da: xr.DataArray, query: WMSGetMetadataQuery) -> dict:
    """
    Returns the timesteps for a given layer
    """
    if "time" not in da.cf:
        logger.error(f"layer {da.name} does not have a time dimension")
        raise HTTPException(
            422,
            detail=f"layer {da.name} does not have a time dimension",
        )

    day = query.day
    if day:
        day_start = dt.datetime.strptime(day, "%Y-%m-%d")
        day_start = day_start.replace(hour=0, minute=0, second=0, microsecond=0)
        day_end = day_start + dt.timedelta(days=1)
        da = da.cf.sel(time=slice(day_start, day_end))

    range = query.range
    if range:
        start, end = range.split("/")
        start = dt.datetime.strptime(start, "%Y-%m-%dT%H:%M:%SZ")
        end = dt.datetime.strptime(end, "%Y-%m-%dT%H:%M:%SZ")
        da = da.cf.sel(time=slice(start, end))

    timesteps = format_timestamp(da.cf["time"]).tolist()
    return {
        "timesteps": timesteps,
    }


def get_minmax(
    ds: xr.Dataset,
    cache: cachey.Cache,
    query: WMSGetMetadataQuery,
    query_params: dict,
    array_get_map_render_threshold_bytes: int,
) -> dict:
    """
    Returns the min and max range of values for a given layer in a given area

    If BBOX is not specified, the entire selected temporal and elevation range is used.
    """
    entire_layer = query.bbox is None
    getmap_query = WMSGetMapQuery(
        service=query.service,
        version=query.version,
        request="GetMap",
        layers=query.layers and ",".join(query.layers),
        bbox=query.bbox if not entire_layer else "-180,-90,180,90",
        width=1 if entire_layer else 512,
        height=1 if entire_layer else 512,
        crs=query.crs,
        time=query.time,
        elevation=query.elevation,
        styles="raster/default" if len(query.layers or []) == 1 else "vector/none",
        colorscalerange="nan,nan",
    )

    getmap = GetMap(
        cache=cache,
        array_render_threshold_bytes=array_get_map_render_threshold_bytes,
    )
    return getmap.get_minmax(ds, getmap_query, query_params, entire_layer)


def get_layer_details(ds: xr.Dataset, layers: List[str]) -> dict:
    """
    Returns a subset of layer details for the requested layers
    """
    das = [ds[layer] for layer in layers]

    # We mostly just assume that multiple layers are vector components and that
    # the client knows what they're doing but here we do a simple validation for it
    all_units = {da.attrs.get("units", "") for da in das}
    if len(all_units) > 1:
        raise HTTPException(422, "Selected layers have different units")
    units = next(iter(all_units))

    supported_styles = [
        s
        for s in GET_MAP_STYLE_METHODS
        if s.startswith("raster" if len(das) == 1 else "vector")
    ]

    # Otherwise take metadata from the first layer and assume second is the same
    da1 = das[0]
    bbox = ds.gridded.bbox(da1)
    if ds.gridded.has_elevation(da1):
        elevation = ds.gridded.elevations(da1).values.round(5).tolist()
        elevation_positive = ds.gridded.elevation_positive_direction(da1)
        elevation_units = ds.gridded.elevation_units(da1)
    else:
        elevation = None
        elevation_positive = None
        elevation_units = None
    if "time" in da1.cf:
        timesteps = format_timestamp(da1.cf["time"]).tolist()
    else:
        timesteps = None

    additional_coords = ds.gridded.additional_coords(da1)
    additional_coord_values = {
        coord: (
            da1.cf.coords[coord] if coord in da1.cf.coords else da1[coord]
        ).values.tolist()
        for coord in additional_coords
    }

    return {
        "layerName": ",".join(str(da.name) for da in das),
        "standard_name": ",".join(
            da.cf.attrs.get("standard_name", da.name) for da in das
        ),
        "long_name": ",".join(da.cf.attrs.get("long_name", da.name) for da in das),
        "bbox": bbox,
        "units": units,
        "supportedStyles": supported_styles,
        "elevation": elevation,
        "elevation_positive": elevation_positive,
        "elevation_units": elevation_units,
        "timesteps": timesteps,
        "additional_coords": additional_coords,
        **additional_coord_values,
    }


def get_menu(ds: xr.Dataset):
    """
    Returns the dataset menu items for the xreds viewer
    TODO - support grouped layers?
    """
    results = {"children": [], "label": ds.attrs.get("title", "")}

    for var in ds.data_vars:
        da = ds[var]
        results["children"].append(
            {
                "plottable": "longitude" in da.cf.coords,
                "id": var,
                "label": da.attrs.get("long_name", da.attrs.get("name", var)),
            },
        )

    return results
