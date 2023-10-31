import datetime as dt

import cachey
import cf_xarray  # noqa
import xarray as xr
from fastapi import HTTPException, Response
from fastapi.responses import JSONResponse

from xpublish_wms.utils import format_timestamp

from .get_map import GetMap


def get_metadata(ds: xr.Dataset, cache: cachey.Cache, params: dict) -> Response:
    """
    Return the WMS metadata for the dataset

    This is compliant subset of ncwms2's GetMetadata handler. Specifically, layerdetails, timesteps and minmax are supported.
    """
    layer_name = params.get("layername", None)
    metadata_type = params.get("item", "layerdetails").lower()

    if not layer_name and metadata_type != "minmax" and metadata_type != "menu":
        raise HTTPException(
            status_code=400,
            detail="layerName must be specified",
        )
    elif layer_name not in ds and metadata_type != "minmax" and metadata_type != "menu":
        raise HTTPException(
            status_code=400,
            detail=f"layerName {layer_name} not found in dataset",
        )

    if metadata_type == "menu":
        payload = get_menu(ds)
    elif metadata_type == "layerdetails":
        payload = get_layer_details(ds, layer_name)
    elif metadata_type == "timesteps":
        da = ds[layer_name]
        payload = get_timesteps(da, params)
    elif metadata_type == "minmax":
        payload = get_minmax(ds, cache, params)
    else:
        raise HTTPException(
            status_code=400,
            detail=f"item {metadata_type} not supported",
        )

    return JSONResponse(content=payload)


def get_timesteps(da: xr.DataArray, params: dict) -> dict:
    """
    Returns the timesteps for a given layer
    """
    if "time" not in da.cf:
        raise HTTPException(
            status_code=400,
            detail=f"layer {da.name} does not have a time dimension",
        )

    day = params.get("day", None)
    if day:
        day_start = dt.datetime.strptime(day, "%Y-%m-%d")
        day_start = day_start.replace(hour=0, minute=0, second=0, microsecond=0)
        day_end = day_start + dt.timedelta(days=1)
        da = da.cf.sel(time=slice(day_start, day_end))

    range = params.get("range", None)
    if range:
        start, end = range.split("/")
        start = dt.datetime.strptime(start, "%Y-%m-%dT%H:%M:%SZ")
        end = dt.datetime.strptime(end, "%Y-%m-%dT%H:%M:%SZ")
        da = da.cf.sel(time=slice(start, end))

    timesteps = format_timestamp(da.cf["time"]).tolist()
    return {
        "timesteps": timesteps,
    }


def get_minmax(ds: xr.Dataset, cache: cachey.Cache, params: dict) -> dict:
    """
    Returns the min and max range of values for a given layer in a given area

    If BBOX is not specified, the entire selected temporal and elevation range is used.
    """
    getmap = GetMap(cache=cache)
    return getmap.get_minmax(ds, params)


def get_layer_details(ds: xr.Dataset, layer_name: str) -> dict:
    """
    Returns a subset of layer details for the requested layer
    """
    da = ds[layer_name]
    units = da.attrs.get("units", "")
    supported_styles = "raster"  # TODO: more styles
    bbox = ds.gridded.bbox(da)
    if ds.gridded.has_elevation(da):
        elevation = ds.gridded.elevations(da).values.round(5).tolist()
        elevation_positive = ds.gridded.elevation_positive_direction(da)
        elevation_units = ds.gridded.elevation_units(da)
    else:
        elevation = None
        elevation_positive = None
        elevation_units = None
    if "time" in da.cf:
        timesteps = format_timestamp(da.cf["time"]).tolist()
    else:
        timesteps = None

    return {
        "layerName": da.name,
        "standard_name": da.cf.attrs.get("standard_name", da.name),
        "long_name": da.cf.attrs.get("long_name", da.name),
        "bbox": bbox,
        "units": units,
        "supportedStyles": [supported_styles],
        "elevation": elevation,
        "elevation_positive": elevation_positive,
        "elevation_units": elevation_units,
        "timesteps": timesteps,
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
