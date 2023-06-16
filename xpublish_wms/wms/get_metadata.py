import cf_xarray # noqa
from fastapi import HTTPException, Response 
from fastapi.responses import JSONResponse
import xarray as xr
import datetime as dt
import cachey

from xpublish_wms.utils import format_timestamp
from .get_map import GetMap

def get_metadata(ds: xr.Dataset, cache: cachey.Cache, params: dict) -> Response: 
    """
    Return the WMS metadata for the dataset

    This is compliant subset of ncwms2's GetMetadata handler. Specifically, timesteps and minmax are supported.
    """
    layer_name = params.get("layername", None)
    metadata_type = params.get("item", "minmax")
    
    if not layer_name and metadata_type != 'minmax':
        raise HTTPException(
            status_code=400,
            detail="layerName must be specified",
        )
    elif layer_name not in ds and metadata_type != 'minmax':
        raise HTTPException(
            status_code=400,
            detail=f"layerName {layer_name} not found in dataset",
        )

    if metadata_type == "timesteps":
        da = ds[layer_name]
        payload = get_timesteps(da, params)
    elif metadata_type == 'minmax':
        payload = get_minmax(ds, cache, params)
    else: 
        raise HTTPException(
            status_code=400,
            detail=f"item {metadata_type} not supported",
        )

    return JSONResponse(content=payload)


def get_timesteps(da: xr.DataArray, params: dict) -> dict:
    '''
    Returns the timesteps for a given layer
    '''
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
    '''
    Returns the min and max range of values for a given layer in a given area

    If BBOX is not specified, the entire selected temporal and elevation range is used. 
    '''
    getmap = GetMap(cache=cache)
    return getmap.get_minmax(ds, params)
