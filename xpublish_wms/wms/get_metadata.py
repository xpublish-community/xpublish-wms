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

    This is compliant with ncwms2's GetMetadata handler
    """
    layer_name = params.get("layername", None)
    if not layer_name:
        raise HTTPException(
            status_code=400,
            detail="layerName must be specified",
        )
    elif layer_name not in ds:
        raise HTTPException(
            status_code=400,
            detail=f"layerName {layer_name} not found in dataset",
        )

    da = ds[layer_name]

    metadata_type = params.get("item", "layerDetails")
    if metadata_type == "layerDetails":
        payload = {}
    elif metadata_type == "timesteps":
        payload = get_timesteps(da, params)
    elif metadata_type == 'minmax':
        payload = get_minmax(da, cache, params)

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


def get_minmax(da: xr.DataArray, cache: cachey.Cache, params: dict) -> dict:
    '''
    Returns the min and max range of values for a given layer in a given area
    '''
    getmap = GetMap(cache=cache)
    return getmap.get_minmax(da, params)