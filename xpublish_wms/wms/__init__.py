"""
OGC WMS router for datasets with CF convention metadata
"""

import logging

import cachey
import cf_xarray  # noqa
import xarray as xr
from fastapi import Depends, HTTPException, Request, Response
from xpublish.dependencies import get_cache, get_dataset

from xpublish_wms.utils import lower_case_keys
from xpublish_wms.wms.get_map import GetMap

from .get_capabilities import get_capabilities
from .get_feature_info import get_feature_info
from .get_legend_info import get_legend_info
from .get_metadata import get_metadata

logger = logging.getLogger("uvicorn")


def wms_handler(
    request: Request,
    dataset: xr.Dataset = Depends(get_dataset),
    cache: cachey.Cache = Depends(get_cache),
) -> Response:
    query_params = lower_case_keys(request.query_params)
    method = query_params.get("request", "").lower()
    logger.info(f"WMS: {method}")
    if method == "getcapabilities":
        return get_capabilities(dataset, request, query_params)
    elif method == "getmap":
        getmap_service = GetMap(cache=cache)
        return getmap_service.get_map(dataset, query_params)
    elif method == "getfeatureinfo" or method == "gettimeseries":
        return get_feature_info(dataset, query_params)
    elif method == "getverticalprofile":
        query_params["elevation"] = "all"
        return get_feature_info(dataset, query_params)
    elif method == "getmetadata":
        return get_metadata(dataset, cache, query_params)
    elif method == "getlegendgraphic":
        return get_legend_info(dataset, query_params)
    else:
        raise HTTPException(
            status_code=404,
            detail=f"{method} is not a valid option for REQUEST",
        )
