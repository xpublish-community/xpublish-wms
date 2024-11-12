"""
OGC WMS router for datasets with CF convention metadata
"""

import logging
from typing import Union

import cachey
import cf_xarray  # noqa
import xarray as xr
from fastapi import HTTPException, Request, Response

from xpublish_wms.query import (
    WMS_FILTERED_QUERY_PARAMS,
    WMSGetCapabilitiesQuery,
    WMSGetFeatureInfoQuery,
    WMSGetLegendInfoQuery,
    WMSGetMapQuery,
    WMSGetMetadataQuery,
)
from xpublish_wms.utils import lower_case_keys
from xpublish_wms.wms.get_map import GetMap

from .get_capabilities import get_capabilities
from .get_feature_info import get_feature_info
from .get_legend_info import get_legend_info
from .get_metadata import get_metadata

logger = logging.getLogger("uvicorn")


def wms_handler(
    request: Request,
    query: Union[
        WMSGetCapabilitiesQuery,
        WMSGetMetadataQuery,
        WMSGetMapQuery,
        WMSGetFeatureInfoQuery,
        WMSGetLegendInfoQuery,
    ],
    dataset: xr.Dataset,
    cache: cachey.Cache,
) -> Response:
    query_params = lower_case_keys(request.query_params)
    query_keys = list(query_params.keys())
    for query_key in query_keys:
        if query_key in WMS_FILTERED_QUERY_PARAMS:
            del query_params[query_key]

    logger.debug(f"Received wms request: {request.url}")

    if isinstance(query, WMSGetCapabilitiesQuery):
        return get_capabilities(dataset, request, query)
    elif isinstance(query, WMSGetMetadataQuery):
        return get_metadata(dataset, cache, query, query_params)
    elif isinstance(query, WMSGetMapQuery):
        getmap_service = GetMap(cache=cache)
        return getmap_service.get_map(dataset, query, query_params)
    elif isinstance(query, WMSGetFeatureInfoQuery):
        return get_feature_info(dataset, query, query_params)
    elif isinstance(query, WMSGetLegendInfoQuery):
        return get_legend_info(dataset, query)
    else:
        raise HTTPException(
            status_code=404,
            detail=f"Unknown WMS request: {request.query_params}",
        )
