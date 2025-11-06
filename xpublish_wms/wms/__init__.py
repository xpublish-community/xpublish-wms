"""
OGC WMS router for datasets with CF convention metadata
"""

import asyncio
from typing import Union

import cachey
import cf_xarray  # noqa
import xarray as xr
from fastapi import HTTPException, Request, Response

from xpublish_wms.logger import logger
from xpublish_wms.query import (
    WMSGetCapabilitiesQuery,
    WMSGetFeatureInfoQuery,
    WMSGetLegendInfoQuery,
    WMSGetMapQuery,
    WMSGetMetadataQuery,
)
from xpublish_wms.wms.get_map import GetMap

from .get_capabilities import get_capabilities
from .get_feature_info import get_feature_info
from .get_legend_info import get_legend_info
from .get_metadata import get_metadata


async def wms_handler(
    request: Request,
    query: Union[
        WMSGetCapabilitiesQuery,
        WMSGetMetadataQuery,
        WMSGetMapQuery,
        WMSGetFeatureInfoQuery,
        WMSGetLegendInfoQuery,
    ],
    extra_query_params: dict,
    dataset: xr.Dataset,
    array_get_map_render_threshold_bytes: int,
    cache: cachey.Cache | None = None,
) -> Response:
    logger.debug(f"Received wms request: {request.url}")

    match query:
        case WMSGetCapabilitiesQuery():
            return await asyncio.to_thread(get_capabilities, dataset, request, query)
        case WMSGetMetadataQuery():
            return await get_metadata(
                dataset,
                cache,
                query,
                extra_query_params,
                array_get_map_render_threshold_bytes=array_get_map_render_threshold_bytes,
            )
        case WMSGetMapQuery():
            getmap_service = GetMap(
                cache=cache,
                array_render_threshold_bytes=array_get_map_render_threshold_bytes,
            )
            return await getmap_service.get_map(dataset, query, extra_query_params)
        case WMSGetFeatureInfoQuery():
            return await asyncio.to_thread(
                get_feature_info,
                dataset,
                query,
                extra_query_params,
            )
        case WMSGetLegendInfoQuery():
            return await asyncio.to_thread(get_legend_info, dataset, query)
        case _:
            raise HTTPException(
                status_code=404,
                detail=f"Unknown WMS request: {request.query_params}",
            )
