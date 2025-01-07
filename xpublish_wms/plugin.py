import logging
from typing import List, Union

import cachey
import xarray as xr
from fastapi import APIRouter, Depends, Request
from xpublish import Dependencies, Plugin, hookimpl

from xpublish_wms.query import (
    WMSGetCapabilitiesQuery,
    WMSGetFeatureInfoQuery,
    WMSGetLegendInfoQuery,
    WMSGetMapQuery,
    WMSGetMetadataQuery,
    wms_query,
)

from .wms import wms_handler

logger = logging.getLogger("uvicorn")
xr.set_options(keep_attrs=True)


class CfWmsPlugin(Plugin):
    """
    OGC WMS plugin for xpublish
    """

    name: str = "cf_wms"

    dataset_router_prefix: str = "/wms"
    dataset_router_tags: List[str] = ["wms"]

    # Limit for rendering arrays in get_map after subsetting to the requested
    # bounding box. If the array is larger than this threshold, an error will be thrown.
    # Default is 1e9 bytes (1 GB)
    array_get_map_render_threshold_bytes: int = 1e9

    @hookimpl
    def dataset_router(self, deps: Dependencies) -> APIRouter:
        """Register dataset level router for WMS endpoints"""

        router = APIRouter(
            prefix=self.dataset_router_prefix,
            tags=self.dataset_router_tags,
        )

        @router.get("", include_in_schema=False)
        @router.get("/")
        def wms_root(
            request: Request,
            wms_query: Union[
                WMSGetCapabilitiesQuery,
                WMSGetMetadataQuery,
                WMSGetMapQuery,
                WMSGetFeatureInfoQuery,
                WMSGetLegendInfoQuery,
            ] = Depends(wms_query),
            dataset: xr.Dataset = Depends(deps.dataset),
            cache: cachey.Cache = Depends(deps.cache),
        ):
            # TODO: Make threshold configurable
            return wms_handler(
                request,
                wms_query,
                dataset,
                cache,
                array_get_map_render_threshold_bytes=self.array_get_map_render_threshold_bytes,
            )

        return router
