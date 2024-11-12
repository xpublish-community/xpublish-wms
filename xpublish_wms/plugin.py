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
            return wms_handler(request, wms_query, dataset, cache)

        return router
