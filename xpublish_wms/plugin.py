import logging
from typing import List

import cachey
import xarray as xr
from fastapi import APIRouter, Depends, Request
from xpublish import Dependencies, Plugin, hookimpl

from .wms import wms_handler

logger = logging.getLogger("uvicorn")


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

        @router.get("/")
        def wms_root(
            request: Request,
            dataset: xr.Dataset = Depends(deps.dataset),
            cache: cachey.Cache = Depends(deps.cache),
        ):
            return wms_handler(request, dataset, cache)

        return router
