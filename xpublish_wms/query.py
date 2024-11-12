from typing import Literal, Optional, Union

from fastapi import Query
from pydantic import BaseModel, Field


class WMSBaseQuery(BaseModel):
    service: Literal["WMS"] = Field(..., description="Service type. Must be WMS")
    version: Literal["1.1.1", "1.3.0"] = Field(
        ...,
        description="Version of the WMS service",
    )


class WMSGetCapabilitiesQuery(WMSBaseQuery):
    """WMS GetCapabilities query"""

    request: Literal["GetCapabilities"] = Field(..., description="Request type")


class WMSGetMetadataQuery(WMSBaseQuery):
    """WMS GetMetadata query"""

    request: Literal["GetMetadata"] = Field(..., description="Request type")
    layername: Optional[str] = Field(
        None,
        description="Name of the layer to get metadata for",
    )
    item: Literal["layerdetails", "timesteps", "minmax", "menu"] = Field(
        ...,
        description="The type of GetMetadata request",
    )
    day: Optional[str] = Field(
        None,
        description="Optional day to get timesteps for in Y-m-d format. Only valid when item=timesteps and layer has a time dimension",
    )
    range: Optional[str] = Field(
        None,
        description="Optional range to get timesteps for in Y-m-dTH:M:SZ/Y-m-dTH:M:SZ format. Only valid when item=timesteps and layer has a time dimension",
    )
    bbox: Optional[str] = (
        Field(
            None,
            description="Bounding box to use for calculating min and max in the format 'minx,miny,maxx,maxy'",
        ),
    )
    time: Optional[str] = (
        Field(
            None,
            description="Optional time to get the min and max for in Y-m-dTH:M:SZ format. Only valid when the layer has a time dimension",
        ),
    )
    elevation: Optional[str] = Field(
        None,
        description="Optional elevation to get the min and max for. Only valid when the layer has an elevation dimension",
    )


class WMSGetMapQuery(WMSBaseQuery):
    """WMS GetMap query"""

    request: Literal["GetMap"] = Field(..., description="Request type")
    layers: str
    styles: str = Field(
        "raster/default",
        description="Style to use for the query. Defaults to raster/default. Default may be replaced by the name of any colormap defined by matplotlibs defaults",
    )
    crs: Literal["EPSG:4326", "EPSG:3857"] = Field(
        "EPSG:4326",
        description="Coordinate reference system to use for the query. EPSG:4326 and EPSG:3857 are supported for this request",
    )
    time: Optional[str] = Field(
        None,
        description="Optional time to get map for in Y-m-dTH:M:SZ format. Only valid when the layer has a time dimension. When not specified, the default time is used",
    )
    elevation: Optional[str] = Field(
        None,
        description="Optional elevation to get map for. Only valid when the layer has an elevation dimension. When not specified, the default elevation is used",
    )
    bbox: Optional[str] = Field(
        None,
        description="Bounding box to use for the query in the format 'minx,miny,maxx,maxy'",
    )
    tile: Optional[str] = Field(
        None,
        description="Tile to use for the query in the format 'x,y,z'. If specified, bbox is ignored",
    )
    width: int = Field(
        ...,
        description="The width of the image to return in pixels",
    )
    height: int = Field(
        ...,
        description="The height of the image to return in pixels",
    )
    colorscalerange: str = Field(
        None,
        description="Optional color scale range to use for the query in the format 'min,max'",
    )
    autoscale: bool = Field(
        False,
        description="Whether to automatically scale the color scale range based on the data. When specified, colorscalerange is ignored",
    )


class WMSGetFeatureInfoQuery(WMSBaseQuery):
    """WMS GetFeatureInfo query"""

    request: Literal["GetFeatureInfo", "GetTimeseries", "GetVerticalProfile"] = Field(
        ...,
        description="Request type",
    )
    query_layers: str
    time: Optional[str] = Field(
        None,
        description="Optional time to get feature info for in Y-m-dTH:M:SZ format. Only valid when the layer has a time dimension. To get a range of times, use 'start/end'",
    )
    elevation: Optional[str] = Field(
        None,
        description="Optional elevation to get feature info for. Only valid when the layer has an elevation dimension. To get all elevations, use 'all', to get a range of elevations, use 'start/end'",
    )
    crs: Literal["EPSG:4326"] = Field(
        "EPSG:4326",
        description="Coordinate reference system to use for the query. Currently only EPSG:4326 is supported for this request",
    )
    bbox: str = Field(
        ...,
        description="Bounding box to use for the query in the format 'minx,miny,maxx,maxy'",
    )
    width: int = Field(
        ...,
        description="Width of the image to query against. This is the number of points between minx and maxx",
    )
    height: int = Field(
        ...,
        description="Height of the image to query against. This is the number of points between miny and maxy",
    )
    x: int = Field(
        ...,
        description="The x coordinate of the point to query. This is the index of the point in the x dimension",
    )
    y: int = Field(
        ...,
        description="The y coordinate of the point to query. This is the index of the point in the y dimension",
    )


class WMSGetLegendInfoQuery(WMSBaseQuery):
    """WMS GetLegendInfo query"""

    request: Literal["GetLegendGraphic"] = Field(..., description="Request type")
    layers: str
    width: int
    height: int
    vertical: bool = False
    colorscalerange: str = "nan,nan"
    autoscale: bool = True
    styles: str = "raster/default"


def wms_query(
    service: Literal["WMS"] = Query(..., description="Service type. Must be WMS"),
    version: Literal["1.1.1", "1.3.0"] = Query(
        ...,
        description="Version of the WMS service",
    ),
    request: Literal[
        "GetCapabilities",
        "GetMetadata",
        "GetMap",
        "GetFeatureInfo",
        "GetTimeseries",
        "GetVerticalProfile",
        "GetLegendInfo",
        "GetLegendGraphic",
    ] = Query(..., description="Request type"),
    layername: Optional[str] = Query(
        None,
        description="Name of the layer to get metadata for. Only valid for GetMetadata requests",
    ),
    item: Literal["layerdetails", "timesteps", "minmax", "menu"] | None = Query(
        None,
        description="The type of GetMetadata request. Only valid for GetMetadata requests",
    ),
    day: Optional[str] = Query(
        None,
        description="Optional day to get timesteps for in Y-m-d format. Only valid for GetMetadata requests when item=timesteps and layer has a time dimension",
    ),
    range: Optional[str] = Query(
        None,
        description="Optional range to get timesteps for in Y-m-dTH:M:SZ/Y-m-dTH:M:SZ format. Only valid for GetMetadta requests when item=timesteps and layer has a time dimension",
    ),
    layers: Optional[str] = Query(
        None,
        description="Comma separated list of layer names. Valid for GetMap and GetLegendInfo requests",
    ),
    query_layers: Optional[str] = Query(
        None,
        description="Comma separated list of layer names to query. Valid for GetFeatureInfo requests",
    ),
    styles: str = Query(
        "raster/default",
        description="Style to use for the query. Defaults to raster/default. Default may be replaced by the name of any colormap defined by matplotlibs defaults. Valid for GetMap and GetLegendInfo requests",
    ),
    crs: Literal["EPSG:4326", "EPSG:3857"] = Query(
        "EPSG:4326",
        description="Coordinate reference system to use for the query. EPSG:4326 and EPSG:3857 are supported for this request",
    ),
    srs: Optional[Literal["EPSG:4326", "EPSG:3857"]] = Query(
        None,
        description="Coordinate reference system to use for the query. EPSG:4326 and EPSG:3857 are supported for this request",
    ),
    time: Optional[str] = Query(
        None,
        description="Optional time to get map for in Y-m-dTH:M:SZ format. Only valid when the layer has a time dimension. When not specified, the default time is used. Valid for GetMap and GetFeatureInfo requests. For GetFeatureInfo, to get a range of times, use 'start/end'",
    ),
    elevation: Optional[str] = Query(
        None,
        description="Optional elevation to get map for. Only valid when the layer has an elevation dimension. When not specified, the default elevation is used. Valid for GetMap and GetFeatureInfo requests. For GetFeatureInfo, to get all elevations, use 'all', to get a range of elevations, use 'start/end'",
    ),
    bbox: Optional[str] = Query(
        None,
        description="Bounding box to use for the query in the format 'minx,miny,maxx,maxy'. Valid for GetMap and GetFeatureInfo requests",
    ),
    tile: Optional[str] = Query(
        None,
        description="Tile to use for the query in the format 'x,y,z'. If specified, bbox is ignored. Only valid for GetMap requests",
    ),
    width: Optional[int] = Query(
        None,
        description="Valid for GetMap, GetLegendInfo, and GetFeatureInfo requests. For GetMap and GetFeatureInfo this is the width of the image to return in pixels. For GetLegendInfo this is the number of points in the x dimension to select from the dataset",
    ),
    height: Optional[int] = Query(
        None,
        description="Valid for GetMap, GetLegendInfo, and GetFeatureInfo requests. For GetMap and GetFeatureInfo this is the height of the image to return in pixels. For GetLegendInfo this is the number of points in the y dimension to select from the dataset",
    ),
    x: Optional[int] = Query(
        None,
        description="The x coordinate of the point to query. This is the index of the point in the x dimension. Only valid for GetFeatureInfo requests",
    ),
    y: Optional[int] = Query(
        None,
        description="The y coordinate of the point to query. This is the index of the point in the y dimension. Only valid for GetFeatureInfo requests",
    ),
    colorscalerange: Optional[str] = Query(
        None,
        description="Optional color scale range to use for the query in the format 'min,max'. Valid for GetMap and GetLegendInfo requests",
    ),
    autoscale: Optional[bool] = Query(
        False,
        description="Whether to automatically scale the color scale range based on the data. When specified, colorscalerange is ignored. Only valid for GetMap and GetFeatureInfo requests",
    ),
) -> Union[
    WMSGetCapabilitiesQuery,
    WMSGetMetadataQuery,
    WMSGetMapQuery,
    WMSGetFeatureInfoQuery,
    WMSGetLegendInfoQuery,
]:
    if request == "GetCapabilities":
        return WMSGetCapabilitiesQuery(
            service=service,
            version=version,
            request=request,
        )
    elif request == "GetMetadata":
        return WMSGetMetadataQuery(
            service=service,
            version=version,
            request=request,
            layername=layername,
            item=item,
            day=day,
            range=range,
            bbox=bbox,
            time=time,
            elevation=elevation,
        )
    elif request == "GetMap":
        return WMSGetMapQuery(
            service=service,
            version=version,
            request=request,
            layers=layers,
            styles=styles,
            crs=crs if srs is None else srs,
            time=time,
            elevation=elevation,
            bbox=bbox,
            tile=tile,
            width=width,
            height=height,
            colorscalerange=colorscalerange,
            autoscale=autoscale,
        )
    elif (
        request == "GetFeatureInfo"
        or request == "GetTimeseries"
        or request == "GetVerticalProfile"
    ):
        return WMSGetFeatureInfoQuery(
            service=service,
            version=version,
            request=request,
            query_layers=query_layers,
            time=time,
            elevation="all" if request == "GetVerticalProfile" else elevation,
            crs=crs if srs is None else srs,
            bbox=bbox,
            width=width,
            height=height,
            x=x,
            y=y,
        )
    elif request == "GetLegendInfo" or request == "GetLegendGraphic":
        return WMSGetLegendInfoQuery(
            service=service,
            version=version,
            request=request,
            layers=layers,
            width=width,
            height=height,
            vertical=False,
            colorscalerange=colorscalerange,
            autoscale=autoscale,
            styles=styles,
        )
    else:
        raise ValueError(f"Unknown WMS request type: {request}")


# These params are used for GetMap and GetFeatureInfo requests, and can be filtered out of the query params for any requests that are handled
WMS_FILTERED_QUERY_PARAMS = {
    "service",
    "version",
    "request",
    "layers",
    "query_layers",
    "styles",
    "crs",
    "srs",
    "time",
    "elevation",
    "bbox",
    "tile",
    "width",
    "height",
    "colorscalerange",
    "autoscale",
}
