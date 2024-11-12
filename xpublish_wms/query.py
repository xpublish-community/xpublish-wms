from typing import Literal, Optional
from pydantic import BaseModel, Field


class WMSQuery(BaseModel):
    """Base WMS query"""

    service: Literal["WMS"] = "WMS"
    version: Literal["1.1.1", "1.3.0"] = "1.3.0"


class WMSGetCapabilitiesQuery(WMSQuery):
    """WMS GetCapabilities query"""

    request: Literal["GetCapabilities"] = "GetCapabilities"


class WMSGetMetadataQuery(WMSQuery):
    """WMS GetMetadata query"""

    request: Literal["GetMetadata"] = "GetMetadata"
    layername: str
    item: Literal["layerdetails", "timesteps", "minmax", "menu"]
    day: Optional[str] = Field(
        None,
        description="Optional day to get timesteps for in Y-m-d format. Only valid when item=timesteps and layer has a time dimension",
    )
    range: Optional[str] = Field(
        None,
        description="Optional range to get timesteps for in Y-m-dTH:M:SZ/Y-m-dTH:M:SZ format. Only valid when item=timesteps and layer has a time dimension",
    )


class WMSGetMapQuery(WMSQuery):
    """WMS GetMap query"""

    request: Literal["GetMap"] = "GetMap"
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


class WMSGetFeatureInfoQuery(WMSQuery):
    """WMS GetFeatureInfo query"""

    request: Literal["GetFeatureInfo", "GetTimeseries"] = "GetFeatureInfo"
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


class WMSGetLegendInfoQuery(WMSQuery):
    """WMS GetLegendInfo query"""

    request: Literal["GetLegendInfo"] = "GetLegendInfo"
    layers: str
    width: int
    height: int
    vertical: bool = False
    colorscalerange: str = "nan,nan"
    autoscale: bool = True
    styles: str = "raster/default"
