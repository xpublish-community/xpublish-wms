import xml.etree.ElementTree as ET
from typing import List
import cf_xarray  # noqa
import xarray as xr
from fastapi import Request, Response
from xpublish_wms.grid import GridType, sel2d
from xpublish_wms.utils import ds_bbox, format_timestamp


# WMS Styles declaration
# TODO: Add others beyond just simple raster
styles = [
    {
        "name": "raster/default",
        "title": "Raster",
        "abstract": "The default raster styling, scaled to the given range. The palette can be overridden by replacing default with a matplotlib colormap name",
    },
]


def create_text_element(root, name: str, text: str) -> ET.Element:
    element = ET.SubElement(root, name)
    element.text = text
    return element


def create_capability_element(root, name: str, url: str, formats: List[str]) -> ET.Element:
    cap = ET.SubElement(root, name)
    # TODO: Add more image formats
    for fmt in formats:
        create_text_element(cap, "Format", fmt)

    dcp_type = ET.SubElement(cap, "DCPType")
    http = ET.SubElement(dcp_type, "HTTP")
    get = ET.SubElement(http, "Get")
    get.append(
        ET.Element(
            "OnlineResource",
            attrib={"xlink:type": "simple", "xlink:href": url},
        ),
    )
    return cap


def get_capabilities(ds: xr.Dataset, request: Request) -> Response:
    """
    Return the WMS capabilities for the dataset
    """
    wms_url = f'{request.base_url}{request.url.path.removeprefix("/")}'

    root = ET.Element(
        "WMS_Capabilities",
        version="1.3.0",
        attrib={
            "xmlns": "http://www.opengis.net/wms",
            "xmlns:xlink": "http://www.w3.org/1999/xlink",
        },
    )

    service = ET.SubElement(root, "Service")
    create_text_element(service, "Name", "WMS")
    create_text_element(service, "Title", "XPublish WMS")
    create_text_element(service, "Abstract", "XPublish WMS")
    service.append(ET.Element("KeywordList"))
    service.append(
        ET.Element(
            "OnlineResource",
            attrib={
                "xlink:type": "simple",
                "xlink:href": "http://www.opengis.net/spec/wms_schema_1/1.3.0",
            },
        ),
    )

    capability = ET.SubElement(root, "Capability")
    # request_tag = ET.SubElement(capability, "Request")

    # get_capabilities = create_capability_element(
    #     request_tag,
    #     "GetCapabilities",
    #     wms_url,
    #     ["text/xml"],
    # )
    # TODO: Add more image formats
    # get_map = create_capability_element(request_tag, "GetMap", wms_url, ["image/png"])
    # TODO: Add more feature info formats
    # get_feature_info = create_capability_element(
    #     request_tag,
    #     "GetFeatureInfo",
    #     wms_url,
    #     ["text/json"],
    # )
    # TODO: Add more image formats
    # get_legend_graphic = create_capability_element(
    #     request_tag,
    #     "GetLegendGraphic",
    #     wms_url,
    #     ["image/png"],
    # )

    exeption_tag = ET.SubElement(capability, "Exception")
    exception_format = ET.SubElement(exeption_tag, "Format")
    exception_format.text = "text/json"

    layer_tag = ET.SubElement(capability, "Layer")
    create_text_element(layer_tag, "Title", ds.attrs.get("title", "Untitled"))
    create_text_element(
        layer_tag,
        "Description",
        ds.attrs.get("description", "No Description"),
    )
    create_text_element(layer_tag, "CRS", "EPSG:4326")
    create_text_element(layer_tag, "CRS", "EPSG:3857")
    create_text_element(layer_tag, "CRS", "CRS:84")


    bbox = ds_bbox(ds)
    bounds = {
        "CRS": "EPSG:4326",
        "minx": f'{bbox[0]}',
        "miny": f'{bbox[1]}',
        "maxx": f'{bbox[2]}',
        "maxy": f'{bbox[3]}',
    }

    for var in ds.data_vars:
        da = ds[var]

        # If there are not spatial coords, we can't view it with this router, sorry
        if "longitude" not in da.cf.coords:
            continue

        attrs = da.cf.attrs
        layer = ET.SubElement(layer_tag, "Layer", attrib={"queryable": "1"})
        create_text_element(layer, "Name", var)
        create_text_element(
            layer,
            "Title",
            attrs.get("long_name", attrs.get("name", var)),
        )
        create_text_element(
            layer,
            "Abstract",
            attrs.get("long_name", attrs.get("name", var)),
        )
        create_text_element(layer, "CRS", "EPSG:4326")
        create_text_element(layer, "CRS", "EPSG:3857")
        create_text_element(layer, "CRS", "CRS:84")

        create_text_element(layer, "Units", attrs.get("units", ""))

        # min_value = float(da.min())
        # create_text_element(layer, 'MinMag', min_value)

        # max_value = float(da.max())
        # create_text_element(layer, 'MaxMag', max_value)

        # Not sure if this can be copied, its possible variables have different extents within
        # a given dataset probably, but for now...
        ET.SubElement(layer, "BoundingBox", attrib=bounds)

        if "T" in da.cf.axes:
            times = format_timestamp(da.cf["T"])

            time_dimension_element = ET.SubElement(
                layer,
                "Dimension",
                attrib={
                    "name": "time",
                    "units": "ISO8601",
                    "default": times[-1],
                },
            )
            # TODO: Add ISO duration specifier
            time_dimension_element.text = f"{','.join(times)}"

        style_tag = ET.SubElement(layer, "Style")

        for style in styles:
            style_element = ET.SubElement(
                style_tag,
                "Style",
                attrib={"name": style["name"]},
            )
            create_text_element(style_element, "Title", style["title"])
            create_text_element(style_element, "Abstract", style["abstract"])

            legend_url = f'{wms_url}?service=WMS&request=GetLegendGraphic&format=image/png&width=20&height=20&layers={var}&styles={style["name"]}'
            create_text_element(style_element, "LegendURL", legend_url)

    ET.indent(root, space="\t", level=0)
    get_caps_xml = ET.tostring(root).decode("utf-8")

    return Response(get_caps_xml, media_type="text/xml")