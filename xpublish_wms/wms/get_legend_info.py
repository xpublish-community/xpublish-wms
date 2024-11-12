import io
from math import isnan

import numpy as np
import xarray as xr
from fastapi import Response
from matplotlib import cm
from PIL import Image

from xpublish_wms.query import WMSGetLegendInfoQuery
from xpublish_wms.utils import parse_float


def get_legend_info(dataset: xr.Dataset, query: WMSGetLegendInfoQuery) -> Response:
    """
    Return the WMS legend graphic for the dataset and given parameters
    """
    parameter = query.layers
    width = query.width
    height = query.height
    vertical = query.vertical
    # colorbaronly = query.get("colorbaronly", "False") == "True"
    colorscalerange = [parse_float(x) for x in query.colorscalerange.split(",")]
    if isnan(colorscalerange[0]):
        autoscale = True
    else:
        autoscale = query.autoscale
    style = query.styles
    stylename, palettename = style.split("/")

    ds = dataset.squeeze()

    # if the user has supplied a color range, use it. Otherwise autoscale
    if autoscale:
        min_value = float(ds[parameter].min())
        max_value = float(ds[parameter].max())
    else:
        min_value = colorscalerange[0]
        max_value = colorscalerange[1]

    scaled = (np.linspace(min_value, max_value, width) - min_value) / (
        max_value - min_value
    )
    data = np.ones((height, width)) * scaled

    if vertical:
        data = np.flipud(data.T)
        data = data.reshape((height, width))

    # Let user pick cm from here https://predictablynoisy.com/matplotlib/gallery/color/colormap_reference.html#sphx-glr-gallery-color-colormap-reference-py
    # Otherwise default to rainbow
    if palettename == "default":
        palettename = "turbo"
    im = Image.fromarray(np.uint8(cm.get_cmap(palettename)(data) * 255))

    image_bytes = io.BytesIO()
    im.save(image_bytes, format="PNG")
    image_bytes = image_bytes.getvalue()

    return Response(content=image_bytes, media_type="image/png")
