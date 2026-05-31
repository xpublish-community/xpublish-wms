import io

import matplotlib
import numpy as np
from fastapi import Response
from PIL import Image

from xpublish_wms.query import WMSGetLegendInfoQuery


def get_legend_info(query: WMSGetLegendInfoQuery) -> Response:
    """
    Return the WMS legend graphic for the dataset and given parameters
    """
    width = query.width
    height = query.height
    vertical = query.vertical
    _, palettename = query.styles

    data = np.ones((height, width)) * np.linspace(0, 1, width)

    if vertical:
        data = np.flipud(data.T)
        data = data.reshape((height, width))

    # Let user pick cm from here https://predictablynoisy.com/matplotlib/gallery/color/colormap_reference.html#sphx-glr-gallery-color-colormap-reference-py
    # Otherwise default to rainbow
    if palettename == "default":
        palettename = "turbo"
    im = Image.fromarray(
        np.uint8(matplotlib.colormaps.get_cmap(palettename)(data) * 255),
    )

    image_bytes = io.BytesIO()
    im.save(image_bytes, format="PNG")
    image_bytes = image_bytes.getvalue()

    return Response(content=image_bytes, media_type="image/png")
