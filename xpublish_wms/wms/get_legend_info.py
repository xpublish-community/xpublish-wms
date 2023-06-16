import io
from math import isnan
from fastapi import Response
import xarray as xr
import numpy as np
from PIL import Image
from matplotlib import cm


def get_legend_info(dataset: xr.Dataset, query: dict):
    """
    Return the WMS legend graphic for the dataset and given parameters
    """
    parameter = query["layers"]
    width: int = int(query["width"])
    height: int = int(query["height"])
    vertical = query.get("vertical", "false") == "true"
    # colorbaronly = query.get("colorbaronly", "False") == "True"
    colorscalerange = [
        float(x) for x in query.get("colorscalerange", "nan,nan").split(",")
    ]
    if isnan(colorscalerange[0]):
        autoscale = True
    else:
        autoscale = query.get("autoscale", "false") != "false"
    style = query["styles"]
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