"""Visualize vector direction."""

from typing import Sequence

import matplotlib
import numpy as np
import xarray as xr
from PIL.Image import Image

from xpublish_wms.wms.get_map.style_types import VectorStyleParams
from xpublish_wms.wms.get_map.vector_style_utils import (
    get_meshgrid,
    render_vector_arrows,
    setup_tile_plot,
)

# Scale arrow length
LENGTH_SCALE = np.array([3, 2, 1]) * 9
# Other arrow size parameters are relative to the tail width
TAIL_WIDTH = np.array([3.5, 2, 1]) * 1.3
HEAD_WIDTH = [2.5, 2.5, 3.5]
# Arrow outline stroke width
LINE_WIDTH = [5, 4, 1]


def visualize_vectors(
    meshes: Sequence[xr.DataArray],
    color: str,
    density: int,
    scaling: VectorStyleParams.GlyphScaling,
    colorscale_range: tuple[float, float] | None = None,
    colormap: str | None = None,
    draw_backing: bool = False,
    arrow_mag_color: bool = False,
) -> Image:
    """Renders a vector tile image."""
    # Create a mesh of grid-points where we will draw arrows/barbs
    if density not in (1, 2, 3):
        raise ValueError(f"Invalid density value {density}")

    assert meshes[0].shape == meshes[1].shape
    tile_height, tile_width = meshes[0].shape

    x_indices, y_indices = get_meshgrid(density, tile_width, tile_height)

    # Select the vector components in a subgrid
    u = meshes[0].isel(x=x_indices, y=y_indices).astype(np.float32)
    v = meshes[1].isel(x=x_indices, y=y_indices).astype(np.float32)
    # use the entire mesh for magnitude not just the sparse u,v
    mag: xr.DataArray = np.sqrt(
        meshes[0] ** 2 + meshes[1] ** 2,
    )  # pyright: ignore[reportAssignmentType]
    # Initialize a plot with appropriate axes
    fig, ax = setup_tile_plot(tile_width, tile_height)

    # If colormap background is desired, draw it now
    if draw_backing and colormap is not None:
        ax.imshow(
            mag,
            cmap=colormap,
            vmin=colorscale_range and colorscale_range[0],
            vmax=colorscale_range and colorscale_range[1],
            extent=(0, tile_width, 0, tile_height),
            origin="lower",
            interpolation="nearest",
        )

    # Scale the length up based on density
    u *= LENGTH_SCALE[density - 1]
    v *= LENGTH_SCALE[density - 1]
    if scaling == VectorStyleParams.GlyphScaling.CONSTANT:
        # normalize the vectors so their size is CONSTANT
        u /= mag.isel(x=x_indices, y=y_indices)
        v /= mag.isel(x=x_indices, y=y_indices)
    else:
        # scale up just a little
        # TODO: this should depend on dataset and its max magnitude
        u *= 3
        v *= 3

    render_args = (x_indices, y_indices, u, v)
    if arrow_mag_color:
        render_args += (mag.isel(x=x_indices, y=y_indices),)

    # Sum the R, G, B values and determine a contrasting edgeline
    edgecolor = "black" if sum(matplotlib.colors.to_rgb(color)) > 1.5 else "white"
    render_kwargs = {
        "color": color,
        "edgecolor": edgecolor,
        "linewidth": LINE_WIDTH[density - 1],
        "linestyle": "solid",
        "width": TAIL_WIDTH[density - 1],
        "headwidth": HEAD_WIDTH[density - 1],
        "headlength": 3,
        "headaxislength": 2.8,
        "cmap": colormap if arrow_mag_color else None,
        "vmin": colorscale_range[0] if arrow_mag_color and colorscale_range else None,
        "vmax": colorscale_range[1] if arrow_mag_color and colorscale_range else None,
    }
    return render_vector_arrows(fig, ax, render_args, render_kwargs)
