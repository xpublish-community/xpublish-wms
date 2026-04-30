"""Visualize vector direction."""

from typing import Sequence

import matplotlib
import numpy as np
import xarray as xr
from numpy.typing import NDArray
from PIL.Image import Image

from xpublish_wms.wms.get_map.style_types import VectorStyleParams
from xpublish_wms.wms.get_map.vector_style_utils import (
    get_grid_step,
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


def get_cell_center_indices(
    das: Sequence[xr.DataArray],
    bbox: tuple[float, float, float, float],
    width: int,
    height: int,
    density: int,
) -> tuple[NDArray[np.intp], NDArray[np.intp]]:
    """Return (px, py) pixel coordinates of subsampled data cell centers."""
    x_full = das[0].x.broadcast_like(das[0])
    y_full = das[0].y.broadcast_like(das[0])
    px = np.floor(
        (x_full.values.ravel() - bbox[0]) / (bbox[2] - bbox[0]) * width,
    ).astype(np.intp)
    py = np.floor(
        (y_full.values.ravel() - bbox[1]) / (bbox[3] - bbox[1]) * height,
    ).astype(np.intp)

    in_tile = (px >= 0) & (px < width) & (py >= 0) & (py < height)
    px, py = px[in_tile], py[in_tile]

    grid_step = get_grid_step(density)
    x_span = bbox[2] - bbox[0]
    y_span = bbox[3] - bbox[1]
    # Use globally aligned buckets so neighboring tiles use the same subsampling phase.
    px_shifted = px + int((bbox[0] / x_span * width) % grid_step)
    py_shifted = py + int((bbox[1] / y_span * height) % grid_step)
    bucket_ids = (px_shifted // grid_step) * (height // grid_step + 1) + (
        py_shifted // grid_step
    )

    _, inverse = np.unique(bucket_ids, return_inverse=True)
    counts = np.bincount(inverse)
    px_sub = (np.bincount(inverse, weights=px) / counts).round().astype(np.intp)
    py_sub = (np.bincount(inverse, weights=py) / counts).round().astype(np.intp)

    return px_sub, py_sub


def visualize_vectors(
    meshes: Sequence[xr.DataArray],
    color: str,
    density: int,
    scaling: VectorStyleParams.GlyphScaling,
    colorscale_range: tuple[float, float] | None = None,
    colormap: str | None = None,
    draw_backing: bool = False,
    arrow_mag_color: bool = False,
    cell_center_indices: tuple[NDArray[np.intp], NDArray[np.intp]] | None = None,
    margin_px: int = 0,
) -> Image:
    """Renders a vector tile image."""
    if density not in (1, 2, 3):
        raise ValueError(f"Invalid density value {density}")

    # NOTE: remember `assert` can be disabled with `python -O`
    # This should never fail, datashader renders both in the shape of the tile
    assert meshes[0].shape == meshes[1].shape
    tile_height, tile_width = meshes[0].shape

    mag: xr.DataArray = np.sqrt(
        meshes[0] ** 2 + meshes[1] ** 2,
    )  # pyright: ignore[reportAssignmentType]

    fig, ax = setup_tile_plot(tile_width, tile_height)

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

    # Build arrays with vector glyph positions
    if cell_center_indices is None:
        x_indices, y_indices = get_meshgrid(density, tile_width, tile_height)
    else:
        x_indices, y_indices = cell_center_indices
        valid = np.isfinite(meshes[0].values[y_indices, x_indices]) & np.isfinite(
            meshes[1].values[y_indices, x_indices],
        )
        x_indices, y_indices = x_indices[valid], y_indices[valid]

    # Build arrays to render vector glyphs
    u = meshes[0].values[y_indices, x_indices].astype(np.float32)
    v = meshes[1].values[y_indices, x_indices].astype(np.float32)

    # Vector scaling
    if scaling == VectorStyleParams.GlyphScaling.CONSTANT:
        m = mag.values[y_indices, x_indices]
        nz = m != 0
        u[nz] /= m[nz]
        v[nz] /= m[nz]
    else:
        # Scale up just a little
        # TODO: this would ideally be based on the dataset max magnitude
        u *= 3
        v *= 3
    # Scale the length based on density
    u *= LENGTH_SCALE[density - 1]
    v *= LENGTH_SCALE[density - 1]

    render_args = (x_indices, y_indices, u, v)
    if arrow_mag_color:
        render_args += (mag.values[y_indices, x_indices],)

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
    im = render_vector_arrows(fig, ax, render_args, render_kwargs)
    if margin_px > 0:
        im = im.crop(
            (margin_px, margin_px, im.width - margin_px, im.height - margin_px),
        )
    return im
