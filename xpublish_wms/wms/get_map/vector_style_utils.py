"""Vector rendering helpers."""

from typing import Any, MutableMapping, Tuple

import matplotlib
import numpy as np
from numpy.typing import NDArray
from PIL.Image import Image, fromarray

matplotlib.use("Agg")

from matplotlib import pyplot as plt  # noqa: E402


def get_grid_step(density: int) -> int:
    """Return vector glyph grid step for given density."""
    return 64 // (2 ** (density - 1))


def get_meshgrid(
    density: int,
    tile_width: int,
    tile_height: int,
) -> Tuple[NDArray[np.intp], NDArray[np.intp]]:
    """Return flat (x, y) pixel index pairs for the vector subgrid."""
    # For a 256x256 tile, there will be:
    # - 4x4 glyphs at density 1,
    # - 8x8 glyphs for density 2,
    # - 16x16 glyphs for density 3.
    grid_step = get_grid_step(density)
    x_indices = np.arange(grid_step // 2, tile_width, grid_step)
    y_indices = np.arange(grid_step // 2, tile_height, grid_step)
    x_indices_grid, y_indices_grid = np.meshgrid(x_indices, y_indices)
    return x_indices_grid.ravel(), y_indices_grid.ravel()


def setup_tile_plot(tile_width: int, tile_height: int) -> tuple[plt.Figure, plt.Axes]:
    """Setup a plot with appropriate axes for rendering a tile."""
    # Make a figure without a frame or axes, this ensures the image has the desired
    # dimensions, as well as not drawing any axes.
    fig = plt.figure(frameon=False, dpi=1, figsize=(tile_width, tile_height))
    ax = plt.Axes(fig, [0, 0, 1, 1])
    ax.set_axis_off()
    # This is extremely important. It ensures that the frame of the figure used has the same
    # dimensions and orientation as the raster tile
    ax.set_xlim(0, tile_width)
    ax.set_ylim(0, tile_height)
    fig.add_axes(ax)

    return fig, ax


VectorArrowsRenderArgs = (
    Tuple[NDArray[np.intp], NDArray[np.intp], NDArray[np.float32], NDArray[np.float32]]
    | Tuple[
        NDArray[np.intp],
        NDArray[np.intp],
        NDArray[np.float32],
        NDArray[np.float32],
        NDArray[np.float32],
    ]
)


def render_vector_arrows(
    fig: plt.Figure,
    ax: plt.Axes,
    render_args: VectorArrowsRenderArgs,
    render_kwargs: MutableMapping[str, Any],
) -> Image:
    """Plot vector arrows using `matplotlib.quiver` and create an `Image`.

    Also call `matplotlib.close` at the end.
    """
    vmin = render_kwargs.pop("vmin")
    vmax = render_kwargs.pop("vmax")
    # Internally matplotlib scales the arrow width and length based on the number of arrows that
    # render, and scale. We explicitly set the width to prevent arrows from being larger on
    # tiles with fewer arrows, and we set scale=1 and units='xy' so we can very carefully set
    # the arrow lengths.
    q = ax.quiver(*render_args, scale=1, units="xy", **render_kwargs)
    if vmin is not None and vmax is not None:
        q.set_clim(vmin, vmax)

    fig.canvas.draw()
    # Copy the matplotlib figure to an Image object
    im = fromarray(np.asarray(fig.canvas.buffer_rgba()))
    plt.close(fig)
    return im
