"""Class definitions for GetMap rendering style information."""

from enum import StrEnum
from typing import Literal, Sequence

from pydantic import BaseModel


class ColormapStyleParams(BaseModel):
    """Container for colormap style parameters."""

    type: Literal["colormap"]
    palettename: str
    colorscale_range: Sequence[float] | None
    autoscale: bool


class VectorStyleParams(BaseModel):
    """Container for vector style parameters."""

    class GlyphScaling(StrEnum):
        CONSTANT = "constant"
        UNIFORM = "uniform"
        # future option could be length/tail

    type: Literal["vector"]
    color: str
    density: int
    scaling: GlyphScaling
    colorscale_range: tuple[float, float] | None
    colormap: str | None
    draw_backing: bool
    arrow_mag_color: bool


ShadingStyleParams = ColormapStyleParams | VectorStyleParams
