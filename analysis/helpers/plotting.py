"""Figure export shared by all analysis notebooks.

Replaces the copy-pasted `if False:` export blocks in the original notebook (cells 32, 46, 56,
62), which each repeated the same deepcopy / transparent-background / write_image sequence.
"""

from copy import deepcopy

import plotly.graph_objects as go

from helpers.config import OUTPUT_DIR


def save_figure(
        fig: go.Figure,
        filename: str,
        width: int = 1000,
        height: int = 600,
        scale: int = 3,
) -> str:
    """Write `fig` to `output/` as a transparent-background PNG.

    The figure is deep-copied first, so the on-screen version keeps its background. `scale`
    multiplies the pixel dimensions, standing in for DPI.
    """
    OUTPUT_DIR.mkdir(exist_ok=True)
    if not filename.endswith(".png"):
        filename = f"{filename}.png"
    path = OUTPUT_DIR / filename
    deepcopy(fig).update_layout(
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
    ).write_image(path, width=width, height=height, scale=scale)
    print(f"Saved figure to {path}")
    return str(path)
