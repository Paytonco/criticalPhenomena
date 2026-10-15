"""Rendering helpers for the sandpile model: turning load-grid snapshots
into compelling stills and video for science communication."""
from __future__ import annotations

from pathlib import Path

import numpy as np
from matplotlib.colors import BoundaryNorm, ListedColormap

from common.viz import animate_grid_with_activity, render_grid

# load buckets 0..z_c-1 (thin snow -> heavy, near-critical load), plus a
# sentinel "just released" highlight -- the same categorical-state trick
# wildfire/viz.py uses for its fire colormap.
SNOW_COLORS = ["#eaf3fb", "#a9cbe8", "#5b8fc4", "#1f3f66", "#ff5050"]
SNOW_CMAP = ListedColormap(SNOW_COLORS)
SNOW_NORM = BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5, 4.5], SNOW_CMAP.N)


def show_grid(ax, grid: np.ndarray, title: str | None = None):
    return render_grid(ax, grid, SNOW_CMAP, SNOW_NORM, title=title)


def animate_grid(frames: np.ndarray, path: str | Path, fps: int = 20,
                  title: str = "", dpi: int = 130, activity: np.ndarray | None = None):
    """Render a (n_frames, L, L) stack of sandpile render-codes to video.

    If `activity` (grains toppling per frame) is given, a small running
    strip-chart is drawn beneath the grid.
    """
    return animate_grid_with_activity(
        frames, SNOW_CMAP, SNOW_NORM, path, fps=fps, title=title, dpi=dpi,
        activity=activity, activity_label="grains toppling per tick",
    )
