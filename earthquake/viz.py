"""Rendering helpers for the OFC earthquake model: turning stress-grid
snapshots into compelling stills and video for science communication."""
from __future__ import annotations

from pathlib import Path

import numpy as np
from matplotlib.colors import BoundaryNorm, ListedColormap

from common.viz import animate_grid_with_activity, render_grid

# stress buckets 0..3 (calm -> heavily loaded, near-failure), plus a
# sentinel "just ruptured" highlight -- the same categorical-state trick
# wildfire/viz.py and sandpile/viz.py use for their colormaps.
QUAKE_COLORS = ["#e9e4d8", "#d9a066", "#b5451f", "#4a1414", "#fff45c"]
QUAKE_CMAP = ListedColormap(QUAKE_COLORS)
QUAKE_NORM = BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5, 4.5], QUAKE_CMAP.N)


def show_grid(ax, grid: np.ndarray, title: str | None = None):
    return render_grid(ax, grid, QUAKE_CMAP, QUAKE_NORM, title=title)


def animate_grid(frames: np.ndarray, path: str | Path, fps: int = 20,
                  title: str = "", dpi: int = 130, activity: np.ndarray | None = None):
    """Render a (n_frames, L, L) stack of fault stress-grid render-codes
    to video.

    If `activity` (sites rupturing per frame) is given, a small running
    strip-chart is drawn beneath the grid.
    """
    return animate_grid_with_activity(
        frames, QUAKE_CMAP, QUAKE_NORM, path, fps=fps, title=title, dpi=dpi,
        activity=activity, activity_label="sites rupturing per tick",
    )
