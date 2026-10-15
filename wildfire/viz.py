"""Animation helpers for the forest-fire model: turning grid snapshots into
compelling video for science communication."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from common.viz import FIRE_CMAP, FIRE_NORM, animate_grid_with_activity


def animate_grid(frames: np.ndarray, path: str | Path, fps: int = 20,
                  title: str = "", dpi: int = 130, activity: np.ndarray | None = None):
    """Render a (n_frames, L, L) stack of forest-fire grid states to video.

    If `activity` (trees ignited per frame) is given, a small running
    strip-chart is drawn beneath the grid so the burst-like nature of
    avalanches is visible alongside the spatial pattern.
    """
    return animate_grid_with_activity(
        frames, FIRE_CMAP, FIRE_NORM, path, fps=fps, title=title, dpi=dpi,
        activity=activity, activity_label="trees ignited per step",
    )
