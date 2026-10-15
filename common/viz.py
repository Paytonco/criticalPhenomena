"""
Shared plotting style and small helpers so every critical-system demo in
this repo looks like part of the same series.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation

try:
    from imageio_ffmpeg import get_ffmpeg_exe
    plt.rcParams["animation.ffmpeg_path"] = get_ffmpeg_exe()
    _HAVE_FFMPEG = True
except Exception:
    _HAVE_FFMPEG = False

BG = "#0f1115"
FG = "#e8e6e1"
GRID_COLOR = "#2a2d34"
ACCENT = "#ff6f3c"     # burning / highlight
ACCENT2 = "#3ea6ff"    # secondary highlight / fit lines
MUTED = "#6b7280"

PALETTE = [ACCENT2, ACCENT, "#4fd1a5", "#c792ea", MUTED]


def set_style():
    plt.rcParams.update({
        "figure.facecolor": BG,
        "axes.facecolor": BG,
        "savefig.facecolor": BG,
        "axes.edgecolor": GRID_COLOR,
        "axes.labelcolor": FG,
        "text.color": FG,
        "xtick.color": FG,
        "ytick.color": FG,
        "axes.titlecolor": FG,
        "grid.color": GRID_COLOR,
        "grid.alpha": 0.6,
        "font.size": 12,
        "axes.titlesize": 14,
        "axes.titleweight": "bold",
        "axes.grid": True,
        "grid.linestyle": "-",
        "grid.linewidth": 0.6,
        "figure.dpi": 110,
        "savefig.dpi": 170,
        "font.family": "DejaVu Sans",
    })


def save_fig(fig, path: str | Path, **kwargs):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight", **kwargs)


# --- Forest-fire grid colormap -------------------------------------------

from matplotlib.colors import ListedColormap, BoundaryNorm  # noqa: E402

# state values: 0=empty (ash), 1=tree, 2=burning
FIRE_COLORS = ["#141414", "#1f7a3f", "#ffb020"]
FIRE_CMAP = ListedColormap(FIRE_COLORS)
FIRE_NORM = BoundaryNorm([-0.5, 0.5, 1.5, 2.5], FIRE_CMAP.N)


def render_grid(ax, grid: np.ndarray, cmap, norm, title: str | None = None):
    ax.imshow(grid, cmap=cmap, norm=norm, interpolation="nearest")
    ax.set_xticks([])
    ax.set_yticks([])
    if title:
        ax.set_title(title)
    return ax


def show_grid(ax, grid: np.ndarray, title: str | None = None):
    """Forest-fire-flavored grid render; kept for backward compatibility.
    Other systems should call `render_grid` with their own cmap/norm."""
    return render_grid(ax, grid, FIRE_CMAP, FIRE_NORM, title=title)


# --- Generic grid-to-video rendering --------------------------------------

def animate_grid_with_activity(frames: np.ndarray, cmap, norm, path: str | Path,
                                fps: int = 20, title: str = "", dpi: int = 130,
                                activity: np.ndarray | None = None,
                                activity_label: str = "activity"):
    """Render a (n_frames, L, L) stack of grid states to an mp4/gif file.

    If `activity` is given, a small running strip-chart is drawn beneath
    the grid so the burst-like nature of avalanches is visible alongside
    the spatial pattern. System-specific wrappers (see `wildfire/viz.py`,
    `sandpile/viz.py`) just supply their own cmap/norm and activity label.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n_frames = len(frames)

    if activity is not None:
        fig, (ax_grid, ax_act) = plt.subplots(
            2, 1, figsize=(5.5, 6.5), dpi=dpi,
            gridspec_kw={"height_ratios": [5, 1.2]},
            facecolor=BG,
        )
        ax_act.set_facecolor(BG)
        window = min(400, n_frames)
        line, = ax_act.plot([], [], color=ACCENT, lw=1.0)
        ax_act.set_xlim(0, window)
        ax_act.set_ylim(0, max(1, activity.max()) * 1.05)
        ax_act.set_yticks([])
        ax_act.set_xticks([])
        ax_act.set_title(activity_label, fontsize=9, color=FG, loc="left")
        for spine in ax_act.spines.values():
            spine.set_color(GRID_COLOR)
    else:
        fig, ax_grid = plt.subplots(figsize=(5.5, 5.5), dpi=dpi, facecolor=BG)

    im = ax_grid.imshow(frames[0], cmap=cmap, norm=norm, interpolation="nearest")
    ax_grid.set_xticks([])
    ax_grid.set_yticks([])
    ax_grid.set_title(title, color=FG)
    fig.tight_layout()

    def update(i):
        im.set_data(frames[i])
        artists = [im]
        if activity is not None:
            lo = max(0, i - window)
            xs = np.arange(lo, i + 1) - lo
            line.set_data(xs, activity[lo:i + 1])
            ax_act.set_xlim(0, window)
            artists.append(line)
        return artists

    anim = animation.FuncAnimation(fig, update, frames=n_frames, blit=True)

    if path.suffix == ".gif":
        anim.save(path, writer=animation.PillowWriter(fps=fps))
    elif _HAVE_FFMPEG:
        anim.save(path, writer=animation.FFMpegWriter(fps=fps, bitrate=1800))
    else:
        anim.save(path.with_suffix(".gif"), writer=animation.PillowWriter(fps=fps))

    plt.close(fig)
    return path
