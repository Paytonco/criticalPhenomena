# Wildfire SOC: the Drossel-Schwabl forest-fire model

A minimal stochastic cellular automaton that self-organizes to a critical
state. See `notebooks/wildfire_soc.ipynb` for the full writeup, figures,
and videos; this is a quick reference for the code.

## Model (`model.py`)

`ForestFireModel(L, p, f, ...)` simulates an `L x L` torus where every
site is empty, a tree, or burning. Every synchronous tick:

1. burning -> empty
2. tree with a burning neighbor -> burning (fire spreads one lattice step)
3. tree with no burning neighbor -> burning, w.p. `f` (lightning)
4. empty -> tree, w.p. `p` (growth)

Criticality requires `f << p` (fire spreads much faster than the forest
regrows); given that, the forest density self-organizes to just below the
site-percolation threshold (~0.593) without being an input parameter.

Key methods:
- `.step()` — advance one tick, returns the ignition count (activity).
- `.run(n_steps, snapshot_every=None, progress=False)` — long runs for
  statistics; returns a `RunResult` with the activity/density time series
  and the extracted avalanche size/duration lists.
- `.run_frames(n_steps)` — short runs that keep every grid frame, for
  animation.
- `.capture_avalanche(min_size, ...)` — runs until a single avalanche at
  least `min_size` trees occurs and returns just that window, for a
  close-up "anatomy of one avalanche" video.

## Rendering (`viz.py`)

`animate_grid(frames, path, ...)` turns a stack of grid states into an
MP4 (via a bundled static `ffmpeg` from `imageio-ffmpeg`, no system
install needed) with an optional activity strip-chart underneath.

## Reproducing the notebook

```
cd notebooks
jupyter nbconvert --to notebook --execute --inplace wildfire_soc.ipynb
```

The "long run" cell (~350k steps at L=64) takes a few minutes; everything
else is fast. Outputs land in `../figures/`, `../videos/`, `../data/`.

## Parameter notes

`L=64, p=1e-3, f=1e-5` was tuned empirically to hit a clean SOC regime
fast enough for an interactive notebook:

- `p` must be small enough that a lattice-spanning fire (duration ~ L/2
  ticks) completes before meaningful regrowth happens in its wake, or the
  fire never fully extinguishes ("smoldering" -- growth keeps feeding new
  fuel into an already-burning region). Rule of thumb: `p * L << 1`.
- `f` must be small enough that ignitions are rare per tick (`f * L^2` on
  the order of `0.01`), or multiple independent fires overlap and merge
  into one runaway conflagration instead of many separable avalanches.

Pushing `L` larger gives more decades of power-law scaling before the
finite-size cutoff, at roughly linear cost in wall-clock time per step.
