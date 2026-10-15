# Earthquake SOC: the Olami-Feder-Christensen model

The standard SOC model built specifically for faults: a grid of
elastically coupled blocks that load slowly and slip suddenly. See
`notebooks/earthquake_soc.ipynb` for the full writeup, figures, and
videos; this is a quick reference for the code.

## Model (`model.py`)

`OFCModel(L, alpha=0.2, f_th=1.0, seed=...)` simulates an `L x L` grid
where every site holds a continuous stress `F(i,j) in [0, f_th)`. Two
phases, never interleaved:

1. **Load:** the whole fault is driven uniformly, so we jump directly to
   the moment the most-stressed site fails (`f_th - F.max()` added
   everywhere at once -- an exact "extremal driving" shortcut).
2. **Rupture:** any site with `F >= f_th` slips -- it drops to 0 and each
   neighbor gains `alpha` times the stress just released. Cascades until
   the fault is stable again.

`alpha` is the key parameter this model has that `wildfire.model` and
`sandpile.model` don't: it's the fraction of released stress that's
*conserved* rather than dissipated. `alpha = 0.25` (2D, 4 neighbors) is
the fully conservative limit -- mathematically identical to the
sandpile's toppling rule -- but real faults are non-conservative (they
radiate energy as seismic waves and heat), which is the entire point of
this model. See the notebook's `alpha`-sweep section for how much
dissipation the system can tolerate before criticality breaks down.

Key methods (same shape as `wildfire.model.ForestFireModel` and
`sandpile.model.SandpileModel`):
- `.step()` -- one load-to-failure + full relaxation; returns (size, duration, area, moment).
- `.run(n_events, snapshot_every=None, progress=False)` -- long runs for
  statistics; returns a `RunResult` with the tick-resolution activity
  time series and per-earthquake size/duration/area/moment lists.
- `.run_frames(n_events)` -- short runs keeping every tick's grid state, for animation.
- `.capture_avalanche(min_size, ...)` -- runs until a single earthquake at
  least `min_size` ruptures occurs, for a close-up video.
- `.render_code()` -- current stress grid bucketed for `earthquake.viz.show_grid`.

## Rendering (`viz.py`)

Thin wrapper around `common.viz.render_grid` / `animate_grid_with_activity`
with a stress-specific colormap (pale = low stress, dark maroon = near
failure, bright yellow = rupturing this frame).

## Reproducing the notebook

```
cd notebooks
jupyter nbconvert --to notebook --execute --inplace earthquake_soc.ipynb
```

The main statistics run (200k earthquakes at L=64, alpha=0.2) takes
about a minute; the alpha-sweep adds another ~30s; everything else is
fast. Outputs land in `../figures/`, `../videos/`, `../data/`.

## Notes on this model vs. the other two

- **A genuinely non-conservative dynamics.** Unlike the sandpile, stress
  is not exactly redistributed -- some is permanently lost at every
  rupture. Whether SOC survives that is itself the interesting question,
  and the answer is conditional: `alpha=0.2` gives a clean power law;
  `alpha <= 0.1` collapses it into something closer to exponential (see
  the notebook's `alpha`-sweep figure). This is the most direct
  illustration in this series that self-organized criticality isn't
  automatic for any local rule -- it depends on where in parameter space
  the model sits.
- **A deterministic-dynamics artifact worth knowing about.** Starting the
  fault from an exactly uniform (all-zero) stress produces a strikingly
  symmetric, wave-like pattern rather than the disordered texture typical
  of the critical state -- because both the loading and rupture rules are
  fully deterministic, so a perfectly symmetric initial condition stays
  perfectly symmetric. The notebook shows this explicitly (and contrasts
  it against a random initial condition) rather than presenting it as the
  generic SOC texture, which it isn't.
- **A direct link to real seismology.** The avalanche-size statistics
  translate into an actual Gutenberg-Richter plot (`log10 N(>=M)` vs.
  magnitude), and the fitted b-value comes out close to the real Earth's
  `b ~ 1` -- the most concrete "this toy model reproduces a named
  real-world law" moment in the series so far.
