# Snow Avalanche SOC: the Bak-Tang-Wiesenfeld sandpile

The original SOC toy model (1987), reframed here as snow loading a slope
until it slides. See `notebooks/snow_avalanche_soc.ipynb` for the full
writeup, figures, and videos; this is a quick reference for the code.

## Model (`model.py`)

`SandpileModel(L, z_c=4, initial="random"|"zeros", seed=...)` simulates
an `L x L` grid where every site holds an integer load `z(i,j)`. Two
phases, never interleaved:

1. **Drive:** add one unit to a uniformly random site.
2. **Relax:** while any site has `z >= z_c`, every such site topples in
   parallel -- it drops by `z_c` and each neighbor gains one. Load that
   would leave the grid edge is dissipated (open boundaries). Repeat
   until stable.

Because driving and relaxation are exact, separate phases by
construction (not a tuned probability, unlike `wildfire.model`), the
model self-organizes to a critical mean load reliably for almost any
`L` -- no parameter search needed.

Key methods (mirroring `wildfire.model.ForestFireModel`'s API):
- `.step()` -- one grain + full relaxation; returns (size, duration, area).
- `.run(n_grains, snapshot_every=None, progress=False)` -- long runs for
  statistics; returns a `RunResult` with the tick-resolution activity
  time series and per-avalanche size/duration/area lists.
- `.run_frames(n_grains)` -- short runs keeping every tick's grid state,
  for animation.
- `.capture_avalanche(min_size, ...)` -- runs until a single avalanche at
  least `min_size` topples occurs, for a close-up video.
- `.render_code()` -- current load grid bucketed for `sandpile.viz.show_grid`.

## Rendering (`viz.py`)

Thin wrappers around `common.viz.render_grid` / `animate_grid_with_activity`
with a snow-specific colormap (pale = thin snow, deep blue = heavily
loaded, bright red = releasing this frame).

## Reproducing the notebook

```
cd notebooks
jupyter nbconvert --to notebook --execute --inplace snow_avalanche_soc.ipynb
```

The "long run" cell (150k grains at L=64) takes a couple of minutes;
everything else is fast. Outputs land in `../figures/`, `../videos/`,
`../data/`.

## Notes on this model vs. the wildfire one

- No pathological parameter regime to avoid: unlike the forest-fire
  model (which needs `p` small enough that a fire finishes before
  meaningful regrowth in its wake, or it "smolders" forever), a finite
  sandpile relaxation is guaranteed to terminate. `L` is basically the
  only knob.
- The full avalanche-size distribution has real, literature-documented
  curvature (see e.g. Priezzhev et al. on BTW multiscaling) -- a clean
  power law only describes the tail (roughly the largest 5-10% of
  avalanches here). The notebook fits this honestly by scanning how the
  likelihood-ratio test result changes with the candidate cutoff,
  rather than reporting whatever `common.powerlaw.fit_power_law`'s
  default settings happen to return.
- The activity power spectrum's fitted exponent is both closer to
  textbook pink noise and much more stable across fitting-window choices
  than the wildfire model's was -- a nice illustration that "1/f-type
  noise in SOC" is a real, robust qualitative phenomenon whose *exact*
  exponent is nonetheless model- and observable-dependent, not a
  universal constant.
