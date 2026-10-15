# SOC Toy Systems

Small, self-contained demonstrations of **self-organized criticality
(SOC)** — systems that drive themselves to a critical point through their
own local dynamics, with no parameter fine-tuned by hand. Each demo is a
minimal model built from a handful of local rules that nonetheless
produces the two classic hallmarks of criticality:

- **Power-law distributed event sizes** — no characteristic scale, verified
  with a proper maximum-likelihood fit and a likelihood-ratio test against
  an exponential alternative (not just "it looks straight on a log-log
  plot").
- **1/f-type ("colored") noise** in the global activity signal — the
  temporal fingerprint of the same criticality. Reported honestly: the
  *exact* exponent turns out to be model- and window-dependent, so each
  notebook fits it properly (log-binned, bounded fitting window) rather
  than asserting a universal beta=1.

## Layout

```
toySystems/
├── common/              shared, system-agnostic analysis code
│   ├── powerlaw.py       discrete power-law MLE fitting + xmin selection
│   ├── spectral.py       Welch PSD + 1/f^beta slope fitting (log-binned)
│   └── viz.py            shared plot style, palette, figure saving,
│                         and the generic grid -> video renderer
│
├── wildfire/            demo 1: Drossel-Schwabl forest-fire model
│   ├── model.py           the cellular automaton
│   ├── viz.py             thin wrapper: fire colormap + common's video renderer
│   ├── notebooks/         wildfire_soc.ipynb -- the full writeup
│   ├── figures/           exported PNGs (populated by the notebook)
│   ├── videos/            exported MP4s (populated by the notebook)
│   └── data/              cached simulation output (.npz)
│
├── sandpile/            demo 2: Bak-Tang-Wiesenfeld sandpile (snow avalanches)
│   ├── model.py           the abelian sandpile automaton
│   ├── viz.py             thin wrapper: snow colormap + common's video renderer
│   ├── notebooks/         snow_avalanche_soc.ipynb -- the full writeup
│   ├── figures/, videos/, data/  -- same shape as wildfire's
│
├── earthquake/          demo 3: Olami-Feder-Christensen model (earthquakes)
│   ├── model.py           the non-conservative fault automaton
│   ├── viz.py             thin wrapper: stress colormap + common's video renderer
│   ├── notebooks/         earthquake_soc.ipynb -- the full writeup
│   ├── figures/, videos/, data/  -- same shape as wildfire's
│
└── <next-system>/       same shape: model.py, viz.py, notebooks/, figures/, videos/, data/
```

## Adding a new system

Each demo is independent but follows the same shape so they read as one
series:

1. Create `<system>/` with `model.py` (the dynamics), `viz.py` (rendering),
   and `notebooks/`, `figures/`, `videos/`, `data/` subfolders.
2. Reuse `common/powerlaw.py` and `common/spectral.py` for the SOC
   diagnostics rather than re-deriving them — they're written to be
   data-agnostic (they just take arrays of event sizes / an activity time
   series).
3. Reuse `common/viz.py` for plot styling and video rendering: define your
   own categorical `ListedColormap`/`BoundaryNorm` for the system's grid
   states (see `wildfire/viz.py` or `sandpile/viz.py` for the pattern --
   both are ~20-line wrappers around `common.viz.render_grid` and
   `common.viz.animate_grid_with_activity`), rather than reimplementing
   the animation loop.
4. The notebook should tell the same three-part story: build the model,
   show it running (figures + video), then demonstrate both SOC
   hallmarks with real statistical tests, not just visual inspection --
   and report what the tests actually say, including when a distribution
   only follows a clean power law in its tail, when a fitted exponent is
   sensitive to the fitting window, or when criticality itself turns out
   to depend on where the model's parameters sit (see all three existing
   notebooks' honest treatment of exactly these).

Candidates for future demos: an Ising model quenched near its critical
temperature, a branching neuronal-avalanche model, or an epidemic
(SIR-type) model tuned to its own percolation threshold.

## Setup

```
pip install numpy scipy matplotlib pandas tqdm ipywidgets imageio imageio-ffmpeg jupyter nbconvert
```

No system `ffmpeg` install is required — `imageio-ffmpeg` bundles a static
binary and `common/viz.py` points matplotlib at it automatically.
