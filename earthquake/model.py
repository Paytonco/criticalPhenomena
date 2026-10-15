"""
The Olami-Feder-Christensen (OFC) model (1992): the standard SOC model
built specifically for earthquakes, as a grid of elastically coupled
fault patches.

Every site on an L x L grid holds a continuous stress F(i,j) in [0,
F_th). Two phases, never interleaved:

1. **Load:** tectonic plates drive the whole fault uniformly, so every
   site's stress increases by the same amount. Rather than simulate many
   tiny increments, we jump straight to the moment the most-stressed
   site reaches failure (`F_th - F.max()` added everywhere at once) --
   an exact, standard "extremal driving" shortcut, not an approximation.
2. **Rupture:** any site with F >= F_th slips -- it drops to 0 and each
   of its (up to four) neighbors gains a fraction `alpha` of the stress
   it just released. If that pushes a neighbor over threshold too, it
   ruptures on the next sweep, and so on until the whole fault is stable
   again -- an earthquake, its magnitude set by how far the rupture
   spread.

Unlike the sandpile's `z_c` (which redistributes *all* released load,
`alpha = 1/4` in 2D), the OFC model is built to be **non-conservative**:
`alpha < 1/4` means some stress is permanently lost at every rupture --
radiated away as seismic waves and heat, physically, rather than passed
on. This is what makes the model specific to earthquakes rather than a
sandpile with the sites relabeled: real fault ruptures dissipate energy,
and Olami, Feder & Christensen's key result was that SOC survives, and
even becomes *more* robust, once you stop insisting on conservation.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import numpy as np

F_TH = 1.0


@dataclass
class RunResult:
    """Time series & earthquake statistics collected from a simulation run."""

    activity: np.ndarray            # sites rupturing per tick, shape (n_ticks,)
    mean_stress: np.ndarray         # mean F after each event, shape (n_events,)
    sizes: np.ndarray               # total ruptures, one entry per earthquake
    durations: np.ndarray           # relaxation sweeps per earthquake
    areas: np.ndarray               # distinct sites ruptured at least once
    moments: np.ndarray             # total stress released (a magnitude proxy)
    snapshots: np.ndarray | None = field(default=None)   # (n_snap, L, L) stress grids
    snapshot_steps: np.ndarray | None = field(default=None)

    @property
    def n_events(self) -> int:
        return len(self.sizes)


class OFCModel:
    """A single L x L Olami-Feder-Christensen fault with open boundaries."""

    def __init__(self, L: int = 64, alpha: float = 0.2, f_th: float = F_TH,
                 seed: int | None = None):
        if not (0 < alpha <= 0.25):
            raise ValueError("alpha must be in (0, 0.25]; 0.25 is the conservative limit")
        self.L = L
        self.alpha = alpha
        self.f_th = f_th
        self.rng = np.random.default_rng(seed)
        self.F = self.rng.uniform(0, f_th, size=(L, L))
        self.t = 0

    @property
    def mean_stress(self) -> float:
        return float(self.F.mean())

    def render_code(self) -> np.ndarray:
        """Current stress grid bucketed into 4 levels, for `earthquake.viz.show_grid`."""
        return self._render_code()

    def _render_code(self, ruptured_mask: np.ndarray | None = None) -> np.ndarray:
        frac = np.clip(self.F / self.f_th, 0.0, 0.9999)
        code = np.floor(frac * 4).astype(np.int8)
        if ruptured_mask is not None:
            code[ruptured_mask] = 4
        return code

    def _load(self) -> float:
        """Uniformly load every site until the most-stressed one fails."""
        delta = self.f_th - self.F.max()
        self.F += delta
        return delta

    def _relax(self, on_microstep=None) -> tuple[list[int], int, float]:
        """Rupture until stable. Returns (per-sweep rupture counts, area, moment).

        Rupture is parallel within a sweep -- every currently unstable
        site releases simultaneously. `on_microstep(n, mask)`, if given,
        is called after each sweep, same convention as
        `sandpile.model.SandpileModel._relax`.
        """
        F, alpha, f_th = self.F, self.alpha, self.f_th
        counts = []
        ever = np.zeros_like(F, dtype=bool)
        moment = 0.0
        while True:
            unstable = F >= f_th
            n = int(unstable.sum())
            if n == 0:
                break
            amounts = np.where(unstable, F, 0.0)
            F[unstable] = 0.0
            F[:-1, :] += alpha * amounts[1:, :]
            F[1:, :] += alpha * amounts[:-1, :]
            F[:, :-1] += alpha * amounts[:, 1:]
            F[:, 1:] += alpha * amounts[:, :-1]
            ever |= unstable
            counts.append(n)
            moment += float(amounts.sum())
            if on_microstep is not None:
                on_microstep(n, unstable)
        return counts, int(ever.sum()), moment

    def step(self) -> tuple[int, int, int, float]:
        """Load the fault to the next failure and let the earthquake run
        its course. Returns (size, duration, area, moment)."""
        self._load()
        counts, area, moment = self._relax()
        self.t += 1
        return sum(counts), len(counts), area, moment

    def run(self, n_events: int, snapshot_every: int | None = None,
            progress: bool = False) -> RunResult:
        """Simulate `n_events` earthquakes and collect statistics.

        `activity` records ruptures-per-tick at full sweep resolution,
        the same convention `wildfire.model` and `sandpile.model` use --
        so the same `common.spectral` tools apply unchanged. Unlike the
        sandpile, every driving step here triggers at least one rupture
        by construction (loading always stops exactly at the next
        failure), so there's no quiescent "0" tick between earthquakes.
        """
        activity_chunks = []
        sizes, durations, areas, moments = [], [], [], []
        mean_stress = np.empty(n_events, dtype=np.float64)
        snaps, snap_steps = [], []
        do_snap = snapshot_every is not None

        iterator = range(n_events)
        if progress:
            from tqdm.auto import tqdm
            iterator = tqdm(iterator, desc="simulating earthquakes")

        for k in iterator:
            self._load()
            counts, area, moment = self._relax()
            self.t += 1
            activity_chunks.append(np.array(counts, dtype=np.int64))
            sizes.append(sum(counts))
            durations.append(len(counts))
            areas.append(area)
            moments.append(moment)
            mean_stress[k] = self.F.mean()
            if do_snap and k % snapshot_every == 0:
                snaps.append(self.F.copy())
                snap_steps.append(k)

        return RunResult(
            activity=np.concatenate(activity_chunks),
            mean_stress=mean_stress,
            sizes=np.array(sizes, dtype=np.int64),
            durations=np.array(durations, dtype=np.int64),
            areas=np.array(areas, dtype=np.int64),
            moments=np.array(moments, dtype=np.float64),
            snapshots=np.array(snaps) if do_snap else None,
            snapshot_steps=np.array(snap_steps) if do_snap else None,
        )

    def _macro_step(self):
        """Load to the next failure and fully relax, returning every
        (frame, activity) tick generated, plus (size, duration, area, moment)."""
        ticks = []

        def record(n, mask):
            ticks.append((self._render_code(mask), n))

        self._load()
        counts, area, moment = self._relax(on_microstep=record)
        self.t += 1
        return ticks, sum(counts), len(counts), area, moment

    def run_frames(self, n_events: int) -> tuple[np.ndarray, np.ndarray]:
        """Run and return (frames, activity) at full tick resolution (for
        animation). Only use this for short runs."""
        all_frames, all_activity = [], []
        for _ in range(n_events):
            ticks, *_ = self._macro_step()
            for frame, a in ticks:
                all_frames.append(frame)
                all_activity.append(a)
        return np.array(all_frames), np.array(all_activity)

    def capture_avalanche(self, min_size: int, max_events: int = 500_000,
                           context_before: int = 5, context_after: int = 15):
        """Run until a single earthquake of at least `min_size` ruptures
        occurs, then return (frames, activity) spanning a short
        pre-event window through the earthquake's full relaxation plus a
        short tail. Returns (None, None) if none qualifies within
        `max_events`.
        """
        pre = deque(maxlen=context_before)
        for _ in range(max_events):
            ticks, size, duration, area, moment = self._macro_step()
            if size >= min_size:
                tail = []
                for _ in range(context_after):
                    t2, s2, *_ = self._macro_step()
                    tail.extend(t2)
                    if s2 >= min_size:
                        # another qualifying earthquake started right away;
                        # stop rather than merging two events into one video
                        break
                all_ticks = list(pre) + ticks + tail
                frames = np.array([f for f, _ in all_ticks])
                activity = np.array([a for _, a in all_ticks])
                return frames, activity
            for tick in ticks[-context_before:]:
                pre.append(tick)
        return None, None
