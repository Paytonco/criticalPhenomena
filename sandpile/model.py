"""
The Bak-Tang-Wiesenfeld (BTW) sandpile model: the original toy model of
self-organized criticality, reframed here as a snowpack sliding toward its
angle of repose.

Every site on an L x L grid holds an integer "load" z(i,j) -- how many
units of snow have piled up there, a proxy for local slope steepness.
Snow falls one unit at a time onto a random site (slow driving). Whenever
a site's load reaches a critical threshold `z_c`, it releases: it drops by
`z_c` and each of its up to four neighbors gains one unit -- a local
slope failure that can trigger neighboring failures in turn, cascading
into an avalanche. Grid edges are open (snow that slides off the boundary
is gone for good), which is what lets the pile relax instead of just
growing forever.

Unlike the wildfire model, there is no probability to tune for a
separation of time scales: driving (add one grain) and relaxation (topple
until stable) are already two distinct phases by construction, so the
model self-organizes to critical reliably for almost any L.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import numpy as np


@dataclass
class RunResult:
    """Time series & avalanche statistics collected from a simulation run."""

    activity: np.ndarray            # grains toppling per tick, shape (n_ticks,)
    mean_load: np.ndarray           # mean z / z_c after each grain drop, shape (n_grains,)
    avalanche_sizes: np.ndarray     # total topples, one entry per avalanche
    avalanche_durations: np.ndarray  # relaxation sweeps (parallel toppling waves)
    avalanche_areas: np.ndarray     # distinct sites toppled at least once
    snapshots: np.ndarray | None = field(default=None)   # (n_snap, L, L) load grids
    snapshot_steps: np.ndarray | None = field(default=None)

    @property
    def n_avalanches(self) -> int:
        return len(self.avalanche_sizes)


class SandpileModel:
    """A single L x L abelian sandpile with open (dissipative) boundaries."""

    # 4-neighbor (von Neumann) toppling: this many units leave a toppling
    # site, one to each neighbor -- the 2D BTW convention.
    Z_C = 4

    def __init__(self, L: int = 64, z_c: int = Z_C, initial: str = "random",
                 seed: int | None = None):
        self.L = L
        self.z_c = z_c
        self.rng = np.random.default_rng(seed)
        if initial == "random":
            self.z = self.rng.integers(0, z_c, size=(L, L)).astype(np.int64)
        else:
            self.z = np.zeros((L, L), dtype=np.int64)
        self.t = 0

    @property
    def mean_load(self) -> float:
        return float(self.z.mean())

    def add_grain(self, site: tuple[int, int] | None = None) -> tuple[int, int]:
        if site is None:
            site = (int(self.rng.integers(self.L)), int(self.rng.integers(self.L)))
        self.z[site] += 1
        return site

    def _relax(self, on_microstep=None) -> tuple[list[int], int]:
        """Topple until stable. Returns (per-microstep topple counts, area).

        Toppling is parallel within a microstep -- every currently
        unstable site releases simultaneously, exactly the abelian
        sandpile algorithm (the final stable configuration and total
        topple count don't depend on the order sites are processed in).
        `on_microstep(n, mask)`, if given, is called after each sweep
        with the topple count and the boolean mask of sites that just
        toppled -- used by `run_frames`/`capture_avalanche` to record
        video frames at full micro-resolution without duplicating this
        loop.
        """
        z, z_c = self.z, self.z_c
        counts = []
        ever_toppled = np.zeros_like(z, dtype=bool)
        while True:
            unstable = z >= z_c
            n = int(unstable.sum())
            if n == 0:
                break
            toppled = unstable.astype(np.int64)
            z -= toppled * z_c
            # each toppling site gives one unit to each in-bounds neighbor;
            # a neighbor off the edge simply loses that unit (open boundary)
            z[:-1, :] += toppled[1:, :]
            z[1:, :] += toppled[:-1, :]
            z[:, :-1] += toppled[:, 1:]
            z[:, 1:] += toppled[:, :-1]
            ever_toppled |= unstable
            counts.append(n)
            if on_microstep is not None:
                on_microstep(n, unstable)
        return counts, int(ever_toppled.sum())

    def step(self) -> tuple[int, int, int]:
        """Drop one grain at a random site and fully relax.

        Returns (size, duration, area), all zero if the grain didn't
        trigger any toppling.
        """
        self.add_grain()
        counts, area = self._relax()
        self.t += 1
        if not counts:
            return 0, 0, 0
        return sum(counts), len(counts), area

    def run(self, n_grains: int, snapshot_every: int | None = None,
            progress: bool = False) -> RunResult:
        """Drop `n_grains` grains (one at a time, each fully relaxed) and
        collect avalanche statistics.

        `activity` records grains-toppling-per-tick at full microstep
        resolution (a quiescent grain drop contributes a single 0), the
        same convention `wildfire.model.ForestFireModel.run` uses for its
        per-step ignition counts -- so the same `common.spectral` tools
        apply unchanged.
        """
        activity_chunks = []
        sizes, durations, areas = [], [], []
        mean_load = np.empty(n_grains, dtype=np.float64)
        snaps, snap_steps = [], []
        do_snap = snapshot_every is not None

        iterator = range(n_grains)
        if progress:
            from tqdm.auto import tqdm
            iterator = tqdm(iterator, desc="dropping grains")

        for k in iterator:
            self.add_grain()
            counts, area = self._relax()
            self.t += 1
            if counts:
                activity_chunks.append(np.array(counts, dtype=np.int64))
                sizes.append(sum(counts))
                durations.append(len(counts))
                areas.append(area)
            else:
                activity_chunks.append(np.zeros(1, dtype=np.int64))
            mean_load[k] = self.z.mean()
            if do_snap and k % snapshot_every == 0:
                snaps.append(self.z.copy())
                snap_steps.append(k)

        return RunResult(
            activity=np.concatenate(activity_chunks),
            mean_load=mean_load,
            avalanche_sizes=np.array(sizes, dtype=np.int64),
            avalanche_durations=np.array(durations, dtype=np.int64),
            avalanche_areas=np.array(areas, dtype=np.int64),
            snapshots=np.array(snaps) if do_snap else None,
            snapshot_steps=np.array(snap_steps) if do_snap else None,
        )

    def render_code(self) -> np.ndarray:
        """The current load grid bucketed to 0..z_c-1, suitable for
        `sandpile.viz.show_grid`."""
        return self._render_code()

    def _render_code(self, toppled_mask: np.ndarray | None = None) -> np.ndarray:
        """Bucket the load grid into 0..z_c-1, flagging just-toppled sites
        with the sentinel value z_c -- the same "state + highlight in one
        integer array" trick `wildfire.model` uses for its categorical
        grid renders."""
        code = np.clip(self.z, 0, self.z_c - 1).astype(np.int8)
        if toppled_mask is not None:
            code[toppled_mask] = self.z_c
        return code

    def _macro_step(self):
        """Add one grain and fully relax, returning every (frame, activity)
        tick generated along the way, plus (size, duration, area)."""
        ticks = []

        def record(n, mask):
            ticks.append((self._render_code(mask), n))

        self.add_grain()
        counts, area = self._relax(on_microstep=record)
        self.t += 1
        if not counts:
            ticks.append((self._render_code(), 0))
            return ticks, 0, 0, 0
        return ticks, sum(counts), len(counts), area

    def run_frames(self, n_grains: int) -> tuple[np.ndarray, np.ndarray]:
        """Run and return (frames, activity) at full tick resolution (for
        animation). Only use this for short runs -- it stores one LxL
        array per tick, and an avalanche can have many ticks."""
        all_frames, all_activity = [], []
        for _ in range(n_grains):
            ticks, *_ = self._macro_step()
            for frame, a in ticks:
                all_frames.append(frame)
                all_activity.append(a)
        return np.array(all_frames), np.array(all_activity)

    def capture_avalanche(self, min_size: int, max_grains: int = 500_000,
                           context_before: int = 5, context_after: int = 15):
        """Run until a single avalanche of at least `min_size` topples
        occurs, then return (frames, activity) spanning a short
        pre-ignition window through the avalanche's full relaxation plus
        a short tail. Returns (None, None) if none qualifies within
        `max_grains`.
        """
        pre = deque(maxlen=context_before)
        for _ in range(max_grains):
            ticks, size, duration, area = self._macro_step()
            if size >= min_size and size > 0:
                tail = []
                for _ in range(context_after):
                    t2, s2, *_ = self._macro_step()
                    tail.extend(t2)
                    if s2 > 0:
                        break
                all_ticks = list(pre) + ticks + tail
                frames = np.array([f for f, _ in all_ticks])
                activity = np.array([a for _, a in all_ticks])
                return frames, activity
            for tick in ticks[-context_before:]:
                pre.append(tick)
        return None, None
