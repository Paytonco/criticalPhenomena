"""
Drossel-Schwabl forest-fire model: a minimal stochastic cellular automaton
that self-organizes to a critical state without any fine-tuned parameter.

Rules, applied synchronously to every cell at each discrete time step:
  1. A burning tree becomes empty (ash).
  2. A tree with at least one burning nearest neighbor catches fire.
  3. A tree with no burning neighbor spontaneously ignites with
     probability `f` (lightning).
  4. An empty site sprouts a tree with probability `p` (growth).

Criticality requires a separation of time scales, p >> f, so that a fire
can burn through a cluster much faster than the forest regrows. The
lattice then self-tunes to hover just below the site-percolation
threshold: forest density is a dynamical output, not an input.

Boundary conditions are periodic (the grid is a torus) to avoid edge
effects truncating large fires.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

EMPTY, TREE, BURNING = 0, 1, 2
_STATE_NAMES = {EMPTY: "empty", TREE: "tree", BURNING: "burning"}


@dataclass
class RunResult:
    """Time series & event statistics collected from a simulation run."""

    activity: np.ndarray            # trees ignited per time step, shape (n_steps,)
    density: np.ndarray             # fraction of grid covered by trees, shape (n_steps,)
    fire_sizes: np.ndarray          # total trees burned, one entry per avalanche
    fire_durations: np.ndarray      # time steps an avalanche stayed alive
    snapshots: np.ndarray | None = field(default=None)  # (n_snap, L, L) state grids
    snapshot_steps: np.ndarray | None = field(default=None)

    @property
    def n_avalanches(self) -> int:
        return len(self.fire_sizes)


class ForestFireModel:
    """A single L x L Drossel-Schwabl forest-fire lattice."""

    def __init__(self, L: int = 64, p: float = 1e-3, f: float = 1e-5,
                 initial_density: float = 0.4, seed: int | None = None):
        if not (0 < f < p < 1):
            raise ValueError("require 0 < f < p < 1 for time-scale separation")
        self.L = L
        self.p = p
        self.f = f
        self.rng = np.random.default_rng(seed)
        self.grid = (self.rng.random((L, L)) < initial_density).astype(np.int8)
        self.t = 0

    @property
    def density(self) -> float:
        return float((self.grid == TREE).mean())

    def step(self) -> int:
        """Advance the automaton by one synchronous time step.

        Returns the number of trees that newly ignited this step (the
        instantaneous "activity").
        """
        L, grid, rng = self.L, self.grid, self.rng
        burning_mask = grid == BURNING
        tree_mask = grid == TREE

        neighbor_burning = (
            np.roll(burning_mask, 1, axis=0) | np.roll(burning_mask, -1, axis=0) |
            np.roll(burning_mask, 1, axis=1) | np.roll(burning_mask, -1, axis=1)
        )
        ignite_by_fire = tree_mask & neighbor_burning
        ignite_by_lightning = tree_mask & ~neighbor_burning & (rng.random((L, L)) < self.f)
        new_burning = ignite_by_fire | ignite_by_lightning

        grid[burning_mask] = EMPTY
        grid[new_burning] = BURNING
        grow = (grid == EMPTY) & (rng.random((L, L)) < self.p)
        grid[grow] = TREE

        self.t += 1
        return int(new_burning.sum())

    def run(self, n_steps: int, snapshot_every: int | None = None,
            progress: bool = False) -> RunResult:
        """Run the automaton for `n_steps` and collect statistics.

        Avalanches are identified from the activity time series: a maximal
        run of consecutive steps with nonzero activity is one avalanche;
        its `size` is the total number of ignitions summed over that run
        and its `duration` is the number of steps it lasted.
        """
        activity = np.empty(n_steps, dtype=np.int32)
        density = np.empty(n_steps, dtype=np.float32)

        snaps, snap_steps = [], []
        do_snap = snapshot_every is not None

        iterator = range(n_steps)
        if progress:
            from tqdm.auto import tqdm
            iterator = tqdm(iterator, desc="simulating")

        for t in iterator:
            a = self.step()
            activity[t] = a
            density[t] = self.density
            if do_snap and t % snapshot_every == 0:
                snaps.append(self.grid.copy())
                snap_steps.append(t)

        fire_sizes, fire_durations = _extract_avalanches(activity)

        return RunResult(
            activity=activity,
            density=density,
            fire_sizes=fire_sizes,
            fire_durations=fire_durations,
            snapshots=np.array(snaps) if do_snap else None,
            snapshot_steps=np.array(snap_steps) if do_snap else None,
        )

    def run_frames(self, n_steps: int) -> tuple[np.ndarray, np.ndarray]:
        """Run and return (frames, activity) for every step (for animation).

        Only use this for short runs -- it stores one LxL array per step.
        """
        frames = np.empty((n_steps, self.L, self.L), dtype=np.int8)
        activity = np.empty(n_steps, dtype=np.int32)
        for t in range(n_steps):
            activity[t] = self.step()
            frames[t] = self.grid
        return frames, activity

    def capture_avalanche(self, min_size: int, max_steps: int = 500_000,
                           context_before: int = 5, context_after: int = 15):
        """Run until a single avalanche of at least `min_size` trees occurs,
        then return (frames, activity) spanning a short pre-ignition window
        through the avalanche's full burnout plus a short tail.

        Used to make a close-up "anatomy of one avalanche" video, as
        opposed to `run_frames`, which just captures whatever happens.
        Returns (None, None) if no qualifying avalanche occurs within
        `max_steps`.
        """
        from collections import deque
        pre_frames = deque(maxlen=context_before)
        pre_activity = deque(maxlen=context_before)

        for _ in range(max_steps):
            pre_frames.append(self.grid.copy())
            a = self.step()
            pre_activity.append(a)

            if a > 0:
                # an avalanche just started (or is continuing) -- record forward
                avalanche_frames = [self.grid.copy()]
                avalanche_activity = [a]
                while True:
                    a2 = self.step()
                    avalanche_frames.append(self.grid.copy())
                    avalanche_activity.append(a2)
                    if a2 == 0:
                        break
                for _ in range(context_after):
                    avalanche_frames.append(self.grid.copy())
                    avalanche_activity.append(self.step())

                total_size = sum(x for x in avalanche_activity if x > 0)
                if total_size >= min_size:
                    frames = np.array(list(pre_frames)[:-1] + avalanche_frames)
                    activity = np.array(list(pre_activity)[:-1] + avalanche_activity)
                    return frames, activity
                # too small: keep only the tail as pre-context and continue
                for fr, ac in zip(avalanche_frames[-context_before:],
                                   avalanche_activity[-context_before:]):
                    pre_frames.append(fr)
                    pre_activity.append(ac)
        return None, None


def _extract_avalanches(activity: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Collapse an activity time series into (size, duration) per avalanche."""
    sizes, durations = [], []
    cur_size = 0
    cur_dur = 0
    for a in activity:
        if a > 0:
            cur_size += a
            cur_dur += 1
        elif cur_dur > 0:
            sizes.append(cur_size)
            durations.append(cur_dur)
            cur_size, cur_dur = 0, 0
    if cur_dur > 0:
        sizes.append(cur_size)
        durations.append(cur_dur)
    return np.array(sizes, dtype=np.int64), np.array(durations, dtype=np.int64)
