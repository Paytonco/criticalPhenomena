"""
Fitting and testing discrete power-law distributions in event-size data.

Implements the Clauset, Shalizi & Newman (2009) approach: maximum-likelihood
exponent estimation for a fixed lower cutoff `xmin`, automatic selection of
`xmin` by minimizing the Kolmogorov-Smirnov distance to the fitted model,
and a log-likelihood-ratio comparison against an exponential alternative
(the most common non-power-law explanation for an apparently heavy tail).

Shared across every critical-system demo in this repo -- nothing here is
specific to wildfires.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import zeta as hurwitz_zeta


@dataclass
class PowerLawFit:
    alpha: float          # scaling exponent: P(x) ~ x^-alpha
    xmin: int             # lower cutoff above which the power law holds
    ks_distance: float    # goodness-of-fit statistic at the chosen xmin
    n_tail: int           # number of observations with x >= xmin
    loglik_ratio_exp: float  # >0 favors power law over exponential
    p_value_exp: float       # two-sided normal-approx significance of that ratio


def _discrete_loglik(x: np.ndarray, alpha: float, xmin: int) -> float:
    n = len(x)
    return -n * np.log(hurwitz_zeta(alpha, xmin)) - alpha * np.sum(np.log(x))


def fit_alpha(x: np.ndarray, xmin: int, bounds=(1.01, 6.0)) -> float:
    """MLE exponent for a discrete power law x >= xmin, x integer >= 1."""
    tail = x[x >= xmin]
    res = minimize_scalar(lambda a: -_discrete_loglik(tail, a, xmin),
                           bounds=bounds, method="bounded")
    return float(res.x)


def _ccdf(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Empirical complementary CDF P(X >= x) at the observed unique values.

    O(n log n) via sorted unique counts, rather than the O(n * n_unique)
    you'd get from asking "how many points are >= v" separately for each v.
    """
    xs, counts = np.unique(x, return_counts=True)
    n = len(x)
    ccdf = 1.0 - (np.cumsum(counts) - counts) / n
    return xs, ccdf


def _model_ccdf(xs: np.ndarray, alpha: float, xmin: int) -> np.ndarray:
    return hurwitz_zeta(alpha, xs) / hurwitz_zeta(alpha, xmin)


def _xmin_candidates(x: np.ndarray, max_candidates: int = 80) -> np.ndarray:
    """A tractable set of xmin candidates to scan.

    Scanning every unique observed value is O(n_unique) MLE fits, each
    itself O(n) -- fine for a few thousand events, painfully slow for the
    tens of thousands an efficient vectorized simulation can produce in
    seconds. Log-spaced candidates lose essentially no resolution (xmin
    is being chosen to the nearest order of magnitude in practice anyway)
    while keeping the search fast at any sample size.
    """
    uniq = np.unique(x)
    if len(uniq) <= max_candidates:
        return uniq
    log_targets = np.logspace(np.log10(uniq[0]), np.log10(uniq[-1]), max_candidates)
    idx = np.unique(np.clip(np.searchsorted(uniq, log_targets), 0, len(uniq) - 1))
    return uniq[idx]


def fit_power_law(x: np.ndarray, xmin_candidates: np.ndarray | None = None,
                   min_tail_fraction: float = 0.2) -> PowerLawFit:
    """Fit a discrete power law with automatic lower-cutoff selection.

    For each candidate xmin, fits alpha by MLE and scores the fit by the
    KS distance between the empirical and model CCDFs restricted to the
    tail x >= xmin; the xmin minimizing KS distance is kept.

    Any finite lattice has a hard cutoff (fires can't exceed L^2 sites),
    so the extreme tail is always sparse and under-dispersed relative to
    a true infinite power law -- naive KS-minimization is well known to
    latch onto that handful of points and report a spuriously steep,
    "well-fit" exponent there. `min_tail_fraction` guards against this by
    requiring the candidate tail to retain at least this fraction of the
    data, keeping the fit in the statistically meaningful scaling regime.
    """
    x = np.asarray(x)
    x = x[x >= 1]
    n_total = len(x)
    min_tail_n = max(30, int(min_tail_fraction * n_total))
    if xmin_candidates is None:
        xmin_candidates = _xmin_candidates(x)

    best = None
    for xmin in xmin_candidates:
        tail = x[x >= xmin]
        if len(tail) < min_tail_n:
            continue
        alpha = fit_alpha(x, int(xmin))
        xs, emp = _ccdf(tail)
        model = _model_ccdf(xs, alpha, int(xmin))
        d = float(np.max(np.abs(emp - model)))
        if best is None or d < best[0]:
            best = (d, int(xmin), alpha, len(tail))

    if best is None:
        raise ValueError("not enough data to fit a power law")
    d, xmin, alpha, n_tail = best

    tail = x[x >= xmin]
    ll_pl = _discrete_loglik(tail, alpha, xmin)
    lam = 1.0 / tail.mean()
    ll_exp = np.sum(np.log(lam) - lam * (tail - xmin))
    R = ll_pl - ll_exp
    per_point = np.log(_pmf_powerlaw(tail, alpha, xmin)) - np.log(_pmf_exp(tail, lam, xmin))
    sigma = per_point.std(ddof=1) if len(per_point) > 1 else 1e-9
    sqrt_n = np.sqrt(len(tail))
    z = R / (sigma * sqrt_n) if sigma > 0 else np.inf
    from scipy.stats import norm
    p_value = 2 * (1 - norm.cdf(abs(z)))

    return PowerLawFit(alpha=alpha, xmin=xmin, ks_distance=d, n_tail=n_tail,
                        loglik_ratio_exp=float(R), p_value_exp=float(p_value))


def _pmf_powerlaw(x: np.ndarray, alpha: float, xmin: int) -> np.ndarray:
    return (x.astype(float) ** -alpha) / hurwitz_zeta(alpha, xmin)


def _pmf_exp(x: np.ndarray, lam: float, xmin: int) -> np.ndarray:
    return lam * np.exp(-lam * (x - xmin))


def log_binned_histogram(x: np.ndarray, n_bins: int = 24):
    """Log-spaced histogram of a positive integer-valued sample.

    Returns bin-center sizes and the normalized probability density in
    each bin -- the standard way to visualize a heavy-tailed distribution
    on log-log axes without the small-sample noise of a linear-bin tail.
    """
    x = np.asarray(x)
    x = x[x > 0]
    bins = np.logspace(0, np.log10(x.max()), n_bins)
    hist, edges = np.histogram(x, bins=bins, density=True)
    centers = np.sqrt(edges[:-1] * edges[1:])
    mask = hist > 0
    return centers[mask], hist[mask]
