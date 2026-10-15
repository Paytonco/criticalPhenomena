"""
Power-spectral-density tools for detecting 1/f^beta ("colored") noise in a
time series of system activity -- the temporal fingerprint of
self-organized criticality, complementing the spatial power-law in event
sizes.

Shared across every critical-system demo in this repo.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import signal


@dataclass
class SpectralFit:
    beta: float           # PSD(freq) ~ freq^-beta
    intercept: float
    freqs: np.ndarray     # frequencies used in the fit
    psd: np.ndarray       # PSD values used in the fit


NOISE_COLORS = [
    (-0.3, 0.3, "white noise (beta ~ 0): no memory, uncorrelated at every scale"),
    (0.3, 1.4, "pink / 1/f noise (beta ~ 1): correlated at every scale, equally"),
    (1.4, 2.3, "red / brown noise (beta ~ 2): a random walk, strongly autocorrelated"),
]


def classify_noise_color(beta: float) -> str:
    """Label a fitted spectral exponent.

    Pink (beta=1) and brown/red (beta=2) are the two most commonly cited
    landmarks, but any beta clearly above 0 and below ~3 belongs to the
    same qualitative family: a scale-free, "colored" spectrum with no
    single characteristic frequency -- as opposed to white noise, which
    has no correlation structure at all. Don't over-read small
    differences in the fitted exponent; which exact power of f a given
    finite, finite-size-cutoff simulation lands on is sensitive to the
    frequency window used to fit it, which is why `fit_spectral_slope`
    reports the window explicitly.
    """
    for lo, hi, label in NOISE_COLORS:
        if lo <= beta <= hi:
            return label
    if beta > 2.3:
        return f"beta = {beta:.2f}: steeper than brown noise, still in the broad 1/f^beta colored-noise family"
    return f"beta = {beta:.2f} (outside the usual white/pink/brown bands)"


def welch_psd(x: np.ndarray, fs: float = 1.0, nperseg: int | None = None):
    """Welch's method power spectral density of a 1-D time series."""
    x = np.asarray(x, dtype=float)
    x = x - x.mean()
    if nperseg is None:
        nperseg = min(16384, len(x) // 4)
    freqs, psd = signal.welch(x, fs=fs, nperseg=nperseg, detrend="constant")
    return freqs, psd


def log_bin_spectrum(freqs: np.ndarray, psd: np.ndarray, n_bins: int = 40):
    """Geometric-mean log-frequency binning of a PSD.

    `scipy.signal.welch` returns *linearly* spaced frequencies, so a
    naive least-squares fit of log(PSD) vs log(freq) implicitly gives the
    high-frequency decades far more sample points -- and therefore far
    more weight -- than the low-frequency ones. Binning geometrically in
    log-frequency first (the same idea as `powerlaw.log_binned_histogram`)
    gives every decade equal say in the fit.
    """
    edges = np.logspace(np.log10(freqs[0]), np.log10(freqs[-1]), n_bins + 1)
    idx = np.digitize(freqs, edges)
    fc, pc = [], []
    for i in range(1, n_bins + 1):
        sel = idx == i
        if sel.any():
            fc.append(np.exp(np.log(freqs[sel]).mean()))
            pc.append(np.exp(np.log(psd[sel]).mean()))
    return np.array(fc), np.array(pc)


def fit_spectral_slope(freqs: np.ndarray, psd: np.ndarray,
                        fmin: float | None = None, fmax: float | None = None,
                        n_bins: int = 40) -> SpectralFit:
    """Least-squares slope of log(PSD) vs log(freq) over a frequency band.

    Excludes freq=0 always; `fmin`/`fmax` additionally restrict the fit to
    the frequency decades over which the scaling actually looks linear --
    real 1/f-type signals are always bounded above by some microscopic
    cutoff (here, the single-step discreteness of the automaton) and
    below by the longest driving/correlation time scale in the system, so
    excluding both ends is standard practice, not cherry-picking. The
    regression itself runs on log-frequency-binned points (see
    `log_bin_spectrum`) so the fit isn't dominated by the many raw Welch
    bins at high frequency.
    """
    mask = freqs > 0
    if fmin is not None:
        mask &= freqs >= fmin
    if fmax is not None:
        mask &= freqs <= fmax
    f, p = freqs[mask], psd[mask]
    fb, pb = log_bin_spectrum(f, p, n_bins=n_bins)
    logf, logp = np.log10(fb), np.log10(pb)
    A = np.vstack([logf, np.ones_like(logf)]).T
    slope, intercept = np.linalg.lstsq(A, logp, rcond=None)[0]
    return SpectralFit(beta=float(-slope), intercept=float(intercept), freqs=f, psd=p)
