"""Shared statistics: the day bootstrap, its percentile intervals and one-sided tests, Benjamini-Hochberg,
rank correlation."""
from __future__ import annotations

import math
import warnings

import numpy as np
import polars as pl


def bootstrap_weights(dow: np.ndarray, B: int, rng: np.random.Generator, stratified: bool = True) -> np.ndarray:
    """(B, D) day weights of a Rao-Wu rescaled day bootstrap (within each weekday when ``stratified``):
    n_h - 1 days drawn with replacement in a stratum of n_h days, multiplicities scaled by
    n_h / (n_h - 1). This removes the (n_h - 1) / n_h variance shrinkage of the naive bootstrap,
    which matters with few days per weekday. A stratum with one day keeps weight 1."""
    D = len(dow)
    W = np.zeros((B, D), np.float32)
    groups = [np.flatnonzero(dow == w) for w in np.unique(dow)] if stratified else [np.arange(D)]
    rows = np.arange(B)[:, None]
    for g in groups:
        n = len(g)
        if n == 0:
            continue
        if n == 1:
            W[:, g] = 1.0
            continue
        pick = g[rng.integers(0, n, (B, n - 1))]
        np.add.at(W, (np.broadcast_to(rows, pick.shape), pick), n / (n - 1))
    return W


def bh(p: np.ndarray, q: float) -> np.ndarray:
    """Benjamini-Hochberg rejections at FDR ``q`` (NaN p-values never rejected)."""
    flat = p.reshape(-1)
    ok = np.isfinite(flat)
    out = np.zeros(flat.shape, bool)
    pv = flat[ok]
    m = len(pv)
    if m == 0:
        return out.reshape(p.shape)
    order = np.argsort(pv)
    passed = pv[order] <= q * np.arange(1, m + 1) / m
    if passed.any():
        kmax = np.flatnonzero(passed).max()
        sel = np.zeros(m, bool)
        sel[order[: kmax + 1]] = True
        out[np.flatnonzero(ok)[sel]] = True
    return out.reshape(p.shape)


_erfc = np.frompyfunc(math.erfc, 1, 1)


def nanquantile0(a: np.ndarray, qs) -> np.ndarray:
    """``np.nanquantile(a, qs, axis=0)`` (linear interpolation), vectorised: one sort along axis 0
    instead of numpy's per-column loop (tens of times faster on (B, H, K) bootstrap draws)."""
    a = np.asarray(a)
    qs = np.atleast_1d(np.asarray(qs, float))
    srt = np.sort(a, axis=0)                          # NaN last
    n = np.isfinite(a).sum(0)
    out = np.full((len(qs),) + a.shape[1:], np.nan, dtype=np.float64)
    for i, q in enumerate(qs):
        pos = (n - 1) * q
        lo = np.floor(pos).astype(np.int64)
        hi = np.minimum(lo + 1, np.maximum(n - 1, 0))
        lo = np.clip(lo, 0, None)
        vlo = np.take_along_axis(srt, lo[None], 0)[0].astype(np.float64)
        vhi = np.take_along_axis(srt, hi[None], 0)[0].astype(np.float64)
        v = vlo + (pos - np.floor(pos)) * (vhi - vlo)
        out[i] = np.where(n > 0, v, np.nan)
    return out


def boot_test(x: np.ndarray, draws: np.ndarray, theta: float, alpha: float):
    """Percentile interval ``(lo, hi)`` at level ``1 - alpha`` of bootstrap ``draws`` (axis 0), their
    standard error ``se`` and the one-sided normal p-value of ``x > theta`` (no spread: 0 or 1)."""
    with np.errstate(all="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        lo, hi = nanquantile0(draws, [alpha / 2, 1 - alpha / 2])
        se = np.nanstd(draws, axis=0, ddof=1)
        z = (x - theta) / se
        z = np.where(se > 0, z, np.where(x > theta, np.inf, -np.inf))
        p = 0.5 * _erfc(z / math.sqrt(2)).astype(float)
    return lo, hi, se, p


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    """Spearman rank correlation over the pairs where both are finite (NaN under 3 pairs or no spread)."""
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 3:
        return float("nan")
    ra = pl.Series(a[ok]).rank("average").to_numpy()
    rb = pl.Series(b[ok]).rank("average").to_numpy()
    if ra.std() == 0 or rb.std() == 0:
        return float("nan")
    return float(np.corrcoef(ra, rb)[0, 1])
