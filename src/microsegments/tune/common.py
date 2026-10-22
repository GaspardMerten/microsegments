"""Shared parts of the tuning: result types, weekday-stratified half splits, tolerant overlaps."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import polars as pl

from ..hotspots import BinStats

SegmentFn = Callable[..., pl.DataFrame]


@dataclass
class TuneResult:
    table: pl.DataFrame                  # one row per L: the criteria
    recommended: float
    rule: str
    per_day_deviance: dict[float, np.ndarray] = field(repr=False, default_factory=dict)


@dataclass
class SensitivityResult:
    table: pl.DataFrame                  # tidy: variant, param, value, x_m, obs_per_h_m (day band profile)
    summary: pl.DataFrame                # one row per variant vs baseline
    hotspots: pl.DataFrame               # baseline top hotspots: boot share, phase count, stable
    stable: bool
    baseline_hotspots: pl.DataFrame = field(repr=False, default_factory=pl.DataFrame)
    n_days: int = 0                      # included days of the baseline analysis


def _split_weights(dow: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """(2, D) indicator weights of a weekday-stratified half split."""
    W = np.zeros((2, len(dow)), np.float32)
    for w in np.unique(dow):
        g = rng.permutation(np.flatnonzero(dow == w))
        h = len(g) // 2 + (rng.random() < 0.5 if len(g) % 2 else 0)
        W[0, g[:h]] = 1
        W[1, g[h:]] = 1
    return W


def _peak_cols(st: BinStats, j0: int, j1: int, hours) -> int:
    hcols = np.flatnonzero(st.cols >= 0)
    hrs = st.cols[hcols]
    hidx = np.array([hcols[np.flatnonzero(hrs == h)[0]] for h in sorted(hours)])
    sx = np.nansum(st.x[hidx][:, j0:j1 + 1], axis=1)
    return int(hidx[int(np.nanargmax(sx))])


def _tol_jaccard(A: np.ndarray, B: np.ndarray, tol: int = 1) -> float:
    """Jaccard-like overlap of two boolean masks with a ±tol tolerance."""
    if not A.any() and not B.any():
        return 1.0
    if not A.any() or not B.any():
        return 0.0
    def dil(M):
        out = M.copy()
        for t in range(1, tol + 1):
            out[t:] |= M[:-t]
            out[:-t] |= M[t:]
        return out
    return float(((A & dil(B)).sum() + (B & dil(A)).sum()) / (A.sum() + B.sum()))
