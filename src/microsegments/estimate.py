"""The estimator: indicators of one direction (``DirData``) under bootstrap day weights, as ratios of sums
(optionally post-stratified by weekday). See ``metrics`` for the definitions."""
from __future__ import annotations

import warnings

import numpy as np

from .config import Params, Quality
from .dirdata import DirData

ALPHA = 0.5              # log2_ratio smoothing
LINE_HOURS = (6, 23)     # [start, end): window of the line-level reference (same as hotspots.ALLDAY_HOURS)


def _ratio(num: np.ndarray, den: np.ndarray, pi: np.ndarray | None, ok: np.ndarray | None = None):
    """num, den: (S, ...) per stratum when pi is given, else plain arrays. Ratio of sums, or
    post-stratified mean of per-stratum ratios (weights renormalised over strata with data)."""
    with np.errstate(divide="ignore", invalid="ignore"):
        if pi is None:
            r = num / den
            r[~(den > 0)] = np.nan
            if ok is not None:
                r[~ok] = np.nan
            return r
        good = den > 0
        if ok is not None:
            good &= ok
        r = np.where(good, num / np.where(good, den, 1), 0.0)
        w = pi.reshape((-1,) + (1,) * (num.ndim - 1)) * good
        tot = w.sum(0)
        out = (w * r).sum(0) / tot
        out[~(tot > 0)] = np.nan
        return out


def _sums(dd: DirData, W: np.ndarray, pi: np.ndarray | None, coverage: bool):
    """Weighted sums (B, Hx, K) of NM, PM, CM, M over the days; with ``pi`` also per weekday stratum S (S, B, Hx,
    K) and the weights renormalised over the strata present. CM is 0 without ``coverage``."""
    X = dd.stacked()
    D, Hx, K = X["NM"].shape
    W = np.asarray(W, np.float32)
    B = W.shape[0]

    def sums(Wm):
        out = {k: (Wm @ X[k].reshape(D, -1)).reshape(B, Hx, K).astype(np.float64)
               for k in (("NM", "PM", "CM", "M") if coverage else ("NM", "PM", "M"))}
        if not coverage:
            out["CM"] = np.zeros((B, Hx, K))
        return out

    if pi is None:
        return sums(W), None, None
    strata = [w for w in range(7) if (dd.dow == w).any()]
    per = [sums(W * (dd.dow == w)[None, :].astype(np.float32)) for w in strata]
    S = {k: np.stack([p[k] for p in per]) for k in ("NM", "PM", "CM", "M")}
    return {k: S[k].sum(0) for k in S}, S, pi[strata] / pi[strata].sum()


def _reference(dd: DirData, s: dict, S: dict | None, pi, opp: np.ndarray, params: Params) -> np.ndarray:
    """(B, K) reference obs per passage: the evening ratio of sums, or (``freeflow``) q20 over hours 5-23."""
    hc = dd.hour_cols
    hours = dd.cols[hc]
    if params.reference == "freeflow":
        sel = hc[(hours >= 5) & (hours <= 23)]
        with np.errstate(all="ignore"), warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            return np.nanquantile(opp[:, sel, :], 0.2, axis=1)
    a, z = params.reference_hours
    sel = hc[(hours >= a) & (hours < z)]
    if S is None:
        return _ratio(s["NM"][:, sel].sum(1), s["PM"][:, sel].sum(1), None)
    return _ratio(S["NM"][:, :, sel].sum(2), S["PM"][:, :, sel].sum(2), pi)


def _estimate(dd: DirData, W: np.ndarray, params: Params, quality: Quality, pi: np.ndarray | None,
              coverage: bool = True) -> dict[str, np.ndarray]:
    s, S, pi = _sums(dd, W, pi, coverage)
    obs, pas, cov, ncell = s["NM"], s["PM"], s["CM"], s["M"]

    def rate(nk, dk, ok=None):
        return _ratio(s[nk], s[dk], None, ok) if S is None else _ratio(S[nk], S[dk], pi, ok)

    with np.errstate(divide="ignore", invalid="ignore"):
        enough = (pas / np.where(ncell > 0, ncell, np.nan)) >= quality.min_passages_per_day
    oph = rate("NM", "CM")
    opp = rate("NM", "PM", enough if S is None else None)
    if S is not None:
        opp[~enough] = np.nan
    ref = _reference(dd, s, S, pi, opp, params)
    with np.errstate(divide="ignore", invalid="ignore"):
        exc_pp = opp - ref[:, None, :]
        if S is None:
            exc_oph = (obs - ref[:, None, :] * pas) / np.where(cov > 0, cov, np.nan)
        else:
            exc_oph = _ratio(S["NM"] - ref[None, :, None, :] * S["PM"], S["CM"], pi)
        log2 = np.log2((opp + ALPHA) / (ref[:, None, :] + ALPHA))
    hc = dd.hour_cols
    level, ref_line = line_level(obs[:, hc], pas[:, hc], dd.cols[hc], dd.seg_len, dd.seg_zone == "running")
    with np.errstate(divide="ignore", invalid="ignore"):
        exc_line = opp - ref_line[:, None, :]
    return {"obs": obs, "passages": pas, "covered_h": cov, "n_cells": ncell, "obs_per_h": oph,
            "obs_per_passage": opp, "ref": ref, "excess_per_passage": exc_pp, "excess_obs_per_h": exc_oph,
            "log2_ratio": log2, "line_level": level, "ref_line": ref_line, "excess_line_per_passage": exc_line}


def running_level(O: np.ndarray, P: np.ndarray, seg_len: np.ndarray, running: np.ndarray):
    """O, P: (B, K) obs and passages summed over a window. Returns obs / passages (B, K) and the level m (B,) =
    median over running segments of obs / passages / metre (NaN when none has passages)."""
    with np.errstate(divide="ignore", invalid="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        occ = np.where(P > 0, O / np.where(P > 0, P, 1), np.nan)
        return occ, np.nanmedian(np.where(running[None, :], occ / seg_len[None, :], np.nan), axis=1)


def line_level(obs: np.ndarray, pas: np.ndarray, hours: np.ndarray, seg_len: np.ndarray, running: np.ndarray,
               window: tuple[int, int] = LINE_HOURS) -> tuple[np.ndarray, np.ndarray]:
    """Line-level reference. ``obs``, ``pas``: (B, H, K) sums per hour column (``hours`` labels).
    Level m (B,) = median over running segments of the ``window`` hours' obs / passages / metre (the
    typical running occupancy per metre of the direction, the "persistent" hotspot criterion's level);
    reference per segment (B, K) = m x length. NaN when no running segment has passages."""
    sel = (hours >= window[0]) & (hours < window[1])
    B, K = obs.shape[0], obs.shape[-1]
    if not sel.any() or not running.any():
        return np.full(B, np.nan), np.full((B, K), np.nan)
    _, m = running_level(obs[:, sel].sum(1), pas[:, sel].sum(1), seg_len, running)
    return m, m[:, None] * seg_len[None, :]
