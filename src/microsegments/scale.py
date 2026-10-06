"""The one fixed colour scale shared by the HTML page (``html/template.html``, constant ``SC``) and the
matplotlib figures (``plot.py``).

Colour is a fixed function of the value: nothing is computed from the data (no percentile), so every
line, period, weekday, hour, band, stop-zone toggle and unit toggle uses the same colours.

Every measure is first brought to a common quantity, ``T`` = seconds per passage for 30 m of track:

* observations per passage: ``T = opp * tick_s * 30 / len_m`` (time each vehicle spends, per 30 m);
* excess over the evening / over the line: ``T = BASE_S + max(0, excess * tick_s * 30 / len_m)``, the
  extra time added to a fluid 30 m (``BASE_S``), so both excesses share one scale and one legend;
* observations per hour: ``T = oph * 30 / len_m * OPH_S`` (``OPH_S`` s per passage per observation per
  hour, i.e. 1 observation per hour ~ 2 s per passage at 10 passages an hour);
* ``obs_per_h_10m`` (per 10 m, the Python API only; the page always shows 30 m) shares the colours of
  ``obs_per_h``.

``T`` is read on the STIB speed ramp (the green -> red km/h scale of the STIB speed pages): 30 m in
``T`` s is ``108 / T`` km/h, so 4 s = 27 km/h (dark green), 7.2 s = 15 km/h, 14.4 s = 7.5 km/h (red),
>= 24 s = 4.5 km/h (dark red). The legend bar runs from ``T_MIN`` to ``T_MAX`` on a LINEAR axis in ``T``
(ticks every 3 s), with the speed ``108 / T`` km/h under each tick; the colour is still the km/h ramp read
at ``108 / T``, so only the bar's axis is linear, not the colours.

Why these constants (weekdays, 30 m segments, 6-21 h, trams 4 7 25 55 81 92 and buses 48 71 95, March
2025; tram 55 Feb-May 2025): between stops the median time per passage is 3.0-4.4 s per 30 m (green),
p90 6.5-9.9 s (light green to orange), p98 12-23 s (red); in stop zones p98 is 33-46 s (saturated dark
red). Positive excess over the evening: median ~1 s, p90 2-7 s, p98 5-17 s; over the line, between
stops: p90 5-11 s, p98 12-23 s; so with ``BASE_S`` = 3 s typical values stay green, and hotspots
(10 s and more per 30 m) are red. Observations per hour per 30 m: between stops median ~2 (4 s, green),
p90 3-6 (6-12 s), stop zones p98 13-25 (26-50 s, dark red). Metro positions only come at stations, so metro
lines saturate at stations whatever the scale.

Comparison of two periods: a fixed symmetric diverging scale, green (faster) - grey - red (slower),
``CMP_S`` = +-10 s per passage per 30 m. The colour eases in with the square root of the difference (so a
few seconds already show), and the legend bar's axis is linear (ticks every 5 s).
"""
from __future__ import annotations

import numpy as np

L_M = 30.0                 # every value is brought to 30 m of track
T_MIN, T_MAX = 3.0, 24.0   # legend bar: <= 3 s (>= 36 km/h) to >= 24 s (<= 4.5 km/h) per passage per 30 m
T_TICKS = (3, 6, 9, 12, 15, 18, 21, 24)   # linear axis
BASE_S = 3.0               # excesses: T = BASE_S + excess (s per passage per 30 m)
OPH_S = 2.0                # observations per hour: T = OPH_S * obs per hour per 30 m
CMP_S = 10.0               # comparison: +-10 s per passage per 30 m
CMP_TICKS = (-10, -5, 0, 5, 10)            # linear axis
KMH_PER_T = 108.0          # 30 m in T s = 108 / T km/h

# the STIB speed ramp (km/h, colour), interpolated in OKLab
RAMP = ((4.5, "#7a1020"), (7.5, "#e5484d"), (10.5, "#f0a13a"), (13.5, "#b8e27f"), (16.5, "#3fb950"),
        (21.0, "#1f8f45"), (27.0, "#11683e"))
# comparison: faster (green) - grey - slower (red)
DIVERGING = ((-1.0, "#11683e"), (-0.45, "#3fb950"), (0.0, "#4a5470"), (0.45, "#f0a13a"), (1.0, "#e5484d"))

CONSTANTS = {"L": L_M, "T0": T_MIN, "T1": T_MAX, "TICKS": list(T_TICKS), "BASE": BASE_S, "OPH": OPH_S,
             "CMP": CMP_S, "CTICKS": list(CMP_TICKS), "K": KMH_PER_T}


def to_seconds(metric: str, value, len_m, tick_s: float = 20.0):
    """``T`` (seconds per passage per 30 m) of a value of ``metric`` in a segment of ``len_m`` metres.
    ``metric``: "obs_per_passage", "excess_per_passage", "excess_line_per_passage", "obs_per_h",
    "obs_per_h_10m" (already per 10 m), "excess_obs_per_h", "log2_ratio" (ratio applied to ``BASE_S``)."""
    v = np.asarray(value, dtype=float)
    k = L_M / np.asarray(len_m, dtype=float)
    if metric == "obs_per_passage":
        return v * tick_s * k
    if metric in ("excess_per_passage", "excess_line_per_passage"):
        return BASE_S + np.clip(v * tick_s * k, 0, None)
    if metric == "obs_per_h":
        return v * k * OPH_S
    if metric == "obs_per_h_10m":
        return v * (L_M / 10.0) * OPH_S
    if metric == "excess_obs_per_h":
        return BASE_S + np.clip(v * k * OPH_S, 0, None)
    if metric == "log2_ratio":
        return BASE_S * 2.0 ** np.clip(v, 0, None)
    raise ValueError(f"no fixed scale for metric {metric!r}")


def from_seconds(metric: str, T):
    """Inverse of :func:`to_seconds` per 30 m of track (for ``obs_per_h_10m``: per 10 m), i.e. the value of
    ``metric`` shown at ``T`` on the legend (excesses: the positive part)."""
    T = np.asarray(T, dtype=float)
    if metric == "obs_per_passage":
        return T
    if metric in ("excess_per_passage", "excess_line_per_passage"):
        return T - BASE_S
    if metric == "obs_per_h":
        return T / OPH_S
    if metric == "obs_per_h_10m":
        return T / OPH_S / (L_M / 10.0)
    if metric == "excess_obs_per_h":
        return (T - BASE_S) / OPH_S
    if metric == "log2_ratio":
        return np.log2(T / BASE_S)
    raise ValueError(f"no fixed scale for metric {metric!r}")


def position(T):
    """Position in [0, 1] of ``T`` on the legend bar (linear axis from ``T_MIN`` to ``T_MAX``)."""
    T = np.clip(np.asarray(T, dtype=float), T_MIN, T_MAX)
    return (T - T_MIN) / (T_MAX - T_MIN)


def cmp_position(d):
    """Position in [0, 1] of a difference ``d`` (s per passage per 30 m) on the comparison legend bar
    (linear axis from ``-CMP_S`` to ``+CMP_S``)."""
    d = np.clip(np.asarray(d, dtype=float), -CMP_S, CMP_S)
    return (d + CMP_S) / (2 * CMP_S)


def kmh(T):
    """Equivalent speed (km/h) of 30 m in ``T`` seconds: the abscissa of the STIB ramp."""
    return KMH_PER_T / np.maximum(np.asarray(T, dtype=float), 1e-9)


__all__ = ["BASE_S", "CMP_S", "CMP_TICKS", "CONSTANTS", "DIVERGING", "L_M", "OPH_S", "RAMP", "T_MAX", "T_MIN",
           "T_TICKS", "cmp_position", "from_seconds", "kmh", "position", "to_seconds"]
