"""Segment-length selection and sensitivity analysis.

``segment_length`` computes, for every candidate length L (segments from ``segment_fn(L, 0.0)``):

* ``cv_deviance``: leave-one-day-out Poisson deviance. The rate per segment and hour (obs per covered
  hour) is fitted on the other days, spread uniformly over the segment, and scored against the held-out
  day's counts on a fine grid (``fine_m``, default 5 m) -- so every L is scored on the same data.
  ``cv_se`` is the standard error of the paired per-day difference to the best L.
* ``ss_cost``: Shimazaki-Shinomoto histogram cost (2 k̄ - v) / (n Δ)^2 on day-band counts.
* ``reliability_excess``: split-half reliability of the **time lost per vehicle** (``excess_per_passage``
  per metre, day band 6-21 h, i.e. the day profile along the line): the days are split at random into two
  halves stratified by weekday (``n_splits`` splits), Pearson correlation of the two halves' profiles,
  Spearman-Brown corrected (2r / (1 + r)), averaged over splits and directions (directions with >= 4 days).
  It is the quantity the map shows, and it grows with L up to an elbow (30 m on the 2025-2026 STIB tram
  and bus lines), so it is the criterion that selects.
* ``reliability``: the same split-half reliability on the hour (6-20) x segment matrix of obs per covered
  hour per metre. Diagnostic only: the stop / between-stop structure dominates that matrix, so it is
  saturated (0.95-1.00 at every L) and separates nothing.
* ``snr``: signal variance / noise variance of that matrix (noise from the half difference).
* ``loc_spread_m``: median over the top-10 hotspots of the bootstrap SD of the peak position.
* ``jaccard``: split-half overlap of hotspot footprints (±1 segment tolerance). Diagnostic: too noisy and
  too dependent on the detection thresholds to gate.
* ``cv_within_1se``: CV deviance within one standard error of the minimum. Diagnostic: the deviance keeps
  falling as L shrinks, so the 1-SE rule always picks the smallest L.
* ``link_total_dev``: max relative deviation of per-link observation totals vs the first L (0 when the
  segmentation partitions every link).

Rule: the smallest L with ``reliability_excess`` >= ``min_reliability`` (0.8) and localisation spread
<= ``max_spread_m`` (30 m, or no hotspot). When no length passes, the default ``params.segment_m``
(30 m) is kept, not the CV minimum (which would pick the finest grid exactly when nothing is stable).

``sensitivity`` re-runs the analysis over phase offsets, gap caps (needs ``coverage_fn``), stop zones
(incl. one estimated from the stop-aligned occupancy profile), references and passage sources, and
compares each variant with the baseline (Spearman on a 5 m profile, hotspot Jaccard, link total change,
share of the baseline hotspots found again), and over other segment lengths (``lengths``).

``sensitivity_table`` runs the suite over several configurations (lines) and returns one tidy table
(line, mode, param, value, spearman, jaccard, link_total_change, hotspots_stable_share, stable) plus a
markdown summary.
"""
from __future__ import annotations

from .common import SegmentFn, SensitivityResult, TuneResult
from .length import segment_length
from .sensitivity import sensitivity, stop_zone_from_profile
from .table import SENSITIVITY_COLUMNS, sensitivity_table

__all__ = ["SENSITIVITY_COLUMNS", "SegmentFn", "SensitivityResult", "TuneResult", "segment_length", "sensitivity",
           "sensitivity_table", "stop_zone_from_profile"]
