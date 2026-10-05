"""JSON exchanged with the HTML page (``html.export``) and the platform (``GET /api/analysis``).

Totals are kept per weekday so the page can recombine any set of weekdays client-side
(sum the selected rows, then divide). Nothing is pre-divided except geometry.

    {
      "v": 1,
      "line": "55", "route_id": "...", "mode": "tram",
      "params": {segment_m, phase_m, grid, tick_s, gap_cap_s, stop_zone: [up, down], reference_hours: [a, b]},
      "period": {"first": "2025-02-17", "last": "2025-05-16",
                 "days": 60,                      # included service days
                 "per_dow": [11, 12, 13, 12, 12, 0, 0],
                 "excluded": [{"date": "2025-03-04", "reason": "low_coverage"}]},
      "hours": [5, ..., 23],                      # hour index h used below
      "coverage": {"dates": ["2025-02-17", ...], "c": [[0.98, ...] per date][per hour]},
      "versions": [{"dir": 0, "pattern_uid": "...", "first": "...", "last": "...", "n_days": 54,
                    "links_added": [...], "links_removed": [...], "display": true}],
      "dirs": [{
        "dir": 0, "pattern_uid": "...", "from": "Da Vinci", "to": "Rogier",
        "stops": [{"id": "...", "name": "...", "ll": [lon, lat], "x": 0.0}],
        "links": [{"key": "...", "from": "...", "to": "...", "len": 252.7, "n_days": 60, "flags": 0}],
        "seg": {"key": [...], "link": [...], "x0": [...], "len": [...], "zone": ["stop"|"running"],
                "c": [[[lon, lat], ...] per segment], "flags": [...]},
        "wk": {"obs":  [dow][h][seg]  int   observation counts,
               "cov":  [dow][h][link] float covered hours (sum of coverage over included days where the link is valid),
               "p":    [dow][h][link] float passages,
               "days": [dow][link]    int   included days where the link is valid,
               "n":    [dow][h][link] int   included day-hours (additive; p / n = passages per hour)},
        "ref": {"evening": [seg], "line": [seg], "line_level_per_m": m}   # additive, obs per passage
      }],
      "hotspots": [HOTSPOT rows as dicts],
      "reference_agreement": [{"direction_id", "line_level_per_m", "evening_ref_per_m", "spearman_day",
                               "spearman_day_all", "top_overlap", "n_hotspots", "n_peak", "n_persistent",
                               "n_both", "hotspots_both_share"}],          # additive
      "ctx": [[[lon, lat], ...]],                 # optional background polylines
      "compare": {...}                            # optional, two-period page only (below)
    }

Two-period page (``microsegments compare``, ``compare.Comparison.to_contract``): the document above
is period B's, plus an additive ``compare`` section (the page then opens in "compare" mode)::

    "compare": {
      "a": {"first", "last", "days", "per_dow", "excluded"},   # period A, before
      "b": {...},                                              # period B, after
      "params": {B, seed, theta_s, theta_bin_s, q, level, freq_change, gap, tick_s, status},
      "cols": [5, ..., 23, "am", "pm", "day", "evening"],     # column labels of the matrices below
      "dirs": [{"dir", "pattern_uid",                          # B's display pattern
                "seg_key": [seg], "matched": [0|1],            # segments without match: null values
                "only_a": n, "only_b": n,                      # unmatched segment counts
                "oa", "ob": [col][seg] obs per passage A / B,
                "xa", "xb": [col][seg] excess over each period's own evening reference,
                "d": [col][seg] delta = ob - oa (obs per vehicle; x tick_s ≈ s), "lo", "hi": its CI,
                "sig": [col][seg] -1 | 0 | 1 (significant decrease / none / increase),
                "fa", "fb": [col][seg] passages per hour A / B}],
      "changes": [compare.STRETCH_SCHEMA rows as dicts, + "dir"]   # ranked per direction
    }

Client-side metrics for a weekday set W and hour h, segment s in link l:
    obs_per_h       = sum_W obs / sum_W cov
    obs_per_passage = sum_W obs / sum_W p
    ref             = sum_{W, h in ref} obs / sum_{W, h in ref} p
    excess          = obs_per_passage - ref
"""
from __future__ import annotations

CONTRACT_VERSION = 1
DOW_FR = ["Lu", "Ma", "Me", "Je", "Ve", "Sa", "Di"]
