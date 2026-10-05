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
               "days": [dow][link]    int   included days where the link is valid}
      }],
      "hotspots": [HOTSPOT rows as dicts],
      "ctx": [[[lon, lat], ...]]                  # optional background polylines
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
