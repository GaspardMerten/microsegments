"""Observations -> Placed: where along which pattern each position fix lies.

* :func:`place_linear` (``linear.py``): "last stop passed + metres since" feeds (STIB vehicle-distance);
* :func:`place_latlon` (``mapmatch.py``): lat/lon fixes (GTFS-RT VehiclePositions, AVL exports);
* :func:`place`: dispatch on ``cfg.input.kind``; anonymous linear feeds are then tracked
  (:func:`microsegments.tracks.track_anonymous`) so rows carry a ``track_id`` and short-working
  standing starts are turned into layover, as in the prototype.
"""
from __future__ import annotations

import polars as pl

from ..config import Config
from .common import Placed
from .linear import place_linear
from .mapmatch import place_latlon


def place(obs, net, cfg: Config | None = None, *, track: bool = True, compat: bool = False) -> Placed:
    """Place ``obs`` (OBSERVATION) on the network according to ``cfg`` (input kind, params, ids).

    ``track``: track anonymous linear feeds (fills ``track_id``, short-working standing starts become
    layover). ``compat``: the prototype's exact rules (no stop-only placement, prototype tracker, see
    :func:`microsegments.locate.linear.place_linear` and :func:`microsegments.tracks.track_anonymous`)."""
    cfg = cfg or Config()
    kind = cfg.input.kind
    if kind in ("linear", "stib"):
        lin = cfg.input.linear
        res = place_linear(obs, net, cfg.params, feed_length=lin.feed_length,
                           terminus_aliases=lin.terminus_aliases or None, tz=cfg.input.timezone,
                           service_day_start=cfg.input.service_day_start, id_normaliser=lin.id_normaliser,
                           compat=compat)
        fr = res.frame
        if track and fr.height and fr["track_id"].null_count() == fr.height:
            from ..tracks import track_anonymous
            before = int(fr["count"].sum())
            fr = track_anonymous(fr, compat=compat)
            res.report["layover_short_working_start"] = before - int(fr["count"].sum())
            res.report["counted"] = int(fr["count"].sum())
            res.frame = fr
        return res
    if kind == "latlon":
        return place_latlon(obs, net, cfg.params, tz=cfg.input.timezone, service_day_start=cfg.input.service_day_start)
    raise ValueError(f"unknown input kind {kind!r}")


__all__ = ["Placed", "place", "place_linear", "place_latlon"]
