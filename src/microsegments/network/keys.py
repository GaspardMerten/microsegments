"""Stable physical ids for inter-stop links across GTFS versions.

``link_key = f"{from_stop}>{to_stop}~{h}"`` where ``h`` is a 6-hex hash of the link geometry as it
was first seen. A link met again (same stop pair) reuses an existing key when it is physically the
same path: length difference within ``max(5 %, 20 m)`` and Hausdorff distance within 20 m. A link
rerouted beyond that gets a new key. Because ``h`` is derived from the geometry, an identical
re-issued feed produces the same keys even without a registry; the registry (persisted with
:meth:`KeyRegistry.to_parquet`) keeps keys stable when geometry jitters slightly between versions.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import polars as pl

from .geometry import as_coords, hausdorff_m, length

REGISTRY_SCHEMA = {
    "link_key": pl.Utf8,
    "from_stop": pl.Utf8,
    "to_stop": pl.Utf8,
    "len_m": pl.Float64,
    "geometry": pl.List(pl.List(pl.Float64)),
}


def geometry_hash(coords, n: int = 6, salt: str = "") -> str:
    c = np.round(as_coords(coords), 5)
    return hashlib.sha1((salt + ";".join(f"{x:.5f},{y:.5f}" for x, y in c)).encode()).hexdigest()[:n]


class KeyRegistry:
    """link_key registry. ``key_for`` returns an existing key or registers a new one."""

    def __init__(self, table: pl.DataFrame | None = None, *, len_tol_rel: float = 0.05,
                 len_tol_m: float = 20.0, hausdorff_tol_m: float = 20.0):
        self.len_tol_rel, self.len_tol_m, self.hausdorff_tol_m = len_tol_rel, len_tol_m, hausdorff_tol_m
        self._rows: list[dict] = []
        self._by_pair: dict[tuple[str, str], list[int]] = {}
        self._keys: set[str] = set()
        if table is not None:
            for r in table.select(list(REGISTRY_SCHEMA)).iter_rows(named=True):
                self._add(r)

    def _add(self, r: dict) -> None:
        self._by_pair.setdefault((r["from_stop"], r["to_stop"]), []).append(len(self._rows))
        self._rows.append(r)
        self._keys.add(r["link_key"])

    def __len__(self) -> int:
        return len(self._rows)

    def __contains__(self, key: str) -> bool:
        return key in self._keys

    def same_link(self, len_a: float, coords_a, len_b: float, coords_b) -> tuple[bool, float]:
        """(physically the same path?, Hausdorff metres)."""
        if abs(len_a - len_b) > max(self.len_tol_rel * max(len_a, len_b), self.len_tol_m):
            return False, float("inf")
        h = hausdorff_m(coords_a, coords_b)
        return h <= self.hausdorff_tol_m, h

    def key_for(self, from_stop: str, to_stop: str, coords, len_m: float | None = None) -> str:
        coords = as_coords(coords)
        len_m = length(coords) if len_m is None else float(len_m)
        best, best_h = None, float("inf")
        for i in self._by_pair.get((from_stop, to_stop), []):
            r = self._rows[i]
            ok, h = self.same_link(len_m, coords, r["len_m"], r["geometry"])
            if ok and h < best_h:
                best, best_h = r["link_key"], h
        if best is not None:
            return best
        salt, n = "", 0
        while True:
            key = f"{from_stop}>{to_stop}~{geometry_hash(coords, salt=salt)}"
            if key not in self._keys:
                break
            n += 1
            salt = f"{n}:"
        self._add({"link_key": key, "from_stop": from_stop, "to_stop": to_stop, "len_m": len_m,
                   "geometry": [[float(x), float(y)] for x, y in coords]})
        return key

    @property
    def table(self) -> pl.DataFrame:
        return pl.DataFrame(self._rows, schema=REGISTRY_SCHEMA)

    def to_parquet(self, path: str | Path) -> None:
        self.table.write_parquet(path)

    @classmethod
    def from_parquet(cls, path: str | Path, **kw) -> KeyRegistry:
        p = Path(path)
        return cls(pl.read_parquet(p) if p.exists() else None, **kw)
