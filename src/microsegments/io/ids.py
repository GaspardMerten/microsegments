"""Stop / terminus id normalisation, so feed ids match GTFS ids.

``"leading_digits"`` mirrors the prototype's ``digits()``: the leading digits without their
leading zeros ("0626" -> "626", "5272F" -> "5272", "0" -> "0"); not starting with a digit -> null.
``"none"`` keeps the id as text. Any other string is a regex whose first group is kept.
"""
from __future__ import annotations

import re

import polars as pl


def normalise_expr(e: pl.Expr, normaliser: str = "leading_digits") -> pl.Expr:
    e = e.cast(pl.Utf8).str.strip_chars()
    if normaliser in (None, "", "none"):
        return e
    if normaliser == "leading_digits":
        return e.str.extract(r"^0*(\d+)", 1)
    if re.compile(normaliser).groups < 1:
        raise ValueError(f"id normaliser regex needs one group: {normaliser!r}")
    return e.str.extract(normaliser, 1)


def normalise_py(value, normaliser: str = "leading_digits") -> str | None:
    """Scalar version of ``normalise_expr``."""
    if value is None:
        return None
    s = str(value).strip()
    if normaliser in (None, "", "none"):
        return s
    pat = r"^0*(\d+)" if normaliser == "leading_digits" else normaliser
    m = re.search(pat, s)
    return m.group(1) if m else None
