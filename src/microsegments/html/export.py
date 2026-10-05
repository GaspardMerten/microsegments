"""Standalone HTML report: ``template.html`` filled with the JSON contract (``report.to_contract``).

The page needs Leaflet JS from cdnjs (and, if ``tiles``, CARTO base-map tiles) for the map; the
matrix, profile, hotspots and coverage calendar work offline. Leaflet's CSS is inlined.
"""
from __future__ import annotations

import html
import warnings
import json
import re
from importlib import resources
from pathlib import Path
from typing import Any

PLACEHOLDERS = ("__LANG__", "__TITLE__", "__DESC__", "__LEAFLETCSS__", "__DATA__")


def _resource(name: str) -> str:
    return resources.files("microsegments.html").joinpath(name).read_text(encoding="utf-8")


def _contract(obj, **kw) -> dict[str, Any]:
    if isinstance(obj, dict):
        return obj
    from ..metrics import Analysis
    from ..report import to_contract
    if isinstance(obj, Analysis):
        return to_contract(obj, **kw)
    if hasattr(obj, "contract"):            # pipeline.RunResult
        return obj.contract(**kw)
    raise TypeError(f"expected an Analysis, a RunResult or a contract dict, got {type(obj).__name__}")


def _json_for_script(data: dict) -> str:
    s = json.dumps(data, separators=(",", ":"), ensure_ascii=False, allow_nan=False, default=str)
    # safe inside <script type="application/json">
    return s.replace("<", "\\u003c").replace(">", "\\u003e").replace("&", "\\u0026") \
        .replace("\u2028", "\\u2028").replace("\u2029", "\\u2029")


def _default_title(c: dict, lang: str) -> str:
    line = c.get("line") or ""
    d = (c.get("dirs") or [{}])[0]
    ends = f"{d.get('from')} ↔ {d.get('to')}" if d.get("from") else ""
    if lang == "en":
        return f"Line {line}, observations per micro-segment" + (f" ({ends})" if ends else "")
    return f"Ligne {line}, observations par microsegment" + (f" ({ends})" if ends else "")


def render(analysis_or_contract, *, title: str | None = None, lang: str = "fr", tiles: bool = True,
           scales: dict | None = None, description: str | None = None, show_hotspots: bool = True, **contract_kw) -> str:
    """The full HTML document as a string. ``contract_kw`` go to ``to_contract`` when an Analysis /
    RunResult is given. The colour scale is fixed for every page (``microsegments.scale``); ``scales`` is
    deprecated and ignored. ``show_hotspots=False`` hides every hotspot element
    (list, map markers and outlines, tile, method paragraph), and the retained changes of the two-period view; the page reads it from the contract key
    ``show_hotspots``, so a page fed with its data later can set that key instead."""
    c = dict(_contract(analysis_or_contract, **contract_kw))
    c["tiles"] = bool(tiles)
    if not show_hotspots:
        c["show_hotspots"] = False
    if scales:
        warnings.warn("render(scales=...) is ignored: the colour scale is fixed (microsegments.scale)",
                      DeprecationWarning, stacklevel=2)
    c.pop("scales", None)
    if title:
        c["title"] = title
    page_title = c.get("title") or _default_title(c, lang)
    p = c.get("period") or {}
    desc = description or (
        f"{page_title}. {p.get('first', '')} – {p.get('last', '')}, {p.get('days', 0)} "
        + ("days." if lang == "en" else "jours."))
    out = _resource("template.html")
    out = out.replace("__LANG__", "en" if lang == "en" else "fr")
    out = out.replace("__TITLE__", html.escape(page_title))
    out = out.replace("__DESC__", html.escape(desc, quote=True))
    out = out.replace("__LEAFLETCSS__", _resource("leaflet.min.css"))
    out = out.replace("__DATA__", _json_for_script(c))
    left = [m for m in re.findall(r"__[A-Z]+__", out) if m in PLACEHOLDERS]
    if left:  # pragma: no cover - template bug
        raise RuntimeError(f"unfilled placeholders {left}")
    return out


def export(analysis_or_contract, path: str | Path, *, title: str | None = None, lang: str = "fr",
           standalone: bool = True, tiles: bool = True, scales: dict | None = None, show_hotspots: bool = True,
           **contract_kw) -> Path:
    """Write the report to ``path``. ``analysis_or_contract``: an ``Analysis``, a ``RunResult`` or a
    contract dict (``report.to_contract``). ``standalone`` (default): a complete HTML document; False:
    only what goes inside ``<body>`` (head elements included inline), to embed in another page."""
    page = render(analysis_or_contract, title=title, lang=lang, tiles=tiles, scales=scales, show_hotspots=show_hotspots,
                  **contract_kw)
    if not standalone:
        head = re.search(r"<head>(.*)</head>", page, re.DOTALL).group(1)
        body = re.search(r"<body>(.*)</body>", page, re.DOTALL).group(1)
        head = re.sub(r"<meta[^>]*>\n?", "", head)
        page = head + body
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(page, encoding="utf-8")
    return path


__all__ = ["export", "render"]
