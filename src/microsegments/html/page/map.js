// ---------------------------------------------------------------- map
let map = null, renderer = null, baseLayer, segLayer, zoneLayer, stopLayer, hotLayer, segs = [];
const mapEl = $('map');
function initMap() {
  if (typeof window.L === 'undefined' || !DIRS.length) { mapEl.innerHTML = `<div class="nomap">${DIRS.length ? tx().nomap : tx().nodir}</div>`; return; }
  const Lf = window.L;
  map = Lf.map('map', {scrollWheelZoom: false, zoomSnap: .25});
  map.attributionControl.setPrefix(false);
  // basemap: OpenStreetMap tiles, darkened by a CSS filter (.osmdark); D.tile_url / D.tile_attr: a host's own tiles
  // (e.g. a keyed dark provider), drawn as they are
  if (D.tiles !== false) Lf.tileLayer(D.tile_url || 'https://tile.openstreetmap.org/{z}/{x}/{y}.png', {maxZoom: 19,
    className: D.tile_url ? '' : 'osmdark', attribution: D.tile_url ? (D.tile_attr || '') : '© <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a>'}).addTo(map);
  if (D.source) map.attributionControl.addAttribution(esc(D.source));
  renderer = Lf.svg({padding: .5});
  for (const c of (D.ctx || [])) Lf.polyline(latlngs(c), {renderer, color: '#1f2b47', weight: 1.5, interactive: false}).addTo(map);
  baseLayer = Lf.layerGroup().addTo(map); hotLayer = Lf.layerGroup().addTo(map); segLayer = Lf.layerGroup().addTo(map); zoneLayer = Lf.layerGroup().addTo(map); stopLayer = Lf.layerGroup().addTo(map);
  const all = DIRS.flatMap(d => [d, ...(d.alt || [])]).flatMap(d => d.seg.c.flatMap(c => latlngs(c)));
  const fit = () => { map.invalidateSize(); if (all.length) map.fitBounds(Lf.latLngBounds(all), {padding: [24, 24]}); };
  fit(); window.addEventListener('load', fit);
  map.on('zoomend', declutter);
}
// Hide stop labels that would overlap an already placed one.
function declutter() {
  if (!map) return;
  const placed = [];
  stopLayer.eachLayer(m => {
    if (!m._lab) return;
    const p = map.latLngToLayerPoint(m.getLatLng()), el = m.getElement && m.getElement();
    if (!el) return;
    const lab = el.querySelector('.lab'); if (!lab) return;
    const ok = !placed.some(q => Math.abs(q.x - p.x) < 70 && Math.abs(q.y - p.y) < 16);
    lab.classList.toggle('hide', !ok);
    if (ok) placed.push(p);
  });
}
// Street View at the clicked point, looking along the direction of travel of the microsegment
function svPopup(c, ll) {
  const [x0, y0] = c[0], [x1, y1] = c[c.length - 1];
  const hd = ((Math.atan2((x1 - x0) * Math.cos(y0 * Math.PI / 180), y1 - y0) * 180 / Math.PI) + 360) % 360;
  const lat = ll.lat.toFixed(6), lng = ll.lng.toFixed(6), h = hd.toFixed(0);
  const link = `https://www.google.com/maps/@?api=1&map_action=pano&viewpoint=${lat},${lng}&heading=${h}`;
  const emb = `https://maps.google.com/maps?layer=c&cbll=${lat},${lng}&cbp=11,${h},0,0,0&output=svembed`;
  window.L.popup({className: 'svpop', maxWidth: 340, minWidth: 320}).setLatLng(ll)
    .setContent(`<iframe src="${emb}" loading="lazy" referrerpolicy="no-referrer-when-downgrade" allowfullscreen></iframe><a href="${link}" target="_blank" rel="noopener">${tx().sv}</a>`)
    .openOn(map);
}
function drawMap() {
  if (!map) return;
  const Lf = window.L, E = entry();
  baseLayer.clearLayers(); segLayer.clearLayers(); zoneLayer.clearLayers(); stopLayer.clearLayers(); segs = [];
  if (!E) return;
  E.seg.c.forEach(c => { if (c.length > 1) Lf.polyline(latlngs(c), {renderer, color: '#05090f', weight: 14, interactive: false}).addTo(baseLayer); });
  // interarrêts: one polyline per unit; a click excludes it from the line totals (or brings it back). Stop zones
  // hidden: the unit is drawn between the zones, the zones thin and hatched (as microsegments).
  if (isIA()) unitsOf(E).U.forEach((u, n) => {
    const ix = runIdx(E, u.idx, !st.stops);
    if (!st.stops) u.idx.filter(i => !ix.includes(i)).forEach(i => Lf.polyline(latlngs(E.seg.c[i]), {renderer, color: MASKC, weight: 4, lineCap: 'butt', interactive: false}).addTo(segLayer));
    if (!ix.length) return segs.push(null);
    const p = Lf.polyline(ix.flatMap(i => latlngs(E.seg.c[i])), {renderer, weight: 10, lineCap: 'butt'}).addTo(segLayer);
    p.bindTooltip(() => { const cm = isCmp(), b = drawn(E, cm ? cvals(E) : vals(E), cm)[n]; return safeTip(() => (cm ? cTipHtml : tipHtml)(E, b, true), E, b); }, {sticky: true});
    p.on('click', e => { toggleXia(u.id); p.closeTooltip(); p.openTooltip(e.latlng); });
    segs.push(p);
  });
  else E.seg.c.forEach((c, i) => {
    const p = Lf.polyline(latlngs(c), {renderer, weight: 10, lineCap: 'butt'}).addTo(segLayer);
    p.bindTooltip(() => { const cm = isCmp(), v = cm ? cvals(E) : vals(E), n = zonesOf(E).zi[i];
      // in a shown stop zone, every microsegment carries the zone's tooltip
      const b = st.stops && n >= 0 ? zvals(E, v, cm)[n] : v[i];
      return safeTip(() => cm ? cTipHtml(E, b) : tipHtml(E, b), E, b); }, {sticky: true});
    p.on('click', e => { p.closeTooltip(); svPopup(c, e.latlng); });
    segs.push(p);
  });
  E.stops.forEach((s, k) => {
    if (!s.ll) return;
    const ll = [s.ll[1], s.ll[0]];
    // label above where the line runs east-west, to the right otherwise
    const nb = E.stops[Math.min(k + 1, E.stops.length - 1)].ll || s.ll, pb = E.stops[Math.max(k - 1, 0)].ll || s.ll;
    const side = Math.abs(nb[0] - pb[0]) * Math.cos(s.ll[1] * Math.PI / 180) > Math.abs(nb[1] - pb[1]) ? 'u' : 'r';
    Lf.circleMarker(ll, {renderer, radius: 4, color: '#0b1224', weight: 2, fillColor: '#9aa4b8', fillOpacity: 1, interactive: false}).addTo(stopLayer);
    const m = Lf.marker(ll, {icon: Lf.divIcon({className: '', html: `<div class="lab ${side}">${esc(s.name)}</div>`, iconSize: [0, 0]}), interactive: false}).addTo(stopLayer);
    m._lab = true;
  });
  drawHotMap();
  setTimeout(declutter, 0);
}
// Halos (hotspots, retained changes) are re-added on every render, and one SVG renderer draws in insertion order:
// push them under the coloured segments, then the dark casing under them, so the colours stay visible.
function haloUnder() { hotLayer.eachLayer(l => l.bringToBack && l.bringToBack()); baseLayer.eachLayer(l => l.bringToBack()); }
function hotSegIdx(E, h) { const out = []; E.seg.x0.forEach((x, i) => { if (x + E.seg.len[i] > h.x0_m + .5 && x < h.x1_m - .5) out.push(i); }); return out; }
function drawHotMap() {
  if (!map) return;
  const E = entry();
  hotLayer.clearLayers();
  if (E && isMain()) {
    if (isCmp()) halos(E, chgDir(), st.chg, selectChg, (h, n) => `<div class="hsm" style="background:${signCol(h.sign)};color:#111">${n + 1}</div>`);
    else halos(E, hotOfDir(), st.hot, selectHot, h => `<div class="hsm">${h.rank}</div>`);
  }
  haloUnder();
}
// yellow halos over listed stretches (hotspots, retained changes; on: the selected rank), a numbered marker (badge) on each
function halos(E, list, on, pick, badge) {
  const Lf = window.L;
  list.forEach((h, n) => {
    const idx = hotSegIdx(E, h); if (!idx.length) return;
    const w = on === h.rank ? {weight: 18, opacity: 1} : {weight: 15, opacity: .8};
    idx.forEach(i => { const c = E.seg.c[i]; if (c.length > 1) Lf.polyline(latlngs(c), {renderer, color: '#ffe600', ...w, lineCap: 'butt', interactive: false}).addTo(hotLayer); });
    const mid = E.seg.c[idx[Math.floor(idx.length / 2)]]; if (!mid || !mid.length) return;
    const pt = mid[Math.floor(mid.length / 2)];
    Lf.marker([pt[1], pt[0]], {icon: Lf.divIcon({className: '', html: badge(h, n), iconSize: [22, 22], iconAnchor: [30, 28]}), zIndexOffset: 1000})
      .on('click', () => pick(h.rank)).addTo(hotLayer);
  });
}
function recolor() {
  const cm = isCmp(), E = entry();
  if (!E) return;
  const v = cm ? cvals(E) : vals(E), {Z, zi} = zonesOf(E), zv = st.stops && !isIA() ? zvals(E, v, cm) : null;
  zoneLayer.clearLayers();
  // excluded interarrêts: thin and dashed, in grey
  if (isIA()) return drawn(E, v, cm).forEach((b, n) => segs[n] && segs[n].setStyle({color: cm ? cColor(b) : colorFor(b), weight: b.x ? 5 : 10, dashArray: b.x ? '3 6' : null}));
  // a shown stop zone: all its microsegments in the zone's colour, under one hatched line
  segs.forEach((p, i) => { const b = zv && zi[i] >= 0 ? zv[zi[i]] : v[i]; p.setStyle({color: cm ? cColor(b) : colorFor(b), weight: shown(b) ? 10 : 4}); });
  if (zv) for (const z of Z) {
    const ll = z.idx.flatMap(i => latlngs(E.seg.c[i]));
    if (ll.length > 1) window.L.polyline(ll, {renderer, color: 'url(#mszone)', weight: 10, lineCap: 'butt', interactive: false}).addTo(zoneLayer);
  }
}

