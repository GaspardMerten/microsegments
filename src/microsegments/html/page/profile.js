
// ---------------------------------------------------------------- profile along the line
const PROF_KMH = 60;   // fixed height of the speed profile, km/h (higher speeds are clipped)
function drawProfile() {
  const E = entry(), svg = $('prof'), l = tx();
  if (!E) { svg.innerHTML = ''; return; }
  if (isCmp()) return cmpProfile();
  if (st.metric === 'ex') {
    gapProfile(E, drawn(E, vals(E), false), lossOf, colorFor, tipHtml, SC.LMAX, true);
    return setSub($('profsub'), l.profsub(E.from, E.to, whenLabel(), false, noun(true)));
  }
  const v = drawn(E, vals(E), false);
  const evening = !(st.h0 >= REFH[0] && st.h1 <= REFH[1]) ? drawn(E, vals(E, REFH), false) : null;
  // fixed height, whatever the line, hour or days; the evening as a line over the bars
  profile(E, v, {val: b => b.kmh, col: colorFor, tip: tipHtml, Y0: 0, YM: PROF_KMH, step: 10, lab: k => nf(k, 0),
    over: (x, y) => { if (!evening) return ''; let pth = '';
      evening.forEach(b => { if (b.kmh != null && shown(b)) pth += `M${x(b.x0).toFixed(1)},${y(b.kmh).toFixed(1)}H${x(b.x0 + b.len).toFixed(1)}`; });
      return `<path d="${pth}" stroke="#eef1f6" stroke-width="1.5" fill="none" opacity=".7"/>`; },
    unit: W => `${W < 560 ? '' : esc(l.mname(noun(false))[st.metric].toLowerCase()) + ', '}km/h`});
  setSub($('profsub'), l.profsub(E.from, E.to, whenLabel(), !!evening, noun(true)));
}
// profile of a signed value (two periods: % of the reference speed; gap with the evening, pos: km/h lost >= 0)
function gapProfile(E, v, val, col, tip, YM, pos) {
  profile(E, v, {val, col, tip, Y0: pos ? 0 : -YM, YM, step: YM / 2, lab: k => Math.abs(k) < 1e-9 ? '0' : pos ? nf(k, 0) : signed(k, 0),
                 unit: () => pos ? 'km/h' : '%'});
}
function cmpProfile() {
  const E = entry();
  gapProfile(E, drawn(E, cvals(E), true), b => b.dr == null ? null : 100 * b.dr, b => b.x ? XC : relColor(b.dr, !b.sig), cTipHtml, 200 * SC.CMPR);
  setSub($('profsub'), ctx().profsub(E.from, E.to, whenLabel(), noun(false), noun(true)));
}
// Bars of val(b) from 0 for the items v (col: their colour, tip: their tooltip) on [Y0, YM], a grid line every step
// (labels lab(k)), the selection, the stops, masked stop zones; over(x, y): drawn on top; unit(W): the axis title.
function profile(E, v, {val, col, tip, Y0, YM, step, lab, over, unit}) {
  const svg = $('prof'), W = svg.clientWidth || 800, H = 300, m = {l: 48, r: 12, t: 18, b: 74};
  const Ltot = E.stops.length ? E.stops[E.stops.length - 1].x : 1;
  const x = u => m.l + u / Ltot * (W - m.l - m.r), y = u => m.t + (1 - (Math.min(Math.max(u, Y0), YM) - Y0) / (YM - Y0)) * (H - m.t - m.b);
  let s = '<g class="axis">';
  for (let k = Y0; k <= YM + 1e-9; k += step)
    s += `<line x1="${m.l}" x2="${W - m.r}" y1="${y(k)}" y2="${y(k)}" stroke-dasharray="${Math.abs(k) < 1e-9 ? '' : '2 4'}"/><text x="${m.l - 6}" y="${y(k) + 4}" text-anchor="end">${lab(k)}</text>`;
  s += '</g>' + selBand(E, x, H, m) + stopTicks(E, x, H, m);
  v.forEach(b => { if (!shown(b)) s += maskBar(x, b, m, H); });
  v.forEach(b => { const u = val(b); if (u == null || !shown(b)) return; const ys = y(u), y0 = y(0);
    const r = [x(b.x0).toFixed(1), Math.min(ys, y0).toFixed(1), Math.max(1, x(b.x0 + b.len) - x(b.x0) - .5).toFixed(1), Math.max(.5, Math.abs(y0 - ys)).toFixed(1)];
    s += `<rect x="${r[0]}" y="${r[1]}" width="${r[2]}" height="${r[3]}" fill="${col(b)}"/>` + (b.z ? zoneHatch(...r) : ''); });
  s += (over ? over(x, y) : '') + `<text x="${m.l}" y="${m.t - 6}" fill="var(--ink-3)" font-size="11" font-family="var(--mono)">${unit(W)}</text>`;
  svg.innerHTML = s;
  svg.onmousemove = e => { const r = svg.getBoundingClientRect(), u = (e.clientX - r.left - m.l) / (W - m.l - m.r) * Ltot; if (u < 0 || u > Ltot) return hideTip();
    showTip(e, tip(E, itemAt(v, u))); };
  svg.onmouseleave = hideTip;
}
// the selection behind a profile
function selBand(E, x, H, m) {
  const S = isSel() && selRange(); if (!S) return '';
  const a = x(E.seg.x0[S[0]]), b = x(E.seg.x0[S[1]] + E.seg.len[S[1]]);
  return `<rect x="${a.toFixed(1)}" y="${m.t}" width="${Math.max(2, b - a).toFixed(1)}" height="${H - m.t - m.b}" fill="#eef1f6" fill-opacity=".08" stroke="#eef1f6" stroke-opacity=".5" stroke-dasharray="4 3"/>`;
}
// stop lines and names under a profile
const stopTicks = (E, x, H, m) => E.stops.map(t => `<line x1="${x(t.x)}" x2="${x(t.x)}" y1="${m.t}" y2="${H - m.b}" stroke="#eef1f6" stroke-opacity=".15"/>`).join('')
  + stopNames(E, x, H - m.b + 14, 'var(--ink-3)', 10);
// stop names along an axis at y, thinned so they do not overlap (profile, matrix)
function stopNames(E, x, y, fill, size) {
  let s = '', last = -1e9;
  for (const t of E.stops) { const xx = x(t.x); if (xx - last < 13) continue; last = xx;
    s += `<text x="${xx}" y="${y}" transform="rotate(-40 ${xx} ${y})" text-anchor="end" fill="${fill}" font-size="${size}" font-family="var(--sans)">${esc(t.name)}</text>`; }
  return s;
}
