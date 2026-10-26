// ---------------------------------------------------------------- matrix (hours x microsegments)
// Fit to width; in full screen every link (stop to stop) gets MXCW px and the matrix scrolls sideways, the hour
// labels (#mxh) staying at the left.
const MXCW = 56;
let MX = null;
function mxScale(E, full, avail, m) {
  const xs = E.stops.map(t => t.x), Ltot = xs.length ? xs[xs.length - 1] : 1, nl = Math.max(1, xs.length - 1);
  if (!full || xs.length < 2 || nl * MXCW + m.l + m.r <= avail) {
    const k = (avail - m.l - m.r) / Ltot;
    return {W: avail, Ltot, x: u => m.l + u * k, inv: px => (px - m.l) / k};
  }
  const seg = u => { let i = 0; while (i < nl - 1 && u >= xs[i + 1]) i++; return i; };
  const x = u => { const i = seg(u); return m.l + (i + (u - xs[i]) / Math.max(1e-9, xs[i + 1] - xs[i])) * MXCW; };
  const inv = px => { const f = (px - m.l) / MXCW, i = Math.max(0, Math.min(nl - 1, Math.floor(f))); return xs[i] + (f - i) * (xs[i + 1] - xs[i]); };
  return {W: m.l + nl * MXCW + m.r, Ltot, x, inv};
}
function drawMatrix() {
  const E = entry(), svg = $('mx'), sh = $('mxh'), l = tx();
  if (!E) { svg.innerHTML = ''; sh.innerHTML = ''; return; }
  const m = {l: 2, r: 8, t: 8, b: 90}, LW = 44;
  const card = $('mxcard'), full = card.classList.contains('full');
  const RH = full ? Math.max(18, Math.floor((window.innerHeight - 220) / D.hours.length)) : 22;
  const cm = isCmp(), VALS = cm ? cvals : vals, COL = cm ? cColor : colorFor;
  const H = m.t + D.hours.length * RH + m.b;
  const {W, Ltot, x, inv} = mxScale(E, full, Math.max(200, $('mxwrap').clientWidth - LW), m);
  svg.style.width = W + 'px'; svg.setAttribute('height', H);
  sh.setAttribute('width', LW); sh.setAttribute('height', m.t + D.hours.length * RH);
  let s = '', lab = '';
  const rows = [];
  D.hours.forEach((h, r) => {
    const v = drawn(E, VALS(E, [h, h + 1]), cm), y = m.t + r * RH; rows.push(v);
    s += `<rect class="mxrow" data-h="${h}" x="${m.l}" y="${y}" width="${W - m.l - m.r}" height="${RH}" fill="#0b1224"/>`;
    v.forEach(b => { const xw = [x(b.x0).toFixed(1), Math.max(.6, x(b.x0 + b.len) - x(b.x0) - .3).toFixed(1)];
      s += `<rect x="${xw[0]}" y="${y + .5}" width="${xw[1]}" height="${RH - 1}" fill="${COL(b)}" pointer-events="none"/>` + (b.z ? zoneHatch(xw[0], y + .5, xw[1], RH - 1) : ''); });
    lab += `<text class="mxrow" data-h="${h}" x="${LW - 6}" y="${y + RH / 2 + 4}" text-anchor="end" fill="var(--ink-2)" font-size="12" font-family="var(--mono)">${h} h</text>`;
  });
  // dark lines: the stops, or the interarrêt bounds (zone exits)
  for (const u of isIA() ? unitsOf(E).U.map(w => w.x0) : E.stops.map(t => t.x)) s += `<line x1="${x(u)}" x2="${x(u)}" y1="${m.t}" y2="${m.t + D.hours.length * RH}" stroke="#0b1224" stroke-width="1.5" pointer-events="none"/>`;
  s += stopNames(E, x, H - m.b + 14, 'var(--ink-2)', 12);
  if (isMain()) for (const h of (cm ? chgDir() : hotOfDir())) for (const [a, b] of hourRuns(h.hours || [])) {
    const r0 = D.hours.indexOf(a), r1 = D.hours.indexOf(b - 1); if (r0 < 0 || r1 < 0) continue;
    const on = cm ? st.chg === h.rank : st.hot === h.rank;
    s += `<rect x="${x(h.x0_m) - 1}" y="${m.t + r0 * RH}" width="${Math.max(3, x(h.x1_m) - x(h.x0_m) + 2)}" height="${(r1 - r0 + 1) * RH}" fill="none" stroke="#ffe600" stroke-opacity="${on ? 1 : .8}" stroke-width="${on ? 2.5 : 1.25}" pointer-events="none"/>`;
  }
  // the hours shown
  const hs = hoursIn(range());
  if (hs.length) s += `<rect id="mxhr" x="${m.l - 1}" y="${m.t + D.hours.indexOf(hs[0]) * RH}" width="${W - m.l - m.r + 2}" height="${hs.length * RH}" fill="none" stroke="#b4bdd0" stroke-width="1.5" pointer-events="none"/>`;
  svg.innerHTML = s; sh.innerHTML = lab;
  MX = {E, rows, m, RH, inv, Ltot, tip: cm ? cTipHtml : tipHtml};
  setSub($('mxsub'), l.mxsub(E.from, E.to, cm ? ctx().days : daysLabel()));
}
$('mx').addEventListener('mousemove', e => {
  if (!MX) return;
  const r = $('mx').getBoundingClientRect(), px = e.clientX - r.left, py = e.clientY - r.top, {m, RH, inv, Ltot, E, rows} = MX;
  const row = Math.floor((py - m.t) / RH), u = inv(px);
  if (row < 0 || row >= rows.length || u < 0 || u > Ltot) return hideTip();
  const b = itemAt(rows[row], u);
  showTip(e, `<span style="font-family:var(--mono)">${rangeLabel([D.hours[row], D.hours[row] + 1])}</span> · ` + safeTip(() => MX.tip(E, b), E, b));
});
$('mx').addEventListener('mouseleave', hideTip);
// a row (or its hour label) selects that hour
for (const id of ['mx', 'mxh']) $(id).addEventListener('click', e => {
  const g = e.target.closest('.mxrow'); if (g) { stop(); setRange(+g.dataset.h, +g.dataset.h + 1); renderRange(); } });

