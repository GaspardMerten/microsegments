
// ---------------------------------------------------------------- sums over the chosen weekdays
// Totals are per weekday in the data (wk); every view sums the chosen weekdays, then divides.
// A ratio needs MINP passages per included day-hour (Quality.min_passages_per_day): p >= MINP x n, n the included
// day-hours of the link (wk.n; an older contract without it: the link's days, every hour).
const MINP = 0.5;
const enough = (p, n) => p > 0 && p >= MINP * n;
const AGG = new Map();
function agg(E, days = st.days) {
  const ck = E.pattern_uid + '|' + E.dir + '|' + days.join(',');
  if (AGG.has(ck)) return AGG.get(ck);
  const ns = E.seg.key.length, nl = E.links.length, o = {}, p = {}, n = {};
  D.hours.forEach((h, j) => {
    const a = new Float64Array(ns), q = new Float64Array(nl), r = new Float64Array(nl);
    for (const w of days) {
      const ow = E.wk.obs[w][j], pw = E.wk.p[w][j], nw = E.wk.n ? E.wk.n[w][j] : E.wk.days[w];
      for (let i = 0; i < ns; i++) a[i] += ow[i];
      for (let l = 0; l < nl; l++) { q[l] += pw[l]; r[l] += nw[l]; }
    }
    o[h] = a; p[h] = q; n[h] = r;
  });
  const sumH = (src, n, hs) => { const r = new Float64Array(n); for (const h of hs) if (src[h]) src[h].forEach((v, i) => r[i] += v); return r; };
  // any hour range [a, b): sums over its hours, cached
  const R = new Map();
  const rng = ([a, b]) => { const k = a + '-' + b; if (!R.has(k)) { const hs = hoursIn([a, b]);
    R.set(k, {o: sumH(o, ns, hs), p: sumH(p, nl, hs), n: sumH(n, nl, hs)}); } return R.get(k); };
  const rs = rng(REFH), ref = E.seg.link.map((l, i) => enough(rs.p[l], rs.n[l]) ? rs.o[i] / rs.p[l] : null);
  const nd = new Float64Array(nl);
  for (const w of days) E.wk.days[w].forEach((v, l) => nd[l] += v);
  const A = {ref, refo: rs.o, refl: lineRef(E, rng(LINEH)), nd, rng};
  AGG.set(ck, A);
  return A;
}
// line-level reference: median over running segments of the 6-23 h obs / passage / m, times the length
function lineRef(E, {o, p}) {
  const per = E.seg.key.map((_, i) => !isStop(E, i) && p[E.seg.link[i]] > 0 ? o[i] / p[E.seg.link[i]] / E.seg.len[i] : null)
    .filter(v => v != null).sort((a, b) => a - b), k = per.length;
  const lvl = k ? (k % 2 ? per[(k - 1) / 2] : (per[k / 2 - 1] + per[k / 2]) / 2) : null;
  return E.seg.len.map(len => lvl == null ? null : lvl * len);
}
function isStop(E, i) { return E.seg.zone[i] === 'stop'; }
const shown = v => st.stops || !v.stop;
// Per microsegment values for an hour range [a, b): obs per passage (opp, its evening ref), and the speeds
// (kmh, evening kmh ek, gap dv = kmh - ek, km/h).
// memo of the per microsegment values (the slider and the matrix ask for the same ranges again and again)
const VMEMO = new Map();
const memo = (k, f) => { if (VMEMO.has(k)) return VMEMO.get(k); if (VMEMO.size > 120) VMEMO.clear(); const r = f(); VMEMO.set(k, r); return r; };
function vals(E, R = range(), days = st.days) {
  return memo(`v|${E.pattern_uid}|${E.dir}|${R}|${days}`, () => vals0(E, R, days));
}
function vals0(E, R, days) {
  const A = agg(E, days), {o, p, n} = A.rng(R);
  return E.seg.key.map((_, i) => {
    const l = E.seg.link[i], len = E.seg.len[i];
    const opp = enough(p[l], n[l]) ? o[i] / p[l] : null, ref = A.ref[i];
    const kmh = kmhOf(opp, len, o[i]), ek = kmhOf(ref, len, A.refo[i]);
    return {i, l, x0: E.seg.x0[i], len, stop: isStop(E, i), flags: E.seg.flags[i] || 0, obs: o[i], robs: A.refo[i],
            opp, ref, kmh, ek, dv: gapOf(ek, kmh), dr: relOf(ek, kmh), pph: n[l] > 0 ? p[l] / n[l] : null, nd: A.nd[l]};
  });
}
// the value of the chosen measure (km/h) and its colour
// gap with the evening: the km/h lost against it (0 when faster), on a linear scale 0 .. SC.LMAX km/h
const lossOf = v => v.dv == null ? null : Math.max(0, -v.dv);
const metricVal = (v, m = st.metric) => m === 'ex' ? lossOf(v) : v.kmh;
// an excluded interarrêt: flat grey (XC)
const XC = '#2a3350';
const colorFor = v => v.x ? XC : !shown(v) ? MASKC : st.metric === 'ex' ? lossColor(lossOf(v)) : kmhColor(v.kmh);
