// ---------------------------------------------------------------- stop zones drawn as one block
// <zones> (pure: no DOM; a test runs this block in node)
// Near a stop the feed cannot place the vehicle (STIB: "distance 0" from ~36 m before to ~36 m after the
// stop point, then a zone-exit fix at ~34-46 m), so the 3-5 microsegments of a stop zone show an
// artefactual brown / green / red pattern. When zones are shown, each one is ONE block: the sum over its
// microsegments of obs / passages (each on its own link), i.e. the total time per vehicle in the zone, and
// its speed = the zone's drawn length / that time. ~30 s per tram over 90 m reads 10.8 km/h (orange) on the
// same fixed scale, not a saturated dark red; a long dwell (> 60 s) still turns dark red.
const ZONE = P.stop_zone || [30, 60];
// grid "stops": the zones follow the feed (from where it switches to the stop to its exit mark, set per stop)
const FEEDZ = P.grid === 'stops';
function zoneSpan(lang, one) {
  const fr = lang === 'fr', a = nf(ZONE[0], 0), b = nf(ZONE[1], 0), m1 = noun(false);
  // "le tram" / "la rame"; the feed is STIB's on the platform, any linear feed otherwise
  const le = fr ? (m1 === 'rame' ? 'la' : 'le') : 'the', who = isStib() ? (fr ? 'la STIB' : 'STIB') : (fr ? 'le flux' : 'the feed');
  if (!FEEDZ) return fr ? `De ${a}\u00a0m avant à ${b}\u00a0m après ${one ? 'l\'arrêt' : 'chaque arrêt'}` : `From ${a} m before to ${b} m after ${one ? 'the stop' : 'each stop'}`;
  return fr ? (one ? `La partie où ${who} situe ${le} ${m1} à l'arrêt, de son approche à son départ` : `Les parties où ${who} situe ${le} ${m1} à un arrêt, de son approche à son départ`)
            : (one ? `The stretch where ${who} ties the ${m1} to the stop, from its approach until it leaves` : `The stretches where ${who} ties the ${m1} to a stop, from its approach until it leaves`);
}
function zonesOf(E) {
  if (E._zones) return E._zones;
  const Z = [], zi = E.seg.key.map(() => -1);
  E.seg.key.forEach((_, i) => {
    if (E.seg.zone[i] !== 'stop') return;
    let z = Z[Z.length - 1];
    // a new block unless this microsegment continues the previous one (same zone run, no gap)
    if (!z || z.idx[z.idx.length - 1] !== i - 1 || Math.abs(E.seg.x0[i] - z.x1) > 1) { z = {idx: [], x0: E.seg.x0[i], x1: E.seg.x0[i]}; Z.push(z); }
    z.idx.push(i); z.x1 = E.seg.x0[i] + E.seg.len[i]; zi[i] = Z.length - 1;
  });
  for (const z of Z) {
    const at = E.stops.filter(t => t.x >= z.x0 - .5 && t.x <= z.x1 + .5);
    z.name = at.length ? at.map(t => t.name).join(' / ') : (stopBefore(E, z.x0) || {}).name || '';
    z.len = z.x1 - z.x0;
  }
  return (E._zones = {Z, zi});
}
// sum of a field over a block's microsegments; null as soon as one of them has no value
const zsum = (bs, f) => bs.every(b => b[f] != null) ? bs.reduce((t, b) => t + b[f], 0) : null;
const zfirst = (bs, f) => { const b = bs.find(x => x[f] != null); return b ? b[f] : null; };
// One-period values of a block (a stop zone, an interarrêt) of microsegment values bs (vals) over len metres; head:
// its own keys. Time per vehicle = the sum of each part's obs / passages, each on its own link; no position report
// at all (e.g. a metro terminus): no data, not 0 s.
function blockVals(bs, len, head) {
  const none = !bs.some(b => b.obs > 0), s = f => none ? null : zsum(bs, f);
  const opp = s('opp'), ref = s('ref'), kmh = kmhOf(opp, len, zsum(bs, 'obs')), ek = kmhOf(ref, len, zsum(bs, 'robs'));
  return {...head, len, flags: bs.reduce((f, b) => f | b.flags, 0), nd: Math.min(...bs.map(b => b.nd)), obs: bs.reduce((t, b) => t + b.obs, 0),
          opp, ref, kmh, ek, dv: gapOf(ek, kmh), dr: relOf(ek, kmh), pph: zfirst(bs, 'pph')};
}
// Two-period values of a block, from the microsegment values bs (cvals): summed differences. It is "sure" when the
// sum of the lower (upper) bounds is above (below) zero: wider than the true interval of the sum, so conservative.
function cBlockVals(bs, len, head) {
  const absent = bs.some(b => b.absent), s = f => absent || !bs.length ? null : zsum(bs, f), lo = s('lo'), hi = s('hi');
  return cKmh({...head, len, nd: 0, flags: 0, absent, d: s('d'), lo, hi, sig: lo != null && hi != null && (lo > 0 || hi < 0),
               oa: s('oa'), ob: s('ob'), fa: zfirst(bs, 'fa'), fb: zfirst(bs, 'fb'), ap: bs.some(b => b.ap)}, len);
}
// values of each zone from the microsegment values v: one period (vals) or two (cvals, cm)
function zvals(E, v, cm = false) {
  return zonesOf(E).Z.map((z, n) => {
    const bs = z.idx.map(i => v[i]);
    return (cm ? cBlockVals : blockVals)(bs, z.len, {z, zone: n, i: z.idx[0], l: bs[bs.length - 1].l, x0: z.x0, stop: true});
  });
}
// two-period speeds over len metres from obs per passage oa / ob and the CI [lo, hi] of ob - oa (obs per passage):
// va, vb, dk = vb - va (dr: relative) and its CI [klo, khi] (oa held fixed; more time = less speed, so the bounds swap).
// When the time CI reaches near zero (oa + lo < oa / 4) the speed has no upper bound: khi is null (one-sided "> klo").
function cKmh(b, len) {
  const va = kmhOf(b.oa, len), vb = kmhOf(b.ob, len), ok = b.oa != null && va != null && va < KMAX, dk = gapOf(va, vb);
  const k = o => { const x = kmhOf(o, len); return x == null || x >= KMAX ? null : x - va; };
  return {...b, va, vb, dk, dr: relOf(va, vb),
          klo: ok && dk != null && b.hi != null ? k(b.oa + b.hi) : null, khi: ok && dk != null && b.lo != null && b.oa + b.lo >= .25 * b.oa ? k(b.oa + b.lo) : null};
}
// "IC 95 % : a à b km/h", one-sided when the upper bound is open
const ciText = v => v.klo == null ? '' : v.khi == null ? tx().ci1(ks(v.klo)) : tx().ci(ks(v.klo), ks(v.khi));
// what is drawn: running microsegments one by one and, when zones are shown, each stop zone as one block
function items(v, E, zv, stops) {
  if (!stops) return v;
  const zi = zonesOf(E).zi, out = [];
  v.forEach((b, i) => { const n = zi[i]; if (n < 0) out.push(b); else if (zv[n].i === i) out.push(zv[n]); });
  return out;
}
// </zones>
