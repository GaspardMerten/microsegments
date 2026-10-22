// <units> (pure: no DOM; a test runs this block in node)
// Interarrêt A→B: from leaving A's stop zone to leaving B's, i.e. the running bins of link A→B, its tail (B's zone
// upstream of B) and the head of B→C (B's zone downstream). A's own zone is in the unit before; the first stop's
// head is in none. Head = the leading stop microsegments of a link (a link without running bins: those whose
// centre is in its first half). Id "A>B" (stop ids), the key of ?xia=.
function unitsOf(E) {
  if (E._units) return E._units;
  const nl = E.stops.length - 1, fr = new Array(Math.max(0, nl)).fill(-1), ui = E.seg.key.map(() => -1), U = [];
  E.seg.link.forEach((l, i) => { if (l >= 0 && fr[l] < 0 && E.seg.zone[i] !== 'stop') fr[l] = i; });
  for (let n = 0; n < nl; n++) U.push({n, idx: [], id: `${E.stops[n].id}>${E.stops[n + 1].id}`, from: E.stops[n].name, to: E.stops[n + 1].name});
  E.seg.link.forEach((l, i) => {
    if (l < 0 || l >= nl) return;
    const head = E.seg.zone[i] === 'stop' && (fr[l] >= 0 ? i < fr[l] : E.seg.x0[i] + E.seg.len[i] / 2 < (E.stops[l].x + E.stops[l + 1].x) / 2);
    const n = head ? l - 1 : l;
    if (n >= 0) { U[n].idx.push(i); ui[i] = n; }
  });
  for (const u of U) { u.idx.sort((a, b) => a - b); const a = u.idx[0], b = u.idx[u.idx.length - 1];
    u.x0 = a == null ? E.stops[u.n].x : E.seg.x0[a]; u.x1 = b == null ? u.x0 : E.seg.x0[b] + E.seg.len[b]; u.len = u.idx.reduce((t, i) => t + E.seg.len[i], 0); }
  return (E._units = {U: U.filter(u => u.idx.length), all: U, ui});
}
// Values of each unit from the microsegment values v: one period (vals) or two (cvals, cm); xs: the excluded ids.
// Each part counts over the passages of its OWN link (running bins and tail: A→B's, head: B→C's), as the stop-zone
// blocks do: a single count (A→B's) would charge B→C's head with vehicles that end at B, or cross an hour boundary
// in between. Each part is itself a ratio of sums over hours and days.
function uvals(E, v, xs, cm = false) {
  return unitsOf(E).U.map((u, n) => (cm ? cBlockVals : blockVals)(u.idx.map(i => v[i]), u.len,
    {u, unit: n, i: u.idx[0], l: u.n, x0: u.x0, stop: false, x: xs.has(u.id)}));
}
// Totals over microsegments idx (a stretch, the line), skipping those outside a counted unit (excluded, or the first
// stop's zone): length, obs per passage (both periods when cm), null as soon as one counted part has no value.
function stretch(E, v, idx, xs, cm) {
  const {all, ui} = unitsOf(E); let len = 0, n = 0, skip = 0; const t = cm ? {oa: 0, ob: 0} : {opp: 0};
  for (const i of idx) {
    const u = ui[i] >= 0 ? all[ui[i]] : null;
    if (!u || xs.has(u.id)) { skip += E.seg.len[i]; continue; }
    len += E.seg.len[i]; n++;
    for (const f in t) t[f] = t[f] == null || v[i][f] == null || v[i].absent ? null : t[f] + v[i][f];
  }
  if (!n) for (const f in t) t[f] = null;
  return {len, skip, ...t, ...(cm ? {va: kmhOf(t.oa, len), vb: kmhOf(t.ob, len)} : {kmh: kmhOf(t.opp, len)})};
}
// </units>
const isIA = () => st.view === 'ia', isSel = () => st.view === 'sel';
// interarrêts: one item per unit; otherwise running microsegments and, when shown, the stop-zone blocks
const drawn = (E, v, cm = isCmp()) => isIA() ? uvals(E, v, st.xia, cm) : items(v, E, st.stops ? zvals(E, v, cm) : null, st.stops);
// light hatch over a zone block (matrix, profile)
const zoneHatch = (x, y, w, h) => `<rect x="${x}" y="${y}" width="${w}" height="${h}" fill="url(#mszone)" pointer-events="none"/>`;

