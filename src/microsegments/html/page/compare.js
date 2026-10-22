// ---------------------------------------------------------------- two-period comparison (D.compare)
const CMP = D.compare || null;
const isCmp = () => !!CMP && st.mode === 'cmp';
const ctx = () => CT[st.lang];
const cdirOf = () => CMP ? CMP.dirs.find(d => d.dir === st.dir) : null;
const CCOLS = CMP ? CMP.cols.map(String) : [];
const perLabel = P => P && P.first ? `${fmtShort(P.first)} – ${fmtShort(P.last)}` : '';
// Comparison values of segment j over an hour range: its own column when the range is one hour or a computed
// band; otherwise pooled from the hours valid in BOTH periods (obs per passage weighted by passages per hour, as a
// ratio of sums over hours with the same day count; CI bounds averaged, wider than the true one, so conservative:
// ap marks this approximate CI).
function cpick(C, R) {
  const b = bandOf(R), one = R[1] - R[0] === 1 ? CCOLS.indexOf(String(R[0])) : -1, bc = b ? CCOLS.indexOf(b) : -1;
  const ci = one >= 0 ? one : bc;
  if (ci >= 0) return j => { const g = a => a[ci][j]; return {d: g(C.d), lo: g(C.lo), hi: g(C.hi), sig: g(C.sig), oa: g(C.oa), ob: g(C.ob), fa: g(C.fa), fb: g(C.fb)}; };
  const cols = hoursIn(R).map(h => CCOLS.indexOf(String(h))).filter(c => c >= 0);
  return j => {
    let sa = 0, wa = 0, sb = 0, wb = 0, sl = 0, sh = 0, w = 0, fa = 0, fb = 0, n = 0;
    for (const c of cols) {
      const oa = C.oa[c][j], ob = C.ob[c][j], a = C.fa[c][j], b = C.fb[c][j];
      if (oa == null || ob == null || !(a > 0) || !(b > 0)) continue;
      sa += oa * a; wa += a; sb += ob * b; wb += b; fa += a; fb += b; n++;
      if (C.lo[c][j] != null && C.hi[c][j] != null) { const u = (a + b) / 2; sl += C.lo[c][j] * u; sh += C.hi[c][j] * u; w += u; }
    }
    if (!n) return {d: null};
    const oa = sa / wa, ob = sb / wb, lo = w ? sl / w : null, hi = w ? sh / w : null;
    return {d: ob - oa, lo, hi, sig: lo != null && (lo > 0 || hi < 0), oa, ob, fa: fa / n, fb: fb / n, ap: n > 1};
  };
}
// Per microsegment comparison values for an hour range of the display entry E (speeds: cKmh).
function cvals(E, R = range()) { return memo(`c|${E.pattern_uid}|${E.dir}|${R}`, () => cvals0(E, R)); }
function cvals0(E, R) {
  const C = CMP.dirs.find(d => d.dir === E.dir) || cdirOf();
  const pos = C ? (C._pos || (C._pos = Object.fromEntries(C.seg_key.map((x, i) => [x, i])))) : {}, pick = C ? cpick(C, R) : null;
  return E.seg.key.map((sk, i) => {
    const j = pos[sk], b = {i, l: E.seg.link[i], x0: E.seg.x0[i], len: E.seg.len[i], stop: isStop(E, i), nd: 0, flags: 0};
    if (j == null || !C.matched[j]) return {...b, absent: true, d: null, dk: null};
    return cKmh({...b, ...pick(j)}, b.len);
  });
}
const cColor = v => v.x ? XC : !shown(v) ? MASKC : v.absent ? NULLC : relColor(v.dr, !v.sig);
function cTipHtml(E, v, map) {
  const t = ctx().tip, l = tx().tip;
  let h = tipHead(E, v);
  const ut = v.u ? uTail(v, map) : '';
  if (v.absent) return h + t.absent + ut;
  if (v.dk == null) return h + l.nodata + ut;
  h += (v.z ? t.zd : t.d)(gapTxt(v.dk, v.dr)) + (v.sig ? '' : ` · <span style="color:var(--ink-3)">${t.unsure}</span>`);
  if (v.klo != null) h += `<br><span style="color:var(--ink-3)">${ciText(v)}${v.ap ? ` · ${tx().ciap}` : ''}</span>`;
  h += '<br>' + t.a(kf(v.va)) + '<br>' + t.b(kf(v.vb));
  if (v.fa != null || v.fb != null) h += '<br>' + t.freq(v.fa == null ? '–' : nf(v.fa, 1), v.fb == null ? '–' : nf(v.fb, 1), noun(true));
  if (v.stop && !v.z) h += `<br><i>${l.stop}</i>`;
  return h + ut;
}
const chgAll = () => CMP && hsOn() ? (CMP.changes || []).filter(c => c.dir === st.dir && (!c.pattern_uid || c.pattern_uid === (dirOf() || {}).pattern_uid)).sort((a, b) => a.rank - b.rank) : [];
const chgShown = () => chgAll().filter(c => st.stops || c.zone !== 'stop');
const chgDir = () => st.hotAll ? chgShown() : chgShown().slice(0, HOT_TOP);
const signCol = s => s < 0 ? FASTC : SLOWC;
// the diverging km/h bar: slower (red) on the left, faster (green) on the right
function gapBar() {
  const M = 100 * SC.CMPR, pos = v => (Math.min(M, Math.max(-M, v)) / M + 1) * 50;
  return rampBar(Array.from({length: 41}, (_, i) => relColor(SC.CMPR * (i - 20) / 20)),
                 SC.CRTICKS.map(v => `<span style="left:${pos(v)}%">${v === M ? '≥ ' : v === -M ? '≤ ' : ''}${v ? signed(v, 0) : '0'} %</span>`));
}
function cmpLegend() {
  const c = ctx().legend;
  const html = gapBar() + `<span class="txt">${esc(c.d(noun(false)))}${helpBtn(c.help.d(noun(false)))}</span>` +
    `<span><i class="sw" style="background:${NULLC}"></i>${c.absent}</span>` +
    zoneOrX() +
    (chgDir().length ? `<span><i class="hk"></i>${c.chg}</span>` : '');
  $('legend').innerHTML = html; $('mxlegend').innerHTML = html;
}
// speeds of a stretch (obs per passage summed over it, over its length); CI of the change in km/h
function stretchKmh(h, len) { return cKmh({oa: h.obs_per_passage_a, ob: h.obs_per_passage_b, lo: h.ci_lo, hi: h.ci_hi}, len); }
function cmpList() {
  const c = ctx(), hs = chgDir();
  $('hstitle').textContent = c.title2;
  $('hsnote').textContent = c.note(noun(false), noun(true));
  const stat = (cdirOf() || {}).status || (CMP.params || {}).status;   // per direction (older data: overall)
  if (stat && stat !== 'computed') { $('hslist').innerHTML = `<p class="empty">${c.few}</p>`; return; }
  if (!hs.length) { $('hslist').innerHTML = `<p class="empty">${chgAll().length ? c.stops : c.none}</p>`; return; }
  fillList(hs.map((h, i) => {
    const k = stretchKmh(h, h.x1_m - h.x0_m);
    const fr = h.passages_per_h_a != null && h.passages_per_h_b != null ? c.freq(nf(h.passages_per_h_a, 1), nf(h.passages_per_h_b, 1), noun(true)) : '';
    const meta = [c.sign[String(h.sign)], tx().zone[h.zone] || h.zone, runsLabel(h.hours || []), fr].filter(Boolean).join(' · ');
    const ci = ciText(k);
    const flag = h.frequency_changed ? `<div class="meta" style="color:var(--acc)">${esc(c.freqflag)}</div>` : '';
    return `<button type="button" class="hs${st.chg === h.rank ? ' on' : ''}" data-r="${h.rank}"><span class="n" style="background:${signCol(h.sign)};color:#111">${i + 1}</span>
      <div><div class="nm">${esc(h.from_stop_name)} <span class="ar">→</span> ${esc(h.to_stop_name)}</div>
      <div class="meta">${esc(meta)}</div><div class="meta">${esc(ci)}</div>${flag}</div>
      <div class="v">${ks(k.dk)}<small>${tx().kmh} (${pc(k.dr)})</small><small>${k.va == null ? '' : esc(c.ab(kf(k.va), kf(k.vb)))}</small></div></button>`;
  }).join(''), chgShown().length, c.more, selectChg);
}
function selectChg(r) { pickListed('chg', chgDir(), r); }
