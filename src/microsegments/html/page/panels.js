// ---------------------------------------------------------------- hotspots list
// Speed over a hotspot stretch (all days): its reference va (evening, or the line's rhythm for "persistent") and
// vb, the reference slowed by the hotspot's excess (obs per passage summed over the stretch); gap dk and its CI (cKmh).
function hotKmh(E, h) {
  // only the "peak" criterion is against the evening; "persistent" and "both" are against the line level
  const A = agg(E, PRESENT), idx = hotSegIdx(E, h), ref = h.criterion === 'peak' ? A.ref : A.refl;
  if (!idx.length || h.excess_per_passage == null || idx.some(i => ref[i] == null)) return null;
  const len = idx.reduce((t, i) => t + E.seg.len[i], 0), r = idx.reduce((t, i) => t + ref[i], 0);
  return cKmh({oa: r, ob: r + h.excess_per_passage, lo: h.ci_lo, hi: h.ci_hi}, len);
}
function drawHotList() {
  if (isIA()) return iaList();
  if (isSel()) return selPanel();
  if (isCmp()) return cmpList();
  $('hstitle').textContent = tx().hstitle;
  const l = tx(), E = entry(), hs = hotOfDir();
  $('hsnote').textContent = l.hsnote(noun(false), noun(true));
  if (!isMain()) { $('hslist').innerHTML = `<p class="empty">${l.hsalt}</p>`; return; }
  if (!hs.length) { $('hslist').innerHTML = `<p class="empty">${D.hotspots_status === 'not_computed' ? l.hsnot : hotDir().length ? l.hsstops : l.hsnone}</p>`; return; }
  fillList(hs.map((h, i) => {
    const k = hotKmh(E, h);
    const meta = [l.zone[h.zone] || h.zone, l.kind[h.kind] || h.kind, h.criterion ? l.crit[h.criterion] : '', h.criterion === 'persistent' ? '' : runsLabel((h.criterion === 'both' ? h.peak_hours : h.hours) || h.hours || [])].filter(Boolean).join(' · ');
    const ci = k ? ciText(k) : '';
    const per = h.persistence != null ? l.daysPct(nf(h.persistence * 100, 0)) : '';
    return `<button type="button" class="hs${st.hot === h.rank ? ' on' : ''}" data-r="${h.rank}"><span class="n">${i + 1}</span>
      <div><div class="nm">${esc(h.from_stop_name)} <span class="ar">→</span> ${esc(h.to_stop_name)}</div>
      <div class="meta">${esc(meta)}</div><div class="meta">${esc([ci, per].filter(Boolean).join(' · '))}</div></div>
      <div class="v">${k ? ks(k.dk) : '—'}<small>${l.kmh}</small><small>${k ? esc(l.hsv(kf(k.vb), kf(k.va))) : ''}</small></div></button>`;
  }).join(''), hotAll().length, l.hsmore, selectHot);
}
// the list of hotspots or retained changes (items: their buttons); over HOT_TOP of n, a "show all / top" toggle (more(n))
function fillList(items, n, more, pick) {
  const el = $('hslist');
  el.innerHTML = items + (n > HOT_TOP ? `<button type="button" class="btn" id="hsmore" style="margin:8px 0 4px">${st.hotAll ? tx().hsless : more(n)}</button>` : '');
  el.querySelectorAll('.hs').forEach(b => b.onclick = () => pick(+b.dataset.r));
  const mb = $('hsmore'); if (mb) mb.onclick = () => { st.hotAll = !st.hotAll; render(); };
}
// ---------------------------------------------------------------- interarrêts and selection panels
const allIdx = E => E._all || (E._all = E.seg.key.map((_, i) => i));
const lenLabel = m => m < 1000 ? `${nf(m, 0)} m` : `${nf(m / 1000, 1)} km`;
const durLabel = s => s == null ? '–' : s < 60 ? `${nf(s, 0)} s` : `${Math.floor(s / 60)} min ${String(Math.round(s % 60)).padStart(2, '0')} s`;
const kmhPair = (S, cm) => cm ? `${kf(S.va)} → ${kf(S.vb)} km/h` : `${kf(S.kmh)} km/h`;
function toggleXia(id, on) {
  if (on ?? !st.xia.has(id)) st.xia.add(id); else st.xia.delete(id);
  saveUrl(); tiles(); render();
}
function iaList() {
  const l = tx(), E = entry(), cm = isCmp(), el = $('hslist');
  $('hstitle').textContent = l.ia.title; $('hsnote').textContent = l.ia.note;
  if (!E) { el.innerHTML = ''; return; }
  const v = cm ? cvals(E) : vals(E), uv = uvals(E, v, st.xia, cm), L = stretch(E, v, allIdx(E), st.xia, cm), nx = uv.filter(b => b.x).length;
  const fid = document.activeElement && document.activeElement.dataset ? document.activeElement.dataset.id : null;
  el.innerHTML = `<p class="lnum"><b>${esc(l.line)} : ${kmhPair(L, cm)}</b> · ${lenLabel(L.len)}${nx ? ` · ${esc(l.nx(nx))} <button type="button" class="lk" id="iareinc">${esc(l.ia.reinc)}</button>` : ''}</p><div class="ialist">` +
    uv.map(b => { const val = cm ? b.dk : metricVal(b), sg = cm || st.metric === 'ex';
      return `<label class="ia${b.x ? ' x' : ''}"><input type="checkbox" data-id="${esc(b.u.id)}"${b.x ? '' : ' checked'}><i class="sw" style="background:${cm ? cColor(b) : colorFor(b)}"></i>` +
        `<span class="nm">${esc(b.u.from)} <span class="ar">→</span> ${esc(b.u.to)}</span><span class="v">${val == null ? '–' : cm ? pc(b.dr) : kf(val)}</span></label>`; }).join('') + '</div>';
  if (fid) { const f = el.querySelector(`input[data-id="${CSS.escape(fid)}"]`); if (f) f.focus(); }
}
// st.sel = {a, b, open}: first and last microsegment (indices of the entry shown); open while the last one is awaited
const selRange = () => st.sel ? [Math.min(st.sel.a, st.sel.b), Math.max(st.sel.a, st.sel.b)] : null;
// a click inside a shown stop zone takes the whole block
function zoneEnds(E, i) { const n = st.stops ? zonesOf(E).zi[i] : -1; if (n < 0) return [i, i]; const z = zonesOf(E).Z[n]; return [z.idx[0], z.idx[z.idx.length - 1]]; }
function pickSel(i) {
  const E = entry(); if (!E || i < 0) return;
  const [a, b] = zoneEnds(E, i);
  st.sel = !st.sel || !st.sel.open ? {a, b, open: true} : {a: Math.min(st.sel.a, st.sel.b, a), b: Math.max(st.sel.a, st.sel.b, b), open: false};
  render();
}
// from leaving stop a's zone to leaving stop b's (the interarrêts a .. b - 1); the whole line: all of them
function pickStop(k) {
  const E = entry(); if (!E) return;
  const U = unitsOf(E).U, last = E.stops.length - 1;
  if (st.sel && st.sel.open && st.sel.k != null && st.sel.k !== k) return selStops(Math.min(st.sel.k, k), Math.max(st.sel.k, k));
  if (st.sel && st.sel.open && st.sel.k == null) { const u = U.find(x => x.n === k - 1) || U.find(x => x.n === k); if (u) pickSel(u.idx[u.idx.length - 1]); return; }
  const u = U.find(x => x.n === Math.min(k, last - 1)); if (!u) return;
  st.sel = {a: u.idx[0], b: u.idx[0], open: true, k};
  render();
}
function selStops(a, b) {
  const U = unitsOf(entry()).U.filter(u => u.n >= a && u.n < b);
  st.sel = U.length ? {a: U[0].idx[0], b: U[U.length - 1].idx[U[U.length - 1].idx.length - 1], open: false} : null;
  render();
}
function selPanel() {
  const l = tx(), s = l.sel, E = entry(), cm = isCmp(), el = $('hslist');
  $('hstitle').textContent = s.title; $('hsnote').textContent = st.sel && st.sel.open ? s.hint2 : s.hint;
  if (!E) { el.innerHTML = ''; return; }
  // the controls are built once per entry and language (so a select keeps its focus), the numbers on every render
  const key = `${E.pattern_uid}|${st.lang}`;
  if (el.dataset.k !== key || !$('selnum')) {
    el.dataset.k = key;
    const o = (k0, k1) => `<option value="">${s.pick}</option>` + E.stops.slice(k0, E.stops.length - k1).map((t, k) => `<option value="${k + k0}">${esc(t.name)}</option>`).join('');
    el.innerHTML = `<div id="selnum"></div><div class="selctl"><button type="button" class="btn" id="selall">${esc(s.all)}</button>` +
      `<label>${esc(s.from)} <select id="selA">${o(0, 1)}</select></label><label>${esc(s.to)} <select id="selB">${o(1, 0)}</select></label>` +
      `<button type="button" class="btn" id="selclr">${esc(s.clear)}</button></div>`;
    $('selall').onclick = () => selStops(0, E.stops.length - 1);
    $('selclr').onclick = () => { st.sel = null; render(); };
    const go = () => { const a = $('selA').value, b = $('selB').value; if (a !== '' && b !== '' && +b > +a) selStops(+a, +b); };
    $('selA').onchange = go; $('selB').onchange = go;
  }
  const v = cm ? cvals(E) : vals(E), L = stretch(E, v, allIdx(E), st.xia, cm), R = selRange();
  const X = R ? stretch(E, v, allIdx(E).slice(R[0], R[1] + 1), st.xia, cm) : null;
  let h = `<p class="lnum"><b>${esc(l.line)} : ${kmhPair(L, cm)}</b>${X ? ` · ${esc(s.title)} : ${kmhPair(X, cm)}` : ''}</p>`;
  if (X) {
    const big = cm ? (X.va != null && X.vb != null ? ks(X.vb - X.va) : '–') : kf(X.kmh), f = cm ? ['oa', 'ob'] : ['opp'];
    const pair = g => f.map(g).join(' → ');
    const share = pair(k => X[k] != null && L[k] > 0 ? nf(100 * X[k] / L[k], 1) + ' %' : '–');
    h += `<div class="selbig"><b>${big}</b> <span>${tx().kmh}</span></div><div class="meta">${esc([`${s.len} ${lenLabel(X.len)}`,
      `${pair(k => durLabel(X[k] == null ? null : X[k] * TICK))} ${s.tpv(noun(false))}`, `${share} ${s.share}`].join(' · '))}</div>` +
      (X.skip > .5 ? `<div class="meta">${esc(s.skip(lenLabel(X.skip)))}</div>` : '') + (f.some(k => X[k] == null) ? `<div class="meta">${esc(s.none)}</div>` : '');
  }
  $('selnum').innerHTML = h;
  // the selects show the stops when the selection runs exactly from one stop's zone exit to another's
  const U = unitsOf(E).U, ua = R && U.find(u => u.idx[0] === R[0]), ub = R && U.find(u => u.idx[u.idx.length - 1] === R[1]);
  if (document.activeElement !== $('selA')) $('selA').value = ua && ub ? ua.n : '';
  if (document.activeElement !== $('selB')) $('selB').value = ua && ub ? ub.n + 1 : '';
}
// Esc clears the selection (before the other Esc handlers: not while a popover or the full-screen matrix is open)
window.addEventListener('keydown', e => { if (e.key === 'Escape' && isSel() && st.sel && !popBtn && !mxCard.classList.contains('full')) { st.sel = null; render(); } }, true);
$('hslist').addEventListener('change', e => { const id = e.target.dataset && e.target.dataset.id; if (id != null && e.target.type === 'checkbox') toggleXia(id, !e.target.checked); });
$('hslist').addEventListener('click', e => { if (e.target.id === 'iareinc') { st.xia.clear(); saveUrl(); tiles(); render(); } });
// a click on a listed stretch (hotspot: key 'hot', change: 'chg'): (un)select it, show its peak hour, zoom onto it
function pickListed(key, list, r) {
  st[key] = st[key] === r ? null : r;
  const h = list.find(x => x.rank === st[key]);
  if (h && h.peak_hour != null && h.peak_hour >= 0 && D.hours.includes(h.peak_hour)) { stop(); setRange(h.peak_hour, h.peak_hour + 1); }
  render();
  if (h && map) { const E = entry(), pts = hotSegIdx(E, h).flatMap(i => latlngs(E.seg.c[i])); if (pts.length) map.fitBounds(window.L.latLngBounds(pts), {maxZoom: 16, padding: [80, 80]}); }
}
function selectHot(r) { pickListed('hot', hotOfDir(), r); }

