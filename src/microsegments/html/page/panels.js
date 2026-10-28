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
      <div class="v">${k ? ks(k.dk) : '–'}<small>${l.kmh}</small><small>${k ? esc(l.hsv(kf(k.vb), kf(k.va))) : ''}</small></div></button>`;
  }).join(''), hotAll().length, l.hsmore, selectHot);
}
// the list of hotspots or retained changes (items: their buttons); over HOT_TOP of n, a "show all / top" toggle (more(n))
function fillList(items, n, more, pick) {
  const el = $('hslist');
  el.innerHTML = items + (n > HOT_TOP ? `<button type="button" class="btn" id="hsmore" style="margin:8px 0 4px">${st.hotAll ? tx().hsless : more(n)}</button>` : '');
  el.querySelectorAll('.hs').forEach(b => b.onclick = () => pick(+b.dataset.r));
  const mb = $('hsmore'); if (mb) mb.onclick = () => { st.hotAll = !st.hotAll; render(); };
}
// ---------------------------------------------------------------- interarrêts panel
const allIdx = E => E._all || (E._all = E.seg.key.map((_, i) => i));
const lenLabel = m => m < 1000 ? `${nf(m, 0)} m` : `${nf(m / 1000, 1)} km`;
const kmhPair = (S, cm) => cm ? `${kf(S.va)} → ${kf(S.vb)} km/h` : `${kf(S.kmh)} km/h`;
function toggleXia(id, on) {
  if (on ?? !st.xia.has(id)) st.xia.add(id); else st.xia.delete(id);
  saveUrl(); tiles(); render();
}
// tick (on) or untick every interarrêt of the entry shown
function setAllXia(on) {
  const E = entry(); if (!E) return;
  for (const u of unitsOf(E).U) if (on) st.xia.delete(u.id); else st.xia.add(u.id);
  saveUrl(); tiles(); render();
}
function iaList() {
  const l = tx(), E = entry(), cm = isCmp(), el = $('hslist');
  $('hstitle').textContent = l.ia.title; $('hsnote').textContent = l.ia.note;
  if (!E) { el.innerHTML = ''; return; }
  const run = !st.stops, v = cm ? cvals(E) : vals(E), uv = uvals(E, v, st.xia, cm, run), L = stretch(E, v, allIdx(E), st.xia, cm, run), nx = uv.filter(b => b.x).length;
  const fid = document.activeElement && document.activeElement.dataset ? document.activeElement.dataset.id : null;
  const btn = (id, t, off) => `<button type="button" class="btn" id="${id}"${off ? ' disabled' : ''}>${esc(t)}</button>`;
  el.innerHTML = `<p class="lnum"><b>${esc(l.line)} : ${kmhPair(L, cm)}</b> · ${lenLabel(L.len)}${nx ? ` · ${esc(l.nx(nx))}` : ''}${run ? ` · ${esc(l.ia.nozone)}` : ''}</p>` +
    `<div class="iactl">${btn('iaall', l.ia.all, !nx)}${btn('ianone', l.ia.none, nx === uv.length)}</div><div class="ialist">` +
    uv.map(b => { const val = cm ? b.dk : metricVal(b), rel = cm || st.metric === 'ex';
      return `<label class="ia${b.x ? ' x' : ''}"><input type="checkbox" data-id="${esc(b.u.id)}"${b.x ? '' : ' checked'}><i class="sw" style="background:${cm ? cColor(b) : colorFor(b)}"></i>` +
        `<span class="nm">${esc(b.u.from)} <span class="ar">→</span> ${esc(b.u.to)}</span><span class="v">${val == null ? '–' : rel ? pc(b.dr) : kf(val)}</span></label>`; }).join('') + '</div>';
  if (fid) { const f = el.querySelector(`input[data-id="${CSS.escape(fid)}"]`); if (f) f.focus(); }
}
$('hslist').addEventListener('change', e => { const id = e.target.dataset && e.target.dataset.id; if (id != null && e.target.type === 'checkbox') toggleXia(id, !e.target.checked); });
$('hslist').addEventListener('click', e => { if (e.target.id === 'iaall' || e.target.id === 'ianone') setAllXia(e.target.id === 'iaall'); });
// a click on a listed stretch (hotspot: key 'hot', change: 'chg'): (un)select it, show its peak hour, zoom onto it
function pickListed(key, list, r) {
  st[key] = st[key] === r ? null : r;
  const h = list.find(x => x.rank === st[key]);
  if (h && h.peak_hour != null && h.peak_hour >= 0 && D.hours.includes(h.peak_hour)) { stop(); setRange(h.peak_hour, h.peak_hour + 1); }
  render();
  if (h && map) { const E = entry(), pts = hotSegIdx(E, h).flatMap(i => latlngs(E.seg.c[i])); if (pts.length) map.fitBounds(window.L.latLngBounds(pts), {maxZoom: 16, padding: [80, 80]}); }
}
function selectHot(r) { pickListed('hot', hotOfDir(), r); }

