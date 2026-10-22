// ---------------------------------------------------------------- helpers
function stopBefore(E, x) { let s = E.stops[0]; for (const t of E.stops) if (t.x <= x + .5) s = t; return s; }
function hourRuns(hs) {
  const a = [...new Set(hs)].sort((x, y) => x - y), out = [];
  for (const h of a) { const r = out[out.length - 1]; if (r && h === r[1]) r[1] = h + 1; else out.push([h, h + 1]); }
  return out;
}
const runsLabel = hs => hourRuns(hs).map(r => rangeLabel(r)).join(', ');
function daysLabel() {
  if (st.days.length === PRESENT.length) return tx().alldays;
  if (st.days.join() === '0,1,2,3,4') return tx().weekdays;
  return st.days.map(w => tx().dowl[w]).join(', ');
}
const whenLabel = () => { const b = bandOf(); return b ? `${tx().bandl[b].toLowerCase()} (${rangeLabel()})` : rangeLabel(); };
// a speed gap in km/h, signed
const ks = v => v == null ? '–' : signed(v, Math.abs(v) < 10 ? 1 : 0);
// a gap in km/h and in % of the reference speed
const gapTxt = (g, r) => g == null ? '–' : `${ks(g)} km/h (${pc(r)})`;
const lossTxt = g => g == null ? '–' : `${kf(Math.max(0, -g))} km/h`;
function tipTail(t, v) {
  let h = '';
  if (v.pph != null) h += '<br>' + t.freq(nf(v.pph, 1), noun(true));
  if (v.flags & 32) h += `<br><span style="color:var(--ink-3)">${t.days(Math.round(v.nd))} · ${t.partial}</span>`;
  return h;
}
// an interarrêt: its name, excluded or not, and what a map click does
const uHead = v => `<b>${tx().tip.uhead(esc(v.u.from), esc(v.u.to))}</b><br>`;
const uTail = (v, map) => (v.x ? `<br><span style="color:var(--ink-3)">${tx().ia.xtip}</span>` : '') + (map ? `<br><span style="color:var(--ink-3)">${v.x ? tx().ia.off : tx().ia.on}</span>` : '');
// what is under the pointer (an interarrêt; a stop zone or microsegment, towards the link's end)
function tipHead(E, v) {
  if (v.u) return uHead(v);
  const t = tx().tip, s = stopBefore(E, v.x0), lk = E.links[v.l];
  return `<b>${v.z ? t.zhead(esc(v.z.name)) : t.after(Math.round(v.x0 - s.x), Math.round(v.x0 - s.x + v.len), esc(s.name))}</b> (${t.towards(esc(lk ? lk.to : E.to))})<br>`;
}
// one period; map: the interarrêt's map click hint
function tipHtml(E, v, map) {
  const t = tx().tip;
  let h = tipHead(E, v) + (v.kmh == null ? t.nodata : (v.z ? t.zopp : t.opp)(kf(v.kmh)) + (v.dv != null ? '<br>' + t.ex(lossTxt(v.dv)) : '')) + tipTail(t, v);
  if (v.stop && !v.z) h += `<br><i>${t.stop}</i>`;
  return h + (v.u ? uTail(v, map) : '');
}
// the item of v (sorted along the line) at u metres; past the end: the last one
const itemAt = (v, u) => { const i = v.findIndex(b => u >= b.x0 && u < b.x0 + b.len); return v[i < 0 ? v.length - 1 : i]; };
// leaflet [lat, lng] of a [lng, lat] polyline
const latlngs = c => c.map(([x, y]) => [y, x]);
const tipEl = document.getElementById('tip');
function showTip(e, html) { tipEl.innerHTML = html; tipEl.style.display = 'block'; const w = tipEl.offsetWidth, hh = tipEl.offsetHeight;
  tipEl.style.left = Math.min(window.innerWidth - w - 8, e.clientX + 14) + 'px'; tipEl.style.top = (e.clientY + 16 + hh > window.innerHeight ? e.clientY - hh - 10 : e.clientY + 16) + 'px'; }
const hideTip = () => { tipEl.style.display = 'none'; };
const HOT_TOP = 8;
// masked stop zone in the profiles: a hatched band over the plot height
const maskBar = (x, b, m, H) => `<rect x="${x(b.x0).toFixed(1)}" y="${m.t}" width="${Math.max(1, x(b.x0 + b.len) - x(b.x0) - .5).toFixed(1)}" height="${H - m.t - m.b}" fill="${MASKC}" opacity=".55"/>`;
// short text + '*' button; the button carries the plain-language explanation for the popover
const helpBtn = t => `<button type="button" class="qm" aria-expanded="false" aria-controls="pop" aria-label="${esc(tx().legend.what)}" data-help="${esc(t)}">*</button>`;
function setSub(el, parts) { el.innerHTML = esc(parts[0]) + helpBtn(parts[1]); }
const popEl = document.getElementById('pop');
let popBtn = null, popPinned = false;
function openPop(b, pin) {
  if (popBtn && popBtn !== b) popBtn.setAttribute('aria-expanded', 'false');
  popBtn = b; popPinned = pin; popEl.textContent = b.dataset.help; popEl.hidden = false; b.setAttribute('aria-expanded', 'true');
  const r = b.getBoundingClientRect(), w = popEl.offsetWidth, h = popEl.offsetHeight;
  popEl.style.left = Math.max(12, Math.min(window.innerWidth - w - 12, r.left + r.width / 2 - w / 2)) + 'px';
  popEl.style.top = (r.bottom + 8 + h > window.innerHeight && r.top - h - 8 > 0 ? r.top - h - 8 : r.bottom + 8) + 'px';
}
function closePop() { if (popBtn) popBtn.setAttribute('aria-expanded', 'false'); popBtn = null; popPinned = false; popEl.hidden = true; }
document.addEventListener('click', e => {
  const b = e.target.closest && e.target.closest('.qm');
  if (b) { e.preventDefault(); if (popBtn === b && popPinned) closePop(); else openPop(b, true); return; }
  if (popBtn && !popEl.contains(e.target)) closePop();
});
document.addEventListener('keydown', e => { if (e.key === 'Escape' && popBtn) { const b = popBtn; closePop(); b.focus(); } });
const canHover = window.matchMedia && matchMedia('(hover: hover)').matches;
document.addEventListener('mouseover', e => { const b = canHover && e.target.closest && e.target.closest('.qm'); if (b && !popPinned) openPop(b, false); });
document.addEventListener('mouseout', e => { const b = e.target.closest && e.target.closest('.qm'); if (b && popBtn === b && !popPinned) closePop(); });
document.addEventListener('focusin', e => { const b = e.target.closest && e.target.closest('.qm'); if (b && !popPinned && b.matches(':focus-visible')) openPop(b, false); });
window.addEventListener('scroll', () => { if (popBtn) { if (popBtn.isConnected) openPop(popBtn, popPinned); else closePop(); } }, {passive: true});
// stop-zone hotspots are listed only when stop zones are shown
const hsOn = () => D.show_hotspots !== false;
const hotDir = () => !hsOn() ? [] : (D.hotspots || []).filter(h => (h.dir ?? h.direction_id) === st.dir && (!h.pattern_uid || h.pattern_uid === (dirOf() || {}).pattern_uid)).sort((a, b) => a.rank - b.rank);
const hotAll = () => hotDir().filter(h => st.stops || h.zone !== 'stop');
// the list (and the map markers) show the top hotspots unless "show all" is on
const hotOfDir = () => { const a = hotAll(); return st.hotAll ? a : a.slice(0, HOT_TOP); };

