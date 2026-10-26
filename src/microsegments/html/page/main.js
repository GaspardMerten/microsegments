// ---------------------------------------------------------------- render
function pressed(el, f) { el.querySelectorAll('button').forEach(b => b.setAttribute('aria-pressed', f(b.dataset.k))); }
function render() {
  pressed($('langs'), k => k === st.lang);
  pressed($('days'), k => st.days.includes(+k));
  pressed($('dirs'), k => +k === st.dir); pressed($('mxdirs'), k => +k === st.dir);
  pressed($('bands'), k => k === bandOf());
  pressed($('metric'), k => k === st.metric);
  if (CMP) pressed($('modes'), k => k === st.mode);
  $('g_days').hidden = isCmp() || HOST;
  pressed($('stopsT'), k => k === String(st.stops));
  pressed($('views'), k => k === st.view);
  const d = dirOf(); if (!$('g_ver').hidden && d) pressed($('vers'), k => k === (st.ver || d.pattern_uid));
  rangeUi();
  // the side panel: hotspots (when shown) or interarrêts
  const nohs = !hsOn() && st.view === 'ms';
  if (document.body.classList.contains('nohs') !== nohs) { document.body.classList.toggle('nohs', nohs); if (map) setTimeout(() => map.invalidateSize(), 0); }
  if (popBtn) closePop();
  recolor(); legend(); drawHotMap(); drawHotList(); drawMatrix(); drawProfile(); drawCalendar();
}
// the hour-range controls
function rangeUi() {
  pressed($('bands'), k => k === bandOf());
  const bn = bandOf(), dl = isCmp() ? ctx().days : daysLabel();
  $('clock').innerHTML = `${rangeLabel()}<small>${esc((bn ? tx().bandl[bn] + ', ' : '') + dl)}</small>`;
  hAEl.value = st.h0; hBEl.value = st.h1;
  $('hsel').style.left = hPos(st.h0); $('hsel').style.width = `calc(${(st.h1 - st.h0) / Math.max(1, HMAX - HMIN)} * (100% - var(--th)))`;
}
// A new hour range only: the matrix (all hours) and the calendar stay; its outline moves; the rest is redrawn
// once per frame.
let rafR = 0;
function renderRange() {
  rangeUi();
  if (rafR) return;
  rafR = requestAnimationFrame(() => {
    rafR = 0; if (popBtn) closePop();
    recolor(); legend(); if (st.view !== 'ms') drawHotList(); drawProfile();
    const r = $('mxhr'), hs = hoursIn(range());
    if (r && MX && hs.length) { r.setAttribute('y', MX.m.t + D.hours.indexOf(hs[0]) * MX.RH); r.setAttribute('height', hs.length * MX.RH); }
  });
}
// embedded: any direction change (a version from ?ver= or the host, the matrix buttons) is posted so the host's control follows
let postedDir = null;
function drawAll() {
  buildVersions(); head(); drawMap(); render();
  if (window.parent !== window && postedDir !== null && st.dir !== postedDir) try { parent.postMessage({ms: 'dir', dir: st.dir}, '*'); } catch (x) {}
  postedDir = st.dir;
}
let timer = null;
function stop() { clearInterval(timer); timer = null; const b = $('play'); b.textContent = tx().play; b.setAttribute('aria-pressed', 'false'); }
// a handle pushes the other one (at least one hour shown)
hAEl.addEventListener('input', () => { stop(); const a = Math.min(+hAEl.value, HMAX - 1); setRange(a, Math.max(st.h1, a + 1)); renderRange(); });
// close handles: the one nearest the pointer goes on top, so either can be grabbed
function nearTop(e) {
  const r = $('dual').getBoundingClientRect(), th = parseFloat(getComputedStyle($('dual')).getPropertyValue('--th')) || 18;
  const h = HMIN + (e.clientX - r.left - th / 2) / Math.max(1, r.width - th) * (HMAX - HMIN), b = h > (st.h0 + st.h1) / 2;
  hAEl.classList.toggle('top', !b); hBEl.classList.toggle('top', b);
}
$('dual').addEventListener('pointermove', e => { if (!e.buttons) nearTop(e); });
$('dual').addEventListener('pointerdown', nearTop, true);
hBEl.addEventListener('input', () => { stop(); const b = Math.max(+hBEl.value, HMIN + 1); setRange(Math.min(st.h0, b - 1), b); renderRange(); });
// play: a one-hour window stepping through the day (from the current hour when one hour is shown)
$('play').addEventListener('click', () => {
  if (timer) return stop();
  const b = $('play'); b.textContent = tx().pause; b.setAttribute('aria-pressed', 'true');
  setRange(st.h1 - st.h0 === 1 && st.h1 < HMAX ? st.h0 : HMIN, 0); renderRange();
  timer = setInterval(() => { const h = st.h0 + 1 >= HMAX ? HMIN : st.h0 + 1; setRange(h, h + 1); renderRange(); }, 900);
});
window.addEventListener('resize', () => { if (map) map.invalidateSize(); drawProfile(); drawMatrix(); });
// Full screen: the card becomes a fixed overlay (works everywhere), plus the browser's full screen when available.
const mxCard = $('mxcard'), mxBtn = $('mxfull');
function setFull(on) {
  mxCard.classList.toggle('full', on); document.body.classList.toggle('mxopen', on);
  mxBtn.textContent = on ? tx().unfull : tx().full; mxBtn.setAttribute('aria-pressed', on);
  if (on && mxCard.requestFullscreen) mxCard.requestFullscreen().catch(() => {});
  if (!on && document.fullscreenElement) document.exitFullscreen().catch(() => {});
  requestAnimationFrame(drawMatrix);
}
mxBtn.onclick = () => setFull(!mxCard.classList.contains('full'));
document.addEventListener('fullscreenchange', () => { if (!document.fullscreenElement && mxCard.classList.contains('full')) setFull(false); else drawMatrix(); });
document.addEventListener('keydown', e => { if (e.key === 'Escape' && mxCard.classList.contains('full')) setFull(false); });

// show a route version by pattern_uid (its direction too); false if unknown. Host: postMessage({ms: 'ver', uid}),
// or ?ver=<pattern_uid> on load
function selectVer(uid) {
  const d = DIRS.find(x => x.pattern_uid === uid || (x.alt || []).some(a => a.pattern_uid === uid));
  if (!d || isCmp()) return false;
  st.dir = d.dir; st.ver = uid === d.pattern_uid ? null : uid; st.hot = null; st.chg = null;
  return true;
}
if (qs.get('ver')) selectVer(qs.get('ver'));
buildControls(); texts(); tiles(); initMap(); drawAll();
window.addEventListener('message', e => {
  const m = e.data || {};
  if (m.ms === 'ver' && selectVer(String(m.uid))) { buildControls(); drawAll(); }
  // the host's excluded interarrêts and display mode (see saveUrl); not echoed back
  if (m.ms === 'xia' && e.source !== window && Array.isArray(m.xia)) {
    st.xia = new Set(m.xia.map(String));
    const vw = viewOf(m.view);
    if (vw && vw !== st.view) { st.view = vw; tiles(); drawAll(); } else { tiles(); render(); }
  }
});
document.querySelector('section.method').hidden = !METHOD;
if (HOST) for (const el of [$('langs'), $('footer'), $('covtitle').closest('section')]) el.style.display = 'none';
// embedded: tell the host the directions, follow its direction control
if (window.parent !== window) {
  postedDir = st.dir;
  try { parent.postMessage({ms: 'dirs', dir: st.dir, dirs: DIRS.map(d => ({dir: d.dir, from: d.from, to: d.to}))}, '*'); } catch (x) {}
  window.addEventListener('message', e => {
    const m = e.data || {};
    if (m.ms === 'dir' && SETDIR && DIRS.some(d => d.dir === +m.dir) && +m.dir !== st.dir) { postedDir = +m.dir; SETDIR(m.dir); }
  });
}
