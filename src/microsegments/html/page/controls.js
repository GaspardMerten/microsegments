// ---------------------------------------------------------------- controls
const $ = id => document.getElementById(id);
function buttons(el, items, onclick) {
  el.innerHTML = '';
  for (const [k, t, title] of items) { const b = document.createElement('button'); b.type = 'button'; b.textContent = t; b.dataset.k = k; if (title) b.title = title; b.onclick = () => onclick(k); el.append(b); }
}
let SETDIR = null;
function buildControls() {
  const l = tx();
  buttons($('langs'), [['fr', 'FR'], ['en', 'EN']], k => { st.lang = k; buildControls(); texts(); tiles(); drawAll(); });
  buttons($('days'), [0,1,2,3,4,5,6].map(w => [w, l.dow[w], l.dowl[w]]), k => {
    const w = +k; if (!PRESENT.includes(w)) return;
    const on = st.days.includes(w);
    if (on && st.days.length === 1) return;
    st.days = on ? st.days.filter(x => x !== w) : [...st.days, w].sort();
    render();
  });
  $('days').querySelectorAll('button').forEach(b => { if (!PRESENT.includes(+b.dataset.k)) b.disabled = true; });
  const dirItems = DIRS.map(d => [d.dir, l.towards(d.to)]);
  const setDir = k => { st.dir = +k; st.ver = null; st.hot = null; st.chg = null; drawAll(); };
  buttons($('dirs'), dirItems, setDir);
  buttons($('mxdirs'), dirItems, setDir);
  $('dirs').closest('.grp').style.display = DIR_CONTROL ? '' : 'none'; $('mxdirs').style.display = DIR_CONTROL ? '' : 'none';
  SETDIR = setDir;
  // a band moves both handles to its hours
  buttons($('bands'), Object.keys(BANDDEF).map(k => [k, l.bandl[k], rangeLabel(BANDDEF[k])]), k => { stop(); setRange(...BANDDEF[k]); renderRange(); });
  $('g_mode').hidden = !CMP || HOST;
  if (CMP) buttons($('modes'), [['one', l.modes.one], ['cmp', l.modes.cmp]], k => { st.mode = k; st.ver = null; st.hot = null; st.chg = null; buildControls(); texts(); tiles(); drawAll(); });
  // two periods: one measure (the speed difference), no buttons
  $('metric').closest('.grp').hidden = isCmp();
  buttons($('metric'), Object.entries(l.metrics(noun(false))).map(([k, t]) => [k, t, l.mname(noun(false))[k]]), k => { st.metric = k; render(); });
  // interarrêts: the units are redrawn without (or with) their stop zones
  buttons($('stopsT'), [['true', l.stopsl.true], ['false', l.stopsl.false]], k => { st.stops = k === 'true'; if (isIA()) drawAll(); else render(); });
  buttons($('views'), Object.entries(l.viewl), k => { if (k === st.view) return; st.view = k; st.hot = null; st.chg = null; saveUrl(); drawAll(); });
  buildVersions();
}
function buildVersions() {
  const d = dirOf(), l = tx();
  const vs = (D.versions || []).filter(v => v.dir === st.dir).sort((a, b) => (b.display - a.display) || (a.first < b.first ? -1 : 1));
  $('g_ver').hidden = isCmp() || !(d && d.alt && d.alt.length);
  if ($('g_ver').hidden) return;
  buttons($('vers'), vs.map(v => [v.pattern_uid, l.vers(fmtShort(v.first), fmtShort(v.last), v.n_days) + (v.display ? ` (${l.verMain})` : ''),
    v.display ? '' : l.verNote(v.links_removed.length, v.links_added.length)]), k => { st.ver = k === d.pattern_uid ? null : k; st.hot = null; drawAll(); });
}
// the two handles: start hour hA in [HMIN, HMAX - 1], end hour hB in [HMIN + 1, HMAX] (end excluded)
const hAEl = $('hA'), hBEl = $('hB');
for (const el of [hAEl, hBEl]) { el.min = HMIN; el.max = HMAX; }
const hPos = h => `calc(var(--th) / 2 + ${(HMAX > HMIN ? (h - HMIN) / (HMAX - HMIN) : 0).toFixed(4)} * (100% - var(--th)))`;

