// ---------------------------------------------------------------- static texts
// "du 2 janvier au 30 mars 2026" / "2 January to 30 March 2026" (year, month written once when shared)
function spanLabel(a, b) {
  if (!a) return '';
  const da = new Date(a + 'T12:00:00'), db = new Date(b + 'T12:00:00'), y = da.getFullYear() === db.getFullYear(), mo = y && da.getMonth() === db.getMonth();
  const first = mo ? (st.lang === 'fr' && da.getDate() === 1 ? '1er' : String(da.getDate())) : fmtDate(a, y ? {day: 'numeric', month: 'long'} : undefined);
  return tx().span(first, fmtDate(b));
}
// one title, "Tram 92 direction Fort-Jaco" (terminus of the shown direction and version), and one period line
function head() {
  const l = tx(), mode = (MODE[st.lang][D.mode] || MODE[st.lang]._), E = entry();
  const badge = D.line ? `<span class="badge">${esc(D.line)}</span>` : '';
  $('h1').innerHTML = DIRS.length || D.line ? l.title(esc(mode), badge, E ? esc(E.to) : '') : esc(D.title || '');
  const x = s => s ? ` <span class="x">· ${esc(s)}</span>` : '';
  $('per').innerHTML = isCmp() ? esc(ctx().per(spanLabel(CMP.a.first, CMP.a.last), spanLabel(CMP.b.first, CMP.b.last)))
    : esc(spanLabel(D.period.first, D.period.last)) + x(exclSummary(D.period.excluded));
}
function texts() {
  const l = tx();
  document.documentElement.lang = st.lang;
  $('stopsQ').innerHTML = helpBtn(stopsHelpText());
  document.getElementById('l_mode').textContent = l.view;
  for (const [id, k] of [['l_views', 'views'], ['l_days', 'days'], ['l_dir', 'dir'], ['l_ver', 'ver'], ['l_bands', 'bands'], ['l_metric', 'metric'], ['l_stops', 'stops']])
    document.getElementById(id).textContent = l[k];
  document.getElementById('stopsT').title = '';
  document.getElementById('hstitle').textContent = l.hstitle;
  document.getElementById('mxtitle').textContent = l.mxtitle;
  document.getElementById('proftitle').textContent = l.proftitle;
  document.getElementById('covtitle').textContent = l.covtitle;
  document.getElementById('mtitle').textContent = l.mtitle;
  document.getElementById('method').innerHTML = l.method(noun(true), noun(false)).filter(x => hsOn() || !/^<b>(Points chauds|Hotspots)\./.test(x)).concat(CMP ? [ctx().method(noun(false), noun(true), perLabel(CMP.a), perLabel(CMP.b))] : []).map(x => `<li>${x}</li>`).join('');
  document.getElementById('footer').textContent = l.footer(D.source);
  document.getElementById('play').textContent = timer ? l.pause : l.play;
  document.getElementById('mxfull').textContent = mxCard.classList.contains('full') ? l.unfull : l.full;
  document.getElementById('map').setAttribute('aria-label', l.mname(noun(false))[st.metric]);
  hAEl.setAttribute('aria-label', l.hstart); hBEl.setAttribute('aria-label', l.hend);
  // hour ticks at the thumbs' true position for that hour
  const tk = [HMIN, 8, 12, 16, 20, HMAX].filter((h, i, a) => h >= HMIN && h <= HMAX && a.indexOf(h) === i);
  document.getElementById('hticks').innerHTML = tk.map(h => `<span style="left:${hPos(h)}">${h} h</span>`).join('');
}
// "4 jours exclus : 2 fériés, 2 flux incomplets" (missing, frozen and empty feed hours count as one reason)
const FEED_REASONS = ['low_coverage', 'frozen', 'no_data'];
function exclSummary(excluded) {
  const l = tx(), ex = excluded || [];
  if (!ex.length) return '';
  const counts = new Map();
  for (const e of ex) { const k = FEED_REASONS.includes(e.reason) ? 'feed' : e.reason; counts.set(k, (counts.get(k) || 0) + 1); }
  const parts = [...counts].sort((a, b) => b[1] - a[1]).map(([k, n]) => {
    const w = l.xr[k] || [l.reasons[k] || k, l.reasons[k] || k];
    return `${nf(n, 0)} ${w[n > 1 ? 1 : 0]}`;
  });
  return l.xsum(ex.length, parts.join(', '));
}
function tiles() {
  const l = tx();
  // line totals: without the microsegments of excluded interarrêts
  let tot = 0, nseg = 0;
  for (const d of DIRS) {
    const {all, ui} = unitsOf(d), keep = d.seg.key.map((_, i) => !(ui[i] >= 0 && st.xia.has(all[ui[i]].id)));
    nseg += keep.filter(Boolean).length;
    for (const w of PRESENT) for (const row of d.wk.obs[w]) row.forEach((v, i) => { if (keep[i]) tot += v; });
  }
  const nh = (D.hotspots || []).length;
  // on the platform the data-quality block has these numbers
  $('tiles').hidden = HOST;
  if (HOST) return;
  if (isCmp()) {
    const c = ctx().tiles, ch = CMP.changes || [];
    document.getElementById('tiles').classList.remove('t3');
    document.getElementById('tiles').classList.toggle('t2', !hsOn());
    const ns = ch.filter(x => x.sign > 0).length, nq = ch.filter(x => x.sign < 0).length;
    document.getElementById('tiles').innerHTML = [[nf(CMP.a.days, 0), `${c.a(CMP.a.days)} (${perLabel(CMP.a)})`], [nf(CMP.b.days, 0), `${c.b(CMP.b.days)} (${perLabel(CMP.b)})`],
      ...(hsOn() ? [[nf(ns, 0), c.slower(ns)], [nf(nq, 0), c.faster(nq)]] : [])]
      .map(([b, s]) => `<div class="tile"><b>${b}</b><span>${esc(s)}</span></div>`).join('');
    return;
  }
  document.getElementById('tiles').classList.remove('t2');
  document.getElementById('tiles').classList.toggle('t3', !hsOn());
  document.getElementById('tiles').innerHTML = [
    [nf(D.period.days, 0), esc(l.tiles.days(D.period.days, ''))],
    [nf(nseg, 0), l.tiles.segs(nseg, nf(P.segment_m || 30, 0))],
    [bigNum(tot), l.tiles.obs],
    ...(hsOn() ? [[nf(nh, 0), l.tiles.hs(nh)]] : []),
  ].map(([b, s]) => `<div class="tile"><b>${b}</b><span>${s}</span></div>`).join('');
}

