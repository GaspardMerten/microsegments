// ---------------------------------------------------------------- coverage calendar
function drawCalendar() {
  const l = tx(), cv = D.coverage || {dates: [], c: [], inc: []}, svg = $('cal');
  const dates = cv.dates || [];
  if (!dates.length) { svg.innerHTML = ''; $('covsub').textContent = ''; return; }
  const exc = Object.fromEntries((D.period.excluded || []).map(e => [e.date, e.reason]));
  const xlab = Object.fromEntries((D.period.excluded || []).filter(e => e.label).map(e => [e.date, e.label]));
  const dayIdx = D.hours.map((h, j) => [h, j]).filter(([h]) => h >= DAYH[0] && h < DAYH[1]).map(([, j]) => j);
  const t0 = new Date(dates[0] + 'T12:00:00Z'), dow0 = (t0.getUTCDay() + 6) % 7;
  const C = 15, G = 3, top = 16, left = 26;
  const weekOf = s => Math.floor(((new Date(s + 'T12:00:00Z') - t0) / 864e5 + dow0) / 7);
  const nW = weekOf(dates[dates.length - 1]) + 1;
  let s = '';
  for (let w = 0; w < 7; w += 2) s += `<text x="0" y="${top + w * (C + G) + C - 3}">${l.dow[w]}</text>`;
  let lastMonth = -1;
  dates.forEach((d, k) => {
    const dt = new Date(d + 'T12:00:00Z'), w = (dt.getUTCDay() + 6) % 7, col = weekOf(d);
    if (dt.getUTCMonth() !== lastMonth && w <= 3) { lastMonth = dt.getUTCMonth(); s += `<text x="${left + col * (C + G)}" y="10">${dt.toLocaleDateString(LOC[st.lang], {month: 'short', timeZone: 'UTC'})}</text>`; }
    const row = (cv.c || [])[k] || [], vs = dayIdx.map(j => row[j]).filter(v => v != null);
    const mc = vs.length ? vs.reduce((a, b) => a + b, 0) / dayIdx.length : null;
    const inc = cv.inc ? cv.inc[k] : (exc[d] ? exc[d] : true);
    const selected = st.days.includes(w) || !PRESENT.includes(w) && inc !== true;
    let fill, status;
    if (inc === true) { fill = st.days.includes(w) ? colorOf(14 + 8 * (mc == null ? 1 : mc)) : '#24324f'; status = st.days.includes(w) ? l.cal.inc : l.cal.off; }
    else if (inc === 'not_selected') { fill = '#16223d'; status = l.cal.off; }
    else { fill = inc === 'no_data' ? '#3a4560' : '#c2414b'; status = `${l.cal.exc} : ${l.reasons[inc] || inc}${xlab[d] ? ` (${xlab[d]})` : ''}`; }
    const tip = `<b>${esc(fmtDate(d, {weekday: 'long', day: 'numeric', month: 'long', year: 'numeric'}))}</b><br>${esc(status)}${mc != null ? '<br>' + esc(l.covday(nf(mc * 100, 0))) : ''}`;
    s += `<rect x="${left + col * (C + G)}" y="${top + w * (C + G)}" width="${C}" height="${C}" rx="3" fill="${fill}" data-tip="${esc(tip)}" opacity="${selected ? 1 : .55}"/>`;
  });
  svg.setAttribute('width', left + nW * (C + G)); svg.setAttribute('height', top + 7 * (C + G));
  svg.innerHTML = s;
  svg.querySelectorAll('rect').forEach(r => { r.onmousemove = e => showTip(e, r.dataset.tip); r.onmouseleave = hideTip; });
  setSub($('covsub'), l.covsub(D.period.days, exclSummary(D.period.excluded)));
  $('calleg').innerHTML = [[colorOf(22), l.cal.inc], ['#c2414b', l.cal.exc], ['#24324f', l.cal.off], ['#3a4560', l.cal.none]]
    .map(([c, t]) => `<span><i class="sw" style="background:${c};width:12px;height:12px"></i>${esc(t)}</span>`).join('');
}

