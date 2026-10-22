// ---------------------------------------------------------------- legend
const isStib = () => /STIB|vehicle-distance/i.test(D.source || '');
const stopsHelpText = () => tx().stopsHelp(isStib());
const zoneSw = () => `<span><i class="sw zone"${isCmp() || st.metric === 'ex' ? ` style="background-color:${DIV[2][1]}"` : ''}></i>${esc(tx().legend.zone)}${helpBtn(stopsHelpText())}</span>`;
// stop zones (shown or hidden); interarrêts: the excluded swatch
const zoneOrX = () => isIA() ? `<span><i class="sw" style="background:${XC}"></i>${esc(tx().ia.excl)}</span>` : st.stops ? zoneSw() : `<span><i class="sw hatch"></i>${esc(tx().legend.masked)}</span>`;
// a legend bar: a gradient through the colours cols, its tick labels
const rampBar = (cols, ticks) => `<div class="ramp"><div class="bar" style="background:linear-gradient(90deg,${cols.join(',')})"></div><div class="tk">${ticks.join('')}</div></div>`;
// The speed bar is linear in km/h, slow (left) to fast (right), the same ramp as the map; the gap with the evening uses the km/h gap bar.
function legend() {
  if (isCmp()) return cmpLegend();
  const l = tx().legend, m = st.metric, n = SC.KTICKS.length;
  let ramp;
  if (m === 'ex') {
    const M = SC.LMAX;
    ramp = rampBar(Array.from({length: 31}, (_, i) => lossColor(M * i / 30)), SC.LTICKS.map(k => `<span style="left:${k / M * 100}%">${k === M ? '≥ ' : ''}${nf(k, 0)}</span>`));
  } else {
    const K1 = SC.KTICKS[n - 1];
    ramp = rampBar(Array.from({length: 37}, (_, i) => colorOf(K1 * i / 36)), SC.KTICKS.map((k, i) => `<span style="left:${k / K1 * 100}%">${i === n - 1 ? '≥ ' : ''}${nf(k, 0)}</span>`));
  }
  const evening = m === 'ex' && st.h0 >= REFH[0] && st.h1 <= REFH[1];
  const txt = esc(l.lab[m](noun(false)) + (evening ? l.evening : '')) + helpBtn(isIA() && m === 'opp' ? tx().ia.help(noun(false)) : l.help[m](noun(false), noun(true)));
  $('legend').innerHTML = ramp + `<span class="txt">${txt}</span><span><i class="sw" style="background:${NULLC}"></i>${l.nodata}</span>` + zoneOrX() +
    (hotOfDir().length && isMain() ? `<span><i class="hk"></i>${l.hot}</span>` : '');
  $('mxlegend').innerHTML = $('legend').innerHTML;
}

