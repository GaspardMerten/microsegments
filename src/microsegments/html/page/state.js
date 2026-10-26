// ---------------------------------------------------------------- state
const qs = new URLSearchParams(location.search);
const st = {lang: T[qs.get('lang')] ? qs.get('lang') : (T[document.documentElement.lang] ? document.documentElement.lang : 'fr'),
            dir: null, ver: null, h0: null, h1: null, metric: 'opp', stops: qs.get('stops') !== 'false', days: [], hot: null,
            mode: D.compare ? 'cmp' : 'one', chg: null, view: 'ms', xia: new Set()};
// URL parameters of the display modes (query, or the hash on a page opened from disk):
//   view=ia       opens "Interarrêts" (default: microsegments; the former view=sel opens "Interarrêts" too);
//   xia=A>B,...   the interarrêts left out of the line totals, by their stop ids ("A>B": dirs[].stops[].id of A then B).
// The page keeps both in its own URL (replaceState); embedded, it also posts {ms: 'xia', xia: [ids], view} to the host
// on every change so the host can put them in the URL it builds, and follows the host's {ms: 'xia', xia: [ids]}.
const hq = new URLSearchParams(location.hash.slice(1)), uq = k => hq.get(k) ?? qs.get(k);
// the display mode of a view parameter (null if unknown); the removed "sel" falls back to interarrêts
const viewOf = v => v === 'ms' || v === 'ia' ? v : v === 'sel' ? 'ia' : null;
if (viewOf(uq('view'))) st.view = viewOf(uq('view'));
for (const x of (uq('xia') || '').split(',')) if (x) st.xia.add(x);
function saveUrl() {
  const xs = [...st.xia].join(','), set = q => { xs ? q.set('xia', xs) : q.delete('xia'); st.view !== 'ms' ? q.set('view', st.view) : q.delete('view'); return q; };
  const h = hq.has('xia') || hq.has('view') || location.protocol === 'file:' ? String(set(hq)) : null;
  const q = h == null ? String(set(new URLSearchParams(location.search))) : location.search.slice(1);
  try { history.replaceState(history.state, '', location.pathname + (q ? '?' + q : '') + (h == null ? location.hash : h ? '#' + h : '')); }
  catch (e) { if (h != null) location.hash = h; }
  if (window.parent !== window) try { parent.postMessage({ms: 'xia', xia: [...st.xia], view: st.view}, '*'); } catch (e) {}
}
const tx = () => T[st.lang];
const noun = (pl) => { const n = NOUN[st.lang][D.mode] || NOUN[st.lang]._; return pl ? n[1] : n[0]; };
const nf = (v, d = 1) => v.toLocaleString(LOC[st.lang], {minimumFractionDigits: d, maximumFractionDigits: d});
const nfa = v => nf(v, Math.abs(v) >= 10 ? 0 : 1);
// a speed in km/h: 1 decimal under 10 km/h
const kf = k => k == null ? '–' : k >= KMAX ? '≥ ' + KMAX : nfa(k);
// big counts in words: 845 000; 1,2 million / 12 millions (en: 1.2 million). Grouped by hand: some browsers
// do not group 6-digit numbers in fr-BE. From 10 000, rounded to the thousand.
const grp = n => String(n).replace(/\B(?=(\d{3})+(?!\d))/g, st.lang === 'fr' ? '\u202f' : ',');
const bigNum = v => {
  v = v >= 1e4 ? Math.round(v / 1e3) * 1e3 : Math.round(v);   // round first: 999 600 -> "1 million"
  if (v < 1e6) return grp(v);
  const x = v / 1e6, d = x < 10 ? 1 : 0, r = +x.toFixed(d);
  return nf(r, r % 1 ? d : 0) + (st.lang === 'fr' ? (r >= 2 ? ' millions' : ' million') : ' million');
};
// no sign on a value that rounds to zero (no '−0 s')
const signed = (v, d = 1) => (Math.abs(v) < .5 * 10 ** -d ? '' : v > 0 ? '+' : '−') + nf(Math.abs(v), d);
const esc = s => String(s == null ? '' : s).replace(/[&<>"]/g, c => ({'&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;'}[c]));
// French writes the first day of a month "1er juillet"
const fmtDate = (s, o = {day: 'numeric', month: 'long', year: 'numeric'}) => {
  const d = new Date(s + 'T12:00:00');
  if (st.lang !== 'fr' || o.day !== 'numeric' || !o.month || /^\d/.test(o.month)) return d.toLocaleDateString(LOC[st.lang], o);
  return new Intl.DateTimeFormat(LOC.fr, o).formatToParts(d).map(p => p.type === 'day' && p.value === '1' ? '1er' : p.value).join('');
};
const fmtShort = s => fmtDate(s, {day: '2-digit', month: '2-digit', year: '2-digit'});
const DIRS = D.dirs || [];
const PRESENT = [0,1,2,3,4,5,6].filter(w => (D.period.per_dow || [])[w] > 0);
st.days = PRESENT.slice();
// direction: ?dir= (or the embedding page's own control, see DIR_CONTROL), else the first one
st.dir = DIRS.some(d => String(d.dir) === qs.get('dir')) ? +qs.get('dir') : DIRS.length ? DIRS[0].dir : null;
// contract options for an embedding page: dir_control false hides the page's direction buttons (the
// host drives the direction with postMessage({ms: 'dir', dir})); method false drops the method section
const DIR_CONTROL = D.dir_control !== false, METHOD = D.method !== false;
// host_controls true: the host page already has the days, period mode, language, coverage and footer, so
// the page drops its own (weekday buttons, one/two-period switch, FR/EN, coverage calendar, footer)
const HOST = D.host_controls === true;
// the hours shown: [h0, h1), end excluded; the morning peak by default
const HMIN = D.hours[0], HMAX = D.hours[D.hours.length - 1] + 1;
const setRange = (a, b) => { st.h0 = Math.max(HMIN, Math.min(a, HMAX - 1)); st.h1 = Math.max(st.h0 + 1, Math.min(b, HMAX)); };
if (BANDDEF.am) setRange(...BANDDEF.am); else setRange(HMIN, HMIN + 1);
const dirOf = () => DIRS.find(x => x.dir === st.dir);
// the entry shown: the display version of the direction, or one of its other versions
const entry = () => { const d = dirOf(); if (!d) return null; if (!st.ver || st.ver === d.pattern_uid) return d; return (d.alt || []).find(a => a.pattern_uid === st.ver) || d; };
const isMain = () => { const d = dirOf(); return !st.ver || (d && st.ver === d.pattern_uid); };
const range = () => [st.h0, st.h1];
const hoursIn = ([a, b]) => D.hours.filter(h => h >= a && h < b);
// the band whose hours are exactly the range shown (null otherwise)
const bandOf = ([a, b] = range()) => Object.keys(BANDDEF).find(k => BANDDEF[k][0] === a && BANDDEF[k][1] === b) || null;
const hh = h => String(h % 24).padStart(2, '0');
// "08:00–09:59": the end hour excluded, written as its last minute
const rangeLabel = ([a, b] = range()) => `${hh(a)}:00–${hh(b - 1)}:59`;

