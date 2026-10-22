// ---------------------------------------------------------------- colours: the prototype's OKLab ramp
// The STIB speed ramp (km/h), interpolated in OKLab: dark red (slow) to dark green (fluid).
const RAMP = [[4.5,'#7a1020'],[7.5,'#e5484d'],[10.5,'#f0a13a'],[13.5,'#b8e27f'],[16.5,'#3fb950'],[21,'#1f8f45'],[27,'#11683e']];
const hex2 = h => [1,3,5].map(i => parseInt(h.slice(i,i+2),16)/255);
const lin = c => c <= .04045 ? c/12.92 : ((c+.055)/1.055)**2.4, gam = c => c <= .0031308 ? 12.92*c : 1.055*c**(1/2.4)-.055;
function toLab([r,g,b]) {
  [r,g,b] = [r,g,b].map(lin);
  const l = Math.cbrt(.4122214708*r+.5363325363*g+.0514459929*b), m = Math.cbrt(.2119034982*r+.6806995451*g+.1073969566*b), s = Math.cbrt(.0883024619*r+.2817188376*g+.6299787005*b);
  return [.2104542553*l+.7936177850*m-.0040720468*s, 1.9779984951*l-2.4285922050*m+.4505937099*s, .0259040371*l+.7827717662*m-.8086757660*s];
}
function fromLab([L,a,b]) {
  const l = (L+.3963377774*a+.2158037573*b)**3, m = (L-.1055613458*a-.0638541728*b)**3, s = (L-.0894841775*a-1.2914855480*b)**3;
  return [4.0767416621*l-3.3077115913*m+.2309699292*s, -1.2684380046*l+2.6097574011*m-.3413193965*s, -.0041960863*l-.7034186147*m+1.7076147010*s]
    .map(c => Math.round(Math.min(1, Math.max(0, gam(c)))*255));
}
const labOf = R => R.map(([v, h]) => [v, toLab(hex2(h))]);
// the colour at f in [0, 1] between stops i and i + 1 of an OKLab ramp
const mixLab = (R, i, f) => `rgb(${fromLab(R[i][1].map((c, k) => c + f * (R[i + 1][1][k] - c))).join(',')})`;
const RAMP_LAB = labOf(RAMP);
function colorOf(v) {
  const R = RAMP_LAB; let i = 0;
  if (v <= R[0][0]) return RAMP[0][1];
  if (v >= R[R.length-1][0]) return RAMP[R.length-1][1];
  while (v > R[i+1][0]) i++;
  return mixLab(R, i, (v - R[i][0]) / (R[i+1][0] - R[i][0]));
}
// ---------------------------------------------------------------- one fixed colour scale for the whole site
// Same constants as microsegments/scale.py CONSTANTS (kept in sync by a test). Colour is a fixed function of the
// value, never computed from the data (no percentile): every line, period, day, hour, band and stop-zone toggle
// shares it. A speed (km/h = 3.6 x length / time per vehicle) is coloured on the STIB ramp (4.5 km/h dark red ..
// 27 km/h dark green); its legend bar is linear in km/h (0 to >= 36 km/h at KTICKS, slow to fast). Gap with the
// evening: km/h lost, linear 0 .. LMAX. Two periods: the gap relative to the reference speed, +-CMPR.
// Why these constants (tram and bus lines, weekdays 2025): between stops the median is 24-36 km/h, the slowest 2 %
// 5-9 km/h.
const SC = {KTICKS: [0, 6, 12, 18, 24, 30, 36], CMPR: 0.25, CRTICKS: [-25, -10, 0, 10, 25], LMAX: 15, LTICKS: [0, 3, 6, 9, 12, 15]};
// no data: flat discreet grey; masked stop zone: grey hatching (SVG pattern #mshatch, legend swatch .sw.hatch)
const NULLC = '#3a4560', MASKC = 'url(#mshatch)';
const kmhColor = k => k == null || !isFinite(k) ? NULLC : colorOf(k);
// km/h lost (0 .. LMAX) on the ramp, from its fastest stop down to its slowest
const lossColor = k => k == null || !isFinite(k) ? NULLC : colorOf(27 - 22.5 * Math.min(k, SC.LMAX) / SC.LMAX);
// Relative speed gaps r = gap / reference speed (-3 km/h is a lot at 10 km/h, noise at 40), diverging ramp: green
// (faster) - grey - red (slower), saturating at +-SC.CMPR. The colour eases in with the square root of the gap (a
// few % already show); the legend axis is linear. pale: an uncertain gap.
const DIV = [[-1, '#11683e'], [-.45, '#3fb950'], [0, '#4a5470'], [.45, '#f0a13a'], [1, '#e5484d']];
const DIV_LAB = labOf(DIV);
const FASTC = '#3fb950', SLOWC = '#e5484d';
function relColor(r, pale) {
  if (r == null || !isFinite(r)) return NULLC;
  let t = -Math.sign(r) * Math.min(1, Math.sqrt(Math.abs(r) / SC.CMPR));
  if (pale) t *= .35;
  let i = 0; while (i < DIV_LAB.length - 2 && t > DIV_LAB[i + 1][0]) i++;
  return mixLab(DIV_LAB, i, Math.min(1, Math.max(0, (t - DIV_LAB[i][0]) / (DIV_LAB[i + 1][0] - DIV_LAB[i][0]))));
}
// km/h of o observations per passage over len metres (time per vehicle = o x TICK s); capped at KMAX (no report at all
// in a short stretch is not an infinite speed)
const KMAX = 80;
// n: the position reports behind o; under MINO the speed is unknown (a fast stretch with 1-2 reports reads 80 km/h)
const MINO = 3;
const kmhOf = (o, len, n) => o == null || !(len > 0) || n != null && n < MINO ? null : Math.min(KMAX, 3.6 * len / Math.max(o * TICK, 1e-9));
// a speed gap b - a and its share of a; never from a missing or capped speed
const gapOf = (a, b) => a != null && b != null && a < KMAX && b < KMAX ? b - a : null;
const relOf = (a, b) => { const g = gapOf(a, b); return g == null || !(a > 0) ? null : g / a; };
const pc = r => r == null ? '–' : signed(100 * r, 0) + ' %';
