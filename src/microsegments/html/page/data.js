"use strict";
const D = JSON.parse(document.getElementById('ms-data').textContent);
const P = D.params || {};
const TICK = P.tick_s || 20;
const REFH = P.reference_hours || [20, 24];
const HTH = P.hotspot_theta_s ?? 2;   // hotspot threshold, seconds per vehicle
const DAYH = D.day_hours || [6, 21];
// the "Période" buttons, [start, end) hours (an older contract's "day" band is not offered)
const BANDDEF = Object.fromEntries(Object.entries({am: [7, 10], midday: [10, 16], pm: [16, 19], evening: [20, 24]}).map(([k, v]) => [k, (D.bands || {})[k] || v]));
const LOC = {fr: 'fr-BE', en: 'en-GB'};
const LINEH = [6, 23];   // window of the line-level reference

