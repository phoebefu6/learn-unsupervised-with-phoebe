/* cluster-live.js - the Mango Lane segment lab (learn-unsupervised-with-phoebe)
   REAL algorithms running in your browser: k-means++ (Lloyd), DBSCAN, spherical-GMM EM,
   PCA (Jacobi eigen), silhouette, and association-rule mining - all on a deterministic
   seeded sample of Mango Lane customers. The data is simulated (that is what lets us
   show a "match to planted truth" score no real project ever gets); the algorithms and
   every metric are genuinely computed. No dependencies. Modes via data-mode attr:
   scale | iterate | pick-k | algos | pca | basket | noise | capstone */
(function () {
  "use strict";
  var mount = document.getElementById("cluster-live");
  if (!mount) return;
  var modeAttr = mount.getAttribute("data-mode");
  var MODE = (modeAttr === null || modeAttr === "") ? "pick-k" : modeAttr;

  /* ---------- deterministic PRNG ---------- */
  function mulberry32(a) { return function () {
    a |= 0; a = a + 0x6D2B79F5 | 0;
    var t = Math.imul(a ^ a >>> 15, 1 | a);
    t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t;
    return ((t ^ t >>> 14) >>> 0) / 4294967296; }; }
  function gauss(rnd) { var u = 0, v = 0; while (u === 0) u = rnd(); while (v === 0) v = rnd();
    return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v); }

  /* ---------- the Mango Lane sample: 300 customers, 5 planted segments ---------- */
  /* features: recency (days since last order), frequency (orders/yr), monetary (avg order in
     CENTS - the export came out of the billing system, and billing systems speak cents).
     Params tuned by in-browser sweep 2026-08-11: raw 5-means purity .70 / scaled 1.00,
     silhouette-by-k peaks at k=5 (.68), k=12 falls to .34. */
  var SEGS = [
    { name: "VIP regulars",       n: 55, c: [25, 22, 11500], s: [8, 1.8, 900] },
    { name: "Steady mid-basket",  n: 90, c: [30, 9, 7000],   s: [10, 1.3, 850] },
    { name: "One-time gifters",   n: 60, c: [35, 1.2, 9800], s: [12, 0.35, 850] },
    { name: "Fresh arrivals",     n: 45, c: [20, 3, 4500],   s: [8, 0.6, 750] },
    { name: "Lapsed big spenders",n: 50, c: [225, 6, 9300],  s: [25, 0.9, 900] }
  ];
  function makeData() {
    var rnd = mulberry32(42), X = [], truth = [];
    SEGS.forEach(function (s, si) {
      for (var i = 0; i < s.n; i++) {
        X.push([Math.max(1, s.c[0] + gauss(rnd) * s.s[0]),
                Math.max(0.2, s.c[1] + gauss(rnd) * s.s[1]),
                Math.max(1500, s.c[2] + gauss(rnd) * s.s[2])]);
        truth.push(si);
      }
    });
    return { X: X, truth: truth };
  }
  var D = makeData();
  function makeNoise() { var rnd = mulberry32(7), X = [];
    for (var i = 0; i < 300; i++) X.push([1 + rnd() * 350, 0.2 + rnd() * 25, 1500 + rnd() * 11000]);
    return X; }
  /* two-moons toy for the algos mode (deterministic) */
  function makeMoons() { var rnd = mulberry32(11), X = [], lab = [];
    for (var i = 0; i < 120; i++) { var t = Math.PI * rnd();
      X.push([Math.cos(t) + gauss(rnd) * .07, Math.sin(t) + gauss(rnd) * .07]); lab.push(0); }
    for (var j = 0; j < 120; j++) { var u = Math.PI * rnd();
      X.push([1 - Math.cos(u) + gauss(rnd) * .07, .5 - Math.sin(u) + gauss(rnd) * .07]); lab.push(1); }
    return { X: X, truth: lab }; }

  /* ---------- math ---------- */
  function scaleZ(X) { var d = X[0].length, m = [], sd = [];
    for (var j = 0; j < d; j++) { var s = 0; X.forEach(function (r) { s += r[j]; }); m[j] = s / X.length;
      var v = 0; X.forEach(function (r) { v += (r[j] - m[j]) * (r[j] - m[j]); }); sd[j] = Math.sqrt(v / X.length) || 1; }
    return X.map(function (r) { return r.map(function (x, j) { return (x - m[j]) / sd[j]; }); }); }
  function dist2(a, b) { var s = 0; for (var i = 0; i < a.length; i++) s += (a[i] - b[i]) * (a[i] - b[i]); return s; }

  function kmeansInit(X, k, rnd) { /* k-means++ */
    var C = [X[Math.floor(rnd() * X.length)].slice()];
    while (C.length < k) {
      var d = X.map(function (x) { var m = Infinity; C.forEach(function (c) { m = Math.min(m, dist2(x, c)); }); return m; });
      var sum = d.reduce(function (a, b) { return a + b; }, 0), r = rnd() * sum, acc = 0, idx = 0;
      for (var i = 0; i < d.length; i++) { acc += d[i]; if (acc >= r) { idx = i; break; } }
      C.push(X[idx].slice());
    }
    return C; }
  function kmeansStep(X, C) {
    var k = C.length, assign = X.map(function (x) {
      var bi = 0, bd = Infinity;
      for (var c = 0; c < k; c++) { var dd = dist2(x, C[c]); if (dd < bd) { bd = dd; bi = c; } }
      return bi; });
    var NC = [], cnt = [];
    for (var c = 0; c < k; c++) { NC.push(new Array(X[0].length).fill(0)); cnt.push(0); }
    X.forEach(function (x, i) { var a = assign[i]; cnt[a]++; x.forEach(function (v, j) { NC[a][j] += v; }); });
    for (var c2 = 0; c2 < k; c2++) { if (cnt[c2] === 0) NC[c2] = C[c2].slice();
      else NC[c2] = NC[c2].map(function (v) { return v / cnt[c2]; }); }
    return { assign: assign, C: NC }; }
  function inertia(X, C, assign) { var s = 0; X.forEach(function (x, i) { s += dist2(x, C[assign[i]]); }); return s; }
  function kmeansOnce(X, k, seed, plain) { /* plain=true -> random init (the unlucky demo) */
    var rnd = mulberry32(seed === undefined ? 1 : seed);
    var C = plain ? (function () { var c = []; var used = {};
        while (c.length < k) { var i = Math.floor(rnd() * X.length); if (!used[i]) { used[i] = 1; c.push(X[i].slice()); } }
        return c; })() : kmeansInit(X, k, rnd);
    var assign = null;
    for (var it = 0; it < 60; it++) {
      var st = kmeansStep(X, C);
      if (assign && st.assign.every(function (a, i) { return a === assign[i]; })) { assign = st.assign; C = st.C; return { assign: assign, C: C, iters: it + 1, inertia: inertia(X, C, assign) }; }
      assign = st.assign; C = st.C;
    }
    return { assign: assign, C: C, iters: 60, inertia: inertia(X, C, assign) }; }
  function kmeans(X, k, seed, plain) { /* best of 5 restarts by inertia (sklearn's old n_init spirit) */
    if (plain) return kmeansOnce(X, k, seed, true);
    var best = null;
    for (var s = 1; s <= 5; s++) { var r = kmeansOnce(X, k, s); if (!best || r.inertia < best.inertia) best = r; }
    return best; }

  function silhouette(X, labels) {
    var n = X.length, ks = {}; labels.forEach(function (l) { ks[l] = (ks[l] || 0) + 1; });
    var keys = Object.keys(ks); if (keys.length < 2) return 0;
    var total = 0, counted = 0;
    for (var i = 0; i < n; i++) {
      if (labels[i] === -1) continue; /* DBSCAN noise excluded */
      var sums = {}, cnts = {};
      for (var j = 0; j < n; j++) { if (i === j || labels[j] === -1) continue;
        var l = labels[j], dd = Math.sqrt(dist2(X[i], X[j]));
        sums[l] = (sums[l] || 0) + dd; cnts[l] = (cnts[l] || 0) + 1; }
      var own = labels[i];
      if (!cnts[own]) continue;
      var a = sums[own] / cnts[own], b = Infinity;
      keys.forEach(function (kk) { if (+kk !== own && cnts[kk]) b = Math.min(b, sums[kk] / cnts[kk]); });
      if (b === Infinity) continue;
      total += (b - a) / Math.max(a, b); counted++;
    }
    return counted ? total / counted : 0; }

  function purity(labels, truth) { /* majority-truth share per cluster, weighted */
    var groups = {};
    labels.forEach(function (l, i) { (groups[l] = groups[l] || []).push(truth[i]); });
    var ok = 0, n = 0;
    Object.keys(groups).forEach(function (g) { if (+g === -1) return;
      var cnt = {}; groups[g].forEach(function (t) { cnt[t] = (cnt[t] || 0) + 1; });
      ok += Math.max.apply(null, Object.keys(cnt).map(function (k) { return cnt[k]; }));
      n += groups[g].length; });
    return n ? ok / n : 0; }

  function dbscan(X, eps, minPts) {
    var n = X.length, labels = new Array(n).fill(undefined), cid = -1, e2 = eps * eps;
    function nbrs(i) { var r = []; for (var j = 0; j < n; j++) if (dist2(X[i], X[j]) <= e2) r.push(j); return r; }
    for (var i = 0; i < n; i++) {
      if (labels[i] !== undefined) continue;
      var N = nbrs(i);
      if (N.length < minPts) { labels[i] = -1; continue; }
      cid++; labels[i] = cid; var queue = N.slice();
      while (queue.length) { var q = queue.shift();
        if (labels[q] === -1) labels[q] = cid;
        if (labels[q] !== undefined) continue;
        labels[q] = cid; var NQ = nbrs(q);
        if (NQ.length >= minPts) queue = queue.concat(NQ); }
    }
    return labels; }

  function gmm(X, k, seed) { /* spherical EM */
    var rnd = mulberry32(seed || 3), d = X[0].length;
    var km = kmeans(X, k, seed || 3);
    var mu = km.C.map(function (c) { return c.slice(); }), va = [], pi = [];
    for (var c = 0; c < k; c++) { va.push(1); pi.push(1 / k); }
    var R = [];
    for (var it = 0; it < 40; it++) {
      R = X.map(function (x) {
        var w = [], s = 0;
        for (var c = 0; c < k; c++) { var p = pi[c] * Math.exp(-dist2(x, mu[c]) / (2 * va[c])) / Math.pow(2 * Math.PI * va[c], d / 2); w.push(p); s += p; }
        return w.map(function (p) { return s > 0 ? p / s : 1 / k; }); });
      for (var c2 = 0; c2 < k; c2++) {
        var nk = 0, nm = new Array(d).fill(0);
        R.forEach(function (r, i) { nk += r[c2]; X[i].forEach(function (v, j) { nm[j] += r[c2] * v; }); });
        if (nk < 1e-6) continue;
        mu[c2] = nm.map(function (v) { return v / nk; });
        var nv = 0; R.forEach(function (r, i) { nv += r[c2] * dist2(X[i], mu[c2]); });
        va[c2] = Math.max(0.05, nv / (nk * d)); pi[c2] = nk / X.length;
      }
    }
    return { resp: R, mu: mu, hard: R.map(function (r) { return r.indexOf(Math.max.apply(null, r)); }) }; }

  function pca2(X) { /* real PCA via Jacobi on the covariance of standardized X (3x3 or 2x2) */
    var Z = scaleZ(X), n = Z.length, d = Z[0].length;
    var C = [];
    for (var i = 0; i < d; i++) { C.push(new Array(d).fill(0)); }
    Z.forEach(function (r) { for (var a = 0; a < d; a++) for (var b = 0; b < d; b++) C[a][b] += r[a] * r[b] / n; });
    var V = []; for (var i2 = 0; i2 < d; i2++) { V.push(new Array(d).fill(0)); V[i2][i2] = 1; }
    for (var sweep = 0; sweep < 30; sweep++) {
      for (var p = 0; p < d - 1; p++) for (var q = p + 1; q < d; q++) {
        if (Math.abs(C[p][q]) < 1e-10) continue;
        var th = 0.5 * Math.atan2(2 * C[p][q], C[q][q] - C[p][p]);
        var c = Math.cos(th), s = Math.sin(th);
        for (var r2 = 0; r2 < d; r2++) { var cp = C[r2][p], cq = C[r2][q];
          C[r2][p] = c * cp - s * cq; C[r2][q] = s * cp + c * cq; }
        for (var r3 = 0; r3 < d; r3++) { var pp = C[p][r3], qq = C[q][r3];
          C[p][r3] = c * pp - s * qq; C[q][r3] = s * pp + c * qq; }
        for (var r4 = 0; r4 < d; r4++) { var vp = V[r4][p], vq = V[r4][q];
          V[r4][p] = c * vp - s * vq; V[r4][q] = s * vp + c * vq; }
      }
    }
    var eig = []; for (var e = 0; e < d; e++) eig.push({ val: C[e][e], vec: V.map(function (row) { return row[e]; }) });
    eig.sort(function (a, b) { return b.val - a.val; });
    var tot = eig.reduce(function (a, b) { return a + b.val; }, 0);
    var proj = Z.map(function (r) { return [
      r.reduce(function (s2, v, j) { return s2 + v * eig[0].vec[j]; }, 0),
      r.reduce(function (s2, v, j) { return s2 + v * eig[1].vec[j]; }, 0)]; });
    return { proj: proj, explained: eig.map(function (e2) { return e2.val / tot; }), vecs: eig }; }

  /* ---------- baskets for association rules (b8) ---------- */
  var PRODUCTS = ["midi dress", "leather belt", "white sneakers", "crew socks", "denim jacket",
    "silk scarf", "tote bag", "gold hoops", "linen shirt", "wide-leg pants", "rain coat", "beanie"];
  var PAIRS = [[0, 1, .62], [2, 3, .55], [4, 11, .3], [8, 9, .5], [5, 7, .42]]; /* planted co-buys */
  function makeBaskets() { var rnd = mulberry32(99), B = [];
    for (var i = 0; i < 500; i++) { var b = {};
      var base = 1 + Math.floor(rnd() * 3);
      for (var j = 0; j < base; j++) b[Math.floor(rnd() * PRODUCTS.length)] = 1;
      PAIRS.forEach(function (p) { if (b[p[0]] && rnd() < p[2]) b[p[1]] = 1; if (b[p[1]] && rnd() < p[2] * .5) b[p[0]] = 1; });
      B.push(Object.keys(b).map(Number)); }
    return B; }
  function mineRules(B, minSup) {
    var n = B.length, cnt1 = {}, cnt2 = {};
    B.forEach(function (b) { b.forEach(function (a) { cnt1[a] = (cnt1[a] || 0) + 1; });
      for (var i = 0; i < b.length; i++) for (var j = 0; j < b.length; j++) if (i !== j)
        cnt2[b[i] + "-" + b[j]] = (cnt2[b[i] + "-" + b[j]] || 0) + 1; });
    var rules = [];
    Object.keys(cnt2).forEach(function (k) { var ab = k.split("-").map(Number), a = ab[0], c = ab[1];
      var sup = cnt2[k] / n; if (sup < minSup) return;
      var conf = cnt2[k] / cnt1[a], lift = conf / (cnt1[c] / n);
      rules.push({ a: PRODUCTS[a], c: PRODUCTS[c], sup: sup, conf: conf, lift: lift }); });
    rules.sort(function (x, y) { return y.lift - x.lift; });
    return rules; }

  /* ---------- theming + shared UI ---------- */
  var css = getComputedStyle(document.documentElement);
  function v(nm, fb) { var x = css.getPropertyValue(nm).trim(); return x || fb; }
  var TH = { accent: v("--indigo", "#7E22CE"), deep: v("--indigo-deep", "#581C87"),
    mid: v("--indigo-mid", "#A855F7"), soft: v("--indigo-soft", "#D8B4FE"), tint: v("--indigo-50", "#F5EDFC"),
    ink: v("--ink", "#221A2E"), muted: v("--muted", "#6B647D"), hairline: v("--hairline", "#EAE5F2"),
    warm: v("--amber", "#D97706"), warmTint: v("--amber-50", "#FBF0DE"), warmInk: v("--amber-ink", "#3F2A08"),
    red: "#991B1B", redTint: "#FEF2F2" };
  var PAL = [TH.accent, TH.warm, "#0E7490", "#15803D", "#BE185D", "#4338CA", "#B45309", "#0F766E", "#7F1D1D", "#3F6212", "#6D28D9", "#A21CAF"];

  var style = document.createElement("style");
  style.textContent =
    "#cluster-live{border:1px solid " + TH.hairline + ";border-radius:14px;background:#fff;padding:1rem 1.1rem 1.2rem;font-feature-settings:'tnum'}" +
    "#cluster-live .cl-hint{font-size:.85rem;color:" + TH.muted + ";margin:0 0 .7rem}" +
    "#cluster-live .cl-row{display:flex;gap:1rem;flex-wrap:wrap;align-items:flex-start}" +
    "#cluster-live .cl-canvas{flex:1 1 320px;min-width:280px}" +
    "#cluster-live canvas{width:100%;height:auto;border:1px solid " + TH.hairline + ";border-radius:10px;background:" + TH.tint + "}" +
    "#cluster-live .cl-side{flex:1 1 230px;min-width:220px}" +
    "#cluster-live .cl-btn{border:none;border-radius:999px;padding:.4rem 1rem;font:inherit;font-weight:800;background:" + TH.accent + ";color:#fff;cursor:pointer;margin:.15rem .3rem .15rem 0}" +
    "#cluster-live .cl-btn:hover{background:" + TH.deep + "}" +
    "#cluster-live .cl-btn.warn{background:" + TH.warm + "}" +
    "#cluster-live .cl-lever{border:1.5px solid " + TH.hairline + ";border-radius:999px;padding:.32rem .8rem;font:inherit;font-size:.82rem;font-weight:700;background:#fff;color:" + TH.muted + ";cursor:pointer;margin:.15rem .3rem .15rem 0}" +
    "#cluster-live .cl-lever.on{border-color:" + TH.warm + ";background:" + TH.warmTint + ";color:" + TH.warmInk + "}" +
    "#cluster-live .cl-stats{display:flex;gap:.6rem;flex-wrap:wrap;margin:.6rem 0}" +
    "#cluster-live .cl-stat{flex:1 1 100px;border:1px solid " + TH.hairline + ";border-radius:10px;padding:.45rem .6rem;text-align:center}" +
    "#cluster-live .cl-stat b{display:block;font-size:1.25rem;font-weight:800;color:" + TH.deep + "}" +
    "#cluster-live .cl-stat span{font-size:.66rem;color:" + TH.muted + ";font-weight:700;text-transform:uppercase;letter-spacing:.05em}" +
    "#cluster-live .cl-stat.bad b{color:" + TH.red + "}" +
    "#cluster-live table{width:100%;border-collapse:collapse;font-size:.82rem;margin-top:.5rem}" +
    "#cluster-live th{font-size:.68rem;text-transform:uppercase;letter-spacing:.05em;color:" + TH.muted + ";text-align:left;padding:.3rem .45rem;border-bottom:1px solid " + TH.hairline + "}" +
    "#cluster-live td{padding:.34rem .45rem;border-bottom:1px solid " + TH.hairline + "}" +
    "#cluster-live .cl-read{margin-top:.7rem;font-size:.85rem;color:" + TH.ink + ";background:#fff;border-left:3px solid " + TH.warm + ";padding:.5rem .8rem;background:" + TH.warmTint + ";border-radius:0 8px 8px 0}" +
    "#cluster-live input[type=range]{width:100%;accent-color:" + TH.accent + "}" +
    "#cluster-live .cl-klabel{font-weight:800;color:" + TH.deep + ";font-size:.9rem}" +
    "#cluster-live .cl-rail{font-size:.73rem;color:" + TH.muted + ";margin-top:.8rem;padding-top:.55rem;border-top:1px dashed " + TH.hairline + "}";
  document.head.appendChild(style);

  function el(tag, cls, html) { var e = document.createElement(tag); if (cls) e.className = cls; if (html != null) e.innerHTML = html; return e; }
  function fmtPct(x) { return (100 * x).toFixed(0) + "%"; }
  function fmt2(x) { return x.toFixed(2); }

  function drawScatter(cv, pts, labels, centroids, opts) {
    opts = opts || {};
    var ctx = cv.getContext("2d"), W = cv.width, H = cv.height;
    ctx.clearRect(0, 0, W, H);
    var xs = pts.map(function (p) { return p[0]; }), ys = pts.map(function (p) { return p[1]; });
    var x0 = Math.min.apply(null, xs), x1 = Math.max.apply(null, xs), y0 = Math.min.apply(null, ys), y1 = Math.max.apply(null, ys);
    var px = function (x) { return 30 + (x - x0) / (x1 - x0 || 1) * (W - 50); };
    var py = function (y) { return H - 26 - (y - y0) / (y1 - y0 || 1) * (H - 46); };
    pts.forEach(function (p, i) {
      var l = labels ? labels[i] : 0;
      ctx.beginPath(); ctx.arc(px(p[0]), py(p[1]), 3.2, 0, 7);
      if (l === -1) { ctx.fillStyle = "#B6AFC4"; ctx.fill(); ctx.strokeStyle = TH.muted; ctx.stroke(); }
      else { ctx.fillStyle = PAL[l % PAL.length]; ctx.globalAlpha = .75; ctx.fill(); ctx.globalAlpha = 1; }
    });
    if (centroids) centroids.forEach(function (c, ci) {
      ctx.beginPath(); ctx.arc(px(c[0]), py(c[1]), 8, 0, 7);
      ctx.fillStyle = "#fff"; ctx.fill(); ctx.lineWidth = 3; ctx.strokeStyle = PAL[ci % PAL.length]; ctx.stroke(); ctx.lineWidth = 1;
    });
    ctx.fillStyle = TH.muted; ctx.font = "600 11px Inter, sans-serif";
    ctx.fillText(opts.xl || "", W / 2 - 30, H - 8);
    ctx.save(); ctx.translate(12, H / 2 + 30); ctx.rotate(-Math.PI / 2); ctx.fillText(opts.yl || "", 0, 0); ctx.restore();
  }
  function drawCurve(cv, xs, ys, opts) {
    var ctx = cv.getContext("2d"), W = cv.width, H = cv.height;
    ctx.clearRect(0, 0, W, H);
    var y0 = Math.min.apply(null, ys), y1 = Math.max.apply(null, ys); if (y1 - y0 < 1e-9) y1 = y0 + 1;
    var px = function (i) { return 34 + i / (xs.length - 1) * (W - 54); };
    var py = function (y) { return H - 28 - (y - y0) / (y1 - y0) * (H - 52); };
    ctx.strokeStyle = TH.accent; ctx.lineWidth = 2.5; ctx.beginPath();
    ys.forEach(function (y, i) { i ? ctx.lineTo(px(i), py(y)) : ctx.moveTo(px(i), py(y)); }); ctx.stroke();
    ctx.fillStyle = TH.deep;
    ys.forEach(function (y, i) { ctx.beginPath(); ctx.arc(px(i), py(y), 3.6, 0, 7); ctx.fill(); });
    if (opts && opts.mark !== undefined) { var mi = xs.indexOf(opts.mark);
      if (mi >= 0) { ctx.beginPath(); ctx.arc(px(mi), py(ys[mi]), 8, 0, 7); ctx.strokeStyle = TH.warm; ctx.lineWidth = 3; ctx.stroke(); } }
    ctx.fillStyle = TH.muted; ctx.font = "600 11px Inter, sans-serif";
    xs.forEach(function (x, i) { ctx.fillText(String(x), px(i) - 4, H - 10); });
    ctx.fillText(opts && opts.yl || "", 6, 14);
  }

  var SEG_ADVICE = ["win-back push", "upsell nudge", "gift reminder", "welcome flow", "VIP perks"];
  function segTable(X, labels, k) {
    var rows = [];
    for (var c = 0; c < k; c++) {
      var idx = []; labels.forEach(function (l, i) { if (l === c) idx.push(i); });
      if (!idx.length) { rows.push({ n: 0 }); continue; }
      var m = [0, 0, 0]; idx.forEach(function (i) { for (var j = 0; j < 3; j++) m[j] += X[i][j]; });
      rows.push({ n: idx.length, r: m[0] / idx.length, f: m[1] / idx.length, mo: m[2] / idx.length });
    }
    return rows; }

  /* ---------- render per mode ---------- */
  var Z = scaleZ(D.X);
  var P = pca2(D.X);

  mount.innerHTML = "";
  var hintText = {
    "scale": "The scatter shows the RAW customer pile: recency in days against average order size - which the billing export delivered in CENTS (4,500-12,000). Run 5-means, read the truth-match. Then flip the scaling lever and run again.",
    "iterate": "Scaled features, k=5, k-means++ start. Step through Lloyd's loop and watch the centroids settle and inertia fall. Then try the unlucky random start.",
    "pick-k": "Slide k, run for every k in one go, and read both curves. Then press the trap button and read the segment table like a marketer would.",
    "algos": "A shape k-means cannot see: two interleaved moons. Run each algorithm on the same points.",
    "pca": "Real PCA on the standardized 3-feature pile. Two components carry most of the variance - the scatter you have been reading all course IS this projection.",
    "basket": "500 Mango Lane baskets, mined live. Rules ranked by lift - how much more often the pair co-occurs than chance.",
    "noise": "This pile is pure uniform noise - no segments were planted. k-means will still answer. Read the metrics, not the colors.",
    "capstone": "The whole project in four presses: raw, scaled, best-k, named. Watch both numbers climb - and know which one a real project never gets."
  };
  mount.appendChild(el("p", "cl-hint", hintText[MODE] || ""));
  var row = el("div", "cl-row"); mount.appendChild(row);
  var left = el("div", "cl-canvas"); row.appendChild(left);
  var cv = document.createElement("canvas"); cv.width = 520; cv.height = 360; left.appendChild(cv);
  var side = el("div", "cl-side"); row.appendChild(side);
  var readout = el("div", "cl-read", "Press a button to run.");
  mount.appendChild(readout);
  mount.appendChild(el("div", "cl-rail", "Honesty rail: the customers are simulated (that is the only reason a \"truth match\" score can exist - real projects never get one); every algorithm and metric on this page runs for real in your browser."));

  function stats(items) { var s = el("div", "cl-stats");
    items.forEach(function (it) { var b = el("div", "cl-stat" + (it.bad ? " bad" : ""));
      b.appendChild(el("b", null, it.v)); b.appendChild(el("span", null, it.l)); s.appendChild(b); });
    return s; }

  if (MODE === "scale") {
    var scaled = false;
    var lever = el("button", "cl-lever", "⚖️ Scale features · off");
    lever.onclick = function () { scaled = !scaled; lever.className = "cl-lever" + (scaled ? " on" : ""); lever.textContent = "⚖️ Scale features · " + (scaled ? "ON" : "off"); };
    var run = el("button", "cl-btn", "▶ Run 5-means");
    side.appendChild(lever); side.appendChild(run); var box = el("div"); side.appendChild(box);
    drawScatter(cv, D.X.map(function (r) { return [r[0], r[2]]; }), null, null, { xl: "recency (days)", yl: "avg order (cents)" });
    run.onclick = function () {
      var Xuse = scaled ? Z : D.X;
      var km = kmeans(Xuse, 5, 1);
      var sil = silhouette(Xuse, km.assign), pur = purity(km.assign, D.truth);
      drawScatter(cv, D.X.map(function (r) { return [r[0], r[2]]; }), km.assign, null, { xl: "recency (days)", yl: "avg order (cents)" });
      box.innerHTML = ""; box.appendChild(stats([
        { v: fmt2(sil), l: "silhouette" },
        { v: fmtPct(pur), l: "truth match", bad: pur < .85 }]));
      readout.innerHTML = scaled
        ? "<b>Scaled:</b> every feature speaks with the same voice, and the five planted segments come back exactly - 100% truth match. One preprocessing line did more than any algorithm choice in this course."
        : "<b>Raw units:</b> the cents column runs 4,500-12,000 while frequency runs 1-22, so distance IS money and nothing else. k-means just drew wallet bands: one-time gifters and lapsed regulars - identical spend, opposite campaigns - land in the same segment. The classic silent failure, courtesy of a unit choice.";
    };
  }

  if (MODE === "iterate") {
    var C = null, it = 0, plain = false;
    var step = el("button", "cl-btn", "⏭ Step");
    var reset = el("button", "cl-lever", "↺ Reset (k-means++)");
    var unlucky = el("button", "cl-lever", "🎲 Unlucky random start");
    side.appendChild(step); side.appendChild(reset); side.appendChild(unlucky);
    var box2 = el("div"); side.appendChild(box2);
    var proj = P.proj;
    function centProj(C) { /* project centroids using same PCA vecs (centroids are in scaled space) */
      return C.map(function (c) { return [
        c.reduce(function (s, vv, j) { return s + vv * P.vecs[0].vec[j]; }, 0),
        c.reduce(function (s, vv, j) { return s + vv * P.vecs[1].vec[j]; }, 0)]; }); }
    function begin(pl) { plain = pl; it = 0;
      var rnd = mulberry32(pl ? 13 : 1);
      C = pl ? (function () { var c = [], used = {}; while (c.length < 5) { var i = Math.floor(rnd() * Z.length); if (!used[i]) { used[i] = 1; c.push(Z[i].slice()); } } return c; })() : kmeansInit(Z, 5, rnd);
      drawScatter(cv, proj, null, centProj(C), { xl: "PC1", yl: "PC2" });
      box2.innerHTML = ""; readout.innerHTML = "Centroids placed (" + (pl ? "random - two landed in the same blob" : "k-means++ spread") + "). Step to run one assign-update round."; }
    step.onclick = function () { if (!C) { begin(false); return; }
      var st = kmeansStep(Z, C); C = st.C; it++;
      var inn = inertia(Z, C, st.assign);
      drawScatter(cv, proj, st.assign, centProj(C), { xl: "PC1", yl: "PC2" });
      box2.innerHTML = ""; box2.appendChild(stats([
        { v: String(it), l: "iterations" }, { v: inn.toFixed(0), l: "inertia (falling)" },
        { v: fmtPct(purity(st.assign, D.truth)), l: "truth match" }]));
      readout.innerHTML = it < 4 ? "<b>Round " + it + ":</b> every customer joins its nearest centroid, then each centroid moves to its members' average. Inertia can only fall." :
        "<b>Settled or nearly:</b> assignments stop changing and the loop ends. " + (plain ? "The unlucky start locked into a worse local optimum - inertia stuck higher than the k-means++ run. This is why initialization matters (and why sklearn's modern default is ONE k-means++ run, not ten random ones)." : "k-means++ starts spread out, so the loop usually lands well in a handful of rounds."); };
    reset.onclick = function () { begin(false); };
    unlucky.onclick = function () { begin(true); };
    begin(false);
  }

  if (MODE === "pick-k") {
    var ks = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12];
    var slider = el("input"); slider.type = "range"; slider.min = 2; slider.max = 12; slider.value = 5;
    var klab = el("div", "cl-klabel", "k = 5");
    slider.oninput = function () { klab.textContent = "k = " + slider.value; };
    var runAll = el("button", "cl-btn", "▶ Run every k");
    var trap = el("button", "cl-btn warn", "✨ More segments! (k=12)");
    side.appendChild(klab); side.appendChild(slider); side.appendChild(runAll); side.appendChild(trap);
    var box3 = el("div"); side.appendChild(box3);
    var cv2 = document.createElement("canvas"); cv2.width = 520; cv2.height = 200; left.appendChild(cv2);
    var tbl = el("div"); mount.insertBefore(tbl, readout);
    var cache = {};
    function runK(k) { if (!cache[k]) { var km = kmeans(Z, k, 1);
        cache[k] = { km: km, sil: silhouette(Z, km.assign), pur: purity(km.assign, D.truth) }; } return cache[k]; }
    function show(k, trapMode) {
      var r = runK(k);
      drawScatter(cv, P.proj, r.km.assign, null, { xl: "PC1", yl: "PC2" });
      var sils = ks.map(function (kk) { return runK(kk).sil; });
      drawCurve(cv2, ks, sils, { yl: "silhouette by k", mark: k });
      box3.innerHTML = ""; box3.appendChild(stats([
        { v: fmt2(r.sil), l: "silhouette", bad: trapMode }, { v: fmtPct(r.pur), l: "truth match" },
        { v: String(k), l: "segments" }]));
      var rows = segTable(D.X, r.km.assign, k);
      var h = "<table><tr><th>segment</th><th>size</th><th>recency</th><th>freq/yr</th><th>avg $</th><th>a campaign?</th></tr>";
      rows.forEach(function (rr, i) { if (!rr.n) return;
        h += "<tr><td><b style='color:" + PAL[i % PAL.length] + "'>■</b> " + (i + 1) + "</td><td>" + rr.n + "</td><td>" + rr.r.toFixed(0) + "d</td><td>" + rr.f.toFixed(1) + "</td><td>$" + (rr.mo / 100).toFixed(0) + "</td><td>" + (rr.n < 18 ? "too small to fund" : SEG_ADVICE[i % SEG_ADVICE.length]) + "</td></tr>"; });
      tbl.innerHTML = h + "</table>";
      var best = ks[sils.indexOf(Math.max.apply(null, sils))];
      readout.innerHTML = trapMode
        ? "<b>The trap:</b> twelve segments LOOKS more sophisticated - but the silhouette fell off the cliff, half the segments are too small to fund a campaign, and several differ by nothing a marketer can act on. More clusters is not more insight; it is the same 300 people cut thinner."
        : "<b>k=" + k + ":</b> silhouette " + fmt2(r.sil) + ". The curve peaks at k=" + best + " - the structure that is actually in the data. (The elbow plot on inertia is the famous heuristic, but it bends ambiguously even on noise - trust silhouette plus the actionability read.)";
    }
    runAll.onclick = function () { show(+slider.value, false); };
    trap.onclick = function () { slider.value = 12; klab.textContent = "k = 12"; show(12, true); };
  }

  if (MODE === "algos") {
    var M = makeMoons(); var MZ = scaleZ(M.X);
    var bkm = el("button", "cl-btn", "▶ k-means (k=2)");
    var bdb = el("button", "cl-btn", "▶ DBSCAN");
    var bgm = el("button", "cl-btn", "▶ GMM (soft)");
    side.appendChild(bkm); side.appendChild(bdb); side.appendChild(bgm);
    var box4 = el("div"); side.appendChild(box4);
    drawScatter(cv, M.X, null, null, { xl: "", yl: "" });
    function report(labels, name, msg, extra) {
      drawScatter(cv, M.X, labels, null, {});
      var pur = purity(labels, M.truth);
      box4.innerHTML = ""; box4.appendChild(stats([{ v: fmtPct(pur), l: "truth match", bad: pur < .9 },
        { v: String(new Set(labels.filter(function (l) { return l >= 0; })).size), l: "clusters found" },
        { v: String(labels.filter(function (l) { return l === -1; }).length), l: "noise points" }].concat(extra || [])));
      readout.innerHTML = "<b>" + name + ":</b> " + msg; }
    bkm.onclick = function () { var km = kmeans(MZ, 2, 1);
      report(km.assign, "k-means", "it drew a straight border through both moons - k-means can only make round, evenly-sized territories (it minimizes distance to a center point). The truth match says it plainly."); };
    bdb.onclick = function () { var lab = dbscan(MZ, 0.32, 5);
      report(lab, "DBSCAN", "no k was given - it followed density and traced each moon's actual shape (it would park any straggler as noise, label -1; this tidy sample has none). The price: two knobs (eps, min_samples) that need tuning - shrink eps here and a moon shatters - and no way to assign a brand-new point without re-running."); };
    bgm.onclick = function () { var g = gmm(MZ, 2, 3);
      report(g.hard, "GMM", "soft assignment - every point gets a probability per cluster, not a verdict. On moons it struggles like k-means (these are spherical gaussians), but on overlapping blobs the maybe-both answer is exactly what marketing needs for borderline customers."); };
  }

  if (MODE === "pca") {
    var bars = el("div"); side.appendChild(bars);
    drawScatter(cv, P.proj, D.truth, null, { xl: "PC1 (the value axis)", yl: "PC2" });
    var h = "<table><tr><th>component</th><th>variance explained</th></tr>";
    P.explained.forEach(function (e, i) { h += "<tr><td>PC" + (i + 1) + "</td><td>" + fmtPct(e) + "</td></tr>"; });
    bars.innerHTML = h + "</table>";
    readout.innerHTML = "<b>Real PCA on the scaled pile:</b> PC1 alone carries " + fmtPct(P.explained[0]) + " of the variance, PC1+PC2 together " + fmtPct(P.explained[0] + P.explained[1]) + ". Three features became two axes and the segment structure survived - that is compression. (Colors here are the PLANTED truth, so you can see what the projection preserved.)";
  }

  if (MODE === "basket") {
    left.removeChild(cv);
    var B = makeBaskets(); var rules = mineRules(B, 0.02).slice(0, 10);
    var h2 = "<table><tr><th>if the basket has</th><th>then also</th><th>support</th><th>confidence</th><th>lift</th></tr>";
    rules.forEach(function (r) { h2 += "<tr><td>" + r.a + "</td><td>" + r.c + "</td><td>" + fmtPct(r.sup) + "</td><td>" + fmtPct(r.conf) + "</td><td><b" + (r.lift > 2 ? " style='color:" + TH.deep + "'" : "") + ">" + r.lift.toFixed(1) + "×</b></td></tr>"; });
    left.innerHTML = h2 + "</table>";
    readout.innerHTML = "<b>Mined live from 500 baskets:</b> lift is the honest column - " + rules[0].a + " → " + rules[0].c + " happens " + rules[0].lift.toFixed(1) + "× more than chance. High confidence with lift ≈ 1 just means the consequent is popular with everyone (put it next to anything). Chase lift, sanity-check support.";
  }

  if (MODE === "noise") {
    var N = makeNoise(); var NZ = scaleZ(N); var NP = pca2(N);
    var run5 = el("button", "cl-btn", "▶ Run 5-means on noise");
    side.appendChild(run5); var box5 = el("div"); side.appendChild(box5);
    drawScatter(cv, NP.proj, null, null, { xl: "PC1", yl: "PC2" });
    run5.onclick = function () {
      var km = kmeans(NZ, 5, 1); var sil = silhouette(NZ, km.assign);
      var realSil = silhouette(Z, kmeans(Z, 5, 1).assign);
      drawScatter(cv, NP.proj, km.assign, null, { xl: "PC1", yl: "PC2" });
      box5.innerHTML = ""; box5.appendChild(stats([
        { v: fmt2(sil), l: "silhouette (noise)", bad: true },
        { v: fmt2(realSil), l: "silhouette (real pile)" }]));
      readout.innerHTML = "<b>It answered anyway.</b> Five confident, colorful segments - on data with NO structure planted. k-means partitions, it does not discover; nothing in the algorithm refuses. The only tell is the silhouette gap (" + fmt2(sil) + " vs " + fmt2(realSil) + " on the real pile). If your real project's silhouette looks like the left number, you have named noise.";
    };
  }

  if (MODE === "capstone") {
    var stages = [
      { label: "1 · Raw 5-means", f: function () { var km = kmeans(D.X, 5, 1);
          return { X2: D.X.map(function (r) { return [r[0], r[2]]; }), a: km.assign, sil: silhouette(D.X, km.assign), pur: purity(km.assign, D.truth), xl: "recency", yl: "cents",
            msg: "Raw units - the cents column owns the distance and the segments are just wallet bands. The floor." }; } },
      { label: "2 · Scale first", f: function () { var km = kmeans(Z, 5, 1);
          return { X2: P.proj, a: km.assign, sil: silhouette(Z, km.assign), pur: purity(km.assign, D.truth), xl: "PC1", yl: "PC2",
            msg: "Standardize, re-run. The single biggest jump in the whole project." }; } },
      { label: "3 · Validate k", f: function () { var best = null, bk = 0;
          [3, 4, 5, 6, 7].forEach(function (k) { var km = kmeans(Z, k, 1); var s = silhouette(Z, km.assign);
            if (!best || s > best.sil) { best = { km: km, sil: s }; bk = k; } });
          return { X2: P.proj, a: best.km.assign, sil: best.sil, pur: purity(best.km.assign, D.truth), xl: "PC1", yl: "PC2",
            msg: "Silhouette sweep confirms k=" + bk + " - the structure that is really there, not the number someone wanted." }; } },
      { label: "4 · Name + action", f: function () { var km = kmeans(Z, 5, 1);
          var rows = segTable(D.X, km.assign, 5);
          var names = rows.map(function (r) {
            if (r.r > 150) return "Lapsed big spenders - win-back";
            if (r.f > 12) return "VIP regulars - perks";
            if (r.f < 2 && r.mo > 7500) return "One-time gifters - occasion nudges";
            if (r.mo < 5500) return "Fresh arrivals - welcome flow";
            return "Steady mid-basket - upsell"; });
          return { X2: P.proj, a: km.assign, sil: silhouette(Z, km.assign), pur: purity(km.assign, D.truth), xl: "PC1", yl: "PC2", names: names,
            msg: "Segments become personas with owners and campaigns. THIS row is the deliverable - the math above was the road." }; } }
    ];
    var box6 = el("div"); side.appendChild(box6);
    var tbl2 = el("div"); mount.insertBefore(tbl2, readout);
    stages.forEach(function (st, i) {
      var b = el("button", "cl-btn", st.label);
      b.onclick = function () { var r = st.f();
        drawScatter(cv, r.X2, r.a, null, { xl: r.xl, yl: r.yl });
        box6.innerHTML = ""; box6.appendChild(stats([
          { v: fmt2(r.sil), l: "silhouette" }, { v: fmtPct(r.pur), l: "truth match" }]));
        if (r.names) { var h3 = "<table><tr><th>segment</th><th>name + campaign</th></tr>";
          r.names.forEach(function (nm, j) { h3 += "<tr><td><b style='color:" + PAL[j % PAL.length] + "'>■</b> " + (j + 1) + "</td><td>" + nm + "</td></tr>"; });
          tbl2.innerHTML = h3 + "</table>"; } else tbl2.innerHTML = "";
        readout.innerHTML = "<b>" + st.label + ":</b> " + r.msg; };
      side.insertBefore(b, box6);
    });
  }
})();
