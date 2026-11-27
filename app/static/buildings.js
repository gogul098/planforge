// Schematic plans of the buildings on the conversion page, redrawn as simple solids (metres, approximate).
// One model per building feeds both drawings: the plan (top view) and the axonometric (3D).
// Element: { p: footprint [[x,y],...], z0, z1, k }  k: "w" wall/solid, "g" glass, "s" slab, "c" roof/dome, "r" see-through roof
// or a free 3D polygon { t: [[x,y,z],...], k }.
const box = (x0, y0, x1, y1, z0, z1, k = "w") => ({ p: [[x0, y0], [x1, y0], [x1, y1], [x0, y1]], z0, z1, k });
const ngon = (cx, cy, r, n, z0, z1, k = "w", rot = Math.PI / n) =>
  ({ p: [...Array(n)].map((_, i) => [cx + r * Math.cos(rot + (i * 2 * Math.PI) / n), cy + r * Math.sin(rot + (i * 2 * Math.PI) / n)]), z0, z1, k });

function farnsworth() {
  const e = [box(-1, 9.6, 15, 15.4, 0, 0.35, "s"), box(0, 0, 23.5, 8.8, 1.2, 1.55, "s"), box(0, 0, 23.5, 8.8, 4.3, 4.6, "c")];
  e.push(box(6.7, 0, 23.5, 0.08, 1.55, 4.3, "g"), box(6.7, 8.72, 23.5, 8.8, 1.55, 4.3, "g"), box(6.7, 0, 6.78, 8.8, 1.55, 4.3, "g"), box(23.42, 0, 23.5, 8.8, 1.55, 4.3, "g"));
  e.push(box(13, 2.6, 20.3, 6.2, 1.55, 4.3, "w"));
  [0, 6.7, 13.4, 20.1, 23.3].forEach((x) => { e.push(box(x, -0.25, x + 0.25, 0, 0, 4.6), box(x, 8.8, x + 0.25, 9.05, 0, 4.6)); });
  return e;
}
function fallingwater() {
  return [
    box(0, 0, 14, 0.7, 0, 3.2), box(0, 0, 0.7, 11, 0, 3.2), box(13.3, 0, 14, 4, 0, 3.2), box(5, -1.5, 7.6, 1.2, 0, 8.5),
    box(0.7, 10.7, 14, 10.9, 0, 3.2, "g"), box(13.8, 4, 14, 10.7, 0, 3.2, "g"),
    box(-0.5, 0, 14.5, 11, 3.2, 3.6, "s"),
    box(1.5, 10.9, 12.5, 15.2, 0, 0.9, "s"), box(14, 3, 19.5, 9.5, 0, 0.9, "s"),
    box(2.5, -0.5, 11.5, 7.5, 3.6, 6.4, "w"), box(2.5, 7.5, 11.5, 7.7, 3.6, 6.4, "g"),
    box(0, 7.7, 13, 12.5, 3.6, 4.5, "s"), box(2, -0.5, 12, 8, 6.4, 6.8, "c"),
  ];
}
function lotus() {
  const e = [ngon(0, 0, 34, 9, 0, 1.2, "s"), ngon(0, 0, 27, 9, 1.2, 2, "s")];
  const ring = (r0, r1, h, half, off) => {
    for (let k = 0; k < 9; k++) {
      const a = off + (k * 2 * Math.PI) / 9, b0 = a - half, b1 = a + half, z = 2;
      e.push({ t: [[r0 * Math.cos(b0), r0 * Math.sin(b0), z], [r0 * Math.cos(b1), r0 * Math.sin(b1), z], [r1 * Math.cos(a), r1 * Math.sin(a), z + h]], k: "c" });
    }
  };
  ring(25, 21, 11, 0.3, 0); ring(19, 12, 22, 0.33, Math.PI / 9); ring(12, 2, 34, 0.36, 0);
  return e;
}
function taj() {
  const e = [box(-48, -48, 48, 48, 0, 6.5, "s")];
  const oct = (r, c) => [[-r + c, -r], [r - c, -r], [r, -r + c], [r, r - c], [r - c, r], [-r + c, r], [-r, r - c], [-r, -r + c]];
  e.push({ p: oct(28.5, 8), z0: 6.5, z1: 36, k: "w" });
  e.push(ngon(0, 0, 13, 16, 36, 46, "w"));
  [[13.5, 46, 52], [12.5, 52, 58], [10.5, 58, 64], [7.5, 64, 68], [3.5, 68, 71], [0.6, 71, 75]].forEach(([r, a, b]) => e.push(ngon(0, 0, r, 16, a, b, "c")));
  [[-18, -18], [18, -18], [18, 18], [-18, 18]].forEach(([x, y]) => { e.push(ngon(x, y, 4, 8, 36, 41, "w"), ngon(x, y, 4.5, 8, 41, 45, "c")); });
  [[-44, -44], [44, -44], [44, 44], [-44, 44]].forEach(([x, y]) => { e.push(ngon(x, y, 2.6, 8, 6.5, 46, "w"), ngon(x, y, 3.2, 8, 46, 47, "s"), ngon(x, y, 2.6, 8, 47, 52, "c")); });
  return e;
}
function gamble() {
  return [
    box(0, 0, 22, 0.35, 0, 3.4), box(0, 13.65, 22, 14, 0, 3.4), box(0, 0, 0.35, 14, 0, 3.4), box(21.65, 0, 22, 9, 0, 3.4),
    box(8, 0, 8.2, 14, 0, 3.4), box(0, 7, 8, 7.2, 0, 3.4), box(14, 0, 14.2, 9, 0, 3.4), box(22, 9, 27, 9.35, 0, 3.4), box(26.65, 9, 27, 14, 0, 3.4), box(22, 13.65, 27, 14, 0, 3.4),
    box(-1.2, -1.2, 28.2, 15.2, 3.4, 3.75, "s"),
    box(2, 0.5, 20, 13.5, 3.75, 6.6, "w"), box(9, -2.5, 16, 0.5, 3.75, 6.6, "g"), box(17, 13.5, 24, 16.5, 3.75, 6.6, "g"),
    box(0.5, -3.5, 22, 17.5, 6.6, 6.95, "r"),
  ];
}

export const MODELS = { fallingwater, lotus, farnsworth, taj, gamble };

// ---------------------------------------------------------------- drawing
const ns = (t, a) => `<${t} ${Object.entries(a).map(([k, v]) => `${k}="${v}"`).join(" ")}/>`;
const f = (v) => v.toFixed(1);

export function planSvg(els) {
  const pts = els.flatMap((e) => (e.p || e.t.map(([x, y]) => [x, y])));
  const xs = pts.map((p) => p[0]), ys = pts.map((p) => p[1]);
  const x0 = Math.min(...xs), y0 = Math.min(...ys), w = Math.max(...xs) - x0, h = Math.max(...ys) - y0;
  const S = 300 / Math.max(w, h), pad = 14;
  const P = ([x, y]) => [pad + (x - x0) * S, pad + (y - y0) * S];
  const lw = Math.max(w, h);
  let out = "", i = 0;
  const order = [...els].sort((a, b) => (({ s: 0, r: 1, c: 1, g: 2, w: 3 })[a.k] - ({ s: 0, r: 1, c: 1, g: 2, w: 3 })[b.k]));
  for (const e of order) {
    const pp = (e.p || e.t.map(([x, y]) => [x, y])).map(P);
    const d = "M" + pp.map((p) => p.map(f).join(" ")).join("L") + "Z";
    const cls = e.k === "w" && (e.z0 ?? 0) < 2.5 ? "pw" : e.k === "w" ? "pu" : e.k === "g" ? "pg" : e.k === "s" ? "ps" : "pc";
    out += ns("path", { d, class: cls, pathLength: 1, style: `--d:${(0.15 + (i++ * 1.6) / els.length).toFixed(2)}s` });
  }
  // a scale bar of round metres
  const bar = lw > 60 ? 20 : lw > 25 ? 10 : 5;
  out += `<g class="pscale"><path d="M${pad} ${pad * 2 + h * S + 8}h${f(bar * S)}"/><text x="${pad}" y="${f(pad * 2 + h * S + 24)}">${bar} m</text></g>`;
  return `<svg viewBox="0 0 ${f(pad * 2 + w * S)} ${f(pad * 2 + h * S + 30)}" aria-hidden="true">${out}</svg>`;
}

export function axoSvg(els) {
  const c = Math.cos(Math.PI / 6), s = 0.5;
  const P = (x, y, z) => [(x - y) * c, (x + y) * s - z];
  const faces = [];
  for (const e of els) {
    if (e.t) { const m = e.t.reduce((a, p) => [a[0] + p[0], a[1] + p[1], a[2] + p[2]], [0, 0, 0]).map((v) => v / e.t.length); faces.push({ pts: e.t.map((p) => P(...p)), k: e.k, depth: m[0] + m[1] + m[2] * 0.02, tone: "a" }); continue; }
    const n = e.p.length, cx = e.p.reduce((a, p) => a + p[0], 0) / n, cy = e.p.reduce((a, p) => a + p[1], 0) / n;
    const base = cx + cy + e.z0 * 0.02;
    for (let i = 0; i < n; i++) {
      const [ax, ay] = e.p[i], [bx, by] = e.p[(i + 1) % n];
      const nx = by - ay, ny = -(bx - ax);                       // outward normal for a counter-clockwise footprint
      const facing = (nx + ny) * (signedArea(e.p) > 0 ? 1 : -1);
      if (facing <= 0) continue;
      faces.push({ pts: [P(ax, ay, e.z0), P(bx, by, e.z0), P(bx, by, e.z1), P(ax, ay, e.z1)], k: e.k, depth: base, tone: nx * (signedArea(e.p) > 0 ? 1 : -1) > ny * (signedArea(e.p) > 0 ? 1 : -1) ? "a" : "b" });
    }
    faces.push({ pts: e.p.map(([x, y]) => P(x, y, e.z1)), k: e.k, depth: base + 0.001, tone: "t" });
  }
  faces.sort((a, b) => a.depth - b.depth);
  const all = faces.flatMap((q) => q.pts), xs = all.map((p) => p[0]), ys = all.map((p) => p[1]);
  const x0 = Math.min(...xs), y0 = Math.min(...ys), W = Math.max(...xs) - x0, H = Math.max(...ys) - y0, S = 300 / Math.max(W, H), pad = 10;
  let out = "";
  faces.forEach((q, i) => {
    const d = "M" + q.pts.map(([x, y]) => f(pad + (x - x0) * S) + " " + f(pad + (y - y0) * S)).join("L") + "Z";
    out += ns("path", { d, class: `ax ${q.k} ${q.tone}`, style: `--d:${(0.3 + (i * 1.8) / faces.length).toFixed(2)}s` });
  });
  return `<svg viewBox="0 0 ${f(pad * 2 + W * S)} ${f(pad * 2 + H * S)}" aria-hidden="true">${out}</svg>`;
}
function signedArea(p) { let a = 0; for (let i = 0; i < p.length; i++) { const [x0, y0] = p[i], [x1, y1] = p[(i + 1) % p.length]; a += x0 * y1 - x1 * y0; } return a / 2; }
