// PlanForge viewer + editor. Plan coords (x, y) in metres  <->  three.js (X = x, Y = up, Z = y).
import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { TransformControls } from "three/addons/controls/TransformControls.js";
import { GLTFLoader } from "three/addons/loaders/GLTFLoader.js";

// API location: same site by default; a static host (Vercel) points it at the
// conversion engine via config.js or ?api=https://... in the URL.
const API = (new URLSearchParams(location.search).get("api") || window.PLANFORGE_API || "").replace(/\/$/, "");
const $ = (s) => document.querySelector(s);
const $$ = (s) => [...document.querySelectorAll(s)];
const app = $("#app");

// ------------------------------------------------------------------ state
const S = {
  file: null, job: null, plan: null, furn: [], uid: 1, sel: null, tab: "3d", tool: "orbit",
  etool: "select", snap: true, xray: false, edges: false, section: false, placing: null,
  hist: [], fut: [], beforeEdit: null, dirty: false,
};
const ACC = 0xc89b5e;                               // brass: selection, measure, markers (both themes)
const CAT_COL = { furniture: 0xb0845c, plumbing: 0xeef0f4, appliance: 0x969ba2 };
const LAYER_COL = { wall: 0xe9e4dc, door: 0x8a5a3b, glass: 0x8fc3e8, floor: 0xc9b9a2, stairs: 0xbfb3a3, plinth: 0xa8a096, ground: 0xa89a82 };
const DEFAULT_SIZE = {
  bed: [2.0, 1.6], sofa: [2.0, 0.9], chair: [0.5, 0.5], table: [1.4, 0.9], wardrobe: [1.8, 0.6],
  tv_cabinet: [1.6, 0.45], toilet: [0.7, 0.4], sink: [0.6, 0.45], bath_tub: [1.7, 0.75],
  gas_stove: [0.6, 0.6], refrigerator: [0.7, 0.7], washing_machine: [0.6, 0.6],
  bedside_cupboard: [0.5, 0.45], high_cabinet: [0.9, 0.6], half_height_cabinet: [1.2, 0.5], closet: [1.2, 0.6],
};
const LIB = ["bed", "sofa", "chair", "table", "wardrobe", "tv_cabinet", "toilet", "sink", "bath_tub",
  "gas_stove", "refrigerator", "washing_machine", "bedside_cupboard", "high_cabinet", "half_height_cabinet"];
let CATALOG = {};
const label = (t) => t.replace(/_/g, " ").replace(/^./, (c) => c.toUpperCase());

// ------------------------------------------------------------------ three setup
const canvas = $("#c3d");
const renderer = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: true, preserveDrawingBuffer: true });
renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
renderer.shadowMap.enabled = true;
renderer.shadowMap.type = THREE.PCFSoftShadowMap;
renderer.localClippingEnabled = false;
const scene = new THREE.Scene();
const persp = new THREE.PerspectiveCamera(45, 1, 0.05, 2000);
const ortho = new THREE.OrthographicCamera(-10, 10, 10, -10, -500, 500);
let camera = persp;
scene.add(new THREE.HemisphereLight(0xfff6ea, 0x5a4a3a, 1.1));
const sun = new THREE.DirectionalLight(0xffffff, 1.6);
sun.castShadow = true;
sun.shadow.mapSize.set(2048, 2048); sun.shadow.bias = -0.0004; sun.shadow.normalBias = 0.03;
scene.add(sun, sun.target);
// ground grid follows the theme (night: Kota stone, day: sandstone)
const GRID_COL = { night: [0x4a524c, 0x343b36], day: [0xb39469, 0xc9ad84] };
let grid = null;
function makeGrid(theme) {
  const pos = grid ? grid.position.clone() : null;
  if (grid) { scene.remove(grid); grid.geometry.dispose(); grid.material.dispose(); }
  const c = GRID_COL[theme] || GRID_COL.night;
  grid = new THREE.GridHelper(200, 200, c[0], c[1]);
  if (pos) grid.position.copy(pos);
  grid.material.transparent = true; grid.material.opacity = 0.55;
  scene.add(grid);
}
makeGrid(document.documentElement.dataset.theme || "night");
window.addEventListener("pf-theme", (e) => makeGrid(e.detail));
const building = new THREE.Group(); building.name = "building";
const furnG = new THREE.Group(); furnG.name = "furniture";
const helpers = new THREE.Group();
scene.add(building, furnG, helpers);

const controls = new OrbitControls(persp, canvas);
controls.enableDamping = true; controls.dampingFactor = 0.12; controls.screenSpacePanning = true;
const tctl = new TransformControls(persp, canvas);
tctl.setSize(0.9);
scene.add(tctl);
tctl.addEventListener("dragging-changed", (e) => {
  controls.enabled = !e.value;
  if (e.value) pushHist(); else { syncFromMesh(); renderProps(); }
});
tctl.addEventListener("objectChange", () => { syncFromMesh(); });

const cutPlane = new THREE.Plane(new THREE.Vector3(0, -1, 0), 1.2);
const loader = new GLTFLoader();
const ray = new THREE.Raycaster();

function resize() {
  const r = $("#view").getBoundingClientRect();
  renderer.setSize(r.width, r.height, false);
  persp.aspect = r.width / r.height; persp.updateProjectionMatrix();
  fitOrtho();
}
addEventListener("resize", resize);

// ------------------------------------------------------------------ model loading
let bbox = new THREE.Box3(new THREE.Vector3(-5, 0, -5), new THREE.Vector3(5, 3, 5));
function matFor(name) {
  const key = Object.keys(LAYER_COL).find((k) => name.toLowerCase().includes(k)) || "wall";
  const m = new THREE.MeshStandardMaterial({
    color: LAYER_COL[key], roughness: key === "glass" ? 0.1 : 0.85, metalness: 0,
    flatShading: true, transparent: key === "glass", opacity: key === "glass" ? 0.45 : 1, side: THREE.DoubleSide,
  });
  m.userData.layer = key; m.userData.baseOpacity = m.opacity; m.userData.baseTransparent = m.transparent;
  return m;
}
async function loadBuilding(url, faithful = false) {
  const gltf = await loader.loadAsync(url + "?t=" + Date.now());
  building.clear();
  gltf.scene.traverse((o) => {
    if (!o.isMesh) return;
    const nm = (o.name || o.parent?.name || "wall") + "";
    if (/furniture|plumbing|appliance/.test(nm)) { o.visible = faithful; }  // editable copies drawn separately
    if (o.geometry.index) o.geometry = o.geometry.toNonIndexed();
    o.geometry.deleteAttribute("color");
    o.geometry.computeVertexNormals();
    o.material = matFor(nm);
    o.userData.layer = o.material.userData.layer;
    o.castShadow = o.receiveShadow = true;
  });
  building.add(gltf.scene);
  bbox = new THREE.Box3().setFromObject(building);
  const c = bbox.getCenter(new THREE.Vector3()), sz = bbox.getSize(new THREE.Vector3());
  grid.position.set(c.x, bbox.min.y - 0.01, c.z);
  sun.position.set(c.x + sz.x, 30, c.z + sz.z * 0.7); sun.target.position.copy(c);
  const d = Math.max(sz.x, sz.z) * 0.7;
  Object.assign(sun.shadow.camera, { left: -d, right: d, top: d, bottom: -d, near: 0.5, far: 120 });
  sun.shadow.camera.updateProjectionMatrix();
  $("#cutH").max = (sz.y + 0.2).toFixed(2);
  applyXray(); applyEdges(); applyLayers();
}

// ------------------------------------------------------------------ furniture (editable)
function fromPlan(f, i) {
  const c = f.corners;
  const e01 = Math.hypot(c[1][0] - c[0][0], c[1][1] - c[0][1]);
  const e12 = Math.hypot(c[2][0] - c[1][0], c[2][1] - c[1][1]);
  const [p, q] = e01 >= e12 ? [c[0], c[1]] : [c[1], c[2]];
  const ang = Math.atan2(q[1] - p[1], q[0] - p[0]);
  const fr = Math.atan2(f.front[1], f.front[0]);
  return {
    uid: S.uid++, type: f.type, center: [...f.center], width: f.width, depth: f.depth, height: f.height,
    angle: ang, frontRel: fr - ang, source: f.source, confidence: f.confidence, category: f.category,
  };
}
function catOf(t) { return CATALOG[t]?.category || (/(sink|bath|toilet|urinal)/.test(t) ? "plumbing" : /(stove|fridge|refrig|wash|appliance|fireplace|chimney)/.test(t) ? "appliance" : "furniture"); }
function heightOf(t) { return CATALOG[t]?.height ?? 0.8; }

function makeFurnMesh(it) {
  const g = new THREE.BoxGeometry(1, 1, 1);
  const m = new THREE.MeshStandardMaterial({ color: CAT_COL[catOf(it.type)] || 0xb0845c, roughness: 0.7 });
  const mesh = new THREE.Mesh(g, m);
  mesh.castShadow = mesh.receiveShadow = true;
  const edge = new THREE.LineSegments(new THREE.EdgesGeometry(g), new THREE.LineBasicMaterial({ color: 0x2b1d12, transparent: true, opacity: 0.55 }));
  mesh.add(edge);
  // front marker: small strip on the front face
  const mk = new THREE.Mesh(new THREE.BoxGeometry(0.6, 0.06, 0.02), new THREE.MeshBasicMaterial({ color: ACC }));
  mk.name = "front"; mesh.add(mk);
  mesh.userData.uid = it.uid;
  return mesh;
}
function placeMesh(mesh, it) {
  mesh.scale.set(it.width, it.height, it.depth);
  mesh.position.set(it.center[0], it.height / 2, it.center[1]);
  mesh.rotation.set(0, -it.angle, 0);
  const mk = mesh.getObjectByName("front");
  // front direction in local (x = long axis, z = short axis): angle frontRel from long axis
  const fx = Math.cos(it.frontRel), fz = Math.sin(it.frontRel);
  mk.position.set(fx * 0.5, 0.3, fz * 0.5);
  mk.rotation.set(0, -it.frontRel + Math.PI / 2, 0);
  mk.scale.set(Math.abs(fz) > 0.5 ? 1 : it.depth / it.width, 1, 1);
  mesh.material.color.set(CAT_COL[catOf(it.type)] || 0xb0845c);
}
function rebuildFurniture() {
  const keep = S.sel;
  tctl.detach();
  furnG.clear();
  for (const it of S.furn) { const m = makeFurnMesh(it); placeMesh(m, it); furnG.add(m); }
  applyXray(); applyLayers();
  if (keep && S.furn.find((f) => f.uid === keep.uid)) select({ kind: "furn", uid: keep.uid }); else if (keep?.kind === "furn") select(null);
  $("#s_furn").textContent = S.furn.length;
  renderPlaced();
}
const meshOf = (uid) => furnG.children.find((m) => m.userData.uid === uid);
const itemOf = (uid) => S.furn.find((f) => f.uid === uid);
function syncFromMesh() {
  if (!S.sel || S.sel.kind !== "furn") return;
  const m = meshOf(S.sel.uid), it = itemOf(S.sel.uid);
  if (!m || !it) return;
  it.center = [+m.position.x.toFixed(4), +m.position.z.toFixed(4)];
  it.angle = -m.rotation.y;
  m.position.y = it.height / 2;
  S.dirty = true;
  updateSelBox();
}

// ------------------------------------------------------------------ geometry helpers
const wallsOf = () => S.plan?.walls || [];
function segDist(p, a, b) {
  const dx = b[0] - a[0], dy = b[1] - a[1], L2 = dx * dx + dy * dy || 1e-9;
  const t = Math.max(0, Math.min(1, ((p[0] - a[0]) * dx + (p[1] - a[1]) * dy) / L2));
  return [Math.hypot(p[0] - a[0] - t * dx, p[1] - a[1] - t * dy), t];
}
function nearestWall(p) {
  let best = null;
  for (const w of wallsOf()) {
    const [d] = segDist(p, w.start, w.end);
    if (!best || d < best.d) best = { w, d };
  }
  return best;
}
function inPoly(p, poly) {
  let ins = false;
  for (let i = 0, j = poly.length - 1; i < poly.length; j = i++) {
    const [xi, yi] = poly[i], [xj, yj] = poly[j];
    if ((yi > p[1]) !== (yj > p[1]) && p[0] < ((xj - xi) * (p[1] - yi)) / (yj - yi) + xi) ins = !ins;
  }
  return ins;
}
const roomAt = (p) => (S.plan?.rooms || []).filter((r) => inPoly(p, r.polygon)).sort((a, b) => a.area_m2 - b.area_m2)[0];

function backToWall(it) {
  const nw = nearestWall(it.center);
  if (!nw) return;
  const { w } = nw;
  const dx = w.end[0] - w.start[0], dy = w.end[1] - w.start[1], L = Math.hypot(dx, dy) || 1;
  const d = [dx / L, dy / L], n = [-d[1], d[0]];
  const rel = [it.center[0] - w.start[0], it.center[1] - w.start[1]];
  const side = Math.sign(rel[0] * n[0] + rel[1] * n[1]) || 1;
  const along = rel[0] * d[0] + rel[1] * d[1];
  const spec = CATALOG[it.type];
  const headOn = ["bed", "toilet", "squat_toilet", "urinal"].includes(it.type);
  const off = w.thickness / 2 + (headOn ? it.width : it.depth) / 2 + 0.005;
  it.center = [w.start[0] + d[0] * along + n[0] * off * side, w.start[1] + d[1] * along + n[1] * off * side];
  const wallAng = Math.atan2(d[1], d[0]);
  // long axis along the wall (or into the room for head-on items); front points away from the wall
  it.angle = headOn ? Math.atan2(n[1] * side, n[0] * side) : wallAng;
  const frontAng = Math.atan2(n[1] * side, n[0] * side);
  it.frontRel = frontAng - it.angle;
  void spec;
}

// ------------------------------------------------------------------ camera / views
function fitOrtho() {
  const r = $("#view").getBoundingClientRect(), asp = r.width / Math.max(r.height, 1);
  const sz = bbox.getSize(new THREE.Vector3());
  const h = Math.max(sz.z, sz.x / asp) * 0.62 + 1;
  Object.assign(ortho, { left: -h * asp, right: h * asp, top: h, bottom: -h });
  ortho.updateProjectionMatrix();
}
function setCamera(cam) {
  camera = cam; controls.object = cam; tctl.camera = cam;
  controls.enableRotate = cam === persp && S.tool !== "walk";
}
function fit(dir = new THREE.Vector3(1, 0.9, 1.15)) {
  const c = bbox.getCenter(new THREE.Vector3()), sz = bbox.getSize(new THREE.Vector3());
  const r = Math.max(sz.x, sz.y, sz.z) * 0.62 + 0.5;
  const dist = r / Math.sin((persp.fov * Math.PI) / 360);
  if (camera === ortho) {
    ortho.position.set(c.x, 100, c.z); ortho.up.set(0, 0, -1); ortho.zoom = 1; fitOrtho();
  } else {
    persp.up.set(0, 1, 0);
    persp.position.copy(c).add(dir.clone().normalize().multiplyScalar(dist * 0.85));
  }
  controls.target.copy(c); controls.update();
}
function setTab(t) {
  S.tab = t;
  $$("#tabs span").forEach((s) => s.classList.toggle("on", s.dataset.t === t));
  $("#traceImg").classList.toggle("hidden", t !== "trace");
  canvas.style.visibility = t === "trace" ? "hidden" : "visible";
  if (t === "plan") { setCamera(ortho); fit(); }
  else if (t === "3d" || t === "raw") { if (camera !== persp) { setCamera(persp); fit(); } }
  if (S.job && (t === "raw") !== !!S.rawLoaded) {
    S.rawLoaded = t === "raw";
    loadBuilding(`${API}/api/files/${S.job}/${t === "raw" ? "model_faithful.glb" : "shell.glb"}`, t === "raw")
      .then(() => { furnG.visible = t !== "raw"; });
  }
}

// view cube ---------------------------------------------------------------
const cubeR = new THREE.WebGLRenderer({ canvas: $("#cube"), antialias: true, alpha: true });
cubeR.setPixelRatio(2); cubeR.setSize(84, 84, false);
const cubeS = new THREE.Scene(), cubeC = new THREE.PerspectiveCamera(35, 1, 0.1, 10);
function faceTex(txt) {
  const cv = document.createElement("canvas"); cv.width = cv.height = 128;
  const g = cv.getContext("2d");
  g.fillStyle = "#e8e0d3"; g.fillRect(0, 0, 128, 128);
  g.strokeStyle = "#9a8b76"; g.lineWidth = 4; g.strokeRect(2, 2, 124, 124);
  g.fillStyle = "#3a2e22"; g.font = "500 22px 'IBM Plex Mono', monospace"; g.textAlign = "center"; g.textBaseline = "middle";
  g.fillText(txt, 64, 64);
  const t = new THREE.CanvasTexture(cv); t.colorSpace = THREE.SRGBColorSpace; return t;
}
// three.js box face order: +x, -x, +y, -y, +z, -z.  Plan "north" = -y = -Z (drawing top)
const cubeFaces = ["RIGHT", "LEFT", "TOP", "BOTTOM", "FRONT", "BACK"];
const cube = new THREE.Mesh(new THREE.BoxGeometry(1, 1, 1), cubeFaces.map((f) => new THREE.MeshBasicMaterial({ map: faceTex(f) })));
cubeS.add(cube);
$("#cube").addEventListener("pointerdown", (e) => {
  const r = e.target.getBoundingClientRect();
  const v = new THREE.Vector2(((e.clientX - r.left) / r.width) * 2 - 1, -((e.clientY - r.top) / r.height) * 2 + 1);
  ray.setFromCamera(v, cubeC);
  const hit = ray.intersectObject(cube)[0];
  if (!hit) return;
  const n = hit.face.normal.clone();
  // corner / edge clicks: blend in neighbouring axes when near the border
  const p = hit.point;
  ["x", "y", "z"].forEach((k) => { if (Math.abs(p[k]) > 0.38 && Math.abs(n[k]) < 0.5) n[k] = Math.sign(p[k]); });
  if (S.tab === "plan" && n.y < 0.99) setTab("3d");
  if (n.y > 0.99 && camera === persp) n.add(new THREE.Vector3(0, 0, 0.001));
  fit(n);
});
$("#btnHome").onclick = () => { if (S.tab !== "3d") setTab("3d"); fit(); };

// ------------------------------------------------------------------ tools
function setTool(t) {
  S.tool = t;
  $$("#tbView [data-tool]").forEach((i) => i.classList.toggle("on", i.dataset.tool === t));
  const L = { orbit: THREE.MOUSE.ROTATE, pan: THREE.MOUSE.PAN, zoom: THREE.MOUSE.DOLLY, measure: THREE.MOUSE.ROTATE, walk: null };
  controls.mouseButtons = { LEFT: L[t] ?? THREE.MOUSE.ROTATE, MIDDLE: THREE.MOUSE.DOLLY, RIGHT: THREE.MOUSE.PAN };
  controls.enabled = t !== "walk";
  controls.enableRotate = camera === persp;
  if (t === "walk") startWalk(); else walk.on = false;
  if (t !== "measure") { meas.pts = []; meas.group.clear(); $("#measOut").classList.add("hidden"); }
  canvas.style.cursor = t === "measure" ? "crosshair" : t === "walk" ? "move" : "";
}
function applyXray() {
  const xr = S.xray;
  building.traverse((o) => {
    if (!o.isMesh) return;
    const m = o.material;
    m.transparent = xr || m.userData.baseTransparent;
    m.opacity = xr ? Math.min(0.22, m.userData.baseOpacity) : m.userData.baseOpacity;
    m.depthWrite = !xr; m.needsUpdate = true;
  });
}
function applyEdges() {
  building.traverse((o) => {
    if (!o.isMesh) return;
    let e = o.children.find((c) => c.userData.isEdge);
    if (S.edges && !e) {
      e = new THREE.LineSegments(new THREE.EdgesGeometry(o.geometry, 25), new THREE.LineBasicMaterial({ color: 0x1b1f25 }));
      e.userData.isEdge = true; o.add(e);
    }
    if (e) e.visible = S.edges;
  });
}
const LAYERS = { wall: true, door: true, glass: true, floor: true, stairs: true, plinth: true, ground: true, furniture: true, grids: false };
function applyLayers() {
  building.traverse((o) => { if (o.isMesh && o.userData.layer && !/furniture|plumbing|appliance/.test(o.name)) o.visible = LAYERS[o.userData.layer] !== false; });
  furnG.visible = LAYERS.furniture && S.tab !== "raw";
  gridLines.visible = LAYERS.grids;
}
const gridLines = new THREE.Group(); helpers.add(gridLines);
function drawGrids() {
  gridLines.clear();
  for (const g of S.plan?.grids || []) {
    const pts = [new THREE.Vector3(g.start[0], 0.01, g.start[1]), new THREE.Vector3(g.end[0], 0.01, g.end[1])];
    const l = new THREE.Line(new THREE.BufferGeometry().setFromPoints(pts), new THREE.LineDashedMaterial({ color: ACC, dashSize: 0.4, gapSize: 0.2 }));
    l.computeLineDistances(); gridLines.add(l);
  }
}
function renderLayersPop() {
  const pop = $("#layersPop");
  pop.innerHTML = Object.keys(LAYERS).map((k) => `<label><input type="checkbox" data-l="${k}" ${LAYERS[k] ? "checked" : ""}> ${label(k)}</label>`).join("");
  pop.querySelectorAll("input").forEach((i) => (i.onchange = () => { LAYERS[i.dataset.l] = i.checked; applyLayers(); }));
}

// measure -----------------------------------------------------------------
const meas = { pts: [], group: new THREE.Group() };
helpers.add(meas.group);
function measureClick(p) {
  if (meas.pts.length >= 2) { meas.pts = []; meas.group.clear(); }
  meas.pts.push(p.clone());
  const dot = new THREE.Mesh(new THREE.SphereGeometry(0.06, 16, 12), new THREE.MeshBasicMaterial({ color: ACC, depthTest: false }));
  dot.position.copy(p); dot.renderOrder = 10; meas.group.add(dot);
  if (meas.pts.length === 2) {
    const l = new THREE.Line(new THREE.BufferGeometry().setFromPoints(meas.pts), new THREE.LineBasicMaterial({ color: ACC, depthTest: false }));
    l.renderOrder = 10; meas.group.add(l);
    const [a, b] = meas.pts, d = a.distanceTo(b);
    const dxz = Math.hypot(a.x - b.x, a.z - b.z);
    $("#measOut").textContent = `${d.toFixed(3)} m  ·  plan ${dxz.toFixed(3)} m  ·  Δh ${Math.abs(a.y - b.y).toFixed(3)} m`;
    $("#measOut").classList.remove("hidden");
  } else { $("#measOut").textContent = "Pick the second point"; $("#measOut").classList.remove("hidden"); }
}

// walk --------------------------------------------------------------------
const walk = { on: false, keys: {}, yaw: 0, pitch: 0, drag: null };
function startWalk() {
  if (S.tab !== "3d") setTab("3d");
  walk.on = true;
  const c = bbox.getCenter(new THREE.Vector3());
  persp.position.set(c.x, 1.6, c.z); persp.up.set(0, 1, 0);
  walk.yaw = 0; walk.pitch = 0;
  toast("Walk: W A S D to move, drag to look, Q/E down/up, Esc to exit");
}
addEventListener("keydown", (e) => {
  if (e.target.matches("input,select")) return;
  walk.keys[e.key.toLowerCase()] = true;
  if (e.key === "Escape") { if (walk.on) setTool("orbit"); select(null); }
  if (!walk.on) {
    const k = e.key.toLowerCase();
    if (app.classList.contains("mode-edit")) {
      if ((e.key === "Delete" || e.key === "Backspace") && S.sel?.kind === "furn") ctxAct("del");
      if (k === "r" && S.sel?.kind === "furn") ctxAct("rot");
      if ((e.ctrlKey || e.metaKey) && k === "z") { e.preventDefault(); e.shiftKey ? redo() : undo(); }
    } else if (S.plan) {
      ({ o: () => setTool("orbit"), p: () => setTool("pan"), z: () => setTool("zoom"), m: () => setTool("measure"), w: () => setTool("walk"), f: () => fit() }[k] || (() => {}))();
    }
  }
});
addEventListener("keyup", (e) => (walk.keys[e.key.toLowerCase()] = false));
function walkStep(dt) {
  const k = walk.keys, sp = (k.shift ? 3.5 : 1.6) * dt;
  const fwd = new THREE.Vector3(Math.sin(walk.yaw), 0, -Math.cos(walk.yaw));
  const right = new THREE.Vector3(-fwd.z, 0, fwd.x);
  if (k.w || k.arrowup) persp.position.addScaledVector(fwd, sp);
  if (k.s || k.arrowdown) persp.position.addScaledVector(fwd, -sp);
  if (k.a || k.arrowleft) persp.position.addScaledVector(right, -sp);
  if (k.d || k.arrowright) persp.position.addScaledVector(right, sp);
  if (k.e) persp.position.y += sp; if (k.q) persp.position.y = Math.max(0.3, persp.position.y - sp);
  const look = new THREE.Vector3(Math.sin(walk.yaw) * Math.cos(walk.pitch), Math.sin(walk.pitch), -Math.cos(walk.yaw) * Math.cos(walk.pitch));
  persp.lookAt(persp.position.clone().add(look));
}

// ------------------------------------------------------------------ picking & selection
function ndc(e) {
  const r = canvas.getBoundingClientRect();
  return new THREE.Vector2(((e.clientX - r.left) / r.width) * 2 - 1, -((e.clientY - r.top) / r.height) * 2 + 1);
}
function pick(e, objs) { ray.setFromCamera(ndc(e), camera); return ray.intersectObjects(objs, true).filter((h) => h.object.visible && !h.object.userData.isEdge && h.object.type === "Mesh")[0]; }
function floorPoint(e) {
  ray.setFromCamera(ndc(e), camera);
  const p = new THREE.Vector3();
  return ray.ray.intersectPlane(new THREE.Plane(new THREE.Vector3(0, 1, 0), 0), p) ? p : null;
}
const selBox = new THREE.BoxHelper(undefined, ACC); selBox.visible = false; helpers.add(selBox);
const wallHi = new THREE.Mesh(new THREE.BoxGeometry(1, 1, 1), new THREE.MeshBasicMaterial({ color: ACC, transparent: true, opacity: 0.35, depthTest: false }));
wallHi.visible = false; wallHi.renderOrder = 5; helpers.add(wallHi);
const roomHi = new THREE.Mesh(new THREE.BufferGeometry(), new THREE.MeshBasicMaterial({ color: ACC, transparent: true, opacity: 0.22, side: THREE.DoubleSide, depthTest: false }));
roomHi.visible = false; roomHi.renderOrder = 4; helpers.add(roomHi);
function updateSelBox() {
  if (S.sel?.kind === "furn") { const m = meshOf(S.sel.uid); if (m) { selBox.setFromObject(m); selBox.visible = true; return; } }
  selBox.visible = false;
}
function select(sel) {
  S.sel = sel;
  wallHi.visible = roomHi.visible = false;
  tctl.detach();
  $("#ctx").classList.add("hidden");
  if (sel?.kind === "furn") {
    const m = meshOf(sel.uid);
    if (m && app.classList.contains("mode-edit") && S.etool !== "select") { tctl.attach(m); setTctlMode(); }
  } else if (sel?.kind === "wall") {
    const w = sel.w, dx = w.end[0] - w.start[0], dy = w.end[1] - w.start[1], L = Math.hypot(dx, dy);
    wallHi.scale.set(L + w.thickness, w.height + 0.02, w.thickness + 0.02);
    wallHi.position.set((w.start[0] + w.end[0]) / 2, w.height / 2, (w.start[1] + w.end[1]) / 2);
    wallHi.rotation.set(0, -Math.atan2(dy, dx), 0); wallHi.visible = true;
  } else if (sel?.kind === "stair") {
    const sh = new THREE.Shape(sel.s.outline.map(([x, y]) => new THREE.Vector2(x, -y)));
    roomHi.geometry.dispose(); roomHi.geometry = new THREE.ShapeGeometry(sh);
    roomHi.rotation.set(-Math.PI / 2, 0, 0); roomHi.position.y = sel.s.top_z + 0.03; roomHi.visible = true;
  } else if (sel?.kind === "room") {
    const sh = new THREE.Shape(sel.r.polygon.map(([x, y]) => new THREE.Vector2(x, -y)));
    roomHi.geometry.dispose(); roomHi.geometry = new THREE.ShapeGeometry(sh);
    roomHi.rotation.set(-Math.PI / 2, 0, 0); roomHi.position.y = 0.03; roomHi.visible = true;
  }
  updateSelBox(); renderProps(); renderPlaced(); renderRooms();
}
function setTctlMode() {
  const rot = S.etool === "rotate";
  tctl.setMode(rot ? "rotate" : "translate");
  tctl.showX = !rot; tctl.showZ = !rot; tctl.showY = rot;
  tctl.setTranslationSnap(S.snap ? 0.05 : null);
  tctl.setRotationSnap(S.snap ? THREE.MathUtils.degToRad(15) : null);
}

let down = null;
canvas.addEventListener("pointerdown", (e) => {
  down = { x: e.clientX, y: e.clientY };
  if (walk.on) { walk.drag = { x: e.clientX, y: e.clientY, yaw: walk.yaw, pitch: walk.pitch }; canvas.setPointerCapture(e.pointerId); }
});
canvas.addEventListener("pointermove", (e) => {
  if (walk.on && walk.drag) {
    walk.yaw = walk.drag.yaw + (e.clientX - walk.drag.x) * 0.005;
    walk.pitch = Math.max(-1.2, Math.min(1.2, walk.drag.pitch - (e.clientY - walk.drag.y) * 0.005));
  }
  hover(e);
});
canvas.addEventListener("pointerleave", () => ($("#tip").style.display = "none"));
canvas.addEventListener("pointerup", (e) => {
  walk.drag = null;
  if (!down || Math.hypot(e.clientX - down.x, e.clientY - down.y) > 4 || tctl.dragging) return;
  click(e);
});
function click(e) {
  if (!S.plan) return;
  const edit = app.classList.contains("mode-edit");
  if (edit && S.placing) {
    const p = floorPoint(e); if (!p) return;
    pushHist();
    const [w, d] = DEFAULT_SIZE[S.placing] || [1, 0.6];
    const it = { uid: S.uid++, type: S.placing, center: [p.x, p.z], width: w, depth: d, height: heightOf(S.placing), angle: 0, frontRel: Math.PI / 2, source: "user", confidence: 1, category: catOf(S.placing) };
    if (S.snap) backToWallIfClose(it);
    S.furn.push(it); S.dirty = true;
    rebuildFurniture(); select({ kind: "furn", uid: it.uid });
    if (!e.shiftKey) { S.placing = null; renderLib(); $("#tip").style.display = "none"; }
    return;
  }
  if (S.tool === "measure" && !edit) {
    const h = pick(e, [building, furnG]); if (h) measureClick(h.point); return;
  }
  const hf = furnG.visible ? pick(e, [furnG]) : null;
  if (hf) {
    let o = hf.object; while (o && o.userData.uid === undefined) o = o.parent;
    select({ kind: "furn", uid: o.userData.uid });
    if (edit) showCtx(e);
    return;
  }
  const hb = pick(e, [building]);
  if (!hb) { select(null); return; }
  const p = [hb.point.x, hb.point.z];
  const st = (S.plan.stairs || []).find((s) => inPoly(p, s.outline));
  if (st) {
    select({ kind: "stair", s: st }); return;
  }
  const nw = nearestWall(p);
  if (hb.object.userData.layer !== "floor" && nw && nw.d < nw.w.thickness + 0.25) select({ kind: "wall", w: nw.w });
  else { const r = roomAt(p); select(r ? { kind: "room", r } : null); }
}
function backToWallIfClose(it) { const nw = nearestWall(it.center); if (nw && nw.d < nw.w.thickness / 2 + it.depth + 0.4) backToWall(it); }
function hover(e) {
  const tip = $("#tip");
  if (!S.plan || walk.on || tctl.dragging) { tip.style.display = "none"; return; }
  const hf = furnG.visible ? pick(e, [furnG]) : null;
  let txt = "";
  if (hf) { let o = hf.object; while (o && o.userData.uid === undefined) o = o.parent; const it = itemOf(o.userData.uid); if (it) txt = `${label(it.type)} · ${it.width.toFixed(2)} × ${it.depth.toFixed(2)} m`; }
  else if (S.placing) txt = `Click to place ${label(S.placing)} (Shift: place several)`;
  else if (S.tool === "measure") { const h = pick(e, [building]); if (h) txt = `x ${h.point.x.toFixed(2)}  y ${h.point.z.toFixed(2)}  z ${h.point.y.toFixed(2)}`; }
  if (!txt) { tip.style.display = "none"; return; }
  const r = $("#view").getBoundingClientRect();
  tip.textContent = txt; tip.style.display = "block";
  tip.style.left = e.clientX - r.left + 14 + "px"; tip.style.top = e.clientY - r.top + 14 + "px";
}
function showCtx(e) {
  const r = $("#view").getBoundingClientRect(), c = $("#ctx");
  c.classList.remove("hidden");
  c.style.left = Math.min(e.clientX - r.left + 10, r.width - 330) + "px";
  c.style.top = Math.min(e.clientY - r.top + 16, r.height - 130) + "px";
}
function ctxAct(a) {
  const it = S.sel?.kind === "furn" && itemOf(S.sel.uid);
  if (!it) return;
  pushHist();
  if (a === "rot") it.angle += Math.PI / 2;
  if (a === "wall") backToWall(it);
  if (a === "dup") { const n = { ...it, uid: S.uid++, center: [it.center[0] + 0.3, it.center[1] + 0.3], source: "user" }; S.furn.push(n); S.dirty = true; rebuildFurniture(); select({ kind: "furn", uid: n.uid }); return; }
  if (a === "del") { S.furn = S.furn.filter((f) => f !== it); S.dirty = true; rebuildFurniture(); select(null); return; }
  S.dirty = true; placeMesh(meshOf(it.uid), it); updateSelBox(); renderProps();
}
$$("#ctx span").forEach((s) => (s.onclick = () => ctxAct(s.dataset.c)));

// ------------------------------------------------------------------ panels
const fmt = (v, n = 3) => (v === undefined || v === null ? "—" : (+v).toFixed(n));
function prop(k, v) { return `<div class="prop"><span>${k}</span><span>${v}</span></div>`; }
function renderProps() {
  const P = $("#props"), sel = S.sel, edit = app.classList.contains("mode-edit");
  if (!sel) { $("#selTitle").textContent = "Selection"; P.innerHTML = `<p class="muted small">${S.plan ? (edit ? "Click a furniture item to edit it, or add one from the left." : "Click a wall, room or furniture item.") : "Nothing loaded yet."}</p>`; return; }
  if (sel.kind === "wall") {
    const w = sel.w, L = Math.hypot(w.end[0] - w.start[0], w.end[1] - w.start[1]);
    const ops = (S.plan.openings || []).filter((o) => o.wall === w.id);
    $("#selTitle").textContent = "Selection · Wall";
    P.innerHTML = prop("Type", `${w.external ? "External" : "Internal"} ${Math.round(w.thickness * 1000)}`) + prop("Length", fmt(L) + " m") +
      prop("Thickness", Math.round(w.thickness * 1000) + " mm") + prop("Height", fmt(w.height) + " m") +
      prop("Doors / windows", `${ops.filter((o) => o.type === "door").length} / ${ops.filter((o) => o.type !== "door").length}`) +
      prop("Room", roomAt([(w.start[0] + w.end[0]) / 2 + 0.3, (w.start[1] + w.end[1]) / 2 + 0.3])?.name || "—") + prop("Wall id", w.id);
    return;
  }
  if (sel.kind === "stair") {
    const s = sel.s, src = s.sources || {};
    const kindName = { entrance: "Entrance steps", interior: "Stair", terrace: "Stepped area" }[s.kind] || "Stair";
    $("#selTitle").textContent = "Selection · " + kindName;
    P.innerHTML = prop("Shape", s.shape) + prop("Steps (risers)", s.risers) + prop("Riser", Math.round(s.riser * 1000) + " mm") +
      prop("Tread", Math.round(s.tread * 1000) + " mm") +
      prop("Width", s.shape === "flared" ? `${fmt(s.width_low, 2)} → ${fmt(s.width_high, 2)} m` : fmt(s.width, 2) + " m") +
      prop("Run", fmt(s.run, 2) + " m") + prop("Levels", `${s.base_z >= 0 ? "+" : ""}${fmt(s.base_z, 2)} → ${s.top_z >= 0 ? "+" : ""}${fmt(s.top_z, 2)} m`) +
      prop("Direction", s.direction || "—") +
      Object.entries(src).map(([k, v]) => prop("From · " + k, `<span class="muted">${v}</span>`)).join("") +
      (s.warnings || []).map((w) => `<p class="small" style="color:#ffb86b">Check: ${w}</p>`).join("") +
      `<div class="stair-edit">
         <p class="small muted">Is it right? Set what these steps really do:</p>
         <div class="seg">
           <button class="btn" data-smode="down" title="Steps go down from the floor to open ground (sandpit, sunken court); the ground beside them is cut out of the floor">Down</button>
           <button class="btn" data-smode="up" title="Raised seating on the floor">Up</button>
           <button class="btn" data-smode="stair" title="A stair to the floor above">Stair</button>
           <button class="btn" data-smode="flat" title="Not steps: plain floor">Flat</button>
         </div>
         <label class="small muted">Height between floor and the other level (m)
           <input id="stairDepth" type="number" min="0.05" max="6" step="0.05" value="${(Math.abs(s.top_z - s.base_z) || S.plan.levels?.plinth || 0.45).toFixed(2)}">
         </label>
       </div>`;
    $$("#props [data-smode]").forEach((b) => (b.onclick = () => editStair(s, b.dataset.smode)));
    return;
  }
  if (sel.kind === "room") {
    const r = sel.r;
    $("#selTitle").textContent = "Selection · Room";
    P.innerHTML = prop("Name", r.name) + prop("Area", fmt(r.area_m2, 2) + " m²") + prop("Furniture", S.furn.filter((f) => roomAt(f.center) === r).length);
    return;
  }
  const it = itemOf(sel.uid);
  if (!it) return;
  $("#selTitle").textContent = "Selection · " + label(it.type);
  const deg = ((THREE.MathUtils.radToDeg(it.angle) % 360) + 360) % 360;
  const src = it.source === "user" ? "added by you" : it.source === "furniture detector" ? `detector ${Math.round(it.confidence * 100)}%` : "floor-plan model";
  if (!edit) {
    P.innerHTML = prop("Type", label(it.type)) + prop("Width", fmt(it.width) + " m") + prop("Depth", fmt(it.depth) + " m") + prop("Height", fmt(it.height) + " m") +
      prop("Rotation", deg.toFixed(1) + "°") + prop("Room", roomAt(it.center)?.name || "—") + prop("Source", src);
    return;
  }
  const opts = [...new Set([...Object.keys(CATALOG), ...LIB, it.type])].map((t) => `<option value="${t}" ${t === it.type ? "selected" : ""}>${label(t)}</option>`).join("");
  const num = (k, v, st = "0.01") => `<input type="number" step="${st}" data-k="${k}" value="${(+v).toFixed(st === "1" ? 0 : 3)}">`;
  P.innerHTML = `<div class="prop"><span>Type</span><select data-k="type">${opts}</select></div>` +
    `<div class="prop"><span>Width (m)</span>${num("width", it.width)}</div><div class="prop"><span>Depth (m)</span>${num("depth", it.depth)}</div>` +
    `<div class="prop"><span>Height (m)</span>${num("height", it.height)}</div><div class="prop"><span>Rotation (°)</span>${num("rot", deg, "1")}</div>` +
    `<div class="prop"><span>X (m)</span>${num("x", it.center[0])}</div><div class="prop"><span>Y (m)</span>${num("y", it.center[1])}</div>` +
    prop("Room", roomAt(it.center)?.name || "—") + prop("Source", src);
  P.querySelectorAll("[data-k]").forEach((inp) => (inp.onchange = () => {
    pushHist();
    const k = inp.dataset.k, v = inp.value;
    if (k === "type") { it.type = v; it.category = catOf(v); it.height = heightOf(v); }
    else if (k === "rot") it.angle = THREE.MathUtils.degToRad(+v);
    else if (k === "x") it.center[0] = +v; else if (k === "y") it.center[1] = +v;
    else it[k] = Math.max(0.05, +v);
    S.dirty = true; placeMesh(meshOf(it.uid), it); updateSelBox(); renderProps(); renderPlaced();
  }));
}
function renderPlaced() {
  $("#placedN").textContent = S.furn.length;
  $("#placed").innerHTML = S.furn.map((f) => {
    const tag = f.source === "user" ? "yours" : f.source === "furniture detector" ? Math.round(f.confidence * 100) + "%" : "model";
    return `<div class="prop ${S.sel?.uid === f.uid ? "sel" : ""}" data-u="${f.uid}"><span>${label(f.type)} · ${roomAt(f.center)?.name || "—"}</span><span>${tag}</span></div>`;
  }).join("");
  $$("#placed .prop").forEach((d) => (d.onclick = () => select({ kind: "furn", uid: +d.dataset.u })));
}
function renderRooms() {
  const rs = S.plan?.rooms || [];
  $("#roomList").innerHTML = rs.map((r) => `<div class="prop ${S.sel?.r === r ? "sel" : ""}" data-r="${r.id}"><span>${r.name}</span><span>${fmt(r.area_m2, 2)} m²</span></div>`).join("") || `<p class="muted small">No rooms yet.</p>`;
  $$("#roomList .prop").forEach((d) => (d.onclick = () => select({ kind: "room", r: rs.find((r) => r.id === +d.dataset.r) })));
}
function renderLib() {
  $("#lib").innerHTML = LIB.map((t) => `<div data-t="${t}" class="${S.placing === t ? "on" : ""}">${label(t)}</div>`).join("");
  $$("#lib div").forEach((d) => (d.onclick = () => { S.placing = S.placing === d.dataset.t ? null : d.dataset.t; renderLib(); if (S.placing && S.tab !== "plan") setTab("plan"); }));
}
function renderStats(st) {
  const p = S.plan, pr = p.precision || {};
  $("#s_cov").textContent = p.coverage !== undefined ? (p.coverage * 100).toFixed(1) + "%" : "—";
  $("#s_cov").className = p.coverage >= 0.97 ? "ok" : "";
  const oneScale = (pr.report || []).some((r) => r.status === "superseded");   // one scale for the sheet: all were read
  $("#s_dim").textContent = pr.dimensions_read ? (oneScale ? `${pr.dimensions_read}` : `${pr.dimensions_exact}/${pr.dimensions_read}`) : "0";
  const dimLbl = $("#s_dim").nextElementSibling; if (dimLbl) dimLbl.textContent = oneScale ? "dims read" : "dims matched";
  $("#s_rooms").textContent = (p.rooms || []).length;
  $("#s_furn").textContent = S.furn.length;
  $("#s_scale").textContent = (st?.exact_scale ? "Scale: exact, from the DXF drawing units." : "Scale: " + (pr.scale_source || p.mapping?.source || "")) +
    (p.levels?.plinth ? ` · Plinth ${p.levels.plinth.toFixed(2)} m (${p.levels.plinth_source})` : "") +
    ((p.stairs || []).some((s) => (s.warnings || []).length) ? " · some stairs need a check: click them" : "");
}

// ------------------------------------------------------------------ history
const snap = () => JSON.stringify(S.furn);
function pushHist() { S.hist.push(snap()); if (S.hist.length > 100) S.hist.shift(); S.fut = []; }
function undo() { if (!S.hist.length) return; S.fut.push(snap()); S.furn = JSON.parse(S.hist.pop()); S.dirty = true; rebuildFurniture(); }
function redo() { if (!S.fut.length) return; S.hist.push(snap()); S.furn = JSON.parse(S.fut.pop()); S.dirty = true; rebuildFurniture(); }

// ------------------------------------------------------------------ modes
function enterEdit() {
  S.beforeEdit = snap(); S.hist = []; S.fut = []; S.dirty = false;
  app.classList.replace("mode-view", "mode-edit");
  $("#crumb").textContent = `/ ${S.file?.name || "plan"} / Furniture`;
  setTool("orbit"); setTab("plan"); renderLib(); select(null);
  LAYERS.furniture = true; applyLayers(); resize();
}
function leaveEdit() {
  app.classList.replace("mode-edit", "mode-view");
  $("#crumb").textContent = `/ ${S.file?.name || "plan"}`;
  S.placing = null; tctl.detach(); select(null); setTab("3d"); resize();
}
async function editStair(s, mode) {
  const btns = $$("#props [data-smode]"); btns.forEach((x) => (x.disabled = true));
  try {
    const depth = parseFloat($("#stairDepth")?.value);
    const r = await fetch(`${API}/api/stair/${S.job}`, { method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ id: s.id, mode, depth: isFinite(depth) ? depth : null }) });
    if (!r.ok) throw new Error((await r.json()).detail || r.statusText);
    S.plan = await (await fetch(`${API}/api/files/${S.job}/plan.json?t=${Date.now()}`)).json();
    await loadBuilding(`${API}/api/files/${S.job}/shell.glb`);
    const again = (S.plan.stairs || []).find((x) => x.id === s.id);
    select(again ? { kind: "stair", s: again } : null);
    renderStats();
    toast({ down: "Steps go down to open ground: floor and plinth cut out", up: "Raised seating on the floor",
            stair: "Stair to the floor above", flat: "Steps removed: plain floor" }[mode] + " · IFC, GLB, plan.json and zip rebuilt");
    $("#saveState").textContent = "stairs edited · exports up to date";
  } catch (err) { toast("Stair change failed: " + err.message, true); btns.forEach((x) => (x.disabled = false)); }
}
$("#btnEdit").onclick = enterEdit;
$("#btnCancel").onclick = () => { S.furn = JSON.parse(S.beforeEdit || "[]"); rebuildFurniture(); leaveEdit(); };
$("#btnApply").onclick = async () => {
  const b = $("#btnApply"); b.disabled = true; b.textContent = "Exporting…";
  try {
    const body = { furniture: S.furn.map((f) => ({
      type: f.type, center: f.center, width: f.width, depth: f.depth, height: f.height,
      angle_deg: THREE.MathUtils.radToDeg(f.angle), front: [Math.cos(f.angle + f.frontRel), Math.sin(f.angle + f.frontRel)],
      source: f.source, confidence: f.confidence })) };
    const r = await fetch(`${API}/api/export/${S.job}`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
    if (!r.ok) throw new Error((await r.json()).detail || r.statusText);
    S.plan = await (await fetch(`${API}/api/files/${S.job}/plan.json?t=${Date.now()}`)).json();
    toast(`Saved · IFC, GLB, plan.json and zip rebuilt with ${S.furn.length} items`);
    $("#saveState").textContent = "furniture edited · exports up to date";
    leaveEdit(); renderStats();
  } catch (err) { toast("Export failed: " + err.message, true); }
  b.disabled = false; b.textContent = "Apply & re-export";
};
$$("#tbEdit [data-etool]").forEach((i) => (i.onclick = () => {
  S.etool = i.dataset.etool;
  $$("#tbEdit [data-etool]").forEach((j) => j.classList.toggle("on", j === i));
  if (S.sel?.kind === "furn") select(S.sel);
}));
$$("#tbEdit [data-etog]").forEach((i) => (i.onclick = () => { S.snap = !S.snap; i.classList.toggle("on", S.snap); setTctlMode(); }));
$$("#tbEdit [data-act]").forEach((i) => (i.onclick = () => (i.dataset.act === "undo" ? undo() : redo())));
$$("#tbView [data-tool]").forEach((i) => (i.onclick = () => S.plan && setTool(i.dataset.tool)));
$$("#tbView [data-act]").forEach((i) => (i.onclick = () => {
  if (i.dataset.act === "fit") fit();
  if (i.dataset.act === "layers") { renderLayersPop(); $("#layersPop").classList.toggle("hidden"); i.classList.toggle("lit"); }
}));
$$("#tbView [data-tog]").forEach((i) => (i.onclick = () => {
  const k = i.dataset.tog; S[k] = !S[k]; i.classList.toggle("lit", S[k]);
  if (k === "xray") applyXray();
  if (k === "edges") applyEdges();
  if (k === "section") { renderer.clippingPlanes = S.section ? [cutPlane] : []; $("#sectionUi").classList.toggle("hidden", !S.section); }
}));
$("#cutH").oninput = (e) => { cutPlane.constant = +e.target.value; $("#cutV").textContent = (+e.target.value).toFixed(2) + " m"; };
$$("#tabs span").forEach((s) => (s.onclick = () => S.plan && setTab(s.dataset.t)));

// ------------------------------------------------------------------ input / convert
const drop = $("#drop");
["dragenter", "dragover"].forEach((ev) => drop.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.add("over"); }));
["dragleave", "drop"].forEach((ev) => drop.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.remove("over"); }));
drop.addEventListener("drop", (e) => e.dataTransfer.files[0] && setFile(e.dataTransfer.files[0]));
$("#file").onchange = (e) => e.target.files[0] && setFile(e.target.files[0]);
function setFile(f) {
  const ext = f.name.split(".").pop().toLowerCase();
  if (!["dxf", "dwg", "pdf", "png", "jpg", "jpeg", "webp", "tif", "tiff", "bmp"].includes(ext)) return toast("Unsupported file: ." + ext, true);
  if (f.size > 50 * 2 ** 20) return toast("File is larger than 50 MB", true);
  S.file = f;
  $("#fileCard").classList.remove("hidden");
  $("#fileExt").textContent = ext.toUpperCase().slice(0, 4);
  $("#fileName").textContent = f.name;
  const note = { dxf: "vector · exact scale", dwg: "save as DXF first", pdf: "page 1 · 300 dpi" }[ext] || "raster image";
  $("#fileMeta").textContent = `${note} · ${(f.size / 2 ** 20).toFixed(1)} MB`;
  $("#fileOk").textContent = ext === "dwg" ? "!" : "✓"; $("#fileOk").className = ext === "dwg" ? "danger" : "ok";
  $$(".dxf-only").forEach((d) => d.classList.toggle("hidden", ext !== "dxf"));
  $("#btnConvert").disabled = ext === "dwg";
  $("#crumb").textContent = "/ " + f.name;
  if (ext === "dwg") toast("DWG is closed: in AutoCAD run SAVEAS → DXF and upload that.", true);
}
const furnState = { need: true, mode: "copy" };
$("#togFurn").onclick = () => {
  furnState.need = !furnState.need;
  $("#togFurn").classList.toggle("on", furnState.need); $("#togFurn").setAttribute("aria-checked", furnState.need);
  $("#furnMode").classList.toggle("off", !furnState.need);
  $("#furnHelp").textContent = !furnState.need ? "No furniture: walls, doors, windows, rooms and stairs only." : furnState.mode === "copy" ? "Furniture drawn on the plan is copied exactly; edit it after conversion." : "Start empty and place furniture yourself in the editor.";
};
$$("#furnMode span").forEach((s) => (s.onclick = () => {
  furnState.mode = s.dataset.v; $$("#furnMode span").forEach((t) => t.classList.toggle("on", t === s));
  $("#togFurn").onclick(); $("#togFurn").onclick();
}));
const val = (id) => { const v = $(id).value.trim(); return v === "" ? null : +v; };
$("#btnConvert").onclick = async () => {
  if (!S.file) return;
  const opt = {
    furniture: furnState.need ? furnState.mode : "none", building_width: val("#o_bw"), wall_height: val("#o_wh") ?? 3,
    ext_wall: val("#o_ew"), int_wall: val("#o_iw"), door_height: val("#o_dh") ?? 2.1, head: val("#o_dh") ?? 2.1,
    sill: val("#o_ws") ?? 0.9, dxf_units: $("#o_units").value || null, plinth: val("#o_pl"),
  };
  const fd = new FormData(); fd.append("file", S.file); fd.append("options", JSON.stringify(opt));
  $("#btnConvert").disabled = true; $("#prog").classList.remove("hidden"); progress(0.02, "Uploading");
  try {
    const r = await fetch(API + "/api/convert", { method: "POST", body: fd });
    if (!r.ok) throw new Error((await r.json()).detail || r.statusText);
    const { job } = await r.json();
    let st;
    let misses = 0;
    for (;;) {
      let r2;
      try { r2 = await fetch(`${API}/api/status/${job}?wait=25`); }      // answers when progress changes
      catch (e) { r2 = null; }
      if (!r2 || r2.status >= 500) {                                      // network blip / engine busy: wait, ask again
        if (++misses > 12) throw new Error("lost contact with the engine (it may have restarted or run out of memory)");
        progress(st ? st.progress : 0.05, "Waiting for the engine…");
        await new Promise((res) => setTimeout(res, 5000));
        continue;
      }
      if (r2.status === 404) throw new Error("the engine restarted and lost this job — please convert again");
      misses = 0;
      st = await r2.json();
      progress(st.progress, st.msg);
      if (st.state === "done" || st.state === "error") break;
    }
    if (st.state === "error") { $("#report").textContent = st.report || ""; throw new Error(st.msg); }
    await loadJob(job, st);
    if (furnState.need && furnState.mode === "manual") enterEdit();
  } catch (err) { toast("Could not convert: " + err.message, true); progress(0, "Failed: " + err.message); }
  $("#btnConvert").disabled = false;
};
function progress(p, m) { $("#progBar").style.width = Math.round(p * 100) + "%"; $("#progMsg").textContent = m || ""; }

async function loadJob(job, st) {
  S.job = job; S.rawLoaded = false;
  S.plan = await (await fetch(`${API}/api/files/${job}/plan.json?t=${Date.now()}`)).json();
  S.furn = (S.plan.furniture || []).map(fromPlan);
  await loadBuilding(`${API}/api/files/${job}/shell.glb`);
  drawGrids(); rebuildFurniture(); renderStats(st); renderRooms();
  $("#report").textContent = st?.report || "";
  $("#traceImg").src = `${API}/api/files/${job}/preview.png?t=${Date.now()}`;
  $$("#dls .dl").forEach((a) => { a.href = `${API}/api/files/${job}/${a.dataset.f}`; a.setAttribute("download", a.dataset.f); a.classList.remove("disabled"); });
  const ex = $("#btnExport"); ex.href = `${API}/api/files/${job}/floorplan_3d.zip`; ex.setAttribute("download", "floorplan_3d.zip"); ex.classList.remove("disabled");
  $("#btnEdit").disabled = false; $("#empty").classList.add("hidden");
  $("#saveState").textContent = "";
  setTab("3d"); setTool("orbit"); fit(); select(null);
  window.__planforge = { ready: true, walls: S.plan.walls.length, furniture: S.furn.length,
    stairs: (S.plan.stairs || []).length, selectStair: (i) => select({ kind: "stair", s: S.plan.stairs[i] }), fit };
}

function toast(m, err = false) {
  const t = $("#toast"); t.textContent = m; t.classList.toggle("err", err); t.classList.remove("hidden");
  clearTimeout(toast.h); toast.h = setTimeout(() => t.classList.add("hidden"), 4200);
}

// ------------------------------------------------------------------ loop
let last = performance.now();
function loop(now) {
  const dt = Math.min(0.05, (now - last) / 1000); last = now;
  if (walk.on) walkStep(dt); else controls.update();
  if (selBox.visible) updateSelBox();
  renderer.render(scene, camera);
  // view cube follows the camera direction
  const dir = new THREE.Vector3(); camera.getWorldDirection(dir);
  cubeC.position.copy(dir.multiplyScalar(-3.2)); cubeC.up.copy(camera.up); cubeC.lookAt(0, 0, 0);
  cubeR.render(cubeS, cubeC);
  requestAnimationFrame(loop);
}
fetch(API + "/api/catalog").then((r) => r.json()).then((c) => (CATALOG = c)).catch(() => {
  $("#progMsg").textContent = "Conversion engine is offline right now: start it, or open the site with ?api=<engine URL>";
  $("#prog").classList.remove("hidden");
});
resize(); fit(); requestAnimationFrame(loop);
// deep link: /?job=<id> reopens a finished conversion
const qj = new URLSearchParams(location.search).get("job");
if (qj) fetch(`${API}/api/status/${qj}`).then((r) => r.ok && r.json()).then((st) => st && st.state === "done" && loadJob(qj, st)).catch(() => {});


// ------------------------------------------------------------------ AutoCAD-style command line + coordinates
const CMD = {
  ORBIT: '#tbView [data-tool="orbit"]', PAN: '#tbView [data-tool="pan"]', ZOOM: '#tbView [data-tool="zoom"]',
  FIT: '#tbView [data-act="fit"]', SECTION: '#tbView [data-tog="section"]', MEASURE: '#tbView [data-tool="measure"]',
  DIST: '#tbView [data-tool="measure"]', WALK: '#tbView [data-tool="walk"]', LAYERS: '#tbView [data-act="layers"]',
  XRAY: '#tbView [data-tog="xray"]', EDGES: '#tbView [data-tog="edges"]', PLAN: '#tabs [data-t="plan"]',
  "3D": '#tabs [data-t="3d"]', TRACE: '#tabs [data-t="trace"]', HOME: "#btnHome",
};
const cmdIn = $("#cmd"), cmdOut = $("#cmdOut");
if (cmdIn) {
  cmdIn.addEventListener("keydown", (e) => {
    if (e.key !== "Enter") return;
    const c = cmdIn.value.trim().toUpperCase().replace(/^_/, "");
    cmdIn.value = "";
    if (!c) return;
    const el = CMD[c] && document.querySelector(CMD[c]);
    if (el) { el.click(); cmdOut.textContent = "Command: " + c; }
    else cmdOut.textContent = "Unknown command: " + c + " · try ORBIT, PAN, FIT, SECTION, MEASURE, WALK, LAYERS, PLAN, 3D";
  });
}
const coordsEl = $("#coords");
const ground = new THREE.Plane(new THREE.Vector3(0, 1, 0), 0);
canvas.addEventListener("pointermove", (e) => {
  if (!coordsEl) return;
  const r = canvas.getBoundingClientRect();
  ray.setFromCamera(new THREE.Vector2(((e.clientX - r.left) / r.width) * 2 - 1, -((e.clientY - r.top) / r.height) * 2 + 1), camera);
  const p = new THREE.Vector3();
  if (ray.ray.intersectPlane(ground, p)) coordsEl.textContent = `${p.x.toFixed(3)}, ${p.z.toFixed(3)}, 0.000`;
});
