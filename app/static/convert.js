// Conversion page: uploads the plan, follows the engine's progress, then opens the viewer.
import { get, put, clear } from "./handoff.js";

const q = new URLSearchParams(location.search);
const API = (q.get("api") || window.PLANFORGE_API || "").replace(/\/$/, "");
const $ = (s) => document.querySelector(s);
const scene = $("#scene");

// ---- the slideshow: real buildings, photographs from Wikimedia Commons (loaded by the visitor's browser) ----
const commons = (f) => "https://commons.wikimedia.org/wiki/Special:FilePath/" + encodeURIComponent(f) + "?width=2200";
const BUILDINGS = [
  { f: "Fallingwater, also known as the Edgar J. Kaufmann, Sr., residence, Pennsylvania, by Carol M. Highsmith.jpg",
    name: "Fallingwater", meta: "Frank Lloyd Wright, Pennsylvania, 1939", credit: "Photo: Carol M. Highsmith, Library of Congress, public domain" },
  { f: "Lotus Temple-Panoroma-Visit During WCI 2016- IMG 6471.jpg",
    name: "Lotus Temple", meta: "Fariborz Sahba, New Delhi, 1986", credit: "Photo: Bijay Chaurasia, Wikimedia Commons, CC BY-SA" },
  { f: "Farnsworth House 2006.jpg",
    name: "Farnsworth House", meta: "Ludwig Mies van der Rohe, Plano, Illinois, 1951", credit: "Photo: Carol M. Highsmith, Library of Congress, public domain" },
  { f: "Taj Mahal (Edited).jpeg",
    name: "Taj Mahal", meta: "Ustad Ahmad Lahauri (attributed), Agra, 1653", credit: "Photo: Yann, Wikimedia Commons, CC BY-SA" },
  { f: "Gamble House 2016-1.jpg",
    name: "Gamble House", meta: "Greene and Greene, Pasadena, 1908", credit: "Photo: Jim Heaphy, Wikimedia Commons, CC BY-SA 3.0" },
];
const ready = [];           // slides whose photograph has loaded, in display order
let at = -1, timer = null;
const KB = [[-1.4, -.8], [1.2, -1], [-1, 1], [1.4, .6], [0, -1.2]];

function addSlide(b, i) {
  const im = new Image(); im.alt = ""; im.decoding = "async"; im.referrerPolicy = "no-referrer";
  im.onload = () => {
    const el = document.createElement("div"); el.className = "ph";
    el.style.setProperty("--kx", KB[i % KB.length][0] + "%"); el.style.setProperty("--ky", KB[i % KB.length][1] + "%");
    el.append(im); $("#photos").append(el);
    ready.push({ el, b });
    if (ready.length === 1) { show(0); timer = setInterval(() => show(at + 1), 7000); }
  };
  im.onerror = () => {};     // a photograph that cannot load is simply skipped
  im.src = commons(b.f);
}
function show(n) {
  if (!ready.length) return;
  const prev = ready[at];
  at = n % ready.length;
  const s = ready[at];
  if (prev && prev !== s) prev.el.classList.remove("on");
  // restart the slow zoom on the incoming photo
  const img = s.el.firstChild; img.style.animation = "none"; void img.offsetWidth; img.style.animation = "";
  s.el.classList.add("on");
  const plate = $("#plate"); plate.classList.remove("swap"); void plate.offsetWidth;
  $("#meta").textContent = s.b.meta; $("#name").textContent = s.b.name; $("#credit").textContent = s.b.credit;
  plate.classList.add("swap");
}
BUILDINGS.forEach(addSlide);
setTimeout(() => { if (!ready.length) scene.classList.add("nophoto"); }, 9000);

// ---- progress ----
const STEPS = [...document.querySelectorAll("#steps li")];
let shown = 0, target = 0, raf = 0;
function paint() {
  shown += (target - shown) * 0.08;
  if (Math.abs(target - shown) < 0.0015) shown = target;
  $("#pct").textContent = Math.round(shown * 100);
  if (shown !== target) raf = requestAnimationFrame(paint); else raf = 0;
}
function progress(p, msg) {
  target = Math.max(target, Math.min(1, p || 0));          // never moves backwards
  $("#fill").style.width = Math.round(target * 100) + "%";
  if (msg) $("#msg").textContent = msg;
  const now = Math.min(STEPS.length - 1, Math.floor(target * STEPS.length));
  STEPS.forEach((li, k) => { li.classList.toggle("done", k < now || target >= 1); li.classList.toggle("now", k === now && target < 1); });
  if (!raf) raf = requestAnimationFrame(paint);
}

function back(params) {
  const p = new URLSearchParams(location.search);
  Object.entries(params).forEach(([k, v]) => p.set(k, v));
  location.replace("app.html?" + p.toString());
}
function fail(msg) {
  scene.classList.add("failed"); $("#fail").classList.remove("hidden"); $("#failMsg").textContent = "Could not convert: " + msg;
  const r = $("#retry"); r.onclick = (e) => { e.preventDefault(); back({ error: "1" }); };
  document.title = "PlanForge — conversion stopped";
}

// ---- run ----
const abort = new AbortController();
$("#cancel").onclick = async () => { abort.abort(); try { const r = await get(); if (r) await put({ file: r.file, opt: r.opt }); } catch (e) {} back({ error: "cancelled" }); };

(async () => {
  let rec;
  try { rec = await get(); } catch (e) { rec = null; }
  if (!rec || !rec.file) return back({});                    // opened directly: nothing to convert
  $("#file").textContent = rec.file.name;
  try {
    let job = rec.job;
    if (!job) {
      progress(0.02, "Uploading");
      const fd = new FormData(); fd.append("file", rec.file); fd.append("options", JSON.stringify(rec.opt || {}));
      const r = await fetch(API + "/api/convert", { method: "POST", body: fd, signal: abort.signal });
      if (!r.ok) throw new Error((await r.json().catch(() => ({}))).detail || r.statusText);
      job = (await r.json()).job;
      await put({ ...rec, job });                            // a reload resumes this job instead of starting another
    } else progress(0.05, "Reconnecting");
    let st, misses = 0;
    for (;;) {
      let r2;
      try { r2 = await fetch(`${API}/api/status/${job}?wait=25`, { signal: abort.signal }); } catch (e) { if (abort.signal.aborted) return; r2 = null; }
      if (!r2 || r2.status >= 500) {
        if (++misses > 12) throw new Error("lost contact with the engine (it may have restarted or run out of memory)");
        progress(st ? st.progress : 0.05, "Waiting for the engine");
        await new Promise((res) => setTimeout(res, 5000)); continue;
      }
      if (r2.status === 404) throw new Error("the engine restarted and lost this job. Please convert again.");
      misses = 0; st = await r2.json();
      progress(st.progress, st.msg);
      if (st.state === "done" || st.state === "error") break;
    }
    if (st.state === "error") throw new Error(st.msg);
    progress(1, "Opening your model");
    await clear().catch(() => {});
    await new Promise((res) => setTimeout(res, 1400));
    back({ job, ...(rec.opt && rec.opt.furniture === "manual" ? { edit: "1" } : {}) });
  } catch (err) {
    if (abort.signal.aborted) return;
    await put({ file: rec.file, opt: rec.opt }).catch(() => {});   // keep the plan so the workspace can restore it
    fail(err.message || String(err));
  }
})();
