// "My plans": every plan this browser sent to the engine, followed in the background on any page.
// When one finishes, the person is told (system notification if allowed, and a note on the page).
const KEY = "pf-jobs";
const API = (new URLSearchParams(location.search).get("api") || window.PLANFORGE_API || "").replace(/\/$/, "");
const listeners = new Set();

export const list = () => { try { return JSON.parse(localStorage.getItem(KEY)) || []; } catch (e) { return []; } };
const save = (a) => { try { localStorage.setItem(KEY, JSON.stringify(a.slice(0, 12))); } catch (e) {} listeners.forEach((f) => f(list())); };
export const onChange = (f) => { listeners.add(f); f(list()); return () => listeners.delete(f); };
export function upsert(job, patch) {
  const a = list(), i = a.findIndex((j) => j.job === job);
  if (i < 0) a.unshift({ job, at: Date.now(), state: "queued", progress: 0, ...patch });
  else a[i] = { ...a[i], ...patch };
  save(a);
}
export const find = (job) => list().find((j) => j.job === job);
// the plan's secret key (given only to the browser that uploaded it) goes with every request for that plan
export const auth = (job, extra = {}) => ({ ...extra, "X-Plan-Key": (find(job) || {}).key || "" });

// remove the plan from the engine (upload + outputs) and from the list
export async function discard(job) {
  const headers = auth(job);
  save(list().filter((j) => j.job !== job));
  try { await fetch(`${API}/api/jobs/${job}`, { method: "DELETE", keepalive: true, headers }); } catch (e) {}
}

export async function askToNotify() {
  if (!("Notification" in window)) return false;
  if (Notification.permission === "granted") return true;
  if (Notification.permission === "denied") return false;
  return (await Notification.requestPermission()) === "granted";
}
function notify(j) {
  const body = j.state === "done" ? "Your 3D model is ready to view and download." : "Conversion stopped: " + (j.msg || "unknown error");
  if ("Notification" in window && Notification.permission === "granted" && document.visibilityState !== "visible") {
    const n = new Notification(j.state === "done" ? `${j.name} is ready` : `${j.name} could not be converted`, { body, icon: "favicon.svg", tag: j.job });
    n.onclick = () => { window.focus(); location.href = "app.html?job=" + j.job; n.close(); };
  }
  window.dispatchEvent(new CustomEvent("pf-job-finished", { detail: j }));
}

// follow every unfinished plan; the convert page follows its own job itself and passes skip
let following = false;
export function follow(skip = null) {
  if (following) return; following = true;
  const tick = async () => {
    for (const j of list().filter((x) => (x.state === "queued" || x.state === "running") && x.job !== skip)) {
      try {
        const r = await fetch(`${API}/api/status/${j.job}`, { headers: auth(j.job) });
        if (r.status === 404) { upsert(j.job, { state: "error", msg: "the engine restarted and lost this plan" }); notify(find(j.job)); continue; }
        if (!r.ok) continue;
        const st = await r.json();
        upsert(j.job, { state: st.state, progress: st.progress, msg: st.msg });
        if (st.state === "done" || st.state === "error") notify(find(j.job));
      } catch (e) {}
    }
    setTimeout(tick, document.visibilityState === "visible" ? 6000 : 15000);
  };
  tick();
}
