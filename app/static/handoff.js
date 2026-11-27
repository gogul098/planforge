// Hands the chosen plan from the workspace page to the conversion page (a File survives navigation in IndexedDB).
const DB = "planforge", ST = "handoff";
const open = () => new Promise((res, rej) => {
  const r = indexedDB.open(DB, 1);
  r.onupgradeneeded = () => r.result.createObjectStore(ST);
  r.onsuccess = () => res(r.result); r.onerror = () => rej(r.error);
});
const tx = async (mode, fn) => { const db = await open(); return new Promise((res, rej) => { const t = db.transaction(ST, mode); const rq = fn(t.objectStore(ST)); t.oncomplete = () => res(rq && rq.result); t.onerror = () => rej(t.error); }); };
export const put = (v) => tx("readwrite", (s) => s.put(v, "pending"));
export const get = () => tx("readonly", (s) => s.get("pending"));
export const clear = () => tx("readwrite", (s) => s.delete("pending"));
