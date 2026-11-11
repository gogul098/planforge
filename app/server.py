"""PlanForge: floor plan (DXF / PDF / PNG / JPG) -> exact 3D + Revit. FastAPI backend."""
import json
import os
import shutil
import tempfile
import threading
import time
import traceback
import uuid
import zipfile
import contextlib
import io

os.environ["P3D_NO_MAIN"] = "1"
os.environ["P3D_NO_INSTALL"] = "1"
os.environ.setdefault("MPLBACKEND", "Agg")

import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

import pipeline as P

HERE = os.path.dirname(os.path.abspath(__file__))
JOBS = os.environ.get("PLANFORGE_JOBS") or os.path.join(tempfile.gettempdir(), "planforge_jobs")
os.makedirs(JOBS, exist_ok=True)
LOCK = threading.Lock()          # the pipeline uses module-level settings -> one plan at a time
STATUS = {}                      # job -> {"state", "progress", "msg", ...}
UNIT_M = {0: None, 1: 0.0254, 2: 0.3048, 4: 0.001, 5: 0.01, 6: 1.0, 14: 0.1}   # DXF $INSUNITS -> metres
OUT_FILES = ("plan.json", "model.ifc", "model.glb", "model_faithful.glb", "preview.png", "build_in_revit.py")

app = FastAPI(title="PlanForge")

from fastapi.middleware.cors import CORSMiddleware  # noqa: E402  the website may be hosted elsewhere
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


# ----------------------------------------------------------------------------- input formats
def dxf_to_png(path, out, unit_override=None, max_px=7000):
    """Rasterise a DXF exactly: every curve flattened in model units, drawn black on white.
    Returns (png, metres per pixel of the PNG)."""
    import cv2
    import ezdxf
    from ezdxf import bbox, disassemble, path as zpath

    doc = ezdxf.readfile(path)
    msp = doc.modelspace()
    upm = unit_override or UNIT_M.get(doc.header.get("$INSUNITS", 0))
    if upm is None:                           # unitless: guess from size (houses are 5-60 m)
        ext = bbox.extents(msp, fast=True)
        big = max(ext.size.x, ext.size.y) if ext.has_data else 0
        upm = 0.001 if big > 2000 else (0.01 if big > 200 else 1.0)
    polys = []
    for e in disassemble.recursive_decompose(msp):          # explodes blocks / inserts
        if e.dxftype() in ("TEXT", "MTEXT", "ATTRIB", "DIMENSION", "HATCH", "POINT", "VIEWPORT"):
            continue
        try:
            p = zpath.make_path(e)
        except Exception:
            continue
        for sub in p.sub_paths():
            pts = [(v.x, v.y) for v in sub.flattening(distance=0.002 / upm)]
            if len(pts) >= 2:
                polys.append(np.array(pts, float))
    if not polys:
        raise ValueError("DXF has no drawable lines")
    allp = np.vstack(polys)
    lo, hi = allp.min(0), allp.max(0)
    w, h = max(hi[0] - lo[0], 1e-9), max(hi[1] - lo[1], 1e-9)
    ppu = min(80 * upm, max_px / max(w, h))   # ~80 px per metre
    pad = 40
    W, H = int(np.ceil(w * ppu)) + 2 * pad, int(np.ceil(h * ppu)) + 2 * pad
    img = np.full((H, W), 255, np.uint8)
    for p in polys:
        q = np.stack([(p[:, 0] - lo[0]) * ppu + pad, H - ((p[:, 1] - lo[1]) * ppu + pad)], 1)
        cv2.polylines(img, [np.round(q * 4).astype(np.int32)], False, 0, 2, cv2.LINE_AA, shift=2)
    cv2.imwrite(out, img)
    return out, upm / ppu


def load_input(path, ext, dxf_units):
    """-> (png path, metres per pixel of that png or None)"""
    if ext == ".dxf":
        return dxf_to_png(path, path + ".png", UNIT_M.get(int(dxf_units)) if dxf_units else None)
    if ext == ".dwg":
        raise HTTPException(415, "DWG is a closed format: in AutoCAD use SAVEAS -> DXF, then upload the DXF.")
    if ext == ".pdf":
        import pymupdf
        doc = pymupdf.open(path)
        doc[0].get_pixmap(dpi=300).save(path + ".png")
        return path + ".png", None
    if ext in (".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"):
        return path, None
    raise HTTPException(415, "Unsupported file type " + ext)


# ----------------------------------------------------------------------------- conversion job
def _set(job, **kw):
    STATUS.setdefault(job, {}).update(kw)
    try:                                   # persisted so a restarted server can still serve the job
        json.dump(STATUS[job], open(os.path.join(JOBS, job, "status.json"), "w"))
    except Exception:
        pass


def _clean(text):
    return "\n".join(l for l in text.splitlines() if l.strip() and "Warning" not in l and "warn(" not in l)


def _package(work, out):
    shutil.copy(os.path.join(HERE, "build_in_revit.py"), os.path.join(out, "build_in_revit.py"))
    zpath = os.path.join(work, "floorplan_3d.zip")
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as z:
        for f in OUT_FILES:
            if os.path.exists(os.path.join(out, f)):
                z.write(os.path.join(out, f), f)


def _shell_glb(plan, out):
    """Building without furniture: the viewer draws furniture itself so it can be edited."""
    P.build_glb(plan, os.path.join(out, "shell.glb"), extra=None)


def run_job(job, src, ext, opt):
    work = os.path.join(JOBS, job)
    out = os.path.join(work, "out")
    log = io.StringIO()
    t0 = time.time()
    with LOCK:
        try:
            _set(job, state="running", progress=0.05, msg="Reading the plan")
            png, mpp_src = load_input(src, ext, opt.get("dxf_units"))
            P.OUT_DIR = out
            P.METERS_PER_PIXEL = None
            P.BUILDING_WIDTH_M = float(opt["building_width"]) if opt.get("building_width") else None
            P.WALL_HEIGHT = float(opt.get("wall_height", 3.0))
            P.DOOR_HEIGHT = float(opt.get("door_height", 2.1))
            P.WINDOW_SILL = float(opt.get("sill", 0.9))
            P.WINDOW_HEAD = float(opt.get("head", 2.1))
            P.EXTERNAL_WALL_M = float(opt["ext_wall"]) / 1000 if opt.get("ext_wall") else None
            P.INTERNAL_WALL_M = float(opt["int_wall"]) / 1000 if opt.get("int_wall") else None
            P.MAKE_FURNITURE = opt.get("furniture", "copy") == "copy"
            with contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
                _set(job, progress=0.15, msg="Detecting walls, doors, windows, rooms")
                if os.environ.get("P3D_TEST_MASKS"):                 # offline test hook
                    d = os.environ["P3D_TEST_MASKS"]
                    img, room, icon = (np.load(os.path.join(d, n + ".npy")) for n in ("img", "room", "icon"))
                    P.LAST_ORIGINAL = img
                else:
                    img, room, icon = P.predict(png)
                if mpp_src:                                          # DXF: exact scale from the drawing units
                    P.METERS_PER_PIXEL = mpp_src * P.LAST_ORIGINAL.shape[1] / img.shape[1]
                _set(job, progress=0.45, msg="Reading dimensions, solving geometry, furniture")
                P.main(img, room, icon, original=P.LAST_ORIGINAL)
            plan = json.load(open(os.path.join(out, "plan.json")))
            _shell_glb(plan, out)
            _package(work, out)
            _set(job, state="done", progress=1.0, msg="Done in {:.0f} s".format(time.time() - t0),
                 report=_clean(log.getvalue()), input_kind=ext.strip("."), exact_scale=bool(mpp_src))
        except Exception as e:
            traceback.print_exc()
            _set(job, state="error", msg=(e.detail if isinstance(e, HTTPException) else
                                          "{}: {}".format(type(e).__name__, str(e)[:300])),
                 report=_clean(log.getvalue())[-4000:])


@app.post("/api/convert")
async def convert(file: UploadFile = File(...), options: str = Form("{}")):
    ext = os.path.splitext(file.filename or "")[1].lower()
    job = uuid.uuid4().hex[:12]
    work = os.path.join(JOBS, job)
    os.makedirs(work)
    src = os.path.join(work, "input" + ext)
    with open(src, "wb") as f:
        shutil.copyfileobj(file.file, f)
    if os.path.getsize(src) > 50 * 2 ** 20:
        raise HTTPException(413, "File is larger than 50 MB")
    opt = json.loads(options or "{}")
    _set(job, state="queued", progress=0.0, msg="Waiting in queue", name=file.filename)
    threading.Thread(target=run_job, args=(job, src, ext, opt), daemon=True).start()
    return {"job": job}


@app.get("/api/status/{job}")
def status(job: str):
    job = os.path.basename(job)
    sp = os.path.join(JOBS, job, "status.json")
    if job not in STATUS and os.path.exists(sp):
        STATUS[job] = json.load(open(sp))
        if STATUS[job].get("state") in ("queued", "running"):
            STATUS[job].update(state="error", msg="Server restarted during conversion: please convert again")
    if job not in STATUS:
        raise HTTPException(404, "unknown job")
    return STATUS[job]


@app.get("/api/files/{job}/{name}")
def files(job: str, name: str):
    base = os.path.join(JOBS, os.path.basename(job))
    p = os.path.join(base, "floorplan_3d.zip") if name == "floorplan_3d.zip" else \
        os.path.join(base, "out", os.path.basename(name))
    if not os.path.exists(p):
        raise HTTPException(404, "not found")
    return FileResponse(p, filename=name, headers={"Cache-Control": "no-store"})


# ----------------------------------------------------------------------------- furniture edits
def furniture_from_edit(items):
    """[{type, center:[x,y], width, depth, angle_deg (long axis)}] -> plan.json furniture records."""
    out = []
    for i, it in enumerate(items):
        t = it["type"]
        height, cat, _ = P.FURN_SPEC.get(t, (0.8, "furniture", None))
        a = np.radians(float(it.get("angle_deg", 0.0)))
        L = np.array([np.cos(a), np.sin(a)])
        S = np.array([-L[1], L[0]])
        c = np.array(it["center"], float)
        w, d = float(it["width"]), float(it["depth"])
        corners = [c + sa * L * w / 2 + sb * S * d / 2 for sa, sb in ((-1, -1), (1, -1), (1, 1), (-1, 1))]
        front = np.array(it.get("front") or S, float)
        out.append({"id": i, "type": t, "category": cat, "source": it.get("source", "user"),
                    "confidence": it.get("confidence", 1.0), "center": [round(float(v), 4) for v in c],
                    "width": round(w, 3), "depth": round(d, 3), "height": float(it.get("height") or height),
                    "front": [round(float(v), 4) for v in front],
                    "rotation_deg": round(float(np.degrees(np.arctan2(front[1], front[0]))), 2),
                    "corners": [[round(float(x), 4), round(float(y), 4)] for x, y in corners]})
    return out


@app.post("/api/export/{job}")
def export(job: str, body: dict):
    work = os.path.join(JOBS, os.path.basename(job))
    out = os.path.join(work, "out")
    jp = os.path.join(out, "plan.json")
    if not os.path.exists(jp):
        raise HTTPException(404, "unknown job")
    plan = json.load(open(jp))
    plan["furniture"] = furniture_from_edit(body.get("furniture", []))
    json.dump(plan, open(jp, "w"), indent=1)
    with LOCK:
        fmesh = P.furniture_meshes(plan["furniture"]) if plan["furniture"] else {}
        P.build_glb(plan, os.path.join(out, "model.glb"), extra=fmesh)
        try:
            P.build_ifc(plan, os.path.join(out, "model.ifc"))
        except Exception as e:
            print("IFC export failed:", e)
        _package(work, out)
    return {"ok": True, "furniture": len(plan["furniture"])}


@app.get("/api/catalog")
def catalog():
    return {k: {"height": v[0], "category": v[1]} for k, v in P.FURN_SPEC.items()}


app.mount("/", StaticFiles(directory=os.path.join(HERE, "static"), html=True), name="static")
