"""Writes sheet_views_kaggle.ipynb: trains the sheet-view detector (head 0 of the v2 spec) on composed sheets.
Run: python training/sheets/build_sheet_nb.py"""
import os

import nbformat as nbf

HERE = os.path.dirname(os.path.abspath(__file__))
TRAIN = os.path.dirname(HERE)
cells = []


def md(s):
    cells.append(nbf.v4.new_markdown_cell(s))


def code(s):
    cells.append(nbf.v4.new_code_cell(s))


md("""# PlanForge sheet-view detector (head 0)

Finds each **plan view**, the **title block** and **schedule tables** on a whole drawing sheet, so the floor-plan
model can read each floor of a G+1 / G+2 sheet separately.

Training data is composed here from real plan drawings (ResPlan renders, plus Structured3D if attached): 1-4 plans
per sheet with titles (GROUND FLOOR PLAN ...), border, title block, notes and tables, so every box is exact.

**Before running:** Settings -> Accelerator **GPU T4 x2**, Internet **On**; Kaggle Secret **HF_TOKEN** (write).
Optional: attach your **private** Structured3D dataset (Structured3D_annotation_3d.zip) for more plan styles.
Output: private HF model repo `<you>/planforge-sheet-views` with `checkpoints/best.pt`, `sheet_views.onnx`,
`sheet_views_meta.json`.""")

code('''# ===================== CONFIG =====================
import os, sys, time, json, shutil, subprocess, random, glob
NOTEBOOK_START = time.time()
MODEL_REPO_NAME = "planforge-sheet-views"     # private HF model repo
BASE_MODEL      = "yolo11s.pt"
N_SHEETS        = 6000                         # composed sheets (10% validation)
N_RESPLAN       = 5000                         # ResPlan plans rendered as the views
N_S3D           = 2500                         # Structured3D houses (only if attached; private, never uploaded)
IMGSZ, BATCH    = 1280, 16
MAX_EPOCHS      = 150
TIME_BUDGET_HOURS = 6.0
SHEET_MAX_SIDE  = 1600                          # sheets are saved at this size (the detector sees the whole sheet)
SRC_DIR, DS_DIR, PLAN_DIR, RUNS_DIR = "/kaggle/working/sheet_src", "/tmp/sheet_ds", "/tmp/sheet_plans", "/tmp/sheet_runs"
for d in (SRC_DIR, DS_DIR, PLAN_DIR, RUNS_DIR): os.makedirs(d, exist_ok=True)
print("config ok")''')

code('''!pip install -q "ultralytics==8.3.253" "huggingface_hub>=0.25" shapely onnx onnxslim 2>shapely onnx 2>&11 | grep -v "already satisfied" | tail -2''')

for path in [os.path.join(TRAIN, "fp_common.py"), os.path.join(TRAIN, "fp_render.py"),
             os.path.join(TRAIN, "fp_extra_s3d.py"), os.path.join(TRAIN, "s3d", "s3d_convert.py"),
             os.path.join(TRAIN, "s3d", "s3d_pairs.py"), os.path.join(HERE, "compose_sheets.py")]:
    code("%%writefile /kaggle/working/sheet_src/" + os.path.basename(path) + "\n" + open(path).read())

code('''%%writefile /kaggle/working/sheet_src/train_sheets.py
import json, sys
from ultralytics import YOLO
cfg = json.load(open(sys.argv[1]))
YOLO(cfg["model"]).train(
    data=cfg["data"], imgsz=cfg["imgsz"], epochs=cfg["epochs"], time=cfg["hours"], batch=cfg["batch"],
    device=cfg["device"], workers=4, project=cfg["project"], name="det", exist_ok=True, patience=40, cos_lr=True,
    seed=42, amp=True, plots=True,
    # drawing sheets are upright and carry text: no rotation / mirror; mild scale and translate only
    hsv_h=0.0, hsv_s=0.0, hsv_v=0.2, degrees=0, fliplr=0.0, flipud=0.0, mosaic=0.3, scale=0.25, translate=0.05,
    close_mosaic=10)
''')

code('''# ===================== SANITY + HF =====================
def fail(msg): raise RuntimeError("\\n\\n*** SETUP PROBLEM: " + msg + " ***\\n")
gpus = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True).stdout.strip().splitlines()
print("\\n".join(gpus)) if gpus else fail("No GPU: Settings -> Accelerator -> GPU T4 x2.")
HF_TOKEN = None
try:
    from kaggle_secrets import UserSecretsClient
    HF_TOKEN = UserSecretsClient().get_secret("HF_TOKEN")
except Exception as e:
    print("Kaggle secret not readable:", e)
HF_TOKEN = HF_TOKEN or os.environ.get("HF_TOKEN")
if not HF_TOKEN: fail("Add a Kaggle Secret named HF_TOKEN and attach it to this notebook.")
from huggingface_hub import HfApi
api = HfApi(token=HF_TOKEN)
MODEL_REPO = f"{api.whoami()['name']}/{MODEL_REPO_NAME}"
api.create_repo(MODEL_REPO, private=True, exist_ok=True)
print("HF repo:", MODEL_REPO)''')

code('''# ===================== PLAN DRAWINGS (the views) =====================
sys.path.insert(0, SRC_DIR)
import cv2, numpy as np
import fp_render
rng = random.Random(42)
t0 = time.time()
if not os.path.exists("/tmp/ResPlan/ResPlan.pkl"):
    subprocess.run(["git", "clone", "-q", "--depth", "1", "https://github.com/m-agour/ResPlan", "/tmp/ResPlan"], check=True)
    import zipfile; zipfile.ZipFile("/tmp/ResPlan/ResPlan.zip").extractall("/tmp/ResPlan")
recs = fp_render.load_resplan("/tmp/ResPlan/ResPlan.pkl")
rng.shuffle(recs)
plans = []                                       # (png path, has room labels)
for k, p in enumerate(recs[:N_RESPLAN]):
    try:
        res = fp_render.render(fp_render.resplan_plan(p), rng.uniform(40, 110), rng)
    except Exception:
        continue
    if res is None: continue
    rgb, room, _ = res
    fn = os.path.join(PLAN_DIR, f"rp{k:05d}")
    cv2.imwrite(fn + "_img.png", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)); cv2.imwrite(fn + "_room.png", room)
    plans.append(fn)
    if len(plans) % 1000 == 0: print(f"  resplan views {len(plans)}")
import fp_extra_s3d
s3d_root = fp_extra_s3d.find_root()
if s3d_root:
    import s3d_convert, s3d_pairs
    scenes = sorted(d for d in os.listdir(s3d_root) if d.startswith("scene_"))[:N_S3D]
    for sc in scenes:
        try:
            res = s3d_convert.convert(os.path.join(s3d_root, sc, "annotation_3d.json"))
            if s3d_convert.is_junk(res): continue
            img, room, _ = s3d_pairs.render(res, rng.randrange(1 << 16))
        except Exception:
            continue
        fn = os.path.join(PLAN_DIR, "s3d_" + sc)
        cv2.imwrite(fn + "_img.png", img); cv2.imwrite(fn + "_room.png", room)
        plans.append(fn)
    print("  + Structured3D views (private, not uploaded)")
else:
    print("  Structured3D not attached: ResPlan views only")
print(f"{len(plans)} plan drawings in {(time.time()-t0)/60:.1f} min")''')

code('''# ===================== COMPOSE SHEETS =====================
import compose_sheets as C
for sub in ("images/train", "images/val", "labels/train", "labels/val"):
    os.makedirs(os.path.join(DS_DIR, sub), exist_ok=True)
def load(fn):
    img = cv2.cvtColor(cv2.imread(fn + "_img.png"), cv2.COLOR_BGR2RGB)
    return img, cv2.imread(fn + "_room.png", cv2.IMREAD_UNCHANGED)
counts = {c: 0 for c in C.CLASSES}
t0 = time.time()
for k in range(N_SHEETS):
    split = "val" if k % 10 == 0 else "train"
    pool = plans[: max(1, len(plans) // 10)] if split == "val" else plans[len(plans) // 10:]   # no plan in both
    n = min(rng.choice([1, 2, 2, 2, 3, 3, 4]), len(pool))
    sheet, boxes = C.compose([load(f) for f in rng.sample(pool, n)], rng)
    H, W = sheet.shape[:2]
    s = SHEET_MAX_SIDE / max(H, W)
    sheet = cv2.resize(sheet, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    if rng.random() < 0.3:                                   # scanned look
        sheet = np.clip(sheet.astype(np.float32) + np.random.normal(0, rng.uniform(2, 6), sheet.shape), 0, 255).astype(np.uint8)
    stem = f"sheet{k:05d}"
    cv2.imwrite(os.path.join(DS_DIR, f"images/{split}/{stem}.jpg"), cv2.cvtColor(sheet, cv2.COLOR_RGB2BGR),
                [cv2.IMWRITE_JPEG_QUALITY, rng.randint(70, 95)])
    open(os.path.join(DS_DIR, f"labels/{split}/{stem}.txt"), "w").write("\\n".join(C.yolo_lines(boxes, W, H)) + "\\n")
    for b in boxes: counts[b[0]] += 1
    if (k + 1) % 1000 == 0: print(f"  sheets {k + 1}/{N_SHEETS}")
open(os.path.join(DS_DIR, "data.yaml"), "w").write(
    f"path: {DS_DIR}\\ntrain: images/train\\nval: images/val\\nnames:\\n" + "".join(f"  {i}: {c}\\n" for i, c in enumerate(C.CLASSES)))
shutil.rmtree(PLAN_DIR, ignore_errors=True)                  # Structured3D renders are deleted here
print("boxes:", counts, f"| {(time.time()-t0)/60:.1f} min")''')

code('''# ===================== TRAIN =====================
hours = (NOTEBOOK_START + TIME_BUDGET_HOURS * 3600 - time.time()) / 3600 - 0.3
ngpu = len(gpus)
cfg = {"model": BASE_MODEL, "data": os.path.join(DS_DIR, "data.yaml"), "imgsz": IMGSZ, "epochs": MAX_EPOCHS,
       "hours": round(hours, 2), "batch": BATCH if ngpu > 1 else BATCH // 2, "device": list(range(ngpu)) if ngpu > 1 else 0,
       "project": RUNS_DIR}
json.dump(cfg, open(os.path.join(SRC_DIR, "run.json"), "w"))
p = subprocess.Popen([sys.executable, os.path.join(SRC_DIR, "train_sheets.py"), os.path.join(SRC_DIR, "run.json")],
                     stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1, cwd="/tmp")
for line in p.stdout:
    line = line.replace("\\r", "\\n").rstrip().split("\\n")[-1]
    if line and not ("it/s" in line and "100%" not in line): print(line, flush=True)
print("trainer exit code", p.wait())''')

code('''# ===================== EXPORT + UPLOAD =====================
from ultralytics import YOLO
W_DIR = os.path.join(RUNS_DIR, "det", "weights")
best = os.path.join(W_DIR, "best.pt")
m = YOLO(best)
r = m.val(data=os.path.join(DS_DIR, "data.yaml"), imgsz=IMGSZ, batch=8, device=0, plots=False)
onnx_path = m.export(format="onnx", imgsz=IMGSZ, dynamic=False, simplify=True, half=False)
meta = {"classes": C.CLASSES, "imgsz": IMGSZ, "sheet_max_side": SHEET_MAX_SIDE,
        "mAP50": float(r.box.map50), "mAP50_95": float(r.box.map),
        "per_class_mAP50_95": {c: float(v) for c, v in zip(C.CLASSES, r.box.maps)},
        "data": "composed sheets from ResPlan" + (" + Structured3D (non-commercial)" if s3d_root else "")}
json.dump(meta, open("/tmp/sheet_views_meta.json", "w"), indent=1)
for src, dst in ((best, "checkpoints/best.pt"), (onnx_path, "sheet_views.onnx"), ("/tmp/sheet_views_meta.json", "sheet_views_meta.json")):
    api.upload_file(path_or_fileobj=src, path_in_repo=dst, repo_id=MODEL_REPO, commit_message="sheet-view detector")
print(json.dumps(meta, indent=1))
print(f"model: https://huggingface.co/{MODEL_REPO}")''')

nb = nbf.v4.new_notebook()
nb["cells"] = cells
nb["metadata"] = {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                  "language_info": {"name": "python"},
                  "kaggle": {"accelerator": "nvidiaTeslaT4", "isInternetEnabled": True, "isGpuEnabled": True}}
nbf.write(nb, os.path.join(HERE, "sheet_views_kaggle.ipynb"))
print("cells", len(cells))
