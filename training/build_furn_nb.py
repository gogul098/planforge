import nbformat as nbf

cells = []
md = lambda s: cells.append(nbf.v4.new_markdown_cell(s))
code = lambda s: cells.append(nbf.v4.new_code_cell(s))

md("""# Furniture detector (rotated boxes) — FloorPlanCAD · YOLO11m-OBB · 2×T4

Detects **bed, sofa, chair, table, wardrobe, cabinets, TV unit, toilet, sink, bath, stove, fridge, washing machine** with their **rotation**, so drawn furniture can be copied exactly into the 3D / Revit model.

**Before running (one time):** Accelerator **GPU T4 x2** · Internet **On** · Add-ons → Secrets → `HF_TOKEN` (write) attached ·
then **Save Version → Save & Run All (Commit)** (runs in the background; you can close the browser).

* First run converts the dataset (~30–45 min) and caches it to your private HF dataset repo; later runs skip that.
* Checkpoints (`last.pt`, `best.pt`, `results.csv`) are pushed to your private HF model repo every epoch.
* If a run is cut off, run the notebook again — it resumes from `last.pt`. A finished run is not retrained.
* Data: FloorPlanCAD (Voxel51 mirror, CC BY-NC 4.0 — non-commercial).
""")

code('''# ===================== CONFIG =====================
import os, sys, time, json, shutil, subprocess, threading, re
NOTEBOOK_START = time.time()
MODEL_REPO_NAME = "floorplan-furniture-obb"          # private HF model repo
DATA_REPO_NAME  = "floorplancad-obb-cache"           # private HF dataset repo (converted data)
BASE_MODEL      = "yolo11m-obb.pt"                   # pretrained rotated-box detector
IMGSZ, BATCH    = 1024, 16                           # 16 = 8 per GPU; auto-retries with 8 if it does not fit
MAX_EPOCHS      = 300                                # upper bound; the time budget decides the real length
TIME_BUDGET_HOURS = 11.25
VAL_FRACTION, SEED = 0.1, 42
DS_DIR   = "/tmp/furn_ds"                            # converted dataset (not saved as notebook output)
RUNS_DIR = "/tmp/furn_runs"
RUN_NAME = "obb"
SRC_DIR  = "/kaggle/working/furn_src"
os.makedirs(SRC_DIR, exist_ok=True)
print("config ok")''')

code('''!pip install -q "ultralytics==8.3.253" "huggingface_hub>=0.25" 2>&1 | grep -v "already satisfied" | tail -2''')

code("%%writefile /kaggle/working/furn_src/furn_common.py\n" + open("/home/claude/fp/furn_common.py").read())

code('''%%writefile /kaggle/working/furn_src/train_furn.py
import json, sys
from ultralytics import YOLO
cfg = json.load(open(sys.argv[1]))
if cfg.get("resume_from"):
    YOLO(cfg["resume_from"]).train(resume=True)
else:
    YOLO(cfg["model"]).train(
        data=cfg["data"], imgsz=cfg["imgsz"], epochs=cfg["epochs"], time=cfg["hours"], batch=cfg["batch"],
        device=cfg["device"], workers=cfg["workers"], project=cfg["project"], name=cfg["name"], exist_ok=True,
        patience=60, cos_lr=True, seed=42, amp=True, plots=True,
        # line drawings: no colour jitter; any rotation / flip is valid for furniture symbols
        hsv_h=0.0, hsv_s=0.0, hsv_v=0.2, degrees=180, fliplr=0.5, flipud=0.5,
        mosaic=1.0, scale=0.5, translate=0.1, close_mosaic=10)
''')

code('''# ===================== SANITY CHECKS =====================
import urllib.request
def fail(msg): raise RuntimeError("\\n\\n*** SETUP PROBLEM: " + msg + " ***\\n")
gpus = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True).stdout.strip().splitlines()
print("\\n".join(gpus))
if len(gpus) < 2: fail("Need 2 GPUs: Settings -> Accelerator -> GPU T4 x2.")
try: urllib.request.urlopen("https://huggingface.co", timeout=20)
except Exception as e: fail(f"No internet ({e}). Settings -> Internet -> On.")
HF_TOKEN = None
try:
    from kaggle_secrets import UserSecretsClient
    HF_TOKEN = UserSecretsClient().get_secret("HF_TOKEN")
except Exception as e:
    print("Kaggle secret not readable:", e)
HF_TOKEN = HF_TOKEN or os.environ.get("HF_TOKEN")
if not HF_TOKEN: fail("Add Kaggle Secret HF_TOKEN (Add-ons -> Secrets) and attach it to this notebook.")
os.environ["HF_TOKEN"] = HF_TOKEN
from huggingface_hub import HfApi, hf_hub_download
api = HfApi(token=HF_TOKEN)
HF_USER = api.whoami()["name"]
MODEL_REPO, DATA_REPO = f"{HF_USER}/{MODEL_REPO_NAME}", f"{HF_USER}/{DATA_REPO_NAME}"
try:
    api.create_repo(MODEL_REPO, private=True, exist_ok=True)
    api.create_repo(DATA_REPO, repo_type="dataset", private=True, exist_ok=True)
    api.upload_file(path_or_fileobj=b"Furniture OBB detector for floor plans (YOLO11m-OBB, FloorPlanCAD, CC BY-NC 4.0).\\n"
                    b"Files: checkpoints/best.pt, checkpoints/last.pt, results.csv, furniture_meta.json\\n",
                    path_in_repo="README.md", repo_id=MODEL_REPO, commit_message="model card")
except Exception as e: fail(f"Token cannot write to Hugging Face: {e}")
free = shutil.disk_usage("/tmp").free / 1e9
print(f"HF OK -> https://huggingface.co/{MODEL_REPO} | free disk {free:.0f} GB")
if free < 15: fail("Need ~15 GB free disk.")''')

code('''# ===================== DATA: HF cache -> else FloorPlanCAD via FiftyOne =====================
sys.path.insert(0, SRC_DIR)
import furn_common as F, random, tarfile, collections
CACHE_NAME = "floorplancad_obb_v1.tar"
META_PATH = os.path.join(DS_DIR, "furniture_meta.json")

def restore_cache():
    try:
        if CACHE_NAME not in api.list_repo_files(DATA_REPO, repo_type="dataset"):
            return False
        print("Restoring converted dataset from HF ...")
        p = hf_hub_download(DATA_REPO, CACHE_NAME, repo_type="dataset", token=HF_TOKEN, local_dir="/tmp/hf_dl")
        os.makedirs(DS_DIR, exist_ok=True)
        with tarfile.open(p) as t: t.extractall(DS_DIR)
        os.remove(p)
        return os.path.exists(META_PATH)
    except Exception as e:
        print("cache restore failed:", e); return False

def build_from_fiftyone():
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "fiftyone"], check=True)
    import fiftyone as fo
    import fiftyone.utils.huggingface as fouh
    print("Downloading FloorPlanCAD (Voxel51 mirror) ...")
    ds = fouh.load_from_hub("Voxel51/FloorPlanCAD")
    schema = ds.get_field_schema()
    det_fields = [k for k, v in schema.items() if "Detections" in str(getattr(v, "document_type", ""))]
    if not det_fields: raise RuntimeError(f"No Detections field found. Fields: {list(schema)}")
    field = det_fields[0]
    print(f"{len(ds)} samples, using field '{field}'")
    all_labels = collections.Counter()
    for s in ds.iter_samples():
        for d in (s[field].detections if s[field] else []):
            all_labels[F.norm_label(d.label)] += 1
    print("labels in dataset:", dict(all_labels.most_common()))
    hit = [c for c in F.FURN_CLASSES if all_labels.get(c)]
    if len(hit) < 3: raise RuntimeError(f"Class names do not match. Dataset labels: {list(all_labels)}")
    missing = [c for c in F.FURN_CLASSES if c not in hit]
    if missing: print("note: classes with no examples:", missing)
    for sub in ("images/train", "images/val", "labels/train", "labels/val"):
        os.makedirs(os.path.join(DS_DIR, sub), exist_ok=True)
    rng = random.Random(SEED); stats = {}; n_obj = 0; t0 = time.time()
    samples = list(ds.iter_samples())
    for k, s in enumerate(samples):
        dets = []
        for d in (s[field].detections if s[field] else []):
            mask = d.mask
            if mask is None and getattr(d, "mask_path", None):
                import cv2; mask = cv2.imread(d.mask_path, cv2.IMREAD_GRAYSCALE)
            dets.append((d.label, d.bounding_box, mask))
        split = "val" if rng.random() < VAL_FRACTION else "train"
        stem = f"{k:05d}"
        n_obj += F.write_sample(s.filepath, dets, os.path.join(DS_DIR, f"images/{split}/{stem}.jpg"),
                                os.path.join(DS_DIR, f"labels/{split}/{stem}.txt"), stats)
        if (k + 1) % 500 == 0 or k + 1 == len(samples):
            el = time.time() - t0
            print(f"  converted {k+1}/{len(samples)} ({el/(k+1)*(len(samples)-k-1)/60:.1f} min left), {n_obj} objects")
    meta = {"classes": F.FURN_CLASSES, "train_px_per_m": F.dataset_px_per_m(stats),
            "counts": {c: len(v) for c, v in stats.items()}, "imgsz": IMGSZ, "source": "Voxel51/FloorPlanCAD"}
    json.dump(meta, open(META_PATH, "w"), indent=1)
    print("objects per class:", meta["counts"], "| drawing scale px/m:", meta["train_px_per_m"])
    print("Uploading converted dataset to HF (so future runs skip this) ...")
    tar = "/tmp/" + CACHE_NAME
    with tarfile.open(tar, "w") as t:
        for item in ("images", "labels", "furniture_meta.json"):
            t.add(os.path.join(DS_DIR, item), arcname=item)
    try:
        api.upload_file(path_or_fileobj=tar, path_in_repo=CACHE_NAME, repo_id=DATA_REPO, repo_type="dataset")
    except Exception as e:
        print("WARNING: cache upload failed (training continues):", e)
    os.remove(tar)

t0 = time.time()
if not restore_cache():
    build_from_fiftyone()
open(os.path.join(DS_DIR, "data.yaml"), "w").write(
    f"path: {DS_DIR}\\ntrain: images/train\\nval: images/val\\nnames:\\n" +
    "".join(f"  {i}: {c}\\n" for i, c in enumerate(F.FURN_CLASSES)))
meta = json.load(open(META_PATH))
n_tr = len(os.listdir(os.path.join(DS_DIR, "images/train"))); n_va = len(os.listdir(os.path.join(DS_DIR, "images/val")))
print(f"data ready in {(time.time()-t0)/60:.1f} min: {n_tr} train / {n_va} val images | scale {meta['train_px_per_m']} px/m")''')

code('''# ===================== RESUME (pull last.pt from HF) =====================
import torch
W_DIR = os.path.join(RUNS_DIR, RUN_NAME, "weights")
os.makedirs(W_DIR, exist_ok=True)
resume_from, finished = None, False
remote = set(api.list_repo_files(MODEL_REPO))
for f in ("checkpoints/last.pt", "checkpoints/best.pt"):
    if f in remote:
        p = hf_hub_download(MODEL_REPO, f, token=HF_TOKEN, local_dir="/tmp/hf_ck")
        shutil.copyfile(p, os.path.join(W_DIR, os.path.basename(f)))
        print("restored", f)
if "checkpoints/results.csv" in remote:
    shutil.copyfile(hf_hub_download(MODEL_REPO, "checkpoints/results.csv", token=HF_TOKEN, local_dir="/tmp/hf_ck"),
                    os.path.join(RUNS_DIR, RUN_NAME, "results.csv"))
last = os.path.join(W_DIR, "last.pt")
if os.path.exists(last):
    ck = torch.load(last, map_location="cpu", weights_only=False)
    if ck.get("epoch", -1) == -1:
        finished = True
        print("Training already finished on HF. Delete checkpoints/ in the repo to train again.")
    else:
        resume_from = last
        print(f"resuming from epoch {ck['epoch'] + 1}")
    del ck
else:
    print("fresh run")''')

code('''# ===================== BACKGROUND UPLOADER (every new epoch -> HF) =====================
def upload_now(msg):
    files = [(os.path.join(W_DIR, "last.pt"), "checkpoints/last.pt"), (os.path.join(W_DIR, "best.pt"), "checkpoints/best.pt"),
             (os.path.join(RUNS_DIR, RUN_NAME, "results.csv"), "checkpoints/results.csv"),
             (os.path.join(RUNS_DIR, RUN_NAME, "args.yaml"), "checkpoints/args.yaml"), (META_PATH, "furniture_meta.json")]
    from huggingface_hub import CommitOperationAdd
    stage = "/tmp/hf_stage"; os.makedirs(stage, exist_ok=True)
    ops = []
    for src, dst in files:
        if os.path.exists(src):
            tmp = os.path.join(stage, dst.replace("/", "_")); shutil.copyfile(src, tmp)
            ops.append(CommitOperationAdd(path_in_repo=dst, path_or_fileobj=tmp))
    for attempt in range(5):
        try:
            api.create_commit(MODEL_REPO, operations=ops, commit_message=msg); return True
        except Exception as e:
            print(f"[hf] upload retry {attempt+1}: {e}", flush=True); time.sleep(20 * (attempt + 1))
    return False

stop_watch = threading.Event()
def watcher():
    seen = None
    while not stop_watch.wait(90):
        try:
            m = os.path.getmtime(last) if os.path.exists(last) else None
            if m and m != seen:
                time.sleep(10)                      # let the trainer finish writing
                if upload_now("epoch checkpoint"):
                    seen = m
                    rows = open(os.path.join(RUNS_DIR, RUN_NAME, "results.csv")).read().strip().splitlines()
                    print(f"[hf] checkpoint uploaded (epoch {len(rows)-1})", flush=True)
        except Exception as e:
            print("[hf] watcher:", e, flush=True)
threading.Thread(target=watcher, daemon=True).start()
print("uploader running")''')

code('''# ===================== TRAIN on both T4s =====================
def run_training(batch):
    hours = (NOTEBOOK_START + TIME_BUDGET_HOURS * 3600 - time.time()) / 3600 - 0.35
    if hours < 0.5: print("Not enough time left in this session."); return 0, ""
    cfg = {"model": BASE_MODEL, "data": os.path.join(DS_DIR, "data.yaml"), "imgsz": IMGSZ, "epochs": MAX_EPOCHS,
           "hours": round(hours, 2), "batch": batch, "device": [0, 1], "workers": 4, "project": RUNS_DIR,
           "name": RUN_NAME, "resume_from": resume_from}
    json.dump(cfg, open(os.path.join(SRC_DIR, "run.json"), "w"))
    print(f"training: batch {batch}, time budget {hours:.2f} h, resume={bool(resume_from)}", flush=True)
    p = subprocess.Popen([sys.executable, os.path.join(SRC_DIR, "train_furn.py"), os.path.join(SRC_DIR, "run.json")],
                         stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1, cwd="/tmp")
    tail = []
    for line in p.stdout:
        line = line.replace("\\r", "\\n").rstrip().split("\\n")[-1]
        if not line or "it/s" in line and "100%" not in line: continue     # skip progress-bar spam
        print(line, flush=True); tail = (tail + [line])[-60:]
    return p.wait(), "\\n".join(tail)

if not finished:
    rc, tail = run_training(BATCH)
    if rc != 0 and "out of memory" in tail.lower() and not os.path.exists(last):
        print("\\nOut of GPU memory -> retrying with batch 8", flush=True)
        rc, tail = run_training(8)
    print("trainer exit code", rc)
stop_watch.set()
ok = upload_now("final checkpoint") if os.path.exists(last) else False
print("final upload:", ok)''')

code('''# ===================== RESULTS =====================
import csv
rp = os.path.join(RUNS_DIR, RUN_NAME, "results.csv")
if os.path.exists(rp):
    rows = list(csv.DictReader(open(rp)))
    rows = [{k.strip(): v for k, v in r.items()} for r in rows]
    key = [k for k in rows[0] if "mAP50-95" in k][0]; k50 = [k for k in rows[0] if k.endswith("mAP50(B)")][0]
    best = max(rows, key=lambda r: float(r[key]))
    print(f"epochs: {len(rows)} | best epoch {best['epoch']}: mAP50 {float(best[k50]):.3f}  mAP50-95 {float(best[key]):.3f}")
best_pt = os.path.join(W_DIR, "best.pt")
if os.path.exists(best_pt):
    from ultralytics import YOLO
    m = YOLO(best_pt)
    r = m.val(data=os.path.join(DS_DIR, "data.yaml"), imgsz=IMGSZ, batch=8, device=0, plots=False, verbose=True)
    shutil.copyfile(best_pt, "/kaggle/working/furniture_best.pt")
    print(f"\\nmodel: https://huggingface.co/{MODEL_REPO}  (checkpoints/best.pt + furniture_meta.json)")''')

nb = nbf.v4.new_notebook()
nb["cells"] = cells
nb["metadata"] = {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                  "language_info": {"name": "python"},
                  "kaggle": {"accelerator": "nvidiaTeslaT4", "isInternetEnabled": True, "isGpuEnabled": True}}
nbf.write(nb, "/home/claude/furniture_detector_kaggle.ipynb")
print("cells", len(cells))
