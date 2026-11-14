import os
import nbformat as nbf

SRC = os.path.dirname(os.path.abspath(__file__)) + "/"   # training/ (run: python training/build_nb.py)
nb = nbf.v4.new_notebook()
cells = []
md = lambda s: cells.append(nbf.v4.new_markdown_cell(s))
code = lambda s: cells.append(nbf.v4.new_code_cell(s))

md("""# Floorplan → 3D: multi-task segmentation (EfficientNet-B3 U-Net, 768 px, 2×T4)

**Predicts:** rooms (incl. walls/railings) · icons (doors, windows, **stairs**, fixtures) · wall-junction / opening heatmaps for vectorising into 3D.
**Data:** CubiCasa5K (auto-downloaded; licence CC BY-NC-SA 4.0 — non-commercial).

### Before running (one time)
1. **Settings → Accelerator → GPU T4 x2**
2. **Settings → Internet → On** (phone may need *Verify phone* on your Kaggle account)
3. **Add-ons → Secrets → Add** `HF_TOKEN` = your Hugging Face **write** token, and tick it for this notebook
4. **Save Version → Save & Run All (Commit)** — runs in the background, so you can close the phone browser.

### While it runs / after
* Progress, metrics and checkpoints appear in your private HF repo (`metrics.json`, `status.json`, `checkpoints/`).
* Kaggle caps a run at 12 h; training sizes its schedule to fit and stops cleanly before that.
* To train longer: run the same notebook again — it resumes from `checkpoints/last.pt` (raise `EXTRA_EPOCHS`).
""")

code('''# ===================== CONFIG (edit here only) =====================
import os, time, json, sys, subprocess, shutil, re
NOTEBOOK_START = time.time()

MODEL_REPO_NAME   = "floorplan-multitask-effb3-768"   # private HF model repo (checkpoints)
DATASET_REPO_NAME = "cubicasa5k-floorplan-cache"      # private HF dataset repo (preprocessed data)
TIME_BUDGET_HOURS = 11.25     # Kaggle kills sessions at 12 h; leaves a safety buffer
MAX_EPOCHS        = 150       # upper bound; schedule auto-shrinks to fit the time budget
EXTRA_EPOCHS      = 0         # on a re-run: >0 extends a finished schedule by this many epochs

ENCODER   = "tu-efficientnet_b3"
CROP      = 768               # training crop size
MAX_SIDE  = 1536              # plans are cached with longest side <= this (keeps thin walls)
SIGMA     = 3.0               # junction heatmap gaussian (px)
LR        = 1e-3              # decoder/head LR; encoder uses LR * 0.5
SEED      = 42

# ---- data beyond CubiCasa5K + rotation (v2). The first run renders / converts them once (~20-30 min) and
# stores them in the HF dataset repo; later runs restore them in minutes.
DATA_VERSION  = "v2"          # changing this starts a new run FROM the current model's weights
EXTRA_SOURCES = ["resplan", "synthetic", "floorplancad"]   # remove any you do not want
N_SYNTHETIC   = 3000          # synthetic angled buildings (wings at 30 / 45 / 60 degrees, curved fronts)
FPCAD_DOORS   = False         # FloorPlanCAD door openings rebuilt from strokes are ~75% right: off by default
SOURCE_WEIGHTS = {"cubicasa": 0.40, "resplan": 0.25, "synthetic": 0.20, "floorplancad": 0.15}
ROT_PROB      = 0.6           # share of crops turned by 30 / 45 / 60 degrees (+-3) or any angle
SCALE_RANGE   = [0.6, 1.8]    # zoom range of training crops

SRC_DIR   = "/kaggle/working/fp_src"
CKPT_DIR  = "/kaggle/working/checkpoints"
BIG_DIR   = "/tmp/fp"            # large scratch (not saved as notebook output)
CACHE_DIR = f"{BIG_DIR}/cache"
os.makedirs(SRC_DIR, exist_ok=True); os.makedirs(CKPT_DIR, exist_ok=True); os.makedirs(BIG_DIR, exist_ok=True)
print("config ok")''')

code('''!pip install -q "segmentation-models-pytorch==0.6.0" "timm==1.0.31" "huggingface_hub>=0.25" svgpathtools shapely 2>&1 | grep -v "already satisfied" | tail -3
!pip install -q hf_xet 2>&1 | tail -1 || true''')

for fn in ["fp_common.py", "fp_prep.py", "fp_render.py", "fp_extra_fpcad.py", "train.py", "viz.py", "ddp_check.py"]:
    body = open(SRC + fn).read()
    code(f"%%writefile {{SRC_DIR}}/{fn}\n".replace("{SRC_DIR}", "/kaggle/working/fp_src") + body)

code('''# ===================== SANITY CHECKS (fail fast, clear messages) =====================
import urllib.request
def fail(msg): raise RuntimeError("\\n\\n*** SETUP PROBLEM: " + msg + " ***\\n")

# GPUs
gpus = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True).stdout.strip().splitlines()
print("\\n".join(gpus))
if len(gpus) < 2: fail("Need 2 GPUs. Settings -> Accelerator -> 'GPU T4 x2'.")

# Internet
try: urllib.request.urlopen("https://huggingface.co", timeout=20)
except Exception as e: fail(f"No internet ({e}). Settings -> Internet -> On.")

# HF token (Kaggle Secret)
HF_TOKEN = None
try:
    from kaggle_secrets import UserSecretsClient
    HF_TOKEN = UserSecretsClient().get_secret("HF_TOKEN")
except Exception as e:
    print("Kaggle secret not readable:", e)
HF_TOKEN = HF_TOKEN or os.environ.get("HF_TOKEN")
if not HF_TOKEN: fail("Add a Kaggle Secret named HF_TOKEN (Add-ons -> Secrets) and attach it to this notebook.")
os.environ["HF_TOKEN"] = HF_TOKEN

from huggingface_hub import HfApi
api = HfApi(token=HF_TOKEN)
try: HF_USER = api.whoami()["name"]
except Exception as e: fail(f"HF token rejected: {e}")
MODEL_REPO, DATASET_REPO = f"{HF_USER}/{MODEL_REPO_NAME}", f"{HF_USER}/{DATASET_REPO_NAME}"
try:
    api.create_repo(MODEL_REPO, private=True, exist_ok=True)
    api.create_repo(DATASET_REPO, repo_type="dataset", private=True, exist_ok=True)
except Exception as e: fail(f"Token cannot create repos (needs WRITE access): {e}")

readme = f"""---
license: cc-by-nc-sa-4.0
tags: [floorplan, segmentation, multitask, cubicasa5k]
---
# Floorplan multi-task model ({ENCODER} U-Net, trained at {CROP}px)
Heads (27 output channels): rooms 0-11, icons 12-23, junction heatmaps 24-26 (sigmoid).
* Rooms: Background, Outdoor, Wall, Kitchen, LivingRoom, Bedroom, Bath, Entry/Hallway, Railing, Storage, Garage, OtherRoom
* Icons: NoIcon, Window, Door, Closet, ElectricalAppliance, Toilet, Sink, SaunaBench, Fireplace, Bathtub, Chimney, Stairs
* Heatmaps: WallJunction, OpeningEndpoint, IconCorner
Input: RGB, ImageNet mean/std normalisation, sides padded to a multiple of 32 (white).
Load: `smp.Unet("{ENCODER}", encoder_weights=None, classes=27, decoder_attention_type="scse")`,
then `load_state_dict(torch.load("checkpoints/best.pt")["ema"])`.
Progress: `metrics.json` (per epoch) and `status.json`. Trained on CubiCasa5K (CC BY-NC-SA 4.0).
"""
try:
    api.upload_file(path_or_fileobj=readme.encode(), path_in_repo="README.md", repo_id=MODEL_REPO,
                    commit_message="model card")
except Exception as e: fail(f"Write test to {MODEL_REPO} failed: {e}")
print("HF OK ->", f"https://huggingface.co/{MODEL_REPO}")

# Disk
free_gb = shutil.disk_usage(BIG_DIR).free / 1e9
print(f"free disk in {BIG_DIR}: {free_gb:.0f} GB")
if free_gb < 14: fail(f"Only {free_gb:.0f} GB free in {BIG_DIR}; need ~14 GB.")

# Pretrained encoder weights (download once, before the 2 workers start)
r = subprocess.run([sys.executable, "-c",
    "import timm; timm.create_model('efficientnet_b3', pretrained=True); print('weights ok')"],
    capture_output=True, text=True)
print(r.stdout.strip() or r.stderr[-800:])
if r.returncode != 0: fail("Could not download EfficientNet-B3 ImageNet weights.")

# 2-GPU communication test (falls back to NCCL_P2P_DISABLE=1 if needed)
DDP_ENV = dict(os.environ, OMP_NUM_THREADS="2", PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True")
def ddp_ok(env):
    try:
        r = subprocess.run(["torchrun", "--standalone", "--nproc_per_node=2", f"{SRC_DIR}/ddp_check.py"],
                           capture_output=True, text=True, timeout=240, env=env)
        return "DDP_OK" in r.stdout, (r.stdout + r.stderr)[-1500:]
    except subprocess.TimeoutExpired:
        return False, "timeout"
ok, out = ddp_ok(DDP_ENV)
if not ok:
    print("Default NCCL failed, retrying with NCCL_P2P_DISABLE=1 ...\\n", out)
    DDP_ENV["NCCL_P2P_DISABLE"] = "1"
    ok, out = ddp_ok(DDP_ENV)
if not ok: fail("2-GPU NCCL test failed:\\n" + out)
print("2-GPU DDP OK", "(P2P disabled)" if DDP_ENV.get("NCCL_P2P_DISABLE") else "")
print(f"setup done in {(time.time()-NOTEBOOK_START)/60:.1f} min")''')

code('''# ===================== DATA (HF cache -> Kaggle input -> Zenodo) =====================
sys.path.insert(0, SRC_DIR)
import fp_prep
t0 = time.time()
index = fp_prep.prepare_data(CACHE_DIR, f"{BIG_DIR}/work", f"{BIG_DIR}/parser", MAX_SIDE,
                             api, DATASET_REPO, HF_TOKEN)
if EXTRA_SOURCES:
    os.makedirs(f"{BIG_DIR}/work", exist_ok=True)
    index = fp_prep.prepare_extra(CACHE_DIR, f"{BIG_DIR}/work", MAX_SIDE, api, DATASET_REPO, HF_TOKEN,
                                  EXTRA_SOURCES, N_SYNTHETIC, FPCAD_DOORS)
import collections
for split in ("train", "val"):
    print(split, dict(collections.Counter(it.get("source", "cubicasa") for it in index[split])))
print({k: len(v) for k, v in index.items() if k != "meta"}, "| meta:",
      {k: v for k, v in index["meta"].items() if k != "failures"},
      "| failed:", len(index["meta"]["failures"]))
print(f"data ready in {(time.time()-t0)/60:.1f} min")''')

code('''# ===================== RESUME (pull last checkpoint from HF if present) =====================
from huggingface_hub import hf_hub_download
remote = set(api.list_repo_files(MODEL_REPO))
for f in ["checkpoints/last.pt", "checkpoints/best.pt"]:
    local = os.path.join(CKPT_DIR, os.path.basename(f))
    if f in remote and not os.path.exists(local):
        for attempt in range(3):
            try:
                p = hf_hub_download(MODEL_REPO, f, token=HF_TOKEN, local_dir=f"{BIG_DIR}/hf_ckpt")
                shutil.copyfile(p, local); print("restored", f); break
            except Exception as e:
                print("download retry", attempt + 1, e); time.sleep(20)
        else:
            raise RuntimeError(f"Could not download {f}; rerun later rather than restarting from scratch.")
force_total = None
if os.path.exists(os.path.join(CKPT_DIR, "last.pt")):
    import torch
    ck = torch.load(os.path.join(CKPT_DIR, "last.pt"), map_location="cpu", weights_only=False)
    print(f"resuming: epoch {ck['epoch']}/{ck['total_epochs']}, best score {ck['best_score']:.4f}")
    if EXTRA_EPOCHS > 0:
        force_total = ck["total_epochs"] + EXTRA_EPOCHS
    elif ck["epoch"] >= ck["total_epochs"]:
        print("Schedule already finished. Set EXTRA_EPOCHS > 0 to keep training.")
    del ck
else:
    print("fresh run")''')

code('''# ===================== RUN CONFIG + BATCH-SIZE PROBE =====================
cpus = os.cpu_count() or 4
cfg = {"cache_root": CACHE_DIR, "ckpt_dir": CKPT_DIR, "encoder": ENCODER, "crop": CROP,
       "max_side": MAX_SIDE, "sigma": SIGMA, "batch_per_gpu": 4, "workers": max(2, min(4, cpus // 2)),
       "lr": LR, "encoder_lr_mult": 0.5, "weight_decay": 1e-2, "warmup_epochs": 1,
       "max_epochs": MAX_EPOCHS, "force_total_epochs": force_total, "ema_decay": 0.999,
       "grad_clip": 1.0, "loss_weights": [1.0, 1.0, 1.0], "val_every": 1, "log_every": 50,
       "seed": SEED, "final_margin_s": 900, "probe_batches": [8, 6, 5, 4, 3, 2],
       "deadline_ts": NOTEBOOK_START + TIME_BUDGET_HOURS * 3600, "hf_repo": MODEL_REPO,
       "data_version": DATA_VERSION, "source_weights": SOURCE_WEIGHTS, "rot_prob": ROT_PROB,
       "scale_range": SCALE_RANGE, "rotated_val_angles": [30, 60], "extra_val_every": 5, "extra_val_limit": 100}
CFG_PATH = f"{SRC_DIR}/run_config.json"
json.dump(cfg, open(CFG_PATH, "w"), indent=1)
r = subprocess.run([sys.executable, f"{SRC_DIR}/train.py", "--config", CFG_PATH, "--probe"],
                   capture_output=True, text=True, cwd=SRC_DIR, env=DDP_ENV)
print("\\n".join(l for l in r.stdout.splitlines() if "probe" in l.lower()))
m = re.search(r"PROBE_BS=(\\d+)", r.stdout)
if not m: print(r.stderr[-2000:]); raise RuntimeError("Batch probe failed (see error above).")
cfg["batch_per_gpu"] = int(m.group(1))
json.dump(cfg, open(CFG_PATH, "w"), indent=1)
print(f"batch per GPU = {cfg['batch_per_gpu']} (global {2 * cfg['batch_per_gpu']}), workers/GPU = {cfg['workers']}")''')

code('''# ===================== TRAIN on both T4s (auto-restarts from last.pt on a crash) =====================
def launch():
    cmd = ["torchrun", "--standalone", "--nproc_per_node=2", "train.py", "--config", CFG_PATH]
    p = subprocess.Popen(cmd, cwd=SRC_DIR, env=DDP_ENV, stdout=subprocess.PIPE,
                         stderr=subprocess.STDOUT, text=True, bufsize=1)
    for line in p.stdout:
        if "Warning" in line or "warnings.warn" in line: continue
        print(line, end="", flush=True)
    return p.wait()

for attempt in range(1, 4):
    rc = launch()
    status_p = os.path.join(CKPT_DIR, "status.json")
    if rc == 0 and os.path.exists(status_p):
        print("\\nTraining exited cleanly:", open(status_p).read()); break
    left_h = (cfg["deadline_ts"] - time.time()) / 3600
    print(f"\\n!!! training exited with code {rc} (attempt {attempt}); {left_h:.1f} h left in budget")
    if left_h < 0.75: break
    cfg["force_total_epochs"] = None   # keep the schedule stored in last.pt
    json.dump(cfg, open(CFG_PATH, "w"), indent=1)
    time.sleep(30)''')

code('''# ===================== PREVIEW + SUMMARY =====================
subprocess.run([sys.executable, f"{SRC_DIR}/viz.py", CFG_PATH], cwd=SRC_DIR, env=DDP_ENV)
from IPython.display import Image, display
png = os.path.join(CKPT_DIR, "val_predictions.png")
if os.path.exists(png): display(Image(png))
hist = json.load(open(os.path.join(CKPT_DIR, "metrics.json"))) if os.path.exists(os.path.join(CKPT_DIR, "metrics.json")) else []
if hist:
    b = max((h for h in hist if "score" in h), key=lambda h: h["score"])
    print(f"best epoch {b['epoch']}: room mIoU {b['room_mIoU']:.4f}  icon mIoU {b['icon_mIoU']:.4f}")
    print("room IoU:", b["room_IoU"]); print("icon IoU:", b["icon_IoU"])
print(f"total runtime {(time.time()-NOTEBOOK_START)/3600:.2f} h  ->  https://huggingface.co/{MODEL_REPO}")''')

nb["cells"] = cells
nb["metadata"] = {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                  "language_info": {"name": "python"},
                  "kaggle": {"accelerator": "nvidiaTeslaT4", "isInternetEnabled": True, "isGpuEnabled": True}}
nbf.write(nb, os.path.join(SRC, "floorplan_multitask_kaggle.ipynb"))
print("cells:", len(cells))
