# PlanForge — 2D floor plan → exact 3D BIM & Revit

Upload a floor plan (**DXF, PDF, PNG or JPG**) and get a measured 3D building model: walls, doors, windows,
stairs, rooms and furniture, exported as **IFC4**, **glTF** and a **native Revit build script**.
Architects then only add elevations and detailing, which cuts most of the manual modelling work.

**Live demo:** _add the Vercel link here_ · built by [@gogul098](https://github.com/gogul098)

![3D viewer](docs/screenshots/viewer.png)

| Furniture editor | Section + X-ray |
|---|---|
| ![editor](docs/screenshots/furniture-editor.png) | ![xray](docs/screenshots/xray-section.png) |

## What it does

- **Any input:** DXF is rasterised at its true drawing units, so the scale is exact. PDF and images are scaled
  from the dimension strings and room sizes written on the plan, read by OCR and solved by least squares.
- **Floor-plan model:** a multitask U-Net (EfficientNet-B3, 768 px) segments 12 room classes and 12 icon
  classes (walls, doors, windows, stairs, fixtures). Best validation score 0.747.
- **Furniture detector:** YOLO11m-OBB trained on FloorPlanCAD (18 classes, mAP50 ≈ 0.87). It keeps the exact
  rotation, checks each object against the room it sits in, and snaps it back against the nearest wall.
- **Vectorisation:** masks become centre-line walls with snapped thicknesses, joined corners and openings hosted
  in walls. On the test plans, 99% of predicted wall pixels end up in BIM walls.
- **Exports:**
  - `model.ifc`: IFC4 with wall types, doors, windows, spaces, slab, stairs and furniture.
  - `plan.json` + `build_in_revit.py`: builds native Revit walls, doors, windows, floors, rooms, grids and
    furniture (pyRevit / RevitPythonShell).
  - `model.glb`: the 3D model as glTF.
- **Web app:** a BIM-style viewer (orbit, pan, view cube, section cut, measure, walk, layers, X-ray), a
  "need furniture placement" switch, and a furniture editor (add, move, rotate, swap, delete, undo) that
  re-exports every file.

## Stairs, entrance steps and stepped areas

Each detected stair is classified and gets exact numbers. Every value records where it came from, best source first:
**written on the plan** → **counted from the drawing** → **your setting** → **standard assumption** (flagged to check).

| | Entrance steps | Interior stair | Stepped area (seating / terraced floor) |
|---|---|---|---|
| How it's recognised | outside the building footprint | inside, straight flight | inside, L / U shaped |
| Rises | from the ground up to the floor (plinth) | floor to floor (wall height) | in tiers towards the walls |
| Step count | written ("12R", "7 steps") or counted from the tread lines | same; plus the riser onto the upper floor | counted tiers |
| Height | plinth: your setting, or a level mark ("+0.45", "PLINTH LVL"), or counted steps × 150 mm | storey height ÷ risers | written riser, else 150 mm (flagged) |
| Width / run | written dimension next to it, else measured | same | measured |
| Shape | straight or **flared** (wider at the ground) | straight | any outline |
| Direction | towards the building | from the "UP" / "DN" label (re-read zoomed if missed), else flagged | towards the walls (flagged) |
| Revit | native stair (straight) or exact stepped solid (flared), from a new **Ground** level | native Revit stair to an upper level | exact stepped solid |

The building sits on a plinth block, with a ground surface below it so entrance steps land on something, in the
viewer, `model.glb`, `model.ifc` and Revit. Riser and tread values outside normal ranges are flagged in the report
and in the viewer (click a stair).

## How it works

```
plan ─► U-Net (rooms + icons) ─► vectorise walls/openings ─► OCR dimensions ─► solve exact scale
     └► YOLO-OBB furniture ─► room check ─► wall snapping ─┘
                                   ▼
          plan.json ─► IFC4 · glTF · Revit script ─► web viewer / editor
```

## Repository

| Path | What |
|---|---|
| `app/` | FastAPI engine (`server.py`), pipeline (`pipeline.py`, generated), Revit script (`build_in_revit.py`), website (`static/`, three.js) |
| `Dockerfile` | Container image for the conversion engine (ONNX Runtime, no PyTorch) |
| `pipeline/` | Pipeline source modules: edit these, then `python pipeline/assemble.py` rebuilds `app/pipeline.py` |
| `training/` | Kaggle notebooks + training code for both models (2× T4, DDP, fp16, EMA) |
| `kaggle/` | `export_onnx.py` (publish ONNX models), `run_planforge_kaggle.py` (run the engine on Kaggle), diagnostics |
| `.github/workflows/azure-engine.yml` | Every push to `main` that changes the engine builds the image and deploys it to the Azure Web App |
| `.github/workflows/checks.yml` | Fails if `app/pipeline.py` or the Kaggle bundle was not rebuilt |

After changing anything in `pipeline/` or `app/`:

```bash
python pipeline/assemble.py     # pipeline/ -> app/pipeline.py
python kaggle/bundle_app.py     # app/ -> the bundle inside kaggle/run_planforge_kaggle.py
```

## Run locally

```bash
pip install -r app/requirements.txt   # ONNX Runtime engine, same as the Docker image
export HF_TOKEN=...                    # read access to the model repos
cd app && uvicorn server:app --port 7860
```

The engine uses ONNX Runtime unless PyTorch is installed (or `PLANFORGE_BACKEND=torch`). To run the PyTorch
checkpoints instead (GPU, Kaggle): `pip install torch torchvision -r app/requirements-torch.txt`.

## Deploy

- **Website:** Vercel, project root `app/static` (free).
- **Conversion engine:** Docker image on an Azure Web App (Linux B1), models cached in Azure Blob Storage.
  Every push to `main` that changes the engine rebuilds and redeploys it.
- It can also run on your PC (`uvicorn server:app`) or in a Kaggle notebook (`kaggle/run_planforge_kaggle.py`
  prints a public link). Open the website with `?api=<engine URL>` to use any of them.

Step-by-step setup: [DEPLOY_AZURE.md](DEPLOY_AZURE.md).

**Models are never stored in this repo.** The engine downloads the weights from the private Hugging Face repos at
runtime with `HF_TOKEN`, and the deploy workflow refuses to build if any weight file is in the repo.

## Credits & licence

Code: MIT. Models trained on [CubiCasa5K](https://github.com/CubiCasa/CubiCasa5k) and
[FloorPlanCAD](https://floorplancad.github.io/) (CC BY-NC 4.0): non-commercial use only.
