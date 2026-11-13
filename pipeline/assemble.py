"""Builds the single-file pipeline from the modules in this folder.
Writes  app/pipeline.py         (used by the engine; do not edit it by hand)
and     pipeline/plan_to_revit.py (the same code, to paste into a Kaggle cell; git-ignored).
Run from anywhere:  python pipeline/assemble.py"""
import os

HERE = os.path.dirname(os.path.abspath(__file__))


def read(name):
    return open(os.path.join(HERE, name)).read()


p3 = read("plan_to_3d.py")
core = p3[p3.index("# ----------------------------------------------------------------------------- 1. model inference"):
          p3.index("# ----------------------------------------------------------------------------- 3. scale + rooms + json")]
dim = read("dimsolve.py")
dim = dim[dim.index("NUM = re.compile"):]
parts = read("revit_parts.py")
fc = read("furn_common.py")
fc = fc[fc.index("# FloorPlanCAD classes"):]
fparts = read("furn_parts.py")
stp = read("stairs.py")
header = '''# ===================== FLOOR PLAN -> EXACT 3D FOR REVIT =====================
# Reads the plan with your model + OCR, solves wall positions from the WRITTEN dimensions,
# and writes plan.json (for build_in_revit.py), model.ifc (Revit: Open/Link IFC) and model.glb (preview).
IMAGE = ""                    # floor-plan image URL or file path (clean scan/export, dimensions visible)
HF_REPO = "floorplan-multitask-effb3-768"
MAX_SIDE = 1280               # inference size (try 1536 for big/dense sheets)
ROTATE_TTA = True             # also read the plan rotated so angled walls / doors lie flat (slower, x(1+angles))
FLIP_TTA = True               # also read the mirrored plan and average (2x model time; the server turns it off)
PREVIEW_3D = True             # preview.png with two 3D renders (slow on CPU; the server draws only the 2D check)

# --- scale: written dimensions are used automatically. Only if the plan has NO dimensions:
METERS_PER_PIXEL = None
BUILDING_WIDTH_M = None       # distance between outermost wall centre-lines
DOOR_WIDTH_M = 0.9            # last-resort estimate

# --- standards (metres). Wall thickness snaps to the nearest option within 35%.
WALL_THICKNESSES = [0.115, 0.15, 0.2, 0.23, 0.3]
EXTERNAL_WALL_M = None        # e.g. 0.23 to force all external walls
INTERNAL_WALL_M = None        # e.g. 0.115 to force all internal walls
DOOR_WIDTHS = [0.75, 0.8, 0.9, 1.0, 1.2]
WALL_HEIGHT, DOOR_HEIGHT = 3.0, 2.1
WINDOW_SILL, WINDOW_HEAD = 0.9, 2.1
SLAB, STAIR_RISE = 0.15, 0.175
# --- stairs / entrance steps: written values on the plan always win; these fill the gaps
PLINTH_HEIGHT = None          # floor level above ground (m), e.g. 0.45; None = from the plan
STAIR_RISE_EXT = 0.15         # riser assumed for outside steps / stepped areas when nothing is written
STEPPED_AREAS = "down"        # stepped areas (L / U steps, seating): "down" = sunken (floor cut out), "up" = raised
# --- the converter PROJECTS what the model marks; it does not fill in for the model.
# Helpers that guess what the model missed are OFF. To switch one on for a plan, add its name:
#   "drawn_windows"  windows read from the drawing where the model marked none
#   "drawn_steps"    steps read from the drawing where the model marked no stairs
#   "stepped_areas"  irregular stair areas become sunken / raised stepped areas with open ground
#   "fill_floor"     carry broken outer walls on and add floor where the wall outline left gaps
#   "widen_doors"    a door marked too narrow gets a standard width
#   "furniture_filter"  drop furniture in the wrong room or outside the building
#   "join_gaps"      join wall pieces across small gaps the model left (up to 1.5 wall thicknesses)
#   "snap_corners"   stretch wall ends (up to 1.5-2 thicknesses) so corners meet
#   "align_walls"    line near-parallel walls up on one centre-line
#   "standard_sizes" snap wall thickness / door width to WALL_THICKNESSES / DOOR_WIDTHS
#   "merge_windows"  merge neighbouring windows on a wall into one
#   "fix_treads"     an implausible tread count -> risers from the storey height
ASSIST = set()                # e.g. ASSIST = {"drawn_windows", "fill_floor"}
OUT_DIR = "/kaggle/working/plan_revit"

# --- furniture: fixtures from the floor-plan model + your trained furniture detector (if available)
MAKE_FURNITURE = True
FURNITURE_REPO = "floorplan-furniture-obb"   # HF model repo from the furniture notebook ("" to skip)
FURNITURE_CONF = 0.4          # always accepted at/above this
FURNITURE_LOW_CONF = 0.15     # weaker hits accepted only when the room fits (bed in a bedroom, ...)
FURNITURE_ZOOMS = (0.75, 1.0, 1.4)   # detector passes at these zooms of the training scale (server: 1.0 only)

import os, sys, re, json, subprocess
if not os.environ.get("P3D_NO_INSTALL"):
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "segmentation-models-pytorch==0.6.0",
                    "timm==1.0.31", "huggingface_hub", "trimesh", "shapely", "mapbox-earcut",
                    "ifcopenshell>=0.8.3", "rapidocr_onnxruntime"], check=False)
import cv2, numpy as np, matplotlib.pyplot as plt
from scipy import ndimage
from skimage.morphology import skeletonize

WALL, WIN, DOOR, STAIRS = 2, 1, 2, 11   # room-head wall id; icon-head ids
LAST_ORIGINAL = None
LAST_ICON_PROB = None

import time as _time
TIMES = {}                    # seconds per stage of the last run (printed at the end of the report)
_LAP = [_time.time()]


def _lap(name=None):
    """Book the time since the previous lap to `name` (None: just restart the clock)."""
    now = _time.time()
    if name:
        TIMES[name] = TIMES.get(name, 0.0) + now - _LAP[0]
    _LAP[0] = now


'''
dim_hdr = "# ----------------------------------------------------------------------------- 2b. dimension solving (OCR)\n"
code = (header + core + dim_hdr + dim + "\n\n" +
        "# ----------------------------------------------------------------------------- furniture detector helpers\n" +
        fc + "\n\n" + fparts + "\n\n" + stp + "\n\n" + parts)
open(os.path.join(HERE, "plan_to_revit.py"), "w").write(code)
open(os.path.join(HERE, "..", "app", "pipeline.py"), "w").write(
    "# generated from pipeline/ by assemble.py - do not edit\n" + code)
print("wrote app/pipeline.py and pipeline/plan_to_revit.py")
