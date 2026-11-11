p3 = open("plan_to_3d.py").read()
core = p3[p3.index("# ----------------------------------------------------------------------------- 1. model inference"):
          p3.index("# ----------------------------------------------------------------------------- 3. scale + rooms + json")]
dim = open("dimsolve.py").read()
dim = dim[dim.index("NUM = re.compile"):]
parts = open("revit_parts.py").read()
fc = open("furn_common.py").read()
fc = fc[fc.index("# FloorPlanCAD classes"):]
fparts = open("furn_parts.py").read()
header = '''# ===================== FLOOR PLAN -> EXACT 3D FOR REVIT =====================
# Reads the plan with your model + OCR, solves wall positions from the WRITTEN dimensions,
# and writes plan.json (for build_in_revit.py), model.ifc (Revit: Open/Link IFC) and model.glb (preview).
IMAGE = ""                    # floor-plan image URL or file path (clean scan/export, dimensions visible)
HF_REPO = "floorplan-multitask-effb3-768"
MAX_SIDE = 1280               # inference size (try 1536 for big/dense sheets)

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
OUT_DIR = "/kaggle/working/plan_revit"

# --- furniture: fixtures from the floor-plan model + your trained furniture detector (if available)
MAKE_FURNITURE = True
FURNITURE_REPO = "floorplan-furniture-obb"   # HF model repo from the furniture notebook ("" to skip)
FURNITURE_CONF = 0.4          # always accepted at/above this
FURNITURE_LOW_CONF = 0.15     # weaker hits accepted only when the room fits (bed in a bedroom, ...)

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


'''
dim_hdr = "# ----------------------------------------------------------------------------- 2b. dimension solving (OCR)\n"
open("plan_to_revit.py", "w").write(header + core + dim_hdr + dim + "\n\n" +
     "# ----------------------------------------------------------------------------- furniture detector helpers\n" +
     fc + "\n\n" + fparts + "\n\n" + parts)
