# -*- coding: utf-8 -*-
"""Build a native Revit model from plan.json (made by plan_to_revit.py on Kaggle).

Creates: grids, walls (own wall types per thickness, centre-line, auto-joined), doors and windows
(your loaded families, sized exactly; plain wall openings if no family is loaded), a floor,
named rooms, stairs (native Revit stairs for straight flights; exact stepped solids for flared entrance
steps and stepped seating areas; a "Ground" level below the plinth for entrance steps), and furniture /
fixtures copied from the drawing (loaded families turned to face the room, or exactly sized boxes).

Run it in Revit 2022 or newer with pyRevit or RevitPythonShell. Start from a new project
(Architectural template), so the default door/window families are loaded.
"""
from __future__ import print_function

import json
import os

import clr
clr.AddReference("RevitAPI")
from Autodesk.Revit.DB import (BuiltInCategory, BuiltInParameter, CompoundStructure, CurveLoop, DirectShape,
                               Element, ElementId, ElementTransformUtils, ElementTypeGroup, FamilySymbol,
                               FilteredElementCollector, Floor, FloorType, GeometryCreationUtilities,
                               GeometryObject, Grid, Level, Line, Material, MaterialFunctionAssignment, Plane,
                               SketchPlane, Transaction, UV, Wall, WallKind, WallType, XYZ)
from Autodesk.Revit.DB.Structure import StructuralType
from System.Collections.Generic import List
try:
    from Autodesk.Revit.DB import (FailureProcessingResult, FailureSeverity, IFailuresPreprocessor,
                                   StairsEditScope)
    from Autodesk.Revit.DB.Architecture import StairsRun, StairsRunJustification
    HAVE_STAIRS_API = True
except Exception:
    HAVE_STAIRS_API = False

# ----------------------------------------------------------------------------- settings
JSON_PATH = None          # leave None to pick the file in a dialog
MAKE_GRIDS = True
MAKE_ROOMS = True
MAKE_FLOOR = True
MAKE_FURNITURE = True
MAKE_STAIRS = True
NATIVE_STAIRS = True       # straight flights as real Revit stairs (else exact stepped solids)
FURNITURE_AS_BOXES = False  # True = always sized boxes (no family matching)
FT = 1.0 / 0.3048         # metres -> Revit internal feet

try:
    doc = __revit__.ActiveUIDocument.Document          # pyRevit / RevitPythonShell
except NameError:
    raise SystemExit("Run this inside Revit with pyRevit or RevitPythonShell.")


# ----------------------------------------------------------------------------- helpers
def pick_json():
    if JSON_PATH:
        return JSON_PATH
    try:
        from pyrevit import forms
        return forms.pick_file(file_ext="json", title="Select plan.json")
    except Exception:
        clr.AddReference("System.Windows.Forms")
        from System.Windows.Forms import DialogResult, OpenFileDialog
        dlg = OpenFileDialog()
        dlg.Filter = "plan.json|*.json"
        dlg.Title = "Select plan.json"
        return dlg.FileName if dlg.ShowDialog() == DialogResult.OK else None


def name_of(e):
    try:
        return e.Name
    except Exception:
        return Element.Name.GetValue(e)


def P(xy, z=0.0):
    """plan metres (x right, y down the sheet) -> Revit XYZ in feet (y up)."""
    return XYZ(xy[0] * FT, -xy[1] * FT, z * FT)


def set_param(elem, bips, names, value):
    for b in bips:
        try:
            p = elem.get_Parameter(b)
            if p and not p.IsReadOnly:
                p.Set(value)
                return True
        except Exception:
            pass
    for n in names:
        p = elem.LookupParameter(n)
        if p and not p.IsReadOnly:
            try:
                p.Set(value)
                return True
            except Exception:
                pass
    return False


def collect(cls=None, cat=None, types=False):
    c = FilteredElementCollector(doc)
    if cls is not None:
        c = c.OfClass(cls)
    if cat is not None:
        c = c.OfCategory(cat)
    c = c.WhereElementIsElementType() if types else c
    return list(c)


def base_level():
    levels = sorted(collect(Level), key=lambda l: abs(l.Elevation))
    if not levels:
        raise SystemExit("No level in this project. Start from an architectural template.")
    return levels[0]


# ----------------------------------------------------------------------------- types
_wall_types = {}


def wall_type(thk_m):
    key = int(round(thk_m * 1000))
    if key in _wall_types:
        return _wall_types[key]
    name = "Plan Masonry {} mm".format(key)
    basics = [w for w in collect(WallType) if w.Kind == WallKind.Basic]
    for w in basics:
        if name_of(w) == name:
            _wall_types[key] = w
            return w
    if not basics:
        raise SystemExit("No basic wall type found in this project.")
    new = basics[0].Duplicate(name)
    mat = ElementId.InvalidElementId
    for m in collect(Material):
        if "brick" in name_of(m).lower() or "masonry" in name_of(m).lower():
            mat = m.Id
            break
    cs = CompoundStructure.CreateSingleLayerCompoundStructure(MaterialFunctionAssignment.Structure,
                                                              thk_m * FT, mat)
    new.SetCompoundStructure(cs)
    _wall_types[key] = new
    return new


_sym_cache = {}


def opening_symbol(kind, width_m, height_m):
    """Duplicate a loaded door/window type at the exact size. Returns None if no family is loaded."""
    key = (kind, int(round(width_m * 1000)), int(round(height_m * 1000)))
    if key in _sym_cache:
        return _sym_cache[key]
    cat = BuiltInCategory.OST_Doors if kind == "door" else BuiltInCategory.OST_Windows
    syms = [s for s in collect(FamilySymbol, cat) if isinstance(s, FamilySymbol)]
    if not syms:
        _sym_cache[key] = None
        return None
    pref = ("single", "flush") if kind == "door" else ("fixed", "casement", "window")
    syms.sort(key=lambda s: 0 if any(p in s.Family.Name.lower() for p in pref) else 1)
    base = syms[0]
    tname = "{} x {}".format(key[1], key[2])
    sym = None
    for s in syms:
        if s.Family.Id == base.Family.Id and name_of(s) == tname:
            sym = s
            break
    if sym is None:
        sym = base.Duplicate(tname)
        wb = [BuiltInParameter.DOOR_WIDTH, BuiltInParameter.FAMILY_WIDTH_PARAM] if kind == "door" else \
             [BuiltInParameter.WINDOW_WIDTH, BuiltInParameter.FAMILY_WIDTH_PARAM]
        hb = [BuiltInParameter.DOOR_HEIGHT, BuiltInParameter.FAMILY_HEIGHT_PARAM] if kind == "door" else \
             [BuiltInParameter.WINDOW_HEIGHT, BuiltInParameter.FAMILY_HEIGHT_PARAM]
        set_param(sym, wb, ["Width", "Rough Width"], width_m * FT)
        set_param(sym, hb, ["Height", "Rough Height"], height_m * FT)
    if not sym.IsActive:
        sym.Activate()
        doc.Regenerate()
    _sym_cache[key] = sym
    return sym


# ----------------------------------------------------------------------------- furniture
FURN_KEYWORDS = {
    "bed": ["bed"], "sofa": ["sofa", "couch"], "chair": ["chair"], "table": ["table"],
    "bedside_cupboard": ["night", "bedside"], "tv_cabinet": ["tv", "media", "entertainment"],
    "half_height_cabinet": ["cabinet", "base"], "high_cabinet": ["tall", "cabinet"],
    "wardrobe": ["wardrobe", "closet"], "closet": ["wardrobe", "closet", "shelving"],
    "sink": ["sink", "lavatory", "basin"], "bath": ["tub", "bath"], "bath_tub": ["tub", "bath"],
    "toilet": ["toilet", "water closet", "wc"], "squat_toilet": ["toilet", "wc"], "urinal": ["urinal"],
    "gas_stove": ["range", "stove", "cooktop"], "refrigerator": ["refrigerator", "fridge"],
    "washing_machine": ["washer", "washing"], "appliance": ["refrigerator", "range", "washer", "appliance"],
    "fireplace": ["fireplace"], "chimney": ["chimney"], "sauna_bench": ["bench"]}
FURN_CATS = [BuiltInCategory.OST_Furniture, BuiltInCategory.OST_Casework, BuiltInCategory.OST_PlumbingFixtures,
             BuiltInCategory.OST_SpecialityEquipment, BuiltInCategory.OST_FurnitureSystems]
_furn_syms = None


def furniture_symbol(ftype):
    global _furn_syms
    if _furn_syms is None:
        _furn_syms = []
        for cat in FURN_CATS:
            try:
                _furn_syms += [s for s in collect(FamilySymbol, cat) if isinstance(s, FamilySymbol)]
            except Exception:
                pass
    keys = FURN_KEYWORDS.get(ftype, [ftype.replace("_", " ")])
    for k in keys:
        for s in _furn_syms:
            label = (s.Family.Name + " " + name_of(s)).lower()
            if k in label:
                return s
    return None


def furniture_box(f, z0):
    """Exactly sized box at the drawn position and rotation (a DirectShape the architect can swap)."""
    loop = CurveLoop()
    c = f["corners"]
    for i in range(4):
        loop.Append(Line.CreateBound(P(c[i], z0), P(c[(i + 1) % 4], z0)))
    loops = List[CurveLoop]()
    loops.Add(loop)
    solid = GeometryCreationUtilities.CreateExtrusionGeometry(loops, XYZ.BasisZ, f["height"] * FT)
    cat = ElementId(BuiltInCategory.OST_Furniture if f["category"] == "furniture" else
                    BuiltInCategory.OST_PlumbingFixtures if f["category"] == "plumbing" else
                    BuiltInCategory.OST_SpecialityEquipment)
    if not DirectShape.IsValidCategoryId(cat, doc):
        cat = ElementId(BuiltInCategory.OST_GenericModel)
    ds = DirectShape.CreateElement(doc, cat)
    shapes = List[GeometryObject]()
    shapes.Add(solid)
    ds.SetShape(shapes)
    try:
        ds.Name = "{} {:.2f} x {:.2f} m".format(f["type"].replace("_", " "), f["width"], f["depth"])
    except Exception:
        pass
    return ds


def place_furniture(f, lvl, z0):
    """Loaded family at the drawn spot, turned to face the room; otherwise an exact box."""
    sym = None if FURNITURE_AS_BOXES else furniture_symbol(f["type"])
    if sym is not None:
        try:
            if not sym.IsActive:
                sym.Activate()
                doc.Regenerate()
            pt = P(f["center"], z0)
            inst = doc.Create.NewFamilyInstance(pt, sym, lvl, StructuralType.NonStructural)
            doc.Regenerate()
            want = XYZ(f["front"][0], -f["front"][1], 0)
            have = inst.FacingOrientation
            ang = have.AngleOnPlaneTo(want, XYZ.BasisZ)
            if abs(ang) > 1e-3:
                ElementTransformUtils.RotateElement(doc, inst.Id, Line.CreateBound(pt, pt + XYZ.BasisZ), ang)
            return "family"
        except Exception:
            pass                                          # hosted / unusual family -> exact box instead
    furniture_box(f, z0)
    return "box"


# ----------------------------------------------------------------------------- stairs
def level_at(z_ft, name):
    """Existing level at this elevation, or a new one (call inside a transaction)."""
    for l in collect(Level):
        if abs(l.Elevation - z_ft) < 0.01:
            return l, False
    l = Level.Create(doc, z_ft)
    try:
        l.Name = name
    except Exception:
        pass
    return l, True


def clean_loop(pts):
    out = []
    for q in pts:
        if not out or abs(q[0] - out[-1][0]) + abs(q[1] - out[-1][1]) > 0.005:
            out.append(q)
    if len(out) > 1 and abs(out[0][0] - out[-1][0]) + abs(out[0][1] - out[-1][1]) <= 0.005:
        out.pop()
    return out


def steps_shape(s, z0):
    """Exact stepped solid, one block per tread (flared steps, stepped areas, or when native stairs fail)."""
    shapes = List[GeometryObject]()
    for tr in s.get("treads", []):
        pts = clean_loop(tr["polygon"])
        if len(pts) < 3 or tr["z1"] - tr["z0"] < 0.005:
            continue
        loop = CurveLoop()
        for i in range(len(pts)):
            loop.Append(Line.CreateBound(P(pts[i], z0 + tr["z0"]), P(pts[(i + 1) % len(pts)], z0 + tr["z0"])))
        loops = List[CurveLoop]()
        loops.Add(loop)
        shapes.Add(GeometryCreationUtilities.CreateExtrusionGeometry(loops, XYZ.BasisZ, (tr["z1"] - tr["z0"]) * FT))
    cat = ElementId(BuiltInCategory.OST_Stairs)
    if not DirectShape.IsValidCategoryId(cat, doc):
        cat = ElementId(BuiltInCategory.OST_GenericModel)
    ds = DirectShape.CreateElement(doc, cat)
    ds.SetShape(shapes)
    try:
        ds.Name = "{} steps: {} x {:.0f} mm risers, {:.0f} mm treads".format(
            s["kind"].title(), s["risers"], s["riser"] * 1000, s["tread"] * 1000)
    except Exception:
        pass
    return ds


if HAVE_STAIRS_API:
    class _QuietWarnings(IFailuresPreprocessor):
        def PreprocessFailures(self, fa):
            for m in fa.GetFailureMessages():
                if m.GetSeverity() == FailureSeverity.Warning:
                    fa.DeleteWarning(m)
            return FailureProcessingResult.Continue


def native_stair(s, base_lvl, top_lvl, z0):
    """Real Revit stair: one straight run with the drawn width, risers from plan.json."""
    scope = StairsEditScope(doc, "Plan to Revit stair")
    sid = scope.Start(base_lvl.Id, top_lvl.Id)
    t = Transaction(doc, "Plan to Revit: stair run")
    t.Start()
    try:
        st = doc.GetElement(sid)
        try:
            st.DesiredRisersNumber = int(s["risers"])
        except Exception:
            pass
        zb = z0 + s["base_z"]
        run = StairsRun.CreateStraightRun(doc, sid, Line.CreateBound(P(s["start"], zb), P(s["end"], zb)),
                                          StairsRunJustification.Center)
        run.ActualRunWidth = s["width"] * FT
        t.Commit()
    except Exception:
        t.RollBack()
        scope.Cancel()
        raise
    scope.Commit(_QuietWarnings())
    return sid


def build_stairs(plan, lvl, z0, stats, errors):
    stairs = plan.get("stairs", [])
    if not stairs:
        return
    t = Transaction(doc, "Plan to Revit: levels for stairs")
    t.Start()
    levels = {}
    for s in stairs:
        for z in (s["base_z"], s["top_z"]):
            key = round(z, 3)
            if key in levels or abs(z) < 1e-4:
                continue
            name = "Ground (plinth {:.2f} m)".format(-z) if z < 0 else "Upper floor (+{:.2f} m)".format(z)
            try:
                levels[key], new = level_at((z0 + z) * FT, name)
                stats["levels_added"] += int(new)
            except Exception as e:
                errors.append("level {:+.2f} m: {}".format(z, e))
    t.Commit()
    lv = lambda z: lvl if abs(z) < 1e-4 else levels.get(round(z, 3))
    for s in stairs:
        done = False
        if NATIVE_STAIRS and HAVE_STAIRS_API and s.get("shape") == "straight" and s["kind"] != "terrace":
            b, tp = lv(s["base_z"]), lv(s["top_z"])
            if b is not None and tp is not None:
                try:
                    native_stair(s, b, tp, z0)
                    stats["stairs_native"] += 1
                    done = True
                except Exception as e:
                    errors.append("stair {} as native Revit stair: {} (drawn as exact steps instead)".format(s["id"], e))
        t = Transaction(doc, "Plan to Revit: steps")
        t.Start()
        try:
            if not done:
                steps_shape(s, z0)
                stats["stairs_shapes"] += 1
            zb = z0 + s["base_z"]
            sp = SketchPlane.Create(doc, Plane.CreateByNormalAndOrigin(XYZ.BasisZ, XYZ(0, 0, zb * FT)))
            o = clean_loop(s["outline"])
            for i in range(len(o)):
                doc.Create.NewModelCurve(Line.CreateBound(P(o[i], zb), P(o[(i + 1) % len(o)], zb)), sp)
            doc.Create.NewModelCurve(Line.CreateBound(P(s["start"], zb), P(s["end"], zb)), sp)
            stats["stair_outlines"] += 1
            t.Commit()
        except Exception as e:
            t.RollBack()
            errors.append("stair {}: {}".format(s["id"], e))


# ----------------------------------------------------------------------------- build
def build(plan):
    lvl = base_level()
    z0 = lvl.Elevation / FT        # metres
    stats = {"grids": 0, "walls": 0, "doors": 0, "windows": 0, "plain_openings": 0, "rooms": 0,
             "floor": 0, "stair_outlines": 0, "stairs_native": 0, "stairs_shapes": 0, "levels_added": 0,
             "furniture_families": 0, "furniture_boxes": 0}
    errors = []

    t = Transaction(doc, "Plan to Revit: grids + walls")
    t.Start()
    if MAKE_GRIDS:
        existing = set(name_of(g) for g in collect(Grid))
        for g in plan.get("grids", []):
            try:
                gr = Grid.Create(doc, Line.CreateBound(P(g["start"], z0), P(g["end"], z0)))
                nm = g["name"] if g["name"] not in existing else "P" + g["name"]
                gr.Name = nm
                existing.add(nm)
                stats["grids"] += 1
            except Exception as e:
                errors.append("grid {}: {}".format(g["name"], e))
    walls = {}
    for w in plan["walls"]:
        try:
            a, b = P(w["start"], z0), P(w["end"], z0)
            if a.DistanceTo(b) < 0.05:
                continue
            wall = Wall.Create(doc, Line.CreateBound(a, b), wall_type(w["thickness"]).Id, lvl.Id,
                               w["height"] * FT, 0.0, False, False)
            set_param(wall, [BuiltInParameter.WALL_KEY_REF_PARAM], [], 0)       # location line = centre-line
            set_param(wall, [BuiltInParameter.ALL_MODEL_INSTANCE_COMMENTS], [],
                      "external" if w.get("external") else "internal")
            walls[w["id"]] = wall
            stats["walls"] += 1
        except Exception as e:
            errors.append("wall {}: {}".format(w["id"], e))
    t.Commit()

    t = Transaction(doc, "Plan to Revit: doors + windows")
    t.Start()
    for o in plan["openings"]:
        host = walls.get(o["wall"])
        if host is None:
            continue
        try:
            sym = opening_symbol(o["type"], o["width"], o["height"])
            if sym is not None:
                inst = doc.Create.NewFamilyInstance(P(o["center"], z0), sym, host, lvl,
                                                    StructuralType.NonStructural)
                set_param(inst, [BuiltInParameter.INSTANCE_SILL_HEIGHT_PARAM], [], o["sill"] * FT)
                # families that size by instance parameters
                set_param(inst, [], ["Width"], o["width"] * FT)
                set_param(inst, [], ["Height"], o["height"] * FT)
                stats["doors" if o["type"] == "door" else "windows"] += 1
            else:
                # no family loaded: cut an exact rectangular hole so the geometry is still right
                wl = [plan["walls"][i] for i in range(len(plan["walls"])) if plan["walls"][i]["id"] == o["wall"]][0]
                ax, ay = wl["start"]; bx, by = wl["end"]
                L = ((bx - ax) ** 2 + (by - ay) ** 2) ** 0.5
                dx, dy = (bx - ax) / L, (by - ay) / L
                c = o["center"]
                p1 = [c[0] - dx * o["width"] / 2, c[1] - dy * o["width"] / 2]
                p2 = [c[0] + dx * o["width"] / 2, c[1] + dy * o["width"] / 2]
                doc.Create.NewOpening(host, P(p1, z0 + o["sill"]), P(p2, z0 + o["sill"] + o["height"]))
                stats["plain_openings"] += 1
        except Exception as e:
            errors.append("{} {}: {}".format(o["type"], o["id"], e))
    t.Commit()

    t = Transaction(doc, "Plan to Revit: floor, rooms")
    t.Start()
    if MAKE_FLOOR and plan.get("floor"):
        try:
            pts = plan["floor"]["polygon"]
            loop = CurveLoop()
            for i in range(len(pts)):
                a, b = P(pts[i], z0), P(pts[(i + 1) % len(pts)], z0)
                if a.DistanceTo(b) > 0.01:
                    loop.Append(Line.CreateBound(a, b))
            ft_id = doc.GetDefaultElementTypeId(ElementTypeGroup.FloorType)
            if ft_id == ElementId.InvalidElementId:
                ft_id = collect(FloorType)[0].Id
            loops = List[CurveLoop]()
            loops.Add(loop)
            Floor.Create(doc, loops, ft_id, lvl.Id)
            stats["floor"] = 1
        except Exception as e:
            errors.append("floor: {} (Revit 2022+ needed for Floor.Create)".format(e))
    doc.Regenerate()
    if MAKE_ROOMS:
        for r in plan.get("rooms", []):
            try:
                room = doc.Create.NewRoom(lvl, UV(r["point"][0] * FT, -r["point"][1] * FT))
                set_param(room, [BuiltInParameter.ROOM_NAME], ["Name"], r["name"])
                stats["rooms"] += 1
            except Exception as e:
                errors.append("room {}: {}".format(r["name"], e))
    t.Commit()

    if MAKE_STAIRS:
        build_stairs(plan, lvl, z0, stats, errors)

    if MAKE_FURNITURE and plan.get("furniture"):
        t = Transaction(doc, "Plan to Revit: furniture")
        t.Start()
        for f in plan["furniture"]:
            try:
                how = place_furniture(f, lvl, z0)
                stats["furniture_families" if how == "family" else "furniture_boxes"] += 1
            except Exception as e:
                errors.append("furniture {} {}: {}".format(f["type"], f["id"], e))
        t.Commit()
    return stats, errors


def main():
    path = pick_json()
    if not path or not os.path.exists(path):
        print("No plan.json selected.")
        return
    with open(path) as fh:
        plan = json.load(fh)
    stats, errors = build(plan)
    pr = plan.get("precision", {})
    print("Plan to Revit finished")
    print("  scale from: {} ({}/{} written dimensions exact)".format(
        pr.get("scale_source"), pr.get("dimensions_exact"), pr.get("dimensions_read")))
    for k in ("grids", "walls", "doors", "windows", "plain_openings", "floor", "rooms", "stairs_native",
              "stairs_shapes", "stair_outlines", "levels_added", "furniture_families", "furniture_boxes"):
        print("  {:<15} {}".format(k, stats[k]))
    if stats["plain_openings"]:
        print("  (no door/window family loaded -> exact wall openings cut instead; load families and rerun)")
    lvls = plan.get("levels", {})
    if lvls.get("plinth"):
        print("  plinth {:.3f} m above ground ({})".format(lvls["plinth"], lvls.get("plinth_source", "")))
    for st in plan.get("stairs", []):
        print("  {} stair {}: {} risers x {:.0f} mm, tread {:.0f} mm, width {:.2f} m".format(
            st["kind"], st["id"], st["risers"], st["riser"] * 1000, st["tread"] * 1000, st["width"]))
        for w in st.get("warnings", []):
            print("      check: " + w)
    if stats["furniture_boxes"]:
        print("  furniture boxes are exact size/position; load matching families and rerun to swap them in")
    if errors:
        print("\n{} items need attention:".format(len(errors)))
        for e in errors[:40]:
            print("  - " + e)


main()
