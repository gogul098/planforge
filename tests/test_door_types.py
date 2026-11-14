"""Door kinds: each door the floor-plan model found takes the kind of the detector's door symbol drawn on it;
no symbol on it -> unknown (nothing guessed). Training side: ArchCAD SVG slices -> label maps."""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "app"))
os.environ.setdefault("P3D_NO_MAIN", "1")
os.environ.setdefault("P3D_NO_INSTALL", "1")
os.environ.setdefault("MPLBACKEND", "Agg")
import pipeline as P  # noqa: E402

MPP = 0.01                                            # 1 px = 1 cm


def _plan():
    mapping = {"scale_m_per_px": MPP, "axes": {"x": {"px": [], "m": []}, "y": {"px": [], "m": []}}}
    walls = [{"id": 0, "start": [0.0, 0.0], "end": [10.0, 0.0], "thickness": 0.2}]
    openings = [{"id": 0, "type": "door", "wall": 0, "center": [2.0, 0.0], "offset": 2.0, "width": 0.9},
                {"id": 1, "type": "door", "wall": 0, "center": [6.0, 0.0], "offset": 6.0, "width": 1.6},
                {"id": 2, "type": "door", "wall": 0, "center": [9.0, 0.0], "offset": 9.0, "width": 0.8},
                {"id": 3, "type": "window", "wall": 0, "center": [4.0, 0.0], "offset": 4.0, "width": 1.2}]
    return {"mapping": mapping, "walls": walls, "openings": openings}


def _box(x0, y0, x1, y1):                              # metres -> px corners
    return np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]]) / MPP


def test_door_kind_from_the_symbol_on_it():
    plan = _plan()
    P._YOLO["doors"] = [
        {"type": "single_door", "conf": 0.9, "corners": _box(1.55, 0.1, 2.45, 1.0)},       # swing of door 0
        {"type": "double_door", "conf": 0.8, "corners": _box(5.2, 0.1, 6.8, 0.9)},         # door 1
        {"type": "sliding_door", "conf": 0.9, "corners": _box(3.0, 3.0, 3.9, 3.2)},        # far away: nothing
        {"type": "sliding_door", "conf": 0.2, "corners": _box(8.6, -0.1, 9.4, 0.1)},       # too weak
    ]
    counts = P.assign_door_types(plan)
    o = {x["id"]: x for x in plan["openings"]}
    assert o[0]["door_type"] == "single_door"
    assert o[1]["door_type"] == "double_door"
    assert "door_type" not in o[2]                     # no reliable symbol on it: unknown, not guessed
    assert "door_type" not in o[3]                     # windows are left alone
    assert counts == {"single_door": 1, "double_door": 1, "unknown": 1}


def test_symbol_that_covers_only_a_bit_of_the_door_is_not_used():
    plan = _plan()
    P._YOLO["doors"] = [{"type": "double_door", "conf": 0.9, "corners": _box(6.6, 0.1, 7.2, 0.8)}]  # edge only
    P.assign_door_types(plan)
    assert "door_type" not in plan["openings"][1]


SVG = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 14000 14000">
 <path d="M 1000 2000 L 13000 2000 M 1000 2240 L 5000 2240 M 6000 2240 L 13000 2240" semantic="wall"/>
 <path d="M 5000 2240 L 5000 3240 A 1000 1000 0 0 0 6000 2240" semantic="single_door" instance="d1"/>
 <rect x="8000" y="7000" width="500" height="500" semantic="21" instance="c1"/>
 <path d="M 1000 12000 L 9000 12000 M 1000 12080 L 9000 12080" semantic="glass" instance="g1"/>
 <path d="M 9000 5000 L 12000 5000 M 9000 5300 L 12000 5300 M 9000 5600 L 12000 5600 M 9000 5000 L 9000 6000"
       semantic="staircase" instance="s1"/>
 <line x1="0" y1="9000" x2="14000" y2="9000" semantic="Axis &amp; Grid"/>
 <text x="3000" y="6000" font-size="400" semantic="others">LIVING</text>
</svg>"""


def test_archcad_slice_to_labels(tmp_path):
    sys.path.insert(0, os.path.join(HERE, "..", "training"))
    try:
        import fp_extra_archcad as A
        from fp_common import ROOM_CLASSES, ICON_CLASSES, NOT_WALL
    except ImportError as e:                           # training deps (torch) not installed in this env
        import pytest
        pytest.skip(str(e))
    f = tmp_path / "s.svg"
    f.write_text(SVG)
    rgb, room, icon = A.convert_sample(str(f), 1400)
    s = 1400 / 14000
    at = lambda x, y: (int(y * s), int(x * s))
    W, Win, St = ROOM_CLASSES.index("Wall"), ICON_CLASSES.index("Window"), ICON_CLASSES.index("Stairs")
    assert room[at(9000, 2120)] == W                   # the two wall lines closed into a band
    assert room[at(8250, 7250)] == W                   # column outline filled as wall
    assert icon[at(5000, 12040)] == Win                # glass along a line = window band
    assert icon[at(10500, 5450)] == St
    assert room[at(13000, 9000)] == NOT_WALL           # axis / grid line: known not wall
    assert room[at(3100, 5950)] == NOT_WALL            # text: known not wall
    _, boxes = A.detector_boxes(str(f), 1000)
    assert [b[0] for b in boxes] == ["single_door"]


def test_structural_slices_are_skipped(tmp_path):
    sys.path.insert(0, os.path.join(HERE, "..", "training"))
    try:
        import fp_extra_archcad as A
    except ImportError as e:
        import pytest
        pytest.skip(str(e))
    f = tmp_path / "f.svg"
    f.write_text(SVG.replace('semantic="21"', 'semantic="foundation"'))
    assert A.convert_sample(str(f), 700) is None
