"""Steps beside open ground the model marks inside the building go DOWN into it (sandpit, sunken court), and the
architect can switch any stair area down / up / stair / flat from the viewer."""
import io
import json
import os
import sys
import contextlib

import cv2
import numpy as np
import pytest

APP = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "app")


@pytest.fixture(scope="module")
def job(tmp_path_factory):
    d = tmp_path_factory.mktemp("sunken")
    os.environ.update(P3D_NO_MAIN="1", P3D_NO_INSTALL="1", MPLBACKEND="Agg", HF_TOKEN="none",
                      P3D_TEST_MASKS=str(d / "masks"), PLANFORGE_JOBS=str(d / "jobs"))
    sys.path.insert(0, APP)
    import server as S
    S.JOBS = str(d / "jobs")
    P = S.P
    R = {"Background": 0, "Outdoor": 1, "Wall": P.WALL, "OtherRoom": 11}   # room-head ids (CubiCasa5K order)
    H, W = 900, 1200
    room = np.full((H, W), R["Background"], np.uint8)
    icon = np.zeros((H, W), np.uint8)
    room[100:800, 100:1100] = R["OtherRoom"]                       # building floor
    for y0, y1, x0, x1 in ((100, 112, 100, 1100), (788, 800, 100, 1100), (100, 800, 100, 112), (100, 800, 1088, 1100)):
        room[y0:y1, x0:x1] = R["Wall"]
    room[400:600, 520:760] = R["Background"]                       # the sandpit: open ground inside the building
    icon[400:600, 420:520] = P.STAIRS                              # tiers beside it
    img = np.full((H, W, 3), 255, np.uint8)
    img[room == R["Wall"]] = 0
    for x in range(430, 520, 18):                                  # drawn tread lines
        cv2.line(img, (x, 400), (x, 600), (0, 0, 0), 2)
    os.makedirs(d / "masks")
    for n, a in (("img", img), ("room", room), ("icon", icon)):
        np.save(d / "masks" / (n + ".npy"), a)
    cv2.imwrite(str(d / "plan.png"), img)
    os.makedirs(os.path.join(S.JOBS, "t"), exist_ok=True)
    with contextlib.redirect_stdout(io.StringIO()):
        S.run_job("t", str(d / "plan.png"), ".png", {"furniture": "none", "plinth": "0.6"})
    assert S.STATUS["t"]["state"] == "done", S.STATUS["t"].get("msg")
    return S, os.path.join(S.JOBS, "t", "out", "plan.json")


def test_steps_beside_open_ground_go_down(job):
    S, jp = job
    plan = json.load(open(jp))
    down = [s for s in plan["stairs"] if s.get("sunken")]
    assert down, [(s["kind"], s.get("direction")) for s in plan["stairs"]]
    st = down[0]
    assert st["pit"], "the open ground the model marked is attached to the steps"
    assert abs(st["base_z"] + 0.6) < 1e-6 and st["top_z"] == 0.0         # floor raised 0.6 m, pit at ground
    z = [t["z1"] for t in st["treads"]]
    assert all(-0.6 < v < 0 for v in z) and len(set(z)) == len(z), z     # every tier between floor and pit
    from shapely.geometry import Polygon
    assert sum(Polygon(h).area for h in plan["floor"]["holes"]) > 0.8 * Polygon(st["pit"][0]).area


def test_architect_switches_stairs(job):
    S, jp = job
    from fastapi.testclient import TestClient
    c = TestClient(S.app)
    sid = json.load(open(jp))["stairs"][0]["id"]
    with contextlib.redirect_stdout(io.StringIO()):
        assert c.post("/api/stair/t", json={"id": sid, "mode": "up", "depth": 0.9}).status_code == 200
        st = [s for s in json.load(open(jp))["stairs"] if s["id"] == sid][0]
        assert not st["sunken"] and st["top_z"] == pytest.approx(0.9)
        assert c.post("/api/stair/t", json={"id": sid, "mode": "down", "depth": 0.45}).status_code == 200
        st = [s for s in json.load(open(jp))["stairs"] if s["id"] == sid][0]
        assert st["sunken"] and st["base_z"] == pytest.approx(-0.45)
        assert c.post("/api/stair/t", json={"id": sid, "mode": "flat"}).status_code == 200
        assert all(s["id"] != sid for s in json.load(open(jp))["stairs"])
        assert c.post("/api/stair/t", json={"id": 999, "mode": "down"}).status_code == 404


def test_stair_note_beside_the_flight_is_read(job):
    """'16R @ 150, T 300' written beside a flight: its centre is further than the search radius from the stair's
    centre, but the text is next to the flight, so it counts (risers, riser height and tread)."""
    S, _ = job
    P = S.P
    toks = [{"kind": "label", "text": "16R @ 150, T 300", "raw": "16R @ 150, T 300", "x": 483.0, "y": 679.0, "h": 13.0},
            {"kind": "label", "text": "UP", "raw": "UP", "x": 433.0, "y": 771.0, "h": 13.0}]
    box = (410, 700, 460, 760)
    far = P._stair_notes(toks, 435, 730, 57)                         # old: from the centre -> the note is missed
    near = P._stair_notes(toks, 435, 730, 57, box=box)
    assert "risers" not in far
    assert near["risers"] == 16 and near["riser"] == pytest.approx(0.15) and near["tread"] == pytest.approx(0.30)
    assert near["up"]
