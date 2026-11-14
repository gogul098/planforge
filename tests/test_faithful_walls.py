"""The converter projects the model's wall mask; it must not add or remove wall.
Synthetic model outputs (room head: 2 = wall, icon head: 1 = window, 2 = door) go through vectorise + build_plan,
and the resulting walls are drawn back onto the mask grid and compared pixel by pixel.

    pip install pytest numpy opencv-python-headless scikit-image scipy shapely matplotlib
    pytest tests
"""
import contextlib
import io
import os
import sys

os.environ.update(P3D_NO_MAIN="1", P3D_NO_INSTALL="1", MPLBACKEND="Agg")
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "app"))

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402

import pipeline as P  # noqa: E402

S = 0.025          # metres per model pixel used in the tests
K3 = np.ones((3, 3), np.uint8)


def house(h=900, w=1400):
    room = np.full((h, w), 3, np.uint8)
    icon = np.zeros((h, w), np.uint8)
    room[100:110, 100:1300] = 2
    room[790:800, 100:1300] = 2
    room[100:800, 100:110] = 2
    room[100:800, 1290:1300] = 2
    return room, icon


def convert(room, icon):
    with contextlib.redirect_stdout(io.StringIO()):
        v = P.vectorise(room, icon, None)
        sol = P.uniform_sol(v, S, "test")
        plan = P.build_plan(v, sol, [], room)
    x0, y0 = sol["axes"]["x"]["px"][0], sol["axes"]["y"]["px"][0]
    walls = []
    for w in plan["walls"]:
        a = np.array([w["start"][0] / S + x0, w["start"][1] / S + y0])
        b = np.array([w["end"][0] / S + x0, w["end"][1] / S + y0])
        walls.append((a, b, w["thickness"] / S, [e / S for e in w["ext"]]))
    return walls


def raster(walls, shape):
    out = np.zeros(shape, np.uint8)
    for a, b, t, ext in walls:
        d = (b - a) / max(np.linalg.norm(b - a), 1e-9)
        n = np.array([-d[1], d[0]]) * t / 2
        a2, b2 = a - d * ext[0], b + d * ext[1]
        cv2.fillPoly(out, [np.round(np.array([a2 - n, b2 - n, b2 + n, a2 + n])).astype(np.int32)], 1)
    return out > 0


def added_missed(room, icon, walls):
    mw = room == 2
    out = raster(walls, mw.shape)
    added = out & ~(cv2.dilate((mw | (icon > 0)).astype(np.uint8), K3) > 0)
    missed = mw & ~(cv2.dilate(out.astype(np.uint8), K3) > 0)
    return int(added.sum()), int(missed.sum())


def wall_through(walls, x, y):
    """Is there an output wall over pixel (x, y)?"""
    p = np.array([x, y], float)
    for a, b, t, ext in walls:
        d = (b - a) / max(np.linalg.norm(b - a), 1e-9)
        u = (p - a) @ d
        if -ext[0] - 0.5 <= u <= np.linalg.norm(b - a) + ext[1] + 0.5 and abs((p - a) @ np.array([-d[1], d[0]])) <= t / 2:
            return True
    return False


@pytest.mark.parametrize("gap", [4, 8, 16, 20, 30])
def test_breaks_in_a_wall_stay_open(gap):
    room, icon = house()
    room[200:700, 697:703] = 2
    room[450 - gap // 2:450 + (gap + 1) // 2, 697:703] = 3
    walls = convert(room, icon)
    assert not wall_through(walls, 700, 450), "a {} px break the model left was walled over".format(gap)
    assert wall_through(walls, 700, 300) and wall_through(walls, 700, 600)


@pytest.mark.parametrize("length", [10, 15, 20, 30])
def test_short_wall_pieces_are_kept(length):
    room, icon = house()
    room[400:406, 600:600 + length] = 2               # free standing
    room[600:606, 110:110 + length] = 2               # attached to the outer wall
    walls = convert(room, icon)
    assert wall_through(walls, 600 + length // 2, 402)
    assert wall_through(walls, 110 + length // 2, 602)


@pytest.mark.parametrize("t", [3, 5, 8, 12, 20])
def test_thickness_is_as_drawn(t):
    room, icon = house()
    room[200:700, 700:700 + t] = 2
    walls = convert(room, icon)
    got = [w[2] for w in walls if abs(w[0][1] - w[1][1]) > 100 and 690 < w[0][0] < 725]
    assert got and abs(got[0] - t) < 0.5, "mask {} px -> wall {}".format(t, got)


def test_walls_are_not_shifted():
    room, icon = house()
    walls = convert(room, icon)
    left = [w for w in walls if abs(w[0][0] - w[1][0]) < 1e-6 and w[0][0] < 200]
    assert left and abs(left[0][0][0] - 104.5) < 0.01     # pixels 100..109 -> centre-line 104.5


def test_column_curve_and_angled_wall_projected():
    room, icon = house()
    room[540:570, 250:280] = 2                                            # square column
    cv2.line(room, (520, 420), (760, 660), 2, 8)                          # 45 degree wall
    cv2.ellipse(room, (800, 250), (120, 120), 0, 180, 270, 2, 8)          # curved wall
    walls = convert(room, icon)
    added, missed = added_missed(room, icon, walls)
    mw = int((room == 2).sum())
    assert missed <= 0.002 * mw, "model wall lost: {} px".format(missed)
    assert added <= 0.005 * mw, "wall added where the model has none: {} px".format(added)
    assert wall_through(walls, 265, 555)


def test_doors_windows_and_open_sides():
    room, icon = house()
    room[790:800, 300:340] = 3; icon[786:804, 300:340] = 2                # door in the bottom wall
    room[100:110, 500:580] = 3; icon[98:112, 500:580] = 1                 # window in the top wall
    room[400:450, 1290:1300] = 3                                          # open side, nothing marked
    walls = convert(room, icon)
    assert not wall_through(walls, 1295, 425), "an open side the model left open was walled over"
    added, missed = added_missed(room, icon, walls)
    assert added == 0 and missed == 0


def _hexagon(noise):
    """A building with 30 / 60 degree walls whose mask edges are ragged (as from a JPEG / phone photo)."""
    room = np.full((1000, 1300), 3, np.uint8)
    c, R = np.array([650, 500]), 380
    pts = [c + R * np.array([np.cos(np.radians(a)), np.sin(np.radians(a))]) for a in range(0, 360, 60)]
    lines = [(pts[i], pts[(i + 1) % 6]) for i in range(6)]
    lines += [(c, c + 0.85 * R * np.array([np.cos(np.radians(a)), np.sin(np.radians(a))])) for a in (30, 150)]
    for a, b in lines:
        cv2.line(room, tuple(np.round(a).astype(int)), tuple(np.round(b).astype(int)), 2, 10)
    if noise:
        rs = np.random.RandomState(0)
        m = room == 2
        edge = (cv2.dilate(m.astype(np.uint8), K3) > 0) & ~(cv2.erode(m.astype(np.uint8), K3) > 0)
        flip = edge & (rs.rand(*room.shape) < noise)
        room[flip & m] = 3
        room[flip & ~m] = 2
    return room, lines


@pytest.mark.parametrize("noise", [0.0, 0.3])
def test_angled_walls_are_one_wall_each(noise):
    room, lines = _hexagon(noise)
    icon = np.zeros_like(room)
    walls = convert(room, icon)
    assert len(walls) == len(lines), "{} walls for {} drawn 30/60 degree walls".format(len(walls), len(lines))
    for a, b in lines:
        mid = (np.array(a) + np.array(b)) / 2
        assert wall_through(walls, *mid)
    added, missed = added_missed(room, icon, walls)
    assert missed <= 0.01 * (room == 2).sum()


def test_walls_each_at_their_own_angle():
    """A photographed / hand-drawn plan: every wing at a slightly different angle (26-33 and 58-64 degrees)."""
    room = np.full((1100, 1400), 3, np.uint8)
    rs = np.random.RandomState(1)
    lines = []
    for k, ang in enumerate([26.5, 29.0, 31.5, 33.0, 58.5, 61.0, 62.5, 64.0, 151.0, 118.5]):
        cx, cy = 200 + (k % 5) * 250, 300 + (k // 5) * 450
        d = np.array([np.cos(np.radians(ang)), np.sin(np.radians(ang))]) * 110
        a, b = np.array([cx, cy]) - d, np.array([cx, cy]) + d
        cv2.line(room, tuple(np.round(a).astype(int)), tuple(np.round(b).astype(int)), 2, 10)
        lines.append((a, b))
    m = room == 2
    edge = (cv2.dilate(m.astype(np.uint8), K3) > 0) & ~(cv2.erode(m.astype(np.uint8), K3) > 0)
    flip = edge & (rs.rand(*room.shape) < 0.3)
    room[flip & m] = 3
    room[flip & ~m] = 2
    walls = convert(room, np.zeros_like(room))
    for a, b in lines:
        d = (b - a) / np.linalg.norm(b - a)
        along = [w for w in walls if abs(abs((w[1] - w[0]) @ d) / max(np.linalg.norm(w[1] - w[0]), 1e-9) - 1) < 2e-3
                 and abs(((w[0] + w[1]) / 2 - a) @ np.array([-d[1], d[0]])) < 6]
        assert len(along) == 1, "wall at {:.1f} deg came out as {} pieces".format(
            np.degrees(np.arctan2(d[1], d[0])) % 180, len(along))
        assert np.linalg.norm(along[0][1] - along[0][0]) > 0.85 * np.linalg.norm(b - a)
