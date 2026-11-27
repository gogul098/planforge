"""Production limits: big sheets are read in tiles (bounded memory), the queue is bounded."""
import os
import sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "app"))
os.environ.setdefault("P3D_NO_MAIN", "1")
os.environ.setdefault("P3D_NO_INSTALL", "1")
os.environ.setdefault("MPLBACKEND", "Agg")
import pipeline as P  # noqa: E402


class FakeOCR:
    """Finds the black squares in a tile; reports a box around each with its global centre as the text."""
    def __init__(self, marks):
        self.marks, self.calls = marks, 0

    def __call__(self, tile):
        self.calls += 1
        out = []
        for (x, y) in self.marks:
            pass
        ys, xs = np.nonzero(tile[:, :, 0] < 50)
        if not len(xs):
            return None, None
        import cv2
        n, lab, st, cen = cv2.connectedComponentsWithStats((tile[:, :, 0] < 50).astype(np.uint8))
        for i in range(1, n):
            x, y, w, h = st[i, :4]
            out.append(([[x, y], [x + w, y], [x + w, y + h], [x, y + h]], "t", 0.9))
        return out, None


def test_big_sheet_read_in_tiles_without_duplicates():
    H, W = 3000, 4000
    img = np.full((H, W, 3), 255, np.uint8)
    marks = [(100, 100), (1350, 900), (1400, 1400), (3900, 2900), (2600, 1250), (1250, 2700)]   # some in overlaps
    for x, y in marks:
        img[y - 10:y + 10, x - 30:x + 30] = 0
    ocr = FakeOCR(marks)
    res = P._ocr_tiled(ocr, img)
    assert ocr.calls > 1                                    # tiled
    centres = sorted((round(sum(q[0] for q in b) / 4), round(sum(q[1] for q in b) / 4)) for b, _, _ in res)
    assert len(centres) == len(marks), centres              # every text once, none twice
    for (x, y), (cx, cy) in zip(sorted(marks), centres):
        assert abs(x - cx) <= 1 and abs(y - cy) <= 1        # in sheet coordinates


def test_normal_upload_read_in_one_go():
    img = np.full((1792, 2400, 3), 255, np.uint8)
    img[500:520, 500:560] = 0
    ocr = FakeOCR([])
    assert len(P._ocr_tiled(ocr, img)) == 1 and ocr.calls == 1


def test_queue_is_bounded():
    import server as S
    from fastapi.testclient import TestClient
    saved = dict(S.STATUS)
    try:
        for i in range(S.MAX_QUEUE):
            S.STATUS["busy%d" % i] = {"state": "queued" if i else "running", "queued_at": i}
        r = TestClient(S.app).post("/api/convert", files={"file": ("a.png", b"x", "image/png")})
        assert r.status_code == 503 and "busy" in r.json()["detail"]
        S.STATUS["late"] = {"state": "queued", "queued_at": 99}
        st = TestClient(S.app, headers={"X-Plan-Key": S.new_key("late")}).get("/api/status/late").json()
        assert "ahead" in st["msg"]
    finally:
        S.STATUS.clear()
        S.STATUS.update(saved)
