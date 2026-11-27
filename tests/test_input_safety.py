"""Untrusted uploads: options are range-checked, image bombs and wrong file types are refused early."""
import os
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "app"))
os.environ.setdefault("P3D_NO_MAIN", "1")
os.environ.setdefault("P3D_NO_INSTALL", "1")
os.environ.setdefault("MPLBACKEND", "Agg")


def test_options_are_checked():
    import server as S
    from fastapi import HTTPException
    o = S.clean_options('{"wall_height": "3.2", "furniture": "<script>", "plinth": ""}')
    assert o["wall_height"] == 3.2 and o["furniture"] == "copy" and o["plinth"] is None
    for bad in ('{"wall_height": 1e9}', '{"wall_height": "nan"}', '{"ext_wall": -5}', "[1]", "{oops", '{"dxf_units": 99}'):
        with pytest.raises(HTTPException):
            S.clean_options(bad)


def test_image_bomb_refused(tmp_path):
    import server as S
    from fastapi import HTTPException
    from PIL import Image
    p = tmp_path / "bomb.png"
    Image.new("1", (20000, 20000)).save(p)          # a few kB on disk, 400 MP once decoded
    assert os.path.getsize(p) < 2 * 2 ** 20
    with pytest.raises(HTTPException) as e:
        S.check_image(str(p))
    assert e.value.status_code == 413
    q = tmp_path / "fake.png"
    q.write_bytes(b"not an image")
    with pytest.raises(HTTPException):
        S.check_image(str(q))


def test_unknown_type_refused_before_saving():
    import server as S
    from fastapi.testclient import TestClient
    r = TestClient(S.app).post("/api/convert", files={"file": ("x.exe", b"MZ", "application/octet-stream")})
    assert r.status_code == 415
