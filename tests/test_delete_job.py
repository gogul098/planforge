"""Plans are private to the browser that uploaded them, and removed on request (never while converting)."""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "app"))
os.environ.setdefault("P3D_NO_MAIN", "1")
os.environ.setdefault("P3D_NO_INSTALL", "1")
os.environ.setdefault("MPLBACKEND", "Agg")


def _job(S, job, state):
    os.makedirs(os.path.join(S.JOBS, job, "out"), exist_ok=True)
    open(os.path.join(S.JOBS, job, "out", "model.ifc"), "w").write("x")
    S.STATUS[job] = {"state": state}
    return S.new_key(job)


def test_only_the_uploader_can_reach_a_plan():
    import server as S
    from fastapi.testclient import TestClient
    key = _job(S, "privtest01", "done")
    c = TestClient(S.app)
    for hdr in ({}, {"X-Plan-Key": "wrong"}, {"X-Plan-Key": S.new_key("someoneelse")}):
        assert c.get("/api/files/privtest01/model.ifc", headers=hdr).status_code == 404
        assert c.get("/api/status/privtest01", headers=hdr).status_code == 404
        assert c.delete("/api/jobs/privtest01", headers=hdr).status_code == 404
        assert c.post("/api/export/privtest01", json={}, headers=hdr).status_code == 404
    assert c.get("/api/files/privtest01/model.ifc", headers={"X-Plan-Key": key}).status_code == 200
    assert "key" not in c.get("/api/status/privtest01", headers={"X-Plan-Key": key}).text.lower()
    assert c.get("/api/files/privtest01/owner.key", headers={"X-Plan-Key": key}).status_code == 404


def test_delete_finished_job_removes_files():
    import server as S
    from fastapi.testclient import TestClient
    key = _job(S, "deltest01", "done")
    c = TestClient(S.app, headers={"X-Plan-Key": key})
    assert c.delete("/api/jobs/deltest01").json() == {"deleted": True}
    assert not os.path.exists(os.path.join(S.JOBS, "deltest01"))
    assert c.get("/api/files/deltest01/model.ifc").status_code == 404


def test_running_job_is_kept():
    import server as S
    from fastapi.testclient import TestClient
    key = _job(S, "deltest02", "running")
    assert TestClient(S.app, headers={"X-Plan-Key": key}).delete("/api/jobs/deltest02").status_code == 409
    assert os.path.isdir(os.path.join(S.JOBS, "deltest02"))
    S.STATUS.pop("deltest02")
