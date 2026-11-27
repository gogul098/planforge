"""A user's plan and its outputs are removed on request, but never while it is still converting."""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "app"))
os.environ.setdefault("P3D_NO_MAIN", "1")
os.environ.setdefault("P3D_NO_INSTALL", "1")
os.environ.setdefault("MPLBACKEND", "Agg")


def test_delete_finished_job_removes_files():
    import server as S
    from fastapi.testclient import TestClient
    job = "deltest01"
    os.makedirs(os.path.join(S.JOBS, job, "out"), exist_ok=True)
    open(os.path.join(S.JOBS, job, "out", "model.ifc"), "w").write("x")
    S.STATUS[job] = {"state": "done"}
    c = TestClient(S.app)
    assert c.delete("/api/jobs/" + job).json() == {"deleted": True}
    assert not os.path.exists(os.path.join(S.JOBS, job))
    assert c.get("/api/files/{}/model.ifc".format(job)).status_code == 404
    assert c.delete("/api/jobs/" + job).json() == {"deleted": False}


def test_running_job_is_kept():
    import server as S
    from fastapi.testclient import TestClient
    job = "deltest02"
    os.makedirs(os.path.join(S.JOBS, job), exist_ok=True)
    S.STATUS[job] = {"state": "running"}
    assert TestClient(S.app).delete("/api/jobs/" + job).status_code == 409
    assert os.path.isdir(os.path.join(S.JOBS, job))
    S.STATUS.pop(job)
