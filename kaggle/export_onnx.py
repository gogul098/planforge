# ===================== EXPORT: PyTorch checkpoints -> ONNX (run once per new model release) =====================
# Kaggle cell. Needs the Kaggle secret HF_TOKEN with WRITE access.
# Publishes, next to the PyTorch checkpoints on Hugging Face:
#   <you>/floorplan-multitask-effb3-768 : checkpoints/model.onnx   (floor-plan model)
#   <you>/floorplan-furniture-obb       : checkpoints/best.onnx    (furniture detector)
# The PlanForge engine (no PyTorch) loads these. If a "prod" tag exists it is moved to the new commit,
# so the engine picks the ONNX files up on its next start.
import os, subprocess, sys, json

subprocess.run([sys.executable, "-m", "pip", "install", "-q", "segmentation-models-pytorch==0.6.0", "timm==1.0.31",
                "onnx", "onnxruntime", "huggingface_hub", "ultralytics==8.3.253"], check=False)
import numpy as np
import torch
import segmentation_models_pytorch as smp
from huggingface_hub import HfApi, hf_hub_download

HF_REPO = "floorplan-multitask-effb3-768"
FURNITURE_REPO = "floorplan-furniture-obb"       # "" to skip
OUT = "/kaggle/working/onnx" if os.path.isdir("/kaggle/working") else "./onnx"
os.makedirs(OUT, exist_ok=True)


def hf_token():
    t = os.environ.get("HF_TOKEN")
    if not t:
        from kaggle_secrets import UserSecretsClient
        t = UserSecretsClient().get_secret("HF_TOKEN")
    return t


def export_floorplan(ckpt_path, out_path):
    """The multitask U-Net -> ONNX with free height/width (any multiple of 32)."""
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    net = smp.Unet(ck.get("encoder", "tu-efficientnet_b3"), encoder_weights=None, classes=27,
                   decoder_attention_type="scse")
    net.load_state_dict(ck["ema"])
    net.eval()
    x = torch.randn(1, 3, 768, 768)
    torch.onnx.export(net, x, out_path, input_names=["image"], output_names=["logits"], opset_version=17,
                      dynamic_axes={"image": {2: "h", 3: "w"}, "logits": {2: "h", 3: "w"}}, dynamo=False)
    import onnx
    m = onnx.load(out_path)
    info = "epoch {}, val score {:.3f}".format(ck.get("epoch", -1) + 1, ck.get("metrics", {}).get("score", float("nan")))
    meta = {"info": info, "encoder": ck.get("encoder", "tu-efficientnet_b3"),
            # trained on plans turned by 30 / 45 / 60 degrees: the engine then reads each plan once
            "rotation_trained": "1" if ck.get("rotation_trained") else "0"}
    for k, v in meta.items():
        e = m.metadata_props.add(); e.key, e.value = k, v
    onnx.save(m, out_path)
    # same answers as PyTorch?
    import onnxruntime as ort
    s = ort.InferenceSession(out_path, providers=["CPUExecutionProvider"])
    xt = torch.randn(1, 3, 512, 640)
    with torch.no_grad():
        ref = net(xt).numpy()
    got = s.run(None, {"image": xt.numpy()})[0]
    err = float(np.abs(ref - got).max())
    print("floor-plan model -> {} ({:.1f} MB), max difference vs PyTorch {:.2e}".format(
        out_path, os.path.getsize(out_path) / 1e6, err))
    assert err < 1e-2, "ONNX output differs from PyTorch"
    return info


def export_furniture(pt_path, out_dir):
    from ultralytics import YOLO
    p = YOLO(pt_path).export(format="onnx", imgsz=1024, opset=17, simplify=True, dynamic=False, nms=False)
    dst = os.path.join(out_dir, "best.onnx")
    os.replace(p, dst)
    print("furniture detector -> {} ({:.1f} MB)".format(dst, os.path.getsize(dst) / 1e6))
    return dst


def publish(api, repo, local, path_in_repo, msg):
    api.upload_file(path_or_fileobj=local, path_in_repo=path_in_repo, repo_id=repo, repo_type="model", commit_message=msg)
    sha = api.model_info(repo, revision="main").sha
    refs = [t.name for t in api.list_repo_refs(repo).tags]
    if "prod" in refs:                                  # keep the live release pointing at a commit with the ONNX file
        api.delete_tag(repo, tag="prod")
        api.create_tag(repo, tag="prod", revision=sha)
        print("  moved tag prod ->", sha[:8])
    print("  published", repo, path_in_repo, "@", sha[:8])


if __name__ == "__main__" and not os.environ.get("PF_EXPORT_TEST"):
    token = hf_token()
    api = HfApi(token=token)
    me = api.whoami()["name"]
    repo = "{}/{}".format(me, HF_REPO)
    rev = "prod" if "prod" in [t.name for t in api.list_repo_refs(repo).tags] else "main"
    ck = hf_hub_download(repo, "checkpoints/best.pt", revision=rev, token=token)
    info = export_floorplan(ck, os.path.join(OUT, "model.onnx"))
    publish(api, repo, os.path.join(OUT, "model.onnx"), "checkpoints/model.onnx", "ONNX export ({})".format(info))
    if FURNITURE_REPO:
        frepo = "{}/{}".format(me, FURNITURE_REPO)
        try:
            frev = "prod" if "prod" in [t.name for t in api.list_repo_refs(frepo).tags] else "main"
            fpt = hf_hub_download(frepo, "checkpoints/best.pt", revision=frev, token=token)
            publish(api, frepo, export_furniture(fpt, OUT), "checkpoints/best.onnx", "ONNX export")
        except Exception as e:
            print("furniture detector not exported:", e)
    print("Done. Restart the Azure Web App (or wait for its next start) to load the ONNX models.")
