"""PlanForge on Modal: one GPU container serves the website + API. Deploy: modal deploy modal_app.py"""
import modal

app = modal.App("planforge")
vol = modal.Volume.from_name("planforge-data", create_if_missing=True)   # jobs + model cache survive restarts

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("libgl1", "libglib2.0-0", "libgomp1")
    .pip_install(
        "torch==2.5.1", "torchvision==0.20.1", "timm==1.0.31", "segmentation-models-pytorch==0.6.0",
        "ultralytics==8.3.253", "huggingface_hub", "opencv-python-headless", "scikit-image", "scipy",
        "shapely", "trimesh", "mapbox-earcut", "ifcopenshell>=0.8.3", "rapidocr_onnxruntime==1.2.3",
        "pymupdf", "matplotlib", "ezdxf", "fastapi", "python-multipart")
    .env({"PLANFORGE_JOBS": "/data/jobs", "HF_HOME": "/data/hf", "YOLO_CONFIG_DIR": "/tmp/yolo",
          "MPLBACKEND": "Agg", "P3D_NO_MAIN": "1", "P3D_NO_INSTALL": "1"})
    .add_local_dir("app", "/root/app", ignore=["__pycache__"])
)


@app.function(
    image=image,
    gpu="T4",                      # ~5x faster than CPU; Modal's free monthly credit covers hundreds of plans
    secrets=[modal.Secret.from_name("hf-token")],
    volumes={"/data": vol},
    max_containers=1,              # the pipeline runs one plan at a time; one container keeps job state together
    scaledown_window=600,          # stay warm 10 min after the last request (time to view / edit furniture)
    timeout=1800,
)
@modal.concurrent(max_inputs=32)   # many browser requests (status polls, files) share the container
@modal.asgi_app()
def web():
    import os
    import sys
    os.makedirs("/data/jobs", exist_ok=True)
    sys.path.insert(0, "/root/app")
    os.chdir("/root/app")
    from server import app as fastapi_app
    return fastapi_app
