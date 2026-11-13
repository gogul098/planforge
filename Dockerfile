FROM python:3.11-slim
RUN apt-get update && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 libgomp1 \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY app/requirements.txt .
RUN pip install --no-cache-dir torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r requirements.txt uvicorn
COPY app/ .
ENV PYTHONUNBUFFERED=1 MPLBACKEND=Agg YOLO_CONFIG_DIR=/tmp/yolo HF_HOME=/tmp/hf \
    P3D_NO_MAIN=1 P3D_NO_INSTALL=1 PLANFORGE_JOBS=/tmp/jobs
RUN useradd -m -u 1000 planforge && mkdir -p /tmp/jobs /tmp/hf /tmp/yolo /tmp/planforge-models \
    && chown -R planforge /app /tmp/jobs /tmp/hf /tmp/yolo /tmp/planforge-models
USER planforge
ARG GIT_SHA=dev
ARG BUILT_AT=unknown
ENV PLANFORGE_VERSION=$GIT_SHA PLANFORGE_BUILT_AT=$BUILT_AT
EXPOSE 8080
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s CMD python -c "import urllib.request;urllib.request.urlopen('http://localhost:8080/api/health')" || exit 1
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8080"]
