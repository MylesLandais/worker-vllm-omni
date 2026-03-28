# infra/vllm-omni/Dockerfile
# Fork of runpod-workers/worker-vllm adapted for vllm-omni audio workloads
# Upstream: https://github.com/runpod-workers/worker-vllm/releases/tag/v2.14.0
#
# Uses python:3.12-slim instead of nvidia/cuda base image. PyTorch/vllm pip
# wheels ship their own CUDA runtime (nvidia-cudnn-cu12, nvidia-cublas-cu12,
# etc.), so the ~4 GB system CUDA from the nvidia base was dead weight.

FROM python:3.12-slim

# Required for nvidia-container-runtime to inject GPU drivers into non-nvidia
# base images. Without these, the container starts without CUDA access.
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility

ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
    git curl ffmpeg libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app

RUN uv venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# vllm base + omni extension
# PyTorch wheels pull nvidia-cudnn-cu12, nvidia-cublas-cu12, etc. as deps.
# No system CUDA needed — pip packages provide all user-space runtime libs.
RUN uv pip install --no-cache-dir vllm --torch-backend=auto \
    && uv pip install --no-cache-dir vllm-omni

COPY requirements.txt /app/requirements.txt
RUN uv pip install --no-cache-dir -r /app/requirements.txt

# Strip build artifacts to reduce image size
RUN find /opt/venv -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null; \
    find /opt/venv -type d -name "tests" -exec rm -rf {} + 2>/dev/null; \
    true

COPY src/ /app/src/
COPY start.sh /app/start.sh
RUN chmod +x /app/start.sh

ENV MODEL_PATH=/runpod-volume/models/Qwen3-TTS
ENV SERVED_MODEL_NAME=tts-1
ENV DTYPE=bfloat16
ENV VLLM_PORT=8091

CMD ["/app/start.sh"]
