# infra/vllm-omni/Dockerfile
# Fork of runpod-workers/worker-vllm adapted for vllm-omni audio workloads
# Upstream: https://github.com/runpod-workers/worker-vllm/releases/tag/v2.14.0

FROM nvidia/cuda:12.9.1-cudnn-runtime-ubuntu24.04

ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.12 python3-pip python3.12-dev python3.12-venv \
    git curl ffmpeg libsndfile1 \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3.12 /usr/bin/python3 \
    && ln -sf /usr/bin/python3.12 /usr/bin/python

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app

RUN uv venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# vllm base + omni extension
# Pre-built wheels target CUDA 12.9 for v0.18.0
# https://docs.vllm.ai/projects/vllm-omni/en/latest/getting_started/installation/gpu/
RUN uv pip install vllm --torch-backend=auto \
    && uv pip install vllm-omni

COPY requirements.txt /app/requirements.txt
RUN uv pip install -r /app/requirements.txt

COPY src/ /app/src/
COPY start.sh /app/start.sh
RUN chmod +x /app/start.sh

ENV MODEL_PATH=/runpod-volume/models/Qwen3-TTS
ENV SERVED_MODEL_NAME=tts-1
ENV DTYPE=bfloat16
ENV VLLM_PORT=8091

CMD ["/app/start.sh"]
