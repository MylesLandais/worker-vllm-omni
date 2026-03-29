# infra/vllm-omni/Dockerfile
# RunPod serverless worker for vllm-omni TTS.
#
# Based on the official vllm/vllm-omni image which bundles vLLM, vllm-omni,
# PyTorch, and CUDA runtime with matching versions. We just add the RunPod
# handler, audio libs, and the entrypoint script.

FROM vllm/vllm-omni:v0.16.0

# Register CUDA 12.9 forward-compatibility libs so cu129 torch works
# on hosts with older drivers (matches runpod-workers/worker-vllm pattern).
RUN ldconfig /usr/local/cuda-12.9/compat/

ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl ffmpeg libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app

COPY requirements.txt /app/requirements.txt
RUN uv pip install --no-cache-dir --system -r /app/requirements.txt

COPY src/ /app/src/
COPY start.sh /app/start.sh
RUN chmod +x /app/start.sh

ENV MODEL_PATH=/runpod-volume/models/Qwen3-TTS
ENV SERVED_MODEL_NAME=tts-1
ENV DTYPE=bfloat16
ENV VLLM_PORT=8091

ENTRYPOINT []
CMD ["/app/start.sh"]
