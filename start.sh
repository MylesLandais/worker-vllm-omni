#!/usr/bin/env bash
set -euo pipefail

MODEL_PATH=${MODEL_PATH:-/runpod-volume/models/Qwen3-TTS}
VLLM_PORT=${VLLM_PORT:-8091}
DTYPE=${DTYPE:-bfloat16}
SERVED_MODEL_NAME=${SERVED_MODEL_NAME:-tts-1}

# If model weights aren't present (e.g. RunPod test pod without volume),
# skip the vllm-omni sidecar and start the handler in degraded mode.
if [ ! -d "$MODEL_PATH" ]; then
  echo "[start] model not found at $MODEL_PATH — starting handler without engine"
  export VLLM_OMNI_SKIP=1
  exec python -u /app/src/handler.py
fi

echo "[start] model=$MODEL_PATH dtype=$DTYPE port=$VLLM_PORT"

# Start vllm-omni OpenAI-compatible server in background
vllm-omni serve "$MODEL_PATH" \
  --omni \
  --served-model-name "$SERVED_MODEL_NAME" \
  --port "$VLLM_PORT" \
  --dtype "$DTYPE" \
  --trust-remote-code \
  --gpu-memory-utilization 0.9 \
  &

VLLM_PID=$!

# Health poll — fail fast if server dies
READY=0
for i in $(seq 1 90); do
  if ! kill -0 $VLLM_PID 2>/dev/null; then
    echo "[start] vllm-omni process died during startup"
    exit 1
  fi
  if curl -sf "http://127.0.0.1:$VLLM_PORT/health" > /dev/null 2>&1; then
    echo "[start] vllm-omni ready (${i}x2s)"
    READY=1
    break
  fi
  sleep 2
done

if [ "$READY" -ne 1 ]; then
  echo "[start] vllm-omni failed to become healthy after 180s"
  exit 1
fi

exec python -u /app/src/handler.py
