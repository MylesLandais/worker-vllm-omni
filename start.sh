#!/usr/bin/env bash
set -euo pipefail

# CUDA forward-compat: ensure compat libs load before host driver libs
# injected by nvidia-container-runtime. LD_LIBRARY_PATH takes precedence
# over the ldconfig cache and is inherited by subprocesses.
export LD_LIBRARY_PATH=/usr/local/cuda-12.9/compat/:${LD_LIBRARY_PATH:-}
ldconfig /usr/local/cuda-12.9/compat/ 2>/dev/null || true

exec python -u /app/src/handler.py
