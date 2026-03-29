#!/usr/bin/env bash
set -euo pipefail

# CUDA forward-compat: register compat libs at runtime in case the
# nvidia-container-runtime replaces driver libs after image build.
ldconfig /usr/local/cuda-12.9/compat/ 2>/dev/null || true

exec python -u /app/src/handler.py
