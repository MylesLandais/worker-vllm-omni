"""
RunPod serverless handler for vllm-omni TTS.

Engine initialization is deferred to the first job request. RunPod may start
containers with NVIDIA_VISIBLE_DEVICES=void and inject the GPU later when a
job is assigned. Starting vllm-omni eagerly at boot would crash before the
worker ever registers.
"""
import os
import sys
import traceback

import runpod
from runpod import RunPodLogger

log = RunPodLogger()

MODEL_PATH = os.environ.get("MODEL_PATH", "/runpod-volume/models/Qwen3-TTS")
_engine_started = False


def _ensure_engine():
    global _engine_started
    if _engine_started:
        return
    from engine import start_engine

    start_engine()
    _engine_started = True


def handler(job: dict) -> dict:
    if not os.path.isdir(MODEL_PATH):
        return {
            "status": "degraded",
            "message": "model not mounted — attach network volume with weights",
        }

    try:
        _ensure_engine()

        from utils import JobInput
        from engine import synthesize

        job_input = JobInput(job["input"])
        return synthesize(job_input)

    except Exception as e:
        error_str = str(e)
        log.error(f"Inference error: {error_str}")
        log.error(traceback.format_exc())

        if "CUDA" in error_str or "cuda" in error_str:
            log.error("CUDA error — exiting to recycle worker")
            sys.exit(1)

        return {"error": error_str}


if __name__ == "__main__":
    log.info(
        f"Starting handler (model_path={MODEL_PATH}, "
        f"exists={os.path.isdir(MODEL_PATH)})"
    )
    runpod.serverless.start({"handler": handler})
