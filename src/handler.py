"""
RunPod serverless handler for vllm-omni TTS.
Fork of runpod-workers/worker-vllm adapted for audio generation.
"""
import os
import sys
import traceback

import runpod
from runpod import RunPodLogger

log = RunPodLogger()

_engine = None
_skip = os.environ.get("VLLM_OMNI_SKIP") == "1"


async def handler(job: dict) -> dict:
    if _skip:
        return {
            "status": "degraded",
            "message": "model not mounted — attach network volume with weights",
        }

    try:
        from utils import JobInput

        job_input = JobInput(job["input"])
        result = await _engine.generate(job_input)
        return result

    except Exception as e:
        error_str = str(e)
        log.error(f"Inference error: {error_str}")
        log.error(traceback.format_exc())

        if "CUDA" in error_str or "cuda" in error_str:
            log.error("CUDA error — exiting to recycle worker")
            sys.exit(1)

        return {"error": error_str}


if __name__ == "__main__":
    if _skip:
        log.info("Starting in degraded mode (no model)")
    else:
        try:
            from engine import TTSEngine

            _engine = TTSEngine()
            log.info("TTSEngine initialized")
        except Exception as e:
            log.error(f"Startup failed: {e}\n{traceback.format_exc()}")
            sys.exit(1)

    runpod.serverless.start({"handler": handler})
