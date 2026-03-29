"""
vllm-omni engine: subprocess manager + HTTP proxy for TTS inference.

Spawns vllm-omni serve as a child process on first use (deferred init) and
proxies OpenAI-compatible /v1/audio/speech requests to it.
"""
import base64
import os
import subprocess
import sys
import time

import httpx
from runpod import RunPodLogger

log = RunPodLogger()

MODEL_PATH = os.environ.get("MODEL_PATH", "/runpod-volume/models/Qwen3-TTS")
VLLM_PORT = int(os.environ.get("VLLM_PORT", "8091"))
SERVED_MODEL = os.environ.get("SERVED_MODEL_NAME", "tts-1")
DTYPE = os.environ.get("DTYPE", "bfloat16")

_proc = None
_client = None


def start_engine():
    """Spawn vllm-omni serve and block until the /health endpoint is ready."""
    global _proc, _client

    log.info(f"Starting vllm-omni (model={MODEL_PATH} port={VLLM_PORT} dtype={DTYPE})")

    _proc = subprocess.Popen(
        [
            "vllm-omni", "serve", MODEL_PATH,
            "--omni",
            "--served-model-name", SERVED_MODEL,
            "--port", str(VLLM_PORT),
            "--dtype", DTYPE,
            "--trust-remote-code",
            "--gpu-memory-utilization", "0.9",
        ],
        stdout=sys.stdout,
        stderr=sys.stderr,
    )

    for i in range(90):
        if _proc.poll() is not None:
            raise RuntimeError(
                f"vllm-omni exited during startup (code={_proc.returncode})"
            )
        try:
            r = httpx.get(f"http://127.0.0.1:{VLLM_PORT}/health", timeout=2.0)
            if r.status_code == 200:
                log.info(f"vllm-omni ready after {i * 2}s")
                _client = httpx.Client(
                    base_url=f"http://127.0.0.1:{VLLM_PORT}",
                    timeout=120.0,
                )
                return
        except Exception:
            pass
        time.sleep(2)

    raise RuntimeError("vllm-omni failed to become healthy after 180s")


def synthesize(job_input) -> dict:
    """POST to the vllm-omni sidecar and return the audio response."""
    payload = {
        "model": SERVED_MODEL,
        "input": job_input.text,
        "voice": job_input.voice,
        "response_format": "wav",
        "speed": job_input.speed,
    }
    if job_input.language:
        payload["language"] = job_input.language
    if job_input.reference_audio_b64:
        payload["ref_audio"] = job_input.reference_audio_b64

    t0 = time.monotonic()
    r = _client.post("/v1/audio/speech", json=payload)
    latency_ms = int((time.monotonic() - t0) * 1000)

    if r.status_code != 200:
        raise RuntimeError(f"vllm-omni {r.status_code}: {r.text[:300]}")

    return {
        "audio_b64": base64.b64encode(r.content).decode(),
        "sample_rate": 24000,
        "generation_time_ms": latency_ms,
        "model": SERVED_MODEL,
    }
