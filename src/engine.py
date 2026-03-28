"""
vllm-omni engine wrapper for TTS inference.

Proxies to the vllm-omni OpenAI-compatible server running on localhost.
References:
  https://github.com/vllm-project/vllm-omni/tree/main/examples/offline_inference/qwen3_tts
"""
import os
import time
import base64

import httpx

VLLM_HOST = f"http://127.0.0.1:{os.environ.get('VLLM_PORT', '8091')}"
SERVED_MODEL = os.environ.get("SERVED_MODEL_NAME", "tts-1")


class TTSEngine:
    """Proxies to the vllm-omni OpenAI-compatible server running on localhost."""

    def __init__(self):
        self._client = httpx.AsyncClient(
            base_url=VLLM_HOST,
            timeout=120.0,
        )

    async def generate(self, job_input: "JobInput") -> dict:
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
        r = await self._client.post("/v1/audio/speech", json=payload)
        latency_ms = int((time.monotonic() - t0) * 1000)

        if r.status_code != 200:
            raise RuntimeError(f"vllm-omni {r.status_code}: {r.text[:300]}")

        return {
            "audio_b64": base64.b64encode(r.content).decode(),
            "sample_rate": 24000,
            "generation_time_ms": latency_ms,
            "model": SERVED_MODEL,
        }
