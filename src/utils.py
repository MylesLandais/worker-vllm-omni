"""JobInput — input parsing and validation for TTS requests."""
import os
from dataclasses import dataclass
from typing import Optional

MAX_TEXT_LEN = int(os.environ.get("MAX_TEXT_LEN", "500"))


@dataclass
class JobInput:
    text: str
    voice: str = "Vivian"
    speed: float = 1.0
    language: Optional[str] = None
    reference_audio_b64: Optional[str] = None

    def __init__(self, raw: dict):
        text = raw.get("text", "").strip()
        if not text:
            raise ValueError("'text' is required")
        self.text = text[:MAX_TEXT_LEN]
        self.voice = raw.get("voice", "Vivian")
        self.speed = float(raw.get("speed", 1.0))
        self.language = raw.get("language")
        self.reference_audio_b64 = raw.get("reference_audio_b64")
