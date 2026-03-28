"""
Download TTS model weights to a RunPod network volume.

Canonical location: /runpod-volume/comfy/models/TTS/<model>/
Symlinks created:   /runpod-volume/models/<model> -> canonical

Run on a hydration pod with the volume mounted:
    pip install huggingface_hub
    python hydrate.py

Set HF_TOKEN env var if any repo is gated.
"""

import os
from pathlib import Path

from huggingface_hub import snapshot_download

VOLUME = Path(os.environ.get("VOLUME_PATH", "/runpod-volume"))
CANONICAL_DIR = VOLUME / "comfy" / "models" / "TTS"
SYMLINK_DIR = VOLUME / "models"

MODELS = [
    ("Qwen/Qwen3-TTS-12Hz-1.7B-Base", "Qwen3-TTS"),
    ("mistralai/Voxtral-4B-TTS-2603", "Voxtral-4B-TTS"),
]


def hydrate():
    token = os.environ.get("HF_TOKEN")
    CANONICAL_DIR.mkdir(parents=True, exist_ok=True)
    SYMLINK_DIR.mkdir(parents=True, exist_ok=True)

    for repo_id, dirname in MODELS:
        dest = CANONICAL_DIR / dirname
        link = SYMLINK_DIR / dirname

        # Download weights
        if dest.is_dir() and any(dest.iterdir()):
            print(f"[skip] {dest} already populated")
        else:
            print(f"[download] {repo_id} -> {dest}")
            snapshot_download(
                repo_id=repo_id,
                local_dir=str(dest),
                local_dir_use_symlinks=False,
                token=token,
            )
            print(f"[done] {dest}")

        # Create or refresh symlink
        if link.is_symlink():
            link.unlink()
        if not link.exists():
            link.symlink_to(dest)
            print(f"[link] {link} -> {dest}")
        else:
            print(f"[warn] {link} exists and is not a symlink, skipping")


if __name__ == "__main__":
    hydrate()
