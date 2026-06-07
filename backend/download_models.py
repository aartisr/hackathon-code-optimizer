"""Model download and cache bootstrap utility for backend inference.

This script is intentionally conservative and repeatable:

- It stores downloaded artifacts under ``backend/model-cache/<model_key>``.
- It writes a small completion marker file after a successful download.
- It validates both marker metadata and essential model files before
    considering a model ready.
- It supports forced re-downloads to recover from stale or partial caches.

Typical usage:

- Download all supported models once:
    ``python backend/download_models.py --all``
- Download a single model:
    ``python backend/download_models.py --model phi3``
- Force re-download:
    ``python backend/download_models.py --all --force``
"""

from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

from huggingface_hub import snapshot_download


MODEL_REPOS = {
    "phi3": "microsoft/Phi-3-mini-4k-instruct",
    "tinyllama": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
}

CACHE_ROOT = Path(__file__).resolve().parent / "model-cache"


def model_cache_dir(model_key: str) -> Path:
    """Return the cache directory path for a specific model key."""
    return CACHE_ROOT / model_key


def marker_path(model_key: str) -> Path:
    """Return the path of the download completion marker for a model."""
    return model_cache_dir(model_key) / ".download-complete.json"


def is_downloaded(model_key: str) -> bool:
    """Check whether a model cache is inference-ready.

    A model is considered downloaded only when all of the following are true:

    - completion marker exists and contains ``status == "ok"``
    - model directory exists
    - tokenizer metadata exists
    - at least one framework checkpoint file exists

    This avoids false positives caused by interrupted or incompatible downloads.
    """
    marker = marker_path(model_key)
    if not marker.exists():
        return False

    try:
        data = json.loads(marker.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return False

    if data.get("status") != "ok":
        return False

    model_dir = model_cache_dir(model_key)
    if not model_dir.exists():
        return False

    has_tokenizer = any(
        (model_dir / file_name).exists()
        for file_name in ["tokenizer.json", "tokenizer.model", "tokenizer_config.json"]
    )
    if not has_tokenizer:
        return False

    has_checkpoint = any(
        model_dir.glob(pattern)
        for pattern in [
            "pytorch_model*.bin",
            "*.safetensors",
            "tf_model.h5",
            "model.ckpt.index",
            "flax_model.msgpack",
        ]
    )
    return has_checkpoint


def write_marker(model_key: str, repo_id: str) -> None:
    """Write a JSON marker indicating a successful model download.

    The marker stores minimal metadata used by readiness checks and diagnostics.
    """
    marker = marker_path(model_key)
    marker.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "status": "ok",
        "repo": repo_id,
        "runtime": "transformers",
        "downloadedAt": datetime.now(timezone.utc).isoformat(),
    }
    marker.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def download_model(model_key: str, force: bool = False) -> None:
    """Download one model snapshot into the local backend cache.

    Args:
        model_key: Canonical model key from ``MODEL_REPOS``.
        force: When true, existing cache content is removed and re-downloaded.

    The function prints user-friendly status lines for CLI visibility and CI logs.
    """
    repo_id = MODEL_REPOS[model_key]
    cache_dir = model_cache_dir(model_key)

    if is_downloaded(model_key) and not force:
        print(f"[skip] {model_key}: already downloaded at {cache_dir}")
        return

    if cache_dir.exists():
        shutil.rmtree(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    print(f"[download] {model_key}: {repo_id}")
    snapshot_download(
        repo_id=repo_id,
        local_dir=str(cache_dir),
        local_dir_use_symlinks=False,
        resume_download=True,
    )
    write_marker(model_key, repo_id)
    print(f"[done] {model_key}: cached at {cache_dir}")


def parse_args() -> argparse.Namespace:
    """Build and parse command-line arguments for this utility."""
    parser = argparse.ArgumentParser(
        description="Download and cache model artifacts once for backend usage."
    )
    parser.add_argument(
        "--model",
        choices=sorted(MODEL_REPOS.keys()),
        help="Download one model by key.",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Download all known models.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-download even if cache marker exists.",
    )
    return parser.parse_args()


def main() -> None:
    """Entry point for command-line execution.

    Validates intent flags and dispatches downloads for requested targets.
    """
    args = parse_args()
    if not args.model and not args.all:
        raise SystemExit("Use --model <key> or --all")

    targets = list(MODEL_REPOS.keys()) if args.all else [args.model]
    for key in targets:
        download_model(key, force=args.force)


if __name__ == "__main__":
    main()
