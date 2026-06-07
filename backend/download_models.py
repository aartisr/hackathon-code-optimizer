from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from huggingface_hub import snapshot_download


MODEL_REPOS = {
    "phi3": "microsoft/Phi-3-mini-4k-instruct",
    "tinyllama": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
}

CACHE_ROOT = Path(__file__).resolve().parent / "model-cache"


def model_cache_dir(model_key: str) -> Path:
    return CACHE_ROOT / model_key


def marker_path(model_key: str) -> Path:
    return model_cache_dir(model_key) / ".download-complete.json"


def is_downloaded(model_key: str) -> bool:
    marker = marker_path(model_key)
    if not marker.exists():
        return False

    try:
        data = json.loads(marker.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return False

    return data.get("status") == "ok"


def write_marker(model_key: str, repo_id: str) -> None:
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
    repo_id = MODEL_REPOS[model_key]
    cache_dir = model_cache_dir(model_key)

    if is_downloaded(model_key) and not force:
        print(f"[skip] {model_key}: already downloaded at {cache_dir}")
        return

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
    args = parse_args()
    if not args.model and not args.all:
        raise SystemExit("Use --model <key> or --all")

    targets = list(MODEL_REPOS.keys()) if args.all else [args.model]
    for key in targets:
        download_model(key, force=args.force)


if __name__ == "__main__":
    main()
