"""FastAPI backend for code optimization.

This service exposes three primary HTTP endpoints:

- ``GET /health``: liveness probe for infrastructure and UI checks.
- ``GET /status``: readiness and diagnostics for model/runtime state.
- ``POST /optimize``: optimization entry point with model-first or heuristic
    fallback behavior.

Design goals:

- Keep request/response contracts explicit with Pydantic models.
- Be honest about optimization source (model vs heuristic).
- Support strict "model-required" mode when users need guaranteed inference.
- Provide transparent timing and model metadata to the frontend.
"""

from __future__ import annotations

import re
import time
from pathlib import Path
from typing import Literal, Optional, Union

from fastapi import FastAPI
from fastapi import HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

try:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
except Exception:  # pragma: no cover - optional dependency/runtime guard
    torch = None
    AutoModelForCausalLM = None
    AutoTokenizer = None


class OptimizeRequest(BaseModel):
    """Input payload for optimization requests.

    Attributes:
        code: Source code to optimize. Must be non-empty.
        language: User-selected language hint.
        goal: User-selected optimization goal.
        strict: Whether behavior-preserving optimization is required.
        performanceMode: Runtime generation profile.
        preferredModel: Backend model preference (or auto-selection).
        requireModel: If true, reject requests when model inference is unavailable.
    """

    code: str = Field(min_length=1)
    language: str = "Auto detect"
    goal: str = "Balanced cleanup"
    strict: bool = True
    performanceMode: Literal["fast", "balanced", "quality"] = "fast"
    preferredModel: Literal["auto", "phi3", "tinyllama"] = "auto"
    requireModel: bool = True


class TimingBreakdown(BaseModel):
    """Detailed timing breakdown returned to the frontend.

    All values are milliseconds and optional to accommodate fallback paths.
    """

    modelLoadMs: Optional[float] = None
    generationMs: Optional[float] = None
    postProcessMs: Optional[float] = None
    localOptimizeMs: Optional[float] = None


class OptimizeResponse(BaseModel):
    """Normalized optimization result returned by the backend."""

    optimizedCode: str
    changes: list[str]
    qualityScore: int
    impactScore: int
    source: str
    sourceLabel: str
    modelUsed: str
    modelUsedLabel: str
    timings: TimingBreakdown


app = FastAPI(title="Code Optimizer Backend", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


MODEL_CACHE_ROOT = Path(__file__).resolve().parent / "model-cache"
MODEL_KEYS = {
    "phi3": "microsoft/Phi-3-mini-4k-instruct",
    "tinyllama": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
}
_loaded_model_key: Optional[str] = None
_loaded_tokenizer = None
_loaded_model = None


def inspect_model_cache(model_key: str) -> dict[str, Union[bool, str, int]]:
    """Inspect cached files for a model and return readiness diagnostics.

    The returned object is used by ``GET /status`` so the UI can explain
    availability issues before a user runs optimization.
    """

    model_dir = MODEL_CACHE_ROOT / model_key
    marker_exists = (model_dir / ".download-complete.json").exists()
    dir_exists = model_dir.exists()

    tokenizer_files = ["tokenizer.json", "tokenizer.model", "tokenizer_config.json"]
    checkpoint_patterns = [
        "pytorch_model*.bin",
        "*.safetensors",
        "tf_model.h5",
        "model.ckpt.index",
        "flax_model.msgpack",
    ]

    tokenizer_present = any((model_dir / file_name).exists() for file_name in tokenizer_files)
    checkpoint_count = sum(len(list(model_dir.glob(pattern))) for pattern in checkpoint_patterns) if dir_exists else 0
    inference_ready = marker_exists and dir_exists and tokenizer_present and checkpoint_count > 0

    return {
        "key": model_key,
        "repo": MODEL_KEYS[model_key],
        "directoryExists": dir_exists,
        "markerExists": marker_exists,
        "tokenizerPresent": tokenizer_present,
        "checkpointCount": checkpoint_count,
        "inferenceReady": inference_ready,
    }


def auto_selection_reason() -> str:
    """Describe why automatic model selection chose its current result."""
    if torch is None:
        return "transformers-runtime-missing"
    if not torch.cuda.is_available():
        return "auto-disabled-on-cpu"
    if model_inference_ready("phi3"):
        return "phi3-ready"
    if model_inference_ready("tinyllama"):
        return "tinyllama-ready"
    return "no-inference-ready-cache"


def model_cached(model_key: str) -> bool:
    """Return True when the model marker file exists in cache."""
    marker = MODEL_CACHE_ROOT / model_key / ".download-complete.json"
    return marker.exists()


def model_inference_ready(model_key: str) -> bool:
    """Return True when a model cache has all files needed for inference.

    Readiness requires marker, directory, tokenizer files, and at least one
    recognized checkpoint artifact.
    """

    model_dir = MODEL_CACHE_ROOT / model_key
    if not model_cached(model_key):
        return False

    if not model_dir.exists():
        return False

    has_tokenizer = any(
        (model_dir / file_name).exists()
        for file_name in ["tokenizer.json", "tokenizer.model", "tokenizer_config.json"]
    )
    if not has_tokenizer:
        return False

    has_checkpoint = any(
        bool(list(model_dir.glob(pattern)))
        for pattern in [
            "pytorch_model*.bin",
            "*.safetensors",
            "tf_model.h5",
            "model.ckpt.index",
            "flax_model.msgpack",
        ]
    )
    return has_checkpoint


def resolve_requested_model(preferred: str) -> str:
    """Resolve model preference into a concrete model key or ``none``.

    ``auto`` mode may intentionally return ``none`` on CPU-only environments to
    avoid very slow first-token latency from large local models.
    """

    if preferred == "phi3" and model_inference_ready("phi3"):
        return "phi3"
    if preferred == "tinyllama" and model_inference_ready("tinyllama"):
        return "tinyllama"
    if preferred == "auto":
        if torch is not None and not torch.cuda.is_available():
            # Avoid automatically loading a large model on CPU for fast frontend response.
            return "none"
        if model_inference_ready("phi3"):
            return "phi3"
        if model_inference_ready("tinyllama"):
            return "tinyllama"
    return "none"


def build_backend_prompt(payload: OptimizeRequest) -> str:
    """Construct the backend generation prompt using frontend request intent."""
    strict_text = "yes" if payload.strict else "no"
    return (
        "You are a senior code optimizer. Optimize the user's code while preserving behavior.\n\n"
        f"Language: {payload.language}\n"
        f"Goal: {payload.goal}\n"
        f"Preserve behavior strictly: {strict_text}\n\n"
        "CRITICAL OUTPUT FORMAT RULES:\n"
        "- Return ONLY a valid JSON object.\n"
        "- No markdown fences.\n"
        "- No preface or explanation.\n"
        "- No extra keys.\n\n"
        "Return exactly this JSON shape:\n"
        "{\n"
        '  "optimizedCode": "the complete optimized code"\n'
        "}\n\n"
        "Code:\n"
        "```\n"
        f"{payload.code}\n"
        "```"
    )


def extract_optimized_code(raw_text: str) -> Optional[str]:
    """Extract optimized code from model output.

    Parsing strategy:

    1. Try strict JSON payload extraction (preferred contract).
    2. Try fenced code block extraction.
    3. Fall back to trimmed raw output if non-empty.
    """

    text = (raw_text or "").strip()
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        try:
            import json

            data = json.loads(text[start : end + 1])
            value = data.get("optimizedCode")
            if isinstance(value, str) and value.strip():
                return value
        except Exception:
            pass

    block_match = re.search(r"```(?:\\w+)?\\n([\\s\\S]*?)```", text)
    if block_match:
        candidate = block_match.group(1).strip()
        if candidate:
            return candidate

    return text if text else None


def generation_config(performance_mode: str) -> dict[str, Union[float, int]]:
    """Return generation parameters for the selected performance profile."""
    if performance_mode == "quality":
        return {"max_new_tokens": 640, "temperature": 0.3, "top_p": 0.95}
    if performance_mode == "balanced":
        return {"max_new_tokens": 420, "temperature": 0.2, "top_p": 0.9}
    return {"max_new_tokens": 260, "temperature": 0.1, "top_p": 0.85}


def get_loaded_runtime(model_key: str):
    """Load (or reuse) tokenizer/model runtime objects for a model key.

    Returns tokenizer, model, and load duration in milliseconds.
    """

    global _loaded_model_key, _loaded_model, _loaded_tokenizer

    if _loaded_model_key == model_key and _loaded_model is not None and _loaded_tokenizer is not None:
        return _loaded_tokenizer, _loaded_model, 0.0

    if AutoTokenizer is None or AutoModelForCausalLM is None or torch is None:
        raise RuntimeError("Transformers runtime is not installed. Install backend requirements first.")

    model_dir = MODEL_CACHE_ROOT / model_key
    if not model_dir.exists():
        raise RuntimeError(f"Model directory missing: {model_dir}")

    load_started = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir), local_files_only=True)

    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        str(model_dir),
        local_files_only=True,
        dtype=dtype,
    )
    model.eval()
    if torch.cuda.is_available():
        model = model.to("cuda")

    _loaded_model_key = model_key
    _loaded_model = model
    _loaded_tokenizer = tokenizer
    load_ms = max(0, round((time.perf_counter() - load_started) * 1000, 2))
    return tokenizer, model, load_ms


def run_model_optimization(payload: OptimizeRequest, model_key: str) -> OptimizeResponse:
    """Execute model-based optimization and normalize backend response shape."""
    prompt = build_backend_prompt(payload)
    tokenizer, model, model_load_ms = get_loaded_runtime(model_key)

    generation_started = time.perf_counter()
    inputs = tokenizer(prompt, return_tensors="pt")
    if torch.cuda.is_available():
        inputs = {k: v.to("cuda") for k, v in inputs.items()}

    cfg = generation_config(payload.performanceMode)
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=int(cfg["max_new_tokens"]),
            do_sample=bool(float(cfg["temperature"]) > 0.0),
            temperature=float(cfg["temperature"]),
            top_p=float(cfg["top_p"]),
            pad_token_id=tokenizer.eos_token_id,
        )

    input_token_count = inputs["input_ids"].shape[-1]
    generated_ids = output_ids[0][input_token_count:]
    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    generation_ms = max(0, round((time.perf_counter() - generation_started) * 1000, 2))

    post_started = time.perf_counter()
    optimized_code = extract_optimized_code(generated_text)
    if not optimized_code:
        raise RuntimeError("Model response did not contain usable optimized code.")

    change_notes = [
        "Backend model inference produced the optimization.",
        "Output was parsed and validated for usable code content.",
    ]

    score_bias = 10 if payload.performanceMode == "quality" else (6 if payload.performanceMode == "balanced" else 2)
    quality_score = min(96, 80 + score_bias)
    impact_score = min(92, 64 + score_bias)
    post_ms = max(0, round((time.perf_counter() - post_started) * 1000, 2))

    model_name = MODEL_KEYS[model_key]
    return OptimizeResponse(
        optimizedCode=optimized_code,
        changes=change_notes,
        qualityScore=quality_score,
        impactScore=impact_score,
        source="backend-model",
        sourceLabel="Backend model inference",
        modelUsed=model_key,
        modelUsedLabel=f"Backend model: {model_name}",
        timings=TimingBreakdown(
            modelLoadMs=model_load_ms,
            generationMs=generation_ms,
            postProcessMs=post_ms,
            localOptimizeMs=0,
        ),
    )


def run_local_optimization(code: str, reason: str, model_used: str, model_used_label: str) -> OptimizeResponse:
    """Execute heuristic optimization as a deterministic fallback path.

    This path avoids LLM dependencies and provides baseline code cleanups with
    transparent source labeling.
    """

    started_at = time.perf_counter()
    optimized = code
    changes: list[str] = []

    if re.search(r"\bvar\b", optimized):
        optimized = re.sub(r"\bvar\b", "let", optimized)
        changes.append("Replaced legacy var declarations with block-scoped let.")

    if re.search(r"(^|[^=!])==(?!=)", optimized) or re.search(r"(^|[^=!])!=(?!=)", optimized):
        optimized = re.sub(r"(^|[^=!])==(?!=)", r"\1===", optimized)
        optimized = re.sub(r"(^|[^=!])!=(?!=)", r"\1!==", optimized)
        changes.append("Replaced loose equality checks with strict comparisons.")

    if re.search(r"= [^;\n]+ \+ ", optimized):
        optimized = re.sub(r"(\w+)\s*=\s*\1\s*\+\s*", r"\1 += ", optimized)
        changes.append("Simplified repeated assignment into compound assignment.")

    if re.search(r"if\s*\(([^)]*)\s*===\s*true\)", optimized):
        optimized = re.sub(r"if\s*\(([^)]*)\s*===\s*true\)", lambda m: f"if ({m.group(1).strip()})", optimized)
        changes.append("Removed redundant boolean comparison.")

    if re.search(r"for\s*\(let i = 0; i < ([^.]+)\.length; i\+\+\)", optimized):
        changes.append("Detected index-based looping; a model pass may convert this to iteration helpers when behavior is safe.")

    if not changes:
        changes.append("Code is already compact enough for the backend local optimizer.")

    quality_score = min(94, 66 + len(changes) * 7 + (8 if len(code) > 120 else 0))
    impact_score = min(88, 42 + len(changes) * 10)
    local_ms = max(0, round((time.perf_counter() - started_at) * 1000, 2))

    return OptimizeResponse(
        optimizedCode=optimized,
        changes=changes,
        qualityScore=quality_score,
        impactScore=impact_score,
        source="backend-local",
        sourceLabel=reason,
        modelUsed=model_used,
        modelUsedLabel=model_used_label,
        timings=TimingBreakdown(
            modelLoadMs=0,
            generationMs=0,
            postProcessMs=0,
            localOptimizeMs=local_ms,
        ),
    )


@app.get("/health")
def health() -> dict[str, str]:
    """Liveness endpoint used for basic service health checks."""
    return {"status": "ok"}


@app.get("/status")
def status() -> dict[str, Union[str, bool, dict, list, None]]:
    """Readiness and diagnostics endpoint for frontend and troubleshooting.

    Includes:

    - runtime availability (Transformers/Torch/CUDA)
    - per-model cache readiness details
    - current auto-selection result and reason
    - currently loaded model key (if any)
    """

    cache_report = [inspect_model_cache("phi3"), inspect_model_cache("tinyllama")]
    return {
        "status": "ok",
        "backend": {
            "api": "ready",
            "modelRequiredDefault": True,
            "transformersRuntimeAvailable": AutoTokenizer is not None and AutoModelForCausalLM is not None and torch is not None,
            "cudaAvailable": bool(torch is not None and torch.cuda.is_available()),
            "loadedModelKey": _loaded_model_key,
        },
        "cache": cache_report,
        "selection": {
            "autoResult": resolve_requested_model("auto"),
            "autoReason": auto_selection_reason(),
            "phi3Result": resolve_requested_model("phi3"),
            "tinyllamaResult": resolve_requested_model("tinyllama"),
        },
    }


@app.post("/optimize", response_model=OptimizeResponse)
def optimize(payload: OptimizeRequest) -> OptimizeResponse:
    """Optimize code using selected backend strategy.

    Flow:

    1. Resolve model selection from request preference and cache readiness.
    2. If no model is available:
       - return 503 when ``requireModel`` is true
       - otherwise use heuristic fallback
    3. If model is available:
       - run model optimization
       - if model fails and ``requireModel`` is false, use heuristic fallback
    """

    reason = "Backend local optimizer (FastAPI)"
    if payload.performanceMode == "quality":
        reason = "Backend local optimizer (quality mode)"
    elif payload.performanceMode == "balanced":
        reason = "Backend local optimizer (balanced mode)"

    selected_model = resolve_requested_model(payload.preferredModel)

    if selected_model == "none":
        if payload.requireModel:
            raise HTTPException(
                status_code=503,
                detail=(
                    "Backend model-required mode is enabled, but no inference-ready cached model was found. "
                    "Run models:download and ensure a Transformers-compatible model is cached."
                ),
            )
        return run_local_optimization(
            payload.code,
            reason,
            "none",
            "No backend LLM inference active (heuristic optimizer only)",
        )

    try:
        return run_model_optimization(payload, selected_model)
    except Exception as exc:
        if payload.requireModel:
            raise HTTPException(
                status_code=500,
                detail=f"Backend model inference failed in model-required mode: {exc}",
            ) from exc

        return run_local_optimization(
            payload.code,
            f"{reason}; model inference failed, local fallback used",
            selected_model,
            f"Downloaded model selected but inference failed: {MODEL_KEYS[selected_model]}",
        )
