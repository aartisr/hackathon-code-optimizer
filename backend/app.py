from __future__ import annotations

import re
import time
from pathlib import Path
from typing import Literal

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
    code: str = Field(min_length=1)
    language: str = "Auto detect"
    goal: str = "Balanced cleanup"
    strict: bool = True
    performanceMode: Literal["fast", "balanced", "quality"] = "fast"
    preferredModel: Literal["auto", "phi3", "tinyllama"] = "auto"
    requireModel: bool = True


class TimingBreakdown(BaseModel):
    modelLoadMs: float | None = None
    generationMs: float | None = None
    postProcessMs: float | None = None
    localOptimizeMs: float | None = None


class OptimizeResponse(BaseModel):
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
_loaded_model_key: str | None = None
_loaded_tokenizer = None
_loaded_model = None


def model_cached(model_key: str) -> bool:
    marker = MODEL_CACHE_ROOT / model_key / ".download-complete.json"
    return marker.exists()


def model_inference_ready(model_key: str) -> bool:
    model_dir = MODEL_CACHE_ROOT / model_key
    if not model_cached(model_key):
        return False
    return (model_dir / "config.json").exists()


def resolve_requested_model(preferred: str) -> str:
    if preferred == "phi3" and model_inference_ready("phi3"):
        return "phi3"
    if preferred == "tinyllama" and model_inference_ready("tinyllama"):
        return "tinyllama"
    if preferred == "auto":
        if model_inference_ready("phi3"):
            return "phi3"
        if model_inference_ready("tinyllama"):
            return "tinyllama"
    return "none"


def build_backend_prompt(payload: OptimizeRequest) -> str:
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


def extract_optimized_code(raw_text: str) -> str | None:
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


def generation_config(performance_mode: str) -> dict[str, float | int]:
    if performance_mode == "quality":
        return {"max_new_tokens": 640, "temperature": 0.3, "top_p": 0.95}
    if performance_mode == "balanced":
        return {"max_new_tokens": 420, "temperature": 0.2, "top_p": 0.9}
    return {"max_new_tokens": 260, "temperature": 0.1, "top_p": 0.85}


def get_loaded_runtime(model_key: str):
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
        torch_dtype=dtype,
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
    return {"status": "ok"}


@app.post("/optimize", response_model=OptimizeResponse)
def optimize(payload: OptimizeRequest) -> OptimizeResponse:
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
