from __future__ import annotations

import re
import time
from typing import Literal

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field


class OptimizeRequest(BaseModel):
    code: str = Field(min_length=1)
    language: str = "Auto detect"
    goal: str = "Balanced cleanup"
    strict: bool = True
    performanceMode: Literal["fast", "balanced", "quality"] = "fast"


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
    timings: TimingBreakdown


app = FastAPI(title="Code Optimizer Backend", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


def run_local_optimization(code: str, reason: str) -> OptimizeResponse:
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

    return run_local_optimization(payload.code, reason)
