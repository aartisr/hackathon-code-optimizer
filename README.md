# Code Optimizer Studio

Code Optimizer Studio is a code optimization playground designed for classrooms, clubs, and quick demos. Users paste source code, choose optimization settings, and receive an improved version with source transparency and timing breakdown.

The app supports two inference paths:

- Browser inference (WebGPU + WebLLM)
- Python backend inference (FastAPI)

In backend mode, optimization now uses locally cached Transformer models when available.

## What This App Does

- Supports `Inference Engine` switching between Browser and Backend API.
- Runs model inference in browser using WebLLM when Browser mode is selected.
- Supports safe runtime modes (`Auto`, `TinyLlama only`, `Local only`) for stability.
- Supports performance modes (`Fast`, `Balanced`, `Quality`) for latency/quality tuning.
- Provides local heuristic optimization fallback when model/API output is unavailable.
- Shows transformed code, quality/impact scores, result source, and phase-by-phase timing.

## Model Strategy

The app uses two in-browser model targets:

- Primary: `Phi-3-mini-4k-instruct-q4f16_1-MLC`
- Fallback: `TinyLlama-1.1B-Chat-v0.4-q4f16_1-MLC`

Runtime behavior:

1. User can explicitly load the model.
2. On optimization, the app attempts model-backed inference.
3. If WebGPU or model loading fails, the app enters local review mode.
4. Local mode still returns optimizations and change notes so work continues.

## Project Structure

```text
.
├── .gitignore
├── backend
│  ├── app.py
│  ├── download_models.py
│  └── requirements.txt
├── index.html
├── README.md
├── package.json
└── src
   ├── main.js
   └── styles.css
```

### File Responsibilities

- `backend/app.py`: FastAPI backend API (`/health`, `/optimize`) for server-side optimization.
- `backend/download_models.py`: one-time model downloader/cacher for backend model artifacts.
- `backend/requirements.txt`: Python dependencies for backend and model download tooling.
- `index.html`: semantic UI structure and control elements.
- `src/styles.css`: visual system, responsive layout, and component styling.
- `src/main.js`: full runtime logic (model loading, prompt creation, optimization pipeline, UI rendering).

## Run Locally

```bash
npm run dev:web
```

Open:

```text
http://127.0.0.1:5179
```

## Backend Mode (Python FastAPI)

This project now supports a backend inference path to avoid browser memory limits.

### Start Backend API

```bash
python3 -m pip install -r backend/requirements.txt
npm run models:download
npm run dev:api
```

`npm run models:download` is a one-time setup step. It stores model artifacts under `backend/model-cache/` and skips re-download on future runs unless you force it.
These downloads are now Transformers-compatible model repos used directly by the backend inference runtime.

Optional targeted downloads:

```bash
npm run models:download:phi3
npm run models:download:tiny
```

The backend runs at:

```text
http://127.0.0.1:8000
```

### Start Frontend

```bash
npm run dev:web
```

### Use Backend In UI

Backend is now the default inference engine in the UI.

1. Set `Inference Engine` to `Backend API (Python)`.
2. Keep `Backend URL` as `http://127.0.0.1:8000` (or change it if hosted elsewhere).
3. Click `Optimize code`.

The app will call `POST /optimize` and display source/timing metadata from the backend response.

Important:

- The UI now includes an explicit `Model used` label for transparency.
- Backend can run in model-required mode via `Require backend model`.
- When `Require backend model` is enabled, backend will fail fast instead of using heuristic fallback.
- When `Require backend model` is disabled, backend may fall back to heuristic optimization if model inference fails.

## UI Controls Overview

- `Inference Engine`: choose Browser or Backend API execution.
- `Backend URL`: backend base URL used when Inference Engine is Backend.
- `Require backend model`: require true backend model inference (no heuristic fallback).
- `Runtime Mode`: browser path selection (auto / tiny-only / local-only).
- `Performance`: latency-vs-quality tradeoff.
- `Result source`: explicitly shows where output came from.
- `Response Time`: shows total plus phase timing (model load, generation, post-process, local optimize).

## Runtime Flow

1. App boots and caches element references.
2. Sample code is inserted into the input editor.
3. User selects inference, runtime, performance, language, and goal.
4. App runs optimization through selected engine (backend or browser).
5. App applies parsing/validation/fallback logic.
6. Parsed result updates:

- optimized code panel
- change list
- quality score
- impact score
- result source label
- timing breakdown

## Core Functions (`src/main.js`)

- `setModelState(kind, text, progress)`: synchronizes model status visuals.
- `updateInputStats()`: updates line/character metrics for input code.
- `parseModelResponse(text)`: parses strict JSON model output when possible.
- `fallbackParse(text)`: gracefully extracts useful output from free-form responses.
- `loadModel(force)`: handles WebGPU checks and model initialization/fallback sequence.
- `buildPrompt(code)`: composes deterministic optimization instructions.
- `runModelOptimization(code)`: executes model inference path, then parses output.
- `runLocalOptimization(code)`: applies heuristic regex transformations.
- `renderResults(result)`: writes transformed code and scoring to the UI.
- `optimizeCode()`: orchestration entry point for optimize button clicks.

## Exact Prompts Sent To Models

The app currently sends one of the following exact prompt templates.

### Standard Prompt (`buildPrompt`)

```text
You are a senior code optimizer. Optimize the user's code while preserving behavior.

Language: ${els.languageSelect.value}
Goal: ${els.goalSelect.value}
Preserve behavior strictly: ${els.strictMode.checked ? "yes" : "no"}

CRITICAL OUTPUT FORMAT RULES:
- Return ONLY a valid JSON object.
- No markdown fences.
- No preface or explanation.
- No extra keys.

Return exactly this JSON shape:
{
  "optimizedCode": "the complete optimized code"
}

Code:
\`\`\`
${code}
\`\`\`
```

### TinyLlama Prompt (`buildTinyPrompt`)

```text
Optimize this code.

Requirements:
- Preserve behavior.
- Keep the same language.
- Improve readability and basic efficiency.
- Do not explain anything.
- Return JSON only.

Output format (exact):
{"optimizedCode":"..."}

Input code:
${code}
```

### Correction Prompt (`requestStrictJsonCorrection`)

This is used only if the first model response is not strict JSON.

```text
Rewrite your previous answer into strict JSON only.

Rules:
- Output must be valid JSON object.
- Output must contain only this key: optimizedCode.
- No markdown.
- No explanation.

Source code:
\`\`\`
${code}
\`\`\`

Previous answer:
${priorResponse}
```

## CSS Architecture Map (`src/styles.css`)

Use this section to quickly find where each UI area is styled.

- App shell and global page frame:
  - `.app-shell`
  - `body`
- Hero area (title + model card):
  - `.hero`
  - `.hero-copy`
  - `.model-card`
  - `.status-row`, `.status-dot`, `.meter`
- Control strip (language/goal/toggles/actions):
  - `.control-strip`
  - `label`, `select`, `.toggle`
  - `.primary`, `.secondary`, `.icon-button`
- Code workspace (before/after editors):
  - `.workspace`
  - `.editor-panel`
  - `textarea`, `pre`
- Insight cards (scores + change notes):
  - `.insights-grid`
  - `.score-panel`
  - `.notes-panel`
- Shared card surface style:
  - `.model-card`, `.control-strip`, `.editor-panel`, `.score-panel`, `.notes-panel`
- Responsive behavior:
  - `@media (max-width: 980px)` for tablet stacking
  - `@media (max-width: 620px)` for mobile spacing and type scaling

Styling conventions used in this file:

- Design tokens are centralized in `:root`.
- Layout containers use CSS Grid for main composition.
- Interactive state changes are intentionally subtle (`transform`, `opacity`, `transition`).
- Code surfaces share mono typography to keep before/after diffs visually comparable.

## Local Heuristic Rules

When model mode is unavailable, local mode currently applies conservative transforms:

- `var` to `let` replacement.
- Loose equality (`==`, `!=`) to strict equality (`===`, `!==`).
- `x = x + y` to compound assignment (`x += y`).
- `if (expr === true)` simplification to `if (expr)`.
- Advisory note when index-based loops are detected.

These are intentionally simple and may not cover edge cases in every language.

## Browser Requirements

- Modern browser environment.
- WebGPU support for model-backed inference.
- Clipboard API support for copy action.

Without WebGPU, the app remains functional in local review mode.

## Troubleshooting

### Browser shows insufficient memory

Recommended order:

1. Switch `Inference Engine` to `Backend API (Python)`.
2. If staying in browser mode, use `Runtime Mode: TinyLlama only`.
3. Use `Performance: Fast` for lower token budgets.
4. Use `Runtime Mode: Local review only` for guaranteed no-model path.

### Status stays in local review mode

Possible causes:

- Browser does not support WebGPU.
- Model download/init failed.
- Device memory constraints during model initialization.

Recommended checks:

1. Use a Chromium-based browser with WebGPU enabled.
2. Retry with the "Load model" button.
3. Keep local mode for lightweight analysis if model loading fails.

### Empty or malformed model response

`parseModelResponse` attempts strict JSON first and then falls back to fenced/free-form parsing. The app should still produce readable output and actionable change notes.

## Development Notes

- The project is intentionally dependency-light and static-host friendly.
- There is no build step; source files are loaded directly by the browser.
- Keep runtime behavior deterministic when adding new heuristic rules.
- Prefer additive UI status updates so users understand when fallback mode is active.

## Possible Extensions

- Add syntax-aware parsing per language before local transforms.
- Persist last input/output in localStorage.
- Add downloadable diff view.
- Add tests for parser and local optimization rules.
