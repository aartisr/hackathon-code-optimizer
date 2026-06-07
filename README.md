# Code Optimizer Studio

Code Optimizer Studio is a browser-only code optimization playground designed for classrooms, clubs, and quick local demos. Users paste source code, choose optimization goals, and receive an improved version plus a short change summary.

No backend service is required.

## What This App Does

- Runs model inference directly in the browser using WebLLM when supported.
- Attempts a primary model first, then automatically falls back to a smaller model.
- Provides an always-available local heuristic optimizer when model inference is unavailable.
- Shows both transformed code and user-friendly quality/impact scoring.

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
├── index.html
├── README.md
├── package.json
└── src
   ├── main.js
   └── styles.css
```

### File Responsibilities

- `index.html`: semantic UI structure and control elements.
- `src/styles.css`: visual system, responsive layout, and component styling.
- `src/main.js`: full runtime logic (model loading, prompt creation, optimization pipeline, UI rendering).

## Run Locally

```bash
npm run dev
```

Open:

```text
http://127.0.0.1:5179
```

## Runtime Flow

1. App boots and caches element references.
2. Sample code is inserted into the input editor.
3. User selects language/goal and triggers optimization.
4. App attempts model run (or local fallback).
5. Parsed result updates:
   - optimized code panel
   - change list
   - quality score
   - impact score

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
