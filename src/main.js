/**
 * Browser model identifiers used by WebLLM.
 * Phi-3 is attempted first, then TinyLlama as an automatic fallback.
 */
const PHI_MODEL = "Phi-3-mini-4k-instruct-q4f16_1-MLC";
const TINY_MODEL = "TinyLlama-1.1B-Chat-v0.4-q4f16_1-MLC";

/**
 * Cached DOM references used throughout the app.
 * Keeping these in one object avoids repeated querySelector calls.
 */
const els = {
  activeModel: document.querySelector("#activeModel"),
  changeList: document.querySelector("#changeList"),
  copyButton: document.querySelector("#copyButton"),
  engineBadge: document.querySelector("#engineBadge"),
  goalSelect: document.querySelector("#goalSelect"),
  impactLabel: document.querySelector("#impactLabel"),
  impactScore: document.querySelector("#impactScore"),
  inputCode: document.querySelector("#inputCode"),
  inputStats: document.querySelector("#inputStats"),
  inferenceMode: document.querySelector("#inferenceMode"),
  languageSelect: document.querySelector("#languageSelect"),
  loadModelButton: document.querySelector("#loadModelButton"),
  modelDot: document.querySelector("#modelDot"),
  modelProgress: document.querySelector("#modelProgress"),
  modelStatus: document.querySelector("#modelStatus"),
  optimizeButton: document.querySelector("#optimizeButton"),
  outputCode: document.querySelector("#outputCode code"),
  performanceMode: document.querySelector("#performanceMode"),
  qualityLabel: document.querySelector("#qualityLabel"),
  qualityScore: document.querySelector("#qualityScore"),
  responseTime: document.querySelector("#responseTime"),
  responseTimeLabel: document.querySelector("#responseTimeLabel"),
  backendUrl: document.querySelector("#backendUrl"),
  backendModel: document.querySelector("#backendModel"),
  requireBackendModel: document.querySelector("#requireBackendModel"),
  phaseTotal: document.querySelector("#phaseTotal"),
  phaseModelLoad: document.querySelector("#phaseModelLoad"),
  phaseGeneration: document.querySelector("#phaseGeneration"),
  phasePostProcess: document.querySelector("#phasePostProcess"),
  phaseLocal: document.querySelector("#phaseLocal"),
  modeSummaryTitle: document.querySelector("#modeSummaryTitle"),
  modeSummaryText: document.querySelector("#modeSummaryText"),
  modelUsed: document.querySelector("#modelUsed"),
  resultSource: document.querySelector("#resultSource"),
  runtimeMode: document.querySelector("#runtimeMode"),
  safeModeBanner: document.querySelector("#safeModeBanner"),
  enableSafeModeButton: document.querySelector("#enableSafeModeButton"),
  sampleButton: document.querySelector("#sampleButton"),
  sampleSelect: document.querySelector("#sampleSelect"),
  strictMode: document.querySelector("#strictMode")
};

/** @type {any | null} WebLLM engine instance once loaded. */
let engine = null;
/** @type {string} Current model id being used by the engine. */
let activeModelId = PHI_MODEL;
/** @type {"model" | "local"} Whether responses come from model inference or local heuristics. */
let engineMode = "local";
/** @type {boolean} Prevents repeated auto-load attempts unless the user forces reload. */
let modelAttempted = false;

/**
 * Library of intentionally non-optimized examples users can load and optimize.
 */
const sampleLibrary = [
  {
    language: "JavaScript",
    code: `function calculateTotal(items) {
  var total = 0;
  for (var i = 0; i < items.length; i++) {
    if (items[i].active == true) {
      total = total + items[i].price * items[i].quantity;
    }
  }
  return total;
}

console.log(calculateTotal(cartItems));`
  },
  {
    language: "JavaScript",
    code: `function cleanup(values) {
  var out = [];
  for (var i = 0; i < values.length; i++) {
    if (values[i] != null) {
      if (out.indexOf(values[i]) == -1) {
        out.push(values[i]);
      }
    }
  }
  return out;
}`
  },
  {
    language: "JavaScript",
    code: `function getOpenTickets(tickets) {
  var result = [];
  for (var i = 0; i < tickets.length; i++) {
    if (tickets[i].closed == false) {
      if (tickets[i].priority == "high" || tickets[i].priority == "urgent") {
        result.push(tickets[i]);
      }
    }
  }
  return result;
}`
  },
  {
    language: "TypeScript",
    code: `type ScoreRow = { name: string; score: number; active: boolean };

function summarize(rows: ScoreRow[]) {
  let total = 0;
  let count = 0;
  for (let i = 0; i < rows.length; i++) {
    if (rows[i].active === true) {
      total = total + rows[i].score;
      count = count + 1;
    }
  }
  const average = count == 0 ? 0 : total / count;
  return { total, count, average };
}`
  },
  {
    language: "Python",
    code: `def summarize_orders(orders):
    total = 0
    high_value = []
    i = 0
    while i < len(orders):
        if orders[i]["paid"] == True:
            total = total + orders[i]["amount"]
            if orders[i]["amount"] > 100:
                high_value.append(orders[i])
        i = i + 1
    return {"total": total, "high_value": high_value}`
  },
  {
    language: "Python",
    code: `def active_user_names(users):
    names = []
    for i in range(0, len(users)):
        if users[i]["disabled"] == False:
            if users[i]["name"] != "":
                names.append(users[i]["name"].strip())
    return names`
  },
  {
    language: "Java",
    code: `public static double invoiceTotal(List<Item> items) {
    double total = 0;
    for (int i = 0; i < items.size(); i++) {
      Item item = items.get(i);
      if (item.isEnabled() == true) {
        total = total + item.getPrice() * item.getQuantity();
      }
    }
    return total;
  }`
  },
  {
    language: "C++",
    code: `int countPositive(const std::vector<int>& nums) {
    int total = 0;
    for (int i = 0; i < nums.size(); i++) {
      if (nums[i] > 0) {
        total = total + 1;
      }
    }
    return total;
  }`
  },
  {
    language: "HTML/CSS",
    code: `<div class="card" style="padding: 16px; border: 1px solid #ddd; border-radius: 6px;">
  <h3 style="margin: 0; color: #333333;">Team Member</h3>
  <p style="margin-top: 12px; margin-bottom: 12px; color: #333333;">Frontend Engineer</p>
  <button style="background: #0a66ff; color: white; border: 0; border-radius: 4px; padding: 8px 12px;">View Profile</button>
</div>`
  },
  {
    language: "JavaScript",
    code: `function searchUsers(users, query) {
  var matches = [];
  for (var i = 0; i < users.length; i++) {
    if (users[i].name.toLowerCase().indexOf(query.toLowerCase()) != -1) {
      if (users[i].active == true) {
        matches.push(users[i]);
      }
    }
  }
  return matches;
}`
  }
];

/**
 * Loads one sample snippet from the library into the editor and syncs controls.
 *
 * @param {number} index Sample index in sampleLibrary.
 */
function loadSample(index) {
  const safeIndex = Number.isInteger(index) ? Math.max(0, Math.min(sampleLibrary.length - 1, index)) : 0;
  const sample = sampleLibrary[safeIndex];

  if (!sample) return;

  els.inputCode.value = sample.code;
  els.sampleSelect.value = String(safeIndex);

  if ([...els.languageSelect.options].some((option) => option.textContent === sample.language)) {
    els.languageSelect.value = sample.language;
  }

  updateInputStats();
  els.inputCode.focus();
}

/**
 * Updates the model status indicators shown in the hero card.
 *
 * @param {"loading" | "ready" | "error"} kind Visual state for the status dot.
 * @param {string} text Human-readable status message.
 * @param {number | null} [progress=null] Optional 0-100 progress value for the meter.
 */
function setModelState(kind, text, progress = null) {
  els.modelDot.className = `status-dot ${kind === "ready" ? "ready" : kind === "error" ? "error" : ""}`;
  els.modelStatus.textContent = text;
  if (progress !== null) {
    els.modelProgress.style.width = `${Math.max(0, Math.min(100, progress))}%`;
  }
}

/**
 * Recomputes and renders line/character stats for the input editor.
 */
function updateInputStats() {
  const lines = els.inputCode.value ? els.inputCode.value.split("\n").length : 0;
  const chars = els.inputCode.value.length;
  els.inputStats.textContent = `${lines} line${lines === 1 ? "" : "s"} · ${chars} chars`;
}

/**
 * Checks whether text is valid JSON containing only optimizedCode.
 *
 * @param {string} text
 * @returns {boolean}
 */
function isStrictOptimizedCodeJson(text) {
  const trimmed = (text || "").trim();
  if (!trimmed.startsWith("{") || !trimmed.endsWith("}")) return false;

  try {
    const parsed = JSON.parse(trimmed);
    const keys = Object.keys(parsed);
    return keys.length === 1 && keys[0] === "optimizedCode" && typeof parsed.optimizedCode === "string";
  } catch {
    return false;
  }
}

/**
 * Heuristic check that distinguishes likely code from prose-only model output.
 *
 * @param {string} text Candidate optimized code text.
 * @returns {boolean}
 */
function looksLikeCode(text) {
  const candidate = (text || "").trim();
  if (!candidate) return false;

  const prosePattern = /(as a senior code optimizer|i propose the following steps|by following these steps)/i;
  if (prosePattern.test(candidate)) return false;

  const codeSignalPattern = /[{}();=]|\b(function|const|let|var|def|class|return|if|for|while|public|static|import)\b/;
  return codeSignalPattern.test(candidate);
}

/**
 * Uses local optimization when model output is not valid/usable code.
 *
 * @param {string} sourceCode Original user input.
 * @param {string} reason Message describing why fallback was applied.
 * @returns {{ optimizedCode: string, changes: string[], qualityScore: number, impactScore: number }}
 */
function localFallbackForInvalidModelOutput(sourceCode, reason) {
  const localResult = runLocalOptimization(sourceCode);
  return {
    ...localResult,
    source: "local-fallback",
    sourceLabel: `Local fallback: ${reason}`,
    changes: [
      reason,
      ...localResult.changes
    ]
  };
}

/**
 * Attempts to parse model output into the expected JSON result contract.
 * Falls back to free-form parsing if strict JSON is not returned.
 *
 * @param {string} text Raw text returned by the model.
 * @returns {{ optimizedCode: string, changes: string[], qualityScore: number, impactScore: number }}
 */
function parseModelResponse(text, sourceCode) {
  const clean = text.trim();
  const jsonStart = clean.indexOf("{");
  const jsonEnd = clean.lastIndexOf("}");

  if (jsonStart >= 0 && jsonEnd > jsonStart) {
    try {
      const parsed = JSON.parse(clean.slice(jsonStart, jsonEnd + 1));
      const modelChanges = Array.isArray(parsed.changes) ? parsed.changes : [];
      const modelResult = {
        optimizedCode: parsed.optimizedCode || parsed.code || clean,
        changes: modelChanges.length ? modelChanges : ["Applied model optimization."],
        qualityScore: Number(parsed.qualityScore) || 82,
        impactScore: Number(parsed.impactScore) || 68,
        source: "model",
        sourceLabel: "Model output (strict JSON)",
        modelUsed: activeModelId,
        modelUsedLabel: `Browser model: ${activeModelId.includes("Phi") ? "Phi-3 Mini" : "TinyLlama"}`
      };

      if (!looksLikeCode(modelResult.optimizedCode)) {
        return localFallbackForInvalidModelOutput(
          sourceCode,
          "Model returned planning prose instead of executable code. Applied local optimization fallback."
        );
      }

      return modelResult;
    } catch {
      return fallbackParse(clean, sourceCode);
    }
  }

  return fallbackParse(clean, sourceCode);
}

/**
 * Best-effort parser for non-JSON model output.
 * Supports fenced code blocks and provides default score/change metadata.
 *
 * @param {string} text Free-form model output.
 * @returns {{ optimizedCode: string, changes: string[], qualityScore: number, impactScore: number }}
 */
function fallbackParse(text, sourceCode) {
  const codeMatch = text.match(/```(?:\w+)?\n([\s\S]*?)```/);
  const extractedCode = codeMatch ? codeMatch[1].trim() : text;

  if (!looksLikeCode(extractedCode)) {
    return localFallbackForInvalidModelOutput(
      sourceCode,
      "Model response was not in a usable code format. Applied local optimization fallback."
    );
  }

  return {
    optimizedCode: extractedCode,
    changes: [
      "The model returned a free-form optimization. Review the diff before using it.",
      "Formatting and naming may have been adjusted for readability."
    ],
    qualityScore: 78,
    impactScore: 58,
    source: "model-freeform",
    sourceLabel: "Model output (free-form parse)",
    modelUsed: activeModelId,
    modelUsedLabel: `Browser model: ${activeModelId.includes("Phi") ? "Phi-3 Mini" : "TinyLlama"}`
  };
}

/**
 * Creates a short, user-facing explanation for model loading failures.
 *
 * @param {unknown} error Original exception thrown by WebLLM/model fetch.
 * @returns {string}
 */
function describeModelLoadError(error) {
  const message = error instanceof Error ? error.message : String(error || "");
  const normalized = message.toLowerCase();

  if (normalized.includes("403") || normalized.includes("forbidden") || normalized.includes("huggingface.co")) {
    return "Model files blocked by network policy (HTTP 403 from Hugging Face).";
  }

  if (normalized.includes("failed to fetch") || normalized.includes("network")) {
    return "Model download failed due to a network connectivity issue.";
  }

  if (normalized.includes("out of memory") || normalized.includes("memory")) {
    return "Model failed to load due to device memory constraints.";
  }

  return "Model initialization failed in this browser session.";
}

/**
 * Estimates whether the current device likely has constrained memory for Phi-3.
 *
 * @returns {boolean}
 */
function isLikelyLowMemoryDevice() {
  const memoryGiB = Number(navigator.deviceMemory || 0);
  return Number.isFinite(memoryGiB) && memoryGiB > 0 && memoryGiB < 8;
}

/**
 * Shows or hides the one-click Safe Mode recommendation banner.
 */
function syncSafeModeBanner() {
  const shouldShow = isLikelyLowMemoryDevice() && els.runtimeMode.value === "auto" && els.inferenceMode.value === "browser";
  if (shouldShow) {
    els.safeModeBanner.hidden = false;
    return;
  }

  els.safeModeBanner.hidden = true;
}

/**
 * Syncs engine controls when switching between browser and backend inference.
 */
function syncInferenceModeUI() {
  const backendMode = els.inferenceMode.value === "backend";
  els.loadModelButton.disabled = backendMode;
  setControlAvailability(els.backendUrl, backendMode);
  setControlAvailability(els.backendModel, backendMode);
  setControlAvailability(els.requireBackendModel, backendMode);
  setControlAvailability(els.runtimeMode, !backendMode);

  if (backendMode) {
    setModelState("ready", "Backend API mode enabled", 100);
    els.engineBadge.textContent = "Backend API";
    els.activeModel.textContent = "Source: Python backend";
    els.loadModelButton.textContent = "Load model";
  } else {
    els.engineBadge.textContent = "Browser runtime";
    els.loadModelButton.textContent = "Load browser model";
    if (!engine) {
      setModelState("loading", "Browser model idle", 0);
      els.activeModel.textContent = "Primary: Phi-3 Mini";
    }
  }

  syncSafeModeBanner();
  updateModeSummary();
}

/**
 * Enables/disables a control and visually mutes its label when not applicable.
 *
 * @param {HTMLElement | null} control
 * @param {boolean} enabled
 */
function setControlAvailability(control, enabled) {
  if (!control) return;
  control.disabled = !enabled;
  const label = control.closest("label");
  if (!label) return;
  label.classList.toggle("muted-control", !enabled);
}

/**
 * Explains the currently selected execution strategy in plain language.
 */
function updateModeSummary() {
  const backendMode = els.inferenceMode.value === "backend";
  if (backendMode) {
    const selected = els.backendModel.value;
    const requireModel = Boolean(els.requireBackendModel.checked);
    const selectedLabel = selected === "phi3" ? "Phi-3" : selected === "tinyllama" ? "TinyLlama" : "Auto model";

    els.modeSummaryTitle.textContent = `Backend API · ${selectedLabel}`;
    els.modeSummaryText.textContent = requireModel
      ? "Strict path enabled: optimization must come from a backend model. If model inference is unavailable, the request fails clearly."
      : "Resilient path enabled: backend attempts model inference first, then falls back to heuristic optimization if needed.";
    return;
  }

  const runtime = els.runtimeMode.value;
  if (runtime === "local-only") {
    els.modeSummaryTitle.textContent = "Browser · Local-only review";
    els.modeSummaryText.textContent = "No model inference is used. Optimizations come from local heuristic rules for maximum reliability.";
    return;
  }

  if (runtime === "tiny-only") {
    els.modeSummaryTitle.textContent = "Browser · TinyLlama only";
    els.modeSummaryText.textContent = "Uses TinyLlama in-browser for lower memory usage and better stability on constrained devices.";
    return;
  }

  els.modeSummaryTitle.textContent = "Browser · Auto runtime";
  els.modeSummaryText.textContent = "Attempts Phi-3 first, then falls back to TinyLlama if needed. Best quality when enough device memory is available.";
}

/**
 * Executes optimization through backend API.
 *
 * @param {string} code
 * @returns {Promise<{ optimizedCode: string, changes: string[], qualityScore: number, impactScore: number, source: string, sourceLabel: string, timings: { modelLoadMs?: number, generationMs?: number, postProcessMs?: number, localOptimizeMs?: number } }>}
 */
async function runBackendOptimization(code) {
  const backendBaseUrl = (els.backendUrl.value || "http://127.0.0.1:8000").replace(/\/$/, "");
  const requestStartedAt = performance.now();

  const response = await fetch(`${backendBaseUrl}/optimize`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json"
    },
    body: JSON.stringify({
      code,
      language: els.languageSelect.value,
      goal: els.goalSelect.value,
      strict: els.strictMode.checked,
      performanceMode: els.performanceMode.value,
      preferredModel: els.backendModel?.value || "auto",
      requireModel: Boolean(els.requireBackendModel?.checked)
    })
  });

  if (!response.ok) {
    let detail = "";
    try {
      const errorPayload = await response.json();
      detail = errorPayload?.detail ? ` ${errorPayload.detail}` : "";
    } catch {
      detail = "";
    }
    throw new Error(`Backend optimization failed (${response.status}).${detail}`);
  }

  const payload = await response.json();
  const roundTripMs = Math.max(0, performance.now() - requestStartedAt);
  return {
    optimizedCode: payload.optimizedCode || code,
    changes: Array.isArray(payload.changes) ? payload.changes : ["Backend optimization completed."],
    qualityScore: Number(payload.qualityScore) || 75,
    impactScore: Number(payload.impactScore) || 55,
    source: payload.source || "backend",
    sourceLabel: payload.sourceLabel || "Backend API optimization",
    modelUsed: payload.modelUsed || "none",
    modelUsedLabel: payload.modelUsedLabel || "No model info from backend",
    timings: {
      modelLoadMs: Number(payload?.timings?.modelLoadMs) || 0,
      generationMs: Number(payload?.timings?.generationMs) || roundTripMs,
      postProcessMs: Number(payload?.timings?.postProcessMs) || 0,
      localOptimizeMs: Number(payload?.timings?.localOptimizeMs) || 0
    }
  };
}

/**
 * Attempts to create one model engine and updates progress/status UI.
 *
 * @param {string} modelId WebLLM model identifier.
 * @param {number} fallbackProgress Progress fallback used when report is missing.
 * @param {string} loadingText Loading message while downloading model artifacts.
 * @param {string} activeLabel Label shown in the model badge on success.
 * @returns {Promise<any>}
 */
async function tryLoadModelCandidate(modelId, fallbackProgress, loadingText, activeLabel) {
  const webllm = await import("https://esm.run/@mlc-ai/web-llm");
  const createEngine = webllm.CreateMLCEngine || webllm.CreateWebWorkerMLCEngine;
  const loadedEngine = await createEngine(modelId, {
    initProgressCallback: (report) => {
      const progress = report.progress ? Math.round(report.progress * 100) : fallbackProgress;
      setModelState("loading", report.text || loadingText, progress);
    }
  });

  engine = loadedEngine;
  activeModelId = modelId;
  engineMode = "model";
  els.activeModel.textContent = activeLabel;
  els.engineBadge.textContent = modelId.includes("Phi") ? "Phi-3 Mini" : "TinyLlama";
  setModelState("ready", `${modelId.includes("Phi") ? "Phi-3 Mini" : "TinyLlama"} ready`, 100);
  return loadedEngine;
}

/**
 * Loads a browser-local model engine if supported.
 *
 * Flow:
 * 1) Skip if already loaded.
 * 2) Abort to local mode if WebGPU is unavailable.
 * 3) Attempt Phi-3, then fallback to TinyLlama.
 * 4) Keep UI status in sync throughout loading.
 *
 * @param {boolean} [force=false] When true, allows a fresh attempt after a previous failure.
 * @returns {Promise<any | null>} Active engine when available; otherwise null for local mode.
 */
async function loadModel(force = false) {
  const runtimeMode = els.runtimeMode.value;
  const perfMode = els.performanceMode.value;
  if (runtimeMode === "local-only") {
    engineMode = "local";
    els.engineBadge.textContent = "Local review mode";
    setModelState("error", "Local-only mode enabled. Browser model loading is disabled.", 0);
    return null;
  }

  if (engine) return engine;
  if (modelAttempted && !force) return null;
  modelAttempted = true;

  if (!("gpu" in navigator)) {
    setModelState("error", "WebGPU unavailable. Local review mode active.", 0);
    els.engineBadge.textContent = "Local review mode";
    return null;
  }

  els.loadModelButton.disabled = true;
  const lowMemoryMode = runtimeMode === "auto" && isLikelyLowMemoryDevice();
  const fastModeAuto = runtimeMode === "auto" && perfMode === "fast";
  const modelPlan =
    runtimeMode === "tiny-only"
      ? [
          {
            id: TINY_MODEL,
            progress: 30,
            loadingText: "Downloading TinyLlama...",
            activeLabel: "Active: TinyLlama (safe mode)"
          }
        ]
      : fastModeAuto
        ? [
            {
              id: TINY_MODEL,
              progress: 30,
              loadingText: "Downloading TinyLlama (fast mode)...",
              activeLabel: "Active: TinyLlama (fast mode)"
            }
          ]
        : lowMemoryMode
        ? [
            {
              id: TINY_MODEL,
              progress: 30,
              loadingText: "Downloading TinyLlama (safe mode)...",
              activeLabel: "Active: TinyLlama (auto safe mode)"
            },
            {
              id: PHI_MODEL,
              progress: 18,
              loadingText: "Downloading Phi-3 Mini...",
              activeLabel: "Active: Phi-3 Mini"
            }
          ]
        : [
            {
              id: PHI_MODEL,
              progress: 18,
              loadingText: "Downloading Phi-3 Mini...",
              activeLabel: "Active: Phi-3 Mini"
            },
            {
              id: TINY_MODEL,
              progress: 30,
              loadingText: "Downloading TinyLlama...",
              activeLabel: "Active: TinyLlama fallback"
            }
          ];

  if (fastModeAuto) {
    setModelState("loading", "Fast mode enabled. Loading TinyLlama for lower latency...", 8);
  } else if (lowMemoryMode) {
    setModelState("loading", "Low memory detected. Trying TinyLlama first...", 8);
  } else if (runtimeMode === "tiny-only") {
    setModelState("loading", "Tiny-only mode enabled. Loading TinyLlama...", 8);
  } else {
    setModelState("loading", "Loading Phi-3 Mini in browser...", 8);
  }

  const failureReasons = [];

  try {
    for (let i = 0; i < modelPlan.length; i++) {
      const candidate = modelPlan[i];

      try {
        if (i > 0) {
          setModelState("loading", `${modelPlan[i - 1].id.includes("Phi") ? "Phi-3" : "TinyLlama"} failed. Trying ${candidate.id.includes("Phi") ? "Phi-3" : "TinyLlama"}...`, 25);
        }
        return await tryLoadModelCandidate(candidate.id, candidate.progress, candidate.loadingText, candidate.activeLabel);
      } catch (candidateError) {
        const reason = describeModelLoadError(candidateError);
        failureReasons.push(reason);
        console.error(`${candidate.id} model load failed:`, candidateError);
      }
    }

    engineMode = "local";
    els.engineBadge.textContent = "Local review mode";
    setModelState("error", `${failureReasons[0] || "Model unavailable."} Local review mode active.`, 0);
    els.changeList.innerHTML = [
      ...failureReasons.map((reason) => `<li>${reason}</li>`),
      "<li>Using local review mode until model access is restored.</li>"
    ].join("");
    return null;
  } finally {
    els.loadModelButton.disabled = false;
  }
}

/**
 * Builds a deterministic instruction prompt for the model run.
 *
 * @param {string} code User-provided source code.
 * @returns {string} Prompt containing goals, constraints, and response schema.
 */
function buildPrompt(code) {
  return `You are a senior code optimizer. Optimize the user's code while preserving behavior.

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
\`\`\``;
}

/**
 * TinyLlama-focused prompt: shorter, rule-driven, and easier to follow.
 *
 * @param {string} code User-provided source code.
 * @returns {string}
 */
function buildTinyPrompt(code) {
  return `Optimize this code.

Requirements:
- Preserve behavior.
- Keep the same language.
- Improve readability and basic efficiency.
- Do not explain anything.
- Return JSON only.

Output format (exact):
{"optimizedCode":"..."}

Input code:
${code}`;
}

/**
 * Returns model-tuned generation settings.
 * TinyLlama benefits from shorter outputs and slightly non-zero temperature.
 *
 * @returns {{ temperature: number, max_tokens: number, top_p?: number }}
 */
function getGenerationSettings() {
  const perfMode = els.performanceMode.value;

  if (activeModelId === TINY_MODEL) {
    if (perfMode === "fast") {
      return {
        temperature: 0,
        max_tokens: 380,
        top_p: 0.85
      };
    }

    if (perfMode === "quality") {
      return {
        temperature: 0.2,
        max_tokens: 1500,
        top_p: 0.95
      };
    }

    return {
      temperature: 0.15,
      max_tokens: 1200,
      top_p: 0.9
    };
  }

  if (perfMode === "fast") {
    return {
      temperature: 0,
      max_tokens: 520,
      top_p: 0.9
    };
  }

  if (perfMode === "quality") {
    return {
      temperature: 0,
      max_tokens: 2200
    };
  }

  return {
    temperature: 0,
    max_tokens: 1800
  };
}

/**
 * Requests a strict JSON-only correction when the first model reply drifts.
 *
 * @param {any} loadedEngine
 * @param {string} code
 * @param {string} priorResponse
 * @returns {Promise<string>}
 */
async function requestStrictJsonCorrection(loadedEngine, code, priorResponse) {
  const correction = await loadedEngine.chat.completions.create({
    messages: [
      { role: "system", content: "You convert drafts into strict JSON output only." },
      {
        role: "user",
        content: `Rewrite your previous answer into strict JSON only.

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
${priorResponse}`
      }
    ],
    temperature: 0,
    max_tokens: 1800
  });

  return correction.choices?.[0]?.message?.content || "";
}

/**
 * Runs model-based optimization when possible, otherwise falls back to local analysis.
 *
 * @param {string} code User-provided source code.
 * @returns {Promise<{ optimizedCode: string, changes: string[], qualityScore: number, impactScore: number }>}
 */
async function runModelOptimization(code) {
  const perfMode = els.performanceMode.value;
  if (perfMode === "fast" && code.length > 2400) {
    return runLocalOptimization(code, "Fast mode used local optimization for large input.");
  }

  const modelLoadStartedAt = performance.now();
  const loadedEngine = await loadModel();
  const modelLoadMs = Math.max(0, performance.now() - modelLoadStartedAt);
  if (!loadedEngine) {
    const fallbackResult = runLocalOptimization(code, "Model unavailable in current runtime mode.");
    fallbackResult.timings = {
      ...(fallbackResult.timings || {}),
      modelLoadMs,
      generationMs: 0,
      postProcessMs: 0
    };
    return fallbackResult;
  }

  const generation = getGenerationSettings();
  const prompt = activeModelId === TINY_MODEL || perfMode === "fast" ? buildTinyPrompt(code) : buildPrompt(code);

  setModelState("loading", `Optimizing with ${activeModelId.includes("Phi") ? "Phi-3 Mini" : "TinyLlama"}...`, 100);
  const generationStartedAt = performance.now();
  const response = await loadedEngine.chat.completions.create({
    messages: [
      { role: "system", content: "You optimize code and return strict JSON only." },
      { role: "user", content: prompt }
    ],
    ...generation
  });

  let modelText = response.choices?.[0]?.message?.content || "";
  let usedCorrection = false;
  if (!isStrictOptimizedCodeJson(modelText) && perfMode !== "fast") {
    usedCorrection = true;
    modelText = await requestStrictJsonCorrection(loadedEngine, code, modelText);
  }
  const generationMs = Math.max(0, performance.now() - generationStartedAt);

  setModelState("ready", `${activeModelId.includes("Phi") ? "Phi-3 Mini" : "TinyLlama"} ready`, 100);
  const postProcessStartedAt = performance.now();
  const result = parseModelResponse(modelText, code);
  const postProcessMs = Math.max(0, performance.now() - postProcessStartedAt);
  result.timings = {
    ...(result.timings || {}),
    modelLoadMs,
    generationMs,
    postProcessMs
  };
  if (usedCorrection && result.source && result.source.startsWith("model")) {
    result.sourceLabel = "Model output (strict JSON after correction pass)";
  }
  return result;
}

/**
 * Performs lightweight regex-based optimization when model inference is not available.
 *
 * This pass intentionally applies conservative transformations aimed at common
 * readability and safety wins without deep semantic rewrites.
 *
 * @param {string} code User-provided source code.
 * @returns {{ optimizedCode: string, changes: string[], qualityScore: number, impactScore: number }}
 */
function runLocalOptimization(code, reason = "Local heuristic optimization (no model).") {
  const startedAt = performance.now();
  let optimized = code;
  const changes = [];

  if (/\bvar\b/.test(optimized)) {
    optimized = optimized.replace(/\bvar\b/g, "let");
    changes.push("Replaced legacy var declarations with block-scoped let.");
  }

  if (/(^|[^=!])==(?!=)/.test(optimized) || /(^|[^=!])!=(?!=)/.test(optimized)) {
    optimized = optimized
      .replace(/(^|[^=!])==(?!=)/g, "$1===")
      .replace(/(^|[^=!])!=(?!=)/g, "$1!==");
    changes.push("Replaced loose equality checks with strict comparisons.");
  }

  if (/= [^;\n]+ \+ /.test(optimized)) {
    optimized = optimized.replace(/(\w+)\s*=\s*\1\s*\+\s*/g, "$1 += ");
    changes.push("Simplified repeated assignment into compound assignment.");
  }

  if (/if\s*\(([^)]*)\s*===\s*true\)/g.test(optimized)) {
    optimized = optimized.replace(/if\s*\(([^)]*)\s*===\s*true\)/g, (_, expression) => `if (${expression.trim()})`);
    changes.push("Removed redundant boolean comparison.");
  }

  if (/for\s*\(let i = 0; i < ([^.]+)\.length; i\+\+\)/.test(optimized)) {
    changes.push("Detected index-based looping; a model pass may convert this to iteration helpers when behavior is safe.");
  }

  if (!changes.length) {
    changes.push("Code is already compact enough for the local reviewer. Load the model for deeper semantic optimization.");
  }

  const qualityScore = Math.min(94, 66 + changes.length * 7 + Math.round(code.length > 120 ? 8 : 0));
  const impactScore = Math.min(88, 42 + changes.length * 10);
  const localOptimizeMs = Math.max(0, Math.round(performance.now() - startedAt));

  return {
    optimizedCode: optimized,
    changes,
    qualityScore,
    impactScore,
    source: "local",
    sourceLabel: reason,
    modelUsed: "none",
    modelUsedLabel: "No model (local heuristic path)",
    timings: {
      localOptimizeMs
    }
  };
}

/**
 * Renders optimization results into the right-side code panel and insight cards.
 *
 * @param {{ optimizedCode: string, changes: string[], qualityScore: number, impactScore: number }} result
 */
function renderResults(result) {
  els.outputCode.textContent = result.optimizedCode || "No optimized code returned.";
  els.changeList.innerHTML = "";
  result.changes.slice(0, 8).forEach((change) => {
    const item = document.createElement("li");
    item.textContent = change;
    els.changeList.appendChild(item);
  });
  els.qualityScore.textContent = Math.round(result.qualityScore);
  els.impactScore.textContent = Math.round(result.impactScore);
  els.qualityLabel.textContent = result.qualityScore >= 85 ? "Strong result" : result.qualityScore >= 70 ? "Useful improvement" : "Needs review";
  els.impactLabel.textContent = result.impactScore >= 75 ? "High value" : result.impactScore >= 50 ? "Moderate value" : "Light cleanup";
  els.modelUsed.textContent = result.modelUsedLabel || "Unknown model usage";
  els.resultSource.textContent = result.sourceLabel || "Unknown source";
}

/**
 * Renders elapsed optimization time.
 *
 * @param {number} elapsedMs
 * @param {{ modelLoadMs?: number, generationMs?: number, postProcessMs?: number, localOptimizeMs?: number }} [timings]
 */
function renderResponseTime(elapsedMs, timings = {}) {
  const safeMs = Math.max(0, Math.round(elapsedMs));
  const seconds = safeMs / 1000;
  els.responseTime.textContent = seconds >= 10 ? `${seconds.toFixed(1)}s` : `${seconds.toFixed(2)}s`;
  els.responseTimeLabel.textContent = `${safeMs.toLocaleString()} ms end-to-end`;

  const formatMs = (value) => {
    if (!Number.isFinite(value) || value < 0) return "--";
    return `${Math.round(value).toLocaleString()} ms`;
  };

  els.phaseTotal.textContent = formatMs(safeMs);
  els.phaseModelLoad.textContent = formatMs(timings.modelLoadMs);
  els.phaseGeneration.textContent = formatMs(timings.generationMs);
  els.phasePostProcess.textContent = formatMs(timings.postProcessMs);
  els.phaseLocal.textContent = formatMs(timings.localOptimizeMs);
}

/**
 * Main click handler for the optimize action.
 * Guards empty input, shows transient working UI, and always re-enables controls.
 *
 * @returns {Promise<void>}
 */
async function optimizeCode() {
  const code = els.inputCode.value.trim();
  if (!code) {
    els.inputCode.focus();
    return;
  }

  const startedAt = performance.now();

  els.optimizeButton.disabled = true;
  els.outputCode.textContent = "Analyzing and optimizing...";
  els.changeList.innerHTML = "<li>Working through structure, readability, and behavior-preserving changes.</li>";
  els.responseTime.textContent = "...";
  els.responseTimeLabel.textContent = "Timing current run...";
  els.phaseTotal.textContent = "...";
  els.phaseModelLoad.textContent = "...";
  els.phaseGeneration.textContent = "...";
  els.phasePostProcess.textContent = "...";
  els.phaseLocal.textContent = "...";

  try {
    const result =
      els.inferenceMode.value === "backend"
        ? await runBackendOptimization(code)
        : await runModelOptimization(code);
    renderResults(result);
    renderResponseTime(performance.now() - startedAt, result.timings || {});
  } catch (error) {
    const result = runLocalOptimization(code, "Local fallback after model runtime error.");
    renderResults(result);
    els.engineBadge.textContent = "Local review mode";
    setModelState("error", "Model run failed. Local review shown.", 0);
    renderResponseTime(performance.now() - startedAt, result.timings || {});
  } finally {
    els.optimizeButton.disabled = false;
  }
}

/**
 * Event wiring and initial UI bootstrap.
 */
els.inputCode.addEventListener("input", updateInputStats);
els.loadModelButton.addEventListener("click", () => loadModel(true));
els.optimizeButton.addEventListener("click", optimizeCode);
els.inferenceMode.addEventListener("change", () => {
  modelAttempted = false;
  syncInferenceModeUI();
});
els.backendModel.addEventListener("change", updateModeSummary);
els.requireBackendModel.addEventListener("change", updateModeSummary);
els.runtimeMode.addEventListener("change", () => {
  modelAttempted = false;
  if (els.runtimeMode.value === "local-only") {
    engineMode = "local";
    els.engineBadge.textContent = "Local review mode";
    setModelState("error", "Local-only mode enabled. Browser model loading is disabled.", 0);
  }
  syncSafeModeBanner();
  updateModeSummary();
});
els.performanceMode.addEventListener("change", () => {
  modelAttempted = false;
});
els.enableSafeModeButton.addEventListener("click", () => {
  els.runtimeMode.value = "tiny-only";
  modelAttempted = false;
  engine = null;
  engineMode = "local";
  els.activeModel.textContent = "Primary: TinyLlama (safe mode)";
  els.engineBadge.textContent = "Safe mode configured";
  setModelState("loading", "Tiny-only mode enabled. Click Load model to continue.", 0);
  syncSafeModeBanner();
});
els.sampleSelect.addEventListener("change", (event) => {
  const target = event.target;
  const selectedIndex = Number(target.value || 0);
  loadSample(selectedIndex);
});
els.sampleButton.addEventListener("click", () => {
  const selectedIndex = Number(els.sampleSelect.value || 0);
  loadSample(selectedIndex);
});
els.copyButton.addEventListener("click", async () => {
  await navigator.clipboard.writeText(els.outputCode.textContent);
  els.copyButton.textContent = "Copied";
  setTimeout(() => {
    els.copyButton.textContent = "Copy";
  }, 1200);
});

loadSample(0);
syncSafeModeBanner();
syncInferenceModeUI();
updateModeSummary();
