const ROOT_ID = "mdp-root";
const DEFAULT_MODEL = "whisper-large-v3-turbo";
const FALLBACK_MODELS = [
  { id: "whisper-large-v3", label: "Whisper large-v3", runtime: "MLX Whisper", available: true },
  { id: "whisper-large-v3-turbo", label: "Whisper large-v3-turbo", runtime: "MLX Whisper", available: true },
  { id: "qwen3-asr-1.7b", label: "Qwen3-ASR-1.7B", runtime: "Qwen3-ASR", available: true },
];

const STYLE = `
:host { all: initial; }
.wrap { position: relative; font-family: ui-sans-serif, system-ui, -apple-system, "Segoe UI", sans-serif; }
.btn {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  height: 36px;
  padding: 0 12px;
  border: 0;
  border-radius: 18px;
  background: #4f46e5;
  color: #fff;
  font-size: 13px;
  font-weight: 650;
  letter-spacing: 0.01em;
  cursor: pointer;
  box-shadow: 0 1px 2px rgba(15, 23, 42, 0.18);
}
.btn:hover { background: #4338ca; }
.btn[disabled] { opacity: 0.65; cursor: default; }
.caret { font-size: 10px; opacity: 0.9; }
.panel {
  position: absolute;
  top: calc(100% + 8px);
  right: 0;
  width: 280px;
  padding: 12px;
  border-radius: 14px;
  background: #0f172a;
  color: #e2e8f0;
  box-shadow: 0 16px 40px rgba(2, 6, 23, 0.35);
  z-index: 10;
}
.panel[hidden] { display: none; }
.title { font-size: 14px; font-weight: 700; color: #fff; margin: 0 0 10px; }
.section { font-size: 11px; font-weight: 700; text-transform: uppercase; letter-spacing: 0.08em; color: #94a3b8; margin-bottom: 8px; }
.option {
  display: flex;
  gap: 8px;
  align-items: flex-start;
  padding: 8px;
  border-radius: 10px;
  cursor: pointer;
}
.option:hover { background: rgba(148, 163, 184, 0.12); }
.option input { margin-top: 3px; }
.option.unavailable { opacity: 0.55; }
.label { font-size: 13px; color: #f8fafc; font-weight: 600; }
.runtime { font-size: 11px; color: #94a3b8; margin-top: 2px; }
.start {
  width: 100%;
  margin-top: 10px;
  height: 34px;
  border: 0;
  border-radius: 10px;
  background: #22c55e;
  color: #052e16;
  font-weight: 800;
  cursor: pointer;
}
.start:hover { background: #16a34a; }
.start[disabled] { opacity: 0.6; cursor: default; }
.status { margin-top: 8px; font-size: 12px; line-height: 1.4; color: #cbd5e1; min-height: 16px; }
.status.ok { color: #86efac; }
.status.err { color: #fda4af; }
`;

let selectedModel = DEFAULT_MODEL;
let models = FALLBACK_MODELS;
let lastUrl = "";

init();

function init() {
  chrome.storage.local.get(["asrModel"], (stored) => {
    if (stored.asrModel) selectedModel = stored.asrModel;
    inject();
  });
  document.addEventListener("yt-navigate-finish", () => inject(true));
  const observer = new MutationObserver(() => {
    if (location.href !== lastUrl) inject(true);
    else if (isVideoPage() && !document.getElementById(ROOT_ID)) inject();
  });
  observer.observe(document.documentElement, { childList: true, subtree: true });
}

function isVideoPage() {
  const host = location.hostname;
  if (host.includes("youtube.com")) return location.pathname === "/watch";
  if (host.includes("bilibili.com")) return /\/video\//.test(location.pathname);
  return false;
}

function inject(force = false) {
  lastUrl = location.href;
  const existing = document.getElementById(ROOT_ID);
  if (!isVideoPage()) {
    existing?.remove();
    return;
  }
  if (existing && !force) return;
  existing?.remove();
  const host = document.createElement("div");
  host.id = ROOT_ID;
  const shadow = host.attachShadow({ mode: "open" });
  shadow.innerHTML = `<style>${STYLE}</style>${renderMarkup()}`;
  mount(host);
  wire(shadow, host);
  refreshModels(shadow);
}

function renderMarkup() {
  return `
    <div class="wrap">
      <button class="btn" type="button" id="toggle">✨ Save &amp; Transcribe <span class="caret">▾</span></button>
      <div class="panel" id="panel" hidden>
        <p class="title">✨ Save &amp; Transcribe</p>
        <div class="section">ASR Model</div>
        <div id="model-list"></div>
        <button class="start" type="button" id="start">Start</button>
        <div class="status" id="status"></div>
      </div>
    </div>
  `;
}

function mount(host) {
    const selectors = location.hostname.includes("youtube")
    ? ["#actions", "#top-level-buttons-computed", "ytd-watch-metadata #top-row", "#owner"]
    : [".video-toolbar-right", ".video-toolbar-container .toolbar-right", ".video-toolbar-container", "#arc_toolbar_report"];
  for (const selector of selectors) {
    const target = document.querySelector(selector);
    if (target) {
      host.classList.remove("mdp-floating");
      target.insertAdjacentElement("afterbegin", host);
      return;
    }
  }
  host.classList.add("mdp-floating");
  document.documentElement.appendChild(host);
}

function wire(shadow, host) {
  const toggle = shadow.getElementById("toggle");
  const panel = shadow.getElementById("panel");
  const start = shadow.getElementById("start");
  toggle.addEventListener("click", (event) => {
    event.preventDefault();
    event.stopPropagation();
    panel.hidden = !panel.hidden;
  });
  start.addEventListener("click", (event) => {
    event.preventDefault();
    event.stopPropagation();
    submitTask(shadow, start);
  });
  document.addEventListener(
    "click",
    (event) => {
      const path = event.composedPath();
      if (!path.includes(host)) panel.hidden = true;
    },
    true
  );
}

function renderModels(shadow) {
  const list = shadow.getElementById("model-list");
  list.innerHTML = models
    .map((model) => {
      const checked = model.id === selectedModel ? "checked" : "";
      const unavailable = model.available === false ? "unavailable" : "";
      const runtime = model.available === false ? `${model.runtime} · not installed` : model.runtime;
      return `
        <label class="option ${unavailable}">
          <input type="radio" name="asr" value="${model.id}" ${checked} ${model.available === false ? "disabled" : ""}>
          <span>
            <span class="label">${escapeHtml(model.label)}</span>
            <div class="runtime">${escapeHtml(runtime)}</div>
          </span>
        </label>
      `;
    })
    .join("");
  list.querySelectorAll("input[name=asr]").forEach((input) => {
    input.addEventListener("change", () => {
      selectedModel = input.value;
      chrome.storage.local.set({ asrModel: selectedModel });
    });
  });
}

async function refreshModels(shadow) {
  renderModels(shadow);
  const response = await send({ type: "MODELS" });
  if (!response.ok) return;
  if (Array.isArray(response.models) && response.models.length) models = response.models;
  const stored = await chrome.storage.local.get(["asrModel"]);
  const preferred = stored.asrModel || response.default || DEFAULT_MODEL;
  const match = models.find((item) => item.id === preferred && item.available !== false);
  selectedModel = match ? match.id : models.find((item) => item.available !== false)?.id || preferred;
  renderModels(shadow);
}

async function submitTask(shadow, start) {
  const status = shadow.getElementById("status");
  status.className = "status";
  status.textContent = "Submitting…";
  start.disabled = true;
  const response = await send({
    type: "CREATE_TASK",
    url: location.href,
    asr_model: selectedModel,
  });
  start.disabled = false;
  if (!response.ok) {
    status.className = "status err";
    status.textContent = response.error || "Could not add the task.";
    return;
  }
  chrome.storage.local.set({ asrModel: selectedModel });
  status.className = "status ok";
  status.textContent = `✓ Added to queue\nASR: ${response.asr_label || selectedModel}`;
}

function send(message) {
  return new Promise((resolve) => {
    chrome.runtime.sendMessage(message, (response) => {
      if (chrome.runtime.lastError) {
        resolve({ ok: false, error: chrome.runtime.lastError.message });
        return;
      }
      resolve(response || { ok: false, error: "No response from the extension" });
    });
  });
}

function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;");
}
