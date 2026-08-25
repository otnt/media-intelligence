const ROOT_ID = "mdp-root";

let selectedModel = MDP_DEFAULT_MODEL;
let models = MDP_FALLBACK_MODELS;
let lastUrl = "";

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
  box-shadow: 0 8px 24px rgba(15, 23, 42, 0.28);
}
.btn:hover { background: #4338ca; }
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
.status { margin-top: 8px; font-size: 12px; line-height: 1.4; color: #cbd5e1; min-height: 16px; white-space: pre-wrap; }
.status.ok { color: #86efac; }
.status.err { color: #fda4af; }
`;

init();

chrome.runtime.onMessage.addListener((message, _sender, sendResponse) => {
  if (message.type === "GET_PAGE") {
    sendResponse({ url: location.href, title: document.title });
  }
});

function init() {
  chrome.storage.local.get(["asrModel"], (stored) => {
    if (stored.asrModel) selectedModel = stored.asrModel;
    inject();
  });
  document.addEventListener("yt-navigate-finish", () => inject(true));
  window.addEventListener("yt-navigate-finish", () => inject(true));
  setInterval(() => {
    if (location.href !== lastUrl) inject(true);
  }, 800);
}

function inject(force = false) {
  lastUrl = location.href;
  const existing = document.getElementById(ROOT_ID);
  if (!mdpIsSupportedVideoUrl(location.href)) {
    existing?.remove();
    return;
  }
  if (existing && !force) return;
  existing?.remove();
  const host = document.createElement("div");
  host.id = ROOT_ID;
  host.classList.add("mdp-floating");
  const shadow = host.attachShadow({ mode: "open" });
  shadow.innerHTML = `<style>${STYLE}</style>
    <div class="wrap">
      <button class="btn" type="button" id="toggle">✨ Save &amp; Transcribe <span class="caret">▾</span></button>
      <div class="panel" id="panel" hidden>
        <p class="title">✨ Save &amp; Transcribe</p>
        <div class="section">ASR Model</div>
        <div id="model-list"></div>
        <button class="start" type="button" id="start">Start</button>
        <div class="status" id="status"></div>
      </div>
    </div>`;
  (document.body || document.documentElement).appendChild(host);
  wire(shadow, host);
  refreshModels(shadow);
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
      if (!event.composedPath().includes(host)) panel.hidden = true;
    },
    true
  );
}

function renderModels(shadow) {
  mdpRenderModelList(shadow.getElementById("model-list"), models, selectedModel, (modelId) => {
    selectedModel = modelId;
    chrome.storage.local.set({ asrModel: selectedModel });
  });
}

async function refreshModels(shadow) {
  renderModels(shadow);
  const response = await mdpSend({ type: "MODELS" });
  if (!response.ok) return;
  if (Array.isArray(response.models) && response.models.length) models = response.models;
  const stored = await chrome.storage.local.get(["asrModel"]);
  const preferred = stored.asrModel || response.default || MDP_DEFAULT_MODEL;
  const match = models.find((item) => item.id === preferred && item.available !== false);
  selectedModel = match ? match.id : models.find((item) => item.available !== false)?.id || preferred;
  renderModels(shadow);
}

async function submitTask(shadow, start) {
  const status = shadow.getElementById("status");
  status.className = "status";
  status.textContent = "Submitting…";
  start.disabled = true;
  const response = await mdpSend({
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
