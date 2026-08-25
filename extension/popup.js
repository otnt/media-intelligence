const healthEl = document.getElementById("health");
const pageEl = document.getElementById("page");
const listEl = document.getElementById("model-list");
const startEl = document.getElementById("start");
const statusEl = document.getElementById("status");
const tasksEl = document.getElementById("tasks");

let selectedModel = MDP_DEFAULT_MODEL;
let models = MDP_FALLBACK_MODELS;
let currentUrl = "";
let serviceOk = false;
let extractKeyframes = false;

init();

async function init() {
  const stored = await chrome.storage.local.get(["asrModel", "extractKeyframes"]);
  if (stored.asrModel) selectedModel = stored.asrModel;
  extractKeyframes = Boolean(stored.extractKeyframes);
  const keyframesEl = document.getElementById("keyframes");
  if (keyframesEl) {
    keyframesEl.checked = extractKeyframes;
    keyframesEl.addEventListener("change", () => {
      extractKeyframes = keyframesEl.checked;
      chrome.storage.local.set({ extractKeyframes });
    });
  }
  await Promise.all([loadHealth(), loadTab(), loadModels(), loadTasks()]);
  renderModels();
  startEl.addEventListener("click", submitTask);
  updateStartEnabled();
}

async function loadHealth() {
  const health = await mdpSend({ type: "HEALTH" });
  serviceOk = Boolean(health.ok);
  if (!health.ok) {
    healthEl.className = "sub err";
    healthEl.textContent = health.error || "Local service is not running.";
    return;
  }
  healthEl.className = "sub ok";
  healthEl.textContent = "Service running on 127.0.0.1";
}

async function loadTab() {
  const tab = await mdpGetActiveTab();
  currentUrl = tab?.url || "";
  if (!currentUrl) {
    pageEl.textContent = "Could not read the current tab. Reload the extension, then open this popup from a video page.";
    return;
  }
  if (!mdpIsSupportedVideoUrl(currentUrl)) {
    pageEl.textContent = `Not a Bilibili, YouTube, or Xiaohongshu page:\n${currentUrl}`;
    return;
  }
  pageEl.textContent = tab.title || currentUrl;
}

async function loadModels() {
  const response = await mdpSend({ type: "MODELS" });
  if (response.ok && Array.isArray(response.models) && response.models.length) {
    models = response.models;
  }
  const preferred = selectedModel || response.default || MDP_DEFAULT_MODEL;
  const match = models.find((item) => item.id === preferred && item.available !== false);
  selectedModel = match ? match.id : models.find((item) => item.available !== false)?.id || preferred;
}

function renderModels() {
  mdpRenderModelList(listEl, models, selectedModel, (modelId) => {
    selectedModel = modelId;
    chrome.storage.local.set({ asrModel: selectedModel });
    updateStartEnabled();
  });
}

async function loadTasks() {
  const listing = await mdpSend({ type: "TASKS" });
  if (!listing.ok) {
    tasksEl.innerHTML = `<li class="meta">${mdpEscapeHtml(listing.error || "Could not load tasks")}</li>`;
    return;
  }
  const tasks = listing.tasks || [];
  if (!tasks.length) {
    tasksEl.innerHTML = `<li class="meta">No tasks yet.</li>`;
    return;
  }
  tasksEl.innerHTML = tasks
    .slice(0, 8)
    .map((task) => {
      const title = task.title || task.video_id || task.url;
      const extra = task.error ? ` · ${task.error}` : "";
      return `<li>
        <div class="title">${mdpEscapeHtml(title)}</div>
        <div class="meta">${mdpEscapeHtml(task.status)} · ${mdpEscapeHtml(task.asr_label || task.asr_model)}${mdpEscapeHtml(extra)}</div>
      </li>`;
    })
    .join("");
}

function updateStartEnabled() {
  const model = models.find((item) => item.id === selectedModel);
  const modelOk = Boolean(model && model.available !== false);
  const urlOk = mdpIsSupportedVideoUrl(currentUrl);
  startEl.disabled = !(serviceOk && urlOk && modelOk);
  if (!startEl.disabled) {
    if (statusEl.textContent.startsWith("Start is disabled")) statusEl.textContent = "";
    return;
  }
  if (!serviceOk) statusEl.textContent = "Start is disabled until the local service is running.";
  else if (!currentUrl) statusEl.textContent = "Start is disabled because this tab URL could not be read.";
  else if (!urlOk) statusEl.textContent = "Start is disabled until you are on a Bilibili, YouTube, or Xiaohongshu page.";
  else if (!modelOk) statusEl.textContent = "Start is disabled because the selected ASR model is not installed.";
}

async function submitTask() {
  statusEl.className = "status";
  statusEl.textContent = "Submitting…";
  startEl.disabled = true;
  const response = await mdpSend({
    type: "CREATE_TASK",
    url: currentUrl,
    asr_model: selectedModel,
    extract_keyframes: extractKeyframes,
  });
  updateStartEnabled();
  if (!response.ok) {
    statusEl.className = "status err";
    statusEl.textContent = response.error || "Could not add the task.";
    return;
  }
  chrome.storage.local.set({ asrModel: selectedModel, extractKeyframes });
  statusEl.className = "status ok";
  statusEl.textContent = `✓ Queued\nASR: ${response.asr_label || selectedModel}${extractKeyframes ? "\nKeyframes: on" : ""}`;
  await loadTasks();
}
