const DEFAULT_BASE = "http://127.0.0.1:8875";
const LEGACY_BASES = new Set([
  "http://127.0.0.1:8765",
  "http://localhost:8765",
]);

chrome.runtime.onMessage.addListener((message, _sender, sendResponse) => {
  handle(message)
    .then(sendResponse)
    .catch((error) => sendResponse({ ok: false, error: error.message || String(error) }));
  return true;
});

async function handle(message) {
  const base = await getBaseUrl();
  if (message.type === "GET_BASE") {
    return { ok: true, base };
  }
  if (message.type === "SET_BASE") {
    const next = normalizeBaseUrl(message.base);
    await chrome.storage.local.set({ apiBase: next });
    return { ok: true, base: next };
  }
  if (message.type === "HEALTH") {
    return request(`${base}/v1/health`);
  }
  if (message.type === "MODELS") {
    return request(`${base}/v1/models`);
  }
  if (message.type === "TASKS") {
    return request(`${base}/v1/tasks`);
  }
  if (message.type === "CREATE_TASK") {
    return request(`${base}/v1/tasks`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        url: message.url,
        asr_model: message.asr_model,
        language: message.language || "auto",
        extract_keyframes: Boolean(message.extract_keyframes),
      }),
    });
  }
  return { ok: false, error: `Unknown message ${message.type}` };
}

function normalizeBaseUrl(value) {
  const raw = String(value || "").trim().replace(/\/+$/, "");
  if (!raw || LEGACY_BASES.has(raw)) return DEFAULT_BASE;
  return raw;
}

async function getBaseUrl() {
  const stored = await chrome.storage.local.get(["apiBase"]);
  const base = normalizeBaseUrl(stored.apiBase || DEFAULT_BASE);
  if (stored.apiBase && stored.apiBase !== base) {
    await chrome.storage.local.set({ apiBase: base });
  }
  return base;
}

async function request(url, options = {}) {
  let response;
  try {
    response = await fetch(url, options);
  } catch (error) {
    throw new Error(
      `Local service is not running at ${url}. Start it with: media-pipeline serve`
    );
  }
  const data = await response.json().catch(() => ({}));
  if (!response.ok) {
    const detail = data.detail || data.error || `${response.status} ${response.statusText}`;
    throw new Error(String(detail));
  }
  return { ok: true, ...data };
}
