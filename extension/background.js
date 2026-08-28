const DEFAULT_BASE = "http://127.0.0.1:8875";

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
    await chrome.storage.local.set({ apiBase: message.base });
    return { ok: true, base: message.base };
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

async function getBaseUrl() {
  const stored = await chrome.storage.local.get(["apiBase"]);
  return stored.apiBase || DEFAULT_BASE;
}

async function request(url, options = {}) {
  let response;
  try {
    response = await fetch(url, options);
  } catch (error) {
    throw new Error(
      "Local service is not running. Start it with: media-pipeline serve"
    );
  }
  const data = await response.json().catch(() => ({}));
  if (!response.ok) {
    const detail = data.detail || data.error || `${response.status} ${response.statusText}`;
    throw new Error(String(detail));
  }
  return { ok: true, ...data };
}
