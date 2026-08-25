const MDP_DEFAULT_MODEL = "whisper-large-v3-turbo";
const MDP_FALLBACK_MODELS = [
  { id: "whisper-large-v3", label: "Whisper large-v3", runtime: "MLX Whisper", available: true },
  { id: "whisper-large-v3-turbo", label: "Whisper large-v3-turbo", runtime: "MLX Whisper", available: true },
  { id: "qwen3-asr-1.7b", label: "Qwen3-ASR-1.7B", runtime: "Qwen3-ASR", available: true },
];

function mdpIsSupportedVideoUrl(urlString) {
  try {
    const url = new URL(urlString);
    const host = (url.hostname || "").replace(/^www\./, "").toLowerCase();
    if (host === "youtu.be") return url.pathname.replace(/\/$/, "").length > 1;
    if (host === "youtube.com" || host === "m.youtube.com") {
      if (url.pathname === "/watch") return Boolean(url.searchParams.get("v"));
      if (url.pathname.startsWith("/shorts/")) return url.pathname.split("/").filter(Boolean).length >= 2;
      return false;
    }
    if (host === "bilibili.com" || host === "m.bilibili.com") {
      return /\/video\/(?:BV|av)/i.test(url.pathname);
    }
    if (host === "b23.tv") return url.pathname.replace(/\/$/, "").length > 1;
    return false;
  } catch (_error) {
    return false;
  }
}

function mdpEscapeHtml(value) {
  return String(value || "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;");
}

function mdpSend(message) {
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

function mdpRenderModelList(container, models, selectedId, onChange) {
  container.innerHTML = models
    .map((model) => {
      const checked = model.id === selectedId ? "checked" : "";
      const unavailable = model.available === false;
      const runtime = unavailable ? `${model.runtime} · not installed` : model.runtime;
      return `
        <label class="option ${unavailable ? "unavailable" : ""}">
          <input type="radio" name="asr" value="${mdpEscapeHtml(model.id)}" ${checked} ${unavailable ? "disabled" : ""}>
          <span>
            <span class="label">${mdpEscapeHtml(model.label)}</span>
            <div class="runtime">${mdpEscapeHtml(runtime)}</div>
          </span>
        </label>
      `;
    })
    .join("");
  container.querySelectorAll("input[name=asr]").forEach((input) => {
    input.addEventListener("change", () => onChange(input.value));
  });
}
