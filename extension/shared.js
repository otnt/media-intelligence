const MDP_DEFAULT_MODEL = "whisper-large-v3-turbo";
const MDP_FALLBACK_MODELS = [
  { id: "whisper-large-v3", label: "Whisper large-v3", runtime: "MLX Whisper", available: true, code_switching: false },
  { id: "whisper-large-v3-turbo", label: "Whisper large-v3-turbo", runtime: "MLX Whisper", available: true, code_switching: false },
  { id: "qwen3-asr-1.7b", label: "Qwen3-ASR-1.7B", runtime: "Qwen3-ASR", available: true, code_switching: true },
];

function mdpIsSupportedVideoUrl(urlString) {
  try {
    const url = new URL(urlString);
    const host = (url.hostname || "").replace(/^www\./, "").toLowerCase();
    if (host === "youtu.be") return url.pathname.replace(/\/$/, "").length > 1;
    if (host === "youtube.com" || host === "m.youtube.com" || host === "music.youtube.com") {
      if (url.searchParams.get("v")) return true;
      if (url.pathname === "/watch") return true;
      if (url.pathname.startsWith("/shorts/")) return url.pathname.split("/").filter(Boolean).length >= 2;
      if (url.pathname.startsWith("/live/")) return url.pathname.split("/").filter(Boolean).length >= 2;
      if (url.pathname.startsWith("/embed/")) return url.pathname.split("/").filter(Boolean).length >= 2;
      return false;
    }
    if (host === "bilibili.com" || host === "m.bilibili.com" || host.endsWith(".bilibili.com")) {
      if (url.searchParams.get("bvid")) return true;
      if (/\/video\/(?:BV|bv|av|AV)/i.test(url.pathname)) return true;
      if (/BV[0-9A-Za-z]+/.test(url.href)) return true;
      return false;
    }
    if (host === "b23.tv") return url.pathname.replace(/\/$/, "").length > 1;
    return false;
  } catch (_error) {
    return false;
  }
}

async function mdpGetActiveTab() {
  const queries = [
    { active: true, lastFocusedWindow: true },
    { active: true, currentWindow: true },
  ];
  for (const query of queries) {
    const tabs = await chrome.tabs.query(query);
    for (const tab of tabs || []) {
      if (!tab?.id || tab.url?.startsWith("chrome-extension://")) continue;
      const fromPage = await mdpAskTabForPage(tab.id);
      const url = fromPage?.url || tab.url || tab.pendingUrl || "";
      if (!url || url.startsWith("chrome://") || url.startsWith("chrome-extension://")) continue;
      return { id: tab.id, url, title: fromPage?.title || tab.title || url };
    }
  }
  return null;
}

function mdpAskTabForPage(tabId) {
  return new Promise((resolve) => {
    chrome.tabs.sendMessage(tabId, { type: "GET_PAGE" }, (response) => {
      if (chrome.runtime.lastError) {
        resolve(null);
        return;
      }
      resolve(response || null);
    });
  });
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
      const bits = [model.runtime];
      if (model.code_switching) bits.push("multilingual");
      if (unavailable) bits.push("not installed");
      const runtime = bits.join(" · ");
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
