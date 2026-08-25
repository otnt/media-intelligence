const MDP_DEFAULT_MODEL = "qwen3-asr-1.7b";
const MDP_FALLBACK_MODELS = [
  { id: "qwen3-asr-1.7b", label: "Qwen3-ASR-1.7B", runtime: "MLX Qwen3-ASR", available: true, code_switching: true },
  { id: "whisper-large-v3", label: "Whisper large-v3", runtime: "MLX Whisper", available: true, code_switching: false },
  { id: "whisper-large-v3-turbo", label: "Whisper large-v3-turbo", runtime: "MLX Whisper", available: true, code_switching: false },
];
const MDP_XHS_NOTE_ID = /^[0-9a-f]{24}$/i;

function mdpHostname(urlString, base) {
  try {
    return new URL(urlString, base || "https://www.xiaohongshu.com").hostname.replace(/^www\./, "").toLowerCase();
  } catch (_error) {
    return "";
  }
}

function mdpIsXiaohongshuHost(urlString) {
  const host = mdpHostname(urlString);
  return (
    host === "xiaohongshu.com" ||
    host.endsWith(".xiaohongshu.com") ||
    host === "rednote.com" ||
    host.endsWith(".rednote.com") ||
    host === "xhslink.com" ||
    host.endsWith(".xhslink.com")
  );
}

function mdpXiaohongshuOrigin(urlString, base) {
  const host = mdpHostname(urlString, base);
  if (host === "rednote.com" || host.endsWith(".rednote.com")) return "https://www.rednote.com";
  return "https://www.xiaohongshu.com";
}

function mdpParseXiaohongshuNote(urlString, base) {
  try {
    const url = new URL(urlString, base || "https://www.xiaohongshu.com");
    const host = (url.hostname || "").replace(/^www\./, "").toLowerCase();
    if (host === "xhslink.com" || host.endsWith(".xhslink.com")) {
      const parts = url.pathname.split("/").filter(Boolean);
      const noteId = parts[0] === "o" ? parts[1] : parts[parts.length - 1];
      if (!noteId) return null;
      return { noteId, url: url.toString(), token: "", source: "", origin: "" };
    }
    const origin = mdpXiaohongshuOrigin(url.toString());
    if (
      host !== "xiaohongshu.com" &&
      !host.endsWith(".xiaohongshu.com") &&
      host !== "rednote.com" &&
      !host.endsWith(".rednote.com")
    ) {
      return null;
    }
    const parts = url.pathname.split("/").filter(Boolean);
    let noteId = "";
    if (parts[0] === "explore" && MDP_XHS_NOTE_ID.test(parts[1] || "")) noteId = parts[1];
    else if (parts[0] === "discovery" && parts[1] === "item" && MDP_XHS_NOTE_ID.test(parts[2] || "")) noteId = parts[2];
    else if (parts[0] === "search_result" && MDP_XHS_NOTE_ID.test(parts[1] || "")) noteId = parts[1];
    else if (parts[0] === "user" && parts[1] === "profile" && MDP_XHS_NOTE_ID.test(parts[3] || "")) noteId = parts[3];
    if (!noteId) return null;
    const token = url.searchParams.get("xsec_token") || "";
    const source = url.searchParams.get("xsec_source") || "";
    return { noteId, url: mdpXiaohongshuNoteUrl(noteId, token, source, origin), token, source, origin };
  } catch (_error) {
    return null;
  }
}

function mdpXiaohongshuNoteUrl(noteId, token, source, origin) {
  const url = new URL(`${origin || "https://www.xiaohongshu.com"}/explore/${noteId}`);
  if (token) url.searchParams.set("xsec_token", token);
  if (source) url.searchParams.set("xsec_source", source);
  return url.toString();
}

function mdpMergeXhsToken(map, id, token, source) {
  if (!id || !token || !MDP_XHS_NOTE_ID.test(String(id))) return map;
  const current = map[String(id)] || {};
  map[String(id)] = { token: String(token), source: String(source || current.source || "") };
  return map;
}

function mdpCollectXhsTokens(state, scriptTexts) {
  const map = {};
  const walk = (value, depth) => {
    if (!value || depth > 10 || typeof value !== "object") return;
    const id =
      value.noteId ||
      value.note_id ||
      value.noteCard?.noteId ||
      ((value.modelType === "note" || value.noteCard) && value.id);
    const token = value.xsecToken || value.xsec_token || value.noteCard?.xsecToken || value.noteCard?.xsec_token;
    const source = value.xsecSource || value.xsec_source || value.noteCard?.xsecSource || "";
    mdpMergeXhsToken(map, id, token, source);
    const children = Array.isArray(value) ? value : Object.values(value);
    for (const child of children) walk(child, depth + 1);
  };
  walk(state, 0);
  const pair = /"id"\s*:\s*"([0-9a-f]{24})"\s*,\s*"modelType"\s*:\s*"note"[\s\S]{0,8000}?"xsecToken"\s*:\s*"([^"]+)"/gi;
  for (const text of scriptTexts || []) {
    pair.lastIndex = 0;
    let match;
    while ((match = pair.exec(String(text || "")))) {
      mdpMergeXhsToken(map, match[1], match[2], "pc_feed");
    }
  }
  return map;
}

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
    if (mdpIsXiaohongshuHost(urlString)) return Boolean(mdpParseXiaohongshuNote(urlString));
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
