const ROOT_ID = "mdp-root";

let selectedModel = ""; // empty: let the server default (asr.default) decide
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

const CARD_STYLE = `
:host {
  all: initial;
  display: block !important;
  pointer-events: auto !important;
}
button {
  display: inline-flex;
  align-items: center;
  gap: 4px;
  height: 28px;
  padding: 0 10px;
  border: 0;
  border-radius: 14px;
  background: rgba(15, 23, 42, 0.88);
  color: #fff;
  font: 650 12px/1 ui-sans-serif, system-ui, -apple-system, sans-serif;
  letter-spacing: 0.01em;
  cursor: pointer;
  box-shadow: 0 6px 16px rgba(15, 23, 42, 0.35);
  pointer-events: auto !important;
}
button:hover { background: #4f46e5; }
button.ok { background: #15803d; }
button.err { background: #b91c1c; }
button[disabled] { opacity: 0.75; cursor: default; }
`;

let cardObserver = null;
let cardScanTimer = 0;
let extractClicksBound = false;
const xhsTokenMap = {};

init();

chrome.runtime.onMessage.addListener((message, _sender, sendResponse) => {
  if (message.type === "GET_PAGE") {
    sendResponse({ url: location.href, title: document.title });
  }
});

const YT_CARD_SELECTORS =
  "ytd-rich-item-renderer, ytd-video-renderer, ytd-grid-video-renderer, ytd-compact-video-renderer, ytd-playlist-video-renderer, ytd-reel-item-renderer";
const BILI_CARD_SELECTORS =
  ".bili-video-card, .video-card, .feed-card, .rank-item, .video-list-item, .small-item, .recommended-card";

function onXhsSurface() {
  return mdpIsXiaohongshuHost(location.href) || Boolean(window.__MDP_TEST_FEED);
}

function onCardSurface() {
  return onXhsSurface() || mdpIsYoutubeHost(location.href) || mdpIsBilibiliHost(location.href);
}

function init() {
  bindExtractClicks();
  bindXhsTokenMessages();
  injectPageBridge();
  const boot = () => {
    chrome.storage.local.get(["asrModel"], (stored) => {
      if (stored.asrModel) selectedModel = stored.asrModel;
      inject();
    });
  };
  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", boot);
  else boot();
  document.addEventListener("yt-navigate-finish", () => inject(true));
  window.addEventListener("yt-navigate-finish", () => inject(true));
  setInterval(() => {
    if (location.href !== lastUrl) {
      inject(true);
      return;
    }
    if (onCardSurface()) scanVideoCards();
  }, 800);
}

function inject(_force = false) {
  lastUrl = location.href;
  const existing = document.getElementById(ROOT_ID);
  // No floating panel — it blocks site UI on XHS/RedNote and elsewhere.
  // Use card ✨ Extract on feeds or the extension popup on note pages.
  existing?.remove();
  if (!onCardSurface()) {
    stopVideoCards();
    return;
  }
  if (onXhsSurface()) injectPageBridge();
  watchVideoCards();
  scanVideoCards();
}

function wire(shadow, host, onNote) {
  const toggle = shadow.getElementById("toggle");
  const panel = shadow.getElementById("panel");
  const start = shadow.getElementById("start");
  const keyframes = shadow.getElementById("keyframes");
  chrome.storage.local.get(["extractKeyframes"], (stored) => {
    keyframes.checked = Boolean(stored.extractKeyframes);
  });
  keyframes.addEventListener("change", () => {
    chrome.storage.local.set({ extractKeyframes: keyframes.checked });
  });
  toggle.addEventListener("click", (event) => {
    event.preventDefault();
    event.stopPropagation();
    panel.hidden = !panel.hidden;
  });
  start.disabled = !onNote;
  start.addEventListener("click", (event) => {
    event.preventDefault();
    event.stopPropagation();
    if (!mdpIsSupportedVideoUrl(location.href)) return;
    submitTask(shadow, start, location.href);
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

async function submitTask(shadow, start, url) {
  const status = shadow.getElementById("status");
  status.className = "status";
  status.textContent = "Submitting…";
  start.disabled = true;
  const keyframes = Boolean(shadow.getElementById("keyframes")?.checked);
  const response = await mdpSend({
    type: "CREATE_TASK",
    url,
    asr_model: selectedModel,
    extract_keyframes: keyframes,
  });
  start.disabled = !mdpIsSupportedVideoUrl(location.href);
  if (!response.ok) {
    status.className = "status err";
    status.textContent = response.error || "Could not add the task.";
    return;
  }
  chrome.storage.local.set({ asrModel: selectedModel, extractKeyframes: keyframes });
  status.className = "status ok";
  status.textContent = `✓ Added to queue\nASR: ${response.asr_label || selectedModel}${keyframes ? "\nKeyframes: on" : ""}`;
}

function watchVideoCards() {
  bindExtractClicks();
  if (cardObserver) {
    scanVideoCards();
    return;
  }
  const schedule = () => {
    if (cardScanTimer) return;
    cardScanTimer = window.setTimeout(() => {
      cardScanTimer = 0;
      scanVideoCards();
    }, 250);
  };
  cardObserver = new MutationObserver(schedule);
  cardObserver.observe(document.documentElement, { childList: true, subtree: true });
  window.addEventListener("scroll", schedule, { passive: true });
  scanVideoCards();
}

function stopVideoCards() {
  if (cardObserver) {
    cardObserver.disconnect();
    cardObserver = null;
  }
  if (cardScanTimer) {
    window.clearTimeout(cardScanTimer);
    cardScanTimer = 0;
  }
  document.querySelectorAll(".mdp-card-extract").forEach((host) => host.remove());
}

function scanVideoCards() {
  if (onXhsSurface()) scanXhsCards();
  else if (mdpIsYoutubeHost(location.href)) scanYoutubeCards();
  else if (mdpIsBilibiliHost(location.href)) scanBilibiliCards();
}

function cleanupOrphanCardButtons(cards) {
  document.querySelectorAll(".mdp-card-extract").forEach((host) => {
    if (!cards.has(host.parentElement)) host.remove();
  });
}

function scanXhsCards() {
  if (!onXhsSurface()) return;
  bindExtractClicks();
  const tokens = collectXhsTokens();
  const cards = new Map();
  const remember = (link) => {
    if (!link || link.closest("#mdp-root, .mdp-card-extract")) return;
    const parsed = mdpParseXiaohongshuNote(link.href, location.href);
    if (!parsed) return;
    const card = cardRoot(link);
    if (!card) return;
    const current = cards.get(card);
    const currentParsed = current ? mdpParseXiaohongshuNote(current.href, location.href) : null;
    if (!current || (parsed.token && !(currentParsed && currentParsed.token))) cards.set(card, link);
  };
  document.querySelectorAll("section.note-item, .note-item").forEach((card) => {
    card.querySelectorAll("a[href]").forEach((link) => remember(link));
  });
  document.querySelectorAll("a[href]").forEach((link) => {
    if (!link.querySelector("img")) return;
    remember(link);
  });
  for (const [card, link] of cards) {
    mountNoteCover(card, link, tokens);
  }
  cleanupOrphanCardButtons(cards);
}

function scanYoutubeCards() {
  if (!mdpIsYoutubeHost(location.href)) return;
  bindExtractClicks();
  const cards = new Map();
  const remember = (link) => {
    if (!link || link.closest("#mdp-root, .mdp-card-extract")) return;
    const parsed = mdpParseYoutubeVideo(link.href, location.href);
    if (!parsed) return;
    const card = ytCardRoot(link);
    if (!card) return;
    if (!cards.has(card)) cards.set(card, link);
  };
  document.querySelectorAll(YT_CARD_SELECTORS).forEach((card) => {
    card.querySelectorAll('a[href*="/watch"], a[href*="/shorts/"], a[href*="/live/"]').forEach(remember);
  });
  document.querySelectorAll('a[href*="/watch?v="], a[href*="/shorts/"], a[href*="/live/"]').forEach((link) => {
    if (link.querySelector("img, yt-image, ytd-thumbnail, #thumbnail")) remember(link);
  });
  for (const [card, link] of cards) {
    const parsed = mdpParseYoutubeVideo(link.href, location.href);
    if (parsed) mountCardButton(card, parsed.url, parsed.videoId);
  }
  cleanupOrphanCardButtons(cards);
}

function ytCardRoot(link) {
  const named = link.closest(YT_CARD_SELECTORS);
  if (named) return named;
  const parent = link.parentElement;
  if (!parent || parent === document.body || parent === document.documentElement) return link;
  return parent;
}

function scanBilibiliCards() {
  if (!mdpIsBilibiliHost(location.href)) return;
  bindExtractClicks();
  const cards = new Map();
  const remember = (link) => {
    if (!link || link.closest("#mdp-root, .mdp-card-extract")) return;
    const parsed = mdpParseBilibiliVideo(link.href, location.href);
    if (!parsed) return;
    const card = biliCardRoot(link);
    if (!card) return;
    if (!cards.has(card)) cards.set(card, link);
  };
  document.querySelectorAll(BILI_CARD_SELECTORS).forEach((card) => {
    card.querySelectorAll('a[href*="/video/"], a[href*="bvid="]').forEach(remember);
  });
  document.querySelectorAll('a[href*="/video/BV"], a[href*="/video/bv"], a[href*="/video/av"]').forEach((link) => {
    if (link.querySelector("img, picture, .bili-video-card__image--wrap, .bili-video-card__cover")) remember(link);
  });
  for (const [card, link] of cards) {
    const parsed = mdpParseBilibiliVideo(link.href, location.href);
    if (parsed) mountCardButton(card, parsed.url, parsed.videoId);
  }
  cleanupOrphanCardButtons(cards);
}

function biliCardRoot(link) {
  const named = link.closest(BILI_CARD_SELECTORS);
  if (named) return named;
  const parent = link.parentElement;
  if (!parent || parent === document.body || parent === document.documentElement) return link;
  return parent;
}

function cardRoot(link) {
  const named = link.closest("section.note-item, .note-item");
  if (named) return named;
  const parent = link.parentElement;
  if (!parent || parent === document.body || parent === document.documentElement) return link;
  const siblings = [...parent.querySelectorAll("a[href]")].filter(
    (item) => item.querySelector("img") && mdpParseXiaohongshuNote(item.href, location.href)
  );
  if (siblings.length <= 1) return parent;
  return link;
}

function mountNoteCover(card, link, tokens) {
  const parsed = mdpParseXiaohongshuNote(link.href, location.href);
  if (!parsed) return "";
  const fromState = tokens[parsed.noteId] || xhsTokenMap[parsed.noteId] || {};
  const token = parsed.token || fromState.token || "";
  const source = parsed.source || fromState.source || (token ? "pc_feed" : "");
  const url = mdpXiaohongshuNoteUrl(parsed.noteId, token, source, parsed.origin || mdpXiaohongshuOrigin(location.href));
  mountCardButton(card, url, parsed.noteId, { xhs: true });
  return parsed.noteId;
}

function collectXhsTokens() {
  const scripts = [...document.querySelectorAll("script")].map((node) => node.textContent || "");
  const collected = mdpCollectXhsTokens(window.__INITIAL_STATE__, scripts);
  Object.assign(xhsTokenMap, collected);
  return xhsTokenMap;
}

function bindXhsTokenMessages() {
  window.addEventListener("message", (event) => {
    if (event.source !== window) return;
    const data = event.data;
    if (!data || data.source !== "mdp-xhs" || data.type !== "TOKENS") return;
    for (const [id, row] of Object.entries(data.tokens || {})) {
      if (row && row.token) xhsTokenMap[id] = { token: String(row.token), source: String(row.source || "") };
    }
  });
}

function injectPageBridge() {
  if (!onXhsSurface() || document.getElementById("mdp-page-bridge")) return;
  if (!chrome.runtime || typeof chrome.runtime.getURL !== "function") return;
  const script = document.createElement("script");
  script.id = "mdp-page-bridge";
  script.src = chrome.runtime.getURL("page-bridge.js");
  (document.documentElement || document.head).appendChild(script);
}

function resolveCardUrl(host) {
  const noteId = host.dataset.mdpNoteId;
  const card = host.parentElement;
  const fromState = (noteId && xhsTokenMap[noteId]) || {};
  let token = fromState.token || "";
  let source = fromState.source || "";
  let origin = "";
  if (card) {
    for (const link of card.querySelectorAll("a[href]")) {
      const parsed = mdpParseXiaohongshuNote(link.href, location.href);
      if (parsed && parsed.noteId === noteId) {
        if (parsed.origin) origin = parsed.origin;
        if (parsed.token) {
          token = parsed.token;
          source = parsed.source || source;
        }
      }
    }
  }
  if (!token && host.dataset.mdpUrl) {
    const stored = mdpParseXiaohongshuNote(host.dataset.mdpUrl, location.href);
    if (stored && stored.token) {
      token = stored.token;
      source = stored.source || source;
    }
    if (stored && stored.origin) origin = origin || stored.origin;
  }
  origin = origin || mdpXiaohongshuOrigin(location.href);
  return {
    noteId,
    token,
    url: noteId ? mdpXiaohongshuNoteUrl(noteId, token, source || (token ? "pc_feed" : ""), origin) : "",
  };
}

function extractHostFromEvent(event) {
  const path = typeof event.composedPath === "function" ? event.composedPath() : [];
  for (const node of path) {
    if (node && node.classList && node.classList.contains("mdp-card-extract")) return node;
  }
  const target = event.target;
  if (target && typeof target.closest === "function") return target.closest(".mdp-card-extract");
  return null;
}

function bindExtractClicks() {
  if (extractClicksBound) return;
  extractClicksBound = true;
  const onClick = (event) => {
    const host = extractHostFromEvent(event);
    if (!host) return;
    event.preventDefault();
    event.stopPropagation();
    event.stopImmediatePropagation();
    const button = host.shadowRoot && host.shadowRoot.querySelector("button");
    if (button) queueCard(host, button);
  };
  const keepClick = (event) => {
    if (!extractHostFromEvent(event)) return;
    event.stopPropagation();
    event.stopImmediatePropagation();
  };
  // Window capture runs before Vue listeners on the card. Stop bubbling so the
  // feed does not open the note, but do not preventDefault on pointerdown or
  // the following click never fires.
  window.addEventListener("click", onClick, true);
  window.addEventListener("auxclick", onClick, true);
  window.addEventListener("pointerdown", keepClick, true);
  window.addEventListener("mousedown", keepClick, true);
}

function mountCardButton(card, url, cardId, options = {}) {
  let host = [...card.children].find((node) => node.classList && node.classList.contains("mdp-card-extract"));
  if (!host) {
    host = document.createElement("div");
    host.className = "mdp-card-extract";
    const shadow = host.attachShadow({ mode: "open" });
    shadow.innerHTML = `<style>${CARD_STYLE}</style><button type="button" title="Extract this post">✨ Extract</button>`;
    host.addEventListener("click", (event) => {
      event.preventDefault();
      event.stopPropagation();
      const button = host.shadowRoot && host.shadowRoot.querySelector("button");
      if (button) queueCard(host, button);
    });
    if (getComputedStyle(card).position === "static") card.classList.add("mdp-extract-host");
    card.appendChild(host);
  }
  host.dataset.mdpCardId = cardId;
  host.dataset.mdpNoteId = options.xhs ? cardId : "";
  host.dataset.mdpXhs = options.xhs ? "1" : "";
  host.dataset.mdpUrl = url;
}

async function queueCard(host, button) {
  let url = host.dataset.mdpUrl || "";
  if (host.dataset.mdpXhs === "1") {
    collectXhsTokens();
    const resolved = resolveCardUrl(host);
    url = resolved.url;
    if (!resolved.token) {
      button.className = "err";
      button.textContent = "Retry";
      button.title = "Missing xsec_token on this card. Scroll it fully into view and try again.";
      return;
    }
  }
  if (!url || !mdpIsSupportedVideoUrl(url) || button.disabled || host.dataset.mdpBusy === "1") return;
  host.dataset.mdpUrl = url;
  host.dataset.mdpBusy = "1";
  const idle = button.textContent;
  button.disabled = true;
  button.className = "";
  button.textContent = "…";
  try {
    const response = await mdpSend({ type: "CREATE_TASK", url });
    if (!response.ok) {
      button.className = "err";
      button.textContent = "Retry";
      button.title = response.error || "Could not add the task.";
      button.disabled = false;
      host.dataset.mdpBusy = "";
      return;
    }
    button.className = "ok";
    button.textContent = "Queued";
    button.title = "Added to queue";
    window.setTimeout(() => {
      button.className = "";
      button.textContent = idle;
      button.title = "Extract this post";
      button.disabled = false;
      host.dataset.mdpBusy = "";
    }, 2500);
  } catch (error) {
    button.className = "err";
    button.textContent = "Retry";
    button.title = error.message || String(error);
    button.disabled = false;
    host.dataset.mdpBusy = "";
  }
}
