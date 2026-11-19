// Clicking the toolbar icon sends the current page straight away: no model or keyframe choices.
// The server picks the ASR (asr.default, Fun-ASR) and skips keyframes.
const pageEl = document.getElementById("page");
const statusEl = document.getElementById("status");
const retryEl = document.getElementById("retry");
const tasksEl = document.getElementById("tasks");

let currentUrl = "";

init();

async function init() {
  retryEl.addEventListener("click", send);
  const tab = await mdpGetActiveTab();
  currentUrl = tab?.url || "";
  pageEl.textContent = tab?.title || currentUrl;
  if (!currentUrl) {
    show("err", "读不到当前标签页的地址。重新加载插件后再试。");
  } else if (!mdpIsSupportedVideoUrl(currentUrl)) {
    show("err", mdpIsXiaohongshuHost(currentUrl)
      ? "小红书首页：点帖子卡片上的 ✨ Extract，或打开帖子后再点插件。"
      : "这个页面不支持（支持 YouTube、B站、小红书、Threads、LinkedIn、X、小宇宙）。");
  } else {
    await send();
  }
  loadTasks();
}

async function send() {
  retryEl.hidden = true;
  show("", "发送中…");
  const response = await mdpSend({ type: "CREATE_TASK", url: currentUrl });
  if (!response.ok) {
    show("err", response.error || "发送失败。");
    retryEl.hidden = false;
    return;
  }
  show("ok", "✓ 已加入队列");
  loadTasks();
}

function show(kind, text) {
  statusEl.className = `status ${kind}`;
  statusEl.textContent = text;
}

async function loadTasks() {
  const listing = await mdpSend({ type: "TASKS" });
  if (!listing.ok) {
    tasksEl.innerHTML = `<li class="meta">${mdpEscapeHtml(listing.error || "服务没有响应")}</li>`;
    return;
  }
  const tasks = (listing.tasks || []).slice(0, 6);
  tasksEl.innerHTML = tasks.length
    ? tasks.map((task) => `<li>
        <div class="title">${mdpEscapeHtml(task.title || task.video_id || task.url)}</div>
        <div class="meta">${mdpEscapeHtml(task.status)}${task.error ? ` · ${mdpEscapeHtml(task.error)}` : ""}</div>
      </li>`).join("")
    : `<li class="meta">还没有任务。</li>`;
}
