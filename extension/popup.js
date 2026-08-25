const healthEl = document.getElementById("health");
const tasksEl = document.getElementById("tasks");

init();

async function init() {
  const health = await send({ type: "HEALTH" });
  if (!health.ok) {
    healthEl.className = "sub err";
    healthEl.textContent = health.error || "Local service is not running.";
    return;
  }
  healthEl.className = "sub ok";
  healthEl.textContent = "Service running on 127.0.0.1";
  const listing = await send({ type: "TASKS" });
  if (!listing.ok) {
    tasksEl.innerHTML = `<li class="meta">${escapeHtml(listing.error || "Could not load tasks")}</li>`;
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
        <div class="title">${escapeHtml(title)}</div>
        <div class="meta">${escapeHtml(task.status)} · ${escapeHtml(task.asr_label || task.asr_model)}${escapeHtml(extra)}</div>
      </li>`;
    })
    .join("");
}

function send(message) {
  return new Promise((resolve) => {
    chrome.runtime.sendMessage(message, (response) => {
      if (chrome.runtime.lastError) {
        resolve({ ok: false, error: chrome.runtime.lastError.message });
        return;
      }
      resolve(response || { ok: false, error: "No response" });
    });
  });
}

function escapeHtml(value) {
  return String(value || "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;");
}
