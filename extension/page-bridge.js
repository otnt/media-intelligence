(() => {
  const SOURCE = "mdp-xhs";
  let last = "";

  function collect() {
    const map = {};
    const seen = new Set();
    const walk = (value, depth) => {
      if (!value || depth > 10 || typeof value !== "object") return;
      if (seen.has(value)) return;
      seen.add(value);
      const id =
        value.noteId ||
        value.note_id ||
        (value.noteCard && value.noteCard.noteId) ||
        ((value.modelType === "note" || value.noteCard) && value.id);
      const token =
        value.xsecToken ||
        value.xsec_token ||
        (value.noteCard && (value.noteCard.xsecToken || value.noteCard.xsec_token));
      const source = value.xsecSource || value.xsec_source || "";
      if (id && token && /^[0-9a-f]{24}$/i.test(String(id))) {
        map[String(id)] = { token: String(token), source: String(source || "") };
      }
      const children = Array.isArray(value) ? value : Object.values(value);
      for (const child of children) walk(child, depth + 1);
    };
    walk(window.__INITIAL_STATE__, 0);
    try {
      const app = document.querySelector("#app");
      const pinia = app && app.__vue_app__ && app.__vue_app__.config.globalProperties.$pinia;
      const state = pinia && (pinia.state.value || pinia.state);
      walk(state, 0);
    } catch (_error) {}
    const serialized = JSON.stringify(map);
    if (serialized === last) return;
    last = serialized;
    window.postMessage({ source: SOURCE, type: "TOKENS", tokens: map }, "*");
  }

  collect();
  window.addEventListener("load", collect);
  window.setInterval(collect, 1000);
})();
