const STORAGE_KEY = "thaillmm-document-chats-v2";
const LEGACY_STORAGE_KEY = "thaillmm-document-chat-v1";
const MAX_STORED_MESSAGES = 40;
const MAX_API_HISTORY = 20;
const MAX_SAVED_CHATS = 30;
const SETTINGS_KEY = "thaillmm-settings-v1";
const DEFAULT_SETTINGS = Object.freeze({
  theme: "system", density: "comfortable", showConfidence: true,
  openSources: false, enterToSend: true,
  fontSize: "default", autoScroll: true, showSuggestions: true,
  answerStyle: "concise", answerLanguage: "auto", retrievalDepth: 5,
  evidenceMode: "balanced", contextLimit: 12,
});
const SETTING_CHOICES = {
  theme: ["system", "light", "dark"], density: ["comfortable", "compact"],
  fontSize: ["small", "default", "large"], answerStyle: ["concise", "balanced", "detailed"],
  answerLanguage: ["auto", "th", "en", "zh"], retrievalDepth: [3, 5, 8],
  evidenceMode: ["balanced", "strict"], contextLimit: [0, 6, 12, 20],
};

const form = document.querySelector("#question-form");
const questionInput = document.querySelector("#question");
const askButton = document.querySelector("#ask-button");
const newChatButton = document.querySelector("#new-chat-button");
const reindexButton = document.querySelector("#reindex-button");
const messagesBox = document.querySelector("#messages");
const emptyState = document.querySelector("#empty-state");
const conversationStage = document.querySelector("#conversation-stage");
const chatList = document.querySelector("#chat-list");
const loading = document.querySelector("#loading");
const errorBox = document.querySelector("#error");
const statusBadge = document.querySelector("#status");
const sidebar = document.querySelector("#sidebar");
const sidebarBackdrop = document.querySelector("#sidebar-backdrop");
const openSidebarButton = document.querySelector("#open-sidebar-button");
const closeSidebarButton = document.querySelector("#close-sidebar-button");

const settingsDialog = document.querySelector("#settings-dialog");
const themeSelect = document.querySelector("#theme-select");
const densitySelect = document.querySelector("#density-select");
const confidenceToggle = document.querySelector("#show-confidence-toggle");
const sourcesToggle = document.querySelector("#open-sources-toggle");
const enterToggle = document.querySelector("#enter-to-send-toggle");
const settingControls = [
  [themeSelect, "theme"], [densitySelect, "density"], [confidenceToggle, "showConfidence"],
  [sourcesToggle, "openSources"], [enterToggle, "enterToSend"],
  ...[["font-size-select", "fontSize"], ["auto-scroll-toggle", "autoScroll"],
    ["suggestions-toggle", "showSuggestions"], ["answer-style-select", "answerStyle"],
    ["answer-language-select", "answerLanguage"], ["retrieval-depth-select", "retrievalDepth"],
    ["evidence-mode-select", "evidenceMode"], ["context-limit-select", "contextLimit"]]
    .map(([id, key]) => [document.querySelector(`#${id}`), key]),
];
const clearChatsButton = document.querySelector("#clear-all-chats-button");
const deviceTheme = window.matchMedia("(prefers-color-scheme: dark)");
let userSettings = loadSettings();
let chatState = loadChatState();
let busy = false;

function loadSettings() {
  try {
    const saved = JSON.parse(localStorage.getItem(SETTINGS_KEY) || "null");
    if (!saved || typeof saved !== "object") return { ...DEFAULT_SETTINGS };
    return Object.fromEntries(Object.entries(DEFAULT_SETTINGS).map(([key, fallback]) => {
      const valid = SETTING_CHOICES[key] ? SETTING_CHOICES[key].includes(saved[key])
        : typeof saved[key] === "boolean";
      return [key, valid ? saved[key] : fallback];
    }));
  } catch { return { ...DEFAULT_SETTINGS }; }
}

function applySettings() {
  document.documentElement.dataset.theme = userSettings.theme === "system"
    ? (deviceTheme.matches ? "dark" : "light") : userSettings.theme;
  document.documentElement.dataset.density = userSettings.density;
  document.documentElement.dataset.fontSize = userSettings.fontSize;
  document.body.classList.toggle("hide-suggestions", !userSettings.showSuggestions);
  document.body.classList.toggle("hide-confidence", !userSettings.showConfidence);
  document.querySelectorAll(".source-details").forEach((details) => {
    details.open = userSettings.openSources;
  });
  for (const [control, key] of settingControls) {
    if (control.type === "checkbox") control.checked = userSettings[key];
    else control.value = String(userSettings[key]);
  }
}

function saveSettings() {
  applySettings();
  try { localStorage.setItem(SETTINGS_KEY, JSON.stringify(userSettings)); }
  catch { showError("Preferences apply for this session, but browser storage is unavailable."); }
}

function makeId() {
  if (window.crypto?.randomUUID) return window.crypto.randomUUID();
  return `chat-${Date.now()}-${Math.random().toString(16).slice(2)}`;
}

function cleanAnswer(text) {
  return String(text || "").replace(/\s*\[\s*SOURCE\s+\d+\s*\]/gi, "").trim();
}

function validMessages(messages) {
  if (!Array.isArray(messages)) return [];
  return messages
    .filter((item) => item && ["user", "assistant"].includes(item.role) && typeof item.content === "string")
    .slice(-MAX_STORED_MESSAGES)
    .map((item) => ({ ...item, content: cleanAnswer(item.content) }));
}

function createChat(messages = []) {
  const now = new Date().toISOString();
  const firstQuestion = messages.find((message) => message.role === "user")?.content;
  return {
    id: makeId(),
    title: firstQuestion ? chatTitle(firstQuestion) : "New conversation",
    createdAt: now,
    updatedAt: now,
    messages: validMessages(messages),
  };
}

function normalizeChat(chat) {
  if (!chat || typeof chat !== "object") return null;
  const messages = validMessages(chat.messages);
  const fallbackTitle = messages.find((message) => message.role === "user")?.content || "New conversation";
  return {
    id: typeof chat.id === "string" ? chat.id : makeId(),
    title: typeof chat.title === "string" && chat.title.trim() ? chat.title.trim().slice(0, 56) : chatTitle(fallbackTitle),
    createdAt: chat.createdAt || new Date().toISOString(),
    updatedAt: chat.updatedAt || new Date().toISOString(),
    messages,
  };
}

function loadChatState() {
  try {
    const saved = JSON.parse(localStorage.getItem(STORAGE_KEY) || "null");
    if (saved && Array.isArray(saved.chats)) {
      const chats = saved.chats.map(normalizeChat).filter(Boolean).slice(0, MAX_SAVED_CHATS);
      if (chats.length) {
        const activeChatId = chats.some((chat) => chat.id === saved.activeChatId) ? saved.activeChatId : chats[0].id;
        return { activeChatId, chats };
      }
    }
  } catch {
    // Continue to legacy migration or a fresh chat.
  }

  try {
    const legacyMessages = validMessages(JSON.parse(localStorage.getItem(LEGACY_STORAGE_KEY) || "[]"));
    if (legacyMessages.length) {
      const migrated = createChat(legacyMessages);
      return { activeChatId: migrated.id, chats: [migrated] };
    }
  } catch {
    // Start fresh below.
  }

  const fresh = createChat();
  return { activeChatId: fresh.id, chats: [fresh] };
}

function saveChatState() {
  chatState.chats = chatState.chats
    .map((chat) => {
      chat.messages = chat.messages.slice(-MAX_STORED_MESSAGES);
      return chat;
    })
    .sort((a, b) => String(b.updatedAt).localeCompare(String(a.updatedAt)))
    .slice(0, MAX_SAVED_CHATS);
  try {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(chatState));
    localStorage.removeItem(LEGACY_STORAGE_KEY);
  } catch { showError("Chat remains available now, but browser storage could not save it."); }
}

function activeChat() {
  let chat = chatState.chats.find((item) => item.id === chatState.activeChatId);
  if (!chat) {
    chat = createChat();
    chatState.chats.unshift(chat);
    chatState.activeChatId = chat.id;
  }
  return chat;
}

function chatTitle(text) {
  const compact = String(text || "New conversation").replace(/\s+/g, " ").trim();
  return compact.length > 42 ? `${compact.slice(0, 42)}…` : compact;
}

function apiHistory(messages) {
  const limit = Math.min(MAX_API_HISTORY, userSettings.contextLimit);
  return limit ? messages.slice(-limit).map(({ role, content }) => ({ role, content })) : [];
}

function requestOptions() {
  return { top_k: userSettings.retrievalDepth, evidence_mode: userSettings.evidenceMode,
    history_messages: userSettings.contextLimit, answer_style: userSettings.answerStyle,
    answer_language: userSettings.answerLanguage };
}

function selectSettingsPage(page) {
  document.querySelectorAll("[data-settings-page]").forEach(button => {
    button.setAttribute("aria-pressed", String(button.dataset.settingsPage === page));
  });
  document.querySelectorAll("[data-settings-panel]").forEach(panel => {
    panel.hidden = panel.dataset.settingsPanel !== page;
  });
  document.querySelector(".settings-content").scrollTop = 0;
}

function buildChatExport(chats) {
  return { format: "thailmm-chat-export", version: 1, exportedAt: new Date().toISOString(),
    chats: chats.map(chat => ({ title: chat.title, createdAt: chat.createdAt, updatedAt: chat.updatedAt,
      messages: chat.messages.map(message => ({ role: message.role, content: message.content,
        sources: Array.isArray(message.sources) ? message.sources.map(source => ({
          document: source.document, page: source.page,
        })) : [],
      })),
    })),
  };
}

function exportChats(all) {
  try {
    const blob = new Blob([JSON.stringify(buildChatExport(all ? chatState.chats : [activeChat()]), null, 2)],
      { type: "application/json;charset=utf-8" });
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    link.download = `thailmm-${all ? "all-chats" : "chat"}-${new Date().toISOString().slice(0, 10)}.json`;
    document.body.append(link);
    link.click();
    link.remove();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  } catch { showError("Could not export conversations. Please try again in your browser."); }
}

function setBusy(isBusy) {
  busy = isBusy;
  clearChatsButton.disabled = isBusy;
  loading.hidden = !isBusy;
  askButton.disabled = isBusy;
  newChatButton.disabled = isBusy;
  reindexButton.disabled = isBusy;
  questionInput.disabled = isBusy;
}

function showError(message) {
  errorBox.textContent = message;
  errorBox.hidden = false;
}

function clearError() {
  errorBox.textContent = "";
  errorBox.hidden = true;
}

async function parseResponse(response) {
  let payload;
  try {
    payload = await response.json();
  } catch {
    throw new Error(`Server returned an unreadable response (${response.status}).`);
  }
  if (!response.ok) throw new Error(payload.detail || `Request failed (${response.status}).`);
  return payload;
}

function renderSources(sources) {
  const details = document.createElement("details");
  details.className = "source-details";
  details.open = userSettings.openSources;
  const summary = document.createElement("summary");
  summary.textContent = `Sources · ${sources.length}`;
  const list = document.createElement("div");
  list.className = "message-sources";
  sources.forEach((source) => {
    const chip = document.createElement("span");
    chip.className = "source-chip";
    chip.textContent = source.page == null ? `▣ ${source.document}` : `▣ ${source.document} · p.${source.page}`;
    list.append(chip);
  });
  details.append(summary, list);
  return details;
}

function renderChatList() {
  chatList.replaceChildren();
  const chats = [...chatState.chats].sort((a, b) => String(b.updatedAt).localeCompare(String(a.updatedAt)));
  chats.forEach((chat) => {
    const row = document.createElement("div");
    row.className = `chat-history-row${chat.id === chatState.activeChatId ? " active" : ""}`;
    const open = document.createElement("button");
    open.type = "button";
    open.className = "chat-history-open";
    open.dataset.chatId = chat.id;
    open.textContent = chat.title;
    open.title = chat.title;
    const remove = document.createElement("button");
    remove.type = "button";
    remove.className = "chat-history-delete";
    remove.dataset.deleteChatId = chat.id;
    remove.setAttribute("aria-label", `Delete ${chat.title}`);
    remove.textContent = "×";
    row.append(open, remove);
    chatList.append(row);
  });
}

function renderConversation() {
  const messages = activeChat().messages;
  messagesBox.replaceChildren();
  emptyState.hidden = messages.length > 0;
  messagesBox.hidden = messages.length === 0;

  messages.forEach((message) => {
    const row = document.createElement("article");
    row.className = `message-row ${message.role}`;
    const copy = document.createElement("div");
    copy.className = "message-copy";
    copy.textContent = cleanAnswer(message.content);

    if (message.role === "user") {
      const bubble = document.createElement("div");
      bubble.className = "message-bubble";
      bubble.append(copy);
      row.append(bubble);
    } else {
      const avatar = document.createElement("div");
      avatar.className = "assistant-avatar";
      avatar.setAttribute("aria-hidden", "true");
      avatar.textContent = "T";
      const content = document.createElement("div");
      content.className = "message-content";
      const name = document.createElement("p");
      name.className = "message-name";
      name.textContent = "ThaiLLM";
      content.append(name, copy);
      const confidence = Math.round((message.retrieval_confidence || 0) * 100);
      const meta = document.createElement("div");
      meta.className = "answer-meta";
      const confidenceChip = document.createElement("span");
      confidenceChip.className = `confidence-chip${confidence < 50 ? " low" : ""}`;
      confidenceChip.textContent = `${confidence}% retrieval match`;
      meta.append(confidenceChip);
      content.append(meta);
      if (Array.isArray(message.sources) && message.sources.length) content.append(renderSources(message.sources));
      row.append(avatar, content);
    }
    messagesBox.append(row);
  });

  renderChatList();
  if (userSettings.autoScroll) {
    requestAnimationFrame(() => { conversationStage.scrollTop = conversationStage.scrollHeight; });
  }
}

function autoResizeComposer() {
  questionInput.style.height = "auto";
  questionInput.style.height = `${Math.min(questionInput.scrollHeight, 180)}px`;
}

function openSidebar() {
  sidebar.classList.add("open");
  sidebarBackdrop.hidden = false;
}

function closeSidebar() {
  sidebar.classList.remove("open");
  sidebarBackdrop.hidden = true;
}

function beginNewChat() {
  const current = activeChat();
  if (current.messages.length === 0) {
    questionInput.focus();
    closeSidebar();
    return;
  }
  const fresh = createChat();
  chatState.chats.unshift(fresh);
  chatState.activeChatId = fresh.id;
  saveChatState();
  clearError();
  renderConversation();
  closeSidebar();
  questionInput.focus();
}

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  if (busy) return;
  const question = questionInput.value.trim();
  if (!question) {
    showError("กรุณาป้อนคำถาม");
    return;
  }

  const chat = activeChat();
  const history = apiHistory(chat.messages);
  const options = requestOptions();
  chat.messages.push({ role: "user", content: question });
  if (chat.messages.filter((message) => message.role === "user").length === 1) chat.title = chatTitle(question);
  chat.updatedAt = new Date().toISOString();
  saveChatState();
  renderConversation();
  questionInput.value = "";
  autoResizeComposer();
  clearError();
  setBusy(true);

  try {
    const response = await fetch("/api/ask", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question, history, options }),
    });
    const payload = await parseResponse(response);
    chat.messages.push({
      role: "assistant",
      content: cleanAnswer(payload.answer),
      sources: payload.sources,
      grounded: payload.grounded,
      retrieval_confidence: payload.retrieval_confidence,
    });
    chat.updatedAt = new Date().toISOString();
    saveChatState();
    renderConversation();
  } catch (error) {
    showError(error.message);
  } finally {
    setBusy(false);
    questionInput.focus();
  }
});

questionInput.addEventListener("input", autoResizeComposer);
questionInput.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey && !event.isComposing && userSettings.enterToSend) {
    event.preventDefault();
    form.requestSubmit();
  }
});

document.querySelectorAll(".suggestion-card").forEach((button) => {
  button.addEventListener("click", () => {
    questionInput.value = button.dataset.question || "";
    autoResizeComposer();
    questionInput.focus();
  });
});

chatList.addEventListener("click", (event) => {
  const openButton = event.target.closest("[data-chat-id]");
  if (openButton) {
    chatState.activeChatId = openButton.dataset.chatId;
    saveChatState();
    clearError();
    renderConversation();
    closeSidebar();
    return;
  }

  const deleteButton = event.target.closest("[data-delete-chat-id]");
  if (!deleteButton || busy) return;
  const chat = chatState.chats.find((item) => item.id === deleteButton.dataset.deleteChatId);
  if (!chat || !window.confirm(`Delete “${chat.title}”?`)) return;
  chatState.chats = chatState.chats.filter((item) => item.id !== chat.id);
  if (!chatState.chats.length) chatState.chats.push(createChat());
  if (chatState.activeChatId === chat.id) chatState.activeChatId = chatState.chats[0].id;
  saveChatState();
  renderConversation();
});

newChatButton.addEventListener("click", beginNewChat);
openSidebarButton.addEventListener("click", openSidebar);
closeSidebarButton.addEventListener("click", closeSidebar);
sidebarBackdrop.addEventListener("click", closeSidebar);

reindexButton.addEventListener("click", async () => {
  clearError();
  setBusy(true);
  const label = reindexButton.lastElementChild;
  const originalLabel = label.textContent;
  label.textContent = "Indexing…";
  try {
    const response = await fetch("/admin/reindex", { method: "POST" });
    const payload = await parseResponse(response);
    statusBadge.textContent = `${payload.chunk_count} chunks ready`;
    statusBadge.className = "status ready";
  } catch (error) {
    showError(error.message);
  } finally {
    label.textContent = originalLabel;
    setBusy(false);
  }
});

async function loadStatus() {
  const connectionStatus = document.querySelector("#settings-connection-status");
  connectionStatus.textContent = "Checking configuration…";
  try {
    const response = await fetch("/api/status");
    const payload = await parseResponse(response);
    if (!payload.thailmm_configured) {
      statusBadge.textContent = "ThaiLLM setup needed";
      statusBadge.className = "status warning";
    } else if (!payload.index_exists) {
      statusBadge.textContent = "Document index needed";
      statusBadge.className = "status warning";
    } else {
      statusBadge.textContent = "System ready";
      statusBadge.className = "status ready";
    }
  } catch {
    statusBadge.textContent = "System unavailable";
    statusBadge.className = "status error-status";
  }
  connectionStatus.textContent = `${statusBadge.textContent}. Configuration check only; this does not verify a live model response.`;
}

document.querySelector("#settings-button").addEventListener("click", () => {
  closeSidebar();
  applySettings();
  settingsDialog.showModal();
});
for (const id of ["#close-settings-button", "#done-settings-button"]) {
  document.querySelector(id).addEventListener("click", () => settingsDialog.close());
}
settingsDialog.addEventListener("click", (event) => {
  if (event.target === settingsDialog) {
    const bounds = settingsDialog.getBoundingClientRect();
    if (event.clientX < bounds.left || event.clientX > bounds.right ||
        event.clientY < bounds.top || event.clientY > bounds.bottom) settingsDialog.close();
  }
});
for (const [control, key] of settingControls) {
  control.addEventListener("change", () => {
    const value = control.type === "checkbox" ? control.checked
      : typeof DEFAULT_SETTINGS[key] === "number" ? Number(control.value) : control.value;
    userSettings[key] = SETTING_CHOICES[key] && !SETTING_CHOICES[key].includes(value)
      ? DEFAULT_SETTINGS[key] : value;
    saveSettings();
  });
}
document.querySelectorAll("[data-settings-page]").forEach(button => {
  button.addEventListener("click", () => selectSettingsPage(button.dataset.settingsPage));
});
document.querySelector("#export-chat-button").addEventListener("click", () => exportChats(false));
document.querySelector("#export-all-button").addEventListener("click", () => exportChats(true));
document.querySelector("#refresh-status-button").addEventListener("click", loadStatus);
deviceTheme.addEventListener("change", () => {
  if (userSettings.theme === "system") applySettings();
});
document.querySelector("#reset-settings-button").addEventListener("click", () => {
  userSettings = { ...DEFAULT_SETTINGS };
  saveSettings();
});
clearChatsButton.addEventListener("click", () => {
  if (busy || !window.confirm("Delete all conversations saved in this browser? This cannot be undone. Documents and settings will not be deleted.")) return;
  const fresh = createChat();
  chatState = { activeChatId: fresh.id, chats: [fresh] };
  saveChatState();
  questionInput.value = "";
  autoResizeComposer();
  renderConversation();
  settingsDialog.close();
});

applySettings();
saveChatState();
renderConversation();
autoResizeComposer();
loadStatus();
