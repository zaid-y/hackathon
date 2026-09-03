const STORAGE_KEY = "thaillmm-document-chat-v1";
const MAX_STORED_MESSAGES = 40;
const MAX_API_HISTORY = 20;

const form = document.querySelector("#question-form");
const questionInput = document.querySelector("#question");
const askButton = document.querySelector("#ask-button");
const newChatButton = document.querySelector("#new-chat-button");
const reindexButton = document.querySelector("#reindex-button");
const messagesBox = document.querySelector("#messages");
const loading = document.querySelector("#loading");
const errorBox = document.querySelector("#error");
const statusBadge = document.querySelector("#status");

let conversation = loadConversation();

function cleanAnswer(text) {
  return String(text || "")
    .replace(/\s*\[\s*SOURCE\s+\d+\s*\]/gi, "")
    .trim();
}

function loadConversation() {
  try {
    const saved = JSON.parse(localStorage.getItem(STORAGE_KEY) || "[]");
    if (!Array.isArray(saved)) return [];
    return saved
      .filter((item) =>
        item &&
        ["user", "assistant"].includes(item.role) &&
        typeof item.content === "string"
      )
      .slice(-MAX_STORED_MESSAGES)
      .map((item) => ({ ...item, content: cleanAnswer(item.content) }));
  } catch {
    return [];
  }
}

function saveConversation() {
  conversation = conversation.slice(-MAX_STORED_MESSAGES);
  localStorage.setItem(STORAGE_KEY, JSON.stringify(conversation));
}

function apiHistory(messages) {
  return messages.slice(-MAX_API_HISTORY).map(({ role, content }) => ({
    role,
    content,
  }));
}

function setBusy(isBusy) {
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
  if (!response.ok) {
    throw new Error(payload.detail || `Request failed (${response.status}).`);
  }
  return payload;
}

function renderSources(sources) {
  const list = document.createElement("div");
  list.className = "message-sources";
  sources.forEach((source) => {
    const chip = document.createElement("span");
    chip.className = "source-chip";
    chip.textContent = source.page == null
      ? `📄 ${source.document}`
      : `📄 ${source.document} · p.${source.page}`;
    list.append(chip);
  });
  return list;
}

function renderConversation() {
  messagesBox.replaceChildren();
  if (!conversation.length) {
    const welcome = document.createElement("div");
    welcome.className = "empty-chat";
    const title = document.createElement("strong");
    title.textContent = "เริ่มบทสนทนากับเอกสาร";
    const copy = document.createElement("span");
    copy.textContent = "ถามคำถามแรก แล้วถามต่อได้โดยไม่ต้องอธิบายเรื่องเดิมซ้ำ";
    welcome.append(title, copy);
    messagesBox.append(welcome);
    return;
  }

  conversation.forEach((message) => {
    const bubble = document.createElement("article");
    bubble.className = `message ${message.role}`;

    const label = document.createElement("span");
    label.className = "message-label";
    label.textContent = message.role === "user" ? "You" : "ThaiLLM";

    const copy = document.createElement("div");
    copy.className = "message-copy";
    copy.textContent = cleanAnswer(message.content);
    bubble.append(label, copy);

    if (message.role === "assistant") {
      const meta = document.createElement("div");
      meta.className = "message-meta";
      meta.textContent = `${Math.round((message.retrieval_confidence || 0) * 100)}% retrieval`;
      bubble.append(meta);
      if (Array.isArray(message.sources) && message.sources.length) {
        bubble.append(renderSources(message.sources));
      }
    }
    messagesBox.append(bubble);
  });
  messagesBox.scrollTop = messagesBox.scrollHeight;
}

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  const question = questionInput.value.trim();
  if (!question) {
    showError("กรุณาป้อนคำถาม");
    return;
  }

  const history = apiHistory(conversation);
  conversation.push({ role: "user", content: question });
  saveConversation();
  renderConversation();
  questionInput.value = "";
  clearError();
  setBusy(true);

  try {
    const response = await fetch("/api/ask", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question, history }),
    });
    const payload = await parseResponse(response);
    conversation.push({
      role: "assistant",
      content: cleanAnswer(payload.answer),
      sources: payload.sources,
      grounded: payload.grounded,
      retrieval_confidence: payload.retrieval_confidence,
    });
    saveConversation();
    renderConversation();
  } catch (error) {
    showError(error.message);
  } finally {
    setBusy(false);
    questionInput.focus();
  }
});

questionInput.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey) {
    event.preventDefault();
    form.requestSubmit();
  }
});

newChatButton.addEventListener("click", () => {
  if (conversation.length && !window.confirm("เริ่มแชตใหม่และล้างบทสนทนาที่บันทึกในเบราว์เซอร์นี้?")) {
    return;
  }
  conversation = [];
  localStorage.removeItem(STORAGE_KEY);
  clearError();
  renderConversation();
  questionInput.focus();
});

reindexButton.addEventListener("click", async () => {
  clearError();
  setBusy(true);
  reindexButton.textContent = "Indexing…";
  try {
    const response = await fetch("/admin/reindex", { method: "POST" });
    const payload = await parseResponse(response);
    statusBadge.textContent = `${payload.chunk_count} chunks ready`;
    statusBadge.className = "status ready";
  } catch (error) {
    showError(error.message);
  } finally {
    reindexButton.textContent = "Re-index documents";
    setBusy(false);
  }
});

async function loadStatus() {
  try {
    const response = await fetch("/api/status");
    const payload = await parseResponse(response);
    if (!payload.thailmm_configured) {
      statusBadge.textContent = "ThaiLLM setup needed";
      statusBadge.className = "status warning";
    } else if (!payload.index_exists) {
      statusBadge.textContent = "Index needed";
      statusBadge.className = "status warning";
    } else {
      statusBadge.textContent = "System ready";
      statusBadge.className = "status ready";
    }
  } catch {
    statusBadge.textContent = "System unavailable";
    statusBadge.className = "status error-status";
  }
}

renderConversation();
loadStatus();
