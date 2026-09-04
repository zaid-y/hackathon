/**
 * ThaiLLM Document Assistant - Frontend Application
 * Competition-ready UI for hackathon demo
 */

// Configuration
const API_BASE = '/api';
const WS_BASE = '/ws';

// State
let currentMode = 'enhanced';
let questionHistory = [];
let isProcessing = false;
let websocket = null;

// DOM Elements
const elements = {
    healthDot: null,
    healthText: null,
    docCount: null,
    docCountText: null,
    modeBadge: null,
    modeText: null,
    modeSelect: null,
    questionInput: null,
    askBtn: null,
    btnText: null,
    btnLoading: null,
    clearBtn: null,
    exampleChips: null,
    answerSection: null,
    answerContent: null,
    answerMeta: null,
    copyBtn: null,
    sourcesSection: null,
    sourcesCount: null,
    sourcesList: null,
    historySection: null,
    historyList: null,
    clearHistoryBtn: null,
};

// Initialize DOM references
function initElements() {
    elements.healthDot = document.getElementById('health-dot');
    elements.healthText = document.getElementById('health-text');
    elements.docCount = document.getElementById('doc-count');
    elements.docCountText = document.getElementById('doc-count-text');
    elements.modeBadge = document.getElementById('mode-badge');
    elements.modeText = document.getElementById('mode-text');
    elements.modeSelect = document.getElementById('mode-select');
    elements.questionInput = document.getElementById('question-input');
    elements.askBtn = document.getElementById('ask-btn');
    elements.btnText = document.querySelector('.btn-text');
    elements.btnLoading = document.querySelector('.btn-loading');
    elements.clearBtn = document.getElementById('clear-btn');
    elements.exampleChips = document.getElementById('example-chips');
    elements.answerSection = document.getElementById('answer-section');
    elements.answerContent = document.getElementById('answer-content');
    elements.answerMeta = document.getElementById('answer-meta');
    elements.copyBtn = document.getElementById('copy-btn');
    elements.sourcesSection = document.getElementById('sources-section');
    elements.sourcesCount = document.getElementById('sources-count');
    elements.sourcesList = document.getElementById('sources-list');
    elements.historySection = document.getElementById('history-section');
    elements.historyList = document.getElementById('history-list');
    elements.clearHistoryBtn = document.getElementById('clear-history-btn');
}

// Utility functions
function show(element) {
    if (element) element.style.display = '';
}

function hide(element) {
    if (element) element.style.display = 'none';
}

function setLoading(loading) {
    isProcessing = loading;
    if (elements.askBtn) {
        elements.askBtn.disabled = loading || !elements.questionInput?.value?.trim();
    }
    if (elements.btnText) elements.btnText.style.display = loading ? 'none' : '';
    if (elements.btnLoading) elements.btnLoading.style.display = loading ? '' : 'none';
    if (elements.questionInput) elements.questionInput.disabled = loading;
    if (elements.modeSelect) elements.modeSelect.disabled = loading;
}

function formatTime(ms) {
    if (ms < 1000) return `${ms.toFixed(0)}ms`;
    return `${(ms / 1000).toFixed(2)}s`;
}

function getConfidenceClass(confidence) {
    if (confidence >= 0.7) return 'confidence-high';
    if (confidence >= 0.4) return 'confidence-medium';
    return 'confidence-low';
}

function getConfidenceLabel(confidence) {
    if (confidence >= 0.7) return 'High';
    if (confidence >= 0.4) return 'Medium';
    return 'Low';
}

function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

// Health check
async function checkHealth() {
    try {
        const response = await fetch(`${API_BASE}/health`);
        const data = await response.json();

        if (elements.healthDot) {
            elements.healthDot.className = 'status-dot ' + (data.thaillm_api ? 'healthy' : 'degraded');
        }
        if (elements.healthText) {
            elements.healthText.textContent = data.thaillm_api ? 'ThaiLLM Connected' : 'ThaiLLM Disconnected';
        }
        if (elements.docCount && data.documents_count > 0) {
            show(elements.docCount);
            elements.docCountText.textContent = `${data.documents_count} document${data.documents_count !== 1 ? 's' : ''} loaded`;
        }
        if (elements.modeBadge && data.competition_mode) {
            show(elements.modeBadge);
            elements.modeText.textContent = 'COMPETITION MODE';
        }
    } catch (error) {
        console.error('Health check failed:', error);
        if (elements.healthDot) elements.healthDot.className = 'status-dot degraded';
        if (elements.healthText) elements.healthText.textContent = 'Connection Failed';
    }
}

// Load example questions
function loadExamples() {
    const examples = [
        "เงื่อนไขการสมัครคืออะไร?",
        "รางวัลอันดับ 1 ได้เท่าไหร่?",
        "กำหนดส่งโครงงานเมื่อไหร่?",
        "ทีมต้องมีสมาชิกกี่คน?",
        "เกณฑ์การตัดสินคืออะไร?",
        "ต้องใช้ Python กี่ขึ้นไป?",
    ];

    if (elements.exampleChips) {
        elements.exampleChips.innerHTML = examples.map(q =>
            `<span data-question="${escapeHtml(q)}">${escapeHtml(q)}</span>`
        ).join('');

        // Add click handlers
        elements.exampleChips.querySelectorAll('span').forEach(chip => {
            chip.addEventListener('click', () => {
                if (elements.questionInput) {
                    elements.questionInput.value = chip.dataset.question;
                    elements.questionInput.focus();
                    updateAskButton();
                }
            });
        });
    }
}

// Update ask button state
function updateAskButton() {
    const hasText = elements.questionInput?.value?.trim().length > 0;
    if (elements.askBtn) {
        elements.askBtn.disabled = !hasText || isProcessing;
    }
}

// Ask question via REST API
async function askQuestion() {
    const question = elements.questionInput?.value?.trim();
    if (!question || isProcessing) return;

    setLoading(true);
    hide(elements.answerSection);
    hide(elements.sourcesSection);

    try {
        const response = await fetch(`${API_BASE}/query`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                query: question,
                mode: currentMode,
            })
        });

        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.detail || 'Query failed');
        }

        const data = await response.json();
        displayAnswer(data);
        addToHistory(question, data);

        // Clear input
        if (elements.questionInput) {
            elements.questionInput.value = '';
            updateAskButton();
        }
    } catch (error) {
        console.error('Query error:', error);
        showError(error.message || 'An error occurred');
    } finally {
        setLoading(false);
    }
}

// Display answer
function displayAnswer(data) {
    // Show answer section
    show(elements.answerSection);

    // Answer content
    if (elements.answerContent) {
        elements.answerContent.textContent = data.answer;
    }

    // Answer meta
    if (elements.answerMeta) {
        const confidenceClass = getConfidenceClass(data.confidence);
        const confidenceLabel = getConfidenceLabel(data.confidence);
        const thresholdStatus = data.retrieval_passed_threshold ? '✅ Retrieved' : '⚠️ Below threshold';

        elements.answerMeta.innerHTML = `
            <span class="confidence-badge ${confidenceClass}">
                Confidence: ${confidenceLabel} (${(data.confidence * 100).toFixed(0)}%)
            </span>
            <span>${thresholdStatus}</span>
            <span>⏱️ Total: ${formatTime(data.total_time_ms)}</span>
            <span>🔍 Retrieval: ${formatTime(data.retrieval_time_ms)}</span>
        `;
    }

    // Sources
    if (data.sources && data.sources.length > 0) {
        show(elements.sourcesSection);
        if (elements.sourcesCount) {
            elements.sourcesCount.textContent = data.sources.length;
        }
        if (elements.sourcesList) {
            elements.sourcesList.innerHTML = data.sources.map((source, idx) => `
                <div class="source-item" style="animation-delay: ${idx * 50}ms">
                    <span class="source-icon">📄</span>
                    <div class="source-content">
                        <div class="source-title">${escapeHtml(source.source)}</div>
                        <div class="source-meta">
                            ${source.page !== null && source.page !== undefined ? `<span>📄 Page ${source.page}</span>` : ''}
                            ${source.chunk_index !== undefined ? `<span>Chunk #${source.chunk_index}</span>` : ''}
                            ${source.heading ? `<span>📑 ${escapeHtml(source.heading)}</span>` : ''}
                            <span class="source-score">Score: ${source.score.toFixed(4)}</span>
                        </div>
                    </div>
                </div>
            `).join('');
        }
    } else {
        hide(elements.sourcesSection);
    }

    // Scroll to answer
    elements.answerSection.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}

// Show error
function showError(message) {
    show(elements.answerSection);
    if (elements.answerContent) {
        elements.answerContent.innerHTML = `<span style="color: var(--accent-danger);">❌ Error: ${escapeHtml(message)}</span>`;
    }
    if (elements.answerMeta) {
        elements.answerMeta.textContent = '';
    }
    hide(elements.sourcesSection);
}

// Add to history
function addToHistory(question, data) {
    questionHistory.unshift({
        question,
        answer: data.answer,
        confidence: data.confidence,
        time: Date.now(),
        sources: data.sources?.length || 0
    });

    // Limit history
    if (questionHistory.length > 20) {
        questionHistory = questionHistory.slice(0, 20);
    }

    renderHistory();
}

// Render history
function renderHistory() {
    if (!elements.historyList) return;

    if (questionHistory.length === 0) {
        elements.historyList.innerHTML = '<div style="padding: 20px; text-align: center; color: var(--text-muted);">No questions yet</div>';
        hide(elements.historySection);
        return;
    }

    show(elements.historySection);
    elements.historyList.innerHTML = questionHistory.map((item, idx) => `
        <div class="history-item" data-index="${idx}">
            <div class="history-question">${escapeHtml(item.question)}</div>
            <div class="history-meta">
                <span>${new Date(item.time).toLocaleTimeString()}</span>
                <span>Confidence: ${(item.confidence * 100).toFixed(0)}%</span>
                <span>${item.sources} sources</span>
            </div>
        </div>
    `).join('');

    // Add click handlers
    elements.historyList.querySelectorAll('.history-item').forEach(item => {
        item.addEventListener('click', () => {
            const idx = parseInt(item.dataset.index);
            const historyItem = questionHistory[idx];
            if (historyItem && elements.questionInput) {
                elements.questionInput.value = historyItem.question;
                elements.questionInput.focus();
                updateAskButton();
            }
        });
    });
}

// Clear history
function clearHistory() {
    questionHistory = [];
    renderHistory();
}

// Copy answer
async function copyAnswer() {
    if (!elements.answerContent) return;

    const text = elements.answerContent.textContent;
    try {
        await navigator.clipboard.writeText(text);
        // Visual feedback
        const originalText = elements.copyBtn.textContent;
        elements.copyBtn.textContent = '✅ Copied!';
        setTimeout(() => {
            elements.copyBtn.textContent = originalText;
        }, 1500);
    } catch (error) {
        console.error('Copy failed:', error);
    }
}

// Event listeners
function setupEventListeners() {
    // Ask button
    elements.askBtn?.addEventListener('click', askQuestion);

    // Enter key in textarea
    elements.questionInput?.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            if (!isProcessing) askQuestion();
        }
    });

    // Input change
    elements.questionInput?.addEventListener('input', updateAskButton);

    // Mode select
    elements.modeSelect?.addEventListener('change', (e) => {
        currentMode = e.target.value;
    });

    // Clear button
    elements.clearBtn?.addEventListener('click', () => {
        if (elements.questionInput) {
            elements.questionInput.value = '';
            updateAskButton();
        }
        hide(elements.answerSection);
        hide(elements.sourcesSection);
    });

    // Clear history
    elements.clearHistoryBtn?.addEventListener('click', clearHistory);

    // Copy answer
    elements.copyBtn?.addEventListener('click', copyAnswer);
}

// Initialize
async function init() {
    initElements();
    setupEventListeners();
    loadExamples();
    await checkHealth();

    // Periodic health check
    setInterval(checkHealth, 30000);

    // Focus input
    elements.questionInput?.focus();

    console.log('ThaiLLM Document Assistant initialized');
}

// Start when DOM is ready
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
} else {
    init();
}