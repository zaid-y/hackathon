const { test } = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const script = fs.readFileSync(path.join(__dirname, '../frontend/app.js'), 'utf8');

function boot(initial = {}, dark = false) {
  const storage = new Map(Object.entries(initial));
  const elements = new Map();
  class Element {
    constructor() {
      this.dataset = {}; this.style = {}; this.children = []; this.handlers = {};
      this.value = ''; this.checked = false; this.open = false; this.type = '';
      this.scrollHeight = 34; this.submissions = 0;
      const names = new Set();
      this.classList = { add: x => names.add(x), remove: x => names.delete(x),
        contains: x => names.has(x), toggle: (x, on) => on ? names.add(x) : names.delete(x) };
    }
    addEventListener(name, fn) { this.handlers[name] = fn; }
    append(...nodes) { this.children.push(...nodes); }
    replaceChildren() { this.children = []; }
    setAttribute(name, value) { this[name] = value; }
    focus() {}
    click() { this.clicked = true; }
    remove() { this.removed = true; }
    showModal() { this.open = true; }
    close() { this.open = false; }
    requestSubmit() { this.submissions++; }
    getBoundingClientRect() { return { left: 0, top: 0, right: 500, bottom: 600 }; }
  }
  const get = id => {
    if (!elements.has(id)) {
      const element = new Element();
      if (id.endsWith('-toggle')) element.type = 'checkbox';
      elements.set(id, element);
    }
    return elements.get(id);
  };
  const media = { matches: dark, addEventListener(name, fn) { this.change = fn; } };
  const pages = ['general', 'chat', 'advanced', 'data', 'about'];
  const nav = pages.map(page => { const el = new Element(); el.dataset.settingsPage = page; return el; });
  const panels = pages.map(page => { const el = new Element(); el.dataset.settingsPanel = page; return el; });
  const document = { querySelector: get, querySelectorAll: selector =>
    selector === '[data-settings-page]' ? nav : selector === '[data-settings-panel]' ? panels : [],
    createElement: () => new Element(), documentElement: new Element(), body: new Element() };
  let id = 0;
  const requests = [], downloads = [];
  const sandbox = { document, Blob, URL: { createObjectURL: blob => { downloads.push(blob); return 'blob:test'; }, revokeObjectURL() {} },
    setTimeout: fn => fn(), localStorage: { getItem: key => storage.get(key) ?? null,
    setItem: (key, value) => storage.set(key, value), removeItem: key => storage.delete(key) },
    window: { crypto: { randomUUID: () => `test-${++id}` }, matchMedia: () => media, confirm: () => false },
    requestAnimationFrame: fn => fn(),
    fetch: async (url, options) => { requests.push({ url, options }); return { ok: true, json: async () => url === '/api/ask'
      ? { answer: 'Test response', sources: [], grounded: true, retrieval_confidence: .7 }
      : { thailmm_configured: true, index_exists: true } }; },
  };
  vm.createContext(sandbox);
  vm.runInContext(script, sandbox);
  return { get, media, storage, document, sandbox, requests, downloads, nav, panels, run: code => vm.runInContext(code, sandbox) };
}

test('invalid stored settings fall back to validated defaults', () => {
  const app = boot({ 'thaillmm-settings-v1': '{"theme":"bad","density":"bad","showConfidence":"false"}' });
  assert.equal(app.get('#theme-select').value, 'system');
  assert.equal(app.get('#density-select').value, 'comfortable');
  assert.equal(app.get('#show-confidence-toggle').checked, true);
});
test('theme changes persist and restore on a fresh page', () => {
  const app = boot();
  app.get('#theme-select').value = 'dark';
  app.get('#theme-select').handlers.change();
  assert.equal(app.document.documentElement.dataset.theme, 'dark');
  const reloaded = boot(Object.fromEntries(app.storage));
  assert.equal(reloaded.document.documentElement.dataset.theme, 'dark');
});
test('system theme follows device changes', () => {
  const app = boot({}, true);
  assert.equal(app.document.documentElement.dataset.theme, 'dark');
  app.media.matches = false; app.media.change();
  assert.equal(app.document.documentElement.dataset.theme, 'light');
});
test('density, confidence and source preferences apply', () => {
  const app = boot();
  app.get('#density-select').value = 'compact'; app.get('#density-select').handlers.change();
  app.get('#show-confidence-toggle').checked = false; app.get('#show-confidence-toggle').handlers.change();
  app.get('#open-sources-toggle').checked = true; app.get('#open-sources-toggle').handlers.change();
  assert.equal(app.document.documentElement.dataset.density, 'compact');
  assert.equal(app.document.body.classList.contains('hide-confidence'), true);
  assert.equal(app.run('renderSources([]).open'), true);
});
test('Enter preference respects newlines and composition input', () => {
  const app = boot();
  const event = { key: 'Enter', shiftKey: false, isComposing: false, preventDefault() {} };
  app.get('#question').handlers.keydown(event);
  assert.equal(app.get('#question-form').submissions, 1);
  app.get('#enter-to-send-toggle').checked = false; app.get('#enter-to-send-toggle').handlers.change();
  app.get('#question').handlers.keydown(event);
  assert.equal(app.get('#question-form').submissions, 1);
});
test('settings dialog opens and closes; reset preserves conversations', () => {
  const app = boot();
  app.get('#settings-button').handlers.click();
  assert.equal(app.get('#settings-dialog').open, true);
  const before = app.storage.get('thaillmm-document-chats-v2');
  app.get('#reset-settings-button').handlers.click();
  assert.equal(app.storage.get('thaillmm-document-chats-v2'), before);
  app.get('#done-settings-button').handlers.click();
  assert.equal(app.get('#settings-dialog').open, false);
});
test('clear requires confirmation and preserves preferences', () => {
  const app = boot();
  const before = app.storage.get('thaillmm-document-chats-v2');
  app.get('#clear-all-chats-button').handlers.click();
  assert.equal(app.storage.get('thaillmm-document-chats-v2'), before);
  app.get('#theme-select').value = 'dark'; app.get('#theme-select').handlers.change();
  app.sandbox.window.confirm = () => true;
  app.get('#clear-all-chats-button').handlers.click();
  const state = JSON.parse(app.storage.get('thaillmm-document-chats-v2'));
  assert.equal(state.chats.length, 1);
  assert.equal(state.chats[0].messages.length, 0);
  assert.equal(JSON.parse(app.storage.get('thaillmm-settings-v1')).theme, 'dark');
});
test('assistant replies survive save and reload after async request', async () => {
  const app = boot();
  app.get('#question').value = 'Test question';
  await app.get('#question-form').handlers.submit({ preventDefault() {} });
  const reloaded = boot(Object.fromEntries(app.storage));
  assert.equal(reloaded.run('activeChat().messages.length'), 2);
  assert.equal(reloaded.run('activeChat().messages[1].content'), 'Test response');
});

test('settings navigation shows exactly the chosen category', () => {
  const app = boot();
  app.nav[2].handlers.click();
  assert.equal(app.nav[2]['aria-pressed'], 'true');
  assert.equal(app.nav[0]['aria-pressed'], 'false');
  assert.deepEqual(app.panels.map(panel => panel.hidden), [true, true, false, true, true]);
});

test('advanced options persist and are sent with requests; context off sends no history', async () => {
  const app = boot();
  for (const [id, value] of [['retrieval-depth-select', '8'], ['evidence-mode-select', 'strict'],
    ['context-limit-select', '0'], ['answer-style-select', 'detailed'], ['answer-language-select', 'en']]) {
    app.get(`#${id}`).value = value; app.get(`#${id}`).handlers.change();
  }
  const reloaded = boot(Object.fromEntries(app.storage));
  assert.equal(reloaded.get('#retrieval-depth-select').value, '8');
  reloaded.run('activeChat().messages.push({role:"user",content:"older question"})');
  reloaded.get('#question').value = 'New question';
  await reloaded.get('#question-form').handlers.submit({ preventDefault() {} });
  const body = JSON.parse(reloaded.requests.find(request => request.url === '/api/ask').options.body);
  assert.deepEqual(body.history, []);
  assert.deepEqual(body.options, { top_k: 8, evidence_mode: 'strict', history_messages: 0,
    answer_style: 'detailed', answer_language: 'en' });
  assert.equal(reloaded.run('activeChat().messages.length'), 3);
});

test('invalid advanced preferences fall back; reset restores all defaults', () => {
  const app = boot({ 'thaillmm-settings-v1': JSON.stringify({ retrievalDepth: 999,
    contextLimit: -1, evidenceMode: 'off', answerStyle: 'invalid', fontSize: 'huge' }) });
  assert.equal(app.get('#retrieval-depth-select').value, '5');
  assert.equal(app.get('#context-limit-select').value, '12');
  app.get('#answer-style-select').value = 'detailed'; app.get('#answer-style-select').handlers.change();
  app.get('#reset-settings-button').handlers.click();
  assert.equal(app.get('#answer-style-select').value, 'concise');
});

test('text size, starters and autoscroll controls apply independently', () => {
  const app = boot();
  app.get('#font-size-select').value = 'large'; app.get('#font-size-select').handlers.change();
  app.get('#suggestions-toggle').checked = false; app.get('#suggestions-toggle').handlers.change();
  app.get('#auto-scroll-toggle').checked = false; app.get('#auto-scroll-toggle').handlers.change();
  app.get('#conversation-stage').scrollTop = 7;
  app.run('renderConversation()');
  assert.equal(app.document.documentElement.dataset.fontSize, 'large');
  assert.equal(app.document.body.classList.contains('hide-suggestions'), true);
  assert.equal(app.get('#conversation-stage').scrollTop, 7);
});

test('exports download only intended conversation fields', async () => {
  const app = boot();
  app.run(`activeChat().messages.push({role:'assistant', content:'Answer', api_key:'secret',
    sources:[{document:'IT.pdf',page:15,api_key:'secret'}]});
    chatState.chats.push(createChat([{role:'user',content:'Another chat'}]));`);
  app.get('#export-chat-button').handlers.click();
  const exported = JSON.parse(await app.downloads[0].text());
  assert.equal(exported.chats.length, 1);
  assert.equal(exported.chats[0].messages[0].content, 'Answer');
  assert.deepEqual(exported.chats[0].messages[0].sources, [{document:'IT.pdf',page:15}]);
  assert.equal(JSON.stringify(exported).includes('secret'), false);
  app.get('#export-all-button').handlers.click();
  assert.equal(JSON.parse(await app.downloads[1].text()).chats.length, 2);
});

test('context count limits prior messages without erasing saved history', () => {
  const app = boot();
  app.get('#context-limit-select').value = '6'; app.get('#context-limit-select').handlers.change();
  assert.equal(app.run('apiHistory(Array.from({length:20},(_,i)=>({role:"user",content:String(i)}))).length'), 6);
});
