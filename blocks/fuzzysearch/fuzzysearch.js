// fuzzysearch.js — client-side "elastic" search over an EDS query-index:
// typo-tolerant fuzzy matching (edit distance) + BM25 relevance ranking, all
// on-device with no backend.
//
// This file is the block's UI only: it parses the authoring config, fetches the
// query-index once, renders the search box and results, and wires up events. All
// the heavy lifting — loading the WebAssembly kernel, building the BM25 index in
// wasm memory, and fuzzy-matching + ranking each keystroke — lives in
// ./search-engine.js.
//
// The engine is created LAZILY (on first interaction or when the block scrolls
// into view) so the synchronous decorate path — and the Lighthouse score — stay
// untouched. There are ZERO network calls at query time: the index is fetched
// once on init, and every keystroke re-scores in-memory in WASM.

import { readBlockConfig } from '../../scripts/aem.js';
import { createSearchEngine } from './search-engine.js';

function wasmUrl() {
  const base = (typeof window !== 'undefined' && window.hlx && window.hlx.codeBasePath) || '';
  return `${base}/blocks/fuzzysearch/fuzzysearch.wasm`;
}

// Pull the raw key/value rows with the platform helper, then coerce to typed
// config with defaults. `readBlockConfig` lowercases/dash-normalizes keys.
function readConfig(block) {
  const raw = readBlockConfig(block);
  const str = (v) => (Array.isArray(v) ? v.join(', ') : `${v ?? ''}`).trim();
  const fields = str(raw.fields).split(',').map((s) => s.trim()).filter(Boolean);
  const fuzziness = parseInt(str(raw.fuzziness), 10);
  return {
    source: str(raw.source || raw.index) || '/query-index.json',
    placeholder: str(raw.placeholder) || 'Search…',
    limit: parseInt(str(raw.limit), 10) || 10,
    fields: fields.length ? fields : ['title', 'description', 'content', 'author', 'place'],
    // Max edit distance for typo tolerance (0/1/2). Default 2 (Elasticsearch AUTO ceiling).
    fuzziness: Number.isFinite(fuzziness) ? Math.max(0, Math.min(2, fuzziness)) : 2,
  };
}

async function loadIndex(source) {
  const base = (window.hlx && window.hlx.codeBasePath) || '';
  const url = /^https?:/.test(source) ? source : `${base}${source}`;
  const res = await fetch(url);
  if (!res.ok) throw new Error(`query-index fetch failed: ${res.status}`);
  const json = await res.json();
  return Array.isArray(json) ? json : (json.data || []);
}

function rowText(row, fields) {
  return fields.map((field) => row[field]).filter(Boolean).join(' ');
}

function renderResults(list, indices, rows) {
  list.textContent = '';
  if (indices.length === 0) {
    const li = document.createElement('li');
    li.className = 'fuzzysearch-empty';
    li.textContent = 'No matches';
    list.append(li);
    return;
  }
  const frag = document.createDocumentFragment();
  indices.forEach((i) => {
    const row = rows[i];
    const li = document.createElement('li');
    li.setAttribute('role', 'option');
    const a = document.createElement('a');
    a.href = row.path || row.url || '#';
    a.textContent = row.title || row.path || '(untitled)';
    li.append(a);
    if (row.description) {
      const desc = document.createElement('p');
      desc.className = 'fuzzysearch-desc';
      desc.textContent = row.description;
      li.append(desc);
    }
    frag.append(li);
  });
  list.append(frag);
}

/**
 * loads and decorates the block
 * @param {Element} block The block element
 */
export default async function decorate(block) {
  const config = readConfig(block);
  block.textContent = '';

  // Synchronous, cheap UI only — no fetch, no wasm. Render stays fast.
  const input = document.createElement('input');
  input.type = 'search';
  input.className = 'fuzzysearch-input';
  input.placeholder = config.placeholder;
  input.setAttribute('aria-label', config.placeholder || 'Search');
  input.autocomplete = 'off';

  const status = document.createElement('p');
  status.className = 'fuzzysearch-status';
  status.setAttribute('aria-live', 'polite');

  const results = document.createElement('ul');
  results.className = 'fuzzysearch-results';
  results.setAttribute('role', 'listbox');

  block.append(input, status, results);

  let rows = [];
  let engine = null;
  let readyPromise = null;

  async function init() {
    if (readyPromise) return readyPromise;
    readyPromise = (async () => {
      status.textContent = 'Loading search index…';
      try {
        rows = await loadIndex(config.source);
      } catch (err) {
        // eslint-disable-next-line no-console
        console.warn('[fuzzysearch] could not load index', err);
        status.textContent = 'Search index unavailable.';
        rows = [];
        return;
      }
      try {
        const texts = rows.map((row) => rowText(row, config.fields));
        engine = await createSearchEngine(wasmUrl(), texts, { maxEdits: config.fuzziness });
        status.textContent = `${rows.length} entries · ${engine.vocabCount} terms · typo-tolerant BM25 (WASM)`;
      } catch (err) {
        // eslint-disable-next-line no-console
        console.warn('[fuzzysearch] WASM search unavailable', err);
        engine = null;
        status.textContent = 'Search unavailable.';
      }
    })();
    return readyPromise;
  }

  function runQuery(text) {
    if (!engine) return;
    if (!text.trim()) {
      results.textContent = '';
      return;
    }
    renderResults(results, engine.search(text.trim(), config.limit), rows);
  }

  let frame = 0;
  input.addEventListener('input', async () => {
    await init();
    cancelAnimationFrame(frame);
    frame = requestAnimationFrame(() => runQuery(input.value));
  });
  input.addEventListener('focus', () => { init(); });

  // Warm up when the block scrolls into view, well before any interaction.
  if (typeof IntersectionObserver !== 'undefined') {
    const io = new IntersectionObserver((entries) => {
      if (entries.some((entry) => entry.isIntersecting)) {
        io.disconnect();
        init();
      }
    }, { rootMargin: '200px' });
    io.observe(block);
  }
}
