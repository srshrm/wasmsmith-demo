// semanticsearch.js — "find similar content" by embedding cosine similarity,
// entirely client-side. Pick a document; the block ranks every other document
// by nearest-neighbour cosine similarity in a SIMD WASM kernel — no server.
//
// This file is UI only: it parses the authoring config, fetches a precomputed
// embeddings file, builds the index once, renders the document list, and wires
// clicks. All the typed-array / wasm work lives in ./semanticsearch-engine.js.
//
// The engine + embeddings load LAZILY (first interaction or when the block
// scrolls into view) so the synchronous decorate path stays fast.

import { readBlockConfig } from '../../scripts/aem.js';
import { createSemanticEngine, normalizeRows } from './semanticsearch-engine.js';

function wasmUrl() {
  const base = (typeof window !== 'undefined' && window.hlx && window.hlx.codeBasePath) || '';
  return `${base}/blocks/semanticsearch/semanticsearch.wasm`;
}

function resolveUrl(src) {
  const base = (window.hlx && window.hlx.codeBasePath) || '';
  return /^(https?:|data:|blob:)/.test(src) ? src : `${base}${src}`;
}

function readConfig(block) {
  const raw = readBlockConfig(block);
  const str = (v) => (Array.isArray(v) ? v.join(', ') : `${v ?? ''}`).trim();
  const num = (v) => {
    const n = parseInt(str(v), 10);
    return Number.isFinite(n) && n > 0 ? n : 0;
  };
  let source = str(raw.source || raw.embeddings);
  if (!source) {
    const link = block.querySelector('a[href$=".json"]');
    if (link) source = link.getAttribute('href');
  }
  return {
    source,
    topk: num(raw.topk) || 8,
    label: str(raw.label) || 'Similar content',
  };
}

// Parse an embeddings file: { dim, docs: [{ title, url?, embedding: [...] }] }.
// Returns a flat unit-normalized corpus plus the doc metadata.
function parseEmbeddings(json) {
  const docs = Array.isArray(json.docs) ? json.docs : [];
  const count = docs.length;
  const dim = json.dim || (docs[0] && docs[0].embedding ? docs[0].embedding.length : 0);
  const corpus = new Float32Array(count * dim);
  for (let r = 0; r < count; r += 1) {
    const emb = docs[r].embedding || [];
    corpus.set(emb.slice(0, dim), r * dim);
  }
  normalizeRows(corpus, count, dim);
  return {
    corpus, count, dim, docs,
  };
}

/**
 * loads and decorates the block
 * @param {Element} block The block element
 */
export default async function decorate(block) {
  const config = readConfig(block);
  block.textContent = '';

  const status = document.createElement('p');
  status.className = 'semanticsearch-status';
  status.setAttribute('aria-live', 'polite');

  const list = document.createElement('ul');
  list.className = 'semanticsearch-list';

  const results = document.createElement('div');
  results.className = 'semanticsearch-results';

  block.append(status, list, results);

  let index = null;
  let data = null;
  let ready = null;

  function showNeighbours(docIndex) {
    const query = data.corpus.slice(docIndex * data.dim, docIndex * data.dim + data.dim);
    const t0 = performance.now();
    const hits = index.search(query, config.topk + 1);
    const ms = (performance.now() - t0).toFixed(1);

    results.textContent = '';
    const heading = document.createElement('p');
    heading.className = 'semanticsearch-heading';
    heading.textContent = `Similar to “${data.docs[docIndex].title}” · ranked in ${ms} ms`;
    results.append(heading);

    const ol = document.createElement('ol');
    hits.filter((h) => h.index !== docIndex).slice(0, config.topk).forEach((h) => {
      const doc = data.docs[h.index];
      const li = document.createElement('li');
      const pct = document.createElement('span');
      pct.className = 'semanticsearch-score';
      pct.textContent = `${(h.score * 100).toFixed(0)}%`;
      const title = doc.url
        ? Object.assign(document.createElement('a'), { href: doc.url, textContent: doc.title })
        : Object.assign(document.createElement('span'), { textContent: doc.title });
      li.append(pct, title);
      ol.append(li);
    });
    results.append(ol);
  }

  function renderList() {
    list.textContent = '';
    data.docs.forEach((doc, i) => {
      const li = document.createElement('li');
      const btn = document.createElement('button');
      btn.type = 'button';
      btn.textContent = doc.title;
      btn.addEventListener('click', () => {
        list.querySelectorAll('button').forEach((b) => b.removeAttribute('aria-current'));
        btn.setAttribute('aria-current', 'true');
        showNeighbours(i);
      });
      li.append(btn);
      list.append(li);
    });
  }

  async function load() {
    if (ready) return ready;
    ready = (async () => {
      if (!config.source) {
        status.textContent = 'No embeddings source configured.';
        return false;
      }
      status.textContent = 'Loading embeddings…';
      const json = await (await fetch(resolveUrl(config.source))).json();
      data = parseEmbeddings(json);
      const engine = await createSemanticEngine(wasmUrl());
      index = engine.index(data.corpus, data.count, data.dim);
      renderList();
      status.textContent = `${data.count} documents indexed · pick one to find similar`;
      return true;
    })().catch((err) => {
      // eslint-disable-next-line no-console
      console.warn('[semanticsearch] unavailable', err);
      status.textContent = 'Semantic search unavailable.';
      return false;
    });
    return ready;
  }

  // Warm up (fetch + instantiate + index) when the block scrolls into view.
  if (typeof IntersectionObserver !== 'undefined') {
    const io = new IntersectionObserver((entries) => {
      if (entries.some((entry) => entry.isIntersecting)) {
        io.disconnect();
        load();
      }
    }, { rootMargin: '200px' });
    io.observe(block);
  } else {
    load();
  }
}
