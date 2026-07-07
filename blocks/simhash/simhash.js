// simhash.js — near-duplicate / SEO-canonical detection over document text,
// entirely client-side. The block fetches a list of documents, computes a
// 64-bit SimHash fingerprint of each in a WASM kernel (native i64 — JS would
// need BigInt), then clusters documents whose fingerprints are within a small
// Hamming distance and renders the near-duplicate groups — no server.
//
// This file is UI only: it parses the authoring config, fetches the document
// list, fingerprints via the engine, clusters, and renders. All typed-array /
// wasm work lives in ./simhash-engine.js.
//
// The engine + document list load LAZILY (when the block scrolls into view) so
// the synchronous decorate path stays fast.

import { readBlockConfig } from '../../scripts/aem.js';
import { createSimhashEngine, toBytes, hamming } from './simhash-engine.js';

function wasmUrl() {
  const base = (typeof window !== 'undefined' && window.hlx && window.hlx.codeBasePath) || '';
  return `${base}/blocks/simhash/simhash.wasm`;
}

function resolveUrl(src) {
  const base = (window.hlx && window.hlx.codeBasePath) || '';
  return /^(https?:|data:|blob:)/.test(src) ? src : `${base}${src}`;
}

function readConfig(block) {
  const raw = readBlockConfig(block);
  const str = (v) => (Array.isArray(v) ? v.join(', ') : `${v ?? ''}`).trim();
  const num = (v, dflt) => {
    const n = parseInt(str(v), 10);
    return Number.isFinite(n) && n >= 0 ? n : dflt;
  };
  let source = str(raw.source || raw.docs);
  if (!source) {
    const link = block.querySelector('a[href$=".json"]');
    if (link) source = link.getAttribute('href');
  }
  return {
    source,
    threshold: num(raw.threshold, 3),
    label: str(raw.label) || 'Near-duplicate groups',
  };
}

// Parse a document list: { docs: [{ title, url?, text }] }.
function parseDocs(json) {
  const docs = Array.isArray(json.docs) ? json.docs : [];
  return docs.map((d, i) => ({
    title: (d.title || `Document ${i + 1}`).trim(),
    url: d.url || '',
    text: d.text || '',
  }));
}

// Cluster documents whose fingerprints are within `threshold` bits (union-find
// on a single-linkage pass). Returns groups of size >= 2 (the near-dup groups),
// each an array of doc indices. O(n^2) over fingerprints — fine for demo sizes.
function clusterNearDups(fingerprints, threshold) {
  const n = fingerprints.length;
  const parent = new Array(n);
  for (let i = 0; i < n; i += 1) parent[i] = i;

  const find = (x) => {
    let root = x;
    while (parent[root] !== root) root = parent[root];
    let cur = x;
    while (parent[cur] !== root) {
      const next = parent[cur];
      parent[cur] = root;
      cur = next;
    }
    return root;
  };
  const union = (a, b) => {
    const ra = find(a);
    const rb = find(b);
    if (ra !== rb) parent[ra] = rb;
  };

  for (let i = 0; i < n; i += 1) {
    for (let j = i + 1; j < n; j += 1) {
      if (hamming(fingerprints[i], fingerprints[j]) <= threshold) union(i, j);
    }
  }

  const byRoot = new Map();
  for (let i = 0; i < n; i += 1) {
    const r = find(i);
    if (!byRoot.has(r)) byRoot.set(r, []);
    byRoot.get(r).push(i);
  }
  return Array.from(byRoot.values()).filter((g) => g.length >= 2);
}

/**
 * loads and decorates the block
 * @param {Element} block The block element
 */
export default async function decorate(block) {
  const config = readConfig(block);
  block.textContent = '';

  const status = document.createElement('p');
  status.className = 'simhash-status';
  status.setAttribute('aria-live', 'polite');

  const results = document.createElement('div');
  results.className = 'simhash-results';

  block.append(status, results);

  let ready = null;

  function renderGroups(docs, groups) {
    results.textContent = '';
    if (!groups.length) {
      const none = document.createElement('p');
      none.className = 'simhash-empty';
      none.textContent = 'No near-duplicates found.';
      results.append(none);
      return;
    }
    const heading = document.createElement('p');
    heading.className = 'simhash-heading';
    heading.textContent = config.label;
    results.append(heading);

    const wrap = document.createElement('div');
    wrap.className = 'simhash-groups';
    groups.forEach((group, gi) => {
      const card = document.createElement('section');
      card.className = 'simhash-group';
      const gh = document.createElement('h4');
      gh.textContent = `Group ${gi + 1} · ${group.length} documents`;
      const ul = document.createElement('ul');
      group.forEach((docIndex) => {
        const doc = docs[docIndex];
        const li = document.createElement('li');
        const node = doc.url
          ? Object.assign(document.createElement('a'), { href: doc.url, textContent: doc.title })
          : Object.assign(document.createElement('span'), { textContent: doc.title });
        li.append(node);
        ul.append(li);
      });
      card.append(gh, ul);
      wrap.append(card);
    });
    results.append(wrap);
  }

  async function load() {
    if (ready) return ready;
    ready = (async () => {
      if (!config.source) {
        status.textContent = 'No document source configured.';
        return false;
      }
      status.textContent = 'Loading documents…';
      const json = await (await fetch(resolveUrl(config.source))).json();
      const docs = parseDocs(json);
      if (!docs.length) {
        status.textContent = 'No documents to fingerprint.';
        return false;
      }

      const engine = await createSimhashEngine(wasmUrl());
      const byteDocs = docs.map((d) => toBytes(d.text));
      const maxDocLen = byteDocs.reduce((m, b) => Math.max(m, b.length), 1);
      const fingerprinter = engine.fingerprinter(maxDocLen);

      const t0 = performance.now();
      const fingerprints = byteDocs.map((b) => fingerprinter.fingerprint(b));
      const ms = (performance.now() - t0).toFixed(1);

      const groups = clusterNearDups(fingerprints, config.threshold);
      renderGroups(docs, groups);
      status.textContent = `${docs.length} documents fingerprinted in ${ms} ms · `
        + `${groups.length} near-duplicate group(s) at Hamming ≤ ${config.threshold}`;
      return true;
    })().catch((err) => {
      // eslint-disable-next-line no-console
      console.warn('[simhash] unavailable', err);
      status.textContent = 'Fingerprinting unavailable.';
      return false;
    });
    return ready;
  }

  // Warm up (fetch + instantiate + fingerprint) when the block scrolls into view.
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
