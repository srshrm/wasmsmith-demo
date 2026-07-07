// imagehash.js — find near-duplicate images in a gallery, entirely client-side.
// Each image is reduced to a 32×32 grayscale, perceptually hashed with a SIMD
// DCT in WebAssembly, and grouped with others whose fingerprints are within a
// small Hamming distance — the kind of dedup pass a DAM runs over an asset
// library, here with no backend.
//
// UI only: it collects image URLs, downscales each on the canvas (native), then
// hands the grayscale batch to ./imagehash-engine.js. The engine loads LAZILY
// (first scroll into view) so the decorate path stays fast.

import { readBlockConfig } from '../../scripts/aem.js';
import {
  createImageHashEngine, toGray, hamming, N,
} from './imagehash-engine.js';

function wasmUrl() {
  const base = (typeof window !== 'undefined' && window.hlx && window.hlx.codeBasePath) || '';
  return `${base}/blocks/imagehash/imagehash.wasm`;
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
    return Number.isFinite(n) && n >= 0 ? n : -1;
  };
  const source = str(raw.source || raw.images);
  const threshold = num(raw.threshold);
  return {
    source,
    threshold: threshold >= 0 ? threshold : 8,
    label: str(raw.label) || 'Near-duplicate groups',
  };
}

// Downscale an image URL to an N×N grayscale block via the canvas (native).
function rasterizeGray(url) {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.crossOrigin = 'anonymous';
    img.onload = () => {
      const canvas = document.createElement('canvas');
      canvas.width = N;
      canvas.height = N;
      const ctx = canvas.getContext('2d');
      ctx.drawImage(img, 0, 0, N, N);
      try {
        resolve(toGray(ctx.getImageData(0, 0, N, N).data));
      } catch (err) {
        reject(err); // tainted canvas (cross-origin image without CORS)
      }
    };
    img.onerror = () => reject(new Error('image failed to load'));
    img.src = url;
  });
}

// Group indices whose fingerprints are within `threshold` bits (union-find).
function clusterByHamming(hashes, threshold) {
  const parent = hashes.map((_, i) => i);
  const find = (i) => { let r = i; while (parent[r] !== r) r = parent[r]; return r; };
  for (let i = 0; i < hashes.length; i += 1) {
    for (let j = i + 1; j < hashes.length; j += 1) {
      if (hamming(hashes[i], hashes[j]) <= threshold) parent[find(j)] = find(i);
    }
  }
  const groups = new Map();
  hashes.forEach((_, i) => {
    const r = find(i);
    if (!groups.has(r)) groups.set(r, []);
    groups.get(r).push(i);
  });
  return [...groups.values()].filter((g) => g.length > 1);
}

/**
 * loads and decorates the block
 * @param {Element} block The block element
 */
export default async function decorate(block) {
  const config = readConfig(block);

  // Collect candidate images before we clear the block.
  let items = [];
  if (config.source) {
    items = null; // fetched during load()
  } else {
    items = [...block.querySelectorAll('img[src], a[href]')].map((el) => ({
      url: el.getAttribute('src') || el.getAttribute('href'),
      title: el.getAttribute('alt') || el.textContent || '',
    })).filter((it) => it.url);
  }

  block.textContent = '';
  const status = document.createElement('p');
  status.className = 'imagehash-status';
  status.setAttribute('aria-live', 'polite');
  const results = document.createElement('div');
  results.className = 'imagehash-results';
  block.append(status, results);

  let ready = null;

  function renderGroups(groups, urls, ms) {
    results.textContent = '';
    if (!groups.length) {
      status.textContent = `No near-duplicates found · ${urls.length} images fingerprinted in ${ms} ms`;
      return;
    }
    status.textContent = `${groups.length} near-duplicate group(s) · ${urls.length} images fingerprinted in ${ms} ms`;
    groups.forEach((group) => {
      const row = document.createElement('div');
      row.className = 'imagehash-group';
      group.forEach((i) => {
        const thumb = document.createElement('img');
        thumb.className = 'imagehash-thumb';
        thumb.src = urls[i];
        thumb.loading = 'lazy';
        thumb.alt = 'near-duplicate';
        row.append(thumb);
      });
      results.append(row);
    });
  }

  async function load() {
    if (ready) return ready;
    ready = (async () => {
      if (config.source) {
        const json = await (await fetch(resolveUrl(config.source))).json();
        const list = Array.isArray(json.images) ? json.images : [];
        items = list.map((it) => ({ url: it.url, title: it.title || '' })).filter((it) => it.url);
      }
      if (!items.length) {
        status.textContent = 'No images to compare.';
        return false;
      }
      status.textContent = `Fingerprinting ${items.length} images…`;
      const urls = items.map((it) => resolveUrl(it.url));
      const grays = await Promise.all(urls.map(rasterizeGray));
      const flat = new Float32Array(grays.length * N * N);
      grays.forEach((g, i) => flat.set(g, i * N * N));

      const engine = await createImageHashEngine(wasmUrl());
      const hasher = engine.hasher(grays.length);
      const t0 = performance.now();
      const hashes = hasher.hashAll(flat, grays.length);
      const ms = (performance.now() - t0).toFixed(1);
      renderGroups(clusterByHamming(hashes, config.threshold), urls, ms);
      return true;
    })().catch((err) => {
      // eslint-disable-next-line no-console
      console.warn('[imagehash] unavailable', err);
      status.textContent = 'Duplicate detection unavailable.';
      return false;
    });
    return ready;
  }

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
