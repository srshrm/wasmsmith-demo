// svgpng.js — convert an SVG to a PNG, entirely client-side.
//
// This file is the block's UI only: it parses the authoring config, rasterizes
// the SVG to RGBA pixels with the browser canvas (native), then hands those
// pixels to ./svgpng-engine.js, which encodes them into PNG bytes in WebAssembly.
//
// The engine is created LAZILY (first click, or when the block scrolls into
// view) so the synchronous decorate path — and the Lighthouse score — stay fast.

import { readBlockConfig } from '../../scripts/aem.js';
import { createSvgPngEngine } from './svgpng-engine.js';

function wasmUrl() {
  const base = (typeof window !== 'undefined' && window.hlx && window.hlx.codeBasePath) || '';
  return `${base}/blocks/svgpng/svgpng.wasm`;
}

function resolveUrl(src) {
  const base = (window.hlx && window.hlx.codeBasePath) || '';
  return /^(https?:|data:|blob:)/.test(src) ? src : `${base}${src}`;
}

// Pull raw rows with the platform helper, then coerce to typed config.
function readConfig(block) {
  const raw = readBlockConfig(block);
  const str = (v) => (Array.isArray(v) ? v.join(', ') : `${v ?? ''}`).trim();
  const num = (v) => {
    const n = parseInt(str(v), 10);
    return Number.isFinite(n) && n > 0 ? n : 0;
  };
  let source = str(raw.source || raw.svg);
  if (!source) {
    const el = block.querySelector('a[href$=".svg"], img[src$=".svg"]');
    if (el) source = el.getAttribute('href') || el.getAttribute('src');
  }
  return {
    source,
    width: num(raw.width),
    height: num(raw.height),
    label: str(raw.label) || 'Convert & download PNG',
  };
}

// Rasterize an SVG URL to RGBA pixels via the canvas (native rendering).
function rasterize(svgUrl, targetW, targetH) {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.crossOrigin = 'anonymous';
    img.onload = () => {
      const width = targetW || img.naturalWidth || 512;
      const height = targetH || img.naturalHeight || 512;
      const canvas = document.createElement('canvas');
      canvas.width = width;
      canvas.height = height;
      const ctx = canvas.getContext('2d');
      ctx.drawImage(img, 0, 0, width, height);
      try {
        const { data } = ctx.getImageData(0, 0, width, height);
        resolve({ pixels: data, width, height });
      } catch (err) {
        reject(err); // tainted canvas (cross-origin SVG without CORS)
      }
    };
    img.onerror = () => reject(new Error('SVG failed to load'));
    img.src = svgUrl;
  });
}

/**
 * loads and decorates the block
 * @param {Element} block The block element
 */
export default async function decorate(block) {
  const config = readConfig(block);
  block.textContent = '';

  const button = document.createElement('button');
  button.type = 'button';
  button.className = 'svgpng-convert';
  button.textContent = config.label;

  const status = document.createElement('p');
  status.className = 'svgpng-status';
  status.setAttribute('aria-live', 'polite');

  const preview = document.createElement('div');
  preview.className = 'svgpng-preview';

  block.append(button, status, preview);

  let enginePromise = null;
  const getEngine = () => {
    if (!enginePromise) enginePromise = createSvgPngEngine(wasmUrl());
    return enginePromise;
  };

  let lastUrl = null;
  function showResult(png, width, height, ms) {
    if (lastUrl) URL.revokeObjectURL(lastUrl);
    preview.textContent = '';
    const url = URL.createObjectURL(new Blob([png], { type: 'image/png' }));
    lastUrl = url;
    const kb = (png.length / 1024).toFixed(1);

    const out = document.createElement('img');
    out.className = 'svgpng-out';
    out.src = url;
    out.alt = 'PNG output';

    const dl = document.createElement('a');
    dl.className = 'svgpng-download';
    dl.href = url;
    dl.download = 'converted.png';
    dl.textContent = `Download PNG (${kb} KB)`;

    preview.append(out, dl);
    status.textContent = `PNG ready · ${width}×${height} · ${kb} KB · encoded in ${ms} ms`;
  }

  async function convert() {
    if (!config.source) {
      status.textContent = 'No SVG source configured.';
      return;
    }
    button.disabled = true;
    status.textContent = 'Rasterizing SVG…';
    try {
      const url = resolveUrl(config.source);
      const { pixels, width, height } = await rasterize(url, config.width, config.height);
      status.textContent = `Encoding ${width}×${height} in WebAssembly…`;
      const engine = await getEngine();
      const t0 = performance.now();
      const png = engine.toPng(pixels, width, height);
      showResult(png, width, height, (performance.now() - t0).toFixed(1));
    } catch (err) {
      // eslint-disable-next-line no-console
      console.warn('[svgpng] conversion failed', err);
      status.textContent = 'PNG conversion unavailable.';
    } finally {
      button.disabled = false;
    }
  }

  button.addEventListener('click', convert);

  // Warm up the engine when the block scrolls into view, before any click.
  if (typeof IntersectionObserver !== 'undefined') {
    const io = new IntersectionObserver((entries) => {
      if (entries.some((entry) => entry.isIntersecting)) {
        io.disconnect();
        getEngine();
      }
    }, { rootMargin: '200px' });
    io.observe(block);
  }
}
