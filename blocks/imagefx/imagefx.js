// imagefx.js — live image blur, entirely client-side. Drag the radius slider and
// the image re-blurs in real time in a SIMD WebAssembly kernel — smooth where a
// per-channel JS loop would jank on a megapixel image.
//
// This file is UI only: it parses the authoring config, rasterizes the source
// image to RGBA pixels with the canvas (native), and hands those pixels to
// ./imagefx-engine.js for the blur. The engine loads LAZILY (first interaction
// or when the block scrolls into view) so the decorate path stays fast.

import { readBlockConfig } from '../../scripts/aem.js';
import { createImageFxEngine } from './imagefx-engine.js';

function wasmUrl() {
  const base = (typeof window !== 'undefined' && window.hlx && window.hlx.codeBasePath) || '';
  return `${base}/blocks/imagefx/imagefx.wasm`;
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
  let source = str(raw.source || raw.image);
  if (!source) {
    const el = block.querySelector('img[src], a[href]');
    if (el) source = el.getAttribute('src') || el.getAttribute('href');
  }
  return {
    source,
    max: num(raw.max) || 30,
    radius: num(raw.radius) || 12,
    label: str(raw.label) || 'Blur radius',
  };
}

// Rasterize an image URL to RGBA pixels via the canvas (native rendering).
function rasterize(url, maxSide) {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.crossOrigin = 'anonymous';
    img.onload = () => {
      const scale = Math.min(1, maxSide / Math.max(img.naturalWidth, img.naturalHeight || 1));
      const width = Math.max(1, Math.round((img.naturalWidth || maxSide) * scale));
      const height = Math.max(1, Math.round((img.naturalHeight || maxSide) * scale));
      const canvas = document.createElement('canvas');
      canvas.width = width;
      canvas.height = height;
      const ctx = canvas.getContext('2d');
      ctx.drawImage(img, 0, 0, width, height);
      try {
        resolve({ pixels: ctx.getImageData(0, 0, width, height).data, width, height });
      } catch (err) {
        reject(err); // tainted canvas (cross-origin image without CORS)
      }
    };
    img.onerror = () => reject(new Error('image failed to load'));
    img.src = url;
  });
}

/**
 * loads and decorates the block
 * @param {Element} block The block element
 */
export default async function decorate(block) {
  const config = readConfig(block);
  block.textContent = '';

  const controls = document.createElement('div');
  controls.className = 'imagefx-controls';
  const slider = document.createElement('input');
  slider.type = 'range';
  slider.min = '0';
  slider.max = String(config.max);
  slider.value = String(Math.min(config.radius, config.max));
  slider.className = 'imagefx-slider';
  const label = document.createElement('label');
  label.className = 'imagefx-label';
  label.textContent = `${config.label}: `;
  const readout = document.createElement('span');
  readout.className = 'imagefx-readout';
  label.append(slider, readout);
  controls.append(label);

  const canvas = document.createElement('canvas');
  canvas.className = 'imagefx-canvas';
  const status = document.createElement('p');
  status.className = 'imagefx-status';
  status.setAttribute('aria-live', 'polite');

  block.append(controls, canvas, status);

  let blurrer = null;
  let source = null; // { pixels, width, height }
  let ctx = null;
  let ready = null;
  let frame = 0;

  function render() {
    if (!blurrer || !source) return;
    const radius = parseInt(slider.value, 10);
    readout.textContent = `${radius}px`;
    const t0 = performance.now();
    const out = blurrer.blur(source.pixels, radius);
    const ms = (performance.now() - t0).toFixed(1);
    const image = new ImageData(new Uint8ClampedArray(out.buffer), source.width, source.height);
    ctx.putImageData(image, 0, 0);
    status.textContent = `${source.width}×${source.height} · blurred in ${ms} ms (WebAssembly SIMD)`;
  }

  // Coalesce rapid slider events to one blur per animation frame.
  function schedule() {
    if (frame) return;
    frame = requestAnimationFrame(() => { frame = 0; render(); });
  }

  async function load() {
    if (ready) return ready;
    ready = (async () => {
      if (!config.source) {
        status.textContent = 'No image source configured.';
        return false;
      }
      status.textContent = 'Loading image…';
      source = await rasterize(resolveUrl(config.source), 900);
      canvas.width = source.width;
      canvas.height = source.height;
      ctx = canvas.getContext('2d');
      const engine = await createImageFxEngine(wasmUrl());
      blurrer = engine.forSize(source.width, source.height);
      render();
      return true;
    })().catch((err) => {
      // eslint-disable-next-line no-console
      console.warn('[imagefx] unavailable', err);
      status.textContent = 'Live blur unavailable.';
      return false;
    });
    return ready;
  }

  slider.addEventListener('input', () => { if (blurrer) schedule(); else load(); });

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
