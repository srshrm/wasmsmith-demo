// imagefx-engine.js — the JS <-> WASM boundary for the imagefx block.
//
// Owns talking to the SIMD blur kernel: loading the module, laying out the
// src/tmp/dst RGBA buffers + the integer weight table in linear memory, and
// running the separable blur. The block (imagefx.js) never touches typed arrays
// or wasm — it rasterizes an image to pixels and calls engine.blur(pixels, r).
//
// Fixed-point integer math (weights + recip/shift) is the whole trick that keeps
// WASM and the JS twin BYTE-IDENTICAL: gaussianWeights() below is the single
// source of those integers, shared by the block, the kernel, and the benchmark.
//
// No browser globals at module top level, so Node (the benchmark) can import it.

const WSCALE = 4096; // integer weights sum to ~this; keeps sums well within i32
const SHIFT = 16; // fixed-point normalization shift (recip = round(2^SHIFT/total))
const MAX_RADIUS = 64; // slider ceiling; sizes the weight buffer

/**
 * Integer Gaussian tap weights for a given radius, plus the reciprocal/shift the
 * kernel uses to normalize. Both WASM and JS compute out = (sum*recip) >> shift,
 * so they agree to the byte. sigma = radius/2 (a soft, natural-looking blur).
 * @param {number} radius
 * @returns {{ weights: Int32Array, radius: number, recip: number, shift: number }}
 */
export function gaussianWeights(radius) {
  if (radius <= 0) {
    return {
      weights: Int32Array.of(WSCALE),
      radius: 0,
      recip: Math.round((2 ** SHIFT) / WSCALE),
      shift: SHIFT,
    };
  }
  const sigma = Math.max(radius / 2, 0.5);
  const size = 2 * radius + 1;
  const raw = new Float64Array(size);
  let sum = 0;
  for (let i = 0; i < size; i += 1) {
    const d = i - radius;
    raw[i] = Math.exp(-(d * d) / (2 * sigma * sigma));
    sum += raw[i];
  }
  const weights = new Int32Array(size);
  let total = 0;
  for (let i = 0; i < size; i += 1) {
    weights[i] = Math.round((raw[i] / sum) * WSCALE);
    total += weights[i];
  }
  return {
    weights, radius, recip: Math.round((2 ** SHIFT) / total), shift: SHIFT,
  };
}

/**
 * Wire a compiled kernel to buffers for one width×height image. All allocations
 * happen up front (alloc may grow memory and detach views) then we snapshot the
 * views, so repeated blur() calls never re-grow memory. Each buffer carries 16
 * bytes of slack so the kernel's 16-byte SIMD pixel loads never over-read past
 * the end of memory.
 * @param {WebAssembly.Instance} instance
 * @param {number} width
 * @param {number} height
 * @returns {{ blur: (pixels: Uint8Array, radius: number) => Uint8Array }}
 */
export function buildBlur(instance, width, height) {
  const {
    reset, alloc, blur, memory,
  } = instance.exports;
  const rawLen = width * height * 4;

  reset();
  const srcPtr = alloc(rawLen + 16);
  const tmpPtr = alloc(rawLen + 16);
  const dstPtr = alloc(rawLen + 16);
  const weightsPtr = alloc((2 * MAX_RADIUS + 1) * 4);
  const mem = new Uint8Array(memory.buffer); // stable: no more growth after this
  const memI32 = new Int32Array(memory.buffer);

  return {
    width,
    height,
    blur(pixels, radius) {
      const r = Math.min(Math.max(Math.trunc(radius), 0), MAX_RADIUS);
      const { weights, recip, shift } = gaussianWeights(r);
      mem.set(pixels, srcPtr);
      memI32.set(weights, weightsPtr / 4);
      blur(srcPtr, tmpPtr, dstPtr, width, height, weightsPtr, r, recip, shift);
      return mem.slice(dstPtr, dstPtr + rawLen);
    },
  };
}

/**
 * Load + instantiate the kernel. instantiateStreaming with a fetch+arrayBuffer
 * fallback for servers that don't send Content-Type: application/wasm.
 * @param {string} url
 * @returns {Promise<WebAssembly.Instance>}
 */
async function loadWasm(url) {
  const importObject = {
    env: { abort() {}, trace() {}, seed() { return Date.now(); } },
  };
  let result;
  try {
    result = await WebAssembly.instantiateStreaming(fetch(url), importObject);
  } catch (streamErr) {
    const bytes = await (await fetch(url)).arrayBuffer();
    result = await WebAssembly.instantiate(bytes, importObject);
  }
  return result.instance;
}

/**
 * High-level engine the block consumes. Loads the kernel once; blur() builds a
 * reusable blurrer for a given image size.
 * @param {string} wasmUrl
 */
export async function createImageFxEngine(wasmUrl) {
  const instance = await loadWasm(wasmUrl);
  return {
    forSize(width, height) {
      return buildBlur(instance, width, height);
    },
  };
}
