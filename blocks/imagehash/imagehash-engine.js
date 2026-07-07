/* eslint-disable no-bitwise */
// imagehash-engine.js — the JS <-> WASM boundary for the imagehash block.
//
// `no-bitwise` is enabled here because perceptual-hash Hamming distance is
// popcount over XORed 64-bit fingerprints — the shifts/masks ARE the algorithm.
//
// Owns talking to the SIMD pHash kernel: building the cosine table, laying out
// the grayscale library + scratch buffers in linear memory, running the DCT
// sweep, and reading back one 64-bit fingerprint per image. The block never
// touches typed arrays or wasm — it downscales images on the canvas (native),
// converts to grayscale, and calls hasher.hashAll(gray, count).
//
// No browser globals at module top level, so Node (the benchmark) can import it.

export const N = 32; // reduced grayscale side
const BLK = 8; // low-frequency block side => 64-bit hash

/**
 * Build the DCT-II cosine matrix with orthonormal scaling folded in:
 *   C[i][j] = alpha(j) * cos(pi*(2i+1)*j / (2N)),  alpha(0)=sqrt(1/N), else sqrt(2/N)
 * The same table drives both separable passes, keeping coefficients bounded.
 * @returns {Float32Array} length N*N, row-major C[i][j]
 */
export function buildCosTable() {
  const c = new Float32Array(N * N);
  const a0 = Math.sqrt(1 / N);
  const aj = Math.sqrt(2 / N);
  for (let i = 0; i < N; i += 1) {
    for (let j = 0; j < N; j += 1) {
      const alpha = j === 0 ? a0 : aj;
      c[i * N + j] = alpha * Math.cos((Math.PI * (2 * i + 1) * j) / (2 * N));
    }
  }
  return c;
}

/**
 * Convert an N*N RGBA block (Uint8) to N*N grayscale floats (Rec. 601 luma).
 * @param {Uint8Array|Uint8ClampedArray} rgba length N*N*4
 * @returns {Float32Array} length N*N
 */
export function toGray(rgba) {
  const g = new Float32Array(N * N);
  for (let i = 0; i < N * N; i += 1) {
    const o = i * 4;
    g[i] = 0.299 * rgba[o] + 0.587 * rgba[o + 1] + 0.114 * rgba[o + 2];
  }
  return g;
}

/**
 * Hamming distance between two 64-bit fingerprints (each a Uint32Array [lo, hi]).
 * @returns {number} number of differing bits (0..64)
 */
export function hamming(a, b) {
  let x = (a[0] ^ b[0]) >>> 0;
  let y = (a[1] ^ b[1]) >>> 0;
  let count = 0;
  while (x) { x &= x - 1; count += 1; }
  while (y) { y &= y - 1; count += 1; }
  return count;
}

/**
 * Wire a compiled kernel to buffers sized for up to `maxCount` images. All
 * allocations happen up front (alloc may grow memory) then we snapshot the views.
 * The cosine table is written once.
 * @param {WebAssembly.Instance} instance
 * @param {number} maxCount max images processed per call
 */
export function buildHasher(instance, maxCount) {
  const {
    reset, alloc, dctBatch, phashBatch, memory,
  } = instance.exports;

  reset();
  const grayPtr = alloc(maxCount * N * N * 4);
  const cosPtr = alloc(N * N * 4);
  const tmpPtr = alloc(N * BLK * 4);
  const coeffAllPtr = alloc(maxCount * 64 * 4); // batch DCT output
  const sortPtr = alloc(64 * 4);
  const outPtr = alloc(maxCount * 8);
  const memF32 = new Float32Array(memory.buffer); // stable after this point
  const memU32 = new Uint32Array(memory.buffer);

  memF32.set(buildCosTable(), cosPtr / 4);

  return {
    // Batch DCT coefficients (count*64) — used by the benchmark's correctness cmp.
    dctAll(grayFlat, count) {
      memF32.set(grayFlat, grayPtr / 4);
      dctBatch(grayPtr, count, cosPtr, tmpPtr, coeffAllPtr);
      return memF32.slice(coeffAllPtr / 4, coeffAllPtr / 4 + count * 64);
    },

    // One 64-bit fingerprint per image — used by the block. Only the hashes
    // cross back (count*8 bytes), never the coefficients.
    hashAll(grayFlat, count) {
      memF32.set(grayFlat, grayPtr / 4);
      phashBatch(grayPtr, count, cosPtr, tmpPtr, coeffAllPtr, sortPtr, outPtr);
      const base = outPtr / 4;
      const out = [];
      for (let i = 0; i < count; i += 1) {
        out.push(Uint32Array.of(memU32[base + i * 2], memU32[base + i * 2 + 1]));
      }
      return out;
    },
  };
}

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
 * High-level engine the block consumes. Loads the kernel once; hasher() builds a
 * reusable fingerprinter for up to maxCount images.
 * @param {string} wasmUrl
 */
export async function createImageHashEngine(wasmUrl) {
  const instance = await loadWasm(wasmUrl);
  return {
    hasher(maxCount) {
      return buildHasher(instance, maxCount);
    },
  };
}
