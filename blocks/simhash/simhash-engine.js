/* eslint-disable no-bitwise */
// simhash-engine.js — the JS <-> WASM boundary for the simhash block.
//
// `no-bitwise` is disabled here because a SimHash fingerprint IS a 64-bit word:
// hamming() counts differing bits across two 32-bit halves and toBytes() clamps
// char codes — the shifts/masks are the fingerprint math, not a pointer-math
// shortcut.
//
// This module owns everything about talking to the WASM SimHash kernel: loading
// the module, laying out one document's bytes + a 256-byte counter scratch + an
// 8-byte output slot in linear memory, calling the kernel, and reading back the
// 64-bit fingerprint as a Uint32Array(2) (low, high). The block (simhash.js)
// never touches typed arrays or wasm — it fetches documents and calls
// engine.fingerprint(bytes) / hamming(a, b).
//
// No browser globals at module top level, so Node (the benchmark) can import it.

/**
 * Lowercase `text` and encode it as one byte per character. The kernel expects
 * lowercased ASCII; any code point > 255 is folded to a space (32) so it acts
 * as a token separator. The JS baseline (bench / compare page) must encode
 * identically for byte-identical fingerprints.
 * @param {string} text
 * @returns {Uint8Array}
 */
export function toBytes(text) {
  const s = String(text).toLowerCase();
  const out = new Uint8Array(s.length);
  for (let i = 0; i < s.length; i += 1) {
    const c = s.charCodeAt(i);
    out[i] = c > 255 ? 32 : c;
  }
  return out;
}

/**
 * Hamming distance between two 64-bit fingerprints held as Uint32Array(2)
 * (index 0 = low 32 bits, index 1 = high 32 bits). This is the popcount of the
 * XOR across all 64 bits — the near-duplicate distance metric.
 * @param {Uint32Array} a fingerprint [low, high]
 * @param {Uint32Array} b fingerprint [low, high]
 * @returns {number} number of differing bits (0..64)
 */
export function hamming(a, b) {
  let dist = 0;
  for (let half = 0; half < 2; half += 1) {
    let x = (a[half] ^ b[half]) >>> 0;
    while (x !== 0) {
      x &= x - 1; // clear the lowest set bit
      dist += 1;
    }
  }
  return dist;
}

/**
 * Wire a compiled kernel instance to buffers sized for one document. All
 * allocations happen up front (alloc may grow memory and detach views), then we
 * snapshot the memory views — so repeated fingerprint() calls never re-grow
 * memory. Documents longer than maxDocLen are truncated to the doc buffer.
 * @param {WebAssembly.Instance} instance
 * @param {number} maxDocLen largest document byte length to be fingerprinted
 * @returns {{ fingerprint: (bytes: Uint8Array) => Uint32Array }}
 */
export function buildFingerprinter(instance, maxDocLen) {
  const {
    reset, alloc, simhash, memory,
  } = instance.exports;

  const cap = Math.max(1, maxDocLen | 0);
  reset();
  const docPtr = alloc(cap);
  const accPtr = alloc(64 * 4); // 256-byte scratch for the per-bit counters
  const outPtr = alloc(2 * 4); // u32[2] output: low, high
  const mem = new Uint8Array(memory.buffer); // stable: no more growth after this
  const u32 = new Uint32Array(memory.buffer);

  return {
    /**
     * @param {Uint8Array} bytes lowercased ASCII document bytes
     * @returns {Uint32Array} fingerprint [low32, high32]
     */
    fingerprint(bytes) {
      const len = Math.min(bytes.length, cap);
      mem.set(len === bytes.length ? bytes : bytes.subarray(0, len), docPtr);
      simhash(docPtr, len, accPtr, outPtr);
      const base = outPtr / 4;
      return u32.slice(base, base + 2);
    },
  };
}

/**
 * Load + instantiate the kernel. instantiateStreaming with a fetch+arrayBuffer
 * fallback for servers that don't send Content-Type: application/wasm. Throws if
 * the module can't be loaded — the block has no JS fingerprinting fallback.
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
 * High-level engine the block consumes. Loads the kernel once; fingerprinter()
 * builds a reusable fingerprinter sized to the longest document it will see.
 * @param {string} wasmUrl
 */
export async function createSimhashEngine(wasmUrl) {
  const instance = await loadWasm(wasmUrl);
  return {
    /**
     * @param {number} maxDocLen largest document byte length to be fingerprinted
     */
    fingerprinter(maxDocLen) {
      return buildFingerprinter(instance, maxDocLen);
    },
  };
}
