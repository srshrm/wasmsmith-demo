// semanticsearch-engine.js — the JS <-> WASM boundary for the semanticsearch block.
//
// Owns everything about talking to the SIMD cosine kernel: loading the module,
// laying out the flat f32 corpus + query in linear memory, and reading back
// either every score (cosine) or just the k nearest neighbours (cosineTopK).
// The block (semanticsearch.js) never touches typed arrays or wasm — it fetches
// embeddings, builds the index once, and calls index.search(queryVec, k).
//
// No browser globals at module top level, so Node (the benchmark) can import it.

const MAX_K = 64; // cosineTopK output buffer holds at most this many neighbours

/**
 * L2-normalize each row of a flat [count * dim] Float32Array IN PLACE, so cosine
 * similarity reduces to a plain dot product. Shared by the block and benchmark
 * (and mirrored on the query) — this is the contract the kernel assumes.
 * @param {Float32Array} flat row-major vectors, length count*dim
 * @param {number} count
 * @param {number} dim
 */
export function normalizeRows(flat, count, dim) {
  for (let r = 0; r < count; r += 1) {
    const base = r * dim;
    let sum = 0;
    for (let i = 0; i < dim; i += 1) sum += flat[base + i] * flat[base + i];
    const inv = sum > 0 ? 1 / Math.sqrt(sum) : 0;
    for (let i = 0; i < dim; i += 1) flat[base + i] *= inv;
  }
  return flat;
}

/** L2-normalize a single vector in place. */
export function normalizeVec(vec) {
  let sum = 0;
  for (let i = 0; i < vec.length; i += 1) sum += vec[i] * vec[i];
  const inv = sum > 0 ? 1 / Math.sqrt(sum) : 0;
  for (let i = 0; i < vec.length; i += 1) vec[i] *= inv;
  return vec;
}

/**
 * Wire a compiled kernel to a fixed corpus. All allocations happen up front
 * (alloc may grow memory and detach views) and we snapshot the typed-array views
 * afterward, so repeated queries never re-grow memory. The corpus is written
 * once; each query only writes `dim` floats.
 * @param {WebAssembly.Instance} instance
 * @param {Float32Array} corpus unit-normalized, row-major, length count*dim
 * @param {number} count number of documents
 * @param {number} dim embedding dimension
 */
export function buildIndex(instance, corpus, count, dim) {
  const {
    reset, alloc, cosine, cosineTopK, memory,
  } = instance.exports;

  reset();
  const corpusPtr = alloc(count * dim * 4);
  const queryPtr = alloc(dim * 4);
  const scoresPtr = alloc(count * 4);
  const topkPtr = alloc(MAX_K * 8);
  const memF32 = new Float32Array(memory.buffer); // stable: no more growth after this
  const memI32 = new Int32Array(memory.buffer);

  memF32.set(corpus, corpusPtr / 4); // write the corpus exactly once

  return {
    count,
    dim,

    // Full scores — used by the benchmark to compare against the JS baseline.
    scoreAll(query) {
      memF32.set(query, queryPtr / 4);
      cosine(queryPtr, corpusPtr, count, dim, scoresPtr);
      return memF32.slice(scoresPtr / 4, scoresPtr / 4 + count);
    },

    // k nearest neighbours — used by the block. Only the winners cross back.
    search(query, k) {
      const kk = Math.min(k, MAX_K);
      memF32.set(query, queryPtr / 4);
      const n = cosineTopK(queryPtr, corpusPtr, count, dim, kk, topkPtr);
      const base = topkPtr / 4; // 8-byte slots => 2 f32 slots each
      const out = [];
      for (let i = 0; i < n; i += 1) {
        out.push({ index: memI32[base + i * 2], score: memF32[base + i * 2 + 1] });
      }
      return out;
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
 * High-level engine the block consumes. Loads the kernel once; index() builds a
 * reusable nearest-neighbour index over a document embedding matrix.
 * @param {string} wasmUrl
 */
export async function createSemanticEngine(wasmUrl) {
  const instance = await loadWasm(wasmUrl);
  return {
    /**
     * @param {Float32Array} corpus unit-normalized, row-major, length count*dim
     * @param {number} count
     * @param {number} dim
     */
    index(corpus, count, dim) {
      return buildIndex(instance, corpus, count, dim);
    },
  };
}
