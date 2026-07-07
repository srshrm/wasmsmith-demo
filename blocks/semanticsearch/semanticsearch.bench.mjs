// semanticsearch.bench.mjs — benchmark fixture for the SIMD cosine kernel.
//
// Consumed by .claude/skills/wasmsmith/scripts/benchmark.mjs. It times a faithful
// scalar-JS cosine sweep against the SIMD WASM kernel on identical unit-normalized
// embeddings and asserts the score arrays agree within a tiny float tolerance
// (SIMD sums four partial lanes, so results differ from a sequential sum only in
// the last bits). "size" is the number of documents in the corpus.
//
// The JS reference here is a REASONABLE, tight typed-array implementation — not a
// deliberately naive one. The gap is pure SIMD lane-width: JS has no vector type.

import { buildIndex, normalizeRows } from './semanticsearch-engine.js';

const DIM = 384; // embedding dimension (MiniLM-class); multiple of 4 for SIMD

// Deterministic PRNG so corpora (and thus scores) are reproducible.
function mulberry32(seed) {
  let a = seed >>> 0;
  return () => {
    a |= 0; a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

// Build `count` embeddings drawn from a handful of latent clusters, so nearest
// neighbours are meaningful (same cluster) rather than uniform noise — closer to
// real content embeddings than pure random vectors.
function makeCorpus(count, dim, seed) {
  const rnd = mulberry32(seed);
  const clusters = 12;
  const centers = new Float32Array(clusters * dim);
  for (let i = 0; i < centers.length; i += 1) centers[i] = rnd() * 2 - 1;
  const flat = new Float32Array(count * dim);
  for (let r = 0; r < count; r += 1) {
    const c = (r % clusters) * dim;
    const base = r * dim;
    for (let i = 0; i < dim; i += 1) flat[base + i] = centers[c + i] + (rnd() * 2 - 1) * 0.35;
  }
  return normalizeRows(flat, count, dim);
}

// Faithful scalar-JS cosine baseline: dot product of the (unit) query against
// every (unit) document. Returns one score per document.
function cosineScoresJS(corpus, count, dim, query) {
  const out = new Float32Array(count);
  for (let r = 0; r < count; r += 1) {
    const base = r * dim;
    let s = 0;
    for (let i = 0; i < dim; i += 1) s += corpus[base + i] * query[i];
    out[r] = s;
  }
  return out;
}

export default {
  name: 'semanticsearch',
  description: 'SIMD cosine-similarity ranking of a query embedding over a document corpus.',
  sizes: [2000, 8000, 20000, 50000], // number of documents in the corpus
  iterations: 20, // full corpus sweeps per query, per size
  tolerance: 1e-3, // SIMD lane-fold vs sequential sum differ only in low bits

  setup(size) {
    const dim = DIM;
    const count = size;
    const corpus = makeCorpus(count, dim, 0xC0FFEE ^ size);
    // A few query vectors: reuse existing documents ("find similar to this").
    const inputs = [0, 1, 7].map((r) => corpus.slice(r * dim, r * dim + dim));
    return {
      dim, count, corpus, inputs,
    };
  },

  scoreJS(ctx, query) {
    return cosineScoresJS(ctx.corpus, ctx.count, ctx.dim, query);
  },

  createWasm(instance, ctx) {
    return buildIndex(instance, ctx.corpus, ctx.count, ctx.dim);
  },

  scoreWasm(engine, query) {
    return engine.scoreAll(query);
  },

  // Domain sample: show the top-5 neighbours of the first query at the last size.
  report(ctx, engine) {
    const hits = engine.search(ctx.inputs[0], 5);
    const list = hits.map((h) => `#${h.index}:${h.score.toFixed(3)}`).join('  ');
    console.log(`\n  ${ctx.count} docs × ${ctx.dim}d — nearest to query[0]:  ${list}`);
  },
};
