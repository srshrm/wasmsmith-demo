// fuzzysearch.bench.mjs — benchmark fixture for the "elastic" fuzzysearch kernel.
//
// Consumed by .claude/skills/wasmsmith/scripts/benchmark.mjs. Exports a "bench
// spec" describing how to build input, run the JS baseline, and run the WASM
// engine on identical data. The harness handles timing + the correctness diff.
//
// The block is WASM-only, so the JS REFERENCE implementation of the scoring
// algorithm lives HERE. It must stay a faithful twin of computeScores() in
// assembly/fuzzysearch.ts — the OSA edit distance and the BM25 math — so the
// harness can assert the two agree on identical input. The index build and the
// marshaling primitives are imported from the shipped engine so the benchmark
// exercises the real code path (only the SCORING is re-implemented here).
//
// Not linted (eslint . only covers .js), so bitwise / for-of are fine.

import {
  buildIndex, buildWasmEngine, tokenize,
  MAX_TERM, DEFAULT_K1, DEFAULT_B, DEFAULT_MAX_EDITS,
} from './search-engine.js';

// ── JS reference (twin of assembly/fuzzysearch.ts) ───────────────────────────

// Restricted Damerau–Levenshtein (OSA) with the same length filter + row-min
// early exit as the kernel. `a` holds one query term's codes; `b` is the shared
// vocab-chars array, term chars at [bBase, bBase+bLen).
function osaJS(a, aLen, b, bBase, bLen, maxEdits, dp) {
  const diff = aLen > bLen ? aLen - bLen : bLen - aLen;
  if (diff > maxEdits) return maxEdits + 1;
  if (aLen === 0) return bLen;
  if (bLen === 0) return aLen;
  const W = bLen + 1;
  for (let j = 0; j <= bLen; j += 1) dp[j] = j;
  for (let i = 1; i <= aLen; i += 1) {
    const rowBase = i * W;
    dp[rowBase] = i;
    const ai = a[i - 1];
    let rowMin = i;
    for (let j = 1; j <= bLen; j += 1) {
      const bj = b[bBase + j - 1];
      const cost = ai === bj ? 0 : 1;
      let v = dp[(i - 1) * W + j] + 1;
      const ins = dp[rowBase + j - 1] + 1;
      if (ins < v) v = ins;
      const sub = dp[(i - 1) * W + j - 1] + cost;
      if (sub < v) v = sub;
      if (i > 1 && j > 1) {
        const aPrev = a[i - 2];
        const bPrev = b[bBase + j - 2];
        if (ai === bPrev && aPrev === bj) {
          const tr = dp[(i - 2) * W + j - 2] + 1;
          if (tr < v) v = tr;
        }
      }
      dp[rowBase + j] = v;
      if (v < rowMin) rowMin = v;
    }
    if (rowMin > maxEdits) return maxEdits + 1;
  }
  return dp[aLen * W + bLen];
}

// Full per-document fuzzy-BM25 scores for one query (array of term strings).
// The kernel reads avgdl/k1/b as f32 from the header, so we fround them here to
// agree to the bit; idf is the SAME precomputed f32 table both sides read.
function scoreAllJS(index, queryTerms, params) {
  const {
    vocabChars, vocabOffsets, postOffsets, postDocs, postTf, idf, docLen,
    docCount, vocabCount,
  } = index;
  const k1 = Math.fround(params.k1);
  const b = Math.fround(params.b);
  const maxEditsCap = params.maxEdits;
  const avgdl = Math.fround(index.avgdl);
  const invAvg = avgdl > 0 ? 1 / avgdl : 0;
  const scores = new Float32Array(docCount); // f32 store => matches kernel rounding
  const best = new Float32Array(docCount);
  const dp = new Int32Array((MAX_TERM + 1) * (MAX_TERM + 1));

  for (let qi = 0; qi < queryTerms.length; qi += 1) {
    const term = queryTerms[qi];
    const qLen = term.length;
    if (qLen === 0) continue;
    const a = new Uint16Array(qLen);
    for (let c = 0; c < qLen; c += 1) a[c] = term.charCodeAt(c);

    let me = qLen <= 2 ? 0 : (qLen <= 5 ? 1 : 2);
    if (me > maxEditsCap) me = maxEditsCap;
    const meDenom = me + 1;

    best.fill(0);
    let matched = false;

    for (let t = 0; t < vocabCount; t += 1) {
      const tStart = vocabOffsets[t];
      const tEnd = vocabOffsets[t + 1];
      const dist = osaJS(a, qLen, vocabChars, tStart, tEnd - tStart, me, dp);
      if (dist > me) continue;
      matched = true;
      const pStart = postOffsets[t];
      const pEnd = postOffsets[t + 1];
      const weight = (1 - dist / meDenom) * idf[t];
      for (let p = pStart; p < pEnd; p += 1) {
        const doc = postDocs[p];
        const tf = postTf[p];
        const dl = docLen[doc];
        const denom = tf + k1 * (1 - b + b * dl * invAvg);
        const s = weight * (tf * (k1 + 1)) / denom;
        if (s > best[doc]) best[doc] = s;
      }
    }

    if (matched) {
      for (let d = 0; d < docCount; d += 1) {
        const bv = best[d];
        if (bv > 0) scores[d] += bv;
      }
    }
  }
  return scores;
}

function topK(scores, k) {
  const out = [];
  for (let i = 0; i < scores.length; i += 1) {
    if (scores[i] > 0) out.push(i);
  }
  out.sort((x, y) => (scores[y] - scores[x]) || (x - y));
  return out.slice(0, k);
}

// ── synthetic corpus with a large vocabulary ─────────────────────────────────

function mulberry32(seed) {
  let a = seed >>> 0;
  return () => {
    a |= 0;
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

// Real-ish domain words: query typos target these, and they are sampled often so
// they end up common in the corpus (interesting IDF + fuzzy matches).
const REAL_WORDS = [
  'edge', 'delivery', 'services', 'search', 'index', 'performance', 'optimization',
  'authoring', 'markup', 'section', 'fragment', 'component', 'tutorial', 'reference',
  'overview', 'configuration', 'sidekick', 'preview', 'publish', 'document',
  'metadata', 'placeholder', 'sitemap', 'redirect', 'analytics', 'experiment',
  'personalization', 'accessibility', 'lighthouse', 'caching', 'security',
  'developer', 'content', 'javascript', 'stylesheet', 'responsive', 'template',
];

const CONS = 'bcdfghklmnprstvw';
const VOW = 'aeiou';

// Build a deterministic pool of pseudo-words + the real words, then sample docs
// from it with a Zipf-ish bias so a realistic vocabulary emerges.
function makeCorpus(size) {
  const rnd = mulberry32(0x9e3779b1 ^ size);
  const poolSize = Math.min(2000 + size, 18000);
  const pool = REAL_WORDS.slice();
  while (pool.length < poolSize) {
    const len = 4 + Math.floor(rnd() * 7); // 4..10
    let w = '';
    for (let i = 0; i < len; i += 1) {
      w += (i % 2 === 0 ? CONS[Math.floor(rnd() * CONS.length)] : VOW[Math.floor(rnd() * VOW.length)]);
    }
    pool.push(w);
  }

  // Zipf-ish pick: bias toward the front of the pool (real words + low indices).
  const pick = () => {
    const r = rnd();
    const idx = Math.floor((r * r) * pool.length); // squaring skews toward 0
    return pool[Math.min(idx, pool.length - 1)];
  };

  const texts = new Array(size);
  const labels = new Array(size);
  for (let i = 0; i < size; i += 1) {
    const titleN = 3 + Math.floor(rnd() * 3);
    const bodyN = 15 + Math.floor(rnd() * 25);
    const title = [];
    for (let j = 0; j < titleN; j += 1) title.push(pick());
    const body = [];
    for (let j = 0; j < bodyN; j += 1) body.push(pick());
    labels[i] = title.join(' ');
    texts[i] = `${labels[i]} ${body.join(' ')}`;
  }
  return { texts, labels };
}

// Query phrases with deliberate typos (transpositions, drops) that AUTO
// fuzziness should still resolve to the intended real words.
const QUERY_PHRASES = [
  'serch index',
  'optimizaton',
  'performnce',
  'edge deliery servces',
  'confguration',
  'accessibilty lighthouse',
  'javascrpt template',
];

export default {
  name: 'fuzzysearch',
  description: 'Typo-tolerant fuzzy matching (edit distance) + BM25 ranking, on-device.',
  sizes: [2000, 8000, 20000, 50000],
  iterations: 12, // full-corpus scorings per query, per size
  tolerance: 1e-4, // byte-identical in practice (shared f32 idf, fround'd params)

  setup(size) {
    const { texts, labels } = makeCorpus(size);
    const index = buildIndex(texts);
    const params = { k1: DEFAULT_K1, b: DEFAULT_B, maxEdits: DEFAULT_MAX_EDITS };
    const inputs = QUERY_PHRASES.map((q) => tokenize(q));
    return {
      index, params, inputs, labels, queryPhrases: QUERY_PHRASES,
    };
  },

  // JS baseline: score the whole corpus for one query (array of term strings).
  scoreJS(ctx, queryTerms) {
    return scoreAllJS(ctx.index, queryTerms, ctx.params);
  },

  // Build a reusable WASM engine once per size.
  createWasm(instance, ctx) {
    return buildWasmEngine(instance, ctx.index, ctx.params);
  },

  scoreWasm(engine, queryTerms) {
    return engine.scoreAll(queryTerms);
  },

  // Domain sample: show a typo query still resolves to the right documents.
  report(ctx, engine) {
    const qi = 3; // 'edge deliery servces' — three typos
    const scores = engine.scoreAll(ctx.inputs[qi]);
    const top = topK(scores, 5);
    console.log(`\n  vocab: ${ctx.index.vocabCount} terms · avgdl ${ctx.index.avgdl.toFixed(1)}`);
    console.log(`  typo query "${ctx.queryPhrases[qi]}" → top matches (WASM, ${ctx.labels.length} docs):`);
    top.forEach((i) => console.log(`    ${scores[i].toFixed(3).padStart(8)}  ${ctx.labels[i]}`));
  },
};
