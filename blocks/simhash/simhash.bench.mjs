// simhash.bench.mjs — benchmark fixture for the SimHash fingerprint kernel.
//
// Consumed by .claude/skills/wasmsmith/scripts/benchmark.mjs. It times a
// faithful JS baseline against the WASM kernel on identical documents and
// asserts the 64-bit fingerprints agree EXACTLY (tolerance 0).
//
// WHY THE JS BASELINE USES BigInt: the kernel computes a 64-bit FNV-1a hash of
// every token. JS numbers are IEEE-754 f64 — they cannot hold exact 64-bit
// integers, so a correct JS twin has no choice but to use BigInt (heap-allocated
// arbitrary-precision integers). BigInt XOR/multiply is ~15-40× slower than the
// native i64/u64 the WASM kernel uses. This is the whole point of the block:
// the gap is the language's missing 64-bit integer type, not a naive baseline.
// (.bench.mjs files are not linted, so BigInt/bitwise are fine here.)
//
// "size" is the approximate document length in words.

import { buildFingerprinter, toBytes } from './simhash-engine.js';

// Deterministic PRNG so documents (and thus fingerprints) are reproducible.
function mulberry32(seed) {
  let a = seed >>> 0;
  return () => {
    a |= 0; a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

// A small vocabulary of lowercase [a-z0-9] words; documents are drawn from it so
// tokens repeat the way real prose does (which is what SimHash exploits).
const VOCAB = [
  'the', 'quick', 'brown', 'fox', 'jumps', 'over', 'lazy', 'dog', 'content',
  'search', 'index', 'canonical', 'duplicate', 'page', 'title', 'document',
  'edge', 'delivery', 'adobe', 'experience', 'manager', 'wasm', 'kernel',
  'hash', 'token', 'vector', 'similarity', 'cluster', 'fingerprint', 'bits',
  'model', 'data', 'query', 'result', 'score', 'rank', 'match', 'near', 'dup',
  'v2', 'v3', 'x64', 'utf8', 'html', 'json', 'ascii', 'byte', 'word', 'text',
];

// Build a pseudo-random document of ~`words` words, space-separated, with a
// little punctuation so the tokenizer has real separators to skip.
function makeDoc(words, seed) {
  const rnd = mulberry32(seed);
  const parts = [];
  for (let i = 0; i < words; i += 1) {
    const w = VOCAB[Math.floor(rnd() * VOCAB.length)];
    parts.push(w);
    if (rnd() < 0.08) parts.push('.\n'); // sentence break
  }
  return parts.join(' ');
}

// ── JS baseline: SimHash via 64-bit FNV-1a in BigInt (the honest twin) ────────
const MASK64 = (1n << 64n) - 1n;
const FNV_OFFSET = 0xcbf29ce484222325n;
const FNV_PRIME = 0x100000001b3n;

function isTokenByte(c) {
  return (c >= 97 && c <= 122) || (c >= 48 && c <= 57);
}

function simhashJS(docBytes) {
  const counters = new Int32Array(64);
  const len = docBytes.length;
  let i = 0;
  while (i < len) {
    if (isTokenByte(docBytes[i])) {
      // FNV-1a over this maximal token run, in exact 64-bit BigInt.
      let h = FNV_OFFSET;
      let j = i;
      while (j < len && isTokenByte(docBytes[j])) {
        h = (h ^ BigInt(docBytes[j])) & MASK64;
        h = (h * FNV_PRIME) & MASK64;
        j += 1;
      }
      for (let b = 0; b < 64; b += 1) {
        const bit = (h >> BigInt(b)) & 1n;
        counters[b] += bit === 1n ? 1 : -1;
      }
      i = j;
    } else {
      i += 1;
    }
  }
  // Collapse counters into a 64-bit fingerprint, emit as [low32, high32].
  let low = 0;
  let high = 0;
  for (let b = 0; b < 64; b += 1) {
    if (counters[b] > 0) {
      if (b < 32) low |= (1 << b);
      else high |= (1 << (b - 32));
    }
  }
  return new Uint32Array([low >>> 0, high >>> 0]);
}

function toHex(fp) {
  const hi = fp[1].toString(16).padStart(8, '0');
  const lo = fp[0].toString(16).padStart(8, '0');
  return `0x${hi}${lo}`;
}

export default {
  name: 'simhash',
  description: 'SimHash content fingerprint via 64-bit FNV-1a; JS baseline is BigInt (no i64).',
  sizes: [200, 800, 2000, 5000], // approximate document length in words
  iterations: 30, // fingerprints per document, per size
  tolerance: 0, // fingerprints must be byte-identical

  setup(size) {
    const seedBase = 0xC0FFEE ^ size;
    // A few deterministic documents of roughly `size` words each.
    const inputs = [0, 1, 2].map((k) => toBytes(makeDoc(size, seedBase + k * 101)));
    const maxDocLen = inputs.reduce((m, b) => Math.max(m, b.length), 0);
    return { size, inputs, maxDocLen };
  },

  scoreJS(ctx, docBytes) {
    return simhashJS(docBytes);
  },

  createWasm(instance, ctx) {
    return buildFingerprinter(instance, ctx.maxDocLen);
  },

  scoreWasm(engine, docBytes) {
    return engine.fingerprint(docBytes);
  },

  // Domain sample: print the fingerprint of the first document at the last size.
  report(ctx, engine) {
    const fp = engine.fingerprint(ctx.inputs[0]);
    console.log(`\n  ${ctx.size}-word doc — fingerprint[0]: ${toHex(fp)}`);
  },
};
