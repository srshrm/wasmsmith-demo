// imagehash.bench.mjs — benchmark fixture for the SIMD pHash DCT kernel.
//
// Consumed by .claude/skills/wasmsmith/scripts/benchmark.mjs. It times a faithful
// scalar-JS 2-D DCT sweep against the SIMD WASM kernel over a whole library of
// 32×32 grayscale images (one call processes the entire batch, so the JS<->WASM
// boundary cost is amortized). "size" is the number of images in the library.
//
// Correctness compares the low-frequency DCT coefficients. The WASM kernel keeps
// intermediates in f32 (SIMD) while the natural-JS twin accumulates in f64, so
// they agree within a small absolute tolerance rather than to the bit — the
// perceptual hash built from these coefficients is identical in practice.

import { buildHasher, buildCosTable, hamming } from './imagehash-engine.js';

const N = 32;
const BLK = 8;

function mulberry32(seed) {
  let a = seed >>> 0;
  return () => {
    a |= 0; a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

const clamp255 = (v) => (v < 0 ? 0 : (v > 255 ? 255 : v));

// A library of low-frequency grayscale "images" (what real photos reduce to at
// 32×32), plus a sprinkling of near-duplicates so the report can find them.
function makeLibrary(count, seed) {
  const rnd = mulberry32(seed);
  const flat = new Float32Array(count * N * N);
  for (let n = 0; n < count; n += 1) {
    const base = n * N * N;
    const fx = 1 + Math.floor(rnd() * 4);
    const fy = 1 + Math.floor(rnd() * 4);
    const phase = rnd() * 6.283;
    const amp = 40 + rnd() * 80;
    const dc = 80 + rnd() * 80;
    for (let y = 0; y < N; y += 1) {
      for (let x = 0; x < N; x += 1) {
        const val = dc + amp * Math.sin(((fx * x + fy * y) / N) * 6.283 + phase) + (rnd() - 0.5) * 10;
        flat[base + y * N + x] = clamp255(val);
      }
    }
  }
  const dupPairs = [];
  const nd = Math.max(1, Math.floor(count * 0.01));
  for (let k = 0; k < nd; k += 1) {
    const src = Math.floor(rnd() * count);
    const dst = (src + 1 + Math.floor(rnd() * (count - 1))) % count;
    for (let i = 0; i < N * N; i += 1) {
      flat[dst * N * N + i] = clamp255(flat[src * N * N + i] + (rnd() - 0.5) * 3);
    }
    dupPairs.push([src, dst]);
  }
  return { flat, dupPairs };
}

// Faithful scalar-JS 2-D DCT (top-left 8×8 block) — same passes as the kernel.
function dctAllJS(grayFlat, count, cos) {
  const out = new Float32Array(count * 64);
  const T = new Float64Array(N * BLK);
  for (let n = 0; n < count; n += 1) {
    const g = n * N * N;
    for (let y = 0; y < N; y += 1) {
      for (let u = 0; u < BLK; u += 1) {
        let s = 0;
        for (let x = 0; x < N; x += 1) s += grayFlat[g + y * N + x] * cos[x * N + u];
        T[y * BLK + u] = s;
      }
    }
    const o = n * 64;
    for (let v = 0; v < BLK; v += 1) {
      for (let u = 0; u < BLK; u += 1) {
        let s = 0;
        for (let y = 0; y < N; y += 1) s += T[y * BLK + u] * cos[y * N + v];
        out[o + v * BLK + u] = s;
      }
    }
  }
  return out;
}

export default {
  name: 'imagehash',
  description: 'SIMD pHash 2-D DCT sweep over an image library vs a scalar-JS twin.',
  sizes: [500, 2000, 5000, 10000], // number of 32×32 images in the library
  iterations: 5, // full library DCT sweeps, per size
  tolerance: 0.05, // f32 (SIMD) vs f64 (JS) DCT accumulation (measured maxΔ ~1.5e-3)

  setup(size) {
    const cos = buildCosTable();
    const { flat, dupPairs } = makeLibrary(size, 0xF00D ^ size);
    return {
      count: size, cos, dupPairs, inputs: [flat],
    };
  },

  scoreJS(ctx, flat) {
    return dctAllJS(flat, ctx.count, ctx.cos);
  },

  createWasm(instance, ctx) {
    const base = buildHasher(instance, ctx.count);
    return {
      dctAll: (flat) => base.dctAll(flat, ctx.count),
      hashAll: (flat) => base.hashAll(flat, ctx.count),
    };
  },

  scoreWasm(engine, flat) {
    return engine.dctAll(flat);
  },

  // Domain sample: fingerprint the library and report near-duplicates found
  // (Hamming <= 8), cross-checking against the pairs we injected.
  report(ctx, engine) {
    const hashes = engine.hashAll(ctx.inputs[0]);
    let found = 0;
    ctx.dupPairs.forEach(([a, b]) => { if (hamming(hashes[a], hashes[b]) <= 8) found += 1; });
    console.log(`\n  ${ctx.count} images fingerprinted · injected ${ctx.dupPairs.length} near-dup pairs · detected ${found} (Hamming ≤ 8)`);
  },
};
