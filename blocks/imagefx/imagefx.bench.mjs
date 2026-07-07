// imagefx.bench.mjs — benchmark fixture for the SIMD Gaussian-blur kernel.
//
// Consumed by .claude/skills/wasmsmith/scripts/benchmark.mjs. It times a faithful
// scalar-JS separable blur against the SIMD WASM kernel on identical RGBA images
// at a fixed radius and asserts the outputs are BYTE-IDENTICAL (tolerance 0) —
// both paths use the same integer weights + reciprocal/shift from the engine's
// gaussianWeights(). "size" is the image side length in pixels.
//
// The JS twin is a reasonable tight typed-array implementation (per-channel
// integer multiply-add), NOT a naive one. The gap is pure SIMD lane width: the
// kernel multiply-accumulates the 4 RGBA channels at once, JS one at a time.

import { buildBlur, gaussianWeights } from './imagefx-engine.js';

const RADIUS = 16; // blur radius used for the comparison (33-tap separable pass)

function mulberry32(seed) {
  let a = seed >>> 0;
  return () => {
    a |= 0; a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

// A photo-like image: smooth gradients plus some structured detail, so the blur
// has real work to average (not a flat field).
function makeImage(side, seed) {
  const rnd = mulberry32(seed);
  const px = new Uint8Array(side * side * 4);
  for (let y = 0; y < side; y += 1) {
    for (let x = 0; x < side; x += 1) {
      const o = (y * side + x) * 4;
      px[o] = (x * 255 / side) | 0;
      px[o + 1] = (y * 255 / side) | 0;
      px[o + 2] = ((x ^ y) & 0xff);
      px[o + 3] = 255;
      if (rnd() < 0.05) px[o + 1] = 255 - px[o + 1]; // sparse structured noise
    }
  }
  return px;
}

// Faithful scalar-JS separable Gaussian blur, byte-exact with the kernel.
function blurJS(pixels, width, height, radius) {
  const rawLen = width * height * 4;
  if (radius <= 0) return pixels.slice(0, rawLen);
  const { weights, recip, shift } = gaussianWeights(radius);
  const taps = 2 * radius + 1;
  const tmp = new Uint8Array(rawLen);
  const dst = new Uint8Array(rawLen);
  const clamp = (c) => (c < 0 ? 0 : (c > 255 ? 255 : c));

  for (let y = 0; y < height; y += 1) {
    const rowBase = y * width;
    for (let x = 0; x < width; x += 1) {
      let a0 = 0; let a1 = 0; let a2 = 0; let a3 = 0;
      for (let k = 0; k < taps; k += 1) {
        let xc = x + k - radius;
        if (xc < 0) xc = 0; else if (xc > width - 1) xc = width - 1;
        const p = (rowBase + xc) * 4;
        const w = weights[k];
        a0 += pixels[p] * w; a1 += pixels[p + 1] * w;
        a2 += pixels[p + 2] * w; a3 += pixels[p + 3] * w;
      }
      const d = (rowBase + x) * 4;
      tmp[d] = clamp((a0 * recip) >> shift); tmp[d + 1] = clamp((a1 * recip) >> shift);
      tmp[d + 2] = clamp((a2 * recip) >> shift); tmp[d + 3] = clamp((a3 * recip) >> shift);
    }
  }
  for (let y = 0; y < height; y += 1) {
    for (let x = 0; x < width; x += 1) {
      let a0 = 0; let a1 = 0; let a2 = 0; let a3 = 0;
      for (let k = 0; k < taps; k += 1) {
        let yc = y + k - radius;
        if (yc < 0) yc = 0; else if (yc > height - 1) yc = height - 1;
        const p = ((yc * width) + x) * 4;
        const w = weights[k];
        a0 += tmp[p] * w; a1 += tmp[p + 1] * w;
        a2 += tmp[p + 2] * w; a3 += tmp[p + 3] * w;
      }
      const d = ((y * width) + x) * 4;
      dst[d] = clamp((a0 * recip) >> shift); dst[d + 1] = clamp((a1 * recip) >> shift);
      dst[d + 2] = clamp((a2 * recip) >> shift); dst[d + 3] = clamp((a3 * recip) >> shift);
    }
  }
  return dst;
}

export default {
  name: 'imagefx',
  description: 'SIMD separable Gaussian blur of RGBA images vs a scalar-JS twin (byte-exact).',
  sizes: [128, 256, 512, 768], // image side length in pixels
  iterations: 6, // full blurs per input, per size
  tolerance: 0, // byte-identical: same integer weights + reciprocal/shift both sides

  setup(size) {
    const width = size;
    const height = size;
    const radius = RADIUS;
    const inputs = [makeImage(size, 0xA11CE ^ size), makeImage(size, 0xB0B ^ size)];
    return {
      width, height, radius, inputs,
    };
  },

  scoreJS(ctx, pixels) {
    return blurJS(pixels, ctx.width, ctx.height, ctx.radius);
  },

  createWasm(instance, ctx) {
    const base = buildBlur(instance, ctx.width, ctx.height);
    return { blur: (pixels) => base.blur(pixels, ctx.radius) };
  },

  scoreWasm(engine, pixels) {
    return engine.blur(pixels);
  },

  report(ctx) {
    const taps = 2 * ctx.radius + 1;
    console.log(`\n  radius ${ctx.radius} (${taps}-tap separable) · ${ctx.width}×${ctx.height} · 4 channels`);
  },
};
