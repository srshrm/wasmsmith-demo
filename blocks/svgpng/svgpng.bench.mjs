// svgpng.bench.mjs — benchmark fixture for the svgpng PNG-encoder kernel.
//
// Consumed by .claude/skills/wasmsmith/scripts/benchmark.mjs. It times the JS twin
// encoder against the WASM kernel on identical RGBA images and asserts they agree
// BYTE-FOR-BYTE (tolerance 0). "size" here is the image side length in pixels.
//
// The JS reference encoder lives in test/png-encoder-js.js (shared with the
// comparison page); the WASM marshaling comes from the block's engine — so the
// benchmark exercises exactly the code paths that ship.

import { buildEncoder } from './svgpng-engine.js';
import encodeZlibJs from '../../test/png-encoder-js.js';

// Deterministic PRNG so images (and thus outputs) are reproducible.
function mulberry32(seed) {
  let a = seed >>> 0;
  return () => {
    a |= 0; a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

// An SVG-render-like image: flat color blocks + a gradient + sparse noise. This
// is what vector art rasterizes to — highly compressible, so it exercises LZ77.
function makeImage(side, seed) {
  const w = side;
  const h = side;
  const rnd = mulberry32(seed);
  const palette = [
    [240, 240, 240], [30, 120, 200], [90, 90, 90], [250, 200, 40], [200, 40, 80],
  ];
  const block = 24 + Math.floor(rnd() * 24);
  const px = new Uint8Array(w * h * 4);
  for (let y = 0; y < h; y += 1) {
    for (let x = 0; x < w; x += 1) {
      const o = (y * w + x) * 4;
      const c = palette[(Math.floor(x / block) + Math.floor(y / block)) % palette.length];
      px[o] = c[0];
      px[o + 1] = c[1];
      px[o + 2] = (x * 255 / w) | 0; // horizontal gradient in blue
      px[o + 3] = 255;
      if ((x * 7 + y * 13) % 101 === 0) px[o] ^= 48; // sparse noise
    }
  }
  return px;
}

export default {
  name: 'svgpng',
  description: 'PNG encode (Up filter + DEFLATE) of synthetic SVG-render-like RGBA images.',
  sizes: [128, 256, 512, 768], // image side length in pixels
  iterations: 5, // full-image encodes per input, per size
  tolerance: 0, // byte-identical: WASM and JS must produce the same zlib stream

  setup(size) {
    const width = size;
    const height = size;
    const inputs = [makeImage(size, 0x1234 ^ size), makeImage(size, 0x9e37 ^ size)];
    return { width, height, inputs };
  },

  scoreJS(ctx, pixels) {
    return encodeZlibJs(pixels, ctx.width, ctx.height);
  },

  createWasm(instance, ctx) {
    return buildEncoder(instance, ctx.width, ctx.height);
  },

  scoreWasm(engine, pixels) {
    return engine.encode(pixels);
  },

  // Domain sample: report compression ratio for one image at the last size.
  report(ctx, engine) {
    const raw = ctx.width * ctx.height * 4;
    const png = engine.encode(ctx.inputs[0]).length;
    console.log(`\n  ${ctx.width}×${ctx.height}: raw ${(raw / 1024).toFixed(0)} KB → PNG ${(png / 1024).toFixed(1)} KB (${(raw / png).toFixed(1)}× compression)`);
  },
};
