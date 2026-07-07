/* eslint-disable no-bitwise */
// svgpng-engine.js — the JS <-> WASM boundary for the svgpng block.
//
// `no-bitwise` is disabled here because PNG container assembly (CRC-32, big-endian
// lengths) is inherently bit-level — the shifts/masks are the format, not a
// pointer-math shortcut.
//
// This module owns everything about talking to the WASM PNG encoder: loading the
// module, laying out the RGBA image + scratch buffers in linear memory, calling
// the kernel to get a zlib stream, and wrapping that stream in PNG chunks. The
// block (svgpng.js) never touches typed arrays or wasm — it rasterizes an SVG to
// pixels and calls engine.toPng(pixels, w, h).
//
// The rasterization (SVG -> pixels) is done by the browser canvas (native); only
// the encode (pixels -> compressed .png bytes) runs here / in WASM.

const HASH_SLOTS = 32768; // hash head/prev tables the kernel needs (matches the kernel)

/**
 * Wire a compiled kernel instance to buffers sized for one width×height image.
 * All allocations happen up front (alloc may grow memory and detach views), then
 * we snapshot the memory view — so repeated encode() calls never re-grow memory.
 * @param {WebAssembly.Instance} instance
 * @param {number} width
 * @param {number} height
 * @returns {{ encode: (pixels: Uint8Array) => Uint8Array }} encode -> zlib stream
 */
export function buildEncoder(instance, width, height) {
  const {
    reset, alloc, encode, memory,
  } = instance.exports;
  const stride = width * 4;
  const rawLen = height * stride;
  const flen = height * (1 + stride);

  reset();
  const pixelsPtr = alloc(rawLen);
  const filteredPtr = alloc(flen);
  const headPtr = alloc(HASH_SLOTS * 4);
  const prevPtr = alloc(HASH_SLOTS * 4);
  const outPtr = alloc(flen * 2 + 1024);
  const mem = new Uint8Array(memory.buffer); // stable: no more growth after this

  return {
    encode(pixels) {
      mem.set(pixels, pixelsPtr);
      const len = encode(pixelsPtr, width, height, filteredPtr, headPtr, prevPtr, outPtr);
      return mem.slice(outPtr, outPtr + len);
    },
  };
}

// ── PNG container (cheap, shared by the WASM and JS paths) ────────────────────

let CRC_TABLE = null;
function crcTable() {
  if (CRC_TABLE) return CRC_TABLE;
  const t = new Uint32Array(256);
  for (let n = 0; n < 256; n += 1) {
    let c = n;
    for (let k = 0; k < 8; k += 1) c = (c & 1) ? (0xedb88320 ^ (c >>> 1)) : (c >>> 1);
    t[n] = c >>> 0;
  }
  CRC_TABLE = t;
  return t;
}

function crc32(buf, start, end) {
  const t = crcTable();
  let c = 0xffffffff;
  for (let i = start; i < end; i += 1) c = t[(c ^ buf[i]) & 0xff] ^ (c >>> 8);
  return (c ^ 0xffffffff) >>> 0;
}

function writeU32BE(arr, off, v) {
  arr[off] = (v >>> 24) & 0xff;
  arr[off + 1] = (v >>> 16) & 0xff;
  arr[off + 2] = (v >>> 8) & 0xff;
  arr[off + 3] = v & 0xff;
}

function chunk(type, data) {
  const out = new Uint8Array(12 + data.length);
  writeU32BE(out, 0, data.length);
  for (let i = 0; i < 4; i += 1) out[4 + i] = type.charCodeAt(i);
  out.set(data, 8);
  writeU32BE(out, 8 + data.length, crc32(out, 4, 8 + data.length));
  return out;
}

/**
 * Wrap a zlib stream (IDAT payload) in a full 8-bit RGBA PNG file.
 * @param {Uint8Array} zlibStream
 * @param {number} width
 * @param {number} height
 * @returns {Uint8Array} complete .png bytes
 */
export function wrapPng(zlibStream, width, height) {
  const sig = Uint8Array.of(137, 80, 78, 71, 13, 10, 26, 10);
  const ihdr = new Uint8Array(13);
  writeU32BE(ihdr, 0, width);
  writeU32BE(ihdr, 4, height);
  ihdr[8] = 8; // bit depth
  ihdr[9] = 6; // color type: RGBA
  // ihdr[10..12] = 0: compression, filter, interlace
  const parts = [sig, chunk('IHDR', ihdr), chunk('IDAT', zlibStream), chunk('IEND', new Uint8Array(0))];
  let total = 0;
  parts.forEach((p) => { total += p.length; });
  const png = new Uint8Array(total);
  let o = 0;
  parts.forEach((p) => { png.set(p, o); o += p.length; });
  return png;
}

/**
 * Load + instantiate the kernel. Uses instantiateStreaming with a fetch+arrayBuffer
 * fallback for servers that don't send Content-Type: application/wasm. Throws if
 * the module can't be loaded — the block has no JS encoding fallback.
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
 * High-level engine the block consumes. Loads the kernel once; each toPng() call
 * lays out the image, encodes it in WASM, and wraps the result as a PNG file.
 * @param {string} wasmUrl
 */
export async function createSvgPngEngine(wasmUrl) {
  const instance = await loadWasm(wasmUrl);
  return {
    /**
     * @param {Uint8Array} pixels row-major RGBA, length width*height*4
     * @param {number} width
     * @param {number} height
     * @returns {Uint8Array} complete .png bytes
     */
    toPng(pixels, width, height) {
      const zlibStream = buildEncoder(instance, width, height).encode(pixels);
      return wrapPng(zlibStream, width, height);
    },
  };
}
