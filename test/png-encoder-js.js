/* eslint-disable no-bitwise */
// png-encoder-js.js — the JS twin of the svgpng WASM kernel.
//
// A DEFLATE/PNG encoder is bit-level by nature (Huffman codes, bit packing,
// checksums), so `no-bitwise` is disabled for this file — the shifts and masks
// ARE the algorithm, not a pointer-math shortcut.
//
// This is a FAITHFUL twin of assembly/svgpng.ts: same "Up" filter, same Adler-32,
// same fixed-Huffman + greedy LZ77 (identical hash, chain depth, tie-breaks), so
// it produces byte-for-byte the same zlib stream. Shared by the benchmark fixture
// (which asserts WASM == JS) and the comparison page (which times JS vs WASM).

const MIN_MATCH = 3;
const MAX_MATCH = 258;
const WSIZE = 32768;
const WMASK = 32767;
const HMASK = 32767;
const MAX_CHAIN = 128;

// Fixed DEFLATE length codes (symbol 257 + index) and distance codes (symbol =
// index), with their base value and extra-bit count (RFC 1951 §3.2.5).
const LEN_BASE = [3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 15, 17, 19, 23, 27, 31, 35, 43,
  51, 59, 67, 83, 99, 115, 131, 163, 195, 227, 258];
const LEN_EXTRA = [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4,
  4, 4, 5, 5, 5, 5, 0];
const DIST_BASE = [1, 2, 3, 4, 5, 7, 9, 13, 17, 25, 33, 49, 65, 97, 129, 193, 257,
  385, 513, 769, 1025, 1537, 2049, 3073, 4097, 6145, 8193, 12289, 16385, 24577];
const DIST_EXTRA = [0, 0, 0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9,
  10, 10, 11, 11, 12, 12, 13, 13];

// PNG "Up" filter: each byte minus the byte directly above, per scanline.
function filterUp(pixels, width, height) {
  const stride = width * 4;
  const filtered = new Uint8Array(height * (1 + stride));
  let fp = 0;
  for (let r = 0; r < height; r += 1) {
    filtered[fp] = 2; // filter type: Up
    fp += 1;
    const row = r * stride;
    for (let i = 0; i < stride; i += 1) {
      const above = r > 0 ? pixels[row - stride + i] : 0;
      filtered[fp] = (pixels[row + i] - above) & 0xff;
      fp += 1;
    }
  }
  return filtered;
}

// Adler-32 over the filtered bytes (zlib trailer), batched to defer the modulo.
function adler32(buf) {
  let a = 1;
  let b = 0;
  let i = 0;
  const total = buf.length;
  while (i < total) {
    let n = total - i;
    if (n > 5552) n = 5552;
    for (let k = 0; k < n; k += 1) {
      a += buf[i + k];
      b += a;
    }
    a %= 65521;
    b %= 65521;
    i += n;
  }
  return ((b << 16) | a) >>> 0;
}

/**
 * Encode an RGBA image to a zlib stream (the PNG IDAT payload) — byte-identical
 * to the WASM kernel's `encode`.
 * @param {Uint8Array} pixels row-major RGBA, length width*height*4
 * @param {number} width
 * @param {number} height
 * @returns {Uint8Array} zlib stream (2-byte header + DEFLATE + 4-byte Adler-32)
 */
export default function encodeZlib(pixels, width, height) {
  const filtered = filterUp(pixels, width, height);
  const flen = filtered.length;
  const adler = adler32(filtered);

  const out = new Uint8Array(flen * 2 + 1024);
  out[0] = 0x78; // CMF
  out[1] = 0x01; // FLG
  let outPos = 0; // bytes written after the 2-byte header
  let bitBuf = 0;
  let bitCnt = 0;

  function sendBits(value, count) {
    bitBuf |= value << bitCnt;
    bitCnt += count;
    while (bitCnt >= 8) {
      out[2 + outPos] = bitBuf & 0xff;
      outPos += 1;
      bitBuf >>>= 8;
      bitCnt -= 8;
    }
  }
  function sendCode(code, len) {
    let r = 0;
    let c = code;
    for (let i = 0; i < len; i += 1) {
      r = (r << 1) | (c & 1);
      c >>= 1;
    }
    sendBits(r, len);
  }
  function sendLitLen(sym) {
    if (sym <= 143) sendCode(0x30 + sym, 8);
    else if (sym <= 255) sendCode(0x190 + (sym - 144), 9);
    else if (sym <= 279) sendCode(sym - 256, 7);
    else sendCode(0xc0 + (sym - 280), 8);
  }
  function sendMatch(len, dist) {
    let li = LEN_BASE.length - 1;
    while (LEN_BASE[li] > len) li -= 1;
    sendLitLen(257 + li);
    if (LEN_EXTRA[li] > 0) sendBits(len - LEN_BASE[li], LEN_EXTRA[li]);

    let di = DIST_BASE.length - 1;
    while (DIST_BASE[di] > dist) di -= 1;
    sendCode(di, 5);
    if (DIST_EXTRA[di] > 0) sendBits(dist - DIST_BASE[di], DIST_EXTRA[di]);
  }
  const hash3 = (p) => ((filtered[p] << 10) ^ (filtered[p + 1] << 5) ^ filtered[p + 2]) & HMASK;

  const head = new Int32Array(32768).fill(-1);
  const prev = new Int32Array(32768);

  sendBits(1, 1); // BFINAL = 1
  sendBits(1, 2); // BTYPE  = 01 (fixed Huffman)

  let pos = 0;
  while (pos < flen) {
    let bestLen = MIN_MATCH - 1;
    let bestDist = 0;

    if (pos + MIN_MATCH <= flen) {
      let maxLen = flen - pos;
      if (maxLen > MAX_MATCH) maxLen = MAX_MATCH;
      const h = hash3(pos);
      let cand = head[h];
      let chain = MAX_CHAIN;
      while (cand >= 0 && (pos - cand) <= WSIZE && chain > 0) {
        if (filtered[cand + bestLen] === filtered[pos + bestLen]) {
          let len = 0;
          while (len < maxLen && filtered[cand + len] === filtered[pos + len]) len += 1;
          if (len > bestLen) {
            bestLen = len;
            bestDist = pos - cand;
            if (len >= maxLen) break;
          }
        }
        cand = prev[cand & WMASK];
        chain -= 1;
      }
    }

    if (bestLen >= MIN_MATCH) {
      sendMatch(bestLen, bestDist);
      const end = pos + bestLen;
      while (pos < end) {
        if (pos + MIN_MATCH <= flen) {
          const hh = hash3(pos);
          prev[pos & WMASK] = head[hh];
          head[hh] = pos;
        }
        pos += 1;
      }
    } else {
      sendLitLen(filtered[pos]);
      if (pos + MIN_MATCH <= flen) {
        const hh = hash3(pos);
        prev[pos & WMASK] = head[hh];
        head[hh] = pos;
      }
      pos += 1;
    }
  }

  sendLitLen(256); // end-of-block
  if (bitCnt > 0) {
    out[2 + outPos] = bitBuf & 0xff;
    outPos += 1;
  }

  const tail = 2 + outPos;
  out[tail] = (adler >>> 24) & 0xff;
  out[tail + 1] = (adler >>> 16) & 0xff;
  out[tail + 2] = (adler >>> 8) & 0xff;
  out[tail + 3] = adler & 0xff;
  return out.subarray(0, 2 + outPos + 4);
}
