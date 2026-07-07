// simhash.ts — WASM kernel for content fingerprinting (near-duplicate / SEO
// canonical detection) via SimHash over document text.
//
// WHY WASM WINS HERE: the hot loop computes a 64-bit FNV-1a hash of every token
// in every document. JavaScript numbers are IEEE-754 f64 and cannot hold exact
// 64-bit integers, so a faithful JS baseline MUST use BigInt — heap-allocated
// big integers that are ~15-40× slower than native machine words. WASM has a
// native i64/u64 type, so the same math runs in registers. No SIMD needed.
//
// Compiled with `--runtime stub`, so there is NO managed heap / GC: we own
// linear memory explicitly through a tiny bump allocator and the host (JS)
// marshals byte arrays in and reads a 64-bit fingerprint out.
//
// ── Interop contract (see blocks/simhash/simhash.manifest.json) ──────────────
//   reset()                              -> rewind the bump allocator
//   alloc(byteLen) -> ptr                -> reserve `byteLen` bytes, byte ptr
//   simhash(docPtr, docLen, accPtr, outPtr) -> fill the 64-bit fingerprint
//
// Memory layout the host arranges before calling `simhash` (little-endian):
//   docPtr : u8[docLen]   document bytes, already lowercased ASCII by the host
//   accPtr : i32[64]      256-byte scratch for the per-bit counters (zeroed here)
//   outPtr : u32[2]       OUTPUT — out[0]=low 32 bits, out[1]=high 32 bits

// ── Explicit linear-memory bump allocator (verbatim from fuzzysearch.ts) ─────
// The first 1 KiB is reserved scratch; allocations start above it and never
// free individually (host calls reset() to rewind for a fresh layout).
let bumpPtr: i32 = 1024;

export function reset(): void {
  bumpPtr = 1024;
}

// Reserve `byteLen` bytes, 8-byte aligned, growing linear memory as needed.
export function alloc(byteLen: i32): i32 {
  const ptr = (bumpPtr + 7) & ~7;
  bumpPtr = ptr + byteLen;
  const needPages = ((bumpPtr + 0xffff) & ~0xffff) >>> 16;
  const havePages = memory.size();
  if (needPages > havePages) memory.grow(needPages - havePages);
  return ptr;
}

// FNV-1a 64-bit constants.
const FNV_OFFSET: u64 = 0xcbf29ce484222325;
const FNV_PRIME: u64 = 0x100000001b3;

// A token byte is [a-z0-9]; every other byte is a separator.
// @inline
function isTokenByte(c: i32): bool {
  return (c >= 97 && c <= 122) || (c >= 48 && c <= 57);
}

// Compute the 64-bit SimHash fingerprint of one document.
//
// Tokenize into maximal [a-z0-9] runs; FNV-1a-hash each token as u64; for each
// bit position b in 0..63, increment counter[b] if bit b of the hash is 1 else
// decrement it; the fingerprint bit b is 1 iff counter[b] > 0. Pure native
// 64-bit integer math — exactly the loop JS can only do in BigInt. Mirrors the
// BigInt baseline in blocks/simhash/simhash.bench.mjs bit-for-bit so the
// benchmark can assert equality (tolerance 0).
export function simhash(docPtr: i32, docLen: i32, accPtr: i32, outPtr: i32): void {
  // Zero the 64 signed counters.
  for (let b = 0; b < 64; b++) {
    store<i32>(accPtr + (b << 2), 0);
  }

  let i = 0;
  while (i < docLen) {
    const c = <i32>load<u8>(docPtr + i);
    if (isTokenByte(c)) {
      // Hash this maximal token run with FNV-1a (64-bit).
      let h: u64 = FNV_OFFSET;
      let j = i;
      while (j < docLen) {
        const tc = <i32>load<u8>(docPtr + j);
        if (!isTokenByte(tc)) break;
        h = h ^ (<u64>tc);
        h = h * FNV_PRIME; // wrapping mul mod 2^64
        j++;
      }
      // Fold this token hash into the per-bit counters.
      for (let b = 0; b < 64; b++) {
        const bit = (h >> (<u64>b)) & 1;
        const v = load<i32>(accPtr + (b << 2));
        store<i32>(accPtr + (b << 2), bit == 1 ? v + 1 : v - 1);
      }
      i = j; // resume after the token
    } else {
      i++;
    }
  }

  // Collapse counters into the 64-bit fingerprint: bit b set iff counter[b] > 0.
  let fp: u64 = 0;
  for (let b = 0; b < 64; b++) {
    if (load<i32>(accPtr + (b << 2)) > 0) {
      fp = fp | ((<u64>1) << (<u64>b));
    }
  }

  store<u32>(outPtr, <u32>(fp & 0xffffffff));
  store<u32>(outPtr + 4, <u32>(fp >> 32));
}
