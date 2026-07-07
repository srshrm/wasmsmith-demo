// semanticsearch.ts — SIMD cosine-similarity search over content embeddings.
//
// The hot loop is a dot product of one query embedding against thousands of
// document embeddings. With unit-normalized vectors, cosine similarity IS the
// dot product, so ranking = one big dot-product sweep. This is a textbook SIMD
// win: JS has no portable vector type and runs the multiply-add scalar, while
// this kernel does 4 lanes per instruction with f32x4.
//
// Compiled with `--runtime stub` (no GC) + `--enable simd`. The host marshals a
// flat f32 corpus and query into linear memory; we read via explicit pointers.
//
// ── Interop contract (see semanticsearch.manifest.json) ──────────────────────
//   reset()                              rewind the bump allocator
//   alloc(byteLen) -> ptr                reserve bytes (8-byte aligned), grows mem
//   cosine(q, corpus, count, dim, out)   OUT f32[count] = similarity per doc
//   cosineTopK(q, corpus, count, dim, k, out) -> n
//                                        OUT {i32 index; f32 score}[k], best first
// Vectors are assumed L2-normalized by the host, so dot == cosine.

// ── Explicit linear-memory bump allocator (identical to the golden example) ──
let bumpPtr: i32 = 1024;

export function reset(): void {
  bumpPtr = 1024;
}

export function alloc(byteLen: i32): i32 {
  const ptr = (bumpPtr + 7) & ~7;
  bumpPtr = ptr + byteLen;
  const needPages = ((bumpPtr + 0xffff) & ~0xffff) >>> 16;
  const havePages = memory.size();
  if (needPages > havePages) memory.grow(needPages - havePages);
  return ptr;
}

// 4-wide SIMD dot product of two f32 vectors. Sums four partial lanes and folds
// them at the end; the scalar tail handles dims not divisible by 4.
@inline
function dot(aPtr: i32, bPtr: i32, dim: i32): f32 {
  let acc = f32x4.splat(0.0);
  let i = 0;
  const limit = dim & ~3;
  for (; i < limit; i += 4) {
    const va = v128.load(aPtr + (i << 2));
    const vb = v128.load(bPtr + (i << 2));
    acc = f32x4.add(acc, f32x4.mul(va, vb));
  }
  let total: f32 = f32x4.extract_lane(acc, 0) + f32x4.extract_lane(acc, 1)
    + f32x4.extract_lane(acc, 2) + f32x4.extract_lane(acc, 3);
  for (; i < dim; i += 1) {
    total += load<f32>(aPtr + (i << 2)) * load<f32>(bPtr + (i << 2));
  }
  return total;
}

// Score every document: cosine (== dot for unit vectors) into out[count].
export function cosine(
  queryPtr: i32,
  corpusPtr: i32,
  count: i32,
  dim: i32,
  outPtr: i32,
): void {
  const stride = dim << 2;
  for (let r = 0; r < count; r += 1) {
    const s = dot(queryPtr, corpusPtr + r * stride, dim);
    store<f32>(outPtr + (r << 2), s);
  }
}

// Fused score + bounded top-k in a single pass — only the k winners cross back
// to JS (no full scores array, no JS-side sort). Ordering: score descending,
// ties broken by lower index (docs visited in ascending order, equal scores
// never displace an incumbent). Mirrors scoreTopK in the golden fuzzysearch.
//   outPtr : { i32 index; f32 score }[k]   OUTPUT, best match first (8 B/slot)
//   returns: number of results written (<= k)
export function cosineTopK(
  queryPtr: i32,
  corpusPtr: i32,
  count: i32,
  dim: i32,
  k: i32,
  outPtr: i32,
): i32 {
  if (k <= 0) return 0;
  const stride = dim << 2;
  let n = 0;
  for (let r = 0; r < count; r += 1) {
    const s = dot(queryPtr, corpusPtr + r * stride, dim);

    // Once full, a new doc must strictly beat the current worst to enter.
    if (n == k) {
      const worst = load<f32>(outPtr + ((k - 1) << 3) + 4);
      if (s <= worst) continue;
    }

    let pos = n < k ? n : k - 1;
    if (n < k) n += 1;
    while (pos > 0) {
      const aboveScore = load<f32>(outPtr + ((pos - 1) << 3) + 4);
      if (aboveScore < s) {
        const aboveIdx = load<i32>(outPtr + ((pos - 1) << 3));
        store<i32>(outPtr + (pos << 3), aboveIdx);
        store<f32>(outPtr + (pos << 3) + 4, aboveScore);
        pos -= 1;
      } else {
        break;
      }
    }
    store<i32>(outPtr + (pos << 3), r);
    store<f32>(outPtr + (pos << 3) + 4, s);
  }
  return n;
}
