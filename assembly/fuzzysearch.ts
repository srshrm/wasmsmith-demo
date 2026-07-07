// fuzzysearch.ts — WASM kernel for on-device "elastic" search: typo-tolerant
// fuzzy matching (edit distance) + BM25 relevance ranking, fully client-side.
//
// This is the compute-heavy core of an Elasticsearch-style query, done in the
// browser with NO backend. A query is tokenized (host side) into terms; for each
// term the kernel:
//   1. FUZZY-EXPANDS it over the whole vocabulary using restricted
//      Damerau–Levenshtein (OSA) edit distance, keeping vocabulary terms within
//      an AUTO fuzziness (0/1/2 edits by term length, capped by maxEdits). This
//      is the hot loop — an O(V · L²) integer DP sweep over V vocabulary terms.
//   2. Scores every document that contains a matched term with BM25
//      (term-frequency · inverse-document-frequency, length-normalized), taking
//      per query term the BEST matched-term contribution per document (so
//      "reference"/"references" don't double count), weighted by a fuzzy boost
//      that decays with edit distance.
// Document scores are summed across query terms, then the top-k is selected in
// the kernel so only the winners cross back to JS.
//
// IDF is a pure function of a term's document frequency and the corpus size —
// both known when the index is built — so the host precomputes idf[t] and the
// kernel just reads it. That keeps the kernel free of transcendental math
// (Math.log miscompiles at -O3 in this module) and makes the JS twin and the
// kernel share one idf table, agreeing to the bit.
//
// Compiled with `--runtime stub`: no managed heap/GC. We own linear memory
// through a bump allocator and the host (search-engine.js) marshals every array
// in and reads results back. The JS reference twin lives in fuzzysearch.bench.mjs
// and must mirror the math here so the benchmark can assert agreement.
//
// ── Interop (see blocks/fuzzysearch/fuzzysearch.manifest.json) ────────────────
// All index arrays live in linear memory; a 56-byte "index header" at idxPtr
// holds their pointers + BM25/fuzziness params so call signatures stay small:
//   idx[0]  i32 vocabCharsPtr   u16[]  all vocab terms' chars concatenated
//   idx[1]  i32 vocabOffsetsPtr i32[V+1] start (u16 units) of each vocab term
//   idx[2]  i32 vocabCount      V
//   idx[3]  i32 postOffsetsPtr  i32[V+1] CSR start (posting units) per term
//   idx[4]  i32 postDocsPtr     i32[P]  docId of each posting
//   idx[5]  i32 postTfPtr       f32[P]  term frequency of each posting
//   idx[6]  i32 docLenPtr       f32[N]  token length of each document
//   idx[7]  i32 docCount        N
//   idx[8]  f32 avgdl           average document length
//   idx[9]  f32 k1              BM25 term-frequency saturation (≈1.2)
//   idx[10] f32 b               BM25 length normalization (≈0.75)
//   idx[11] i32 maxEditsCap     ceiling on AUTO fuzziness (0/1/2)
//   idx[12] i32 idfPtr          f32[V]  precomputed inverse document frequency

// ── Explicit linear-memory bump allocator (mirrors the golden example) ────────
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

// Restricted Damerau–Levenshtein (Optimal String Alignment) with a threshold.
// Returns the edit distance, or maxEdits+1 as soon as it can prove the distance
// exceeds maxEdits (length filter + non-decreasing row-minimum early exit — both
// provably safe for OSA). `dp` points at a reusable (MAX_TERM+1)² i32 matrix.
@inline
function osa(
  aPtr: i32,
  aLen: i32,
  bPtr: i32,
  bLen: i32,
  maxEdits: i32,
  dp: i32,
): i32 {
  const diff = aLen > bLen ? aLen - bLen : bLen - aLen;
  if (diff > maxEdits) return maxEdits + 1; // |Δlen| is a lower bound on distance
  if (aLen == 0) return bLen;
  if (bLen == 0) return aLen;

  const W = bLen + 1;
  for (let j = 0; j <= bLen; j++) store<i32>(dp + (j << 2), j);

  for (let i = 1; i <= aLen; i++) {
    const rowBase = i * W;
    store<i32>(dp + (rowBase << 2), i);
    const ai = load<u16>(aPtr + ((i - 1) << 1));
    let rowMin = i;
    for (let j = 1; j <= bLen; j++) {
      const bj = load<u16>(bPtr + ((j - 1) << 1));
      const cost = ai == bj ? 0 : 1;
      let v = load<i32>(dp + (((i - 1) * W + j) << 2)) + 1; // deletion
      const ins = load<i32>(dp + ((rowBase + j - 1) << 2)) + 1; // insertion
      if (ins < v) v = ins;
      const sub = load<i32>(dp + (((i - 1) * W + j - 1) << 2)) + cost; // substitution
      if (sub < v) v = sub;
      if (i > 1 && j > 1) {
        const aPrev = load<u16>(aPtr + ((i - 2) << 1));
        const bPrev = load<u16>(bPtr + ((j - 2) << 1));
        if (ai == bPrev && aPrev == bj) {
          const tr = load<i32>(dp + (((i - 2) * W + j - 2) << 2)) + 1; // transposition
          if (tr < v) v = tr;
        }
      }
      store<i32>(dp + ((rowBase + j) << 2), v);
      if (v < rowMin) rowMin = v;
    }
    if (rowMin > maxEdits) return maxEdits + 1; // whole remaining matrix ≥ rowMin
  }
  return load<i32>(dp + ((aLen * W + bLen) << 2));
}

// Fill scores[docCount] with the fuzzy-expanded BM25 score of every document for
// the given query terms. `best` is per-document scratch (f32[docCount]) reused
// across query terms; `dp` is the edit-distance matrix scratch.
function computeScores(
  idxPtr: i32,
  qCharsPtr: i32,
  qOffsetsPtr: i32,
  qTermCount: i32,
  dp: i32,
  best: i32,
  scores: i32,
): void {
  const vocabChars = load<i32>(idxPtr + 0);
  const vocabOffsets = load<i32>(idxPtr + 4);
  const vocabCount = load<i32>(idxPtr + 8);
  const postOffsets = load<i32>(idxPtr + 12);
  const postDocs = load<i32>(idxPtr + 16);
  const postTf = load<i32>(idxPtr + 20);
  const docLen = load<i32>(idxPtr + 24);
  const docCount = load<i32>(idxPtr + 28);
  const avgdl = <f64>load<f32>(idxPtr + 32);
  const k1 = <f64>load<f32>(idxPtr + 36);
  const b = <f64>load<f32>(idxPtr + 40);
  const maxEditsCap = load<i32>(idxPtr + 44);
  const idfArr = load<i32>(idxPtr + 48);
  const invAvg = avgdl > 0.0 ? 1.0 / avgdl : 0.0;

  for (let d = 0; d < docCount; d++) store<f32>(scores + (d << 2), 0.0);

  for (let qi = 0; qi < qTermCount; qi++) {
    const qStart = load<i32>(qOffsetsPtr + (qi << 2));
    const qEnd = load<i32>(qOffsetsPtr + ((qi + 1) << 2));
    const qLen = qEnd - qStart;
    if (qLen == 0) continue;
    const qPtr = qCharsPtr + (qStart << 1);

    // Elasticsearch-style AUTO fuzziness, capped by the configured maximum.
    let me = qLen <= 2 ? 0 : (qLen <= 5 ? 1 : 2);
    if (me > maxEditsCap) me = maxEditsCap;
    const meDenom = <f64>(me + 1);

    for (let d = 0; d < docCount; d++) store<f32>(best + (d << 2), 0.0);
    let matched = false;

    for (let t = 0; t < vocabCount; t++) {
      const tStart = load<i32>(vocabOffsets + (t << 2));
      const tEnd = load<i32>(vocabOffsets + ((t + 1) << 2));
      const dist = osa(qPtr, qLen, vocabChars + (tStart << 1), tEnd - tStart, me, dp);
      if (dist > me) continue;
      matched = true;

      const idf = <f64>load<f32>(idfArr + (t << 2)); // precomputed by the host
      const boost = 1.0 - <f64>dist / meDenom;
      const weight = boost * idf;
      const pStart = load<i32>(postOffsets + (t << 2));
      const pEnd = load<i32>(postOffsets + ((t + 1) << 2));

      for (let p = pStart; p < pEnd; p++) {
        const doc = load<i32>(postDocs + (p << 2));
        const tf = <f64>load<f32>(postTf + (p << 2));
        const dl = <f64>load<f32>(docLen + (doc << 2));
        const denom = tf + k1 * (1.0 - b + b * dl * invAvg);
        const s = weight * (tf * (k1 + 1.0)) / denom;
        const cur = <f64>load<f32>(best + (doc << 2));
        if (s > cur) store<f32>(best + (doc << 2), <f32>s); // best matched term per doc
      }
    }

    if (matched) {
      for (let d = 0; d < docCount; d++) {
        const bv = load<f32>(best + (d << 2));
        if (bv > 0.0) {
          const acc = load<f32>(scores + (d << 2));
          store<f32>(scores + (d << 2), acc + bv);
        }
      }
    }
  }
}

// Full per-document score vector. Used by the benchmark to check WASM == JS.
export function scoreAll(
  idxPtr: i32,
  qCharsPtr: i32,
  qOffsetsPtr: i32,
  qTermCount: i32,
  dpPtr: i32,
  bestPtr: i32,
  scoresPtr: i32,
): void {
  computeScores(idxPtr, qCharsPtr, qOffsetsPtr, qTermCount, dpPtr, bestPtr, scoresPtr);
}

// Bounded top-k selection over a scores buffer, best (highest) first, ties by
// LOWER document index (rows visited ascending; equal scores never displace).
//   outPtr : { i32 doc; f32 score }[k]   OUTPUT, 8 B/slot
@inline
function selectTopK(scores: i32, docCount: i32, k: i32, outPtr: i32): i32 {
  if (k <= 0) return 0;
  let count = 0;
  for (let r = 0; r < docCount; r++) {
    const s = load<f32>(scores + (r << 2));
    if (s <= 0.0) continue;
    if (count == k) {
      const worst = load<f32>(outPtr + ((k - 1) << 3) + 4);
      if (s <= worst) continue;
    }
    let pos = count < k ? count : k - 1;
    if (count < k) count++;
    while (pos > 0) {
      const aboveScore = load<f32>(outPtr + ((pos - 1) << 3) + 4);
      if (aboveScore < s) {
        const aboveIdx = load<i32>(outPtr + ((pos - 1) << 3));
        store<i32>(outPtr + (pos << 3), aboveIdx);
        store<f32>(outPtr + (pos << 3) + 4, aboveScore);
        pos--;
      } else {
        break;
      }
    }
    store<i32>(outPtr + (pos << 3), r);
    store<f32>(outPtr + (pos << 3) + 4, s);
  }
  return count;
}

// Fused score + rank: compute all document scores, then return only the top-k
// (doc, score) pairs. This is the block's per-query path — no full scores array
// crosses back to JS and no JS-side sort runs.
export function searchTopK(
  idxPtr: i32,
  qCharsPtr: i32,
  qOffsetsPtr: i32,
  qTermCount: i32,
  dpPtr: i32,
  bestPtr: i32,
  scoresPtr: i32,
  k: i32,
  outPtr: i32,
): i32 {
  computeScores(idxPtr, qCharsPtr, qOffsetsPtr, qTermCount, dpPtr, bestPtr, scoresPtr);
  return selectTopK(scoresPtr, load<i32>(idxPtr + 28), k, outPtr);
}
