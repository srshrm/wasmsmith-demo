// search-engine.js — the JS <-> WASM boundary for the "elastic" fuzzysearch block.
//
// This module owns EVERYTHING about talking to WebAssembly: building the search
// index (tokenize → vocabulary → BM25 postings), laying it out in linear memory,
// and running a typo-tolerant, BM25-ranked query. The block (fuzzysearch.js)
// never touches typed arrays or wasm — it calls createSearchEngine() once, then
// engine.search(text, limit) per keystroke.
//
// The heavy work lives in the kernel (assembly/fuzzysearch.ts): for each query
// term it fuzzy-expands over the whole vocabulary with edit distance, then scores
// documents with BM25. Here we only marshal: WebAssembly understands a flat block
// of bytes and numbers, not JS strings or Maps, so the index is flattened into
// typed arrays (vocab chars + offsets, CSR postings, doc lengths) and copied into
// wasm memory once; each query writes just its terms and reads back the winners.
//
// This block is WASM-only: if the module fails to load, createSearchEngine throws
// and the block reports "Search unavailable" (no JS scoring fallback). The JS
// reference implementation of the scoring math lives in the benchmark, not here.

export const MAX_TERM = 64; // longest token the edit-distance DP handles (chars)
export const MAX_QUERY_TERMS = 32; // most query terms marshaled per query
export const MAX_QUERY_CHARS = 512; // total query chars marshaled per query
export const MAX_K = 256; // largest top-k the fused kernel returns in one call

export const DEFAULT_K1 = 1.2; // BM25 term-frequency saturation
export const DEFAULT_B = 0.75; // BM25 length normalization
export const DEFAULT_MAX_EDITS = 2; // ceiling on AUTO fuzziness (0/1/2 edits)

/**
 * Tokenize text into lowercased alphanumeric terms, each capped at MAX_TERM
 * chars. The SAME tokenizer is used to index documents and to parse queries, so
 * query terms line up with the vocabulary. Shared with the benchmark.
 * @param {string} text
 * @returns {string[]}
 */
export function tokenize(text) {
  const matches = (text || '').toLowerCase().match(/[a-z0-9]+/g);
  if (!matches) return [];
  return matches.map((t) => (t.length > MAX_TERM ? t.slice(0, MAX_TERM) : t));
}

/**
 * Build the flat search index from one searchable string per document:
 *  - vocabChars/vocabOffsets: every unique term's chars, concatenated
 *  - postOffsets/postDocs/postTf: CSR inverted index (term -> {doc, tf} list);
 *    a term's document frequency is postOffsets[t+1] - postOffsets[t]
 *  - idf: precomputed inverse document frequency per term (a pure function of
 *    document frequency + corpus size, so it is index metadata, not query work)
 *  - docLen/avgdl: token length per document + the average (BM25 normalization)
 * Everything the kernel reads is a typed array so it copies straight into memory.
 * @param {string[]} texts one searchable string per document (index === result)
 * @returns {{
 *   vocabChars: Uint16Array, vocabOffsets: Int32Array,
 *   postOffsets: Int32Array, postDocs: Int32Array, postTf: Float32Array,
 *   idf: Float32Array, docLen: Float32Array,
 *   docCount: number, vocabCount: number, avgdl: number
 * }}
 */
export function buildIndex(texts) {
  const docCount = texts.length;
  const docLen = new Float32Array(docCount);
  const perDocCounts = new Array(docCount); // Map<term, tf> per document
  const vocabIndex = new Map(); // term -> id
  const docFreq = new Map(); // term -> number of documents containing it
  let totalLen = 0;

  for (let r = 0; r < docCount; r += 1) {
    const toks = tokenize(texts[r]);
    docLen[r] = toks.length;
    totalLen += toks.length;
    const counts = new Map();
    toks.forEach((tok) => counts.set(tok, (counts.get(tok) || 0) + 1));
    perDocCounts[r] = counts;
    counts.forEach((_, term) => {
      docFreq.set(term, (docFreq.get(term) || 0) + 1);
      if (!vocabIndex.has(term)) vocabIndex.set(term, vocabIndex.size);
    });
  }

  const vocabCount = vocabIndex.size;
  const terms = new Array(vocabCount);
  vocabIndex.forEach((id, term) => { terms[id] = term; });

  // Flatten vocabulary chars.
  let vocabCharLen = 0;
  for (let id = 0; id < vocabCount; id += 1) vocabCharLen += terms[id].length;
  const vocabChars = new Uint16Array(vocabCharLen);
  const vocabOffsets = new Int32Array(vocabCount + 1);
  let vp = 0;
  for (let id = 0; id < vocabCount; id += 1) {
    vocabOffsets[id] = vp;
    const term = terms[id];
    for (let c = 0; c < term.length; c += 1) {
      vocabChars[vp] = term.charCodeAt(c);
      vp += 1;
    }
  }
  vocabOffsets[vocabCount] = vp;

  // CSR postings: prefix-sum document frequencies into offsets, then scatter.
  const postOffsets = new Int32Array(vocabCount + 1);
  for (let id = 0; id < vocabCount; id += 1) {
    postOffsets[id + 1] = postOffsets[id] + docFreq.get(terms[id]);
  }
  const postingCount = postOffsets[vocabCount];
  const postDocs = new Int32Array(postingCount);
  const postTf = new Float32Array(postingCount);
  const cursor = postOffsets.slice(0, vocabCount); // write cursor per term
  for (let r = 0; r < docCount; r += 1) {
    perDocCounts[r].forEach((tf, term) => {
      const id = vocabIndex.get(term);
      const pos = cursor[id];
      cursor[id] = pos + 1;
      postDocs[pos] = r;
      postTf[pos] = tf;
    });
  }

  // Precompute idf per term (index metadata): idf = ln(1 + (N - df + 0.5)/(df + 0.5)).
  const idf = new Float32Array(vocabCount);
  for (let id = 0; id < vocabCount; id += 1) {
    const df = postOffsets[id + 1] - postOffsets[id];
    idf[id] = Math.log(1 + ((docCount - df) + 0.5) / (df + 0.5));
  }

  return {
    vocabChars,
    vocabOffsets,
    postOffsets,
    postDocs,
    postTf,
    idf,
    docLen,
    docCount,
    vocabCount,
    avgdl: docCount ? totalLen / docCount : 0,
  };
}

/**
 * Wire a compiled wasm instance to a built index and return a reusable scorer.
 * All allocations happen up front (alloc may grow memory and detach views), and
 * only afterwards do we snapshot the typed-array views — so the per-query path
 * never re-grows memory.
 * @param {WebAssembly.Instance} instance
 * @param {ReturnType<typeof buildIndex>} index
 * @param {{k1?: number, b?: number, maxEdits?: number}} [params]
 */
export function buildWasmEngine(instance, index, params = {}) {
  const {
    vocabChars, vocabOffsets, postOffsets, postDocs, postTf, idf, docLen,
    docCount, vocabCount, avgdl,
  } = index;
  const k1 = params.k1 ?? DEFAULT_K1;
  const b = params.b ?? DEFAULT_B;
  const maxEdits = params.maxEdits ?? DEFAULT_MAX_EDITS;
  const {
    reset, alloc, scoreAll, searchTopK, memory,
  } = instance.exports;

  reset();
  // Allocate every buffer BEFORE snapshotting views (growth detaches old views).
  const vocabCharsPtr = alloc(Math.max(vocabChars.length, 1) * 2);
  const vocabOffsetsPtr = alloc(vocabOffsets.length * 4);
  const postOffsetsPtr = alloc(postOffsets.length * 4);
  const postDocsPtr = alloc(Math.max(postDocs.length, 1) * 4);
  const postTfPtr = alloc(Math.max(postTf.length, 1) * 4);
  const idfPtr = alloc(Math.max(idf.length, 1) * 4);
  const docLenPtr = alloc(Math.max(docLen.length, 1) * 4);
  const idxPtr = alloc(56); // index header (13 x 4-byte slots)
  const qCharsPtr = alloc(MAX_QUERY_CHARS * 2);
  const qOffsetsPtr = alloc((MAX_QUERY_TERMS + 1) * 4);
  const dpPtr = alloc((MAX_TERM + 1) * (MAX_TERM + 1) * 4);
  const bestPtr = alloc(Math.max(docCount, 1) * 4);
  const scoresPtr = alloc(Math.max(docCount, 1) * 4);
  const topkPtr = alloc(MAX_K * 8); // { i32 doc; f32 score }[MAX_K]

  const u16 = new Uint16Array(memory.buffer);
  const i32 = new Int32Array(memory.buffer);
  const f32 = new Float32Array(memory.buffer);

  u16.set(vocabChars, vocabCharsPtr / 2);
  i32.set(vocabOffsets, vocabOffsetsPtr / 4);
  i32.set(postOffsets, postOffsetsPtr / 4);
  i32.set(postDocs, postDocsPtr / 4);
  f32.set(postTf, postTfPtr / 4);
  f32.set(idf, idfPtr / 4);
  f32.set(docLen, docLenPtr / 4);

  // Index header: interleaved i32/f32 slots the kernel reads with load<i32>/<f32>.
  const h = idxPtr / 4;
  i32[h + 0] = vocabCharsPtr;
  i32[h + 1] = vocabOffsetsPtr;
  i32[h + 2] = vocabCount;
  i32[h + 3] = postOffsetsPtr;
  i32[h + 4] = postDocsPtr;
  i32[h + 5] = postTfPtr;
  i32[h + 6] = docLenPtr;
  i32[h + 7] = docCount;
  f32[h + 8] = avgdl;
  f32[h + 9] = k1;
  f32[h + 10] = b;
  i32[h + 11] = maxEdits;
  i32[h + 12] = idfPtr;

  const qCharsView = u16.subarray(qCharsPtr / 2, qCharsPtr / 2 + MAX_QUERY_CHARS);
  const qOffsetsView = i32.subarray(qOffsetsPtr / 4, qOffsetsPtr / 4 + MAX_QUERY_TERMS + 1);
  const scoresView = f32.subarray(scoresPtr / 4, scoresPtr / 4 + docCount);
  // Interleaved [doc, scoreBits] pairs; we read only the doc indices (even slots).
  const topkI32 = new Int32Array(memory.buffer, topkPtr, MAX_K * 2);

  // Marshal query terms into the query buffers; returns the term count written.
  function writeQuery(queryTerms) {
    let p = 0;
    let count = 0;
    for (let i = 0; i < queryTerms.length && count < MAX_QUERY_TERMS; i += 1) {
      const term = queryTerms[i];
      if (term.length > 0 && p + term.length <= MAX_QUERY_CHARS) {
        qOffsetsView[count] = p;
        for (let c = 0; c < term.length; c += 1) {
          qCharsView[p] = term.charCodeAt(c);
          p += 1;
        }
        count += 1;
      }
    }
    qOffsetsView[count] = p;
    return count;
  }

  return {
    // Full per-document scores (used by the benchmark to check WASM == JS).
    scoreAll(queryTerms) {
      const count = writeQuery(queryTerms);
      scoreAll(idxPtr, qCharsPtr, qOffsetsPtr, count, dpPtr, bestPtr, scoresPtr);
      return scoresView;
    },
    // Fused score + rank: one WASM call returns just the top-k document indices.
    searchTopK(queryTerms, k) {
      const kk = Math.min(k, MAX_K);
      const count = writeQuery(queryTerms);
      const n = searchTopK(
        idxPtr,
        qCharsPtr,
        qOffsetsPtr,
        count,
        dpPtr,
        bestPtr,
        scoresPtr,
        kk,
        topkPtr,
      );
      const out = new Array(n);
      for (let i = 0; i < n; i += 1) out[i] = topkI32[i * 2];
      return out;
    },
  };
}

/**
 * Load + instantiate the kernel. Uses instantiateStreaming with a fallback to
 * fetch+arrayBuffer for servers that don't send Content-Type: application/wasm.
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
 * High-level search engine the block consumes. Loads the kernel, builds the
 * index in wasm memory, and returns a search() that fuzzy-matches + BM25-ranks
 * entirely in WASM and returns the top-K document indices. Throws if the kernel
 * can't be loaded.
 * @param {string} wasmUrl URL of the compiled kernel
 * @param {string[]} texts one searchable string per document (index === result)
 * @param {{k1?: number, b?: number, maxEdits?: number}} [params]
 */
export async function createSearchEngine(wasmUrl, texts, params = {}) {
  const instance = await loadWasm(wasmUrl);
  const index = buildIndex(texts);
  const engine = buildWasmEngine(instance, index, params);
  return {
    vocabCount: index.vocabCount,
    docCount: index.docCount,
    /**
     * Tokenize the query, then fuzzy-match + BM25-rank entirely in WASM.
     * @param {string} text raw query string
     * @param {number} limit max results
     * @returns {number[]} document indices, best match first
     */
    search(text, limit) {
      const terms = tokenize(text);
      if (!terms.length) return [];
      return engine.searchTopK(terms, limit);
    },
  };
}
