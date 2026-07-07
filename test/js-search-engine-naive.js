// js-search-engine-naive.js — DEMO-ONLY "idiomatic JavaScript" fuzzy search.
//
// This is the counterpart to js-search-engine.js. Both run the SAME scoring
// algorithm in plain JS, but this one is written the way a typical developer
// would reach for first: score over the row STRINGS with charCodeAt, build an
// array of {index, score} objects, then filter + full-sort + slice — no
// pre-flattened Uint16Array corpus, no offset table, no bounded selection.
//
// The point of the comparison isn't "JS is bad": it's that the WASM engine's
// speed comes from BOTH compiled code AND a tight data layout (a flat typed
// buffer scored in one pass). js-search-engine.js keeps the layout and shows
// the modest ~1.5x compiled-code win; this file drops the layout and shows what
// most real-world search code actually costs. Label them honestly in the demo.

function isBoundary(c) {
  return c === 32 || c === 45 || c === 47 || c === 95
    || c === 46 || c === 44 || c === 58 || c === 40;
}

// Same fuzzy subsequence score as the kernel, but reading a JS string directly.
function scoreString(query, text) {
  const qlen = query.length;
  if (qlen === 0) return 1;
  const n = text.length;
  let qi = 0;
  let total = 0;
  let prevMatch = -2;
  let consec = 0;
  for (let hi = 0; hi < n && qi < qlen; hi += 1) {
    if (text.charCodeAt(hi) === query.charCodeAt(qi)) {
      let bonus = 1;
      if (hi === prevMatch + 1) {
        consec += 1;
        bonus += consec * 2;
      } else {
        consec = 0;
      }
      if (hi === 0) {
        bonus += 3;
      } else if (isBoundary(text.charCodeAt(hi - 1))) {
        bonus += 2;
      }
      if (n > 0) bonus += (n - hi) / n;
      total += bonus;
      prevMatch = hi;
      qi += 1;
    }
  }
  if (qi < qlen) return 0;
  return total;
}

/**
 * Build an idiomatic pure-JS search engine — same API as the other engines.
 * Scores every row string per query and ranks with a full sort over match
 * objects. The one concession a reasonable dev makes is lowercasing once.
 * @param {string[]} texts one searchable string per row (row index === result index)
 * @returns {{ search: (text: string, limit: number) => number[] }}
 */
export default function createNaiveJsSearchEngine(texts) {
  const lowerTexts = texts.map((t) => (t || '').toLowerCase());
  return {
    search(text, limit) {
      const q = text.toLowerCase();
      return lowerTexts
        .map((t, i) => ({ i, score: scoreString(q, t) }))
        .filter((r) => r.score > 0)
        .sort((a, b) => (b.score - a.score) || (a.i - b.i))
        .slice(0, limit)
        .map((r) => r.i);
    },
  };
}
