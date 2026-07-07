# fuzzysearch block — authoring

On-device **"elastic" search**: typo-tolerant fuzzy matching + BM25 relevance
ranking over an EDS query-index, entirely client-side (no backend, no server
round-trips). Paste this table into a Google Doc / Word / SharePoint document
where you want search-as-you-type. Every cell is optional; defaults are shown.

| fuzzysearch |                    |
| ----------- | ------------------ |
| source      | /query-index.json  |
| placeholder | Search the docs…   |
| limit       | 10                 |
| fields      | title, description, content, author, place |
| fuzziness   | 2                  |

**Fields**

- **source** — path (or absolute URL) to an EDS query-index JSON. Fetched once,
  on lazy init; never re-fetched per keystroke.
- **placeholder** — placeholder text for the input.
- **limit** — maximum number of results shown.
- **fields** — comma-separated query-index columns concatenated into the
  searchable text for each row. Columns missing from a row are ignored, so it is
  safe to list optional ones (e.g. `content`, `author`, `place`).
- **fuzziness** — maximum typo tolerance (edit distance), `0`–`2`. Default `2`.
  Elasticsearch-style AUTO applies underneath: short terms tolerate fewer edits
  (0 for ≤2 chars, 1 for ≤5, 2 otherwise), capped by this value. `0` = exact
  terms only.

**How it works**

- Query terms are matched against the index vocabulary with restricted
  Damerau–Levenshtein edit distance, so `serch` finds *search*, `teh` finds
  *the*, and `optimizaton` finds *optimization* (transpositions, insertions,
  deletions, substitutions all tolerated within the fuzziness).
- Results are ranked by **BM25** — the same term-frequency / inverse-document-
  frequency model Elasticsearch uses by default — with a fuzzy boost that decays
  with edit distance, so exact matches outrank typo'd ones.
- All of it runs in a tiny WebAssembly kernel over a flat in-memory index. A
  query-index is produced by an
  [index configuration](https://www.aem.live/developer/indexing); point `source`
  at a scoped index (e.g. `/docs/query-index.json`) to search a subtree.

**Notes**

- Both the WASM kernel and its JS reference twin read one precomputed idf table
  and produce byte-identical scores (the benchmark asserts maxΔ = 0).
- Speed: vs a hand-optimized typed-array JS twin, WASM is ~1.4× (the conservative
  floor the benchmark measures); vs the idiomatic JS you'd naturally write
  (string edit distance + Maps), it is ~15× — see `test/fuzzysearch.compare.html`,
  a type-your-own-typos demo, and `blocks/fuzzysearch/fuzzysearch.bench.mjs`.
