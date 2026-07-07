# semanticsearch block — authoring

"Find similar content" powered by embedding cosine similarity. The block loads a
precomputed embeddings file, indexes it in a SIMD WebAssembly kernel, and — when
a reader picks a document — ranks every other document by nearest-neighbour
similarity, client-side, with no backend call.

Paste this table into a Google Doc / Word / SharePoint document:

| semanticsearch |                        |
| -------------- | ---------------------- |
| source         | /data/embeddings.json  |
| topk           | 8                      |
| label          | Similar content        |

**Fields**

- **source** — path (or absolute URL) to the embeddings JSON. If omitted, the
  block looks for a `.json` link in its own content.
- **topk** — how many neighbours to show (default 8).
- **label** — heading text.

**Embeddings file format**

```json
{
  "dim": 384,
  "docs": [
    { "title": "Reset your password", "url": "/help/reset", "embedding": [0.01, -0.02, ...] },
    { "title": "Two-factor setup",    "url": "/help/2fa",   "embedding": [0.03,  0.00, ...] }
  ]
}
```

- `dim` is the embedding dimension (e.g. 384 for MiniLM-class models). All
  `embedding` arrays must be that length. A multiple of 4 lets the SIMD kernel
  run with no scalar tail.
- Embeddings are **precomputed at publish time** (by whatever model you use) and
  shipped as a static asset. No model runs in the browser — only the similarity
  ranking does. The query is an existing document's own vector ("find similar
  to this"), so no query-time embedding is needed.

**Notes**

- Vectors are L2-normalized on load, so cosine similarity is a plain dot product.
- See `test/semanticsearch.compare.html` for a WASM-vs-JS ranking-speed
  comparison over a synthetic corpus (no embeddings file required).
