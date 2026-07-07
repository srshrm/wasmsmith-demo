# simhash block — authoring

Near-duplicate and SEO-canonical detection over your content. The block loads a
list of documents, computes a 64-bit **SimHash** fingerprint of each in a
WebAssembly kernel, then groups documents whose fingerprints are within a small
**Hamming distance** — surfacing pages that are near-duplicates of one another,
client-side, with no backend call.

Paste this table into a Google Doc / Word / SharePoint document:

| simhash   |                     |
| --------- | ------------------- |
| source    | /data/documents.json |
| threshold | 3                   |
| label     | Near-duplicate groups |

**Fields**

- **source** — path (or absolute URL) to the documents JSON. If omitted, the
  block looks for a `.json` link in its own content.
- **threshold** — maximum Hamming distance (differing bits, 0–64) for two
  documents to count as near-duplicates (default 3). Larger = looser matching.
- **label** — heading text for the results.

**Documents file format**

```json
{
  "docs": [
    { "title": "Reset your password",     "url": "/help/reset",  "text": "To reset your password, open settings and…" },
    { "title": "Reset password (mirror)",  "url": "/kb/reset",    "text": "To reset your password, open settings and…" },
    { "title": "Enable two-factor auth",   "url": "/help/2fa",    "text": "Two-factor authentication adds a second…" }
  ]
}
```

- Each doc has a `title`, optional `url`, and the `text` to fingerprint.
- The host lowercases the text and hashes maximal `[a-z0-9]` token runs; every
  other character is a separator. Fingerprinting is deterministic.

**Why WebAssembly**

The hot loop computes a **64-bit FNV-1a hash** of every token in every document.
JavaScript numbers are IEEE-754 `f64` and cannot hold exact 64-bit integers, so
a correct JS implementation must fall back to **BigInt** — heap-allocated big
integers that are ~15–40× slower than native machine words. The WASM kernel uses
the native `i64`/`u64` type, so the same math runs in registers. No SIMD needed.

**Notes**

- The fingerprint is a 64-bit word; near-duplicate detection is a bit-count of
  the XOR of two fingerprints (Hamming distance), exposed as `hamming(a, b)` in
  the engine.
- See `test/simhash.compare.html` for a WASM-vs-JS fingerprinting-speed
  comparison over a synthetic document set (no documents file required).
