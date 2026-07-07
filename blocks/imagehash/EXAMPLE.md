# imagehash block — authoring

Find near-duplicate images in a gallery. Each image is reduced to a 32×32
grayscale, perceptually hashed with a SIMD DCT in WebAssembly, and grouped with
others whose fingerprints are within a few bits — the dedup pass a DAM runs over
an asset library, here fully client-side.

Two ways to author it.

**A. Point at a JSON list**

| imagehash |                    |
| --------- | ------------------ |
| source    | /data/assets.json  |
| threshold | 8                  |
| label     | Near-duplicate groups |

```json
{ "images": [ { "url": "/media/a.jpg", "title": "A" }, { "url": "/media/a-copy.jpg" } ] }
```

**B. Just put images in the block** (the block reads its own `img`/links)

| imagehash |
| --------- |
| ![](/media/hero.jpg) ![](/media/hero-2.jpg) ![](/media/other.jpg) |

**Fields**

- **source** — path (or absolute URL) to a JSON `{ images: [{ url, title? }] }`.
  If omitted, the block fingerprints the images already inside it.
- **threshold** — max Hamming distance (differing bits, 0–64) to treat two images
  as near-duplicates. Default 8; lower = stricter.
- **label** — heading text.

**Notes**

- Downscale + grayscale use the native canvas; only the DCT fingerprinting runs
  in WebAssembly. Cross-origin images need CORS headers or the canvas is tainted.
- The DCT runs in f32 (SIMD) vs f64 in plain JS, so fingerprints match in
  practice (to a bit or two) — robust for near-duplicate grouping.
- See `test/imagehash.compare.html` for a WASM-vs-JS fingerprinting-speed
  comparison over a synthetic library.
