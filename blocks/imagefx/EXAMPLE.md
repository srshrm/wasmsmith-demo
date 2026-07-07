# imagefx block — authoring

Live image blur. The block rasterizes an image, then a radius slider re-blurs it
in real time in a SIMD WebAssembly kernel — smooth even on a large image, where a
per-channel JavaScript loop would stutter.

Paste this table into a Google Doc / Word / SharePoint document:

| imagefx |                    |
| ------- | ------------------ |
| source  | /media/hero.jpg    |
| radius  | 12                 |
| max     | 30                 |
| label   | Blur radius        |

**Fields**

- **source** — path (or absolute URL) to an image. If omitted, the block uses the
  first `img`/link in its own content. Large images are scaled to fit ~900 px on
  the long edge before blurring.
- **radius** — initial blur radius (default 12).
- **max** — slider maximum (default 30; the kernel caps at 64).
- **label** — slider label text.

**Notes**

- Image decode/scale uses the native canvas; only the blur runs in WebAssembly.
- The blur uses integer fixed-point math, so its output is byte-identical to a
  plain-JS implementation — the kernel is only faster, not different.
- Cross-origin images need CORS headers, or the canvas is tainted and pixel
  readback is blocked; the block reports this gracefully.
- See `test/imagefx.compare.html` for a live WASM-vs-JS slider comparison.
