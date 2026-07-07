# svgpng block — authoring

Paste this table into a Google Doc / Word / SharePoint document. The block loads
the SVG, rasterizes it with the browser canvas, and encodes a downloadable PNG in
WebAssembly. Every cell is optional except `source`.

| svgpng |                     |
| ------ | ------------------- |
| source | /media/logo.svg     |
| width  | 512                 |
| height | 512                 |
| label  | Convert & download PNG |

**Fields**

- **source** — path (or absolute URL) to an SVG. If omitted, the block looks for
  an `.svg` link or image in its own content.
- **width** / **height** — output raster size in pixels. If omitted, the SVG's
  intrinsic size is used (falling back to 512×512).
- **label** — text on the convert button.

**Notes**

- Rasterization uses the native canvas; only the PNG encoding (filter + DEFLATE
  compression) runs in WebAssembly. Output is a true 8-bit RGBA PNG.
- The SVG must be same-origin (or CORS-enabled); otherwise the canvas is tainted
  and the browser blocks pixel readback. The block reports this gracefully.
- See `test/svgpng.compare.html` for a WASM-vs-JS encode-speed comparison.
