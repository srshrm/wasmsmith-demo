// imagefx.ts — SIMD separable Gaussian blur for live image effects.
//
// A large-radius blur is a direct weighted sum over (2*radius+1) taps per pixel,
// per axis — millions of multiply-adds for a megapixel image, redone every time
// the radius slider moves. That is exactly where WASM SIMD pulls away: each RGBA
// pixel is widened to an i32x4 and the whole multiply-accumulate runs 4 channels
// at once, while JS has no vector type and grinds one channel at a time.
//
// Compiled with `--runtime stub` (no GC) + `--enable simd`. Integer fixed-point
// math throughout, so the output is BYTE-IDENTICAL to the JS twin (tolerance 0):
// the host supplies integer Gaussian weights plus a reciprocal/shift pair and
// both paths compute out = clamp((sum * recip) >> shift, 0, 255).
//
// ── Interop contract (see imagefx.manifest.json) ─────────────────────────────
//   reset()                                        rewind the bump allocator
//   alloc(byteLen) -> ptr                          reserve bytes (8-aligned)
//   blur(src, tmp, dst, w, h, weights, r, recip, shift)
//        src/tmp/dst : u8[w*h*4] RGBA (tmp is scratch for the separable passes)
//        weights     : i32[2*r+1] integer Gaussian taps
// Edges use clamp-to-edge. src/tmp/dst must be distinct buffers.

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

@inline
function clampi(v: i32, lo: i32, hi: i32): i32 {
  return v < lo ? lo : (v > hi ? hi : v);
}

// Widen the 4 RGBA bytes at `p` into an i32x4 lane vector (one op chain, no
// scalar lane inserts). v128.load reads 16 bytes; only the low 4 are used, and
// callers guarantee another buffer follows so the over-read never traps.
@inline
function loadPixel(p: i32): v128 {
  const bytes = v128.load(p);
  const lo16 = i16x8.extend_low_i8x16_u(bytes);
  return i32x4.extend_low_i16x8_u(lo16);
}

// Normalize an i32x4 accumulator (fixed-point) and store it as 4 clamped bytes.
@inline
function storePixel(dp: i32, acc: v128, recip: i32, shift: i32): void {
  const out = i32x4.shr_s(i32x4.mul(acc, i32x4.splat(recip)), shift);
  store<u8>(dp, <u8>clampi(i32x4.extract_lane(out, 0), 0, 255));
  store<u8>(dp + 1, <u8>clampi(i32x4.extract_lane(out, 1), 0, 255));
  store<u8>(dp + 2, <u8>clampi(i32x4.extract_lane(out, 2), 0, 255));
  store<u8>(dp + 3, <u8>clampi(i32x4.extract_lane(out, 3), 0, 255));
}

// Horizontal 1-D convolution: src -> dst, clamp-to-edge in x.
function hpass(
  srcPtr: i32,
  dstPtr: i32,
  width: i32,
  height: i32,
  weightsPtr: i32,
  radius: i32,
  recip: i32,
  shift: i32,
): void {
  const taps = 2 * radius + 1;
  const last = width - 1;
  for (let y = 0; y < height; y += 1) {
    const rowBase = y * width;
    for (let x = 0; x < width; x += 1) {
      let acc = i32x4.splat(0);
      for (let k = 0; k < taps; k += 1) {
        const xc = clampi(x + k - radius, 0, last);
        const pix = loadPixel(srcPtr + ((rowBase + xc) << 2));
        const w = load<i32>(weightsPtr + (k << 2));
        acc = i32x4.add(acc, i32x4.mul(pix, i32x4.splat(w)));
      }
      storePixel(dstPtr + ((rowBase + x) << 2), acc, recip, shift);
    }
  }
}

// Vertical 1-D convolution: src -> dst, clamp-to-edge in y.
function vpass(
  srcPtr: i32,
  dstPtr: i32,
  width: i32,
  height: i32,
  weightsPtr: i32,
  radius: i32,
  recip: i32,
  shift: i32,
): void {
  const taps = 2 * radius + 1;
  const last = height - 1;
  for (let y = 0; y < height; y += 1) {
    const dstRow = y * width;
    for (let x = 0; x < width; x += 1) {
      let acc = i32x4.splat(0);
      for (let k = 0; k < taps; k += 1) {
        const yc = clampi(y + k - radius, 0, last);
        const pix = loadPixel(srcPtr + (((yc * width) + x) << 2));
        const w = load<i32>(weightsPtr + (k << 2));
        acc = i32x4.add(acc, i32x4.mul(pix, i32x4.splat(w)));
      }
      storePixel(dstPtr + ((dstRow + x) << 2), acc, recip, shift);
    }
  }
}

// Separable Gaussian blur: horizontal pass into tmp, then vertical into dst.
export function blur(
  srcPtr: i32,
  tmpPtr: i32,
  dstPtr: i32,
  width: i32,
  height: i32,
  weightsPtr: i32,
  radius: i32,
  recip: i32,
  shift: i32,
): void {
  if (radius <= 0) {
    memory.copy(dstPtr, srcPtr, <usize>(width * height * 4));
    return;
  }
  hpass(srcPtr, tmpPtr, width, height, weightsPtr, radius, recip, shift);
  vpass(tmpPtr, dstPtr, width, height, weightsPtr, radius, recip, shift);
}
