// imagehash.ts — SIMD perceptual hash (pHash) for near-duplicate image detection.
//
// pHash fingerprints an image by taking a 32×32 grayscale reduction, running a 2-D
// DCT, keeping the top-left 8×8 low-frequency block, and setting one bit per
// coefficient (above the block median = 1). Near-duplicates then differ in only a
// few bits (small Hamming distance). Scanning a whole asset library means running
// that DCT thousands of times — the compute-bound part worth moving to WASM.
//
// The DCT is a separable matrix multiply. We vectorize ACROSS 4 output frequencies
// with f32x4, so each coefficient still accumulates over its input index in scalar
// order — the numerics stay a faithful match to the JS twin. Compiled with
// `--runtime stub` + `--enable simd`.
//
// ── Interop contract (see imagehash.manifest.json) ───────────────────────────
//   reset()/alloc(byteLen)  standard bump allocator
//   dct8x8(gray, cos, tmp, outCoeff)      OUT f32[64] low-freq DCT block (bench)
//   phashBatch(gray, count, cos, tmp, coeff, sort, outHash)
//                                         OUT u32[2*count] 64-bit hash per image
// gray: f32[32*32] per image (host builds it: canvas downscale + grayscale).
// cos : f32[32*32] host-built table  C[i][j] = alpha(j)*cos(pi*(2i+1)*j/(2*32)).

const N: i32 = 32; // reduced grayscale side
const BLK: i32 = 8; // low-frequency block side (64-bit hash)

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

// Separable 2-D DCT keeping only the 8×8 low-frequency block.
//   stage 1 (rows): T[y][u] = sum_x gray[y][x] * C[x][u]   for u in 0..7
//   stage 2 (cols): D[v][u] = sum_y T[y][u]   * C[y][v]   for v,u in 0..7
// tmp holds T as [N][BLK] f32; coeff receives D as [BLK][BLK] f32.
function computeDct(grayPtr: i32, cosPtr: i32, tmpPtr: i32, coeffPtr: i32): void {
  // stage 1: for each row, 8 horizontal-frequency outputs (two f32x4 lanes)
  for (let y = 0; y < N; y += 1) {
    const rowGray = grayPtr + y * N * 4;
    let acc0 = f32x4.splat(0.0);
    let acc4 = f32x4.splat(0.0);
    for (let x = 0; x < N; x += 1) {
      const fv = f32x4.splat(load<f32>(rowGray + (x << 2)));
      const crow = cosPtr + x * N * 4; // C[x][*]
      acc0 = f32x4.add(acc0, f32x4.mul(fv, v128.load(crow)));
      acc4 = f32x4.add(acc4, f32x4.mul(fv, v128.load(crow + 16)));
    }
    v128.store(tmpPtr + y * BLK * 4, acc0);
    v128.store(tmpPtr + y * BLK * 4 + 16, acc4);
  }
  // stage 2: for each vertical frequency v, 8 outputs summed over rows
  for (let v = 0; v < BLK; v += 1) {
    let acc0 = f32x4.splat(0.0);
    let acc4 = f32x4.splat(0.0);
    for (let y = 0; y < N; y += 1) {
      const cy = f32x4.splat(load<f32>(cosPtr + ((y * N + v) << 2))); // C[y][v]
      const trow = tmpPtr + y * BLK * 4;
      acc0 = f32x4.add(acc0, f32x4.mul(cy, v128.load(trow)));
      acc4 = f32x4.add(acc4, f32x4.mul(cy, v128.load(trow + 16)));
    }
    v128.store(coeffPtr + v * BLK * 4, acc0);
    v128.store(coeffPtr + v * BLK * 4 + 16, acc4);
  }
}

// 64 coefficients -> 64-bit hash: bit set where coeff > block median.
// sort is 64-f32 scratch; out receives {u32 low; u32 high}.
function hashCoeffs(coeffPtr: i32, sortPtr: i32, outPtr: i32): void {
  for (let i = 0; i < 64; i += 1) {
    store<f32>(sortPtr + (i << 2), load<f32>(coeffPtr + (i << 2)));
  }
  // insertion sort ascending (64 elements — tiny)
  for (let i = 1; i < 64; i += 1) {
    const key = load<f32>(sortPtr + (i << 2));
    let j = i - 1;
    while (j >= 0 && load<f32>(sortPtr + (j << 2)) > key) {
      store<f32>(sortPtr + ((j + 1) << 2), load<f32>(sortPtr + (j << 2)));
      j -= 1;
    }
    store<f32>(sortPtr + ((j + 1) << 2), key);
  }
  const median: f32 = (load<f32>(sortPtr + (31 << 2)) + load<f32>(sortPtr + (32 << 2))) * <f32>0.5;

  let lo: u32 = 0;
  let hi: u32 = 0;
  for (let i = 0; i < 64; i += 1) {
    if (load<f32>(coeffPtr + (i << 2)) > median) {
      if (i < 32) lo |= (<u32>1 << <u32>i);
      else hi |= (<u32>1 << <u32>(i - 32));
    }
  }
  store<u32>(outPtr, lo);
  store<u32>(outPtr + 4, hi);
}

// Low-frequency 8×8 DCT block for one image.
export function dct8x8(grayPtr: i32, cosPtr: i32, tmpPtr: i32, outCoeffPtr: i32): void {
  computeDct(grayPtr, cosPtr, tmpPtr, outCoeffPtr);
}

// DCT blocks for `count` images in one call — the library sweep the benchmark
// times. Writes count*64 coefficients back-to-back at outCoeffPtr.
export function dctBatch(
  grayPtr: i32,
  count: i32,
  cosPtr: i32,
  tmpPtr: i32,
  outCoeffPtr: i32,
): void {
  const imgStride = N * N * 4;
  for (let n = 0; n < count; n += 1) {
    computeDct(grayPtr + n * imgStride, cosPtr, tmpPtr, outCoeffPtr + n * 64 * 4);
  }
}

// Fingerprint `count` images back-to-back; only the count*8-byte hash table
// crosses back to JS. tmp/coeff/sort are shared scratch reused each image.
export function phashBatch(
  grayPtr: i32,
  count: i32,
  cosPtr: i32,
  tmpPtr: i32,
  coeffPtr: i32,
  sortPtr: i32,
  outPtr: i32,
): void {
  const imgStride = N * N * 4;
  for (let n = 0; n < count; n += 1) {
    computeDct(grayPtr + n * imgStride, cosPtr, tmpPtr, coeffPtr);
    hashCoeffs(coeffPtr, sortPtr, outPtr + n * 8);
  }
}
