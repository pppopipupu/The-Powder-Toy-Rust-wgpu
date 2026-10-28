//! Radix-2 FFT kernels for the Newtonian gravity Poisson solve.
//!
//! The solve is a cyclic convolution of the padded mass grid with the
//! point-mass velocity field, so it needs one forward and two inverse 2-D
//! transforms per frame (see [`super`] for the frame wiring). Each transform is
//! a bit-reversal pass plus `log2(n)` butterfly passes; every pass is its own
//! dispatch, which keeps the kernels free of barriers and shared memory and
//! makes the whole pipeline run on the device without a read-back.
//!
//! Complex samples are interleaved (`2 * i` = real, `2 * i + 1` = imaginary) in
//! one `f32` buffer per grid, so a transform stays in place and each pass binds
//! a single data buffer plus the shared twiddle table.
//!
//! # Index layout
//!
//! Every pass takes `(rows, len, inner, stride)`: sequence `r` occupies the
//! indices `r * inner + i * stride` for `i in 0..len`. A row transform is
//! `(height, width, width, 1)`, a column transform `(width, height, 1, width)`,
//! so the same kernel serves both axes.

use cubecl::prelude::*;

/// Runtime-seeded `u32`.
///
/// A literal `let mut n = 0u32;` expands to a plain host integer, and the next
/// runtime assignment then fails to compile with `expected u32, found
/// NativeExpand<u32>`; seeding through a `#[cube]` call keeps it a device value.
#[cube]
fn seed_u32(v: u32) -> u32 {
    v
}

/// In-place bit-reversal permutation of `rows` sequences of length `len`.
#[cube(launch_unchecked)]
pub fn fft_bitrev_kernel(
    data: &mut [f32],
    rows: u32,
    len: u32,
    inner: u32,
    stride: u32,
    bits: u32,
) {
    let id = ABSOLUTE_POS;
    if id >= (rows * len) as usize {
        terminate!();
    }
    let r = (id as u32) / len;
    let i = (id as u32) % len;

    let mut src = i;
    let mut rev = seed_u32(0u32);
    let mut b = seed_u32(0u32);
    while b < bits {
        rev = rev * 2u32 + src % 2u32;
        src /= 2u32;
        b += 1u32;
    }

    if i < rev {
        let a = ((r * inner + i * stride) * 2u32) as usize;
        let c = ((r * inner + rev * stride) * 2u32) as usize;
        let re = data[a];
        let im = data[a + 1usize];
        data[a] = data[c];
        data[a + 1usize] = data[c + 1usize];
        data[c] = re;
        data[c + 1usize] = im;
    }
}

/// One butterfly pass of a decimation-in-time radix-2 FFT.
///
/// `half` is `len / 2` for this pass; a butterfly pairs `i0` with `i0 + half`
/// inside every block of `len` samples. Each sample belongs to exactly one
/// butterfly, so the in-place update needs no synchronisation. `sign` is `+1`
/// for the forward transform and `-1` for the inverse (the twiddle table stores
/// `exp(-2*pi*i*k/max_len)`).
#[cube(launch_unchecked)]
pub fn fft_stage_kernel(
    data: &mut [f32],
    rows: u32,
    n: u32,
    half: u32,
    inner: u32,
    stride: u32,
    tw_re: &[f32],
    tw_im: &[f32],
    tw_shift: u32,
    sign: f32,
) {
    let id = ABSOLUTE_POS;
    let per_row = n / 2u32;
    if id >= (rows * per_row) as usize {
        terminate!();
    }
    let r = (id as u32) / per_row;
    let k = (id as u32) % per_row;
    let len = half * 2u32;
    let block = k / half;
    let pos = k % half;
    let i0 = block * len + pos;
    let i1 = i0 + half;

    let t = (pos << tw_shift) as usize;
    let wr = tw_re[t];
    let wi = tw_im[t] * sign;

    let a = ((r * inner + i0 * stride) * 2u32) as usize;
    let b = ((r * inner + i1 * stride) * 2u32) as usize;
    let xr = data[a];
    let xi = data[a + 1usize];
    let yr = data[b];
    let yi = data[b + 1usize];

    let tr = wr * yr - wi * yi;
    let ti = wr * yi + wi * yr;

    data[a] = xr + tr;
    data[a + 1usize] = xi + ti;
    data[b] = xr - tr;
    data[b + 1usize] = xi - ti;
}

/// Pointwise complex product `out = work * kernel`.
///
/// The convolution theorem needs `FFT(mass) * FFT(kernel)` for both force
/// components; one launch fills both spectra so the mass spectrum is read once.
#[cube(launch_unchecked)]
pub fn complex_mul_kernel(work: &[f32], kernel: &[f32], out: &mut [f32], cells: u32) {
    let i = ABSOLUTE_POS;
    if i < cells as usize {
        let a = i * 2usize;
        let wr = work[a];
        let wi = work[a + 1usize];
        let kr = kernel[a];
        let ki = kernel[a + 1usize];
        out[a] = wr * kr - wi * ki;
        out[a + 1usize] = wr * ki + wi * kr;
    }
}

/// Zero-pads the coarse mass grid into the top-left quadrant of the transform
/// grid.
///
/// The kernel is built centred on index `(0, 0)` and wrapped (`ker[u] = K(u)`
/// with `u` folded into `(-N/2, N/2]`), so the cyclic convolution of the FFT
/// equals the physical one as long as the padding is at least twice the grid:
/// every offset inside the quadrant is then shorter than half a period and
/// never wraps onto the result. The mass accumulator is fixed-point
/// (`1 / 65536` mass per unit), so the conversion happens here; cells the
/// gravity mask blocks contribute nothing.
#[cube(launch_unchecked)]
pub fn build_padded_mass_kernel(
    mass: &[i32],
    mask: &[u32],
    work: &mut [f32],
    width: u32,
    height: u32,
    xcells: u32,
    ycells: u32,
    mass_scale_inv: f32,
) {
    let id = ABSOLUTE_POS;
    if id < (width * height) as usize {
        let px = (id as u32) % width;
        let py = (id as u32) / width;
        let mut value = 0.0f32;
        if px < xcells && py < ycells {
            let cell = py * xcells + px;
            if mask[cell as usize] != 0u32 {
                value = mass[cell as usize] as f32 * mass_scale_inv;
            }
        }
        let a = id * 2usize;
        work[a] = value;
        work[a + 1usize] = 0.0f32;
    }
}

/// Writes the real part of the top-left transform quadrant into the force grid.
#[cube(launch_unchecked)]
pub fn extract_force_kernel(
    spec: &[f32],
    mask: &[u32],
    out: &mut [f32],
    width: u32,
    xcells: u32,
    ycells: u32,
) {
    let id = ABSOLUTE_POS;
    if id < (xcells * ycells) as usize {
        let px = (id as u32) % xcells;
        let py = (id as u32) / xcells;
        let mut value = 0.0f32;
        if mask[id] != 0u32 {
            value = spec[(py * width + px) as usize * 2usize];
        }
        out[id] = value;
    }
}

/// Point-mass velocity field used as the convolution kernel.
///
/// `out` receives `scale * d / |d|^3` for the offset `d` of each cell from the
/// transform origin: index `(px, py)` maps to `d = (px, py)` folded into
/// `(-N/2, N/2]`, so the array is the kernel wrapped for a cyclic convolution.
/// `axis` selects the x (`0`) or y (`1`) component.
#[cube(launch_unchecked)]
pub fn build_kernel_field_kernel(out: &mut [f32], width: u32, height: u32, scale: f32, axis: u32) {
    let id = ABSOLUTE_POS;
    if id < (width * height) as usize {
        let px = (id as u32) % width;
        let py = (id as u32) / width;
        let mut dx = px as f32;
        if px * 2u32 >= width {
            dx = px as f32 - width as f32;
        }
        let mut dy = py as f32;
        if py * 2u32 >= height {
            dy = py as f32 - height as f32;
        }
        let mut value = 0.0f32;
        if dx != 0.0f32 || dy != 0.0f32 {
            let distance = (dx * dx + dy * dy).sqrt();
            let inv = scale / (distance * distance * distance);
            if axis == 0u32 {
                value = dx * inv;
            } else {
                value = dy * inv;
            }
        }
        let a = id * 2usize;
        out[a] = value;
        out[a + 1usize] = 0.0f32;
    }
}

/// Builds the twiddle table `exp(-2*pi*i*k / max_len)` for `k < max_len / 2`.
///
/// Every pass reads it at `pos << (log2(max_len) - log2(len))`, so one table
/// serves all transform lengths that are powers of two up to `max_len`.
pub fn build_twiddle(max_len: usize) -> (Vec<f32>, Vec<f32>) {
    let half = (max_len / 2).max(1);
    let mut re = Vec::with_capacity(half);
    let mut im = Vec::with_capacity(half);
    for k in 0..half {
        let angle = -2.0 * std::f32::consts::PI * (k as f32) / (max_len as f32);
        re.push(angle.cos());
        im.push(angle.sin());
    }
    (re, im)
}

/// Number of bits needed to represent `n - 1`, i.e. `log2(n)` for powers of two.
pub fn bit_count(n: u32) -> u32 {
    let mut bits = 0u32;
    let mut v = n.max(1) - 1;
    while v > 0 {
        v >>= 1;
        bits += 1;
    }
    bits
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Host mirror of [`fft_bitrev_kernel`] + [`fft_stage_kernel`].
    ///
    /// The kernels cannot be called from a host test, so the exact arithmetic
    /// they perform is repeated here and checked against a naive DFT.
    #[allow(clippy::too_many_arguments)]
    fn fft_host(
        data: &mut [f32],
        rows: usize,
        n: usize,
        inner: usize,
        stride: usize,
        tw_re: &[f32],
        tw_im: &[f32],
        tw_bits: u32,
        sign: f32,
    ) {
        let bits = bit_count(n as u32);
        for r in 0..rows {
            for i in 0..n {
                let mut src = i as u32;
                let mut rev = 0u32;
                for _ in 0..bits {
                    rev = rev * 2 + src % 2;
                    src /= 2;
                }
                let rev = rev as usize;
                if i < rev {
                    let a = (r * inner + i * stride) * 2;
                    let c = (r * inner + rev * stride) * 2;
                    data.swap(a, c);
                    data.swap(a + 1, c + 1);
                }
            }
            let mut half = 1usize;
            while half < n {
                let len = half * 2;
                let tw_shift = tw_bits - bit_count(len as u32);
                for k in 0..n / 2 {
                    let block = k / half;
                    let pos = k % half;
                    let i0 = block * len + pos;
                    let i1 = i0 + half;
                    let t = pos << tw_shift;
                    let wr = tw_re[t];
                    let wi = tw_im[t] * sign;
                    let a = (r * inner + i0 * stride) * 2;
                    let b = (r * inner + i1 * stride) * 2;
                    let (xr, xi) = (data[a], data[a + 1]);
                    let (yr, yi) = (data[b], data[b + 1]);
                    let tr = wr * yr - wi * yi;
                    let ti = wr * yi + wi * yr;
                    data[a] = xr + tr;
                    data[a + 1] = xi + ti;
                    data[b] = xr - tr;
                    data[b + 1] = xi - ti;
                }
                half = len;
            }
        }
    }

    /// Naive 2-D DFT of a row-major `h x w` complex grid.
    fn dft2d(re: &[f32], im: &[f32], w: usize, h: usize, sign: f32) -> (Vec<f32>, Vec<f32>) {
        let mut out_re = vec![0.0f32; w * h];
        let mut out_im = vec![0.0f32; w * h];
        for ky in 0..h {
            for kx in 0..w {
                let mut sr = 0.0f32;
                let mut si = 0.0f32;
                for y in 0..h {
                    for x in 0..w {
                        let angle = sign
                            * 2.0
                            * std::f32::consts::PI
                            * ((kx * x) as f32 / w as f32 + (ky * y) as f32 / h as f32);
                        let (s, c) = angle.sin_cos();
                        let a = y * w + x;
                        sr += re[a] * c - im[a] * s;
                        si += re[a] * s + im[a] * c;
                    }
                }
                out_re[ky * w + kx] = sr;
                out_im[ky * w + kx] = si;
            }
        }
        (out_re, out_im)
    }

    #[test]
    fn mirrored_passes_match_a_naive_dft() {
        let (w, h) = (8usize, 8usize);
        let tw_bits = bit_count(w.max(h) as u32);
        let (tw_re, tw_im) = build_twiddle(1usize << tw_bits);

        let mut re = vec![0.0f32; w * h];
        let mut im = vec![0.0f32; w * h];
        for y in 0..h {
            for x in 0..w {
                re[y * w + x] = ((x * 7 + y * 3) % 11) as f32 - 5.0;
                im[y * w + x] = ((x * 5 + y * 9) % 7) as f32 - 3.0;
            }
        }

        let mut data = vec![0.0f32; w * h * 2];
        for i in 0..w * h {
            data[i * 2] = re[i];
            data[i * 2 + 1] = im[i];
        }
        // Row pass, then column pass (the exact sequence the driver issues).
        fft_host(&mut data, h, w, w, 1, &tw_re, &tw_im, tw_bits, 1.0);
        fft_host(&mut data, w, h, 1, w, &tw_re, &tw_im, tw_bits, 1.0);

        let (want_re, want_im) = dft2d(&re, &im, w, h, -1.0);
        for i in 0..w * h {
            let got_re = data[i * 2];
            let got_im = data[i * 2 + 1];
            assert!(
                (got_re - want_re[i]).abs() < 1e-3 && (got_im - want_im[i]).abs() < 1e-3,
                "bin {i}: got ({got_re}, {got_im}) want ({}, {})",
                want_re[i],
                want_im[i]
            );
        }
    }

    #[test]
    fn forward_then_inverse_scales_by_the_sample_count() {
        let (w, h) = (4usize, 8usize);
        let tw_bits = bit_count(w.max(h) as u32);
        let (tw_re, tw_im) = build_twiddle(1usize << tw_bits);
        let mut data = vec![0.0f32; w * h * 2];
        for i in 0..w * h {
            data[i * 2] = (i as f32) * 0.25 - 1.0;
        }
        let original = data.clone();

        fft_host(&mut data, h, w, w, 1, &tw_re, &tw_im, tw_bits, 1.0);
        fft_host(&mut data, w, h, 1, w, &tw_re, &tw_im, tw_bits, 1.0);
        fft_host(&mut data, h, w, w, 1, &tw_re, &tw_im, tw_bits, -1.0);
        fft_host(&mut data, w, h, 1, w, &tw_re, &tw_im, tw_bits, -1.0);

        let scale = (w * h) as f32;
        for i in 0..w * h {
            assert!((data[i * 2] / scale - original[i * 2]).abs() < 1e-3);
        }
    }
}
