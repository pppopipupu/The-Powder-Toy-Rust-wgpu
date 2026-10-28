//! Device-side FFT Poisson solve for Newtonian gravity.
//!
//! Port of `src/simulation/gravity/Fft.cpp` that keeps the whole solve on the
//! GPU: the mass grid is zero-padded into the transform grid, forward
//! transformed, multiplied by the precomputed point-mass spectra and inverse
//! transformed back into the force grids, all in the frame's command stream.
//! No per-frame host work and no read-back remain; the host shadows are
//! refreshed on demand instead of every tick.
//!
//! # Transform size, mass placement and scale vs `Fft.cpp`
//!
//! The reference uses `blocks = CELLS * 2` exactly (`Fft.cpp:5-7`), places the
//! mass at `massBig[p + CELLS]` (`Fft.cpp:110`), reads the force at `p`
//! (`Fft.cpp:130-131`), builds the kernel at `d = p - CELLS` (`Fft.cpp:165`)
//! and scales it by `-M_GRAV / (NCELL * 4)` (`Fft.cpp:7`).
//!
//! This solver must use a radix-2 transform, so each axis is padded to
//! `next_power_of_two(2 * cells)` (`>= 2 * cells`). With the mass in the
//! top-left quadrant and the kernel stored wrapped around the transform origin
//! (`ker[u] = K(u)`, `u` folded into `(-N/2, N/2]`), the cyclic convolution is
//!
//! ```text
//! force[p] = sum_m mass[m] * K((p - m) mod N)
//! ```
//!
//! and every intra-quadrant offset satisfies `|p - m| < cells <= N/2`, so the
//! wrapped index folds back to exactly `p - m`: the result is the same
//! physical sum as the reference's `mass at +CELLS` / `kernel centred at
//! +CELLS` layout for its `2 * cells` period. The larger period can only
//! remove wrap contamination, never add it.
//!
//! The scale is `-M_GRAV / (width * height)` rather than `-M_GRAV / (NCELL * 4)`
//! because the inverse transform is unnormalised and multiplies the convolution
//! by the **actual** padded cell count `width * height`; for the reference's
//! `2 * CELLS` grid those two expressions are identical
//! (`NCELL * 4 = (2 * XCELLS) * (2 * YCELLS)`). In both cases the transform
//! factor cancels and the kernel evaluates to the physical
//! `-M_GRAV * d / |d|^3`. The device-vs-brute-force regression test
//! (`tests/gpu_sim/gravity_fft.rs`) pins this equivalence on a canvas whose
//! padded size is strictly larger than `2 * cells` (`96x96` -> `24x24` cells ->
//! `64x64` transform).
//!
//! Kernel spectra depend on the grid size alone, so they are built once per
//! canvas size (two forward transforms) and reused until the canvas changes.

pub mod fft;

use cubecl::bytes::Bytes;
use cubecl::client::Client;
use cubecl::prelude::*;
use cubecl::server::Handle;

use crate::buffers::SimBuffers;

/// Work units per cube; the portable WebGPU minimum for a 1-D cube.
const CUBE_DIM: u32 = 256;

/// Bytes per `f32` device element.
const WORD: usize = 4;

/// TPT's gravitational constant (`M_GRAV`).
const M_GRAV: f32 = 6.673e-1;

/// Device buffers for one canvas size.
struct FftBuffers {
    /// Padded transform width (power of two, `>= 2 * xcells`).
    width: u32,
    /// Padded transform height (power of two, `>= 2 * ycells`).
    height: u32,
    xcells: u32,
    ycells: u32,
    /// `log2(max(width, height))`, the exponent of the twiddle table.
    tw_bits: u32,
    /// Padded mass grid, transformed in place into its spectrum.
    work: Handle,
    /// Force spectra for x and y.
    spec_x: Handle,
    spec_y: Handle,
    /// Point-mass velocity spectra for x and y, rebuilt on a grid change.
    kernel_x: Handle,
    kernel_y: Handle,
    tw_re: Handle,
    tw_im: Handle,
    /// Whether `kernel_x` / `kernel_y` hold a valid spectrum.
    kernels_ready: bool,
}

/// GPU FFT Poisson solver for Newtonian gravity.
pub struct GravityFft {
    buffers: Option<FftBuffers>,
}

impl Default for GravityFft {
    fn default() -> Self {
        Self::new()
    }
}

impl GravityFft {
    /// Creates a solver with no device buffers; the first [`Self::solve`]
    /// allocates them for the current canvas.
    pub fn new() -> Self {
        Self { buffers: None }
    }

    /// Runs the whole solve for the given mass grid and writes `grav_fx` /
    /// `grav_fy` on the device.
    ///
    /// `mass` is the fixed-point mass accumulator to transform
    /// ([`crate::buffers::SimBuffers::gpu_grav_mass_prev`], the previous
    /// frame's accumulation, matching `BeforeSim`'s dispatch-then-clear order,
    /// `Simulation.cpp:3737-3742`). `mass_scale_inv` converts the fixed-point
    /// words back to mass units (the accumulator stores `mass * 65536`).
    pub fn solve(&mut self, bufs: &SimBuffers, mass: &Handle, mass_scale_inv: f32) {
        let xcells = bufs.config.xcells();
        let ycells = bufs.config.ycells();
        if xcells == 0 || ycells == 0 {
            return;
        }
        let client = bufs.client.clone();

        let rebuild = match &self.buffers {
            Some(state) => state.xcells != xcells || state.ycells != ycells,
            None => true,
        };
        if rebuild {
            self.buffers = Some(FftBuffers::allocate(&client, xcells, ycells));
        }
        let state = self.buffers.as_mut().expect("buffers were just allocated");

        if !state.kernels_ready {
            state.build_kernels(&client);
        }
        state.solve(&client, bufs, mass, mass_scale_inv);
    }
}

impl FftBuffers {
    /// Allocates the transform buffers and uploads the twiddle table.
    fn allocate(client: &Client, xcells: u32, ycells: u32) -> Self {
        let width = (xcells * 2).next_power_of_two();
        let height = (ycells * 2).next_power_of_two();
        let tw_bits = fft::bit_count(width.max(height));

        let bytes = (width as usize) * (height as usize) * 2 * WORD;
        let (tw_re, tw_im) = fft::build_twiddle(1usize << tw_bits);
        let tw_re_handle = client.empty(tw_re.len() * WORD);
        let tw_im_handle = client.empty(tw_im.len() * WORD);
        client.write(&tw_re_handle, Bytes::from_elems(tw_re));
        client.write(&tw_im_handle, Bytes::from_elems(tw_im));

        Self {
            width,
            height,
            xcells,
            ycells,
            tw_bits,
            work: client.empty(bytes),
            spec_x: client.empty(bytes),
            spec_y: client.empty(bytes),
            kernel_x: client.empty(bytes),
            kernel_y: client.empty(bytes),
            tw_re: tw_re_handle,
            tw_im: tw_im_handle,
            kernels_ready: false,
        }
    }

    /// Builds and forward-transforms both point-mass kernels.
    fn build_kernels(&mut self, client: &Client) {
        let width = self.width;
        let height = self.height;
        let cells = (width * height) as usize;
        let elements = cells * 2;

        // The inverse transform is unnormalised, so it multiplies the
        // convolution by the padded cell count; the kernel carries the inverse
        // of that factor, which makes the result the physical `-M_GRAV * d/|d|^3`
        // sum (the same cancellation FFTW relies on in `Fft.cpp`).
        let scale = -M_GRAV / (width * height) as f32;

        let (cubes, dim) = launch_grid(cells);
        for (buffer, axis) in [(self.kernel_x.clone(), 0u32), (self.kernel_y.clone(), 1u32)] {
            unsafe {
                fft::build_kernel_field_kernel::launch_unchecked(
                    client,
                    CubeCount::Static(cubes, 1, 1),
                    CubeDim::new_1d(dim),
                    BufferArg::from_raw_parts(buffer.clone(), elements),
                    width,
                    height,
                    scale,
                    axis,
                );
            }
            self.transform(client, &buffer, elements, height, width, width, 1, 1.0);
            self.transform(client, &buffer, elements, width, height, 1, width, 1.0);
        }
        self.kernels_ready = true;
    }

    /// Builds the padded mass spectrum, multiplies it by both kernels, inverts
    /// the products and extracts the force grids.
    fn solve(&self, client: &Client, bufs: &SimBuffers, mass: &Handle, mass_scale_inv: f32) {
        let width = self.width;
        let height = self.height;
        let xcells = self.xcells;
        let ycells = self.ycells;
        let cells = (width * height) as usize;
        let elements = cells * 2;
        let force_cells = (xcells * ycells) as usize;

        let (cubes, dim) = launch_grid(cells);
        unsafe {
            fft::build_padded_mass_kernel::launch_unchecked(
                client,
                CubeCount::Static(cubes, 1, 1),
                CubeDim::new_1d(dim),
                BufferArg::from_raw_parts(mass.clone(), force_cells),
                BufferArg::from_raw_parts(bufs.gpu_grav_mask.clone(), force_cells),
                BufferArg::from_raw_parts(self.work.clone(), elements),
                width,
                height,
                xcells,
                ycells,
                mass_scale_inv,
            );
        }

        self.transform(client, &self.work, elements, height, width, width, 1, 1.0);
        self.transform(client, &self.work, elements, width, height, 1, width, 1.0);

        unsafe {
            fft::complex_mul_kernel::launch_unchecked(
                client,
                CubeCount::Static(cubes, 1, 1),
                CubeDim::new_1d(dim),
                BufferArg::from_raw_parts(self.work.clone(), elements),
                BufferArg::from_raw_parts(self.kernel_x.clone(), elements),
                BufferArg::from_raw_parts(self.spec_x.clone(), elements),
                cells as u32,
            );
            fft::complex_mul_kernel::launch_unchecked(
                client,
                CubeCount::Static(cubes, 1, 1),
                CubeDim::new_1d(dim),
                BufferArg::from_raw_parts(self.work.clone(), elements),
                BufferArg::from_raw_parts(self.kernel_y.clone(), elements),
                BufferArg::from_raw_parts(self.spec_y.clone(), elements),
                cells as u32,
            );
        }

        for spec in [&self.spec_x, &self.spec_y] {
            self.transform(client, spec, elements, height, width, width, 1, -1.0);
            self.transform(client, spec, elements, width, height, 1, width, -1.0);
        }

        let (cubes, dim) = launch_grid(force_cells);
        for (spec, force) in [
            (self.spec_x.clone(), bufs.gpu_grav_fx.clone()),
            (self.spec_y.clone(), bufs.gpu_grav_fy.clone()),
        ] {
            unsafe {
                fft::extract_force_kernel::launch_unchecked(
                    client,
                    CubeCount::Static(cubes, 1, 1),
                    CubeDim::new_1d(dim),
                    BufferArg::from_raw_parts(spec, elements),
                    BufferArg::from_raw_parts(bufs.gpu_grav_mask.clone(), force_cells),
                    BufferArg::from_raw_parts(force, force_cells),
                    width,
                    xcells,
                    ycells,
                );
            }
        }
    }

    /// One 1-D transform over `rows` sequences of length `len`.
    ///
    /// Sequence `r` lives at `r * inner + i * stride`; `sign` is `+1` for the
    /// forward transform and `-1` for the inverse.
    #[allow(clippy::too_many_arguments)]
    fn transform(
        &self,
        client: &Client,
        data: &Handle,
        elements: usize,
        rows: u32,
        len: u32,
        inner: u32,
        stride: u32,
        sign: f32,
    ) {
        let (cubes, dim) = launch_grid((rows * len) as usize);
        unsafe {
            fft::fft_bitrev_kernel::launch_unchecked(
                client,
                CubeCount::Static(cubes, 1, 1),
                CubeDim::new_1d(dim),
                BufferArg::from_raw_parts(data.clone(), elements),
                rows,
                len,
                inner,
                stride,
                fft::bit_count(len),
            );
        }

        let tw_len = 1usize << (self.tw_bits - 1);
        let mut half = 1u32;
        while half < len {
            let (cubes, dim) = launch_grid((rows * (len / 2)) as usize);
            let shift = self.tw_bits - fft::bit_count(half * 2);
            unsafe {
                fft::fft_stage_kernel::launch_unchecked(
                    client,
                    CubeCount::Static(cubes, 1, 1),
                    CubeDim::new_1d(dim),
                    BufferArg::from_raw_parts(data.clone(), elements),
                    rows,
                    len,
                    half,
                    inner,
                    stride,
                    BufferArg::from_raw_parts(self.tw_re.clone(), tw_len),
                    BufferArg::from_raw_parts(self.tw_im.clone(), tw_len),
                    shift,
                    sign,
                );
            }
            half *= 2;
        }
    }
}

/// Cube count and cube size for `units` independent work items.
fn launch_grid(units: usize) -> (u32, u32) {
    let cubes = units.div_ceil(CUBE_DIM as usize).max(1) as u32;
    (cubes, CUBE_DIM)
}
