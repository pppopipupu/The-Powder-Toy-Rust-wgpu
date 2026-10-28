//! Scattered particle uploads for brush strokes.
//!
//! A brush frame changes a few thousand slots that are scattered across the
//! pool. Uploading them as one sub-range per array run cost one `write_buffer`
//! per (run, array): a 100-pixel brush touched ~31 000 slots in ~250 runs, which
//! issued ~3600 writes and spent ~15 ms of a 16 ms upload. The number of calls,
//! not the bytes, was the cost.
//!
//! Here the host packs one record per dirty slot — the slot id, the map word and
//! the particle's 14 fields — into a single staging buffer and two small kernels
//! scatter it into the SoA. One upload call and two launches per brush frame,
//! and the device still receives exactly the host's values for exactly the
//! slots the host touched.

use cubecl::bytes::Bytes;
use cubecl::prelude::*;
use cubecl::server::Handle;

use super::{SimBuffers, WORD};

/// Words per patch record: a four-word header plus the 14 particle fields.
///
/// Kept a multiple of four so every record stays 16-byte aligned in the staging
/// buffer; the kernel mirrors the value as a literal (see `PATCH_STRIDE_GUARD`).
pub(super) const PATCH_STRIDE: usize = 20;

/// Record offsets. Header first, then the particle's words in SoA order.
const R_SLOT: usize = 0;
const R_PIXEL: usize = 1;
const R_PMAP: usize = 2;
const R_PHOTONS: usize = 3;
const R_TY: usize = 4;
const R_LIFE: usize = 5;
const R_CTYPE: usize = 6;
const R_X: usize = 7;
const R_Y: usize = 8;
const R_VX: usize = 9;
const R_VY: usize = 10;
const R_TEMP: usize = 11;
const R_FLAGS: usize = 12;
const R_TMP: usize = 13;
const R_TMP2: usize = 14;
const R_TMP3: usize = 15;
const R_TMP4: usize = 16;
const R_DCOLOUR: usize = 17;

/// Record pixel value of a slot whose position is outside the canvas.
const NO_PIXEL: u32 = u32::MAX;

/// Threads per patch kernel invocation.
const PATCH_THREADS: u32 = 256;

/// Smallest staging allocation, so a single painted pixel does not reallocate.
const PATCH_MIN_WORDS: usize = 2048;

/// Writes the integer fields, position and velocity of every record.
#[cube(launch_unchecked)]
#[allow(clippy::too_many_arguments)]
fn patch_body_kernel(
    recs: &[u32],
    part_ty: &mut [u32],
    part_life: &mut [u32],
    part_ctype: &mut [u32],
    part_x: &mut [f32],
    part_y: &mut [f32],
    part_vx: &mut [f32],
    part_vy: &mut [f32],
    part_temp: &mut [f32],
    count: u32,
) {
    let index = ABSOLUTE_POS;
    if index < count as usize {
        let base = index * 20usize;
        let slot = recs[base] as usize;
        part_ty[slot] = recs[base + 4usize];
        part_life[slot] = recs[base + 5usize];
        part_ctype[slot] = recs[base + 6usize];
        part_x[slot] = f32::reinterpret(recs[base + 7usize]);
        part_y[slot] = f32::reinterpret(recs[base + 8usize]);
        part_vx[slot] = f32::reinterpret(recs[base + 9usize]);
        part_vy[slot] = f32::reinterpret(recs[base + 10usize]);
        part_temp[slot] = f32::reinterpret(recs[base + 11usize]);
    }
}

/// Writes the scratch words, the decoration colour and the pixel-map words.
#[cube(launch_unchecked)]
#[allow(clippy::too_many_arguments)]
fn patch_tail_kernel(
    recs: &[u32],
    part_flags: &mut [u32],
    part_tmp: &mut [u32],
    part_tmp2: &mut [u32],
    part_tmp3: &mut [u32],
    part_tmp4: &mut [u32],
    part_dcolour: &mut [u32],
    pmap: &mut [u32],
    photons: &mut [u32],
    count: u32,
) {
    let index = ABSOLUTE_POS;
    if index < count as usize {
        let base = index * 20usize;
        let slot = recs[base] as usize;
        part_flags[slot] = recs[base + 12usize];
        part_tmp[slot] = recs[base + 13usize];
        part_tmp2[slot] = recs[base + 14usize];
        part_tmp3[slot] = recs[base + 15usize];
        part_tmp4[slot] = recs[base + 16usize];
        part_dcolour[slot] = recs[base + 17usize];
        let pixel = recs[base + 1usize] as usize;
        if pixel != 0xFFFF_FFFFusize {
            pmap[pixel] = recs[base + 2usize];
            photons[pixel] = recs[base + 3usize];
        }
    }
}

// The kernels index the record with literals (a `#[cube]` body cannot read a
// host constant); these guards keep the two layouts tied.
const _: () = assert!(PATCH_STRIDE == 20);
const _: () = assert!(R_SLOT == 0 && R_PIXEL == 1 && R_PMAP == 2 && R_PHOTONS == 3);
const _: () = assert!(R_TY == 4 && R_LIFE == 5 && R_CTYPE == 6);
const _: () = assert!(R_X == 7 && R_Y == 8 && R_VX == 9 && R_VY == 10 && R_TEMP == 11);
const _: () = assert!(R_FLAGS == 12 && R_TMP == 13 && R_TMP2 == 14 && R_TMP3 == 15);
const _: () = assert!(R_TMP4 == 16 && R_DCOLOUR == 17 && R_DCOLOUR < PATCH_STRIDE);

impl SimBuffers {
    /// Uploads the SoA fields and map words of `slots` through a staging buffer.
    ///
    /// `slots` must be sorted, deduplicated and free of [`FULL_PARTICLE_UPLOAD`];
    /// the caller falls back to a whole-array upload otherwise. Records whose
    /// position lies outside the canvas skip the map write, exactly like the
    /// per-pixel path did.
    pub(super) fn upload_particle_patch(&mut self, slots: &[u32]) {
        let xres = self.config.xres as i32;
        let yres = self.config.yres as i32;
        let stride = self.config.xres as usize;
        let mut recs = Vec::with_capacity(slots.len() * PATCH_STRIDE);
        for &slot in slots {
            let id = slot as usize;
            if id >= self.capacity {
                continue;
            }
            let x = self.part_x[id].round() as i32;
            let y = self.part_y[id].round() as i32;
            let in_canvas = x >= 0 && y >= 0 && x < xres && y < yres;
            let pixel = if in_canvas {
                y as usize * stride + x as usize
            } else {
                0
            };
            let mut rec = [0u32; PATCH_STRIDE];
            rec[R_SLOT] = slot;
            rec[R_PIXEL] = if in_canvas { pixel as u32 } else { NO_PIXEL };
            if in_canvas {
                rec[R_PMAP] = self.pmap[pixel];
                rec[R_PHOTONS] = self.photons[pixel];
            }
            rec[R_TY] = self.part_ty[id];
            rec[R_LIFE] = self.part_life[id] as u32;
            rec[R_CTYPE] = self.part_ctype[id];
            rec[R_X] = self.part_x[id].to_bits();
            rec[R_Y] = self.part_y[id].to_bits();
            rec[R_VX] = self.part_vx[id].to_bits();
            rec[R_VY] = self.part_vy[id].to_bits();
            rec[R_TEMP] = self.part_temp[id].to_bits();
            rec[R_FLAGS] = self.part_flags[id];
            rec[R_TMP] = self.part_tmp[id] as u32;
            rec[R_TMP2] = self.part_tmp2[id] as u32;
            rec[R_TMP3] = self.part_tmp3[id] as u32;
            rec[R_TMP4] = self.part_tmp4[id] as u32;
            rec[R_DCOLOUR] = self.part_dcolour[id];
            // The last two words stay zero; the kernels never read them.
            recs.extend_from_slice(&rec);
        }
        let count = recs.len() / PATCH_STRIDE;
        if count == 0 {
            return;
        }
        let words = recs.len();
        let handle = self.patch_scratch(words);
        self.client.write(&handle, Bytes::from_elems(recs));

        let threads = PATCH_THREADS;
        let cubes = (count as u32).div_ceil(threads);
        let count = count as u32;
        unsafe {
            patch_body_kernel::launch_unchecked(
                &self.client,
                CubeCount::Static(cubes, 1, 1),
                CubeDim::new_1d(threads),
                BufferArg::from_raw_parts(handle.clone(), words),
                BufferArg::from_raw_parts(self.gpu_part_ty.clone(), self.capacity),
                BufferArg::from_raw_parts(self.gpu_part_life.clone(), self.capacity),
                BufferArg::from_raw_parts(self.gpu_part_ctype.clone(), self.capacity),
                BufferArg::from_raw_parts(self.gpu_part_x.clone(), self.capacity),
                BufferArg::from_raw_parts(self.gpu_part_y.clone(), self.capacity),
                BufferArg::from_raw_parts(self.gpu_part_vx.clone(), self.capacity),
                BufferArg::from_raw_parts(self.gpu_part_vy.clone(), self.capacity),
                BufferArg::from_raw_parts(self.gpu_part_temp.clone(), self.capacity),
                count,
            );
            patch_tail_kernel::launch_unchecked(
                &self.client,
                CubeCount::Static(cubes, 1, 1),
                CubeDim::new_1d(threads),
                BufferArg::from_raw_parts(handle, words),
                BufferArg::from_raw_parts(self.gpu_part_flags.clone(), self.capacity),
                BufferArg::from_raw_parts(self.gpu_part_tmp.clone(), self.capacity),
                BufferArg::from_raw_parts(self.gpu_part_tmp2.clone(), self.capacity),
                BufferArg::from_raw_parts(self.gpu_part_tmp3.clone(), self.capacity),
                BufferArg::from_raw_parts(self.gpu_part_tmp4.clone(), self.capacity),
                BufferArg::from_raw_parts(self.gpu_part_dcolour.clone(), self.capacity),
                BufferArg::from_raw_parts(self.gpu_pmap.clone(), self.pmap.len()),
                BufferArg::from_raw_parts(self.gpu_photons.clone(), self.photons.len()),
                count,
            );
        }
    }

    /// Returns the staging buffer, growing it (never shrinking) to hold `words`.
    fn patch_scratch(&mut self, words: usize) -> Handle {
        if self.patch_words < words {
            let want = words.next_power_of_two().max(PATCH_MIN_WORDS);
            self.patch_buf = Some(self.client.empty(want * WORD));
            self.patch_words = want;
        }
        self.patch_buf
            .as_ref()
            .expect("patch staging buffer allocated above")
            .clone()
    }
}
