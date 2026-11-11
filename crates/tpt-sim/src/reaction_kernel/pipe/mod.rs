//! Pipe elements: `PIPE` (`SC_FORCE`) and `PPIP` (`SC_POWERED`).
//!
//! A pipe holds at most one particle of cargo and relays it along a chain of
//! pipes whose colour flags cycle red → blue → green → red. The cargo occupies
//! four particle fields, exactly as in TPT (`PIPE.cpp:421-448`):
//!
//! | field | cargo meaning | TPT name |
//! |-------|---------------|----------|
//! | `ctype` | element id | `ctype` |
//! | `tmp2` | `life` | `tmp2` |
//! | `tmp3` | `tmp` | `tmp3` |
//! | `tmp4` | `ctype` | `tmp4` |
//! | `temp` | the pipe's temperature, mixed with the cargo as it arrives | `temp` |
//!
//! `tmp` itself is the pipe's own state: `PFLAG_COLORS` (bits 18-19 — the same
//! bits `render_kernel::particles::pipe_graphics` reads for the pipe tint),
//! `PFLAG_INITIALIZING`, the single-pixel nibbles and, for `PPIP`, the trigger
//! and pause bits. The renderer contract is therefore unchanged: the cargo type
//! is in `ctype`, the pipe colour in `tmp`.
//!
//! # Organization
//!
//! The kernels are partitioned into three child modules:
//!
//! | module | work |
//! |--------|------|
//! | [`pump`] | [`pump::pipe_flood_kernel`], [`pump::pipe_cascade_kernel`], [`pump::pipe_brck_kernel`], [`pump::pipe_state_kernel`] — connectivity flood, colour cascade, pause marker on the `BRCK` border, state countdown/`BRCK` border |
//! | [`handshake`] | [`handshake::pipe_claim_kernel`] — transport and eat handshake decisions |
//! | [`transport`] | [`transport::pipe_copy_kernel`], [`transport::pipe_entrance_kernel`], [`transport::pipe_victim_kernel`], [`transport::pipe_finish_kernel`] — cargo transfer, pickup, particle death and release |
//!
//! # TPT references
//!
//! * `PIPE.cpp:11-52` / `PPIP.cpp:9-46` — element table: `TYPE_SOLID |
//!   PROP_LIFE_DEC`, `life` 60, `HeatConduct` 251, `PIPE` melts into `BRMT`
//!   above 10 pressure. Both are already in `tpt-core`'s table, and the life
//!   countdown is `life::life_and_local_reaction_kernel`'s `PROP_LIFE_DEC`.
//! * `PIPE.cpp:57-62`, `PPIP.cpp:49-56` — the `PPIP` trigger/pause/reverse bits.
//! * `PIPE.cpp:64-72` — `Element_PIPE_offsets`: the eight neighbour offsets in
//!   TPT's `rx`-outer / `ry`-inner scan order ([`pipe_offset`]).
//! * `PIPE.cpp:81-103` — `prevColor` / `nextColor`, the flow cycle
//!   ([`next_colour`], [`prev_colour`]).
//! * `PIPE.cpp:105-172` — trigger handling: `TRIGGER_ON` beats `TRIGGER_OFF`,
//!   `TRIGGER_REVERSE` toggles `REVERSED`, swaps red↔blue and swaps the two
//!   single-pixel nibbles. A trigger that flips the pause bit also writes the
//!   stasis marker into every `BRCK` within two pixels (`PIPE.cpp:126-145`):
//!   `tmp = 0` when the pipe ends up paused, `1` when it runs again, which is
//!   the additive glow `BRCK.cpp:48-59` draws.
//! * `PIPE.cpp:165-232` — colour cascade: a coloured pipe at `life == 3` seeds
//!   every `PFLAG_INITIALIZING` neighbour with `nextColor` and `life = 6`; a
//!   one-wide (`0x100`) source also seeds the `0x200`/nibble fixed-direction
//!   words (`PIPE.cpp:186-199`).
//! * `PIPE.cpp:238-278` — `BRCK` border while `life <= 10`, and
//!   `PFLAG_INITIALIZING` at `life <= 1`; a `PPIP` creates its border bricks
//!   glowing (`tmp = 1`, `PIPE.cpp:279-285`), a plain `PIPE` does not.
//! * `PIPE.cpp:280-335` — initialisation countdown: `life == 0` waits for free
//!   space and restarts at 50, `life == 5` marks a one-wide run, `life == 2`
//!   turns the pipe red.
//! * `PIPE.cpp:233-262` — end-of-run release (create the cargo in a free
//!   neighbour) and entrance eating (`transfer_part_to_pipe` + `kill_part`).
//! * `PIPE.cpp:541-629` — `pushParticle`: transfer to a neighbour of colour
//!   `nextColor` that has no cargo, up to two hops, and the single-pixel
//!   fixed-direction transfer (`coords = 7 - ((tmp >> 10) & 7)`).
//! * `PIPE.cpp:470-495` / `PIPE.cpp:497-539` — `transfer_part_to_pipe` and
//!   `transfer_pipe_to_pipe`: which cargo fields move and how the pipe's
//!   temperature is mixed (`PFLAG_CAN_CONDUCT`).
//! * `PPIP.cpp:64-150` — `Element_PPIP_flood_trigger`: the whole connected run
//!   of pipes toggles together. Seeded by a spark on a `PSCN` / `NSCN` / `INST`
//!   conductor (`SPRK.cpp:264-269`), by `INST` through the same call, or by
//!   `HEAC` next to a coloured pipe at `life == 3` (`PIPE.cpp:221-222`); the
//!   pending bits are folded into the active trigger bits before the pipe reads
//!   them (`Simulation.cpp:3880-3890`).
//! * `PIPE.cpp:244-263` — the end-of-run probe reads `pmap` and then `photons`,
//!   so a `TYPE_ENERGY` cargo (`SPRK`, `PHOT`, `NEUT`, `ELEC`) is released and
//!   eaten exactly like matter, and a `STOR` next to the probe hands its stored
//!   element over (`PIPE.cpp:259-263`).
//!
//! # Gather instead of push
//!
//! TPT mutates the *neighbour*: `pushParticle` writes the destination pipe's
//! cargo fields, clears the source's `ctype` and then recurses from the
//! destination, so one cargo can travel two pipes in a frame; the `nt` branch
//! of `PIPE_update` creates a particle in a random free neighbour and
//! `transfer_part_to_pipe` kills the particle it eats. A slot-parallel GPU
//! dispatch cannot do any of that: a thread may only write its own slot.
//!
//! Every kernel here therefore recomputes the same *edge* from both ends. An
//! edge `p → q` is live exactly when both of these read-only predicates agree:
//!
//! * sender choice: the lowest-slot neighbour of colour `nextColor(p)` that
//!   has no cargo (TPT scans three of the eight offsets; the deterministic lowest
//!   slot keeps the result independent of thread order).
//! * receiver choice: the lowest-slot neighbour of colour `prevColor(q)` that
//!   carries cargo and is not paused.
//!
//! The sender clears its own cargo only when the receiver's choice is itself,
//! and the receiver copies the cargo only when the sender's choice is itself;
//! the two tests are the same predicate, so they cannot disagree and the cargo
//! is neither duplicated nor lost. Both threads write only their own slot.
//!
//! The handshake cannot run in one dispatch: a receiver reading the cargo of a
//! sender that is clearing it in the same dispatch would see either value, and a
//! half-cleared cargo is a lost cargo. The pass is therefore staged, and each
//! dispatch only reads fields that no thread of that dispatch writes:
//!
//! * [`pump::pipe_flood_kernel`] — [`FLOOD_ROUNDS`] rounds of the connectivity
//!   flood, decided from the previous round's marks only;
//! * [`pump::pipe_cascade_kernel`] — colour cascade, staged in `tmp3`;
//! * [`pump::pipe_brck_kernel`] — the pause flip's stasis marker on the border
//!   bricks, gathered from the pre-trigger pipe state;
//! * [`pump::pipe_state_kernel`] — applies it and owns the pipe's own fields;
//! * [`handshake::pipe_claim_kernel`] — both handshakes decided from the frozen state and
//!   recorded in `part_flags` ([`PIPE_FLAG_PUSH`], [`PIPE_FLAG_RECV`] with the
//!   upstream direction, [`PIPE_FLAG_EAT`], [`PIPE_FLAG_VICTIM`],
//!   [`PIPE_FLAG_STOR`], [`PIPE_FLAG_RELEASE`]);
//! * [`transport::pipe_copy_kernel`] — the receiver copies; the sender is idle, so the
//!   fields it reads cannot change under it;
//! * [`transport::pipe_entrance_kernel`] — the pipe copies the cargo of the particle it
//!   eats, [`transport::pipe_victim_kernel`] — that particle dies;
//! * [`transport::pipe_finish_kernel`] — the sender clears the cargo it handed over, and a
//!   storer whose element was taken is emptied.
//!
//! Launched from [`crate::reaction_kernel::step_reactions_gpu`] after the force
//! pass and before wall conduction.

use cubecl::prelude::*;
use tpt_core::constants::*;

use super::PassDims;
use super::shared::{launch_grid, pmap_at};
use crate::buffers::SimBuffers;

pub(super) mod handshake;
pub(super) mod pump;
pub(super) mod transport;

use handshake::pipe_claim_kernel;
use pump::{pipe_brck_kernel, pipe_cascade_kernel, pipe_flood_kernel, pipe_state_kernel};
use transport::{pipe_copy_kernel, pipe_entrance_kernel, pipe_finish_kernel, pipe_victim_kernel};

// `tmp` layout (`PIPE.h:13-25`, `PIPE.cpp:57-62`)

/// `PFLAG_CAN_CONDUCT` (`PIPE.h:18`): mix the cargo's heat capacity.
pub(super) const PFLAG_CAN_CONDUCT: i32 = 0x0000_0001;
/// `0x100` (`PIPE.h:13`, `PIPE.cpp:220`): one-wide run marker.
pub(super) const PFLAG_SINGLE_PIXEL: i32 = 0x0000_0100;
/// `0x200` (`PIPE.h:14`, `PIPE.cpp:548`): transfer in a fixed direction.
pub(super) const PFLAG_SINGLE_FORWARD: i32 = 0x0000_0200;
/// `PFLAG_INITIALIZING` (`PIPE.h:21`): the colour has not been assigned yet.
pub(super) const PFLAG_INITIALIZING: i32 = 0x0002_0000;
/// `PFLAG_COLOR_RED` (`PIPE.h:22`).
pub(super) const PFLAG_COLOR_RED: i32 = 0x0004_0000;
/// `PFLAG_COLOR_GREEN` (`PIPE.h:23`).
pub(super) const PFLAG_COLOR_GREEN: i32 = 0x0008_0000;
/// `PFLAG_COLOR_BLUE` (`PIPE.h:24`).
pub(super) const PFLAG_COLOR_BLUE: i32 = 0x000C_0000;
/// `PFLAG_COLORS` (`PIPE.h:25`): the bits the renderer tints the pipe with.
pub(super) const PFLAG_COLORS: i32 = 0x000C_0000;
/// `PPIP_TMPFLAG_REVERSED` (`PIPE.cpp:57`).
pub(super) const PPIP_REVERSED: i32 = 0x0100_0000;
/// `PPIP_TMPFLAG_PAUSED` (`PIPE.cpp:58`).
pub(super) const PPIP_PAUSED: i32 = 0x0200_0000;
/// `PPIP_TMPFLAG_TRIGGER_REVERSE` (`PIPE.cpp:59`).
pub(super) const PPIP_TRIGGER_REVERSE: i32 = 0x0400_0000;
/// `PPIP_TMPFLAG_TRIGGER_OFF` (`PIPE.cpp:60`).
pub(super) const PPIP_TRIGGER_OFF: i32 = 0x0800_0000;
/// `PPIP_TMPFLAG_TRIGGER_ON` (`PIPE.cpp:61`).
pub(super) const PPIP_TRIGGER_ON: i32 = 0x1000_0000;
/// `PPIP_TMPFLAG_TRIGGERS` (`PIPE.cpp:62`).
pub(super) const PPIP_TRIGGERS: i32 = 0x1C00_0000;
/// Port-only: a `FLAG_SPAWN_CLONE` request is waiting for the host.
pub(super) const PIPE_TMP_RELEASE_PENDING: i32 = 0x0001_0000;

// Transient `part_flags` bits for the two handshakes.
pub(super) const PIPE_FLAG_PUSH: u32 = 0x0000_4000;
pub(super) const PIPE_FLAG_RECV: u32 = 0x0000_8000;
pub(super) const PIPE_FLAG_DIR_SHIFT: u32 = 16;
pub(super) const PIPE_FLAG_DIR_MASK: u32 = 0x0007_0000;
pub(super) const PIPE_FLAG_EAT: u32 = 0x0008_0000;
pub(super) const PIPE_FLAG_RELEASE: u32 = 0x0010_0000;
pub(super) const PIPE_FLAG_VICTIM: u32 = 0x0020_0000;
pub(super) const PIPE_FLAG_STOR: u32 = 0x0040_0000;
pub(super) const PIPE_FLAG_STOR_TAKEN: u32 = 0x0080_0000;

// Connectivity flood (`PPIP.cpp:64-150`).
pub(super) const PIPE_FLOOD_ON: u32 = 0x0100_0000;
pub(super) const PIPE_FLOOD_OFF: u32 = 0x0200_0000;
pub(super) const PIPE_FLOOD_REVERSE: u32 = 0x0400_0000;
pub(super) const PIPE_FLOOD_CONDUCT: u32 = 0x0800_0000;
pub(super) const PIPE_FLOOD_KIND: u32 = 0x0F00_0000;
pub(super) const PIPE_FLOOD_GEN_SHIFT: u32 = 28;
pub(super) const PIPE_FLOOD_GEN: u32 = 0x7000_0000;
pub(super) const PIPE_FLOOD_GEN_MOD: u32 = 8;
pub(super) const PIPE_FLOOD_APPLIED: u32 = 0x8000_0000;

pub(super) const FLOOD_ROUNDS: u32 = 8;

pub(super) const PMAP_TYPE: u32 = 0x1FF;
pub(super) const PT_NUM_U32: u32 = PT_NUM as u32;
const _: () = assert!(PT_NUM_U32 as usize == PT_NUM);

pub(super) const MOVABLE_MASK: u32 = TYPE_PART | TYPE_LIQUID | TYPE_GAS | TYPE_ENERGY;

pub(super) const NO_DIR: u32 = 8;
pub(super) const NO_SLOT: u32 = 0xFFFF_FFFF;

pub(super) const EXIT_NONE: u32 = 0;
pub(super) const EXIT_PIPE: u32 = 1;
pub(super) const EXIT_FREE: u32 = 2;

/// `PIPE.cpp:64-72`: the eight neighbour offsets in TPT's scan order.
#[cube]
pub(super) fn pipe_offset(k: u32) -> (i32, i32) {
    let mut dx = 1i32;
    if k < 3u32 {
        dx = -1i32;
    }
    if k == 3u32 || k == 4u32 {
        dx = 0i32;
    }
    let mut dy = (k as i32) - 6i32;
    if k < 3u32 {
        dy = (k as i32) - 1i32;
    }
    if k == 3u32 {
        dy = -1i32;
    }
    if k == 4u32 {
        dy = 1i32;
    }
    (dx, dy)
}

/// `nextColor` (`PIPE.cpp:93-103`): red → blue → green → red.
#[cube]
pub(super) fn next_colour(colour: i32) -> i32 {
    if colour == PFLAG_COLOR_RED {
        PFLAG_COLOR_BLUE
    } else if colour == PFLAG_COLOR_BLUE {
        PFLAG_COLOR_GREEN
    } else if colour == PFLAG_COLOR_GREEN {
        PFLAG_COLOR_RED
    } else {
        PFLAG_COLOR_GREEN
    }
}

/// `prevColor` (`PIPE.cpp:81-91`): red → green → blue → red.
#[cube]
pub(super) fn prev_colour(colour: i32) -> i32 {
    if colour == PFLAG_COLOR_RED {
        PFLAG_COLOR_GREEN
    } else if colour == PFLAG_COLOR_GREEN {
        PFLAG_COLOR_BLUE
    } else {
        PFLAG_COLOR_RED
    }
}

/// A sentinel in the cube value domain.
#[allow(clippy::erasing_op)]
#[cube]
pub(super) fn sentinel(seed: u32, value: u32) -> u32 {
    (seed & 0u32) | value
}

/// `PIPE.cpp:592`: `coords = 7 - ((tmp >> 10) & 7)` — the fixed transfer
/// direction of a `0x200` pipe, in [`pipe_offset`] order.
#[cube]
pub(super) fn single_pixel_exit_dir(tmp: i32) -> u32 {
    7u32 - (((tmp as u32) >> 10u32) & 7u32)
}

/// Packed word at `(x, y)` from the matter layer, falling back to the energy
/// layer, or `0` when both are empty.
#[cube]
pub(super) fn map_at_both(
    pmap: &[u32],
    photons: &[u32],
    xres: u32,
    yres: u32,
    x: i32,
    y: i32,
) -> u32 {
    let mut v = pmap_at(pmap, xres, yres, x, y);
    if v == 0u32 {
        v = pmap_at(photons, xres, yres, x, y);
    }
    v
}

/// `true` when one of the eight neighbours has no particle in either layer.
#[cube]
pub(super) fn has_free_neighbour_both(
    pmap: &[u32],
    photons: &[u32],
    xres: u32,
    yres: u32,
    x: i32,
    y: i32,
) -> bool {
    let mut free = false;
    for k in 0u32..8u32 {
        let (dx, dy) = pipe_offset(k);
        if map_at_both(pmap, photons, xres, yres, x + dx, y + dy) == 0u32 {
            free = true;
        }
    }
    free
}

/// Run the pipe pass for one frame.
pub(super) fn launch(bufs: &SimBuffers, dims: &PassDims) {
    if dims.pixels == 0 || dims.capacity == 0 || dims.elem_len == 0 {
        return;
    }
    if bufs.pmap.len() < dims.pixels
        || bufs.photons.len() < dims.pixels
        || bufs.part_ty.len() < dims.capacity
        || bufs.part_ctype.len() < dims.capacity
        || bufs.part_tmp.len() < dims.capacity
        || bufs.part_tmp2.len() < dims.capacity
        || bufs.part_tmp3.len() < dims.capacity
        || bufs.part_tmp4.len() < dims.capacity
        || bufs.part_life.len() < dims.capacity
        || bufs.part_temp.len() < dims.capacity
        || bufs.part_flags.len() < dims.capacity
    {
        return;
    }

    let (cubes, dim) = launch_grid(dims.pixels);
    let grid = CubeCount::Static(cubes, 1, 1);
    let cube_dim = CubeDim::new_1d(dim);
    let (slot_cubes, slot_dim) = launch_grid(dims.used_slots);
    let slot_grid = CubeCount::Static(slot_cubes, 1, 1);
    let slot_cube_dim = CubeDim::new_1d(slot_dim);

    for _ in 0..FLOOD_ROUNDS {
        unsafe {
            pipe_flood_kernel::launch_unchecked(
                &bufs.client,
                grid.clone(),
                cube_dim,
                BufferArg::from_raw_parts(bufs.gpu_pmap.clone(), dims.pixels),
                BufferArg::from_raw_parts(bufs.gpu_part_tmp.clone(), dims.capacity),
                BufferArg::from_raw_parts(bufs.gpu_part_life.clone(), dims.capacity),
                BufferArg::from_raw_parts(bufs.gpu_part_ctype.clone(), dims.capacity),
                BufferArg::from_raw_parts(bufs.gpu_part_flags.clone(), dims.capacity),
                dims.xres,
                dims.yres,
            );
        }
    }

    unsafe {
        pipe_cascade_kernel::launch_unchecked(
            &bufs.client,
            grid.clone(),
            cube_dim,
            BufferArg::from_raw_parts(bufs.gpu_pmap.clone(), dims.pixels),
            BufferArg::from_raw_parts(bufs.gpu_part_tmp.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_life.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_tmp3.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_flags.clone(), dims.capacity),
            dims.xres,
            dims.yres,
        );
    }

    // `PIPE.cpp:126-145`: a pause flip mirrors onto the border bricks. The pass
    // runs before the state half so it reads the pipe's pre-trigger `tmp` — the
    // state `pause_changed` is decided from — and it writes only the bricks' own
    // slots, which no other pass in this frame touches.
    unsafe {
        pipe_brck_kernel::launch_unchecked(
            &bufs.client,
            grid.clone(),
            cube_dim,
            BufferArg::from_raw_parts(bufs.gpu_pmap.clone(), dims.pixels),
            BufferArg::from_raw_parts(bufs.gpu_part_tmp.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_flags.clone(), dims.capacity),
            dims.xres,
            dims.yres,
        );
    }

    unsafe {
        pipe_state_kernel::launch_unchecked(
            &bufs.client,
            grid.clone(),
            cube_dim,
            BufferArg::from_raw_parts(bufs.gpu_pmap.clone(), dims.pixels),
            BufferArg::from_raw_parts(bufs.gpu_part_ctype.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_tmp.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_life.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_tmp2.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_tmp3.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_tmp4.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_flags.clone(), dims.capacity),
            dims.xres,
            dims.yres,
        );
    }

    unsafe {
        pipe_claim_kernel::launch_unchecked(
            &bufs.client,
            grid.clone(),
            cube_dim,
            BufferArg::from_raw_parts(bufs.gpu_pmap.clone(), dims.pixels),
            BufferArg::from_raw_parts(bufs.gpu_photons.clone(), dims.pixels),
            BufferArg::from_raw_parts(bufs.gpu_part_ctype.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_tmp.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_elem_props.clone(), dims.elem_len),
            BufferArg::from_raw_parts(bufs.gpu_part_flags.clone(), dims.capacity),
            dims.xres,
            dims.yres,
            dims.frame_seed,
        );
    }

    unsafe {
        pipe_copy_kernel::launch_unchecked(
            &bufs.client,
            grid.clone(),
            cube_dim,
            BufferArg::from_raw_parts(bufs.gpu_pmap.clone(), dims.pixels),
            BufferArg::from_raw_parts(bufs.gpu_part_ctype.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_tmp.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_tmp2.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_tmp3.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_tmp4.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_temp.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_flags.clone(), dims.capacity),
            dims.xres,
            dims.yres,
        );
    }

    // `photons` layer, so the pipe looks in both maps.
    unsafe {
        pipe_entrance_kernel::launch_unchecked(
            &bufs.client,
            grid.clone(),
            cube_dim,
            BufferArg::from_raw_parts(bufs.gpu_pmap.clone(), dims.pixels),
            BufferArg::from_raw_parts(bufs.gpu_photons.clone(), dims.pixels),
            BufferArg::from_raw_parts(bufs.gpu_part_ty.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_ctype.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_tmp.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_life.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_temp.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_tmp2.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_flags.clone(), dims.capacity),
            dims.xres,
            dims.yres,
            dims.capacity as u32,
            dims.frame_seed,
        );
    }

    // 5b. Victim death — 3 bindings, slot parallel.
    unsafe {
        pipe_victim_kernel::launch_unchecked(
            &bufs.client,
            slot_grid,
            slot_cube_dim,
            BufferArg::from_raw_parts(bufs.gpu_part_ty.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_flags.clone(), dims.capacity),
            dims.used_slots as u32,
        );
    }

    unsafe {
        pipe_finish_kernel::launch_unchecked(
            &bufs.client,
            grid,
            cube_dim,
            BufferArg::from_raw_parts(bufs.gpu_pmap.clone(), dims.pixels),
            BufferArg::from_raw_parts(bufs.gpu_part_ctype.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_tmp.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_tmp2.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_tmp3.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_tmp4.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_flags.clone(), dims.capacity),
            dims.xres,
            dims.yres,
        );
    }
}

#[cfg(test)]
pub(super) fn pack(id: u32, ty: u32) -> u32 {
    ((id + 1) << 9) | (ty & PMAP_TYPE)
}

#[cfg(test)]
mod tests {
    use super::transport::{pipe_eater_slot, pipe_eats, pipe_entrance_dir};
    use super::*;
    use crate::reaction_kernel::shared::{FLAG_SPAWN_CLONE, WORD_PROPERTIES};

    #[test]
    fn offsets_are_tpt_scan_order() {
        // `PIPE.cpp:64-72`.
        let want = [
            (-1, -1),
            (-1, 0),
            (-1, 1),
            (0, -1),
            (0, 1),
            (1, -1),
            (1, 0),
            (1, 1),
        ];
        for (k, expect) in want.iter().enumerate() {
            assert_eq!(pipe_offset(k as u32), *expect, "offset {k}");
        }
        // The transfer handshake relies on `offset(7 - k) == -offset(k)`: the
        // direction from the receiver back to the sender is the reverse index.
        for k in 0u32..8u32 {
            let (dx, dy) = pipe_offset(k);
            assert_eq!(pipe_offset(7u32 - k), (-dx, -dy));
        }
    }

    #[test]
    fn colour_cycle_matches_tpt() {
        // `PIPE.cpp:81-103`.
        assert_eq!(next_colour(PFLAG_COLOR_RED), PFLAG_COLOR_BLUE);
        assert_eq!(next_colour(PFLAG_COLOR_BLUE), PFLAG_COLOR_GREEN);
        assert_eq!(next_colour(PFLAG_COLOR_GREEN), PFLAG_COLOR_RED);
        assert_eq!(next_colour(0), PFLAG_COLOR_GREEN);

        assert_eq!(prev_colour(PFLAG_COLOR_RED), PFLAG_COLOR_GREEN);
        assert_eq!(prev_colour(PFLAG_COLOR_GREEN), PFLAG_COLOR_BLUE);
        assert_eq!(prev_colour(PFLAG_COLOR_BLUE), PFLAG_COLOR_RED);
        assert_eq!(prev_colour(0), PFLAG_COLOR_RED);

        // `prev` is the inverse of `next` on the cycle, which is what makes a
        // sender and its receiver agree on the same edge.
        for colour in [PFLAG_COLOR_RED, PFLAG_COLOR_GREEN, PFLAG_COLOR_BLUE] {
            assert_eq!(prev_colour(next_colour(colour)), colour);
            assert_eq!(next_colour(prev_colour(colour)), colour);
        }
    }

    #[test]
    fn cargo_fields_match_tpt_and_the_renderer() {
        // `PIPE.cpp:421-448`: the cargo type is `ctype`, its life `tmp2`, its
        // own `tmp` `tmp3` and its own `ctype` `tmp4`.
        //
        // The renderer reads the same two fields
        // (`render_kernel::particles::pipe_graphics`): the cargo colour comes
        // from `ctype` and the pipe tint from `tmp & PFLAG_COLORS`, so the
        // encoding above must not touch bits 18-19.
        assert_eq!(PFLAG_COLORS, 0x000C_0000);
        assert_eq!(PFLAG_COLOR_RED, 0x0004_0000);
        assert_eq!(PFLAG_COLOR_GREEN, 0x0008_0000);
        assert_eq!(PFLAG_COLOR_BLUE, 0x000C_0000);

        let pipe_bits = [
            PFLAG_CAN_CONDUCT,
            // `PFLAG_PARTICLE_DECO` (`PIPE.h:19`): unused here, `dcolour` is
            // not bound, but it must stay clear of the colour mask.
            0x0000_0002,
            PFLAG_SINGLE_PIXEL,
            PFLAG_SINGLE_FORWARD,
            PFLAG_INITIALIZING,
            PIPE_TMP_RELEASE_PENDING,
            PPIP_REVERSED,
            PPIP_PAUSED,
            PPIP_TRIGGER_ON,
            PPIP_TRIGGER_OFF,
            PPIP_TRIGGER_REVERSE,
        ];
        for bit in pipe_bits {
            assert_eq!(bit & PFLAG_COLORS, 0, "bit {bit:#x} tints the pipe");
        }
        // The port-only pending bit reuses TPT's `PFLAG_NORMALSPEED` position,
        // which TPT keeps in `parts[].flags`, never in `tmp` (`PIPE.h:20`).
        assert_eq!(PIPE_TMP_RELEASE_PENDING, 0x0001_0000);
        // The release is carried by `FLAG_SPAWN_CLONE`, which the host resolves
        // from `part_ctype`.
        assert_eq!(FLAG_SPAWN_CLONE, 0x0000_0100);
    }

    #[test]
    fn single_pixel_exit_decodes_the_tpt_nibble() {
        // `PIPE.cpp:592`: `coords = 7 - ((tmp >> 10) & 7)`.
        for d in 0u32..8u32 {
            let tmp = PFLAG_SINGLE_FORWARD | ((d as i32) << 10);
            assert_eq!(single_pixel_exit_dir(tmp), 7u32 - d);
        }
        // Bits 14-16 belong to the reverse nibble and must not leak in.
        let tmp = PFLAG_SINGLE_FORWARD | (2i32 << 10) | (5i32 << 14);
        assert_eq!(single_pixel_exit_dir(tmp), 5u32);
    }

    #[test]
    fn energy_cargo_comes_from_the_energy_layer() {
        // `PIPE.cpp:244-245` / `PIPE.cpp:253-254`: the probe reads `pmap` and
        // then `photons`, so a `PHOT` in the energy layer is a cargo.
        let xres = 8u32;
        let yres = 8u32;
        let mut pmap = vec![0u32; (xres * yres) as usize];
        let mut photons = vec![0u32; (xres * yres) as usize];
        pmap[4 * 8 + 3] = pack(1, PT_PIPE);
        photons[4 * 8 + 4] = pack(0, PT_PHOT);

        let ctype = vec![0u32; 2];
        let mut tmp = vec![0i32; 2];
        tmp[1] = PFLAG_COLOR_RED;
        let mut props = vec![0u32; 32 * 256];
        props[PT_PHOT as usize * 32 + WORD_PROPERTIES as usize] = TYPE_ENERGY;

        let seed = 5u32;
        assert_eq!(pipe_entrance_dir(1, seed), 6);
        assert!(pipe_eats(
            &pmap, &photons, &ctype, &tmp, &props, xres, yres, 3, 4, 1, tmp[1], seed
        ));
        // The photon's own slot names the same pipe, so exactly one of the two
        // sides takes the cargo and the other dies.
        assert_eq!(
            pipe_eater_slot(&pmap, &ctype, &tmp, xres, yres, 4, 4, seed),
            1
        );

        // Without the energy layer the pipe sees nothing to eat.
        let empty = vec![0u32; (xres * yres) as usize];
        assert!(!pipe_eats(
            &pmap, &empty, &ctype, &tmp, &props, xres, yres, 3, 4, 1, tmp[1], seed
        ));
        // `map_at_both` is the whole difference: it falls back to `photons`.
        assert_eq!(
            map_at_both(&pmap, &photons, xres, yres, 4, 4),
            pack(0, PT_PHOT)
        );
        assert_eq!(
            map_at_both(&pmap, &photons, xres, yres, 3, 4),
            pack(1, PT_PIPE)
        );
        assert_eq!(map_at_both(&pmap, &photons, xres, yres, 0, 0), 0);
    }
}
