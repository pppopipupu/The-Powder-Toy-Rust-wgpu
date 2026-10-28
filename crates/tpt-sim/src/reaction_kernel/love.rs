//! `LOVE` / `LOLZ` — the 9x9 rule-table blocks (`Simulation.cpp:3792-3857`).
//!
//! `LOVE.cpp:45-58` and `LOLZ.cpp:46-58` carry a 9x9 cell table each. The
//! `BeforeSim` block ports as two things: a marking scan that kills the border
//! cells and records which 9x9 blocks hold the element, then a per-block
//! re-stamp. An empty cell whose table entry is `1` becomes the element; a cell
//! of that element whose entry is `0` is killed. Cells of other elements are
//! never touched and a `1` cell that already holds the element survives.
//!
//! # TPT's scan order
//!
//! The reference walk is
//!
//! ```text
//! for (nx = 9; nx <= XRES-18; nx++)
//!     for (ny = 9; ny <= YRES-7; ny++)
//! ```
//!
//! with the marker read as `Element_LOVE_love[nx/9][ny/9]` and cleared at the
//! bottom of the body (`Simulation.cpp:3817-3857`). The marker is cleared on
//! the *first* pixel of a block the walk reaches, which is the block's own
//! origin `(9*bx, 9*by)`; the step of `1` therefore never reaches a second
//! pixel of that block. The walk is exactly "evaluate every marked block once,
//! at its origin", in ascending `(bx, by)` order, just as if the loops stepped
//! by `9`. The 9x9 windows of two blocks are disjoint, so the port can give one
//! block to one thread with no ordering constraint. The two rule phases of one
//! block stay in that thread because the `LOLZ` phase reads the pixels the
//! `LOVE` phase changed, exactly as the sequential reference does.
//!
//! The marking scan is a separate phase in the reference
//! (`Simulation.cpp:3796-3815`) and runs completely before any rule phase, but
//! it only reads `pmap` and kills border cells; a rule phase can only observe a
//! border kill inside its own block (the window is the block), so folding the
//! marking scan into the block thread is exact.
//!
//! # `parts[].tmp`
//!
//! The reference pass keeps no state on the particles: its only bookkeeping is
//! the two block-marker arrays `Element_LOVE_love` / `Element_LOLZ_lolz`
//! (`Simulation.h:148-149`), reset by the `memset` at
//! `Simulation.cpp:1046-1047` and recomputed from `pmap` every frame. The port
//! carries no `tmp` counter either: the marker is a local of the block thread,
//! and a created cell takes the element's `DefaultProperties` (`tmp = 0`) from
//! the shared creation queue, exactly as `create_part` copies the whole default
//! row (`Simulation.cpp:1942-1945`).
//!
//! # The `elementCount` gate
//!
//! `Simulation.cpp:3793` skips the whole block when neither element exists
//! (`elementCount[PT_LOVE] <= 0 && elementCount[PT_LOLZ] <= 0`). The port keeps
//! that gate on the device: [`love_gate_scan_kernel`] reduces `part_ty` over the
//! used slots into one word of the otherwise unused `gpu_elem_phys` allocation
//! ([`LOVE_GATE_WORD`]), and [`love_kernel`] terminates every block thread
//! before touching `pmap` when the word is zero. The extra scan is one
//! `part_ty` load per used slot — the same range every slot-parallel reaction
//! pass already visits — and replaces the 81 `pmap` loads per block the absent
//! case used to cost.
//!
//! # Deviations
//!
//! * `kill_part` also clears the pixel's `pmap` word (`Simulation.cpp:1754-1775`);
//!   the port only writes `part_ty`, the shared death convention. A cell that is
//!   killed and re-created in the same block (a border cell, or a `LOVE` that
//!   the `LOLZ` phase replaces) is queued with [`CR_NO_OCCUPANCY`] so the drain
//!   overwrites the stale map word with the new slot, which is what
//!   `kill_part` + `create_part` produce. For the same reason a pixel another
//!   pass emptied earlier in the same reaction step still looks occupied here,
//!   so a birth into it waits for the next frame's map rebuild; `LOVE` and
//!   `LOLZ` themselves are never touched by another pass, so the map type this
//!   pass tests is always their current type.
//! * TPT runs this block in `BeforeSim` before the element updates; the port
//!   runs it inside the reaction step after the thermal pass. Neither element
//!   has an `Update` and neither has a phase transition, so the ordering cannot
//!   change the result.
//! * The rule phases resolve one cell at a time from that cell's pre-pass map
//!   word and the two marker flags, replaying the reference's `LOVE`-then-`LOLZ`
//!   order. A queued birth is never read back, so no staged buffer is needed and
//!   no cell is visited twice.

// Clippy's rewrite for these lints uses `Range::contains` / `clamp` / let-chains,
// which the device IR does not have, so the plain comparisons and nested `if`s
// are deliberate.
#![allow(clippy::manual_range_contains)]
#![allow(clippy::collapsible_if)]

use cubecl::prelude::*;
use tpt_core::constants::*;

use super::PassDims;
use super::shared::{CR_NO_OCCUPANCY, CREATE_QUEUE_WORDS, launch_grid, request_create};
use crate::buffers::SimBuffers;

/// Word of the unused `gpu_elem_phys` allocation that carries the
/// `elementCount` gate (`1` = the pool holds a `LOVE` or `LOLZ`).
const LOVE_GATE_WORD: usize = 0;

/// Low 9 bits of a packed `pmap` word hold the element type.
const PMAP_TYPE_MASK: u32 = 0x1FF;
const _: () = assert!(PMAP_TYPE_MASK == 0x1FF);

/// `PT_LOVE` (`Simulation.cpp:3807`), mirrored for the kernel body.
const PT_LOVE_U32: u32 = PT_LOVE;
const _: () = assert!(PT_LOVE_U32 == 110);

/// `PT_LOLZ` (`Simulation.cpp:3811`).
const PT_LOLZ_U32: u32 = PT_LOLZ;
const _: () = assert!(PT_LOLZ_U32 == 132);

/// `PT_NONE`, the free-slot marker the reaction passes kill with.
const PT_NONE_U32: u32 = PT_NONE;
const _: () = assert!(PT_NONE_U32 == 0);

/// The block edge: a rule cell is a `9x9` square (`LOVE.cpp:45`).
const BLOCK: u32 = 9;

/// `Element_LOVE_RuleTable[nnx][nny]` (`LOVE.cpp:45-58`).
///
/// The first index is the x offset and the second the y offset, exactly like
/// the C array, so the kernel's linear index is `nnx * 9 + nny`.
const LOVE_RULE_TABLE: [[u32; 9]; 9] = [
    [0, 0, 1, 1, 0, 0, 0, 0, 0],
    [0, 1, 0, 0, 1, 1, 0, 0, 0],
    [1, 0, 0, 0, 0, 0, 1, 0, 0],
    [1, 0, 0, 0, 0, 0, 0, 1, 0],
    [0, 1, 0, 0, 0, 0, 0, 0, 1],
    [1, 0, 0, 0, 0, 0, 0, 1, 0],
    [1, 0, 0, 0, 0, 0, 1, 0, 0],
    [0, 1, 0, 0, 1, 1, 0, 0, 0],
    [0, 0, 1, 1, 0, 0, 0, 0, 0],
];

/// `Element_LOLZ_RuleTable[nny][nnx]` (`LOLZ.cpp:46-58`).
///
/// The pass indexes this table transposed (`Simulation.cpp:3847`), so the
/// kernel's linear index for a cell at `(dx, dy)` is `dy * 9 + dx`.
const LOLZ_RULE_TABLE: [[u32; 9]; 9] = [
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [1, 0, 0, 0, 0, 0, 1, 0, 0],
    [1, 0, 0, 0, 0, 0, 1, 0, 0],
    [1, 0, 0, 1, 1, 0, 0, 1, 0],
    [1, 0, 1, 0, 0, 1, 0, 1, 0],
    [1, 0, 1, 0, 0, 1, 0, 1, 0],
    [0, 1, 0, 1, 1, 0, 0, 1, 0],
    [0, 1, 0, 0, 0, 0, 0, 1, 0],
    [0, 1, 0, 0, 0, 0, 0, 1, 0],
];

/// Packs a `9x9` table into three little-endian `u32` words, `index = x*9 + y`.
///
/// A `const fn` so the words never cost a runtime table walk; [`rule_bit`]
/// decodes one bit on the device.
const fn rule_words(table: &[[u32; 9]; 9]) -> [u32; 3] {
    let mut words = [0u32; 3];
    let mut x = 0usize;
    while x < 9 {
        let mut y = 0usize;
        while y < 9 {
            if table[x][y] != 0 {
                let bit = x * 9 + y;
                words[bit / 32] |= 1u32 << (bit % 32);
            }
            y += 1;
        }
        x += 1;
    }
    words
}

/// `Element_LOVE_RuleTable` as three bit words.
const LOVE_RULE_WORDS: [u32; 3] = rule_words(&LOVE_RULE_TABLE);
/// `Element_LOLZ_RuleTable` as three bit words (still indexed `x*9 + y`; the
/// caller applies the transposition).
const LOLZ_RULE_WORDS: [u32; 3] = rule_words(&LOLZ_RULE_TABLE);

/// Bit `index` of a packed rule table (`Simulation.cpp:3829`, `:3833`).
///
/// The word is selected with plain assignments from the three runtime seeds so
/// CubeCL never folds the local back into its initialiser constant.
#[cube]
fn rule_bit(w0: u32, w1: u32, w2: u32, index: u32) -> u32 {
    let word = index / 32u32;
    let mut w = w0;
    if word == 1u32 {
        w = w1;
    }
    if word == 2u32 {
        w = w2;
    }
    (w >> (index % 32u32)) & 1u32
}

/// The marking scan for one block: border kills plus the two block markers
/// (`Simulation.cpp:3796-3815`).
///
/// Returns bit 0 set for `LOVE` and bit 1 set for `LOLZ`, i.e. the two
/// `Element_*_...` marker slots the reference keeps for the block. The border
/// test is the reference's `(ny<9||nx<9||ny>YRES-7||nx>XRES-10)` rewritten as
/// `y+7 > YRES || x+10 > XRES` so no unsigned subtraction can underflow on a
/// canvas smaller than the border.
#[cube]
#[allow(clippy::too_many_arguments)]
fn mark_block(
    pmap: &[u32],
    part_ty: &mut [u32],
    x0: u32,
    y0: u32,
    scan_x: u32,
    scan_y: u32,
    xres: u32,
    yres: u32,
    capacity: u32,
) -> u32 {
    let mut marks = 0u32;
    for dy in 0u32..9u32 {
        let y = y0 + dy;
        if y < scan_y {
            for dx in 0u32..9u32 {
                let x = x0 + dx;
                if x < scan_x {
                    let entry = pmap[(y * xres + x) as usize];
                    if entry != 0u32 {
                        let ty = entry & PMAP_TYPE_MASK;
                        let mapped = entry >> 9u32;
                        let border = y < 9u32 || x < 9u32 || y + 7u32 > yres || x + 10u32 > xres;
                        if border && (ty == PT_LOVE_U32 || ty == PT_LOLZ_U32) {
                            // `kill_part(ID(r))`: the map entry may be a slot
                            // the pool already recycled, so bound it.
                            if mapped >= 1u32 && mapped <= capacity {
                                part_ty[(mapped - 1u32) as usize] = PT_NONE_U32;
                            }
                        } else if ty == PT_LOVE_U32 {
                            marks += 1u32;
                        } else if ty == PT_LOLZ_U32 {
                            marks += 2u32;
                        }
                    }
                }
            }
        }
    }
    marks
}

/// Clears the [`LOVE_GATE_WORD`] before [`love_gate_scan_kernel`].
#[cube(launch_unchecked)]
fn love_gate_clear_kernel(gate: &mut [Atomic<u32>]) {
    if ABSOLUTE_POS == 0usize {
        gate[LOVE_GATE_WORD].store(0u32);
    }
}

/// Recomputes the `elementCount[PT_LOVE] > 0 || elementCount[PT_LOLZ] > 0`
/// gate (`Simulation.cpp:3793`) from the live pool.
#[cube(launch_unchecked)]
fn love_gate_scan_kernel(part_ty: &[u32], gate: &mut [Atomic<u32>], bound: u32) {
    let i = ABSOLUTE_POS as u32;
    if i < bound {
        let ty = part_ty[i as usize];
        if ty == PT_LOVE_U32 || ty == PT_LOLZ_U32 {
            gate[LOVE_GATE_WORD].fetch_or(1u32);
        }
    }
}

/// Re-stamps one 9x9 block (`Simulation.cpp:3821-3857`).
///
/// Every cell is resolved from its own pre-pass map word and the two markers;
/// the two phases are replayed in the reference order (`LOVE` then `LOLZ`)
/// without a staged buffer, because a block's phases can only observe their own
/// cell: births are queued, never read back, and each cell is visited once.
#[cube]
#[allow(clippy::too_many_arguments)]
fn eval_block(
    pmap: &[u32],
    part_ty: &mut [u32],
    create_queue: &mut [Atomic<u32>],
    wall_bmap: &[u32],
    x0: u32,
    y0: u32,
    love_mark: u32,
    lolz_mark: u32,
    xres: u32,
    yres: u32,
    xcells: u32,
    capacity: u32,
    love_w0: u32,
    love_w1: u32,
    love_w2: u32,
    lolz_w0: u32,
    lolz_w1: u32,
    lolz_w2: u32,
) {
    for dx in 0u32..9u32 {
        for dy in 0u32..9u32 {
            let x = x0 + dx;
            let y = y0 + dy;
            // `if (ny+nny>0&&ny+nny<YRES&&nx+nnx>=0&&nx+nnx<XRES)`
            // (`Simulation.cpp:3826`). The lower bounds are always true for a
            // block origin, but the upper ones are not: the last block row can
            // reach past `YRES` on a canvas whose height is not a multiple of 9.
            if y < yres && x < xres {
                let entry = pmap[(y * xres + x) as usize];
                let ty = entry & PMAP_TYPE_MASK;
                let mapped = entry >> 9u32;
                let border = y < 9u32 || x < 9u32 || y + 7u32 > yres || x + 10u32 > xres;

                // The marking scan covers `[0, XRES-4) x [0, YRES-4)`
                // (`Simulation.cpp:3796`). Only a cell it visits was
                // border-killed; the last block row can reach past that limit
                // (`y >= YRES-4`), and such a cell is still ruled on by the
                // reference, so it must not be treated as empty here.
                let scanned = x < xres - 4u32 && y < yres - 4u32;
                let killed_by_border =
                    scanned && border && (ty == PT_LOVE_U32 || ty == PT_LOLZ_U32);
                let empty_after_mark = entry == 0u32 || killed_by_border;
                let base_love = ty == PT_LOVE_U32 && !killed_by_border;
                let base_lolz = ty == PT_LOLZ_U32 && !killed_by_border;

                // LOVE phase, `Element_LOVE_RuleTable[nnx][nny]`.
                let love_bit = rule_bit(love_w0, love_w1, love_w2, dx * 9u32 + dy);
                let love_created = love_mark != 0u32 && empty_after_mark && love_bit != 0u32;
                let love_killed = love_mark != 0u32 && base_love && love_bit == 0u32;

                // State after the LOVE phase. A cell the LOVE phase filled
                // blocks the LOLZ phase even if the table would birth there.
                let after_empty = (empty_after_mark && !love_created) || love_killed;

                // LOLZ phase, `Element_LOLZ_RuleTable[nny][nnx]` — note the
                // transposed index (`Simulation.cpp:3847`).
                let lolz_bit = rule_bit(lolz_w0, lolz_w1, lolz_w2, dy * 9u32 + dx);
                let lolz_killed = lolz_mark != 0u32 && base_lolz && lolz_bit == 0u32;
                let lolz_created = lolz_mark != 0u32 && after_empty && lolz_bit != 0u32;

                if love_killed || lolz_killed {
                    // `kill_part(ID(rt))` only writes `type = PT_NONE` here;
                    // the next map rebuild clears the entry.
                    if mapped >= 1u32 && mapped <= capacity {
                        part_ty[(mapped - 1u32) as usize] = PT_NONE_U32;
                    }
                }
                // `create_part(-1, x, y, PT_LOVE/PT_LOLZ)` only refuses an
                // empty-pixel creation when the pixel sits in a wall
                // (`Simulation.cpp:1907-1910`: `bmap && eval_move(...) == 0`,
                // and `eval_move` reports 0 for an empty pixel under every wall
                // id these TYPE_SOLID elements cannot pass). The pixel is free
                // by construction (empty, or a cell this pass just killed), but
                // the device map still names the killed particle until the
                // drain runs, so the record opts out of the occupancy test.
                let wall_cell = ((y / 4u32) * xcells + (x / 4u32)) as usize;
                let walled = wall_bmap[wall_cell] != 0u32;
                if love_created && !walled {
                    request_create(
                        create_queue,
                        PT_LOVE_U32,
                        x as i32,
                        y as i32,
                        CR_NO_OCCUPANCY,
                        0i32,
                        0u32,
                        0.0f32,
                        0i32,
                        0i32,
                        0i32,
                        0i32,
                        0u32,
                        0u32,
                        0.0f32,
                        0.0f32,
                    );
                }
                if lolz_created && !walled {
                    request_create(
                        create_queue,
                        PT_LOLZ_U32,
                        x as i32,
                        y as i32,
                        CR_NO_OCCUPANCY,
                        0i32,
                        0u32,
                        0.0f32,
                        0i32,
                        0i32,
                        0i32,
                        0i32,
                        0u32,
                        0u32,
                        0.0f32,
                        0.0f32,
                    );
                }
            }
        }
    }
}

/// One thread per 9x9 block of the scanned region (`Simulation.cpp:3792-3857`).
#[cube(launch_unchecked)]
#[allow(clippy::too_many_arguments)]
pub(super) fn love_kernel(
    pmap: &[u32],
    part_ty: &mut [u32],
    create_queue: &mut [Atomic<u32>],
    wall_bmap: &[u32],
    gate: &mut [Atomic<u32>],
    xres: u32,
    yres: u32,
    xcells: u32,
    capacity: u32,
    bx_count: u32,
    blocks: u32,
    last_bx: u32,
    last_by: u32,
    love_w0: u32,
    love_w1: u32,
    love_w2: u32,
    lolz_w0: u32,
    lolz_w1: u32,
    lolz_w2: u32,
) {
    let block = ABSOLUTE_POS as u32;
    if block >= blocks {
        terminate!();
    }
    // `Simulation.cpp:3793`: the reference skips the whole block scan when
    // `elementCount[PT_LOVE] <= 0 && elementCount[PT_LOLZ] <= 0`. The gate word
    // is the device's per-frame reduction of `part_ty`; a zero makes every
    // block thread exit before it touches `pmap`. `fetch_max(0)` is the
    // load that keeps the binding `read_write` for the atomic (commit.rs uses
    // the same trick for `cell_winner`).
    let gate_state = gate[LOVE_GATE_WORD].fetch_max(0u32);
    if gate_state == 0u32 {
        terminate!();
    }
    let bx = block % bx_count;
    let by = block / bx_count;
    let x0 = bx * 9u32;
    let y0 = by * 9u32;
    let scan_x = xres - 4u32;
    let scan_y = yres - 4u32;

    let marks = mark_block(pmap, part_ty, x0, y0, scan_x, scan_y, xres, yres, capacity);

    // `for (nx=9; nx<=XRES-18; nx++)` / `for (ny=9; ny<=YRES-7; ny++)`: only
    // blocks whose origin is inside those bounds are ever evaluated, and only
    // when the block holds the element.
    if bx >= 1u32 && by >= 1u32 && bx <= last_bx && by <= last_by && marks != 0u32 {
        eval_block(
            pmap,
            part_ty,
            create_queue,
            wall_bmap,
            x0,
            y0,
            marks & 1u32,
            marks & 2u32,
            xres,
            yres,
            xcells,
            capacity,
            love_w0,
            love_w1,
            love_w2,
            lolz_w0,
            lolz_w1,
            lolz_w2,
        );
    }
}

/// Launches the `LOVE` / `LOLZ` rule block for one reaction step.
pub(super) fn launch(bufs: &SimBuffers, dims: &PassDims) {
    let xres = dims.xres;
    let yres = dims.yres;
    // The marking scan covers `[0, XRES-4) x [0, YRES-4)` (`Simulation.cpp:3796`).
    if xres < 4 || yres < 4 {
        return;
    }
    let scan_x = xres - 4;
    let scan_y = yres - 4;
    let bx_count = scan_x.div_ceil(BLOCK);
    let by_count = scan_y.div_ceil(BLOCK);
    let blocks = bx_count * by_count;
    if blocks == 0 {
        return;
    }
    // Row/column of the last block the rule loops reach:
    // `9*bx <= XRES-18`, `9*by <= YRES-7` (`Simulation.cpp:3817-3819`).
    let last_bx = xres.saturating_sub(18) / BLOCK;
    let last_by = yres.saturating_sub(7) / BLOCK;

    // `elementCount` gate (`Simulation.cpp:3793`): reduce the live pool into
    // one device word, then let every block thread early-out on it.
    let gate_len = crate::move_kernel::elem_phys_len();
    unsafe {
        love_gate_clear_kernel::launch_unchecked(
            &bufs.client,
            CubeCount::Static(1, 1, 1),
            CubeDim::new_1d(1),
            BufferArg::from_raw_parts(bufs.gpu_elem_phys.clone(), gate_len),
        );
    }
    let bound = dims.used_slots as u32;
    if bound > 0 {
        let (gate_cubes, gate_dim) = launch_grid(bound as usize);
        unsafe {
            love_gate_scan_kernel::launch_unchecked(
                &bufs.client,
                CubeCount::Static(gate_cubes, 1, 1),
                CubeDim::new_1d(gate_dim),
                BufferArg::from_raw_parts(bufs.gpu_part_ty.clone(), dims.capacity),
                BufferArg::from_raw_parts(bufs.gpu_elem_phys.clone(), gate_len),
                bound,
            );
        }
    }

    let (cubes, dim) = launch_grid(blocks as usize);
    unsafe {
        love_kernel::launch_unchecked(
            &bufs.client,
            CubeCount::Static(cubes, 1, 1),
            CubeDim::new_1d(dim),
            BufferArg::from_raw_parts(bufs.gpu_pmap.clone(), dims.pixels),
            BufferArg::from_raw_parts(bufs.gpu_part_ty.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_create_queue.clone(), CREATE_QUEUE_WORDS),
            BufferArg::from_raw_parts(bufs.gpu_wall_bmap.clone(), dims.cells),
            BufferArg::from_raw_parts(bufs.gpu_elem_phys.clone(), gate_len),
            dims.xres,
            dims.yres,
            dims.xcells,
            dims.capacity as u32,
            bx_count,
            blocks,
            last_bx,
            last_by,
            LOVE_RULE_WORDS[0],
            LOVE_RULE_WORDS[1],
            LOVE_RULE_WORDS[2],
            LOLZ_RULE_WORDS[0],
            LOLZ_RULE_WORDS[1],
            LOLZ_RULE_WORDS[2],
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Decodes the packed words back into the reference's cell layout.
    fn decoded(words: [u32; 3]) -> [[u32; 9]; 9] {
        let mut table = [[0u32; 9]; 9];
        for (x, row) in table.iter_mut().enumerate() {
            for (y, cell) in row.iter_mut().enumerate() {
                let index = x * 9 + y;
                *cell = (words[index / 32] >> (index % 32)) & 1;
            }
        }
        table
    }

    #[test]
    fn love_words_match_the_reference_table() {
        assert_eq!(decoded(LOVE_RULE_WORDS), LOVE_RULE_TABLE);
        // Spot checks against `LOVE.cpp:45-58`.
        assert_eq!(LOVE_RULE_TABLE[0], [0, 0, 1, 1, 0, 0, 0, 0, 0]);
        assert_eq!(LOVE_RULE_TABLE[4], [0, 1, 0, 0, 0, 0, 0, 0, 1]);
        assert_eq!(LOVE_RULE_TABLE[8], [0, 0, 1, 1, 0, 0, 0, 0, 0]);
    }

    #[test]
    fn lolz_words_match_the_reference_table() {
        assert_eq!(decoded(LOLZ_RULE_WORDS), LOLZ_RULE_TABLE);
        // Spot checks against `LOLZ.cpp:46-58`.
        assert_eq!(LOLZ_RULE_TABLE[0], [0, 0, 0, 0, 0, 0, 0, 0, 0]);
        assert_eq!(LOLZ_RULE_TABLE[3], [1, 0, 0, 1, 1, 0, 0, 1, 0]);
        assert_eq!(LOLZ_RULE_TABLE[8], [0, 1, 0, 0, 0, 0, 0, 1, 0]);
        // The transposed decode of the cell layout is what the kernel indexes:
        // `Element_LOLZ_RuleTable[nny][nnx]` for a cell at `(nnx, nny)`.
        let mut cells = [[0u32; 9]; 9];
        for dx in 0..9 {
            for dy in 0..9 {
                cells[dx][dy] = LOLZ_RULE_TABLE[dy][dx];
            }
        }
        assert_eq!(cells[0], [0, 1, 1, 1, 1, 1, 0, 0, 0]);
        assert_ne!(cells, LOLZ_RULE_TABLE, "the LOLZ table is not symmetric");
    }
}
