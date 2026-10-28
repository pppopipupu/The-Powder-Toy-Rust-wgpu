//! Device-resident parallel water equalisation (TPT `flood_water` GPU port).
//!
//! # Background & Physics
//!
//! In discrete cellular automata, gravity moves liquids downwards (+y) or diagonally/laterally.
//! Liquids cannot naturally flow upwards (-y) against gravity. In communicating vessels such as
//! U-tubes, water in one arm creates hydrostatic pressure that in reality pushes liquid up the
//! other arm. TPT simulates this communicating-vessel equalisation via `Simulation::flood_water`:
//! eligible liquid particles have a stochastic (1-in-200) chance per frame to search their
//! connected liquid body and, if that body has an open surface at a lower row (greater y),
//! teleport to that surface.
//!
//! # Parallel GPU Formulation
//!
//! The original host implementation ran a serial BFS/DFS flood fill per particle over host shadows,
//! requiring costly per-frame CPU-GPU readbacks and uploads. A naive full-canvas jump-flooding or
//! iterative label propagation on GPU would either require ~10+ ping-pong dispatches per frame
//! (exceeding dispatch budgets) or risk jumping across walls.
//!
//! We formulate a deterministic, 4-pass device-resident pipeline:
//!
//! 1. **Column Surface Analysis ([`column::water_column_analyze_kernel`])**:
//!    One thread per column `x in 0..xres`. Scans vertically down column `x` to identify liquid
//!    columns, their topmost row `top_y`, bottommost row `bot_y`, and whether the pixel directly
//!    above `top_y` is empty and unblocked by walls. Clears the column target table.
//! 2. **Horizontal Span Connectivity & Target Assignment ([`row::water_row_connect_kernel`])**:
//!    One thread per row `y in 0..yres`. Scans horizontally to find contiguous spans of liquid
//!    across row `y`. For each connected liquid span, evaluates the surfaces of all columns touching
//!    the span: identifies the highest surface (`min_top_y`) and lowest free surface (`max_free_dest_y`).
//!    If `max_free_dest_y > min_top_y`, columns belonging to the higher surface are assigned
//!    destination targets distributed across the receiving columns.
//! 3. **Candidate Intent & Arbitration ([`intent::water_intent_kernel`])**:
//!    One thread per particle slot `id in 0..num_slots`. Gated by `Falldown == 2`, non-rigid,
//!    `!FLAG_SKIPMOVE`, and the deterministic 1-in-200 frame seed hash. Eligible particles whose
//!    current row lies strictly above their column's assigned destination row submit move intents
//!    and compete for target pixels via `Atomic<u32>::fetch_min(id + 1)`.
//! 4. **Commit Stage ([`commit::water_commit_kernel`])**:
//!    One thread per particle slot `id in 0..num_slots`. Winning particles (`winner == id + 1`)
//!    atomically update `part_x`, `part_y`, clear the old position in `pmap`, and claim the new
//!    pixel in `pmap`.
//!
//! # Properties
//!
//! - **Dispatch count**: Exactly 4 dispatches per frame.
//! - **Storage buffers**: Every kernel binds <= 8 storage buffers (well within WebGPU/device limits).
//! - **Host overhead**: Zero per-frame readbacks and zero host loops.
//! - **Obstacle safety**: Only contiguous liquid spans connect; particle walls (`PT_WALL`, etc.)
//!   and coarse walls (`WL_*`) never allow teleportation across boundaries.

pub mod column;
pub mod commit;
pub mod intent;
pub mod row;

use cubecl::prelude::*;
use tpt_core::constants::PT_NUM;

use crate::buffers::SimBuffers;

pub use column::water_column_analyze_kernel;
pub use commit::water_commit_kernel;
pub use intent::water_intent_kernel;
pub use row::water_row_connect_kernel;

const BLOCK: u32 = 256;

/// Runs one device-resident water equalisation step if enabled.
pub fn step_water_equal_gpu(bufs: &mut SimBuffers, frame_seed: u32) {
    if !bufs.config.water_equal_test {
        return;
    }
    let xres = bufs.config.xres;
    let yres = bufs.config.yres;
    let xcells = bufs.config.xcells();
    let num_pixels = bufs.config.num_pixels();
    let num_cells = bufs.config.num_cells();
    let num_slots = bufs.next_slot.min(bufs.capacity) as u32;

    if xres == 0 || yres == 0 || num_slots == 0 {
        return;
    }

    let client = bufs.client.clone();

    unsafe {
        // Pass 1: Column Analysis (1 thread per column)
        water_column_analyze_kernel::launch_unchecked(
            &client,
            CubeCount::new_1d(xres.div_ceil(BLOCK)),
            CubeDim::new_1d(BLOCK),
            BufferArg::from_raw_parts(bufs.gpu_water_col_surf.clone(), num_pixels),
            BufferArg::from_raw_parts(bufs.gpu_water_col_target.clone(), num_pixels),
            BufferArg::from_raw_parts(bufs.gpu_pmap.clone(), num_pixels),
            BufferArg::from_raw_parts(bufs.gpu_wall_bmap.clone(), num_cells),
            BufferArg::from_raw_parts(bufs.gpu_wall_emap.clone(), num_cells),
            BufferArg::from_raw_parts(bufs.gpu_elem_props.clone(), PT_NUM * 32),
            xres,
            yres,
            xcells,
        );

        // Pass 2: Row Scan & Horizontal Connectivity (1 thread per row)
        water_row_connect_kernel::launch_unchecked(
            &client,
            CubeCount::new_1d(yres.div_ceil(BLOCK)),
            CubeDim::new_1d(BLOCK),
            BufferArg::from_raw_parts(bufs.gpu_water_col_surf.clone(), num_pixels),
            BufferArg::from_raw_parts(bufs.gpu_pmap.clone(), num_pixels),
            BufferArg::from_raw_parts(bufs.gpu_wall_bmap.clone(), num_cells),
            BufferArg::from_raw_parts(bufs.gpu_wall_emap.clone(), num_cells),
            BufferArg::from_raw_parts(bufs.gpu_elem_props.clone(), PT_NUM * 32),
            BufferArg::from_raw_parts(bufs.gpu_water_col_target.clone(), num_pixels),
            BufferArg::from_raw_parts(bufs.gpu_water_winner.clone(), num_pixels),
            xres,
            yres,
            xcells,
            frame_seed,
        );

        // Pass 3: Candidate Move Intent & Arbitration (1 thread per particle slot)
        water_intent_kernel::launch_unchecked(
            &client,
            CubeCount::new_1d(num_slots.div_ceil(BLOCK)),
            CubeDim::new_1d(BLOCK),
            BufferArg::from_raw_parts(bufs.gpu_part_ty.clone(), bufs.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_x.clone(), bufs.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_y.clone(), bufs.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_flags.clone(), bufs.capacity),
            BufferArg::from_raw_parts(bufs.gpu_pmap.clone(), num_pixels),
            BufferArg::from_raw_parts(bufs.gpu_elem_props.clone(), PT_NUM * 32),
            BufferArg::from_raw_parts(bufs.gpu_wall_bmap.clone(), num_cells),
            BufferArg::from_raw_parts(bufs.gpu_wall_emap.clone(), num_cells),
            BufferArg::from_raw_parts(
                bufs.gpu_move_lut.clone(),
                crate::move_kernel::build_move_lut_len(),
            ),
            BufferArg::from_raw_parts(bufs.gpu_water_col_target.clone(), num_pixels),
            BufferArg::from_raw_parts(bufs.gpu_water_winner.clone(), num_pixels),
            num_slots,
            xres,
            yres,
            xcells,
            frame_seed,
        );

        // Pass 4: Commit Moves (1 thread per particle slot)
        water_commit_kernel::launch_unchecked(
            &client,
            CubeCount::new_1d(num_slots.div_ceil(BLOCK)),
            CubeDim::new_1d(BLOCK),
            BufferArg::from_raw_parts(bufs.gpu_part_ty.clone(), bufs.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_x.clone(), bufs.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_y.clone(), bufs.capacity),
            BufferArg::from_raw_parts(bufs.gpu_pmap.clone(), num_pixels),
            BufferArg::from_raw_parts(bufs.gpu_water_col_target.clone(), num_pixels),
            BufferArg::from_raw_parts(bufs.gpu_water_winner.clone(), num_pixels),
            num_slots,
            xres,
            yres,
        );
    }

    bufs.gpu_dirty = true;
    bufs.gpu_particles_dirty = true;
    bufs.gpu_maps_dirty = true;
}
