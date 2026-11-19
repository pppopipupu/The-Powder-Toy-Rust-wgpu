//! Host side of the Game of Life pass: the scratch buffers, the generation
//! bookkeeping and the `GOLTool` brush.

use cubecl::client::Client;
use cubecl::prelude::*;
use cubecl::server::Handle;
use tpt_core::commands::SimCommand;
use tpt_core::constants::PT_LIFE;
use tpt_core::elements::ElementRegistry;

use crate::SimulationEngine;
use crate::buffers::SimBuffers;
use crate::reaction_kernel::dispatch::{SIM_SLOT_GOL_PIX, sim_grid};

use super::kernels::{gol_analyze_kernel, gol_birth_kernel, gol_update_kernel};
use super::rules::{BUILTIN_GOL, GOL_RULE_MASK, NGOL, gol_colours, gol_max_state};
use super::CUBE_DIM;

/// Words in `SimBuffers::gpu_create_queue` (`[count, record * GPU_CREATE_COUNT]`,
/// see `reaction_kernel::shared::request_create`).
const CREATE_QUEUE_WORDS: usize = SimBuffers::GPU_CREATE_COUNT * SimBuffers::CREATE_WORDS + 1;

/// Device scratch of the Game of Life pass for one canvas size.
#[derive(Clone)]
struct GolScratch {
    /// `BUILTIN_GOL[i].ruleset` for `i < NGOL`, then each entry's
    /// `(colour, colour2)` pair.
    rules: Handle,
    /// One packed analysis word per pixel.
    meta: Handle,
    /// Canvas size the buffers were sized for.
    pixels: usize,
}

/// Device scratch space and bookkeeping of the Game of Life pass.
#[derive(Default)]
pub struct GolState {
    scratch: Option<GolScratch>,
    /// Generations executed since the engine started.
    pub generations: u64,
    /// `CGOL`: frames elapsed since the last generation (`SimulateGoL` resets
    /// it at the top of every run, `Simulation.cpp:3446`). `GSPEED` frames per
    /// generation.
    pub cgol: u32,
}

impl GolState {
    /// Empty state; the device scratch is created by the first step.
    pub fn new() -> Self {
        Self::default()
    }

    /// Requests a generation on the next step.
    ///
    /// The palette calls this after painting `LIFE`, so a fresh cell evolves
    /// immediately instead of waiting for the next `GSPEED` window.
    pub fn wake(&mut self) {
        // `GSPEED` counts frames since the last generation; a fresh paint
        // starts that clock over so the first generation is not delayed.
        self.cgol = 0;
    }

    /// Drops the scratch space and the generation bookkeeping.
    ///
    /// Called when the canvas is resized or cleared: the device buffers belong
    /// to the old canvas and the next step allocates new ones.
    pub fn reset(&mut self) {
        self.scratch = None;
        self.cgol = 0;
    }

    /// Returns the scratch for a canvas with `pixels` pixels, rebuilding it
    /// when the canvas size changed.
    fn prepare(&mut self, client: &Client, pixels: usize) -> GolScratch {
        if let Some(scratch) = &self.scratch
            && scratch.pixels == pixels
        {
            return scratch.clone();
        }
        // The rulesets stay in the first `NGOL` words so the kernels can index
        // them by `ctype`; the colour pairs follow in the same order.
        let mut table: Vec<u32> = BUILTIN_GOL.iter().map(|rule| rule.ruleset).collect();
        for rule in BUILTIN_GOL.iter() {
            table.push(rule.colour);
            table.push(rule.colour2);
        }
        let scratch = GolScratch {
            rules: client.create_from_slice(bytemuck::cast_slice(&table)),
            meta: client.empty(pixels.max(1) * 4),
            pixels,
        };
        self.scratch = Some(scratch.clone());
        scratch
    }
}

/// Runs one Game of Life generation.
///
/// The whole skip decision stays on the device: the element mask populated by
/// `reaction_kernel::dispatch::prepare_sim_dispatch` resolves the
/// `SIM_SLOT_GOL_PIX` indirect-dispatch slot, so a canvas without a live `LIFE`
/// launches zero workgroups and costs no host read-back. The host-side
/// [`SimBuffers::has_seen_elem`] gate in `SimulationEngine::step` only skips
/// the group while `LIFE` has never been introduced.
///
/// Births are queued as `reaction_kernel::shared::request_create` records into
/// `SimBuffers::gpu_create_queue` and materialised by
/// `reaction_kernel::drain_pending_creates` in the same tick (the engine calls
/// `begin_pending_creates` before this pass and the drain right after it). The
/// queue holds `SimBuffers::GPU_CREATE_COUNT` records; a generation that would
/// create more (a whole-canvas birth) drops the surplus, which is TPT's
/// behaviour when `create_part` finds no free slot.
///
/// The `registry` is unused: births no longer allocate on the host. It stays in
/// the signature so the engine's call site does not have to change.
pub fn step_gol_gpu(
    bufs: &mut SimBuffers,
    gol: &mut GolState,
    _registry: &ElementRegistry,
    gspeed: u32,
) {
    let xres = bufs.config.xres;
    let yres = bufs.config.yres;
    if xres == 0 || yres == 0 || bufs.capacity == 0 {
        return;
    }
    let pixels = (xres as usize) * (yres as usize);
    let xcells = bufs.config.xcells();
    if pixels == 0 || xcells == 0 {
        return;
    }

    // `Simulation::BeforeSim` (`Simulation.cpp:3885-3888`):
    // `if (elementCount[LIFE] > 0 && ++CGOL >= GSPEED) SimulateGoL();`
    // `CGOL` is reset inside `SimulateGoL` (`Simulation.cpp:3446`).
    gol.cgol = gol.cgol.wrapping_add(1);
    let speed = gspeed.max(1);
    if gol.cgol < speed {
        return;
    }
    gol.cgol = 0;

    let scratch = gol.prepare(&bufs.client, pixels);
    let gol_grid = sim_grid(bufs, SIM_SLOT_GOL_PIX);
    unsafe {
        gol_analyze_kernel::launch_unchecked(
            &bufs.client,
            gol_grid.clone(),
            CubeDim::new_1d(CUBE_DIM),
            BufferArg::from_raw_parts(bufs.gpu_pmap.clone(), pixels),
            BufferArg::from_raw_parts(bufs.gpu_part_ctype.clone(), bufs.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_tmp2.clone(), bufs.capacity),
            BufferArg::from_raw_parts(scratch.rules.clone(), NGOL * 3),
            BufferArg::from_raw_parts(scratch.meta.clone(), pixels),
            xres,
            yres,
            bufs.capacity as u32,
            pixels as u32,
        );
        gol_update_kernel::launch_unchecked(
            &bufs.client,
            gol_grid.clone(),
            CubeDim::new_1d(CUBE_DIM),
            BufferArg::from_raw_parts(bufs.gpu_part_ty.clone(), bufs.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_tmp2.clone(), bufs.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_ctype.clone(), bufs.capacity),
            BufferArg::from_raw_parts(scratch.rules.clone(), NGOL * 3),
            BufferArg::from_raw_parts(scratch.meta.clone(), pixels),
            BufferArg::from_raw_parts(bufs.gpu_wall_bmap.clone(), bufs.wall_bmap.len()),
            BufferArg::from_raw_parts(bufs.gpu_wall_emap.clone(), bufs.wall_emap.len()),
            xres,
            xcells,
            bufs.capacity as u32,
            pixels as u32,
        );
        gol_birth_kernel::launch_unchecked(
            &bufs.client,
            gol_grid,
            CubeDim::new_1d(CUBE_DIM),
            BufferArg::from_raw_parts(bufs.gpu_pmap.clone(), pixels),
            BufferArg::from_raw_parts(bufs.gpu_part_ctype.clone(), bufs.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_tmp.clone(), bufs.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_dcolour.clone(), bufs.capacity),
            BufferArg::from_raw_parts(scratch.rules.clone(), NGOL * 3),
            BufferArg::from_raw_parts(scratch.meta.clone(), pixels),
            BufferArg::from_raw_parts(bufs.gpu_create_queue.clone(), CREATE_QUEUE_WORDS),
            bufs.capacity as u32,
            xres,
            yres,
            pixels as u32,
        );
    }

    gol.generations = gol.generations.wrapping_add(1);
    bufs.gpu_particles_dirty = true;
    bufs.gpu_maps_dirty = true;
    bufs.gpu_dirty = true;
}

/// Paint `LIFE` particles with the given `ctype`, as
/// `ElementTool(PMAP(ctype, PT_LIFE))` does in the reference game.
///
/// Paint goes through the normal placement path and then stamps TPT's `LIFE`
/// create fields (`ctype`, `tmp2`, `dcolour`, `tmp`) on the slots the stroke
/// created, without a device read-back: the host shadow is authoritative right
/// after `PlaceParticles`.
pub fn paint_life(
    engine: &mut SimulationEngine,
    points: &[(i32, i32)],
    ctype: u32,
    replace_mode: bool,
) {
    let ctype = ctype & GOL_RULE_MASK;
    if points.is_empty() {
        return;
    }
    // Outside replace mode `CreateBox` leaves an occupied pixel alone, so a
    // stroke over `LIFE` of another ruleset must not restamp the old cell.
    let occupied: Vec<bool> = if replace_mode {
        Vec::new()
    } else {
        points
            .iter()
            .map(|&(x, y)| engine.bufs.particle_index_at(x, y).is_some())
            .collect()
    };
    engine.apply_command(SimCommand::PlaceParticles {
        points: points.to_vec(),
        elem_ty: PT_LIFE,
        replace_mode,
        dcolour: 0,
    });

    // `LIFE.cpp:104-114`: a builtin rule takes its palette colours, a custom
    // ruleset the colours registered for it in the engine's table.
    let (mut colour1, mut colour2) = gol_colours(ctype);
    if ctype >= NGOL as u32
        && let Some(custom) = engine.custom_gol.iter().find(|g| g.ruleset == ctype)
    {
        colour1 = custom.colour;
        colour2 = custom.colour2 as i32;
    }
    let tmp2 = gol_max_state(ctype);
    let bufs = &mut engine.bufs;
    let mut touched: Vec<u32> = Vec::new();
    for (index, &(x, y)) in points.iter().enumerate() {
        if occupied.get(index) == Some(&true) {
            continue;
        }
        let Some(id) = bufs.particle_index_at(x, y) else {
            continue;
        };
        if id >= bufs.capacity || bufs.part_ty[id] != PT_LIFE {
            continue;
        }
        bufs.part_ctype[id] = ctype;
        bufs.part_tmp2[id] = tmp2;
        if colour1 != 0 || colour2 != 0 {
            bufs.part_dcolour[id] = colour1;
            bufs.part_tmp[id] = colour2;
        }
        touched.push(id as u32);
    }
    if touched.is_empty() {
        return;
    }
    for id in touched {
        bufs.mark_particle_slot_dirty(id as usize);
    }
    bufs.sync_host_to_gpu();
    engine.gol.wake();
}
