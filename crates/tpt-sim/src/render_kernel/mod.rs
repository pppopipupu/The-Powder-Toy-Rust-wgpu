//! GPU rendering pass for The Powder Toy.
//!
//! Port of TPT's fancy renderer (`src/graphics/Renderer.cpp`) split into
//! pixel-parallel passes. TPT draws particles one at a time and *scatters* each
//! effect into the framebuffer; a GPU cannot do that without atomics, so every
//! effect is turned into an equivalent *gather*: a pixel sums the contributions
//! of the particles around it. Additive effects are exactly equivalent, blend
//! effects differ only in the order neighbouring particles are composited in.
//!
//! | pass | TPT reference | storage buffers |
//! |------|---------------|-----------------|
//! | [`render_bg_kernel`] | `Renderer::Clear` + `draw_air` | 5 |
//! | [`render_walls_kernel`] | `Renderer::DrawWalls` | 7 |
//! | [`clear_fx_cells_kernel`] | port-only effect-flag clear | 2 |
//! | [`build_fx_rows_kernel`] | port-only spark row/column mask | 2 |
//! | [`render_particles_base_kernel`] | `render_parts` FLAT/BLEND/ADD + `EFFECT_LINES` bonds | 16 |
//! | [`render_stickmen_and_orbitals_kernel`] | `Renderer.cpp:540-660,793-847` STKM rigs, orbitals and the portal debug lines | 12 |
//! | [`render_particles_glow_kernel`] | `render_parts` BLOB/GLOW/BLUR | 16 |
//! | [`render_particles_rays_kernel`] | `render_parts` SPARK/FLARE/LFLARE | 16 |
//! | [`render_fire_accum_matter_kernel`] | `render_parts` fire buffers | 15 |
//! | [`render_fire_accum_energy_kernel`] | `render_parts` fire buffers | 15 |
//! | [`render_fire_spread_kernel`] | `Renderer::render_fire` glow gather | 6 |
//! | [`render_fire_add_kernel`] | `Renderer::render_fire` composite | 3 |
//! | [`render_fire_decay_kernel`] | `Renderer::render_fire` decay | 3 |
//! | [`copy_u32_kernel`], [`render_warp_kernel`] | `render_gravlensing` | 2, 4 |
//!
//! The particle-colouring passes bind per-particle arrays (`flags`, `life`,
//! `ctype`, `temp`, `dcolour`, `tmp`, `tmp2`, `tmp3`, `tmp4`, `vx`, `vy`) plus
//! the element table, because an element `Graphics` rule may read any of them;
//! they all stay well inside the 16-storage-buffer device limit. The `tick` scalar is
//! TPT's `sim->currentTick`.
//!
//! Every effect is gated twice, exactly like TPT: by the `RENDER_*` flags in
//! `SimulationConfig::render_mode` and by the element's `PMODE_*`/`FIRE_*`
//! bits from its `Graphics` function (stored in the flat `GpuElementProps`
//! table). The fallback chain (`FIRE_ADD` -> `GLOW` -> `BLEND`, `FIRE_BLEND` ->
//! `BLUR` -> `FLAT`, ...) is applied before the mask, so an effect that is
//! switched off degrades the way TPT degrades it instead of disappearing.
//!
//! [`render_gpu_only`] dispatches every pass without touching host memory;
//! [`render_simulation_gpu`] adds the framebuffer read-back used by the wgpu
//! texture upload (and by the headless tests).
//!
//! # Buffer layout notes
//!
//! * Every array is bound as a flat slice; struct arrays are bound as `&[u32]`
//!   words and indexed with the compile-time offsets below.
//! * `fire_r/g/b` are read-modify-written from many pixels of the same 4x4
//!   cell, exactly like TPT's per-particle accumulation. Concurrent updates can
//!   lose a contribution; the buffer only drives a soft additive bloom, so a
//!   lost update is a sub-percent brightness difference.
//! * The framebuffer packs `0xAABBGGRR`, i.e. the little-endian byte order
//!   `[R, G, B, A]` that `wgpu::TextureFormat::Rgba8Unorm` expects.

use core::cell::RefCell;
use core::mem::{offset_of, size_of};

use std::sync::LazyLock;
use std::sync::atomic::{AtomicU32, Ordering};
use std::thread::LocalKey;

use cubecl::prelude::*;
use cubecl::server::Handle;

use tpt_core::constants::*;
use tpt_core::elements::{ElementRegistry, GpuElementProps};
use tpt_core::walls::default_walls;

use crate::buffers::SimBuffers;

/// Words per `GpuElementProps` entry in the flat element-property buffer.
const ELEM_WORDS: u32 = (size_of::<GpuElementProps>() / size_of::<u32>()) as u32;
/// Word offset of `GpuElementProps::colour` (`0x00RRGGBB`).
const ELEM_COLOUR: u32 = (offset_of!(GpuElementProps, colour) / size_of::<u32>()) as u32;
/// Word offset of `GpuElementProps::properties` (`PROP_*` / `TYPE_*` bitflags).
const ELEM_PROPS: u32 = (offset_of!(GpuElementProps, properties) / size_of::<u32>()) as u32;
/// Word offset of `GpuElementProps::high_temp`.
const ELEM_HIGH_TEMP: u32 = (offset_of!(GpuElementProps, high_temp) / size_of::<u32>()) as u32;
/// Word offset of `GpuElementProps::pixel_mode` (`PMODE_*` / `FIRE_*` flags).
const ELEM_PIXEL_MODE: u32 = (offset_of!(GpuElementProps, pixel_mode) / size_of::<u32>()) as u32;
/// Word offset of `GpuElementProps::fire_colour` (`0xAARRGGBB`, `0` = no glow).
const ELEM_FIRE_COLOUR: u32 = (offset_of!(GpuElementProps, fire_colour) / size_of::<u32>()) as u32;
/// Word offset of `GpuElementProps::cola` (the `Graphics` alpha, `0..=255`).
const ELEM_COLA: u32 = (offset_of!(GpuElementProps, cola) / size_of::<u32>()) as u32;
/// Word offset of `GpuElementProps::flammable` (`DestructibilityScore`).
const ELEM_FLAMMABLE: u32 = (offset_of!(GpuElementProps, flammable) / size_of::<u32>()) as u32;
/// Word offset of `GpuElementProps::hardness` (`DestructibilityScore`).
const ELEM_HARDNESS: u32 = (offset_of!(GpuElementProps, hardness) / size_of::<u32>()) as u32;
/// Word offset of `GpuElementProps::low_pressure_transition`.
const ELEM_LOW_PRESSURE_TRANSITION: u32 =
    (offset_of!(GpuElementProps, low_pressure_transition) / size_of::<u32>()) as u32;
/// Word offset of `GpuElementProps::high_pressure_transition`.
const ELEM_HIGH_PRESSURE_TRANSITION: u32 =
    (offset_of!(GpuElementProps, high_pressure_transition) / size_of::<u32>()) as u32;
/// Word offset of `GpuElementProps::low_temp_transition`.
const ELEM_LOW_TEMP_TRANSITION: u32 =
    (offset_of!(GpuElementProps, low_temp_transition) / size_of::<u32>()) as u32;
/// Word offset of `GpuElementProps::high_temp_transition`.
const ELEM_HIGH_TEMP_TRANSITION: u32 =
    (offset_of!(GpuElementProps, high_temp_transition) / size_of::<u32>()) as u32;

/// Words per wall entry in the render wall table (`colour`, `eglow`, `drawstyle`, pad).
const WALL_WORDS: u32 = 4;
/// Wall table word holding `WallDef::colour`.
const WALL_COLOUR: u32 = 0;
/// Wall table word holding `WallDef::eglow`.
const WALL_EGLOW: u32 = 1;
/// Wall table word holding `WallDef::drawstyle`.
const WALL_STYLE: u32 = 2;

/// Number of wall ids in the render wall table.
const WALL_COUNT: u32 = WL_NUM as u32;

/// Wall ids as `u32` so they can be compared against kernel `u32` values.
const W_WALLELEC: u32 = WL_WALLELEC as u32;
const W_EWALL: u32 = WL_EWALL as u32;
const W_DETECT: u32 = WL_DETECT as u32;
const W_FAN: u32 = WL_FAN as u32;
const W_ALLOWLIQUID: u32 = WL_ALLOWLIQUID as u32;
const W_WALL: u32 = WL_WALL as u32;
const W_ALLOWAIR: u32 = WL_ALLOWAIR as u32;
const W_ALLOWPOWDER: u32 = WL_ALLOWPOWDER as u32;
const W_ALLOWALLELEC: u32 = WL_ALLOWALLELEC as u32;
const W_EHOLE: u32 = WL_EHOLE as u32;
const W_ALLOWGAS: u32 = WL_ALLOWGAS as u32;
const W_ALLOWENERGY: u32 = WL_ALLOWENERGY as u32;
const W_STASIS: u32 = WL_STASIS as u32;

/// Resolved wall draw style: solid fill.
const STYLE_SOLID: u32 = 3;
/// Resolved wall draw style: sparse dots.
const STYLE_DOTS: u32 = 1;
/// Resolved wall draw style: dithered mesh.
const STYLE_MESH: u32 = 2;
/// Resolved wall draw style: diagonal stripes.
const STYLE_STRIPES: u32 = 4;
/// Resolved wall draw style: dark hole (only the cell rim is drawn).
const STYLE_HOLE: u32 = 5;

/// `PMODE_*` / `FIRE_*` bits, mirrored from `src/simulation/ElementGraphics.h`.
///
/// They are stored verbatim in [`tpt_core::elements::ElementDef::pixel_mode`], so
/// the numeric values are part of the element-table contract.
const PMODE_FLAT: u32 = 0x0000_0001;
const PMODE_BLOB: u32 = 0x0000_0002;
const PMODE_BLUR: u32 = 0x0000_0004;
const PMODE_GLOW: u32 = 0x0000_0008;
const PMODE_SPARK: u32 = 0x0000_0010;
const PMODE_FLARE: u32 = 0x0000_0020;
const PMODE_LFLARE: u32 = 0x0000_0040;
const PMODE_ADD: u32 = 0x0000_0080;
const PMODE_BLEND: u32 = 0x0000_0100;
/// `CELL` as the kernels spell it: host constants never enter a kernel body.
const _: () = assert!(CELL == 4);

/// `fx_cell` bit: a pixel in the cell draws with a blob/glow/blur mode.
pub(crate) const FX_GLOW: u32 = 0x0000_0001;
/// `fx_cell` bit: a pixel in the cell draws with a spark/flare mode.
pub(crate) const FX_SPARK: u32 = 0x0000_0002;
/// `fx_cell` bit: a pixel in the cell draws with `PMODE_GLOW`. Finer than
/// [`FX_GLOW`] so the glow gather can reject a neighbour with one load instead
/// of re-deriving its effective pixel mode.
pub(crate) const FX_PMGLOW: u32 = 0x0000_0004;
/// `fx_cell` bit: a pixel in the cell draws with `PMODE_BLUR`.
pub(crate) const FX_PMBLUR: u32 = 0x0000_0008;

/// TPT `OPTIONS` mask; it carries `NO_DECO`/`DECO_FIRE` through the render mask.
const PMODE_OPTIONS: u32 = 0x0000_F000;
const NO_DECO: u32 = 0x0000_1000;
/// TPT `FLAG_PHOTDECO` (`src/simulation/ElementDefs.h:39`): photon decoration enabled.
pub(crate) const FLAG_PHOTDECO: u32 = 0x0000_0008;
const _: () = assert!(FLAG_PHOTDECO == 0x8);
/// TPT `DECO_FIRE`: `dcolour` decoration also tints the fire channels.
const DECO_FIRE: u32 = 0x0000_2000;
/// TPT `FIREMODE` mask (`FIRE_ADD | FIRE_BLEND | FIRE_SPARK`).
const FIREMODE: u32 = 0x00FF_0000;
const FIRE_ADD: u32 = 0x0001_0000;
const FIRE_BLEND: u32 = 0x0002_0000;
const FIRE_SPARK: u32 = 0x0004_0000;

/// `!(FIREMODE | PMODE_GLOW)`, used by the `COLOUR_HEAT`/`COLOUR_GRAD` rewrites.
const CLEAR_FIRE_GLOW: u32 = 0xFF00_FFF7;
/// `!(PMODE_BLEND | PMODE_ADD)`.
const CLEAR_BLEND_ADD: u32 = 0xFFFF_FE7F;
/// `!COLOUR_GRAD`, the mask TPT tests to decide whether decoration is visible.
const NOT_GRAD_ONLY: u32 = 0xFFFF_FFFB;

/// TPT `RendererSettings::DecorationLevel` (`RendererSettings.h:24-28`).
///
/// The render passes take the level as a scalar instead of a folded
/// `colour_mode` bit, because the anti-clickbait level needs a third state:
/// `Renderer.cpp:300-314` collapses every `dcolour` to "no decoration" unless
/// it is a near-black 250+ alpha, and `EMBR`/`LCRY`/`LIFE` test the level
/// themselves (`EMBR.cpp:104-122`, `LCRY.cpp:107-116`, `LIFE.cpp:51-91`).
pub(crate) const DECORATION_DISABLED: u32 = 0;
pub(crate) const DECORATION_ENABLED: u32 = 1;
pub(crate) const DECORATION_ANTI_CLICKBAIT: u32 = 2;
const _: () = assert!(
    DECORATION_DISABLED < DECORATION_ENABLED && DECORATION_ENABLED < DECORATION_ANTI_CLICKBAIT
);

/// Per-particle weight of an electrified wall's glow contribution.
const WALL_GLOW_SHIFT: u32 = 8;
/// Number of `fire_alpha` entries per axis (`3 * CELL`).
const FIRE_ALPHA_DIM: usize = 3 * CELL as usize;
/// [`FIRE_ALPHA_DIM`] as the `i32` the kernels index with.
const FIRE_ALPHA_SPAN: i32 = FIRE_ALPHA_DIM as i32;

/// Ray length cap for `PMODE_SPARK`/`PMODE_FLARE`/`PMODE_LFLARE`.
///
/// TPT walks each ray until its alpha drops below `0.5`: `SPARK` (`/1.5`) needs
/// at most 10 pixels, `FLARE` (`/1.2`) 32 (`255 / 1.2^31 < 1`). `LFLARE`
/// (`/1.01`) has no useful bound and is truncated here; no built-in element uses
/// it.
const RAY_STEPS: u32 = 32;

/// Threads per cube for the pixel-parallel render passes.
///
/// `256` matches every other slot/pixel-parallel pass in the engine because the
/// saturated pool (`SimBuffers::MAX_PARTICLE_CAPACITY = 0x7F_FFFF`, which a
/// 2048x2048 canvas asks for) needs `div_ceil(cap, 128) = 65 536` cubes for the
/// slot-parallel passes, one past the `65 535` workgroups-per-dimension ceiling
/// that `wgpu` validates on every dispatch.
const RENDER_CUBE_DIM: u32 = 256;

/// Workgroup ceiling per dispatch dimension, mirrored from the WebGPU limit
/// `max_compute_workgroups_per_dimension`.
const MAX_CUBES_PER_DIM: u32 = 65_535;

// Every dispatch dimension the render passes use must stay under the workgroup
// ceiling at the largest supported canvas and the largest supported pool; a
// regression here is a device-validation panic, not a wrong pixel.
const _: () = assert!(
    SimBuffers::MAX_PARTICLE_CAPACITY.div_ceil(RENDER_CUBE_DIM as usize)
        <= MAX_CUBES_PER_DIM as usize,
    "the slot-parallel render passes exceed the workgroup ceiling"
);
const _: () = assert!(
    (SimBuffers::MAX_SUPPORTED_CANVAS_SIDE as usize)
        .pow(2)
        .div_ceil(RENDER_CUBE_DIM as usize)
        <= MAX_CUBES_PER_DIM as usize,
    "the pixel-parallel render passes exceed the workgroup ceiling"
);

/// `pmap`/`photons` mask for the element type field.
const PMAP_TYPE_MASK: u32 = 0x1FF;
/// `pmap`/`photons` shift for the `particle id + 1` field.
const PMAP_ID_SHIFT: u32 = 9;

/// Whether any cell within `radius` cells of `(cx, cy)` carries `bit`.
///
/// The glow gather reaches five pixels, so a 5x5 cell window (`radius == 2`)
/// covers it; a region without a matching draw mode cannot produce output, so
/// the pass skips it.
#[cube]
pub(crate) fn cell_fx_any(
    fx_cell: &[u32],
    cx: u32,
    cy: u32,
    xcells: u32,
    ycells: u32,
    bit: u32,
    radius: u32,
) -> bool {
    let mut any = false;
    let span = radius * 2u32 + 1u32;
    for cj in 0u32..span {
        for ci in 0u32..span {
            let sx = cx as i32 + ci as i32 - radius as i32;
            let sy = cy as i32 + cj as i32 - radius as i32;
            if sx >= 0i32
                && sy >= 0i32
                && (sx as u32) < xcells
                && (sy as u32) < ycells
                && (fx_cell[((sy as u32) * xcells + (sx as u32)) as usize] & bit) != 0u32
            {
                any = true;
            }
        }
    }
    any
}

/// Clears the per-cell effect flags before the particle passes rebuild them.
///
/// One dispatch covers the cells, the rows and the columns: each thread clears
/// whichever of the three it lands in.
#[cube(launch_unchecked)]
pub fn clear_fx_cells_kernel(
    fx_cell: &mut [u32],
    fx_axis: &mut [u32],
    xcells: u32,
    ycells: u32,
    xres: u32,
    yres: u32,
) {
    let idx = ABSOLUTE_POS as u32;
    if idx < xcells * ycells {
        fx_cell[idx as usize] = 0u32;
    }
    if idx < xres + yres {
        fx_axis[idx as usize] = 0u32;
    }
}

/// Marks the rows and columns of every cell that carries [`FX_SPARK`].
///
/// The ray pass walks the four axis directions, so a pixel can only receive a
/// ray from its own row or its own column; two words per pixel replace the
/// 65-cell scan a 32-pixel reach would otherwise need.
#[cube(launch_unchecked)]
pub fn build_fx_rows_kernel(
    fx_cell: &[u32],
    fx_axis: &mut [u32],
    xcells: u32,
    ycells: u32,
    xres: u32,
    yres: u32,
) {
    let idx = ABSOLUTE_POS as u32;
    if idx < xcells * ycells && (fx_cell[idx as usize] & FX_SPARK) != 0u32 {
        let cx = idx % xcells;
        let cy = idx / xcells;
        for d in 0u32..4u32 {
            let y = cy * 4u32 + d;
            if y < yres {
                fx_axis[y as usize] = 1u32;
            }
            let x = cx * 4u32 + d;
            if x < xres {
                fx_axis[(yres + x) as usize] = 1u32;
            }
        }
    }
}

mod background;
mod colour_scale;
mod fancy;
mod fire;
mod framebuffer;
mod overlays;
pub(crate) mod particles;
mod rays;
mod warp;

pub use background::*;
pub use colour_scale::current_hdisp_limits;
pub use fancy::*;
pub use fire::*;
pub use overlays::*;
pub use particles::*;
pub use rays::*;
pub use warp::*;

/// Wall render table (`colour`, `eglow`, `drawstyle`, pad) built from [`default_walls`].
fn wall_table() -> &'static [u32] {
    static TABLE: LazyLock<Vec<u32>> = LazyLock::new(|| {
        let mut table = vec![0u32; WL_NUM * WALL_WORDS as usize];
        for wall in default_walls().iter().take(WL_NUM) {
            let base = wall.id as usize * WALL_WORDS as usize;
            table[base + WALL_COLOUR as usize] = wall.colour;
            table[base + WALL_EGLOW as usize] = wall.eglow;
            table[base + WALL_STYLE as usize] = wall.drawstyle as u32;
        }
        table
    });
    TABLE.as_slice()
}

/// TPT `Renderer::prepare_alpha`: the 12x12 `fire_alpha` kernel at intensity 1.
///
/// `fire_alpha[y][x] = 255 * sum(exp(-0.1 * (i^2 + j^2))) / CELL^2` over the
/// `CELL x CELL` source cell offsets that land on `(x, y)`.
fn fire_alpha_table() -> &'static [u32] {
    static TABLE: LazyLock<Vec<u32>> = LazyLock::new(|| {
        let cell = CELL as i32;
        let dim = FIRE_ALPHA_DIM;
        let mut temp = vec![0.0f32; dim * dim];
        for x in 0..cell {
            for y in 0..cell {
                for i in -cell..cell {
                    for j in -cell..cell {
                        let row = (y + cell + j) as usize;
                        let col = (x + cell + i) as usize;
                        temp[row * dim + col] += (-0.1f32 * ((i * i + j * j) as f32)).exp();
                    }
                }
            }
        }
        let scale = 255.0f32 / ((cell * cell) as f32);
        temp.iter().map(|v| (v * scale) as u32).collect()
    });
    TABLE.as_slice()
}

/// A device buffer cached across frames.
struct CachedBuffer {
    /// Address of the owning `Client`; a different client means a new device.
    client: usize,
    /// Length in `u32` words.
    len: usize,
    handle: Handle,
}

// Each initialiser is already a `const` block; clippy 1.99 still reports the
// statics whose value type owns a device handle, so the lint is silenced per
// static instead of leaving warnings in the build.
thread_local! {
    /// Device copy of the immutable wall palette.
    #[allow(clippy::missing_const_for_thread_local)]
    static WALL_TABLE_GPU: RefCell<Option<CachedBuffer>> =
        const { RefCell::new(None::<CachedBuffer>) };
    /// Device copy of the immutable `fire_alpha` kernel.
    #[allow(clippy::missing_const_for_thread_local)]
    static FIRE_ALPHA_GPU: RefCell<Option<CachedBuffer>> =
        const { RefCell::new(None::<CachedBuffer>) };
    /// Scratch framebuffer used by the gravity-lensing pass.
    #[allow(clippy::missing_const_for_thread_local)]
    static WARP_SCRATCH_GPU: RefCell<Option<CachedBuffer>> =
        const { RefCell::new(None::<CachedBuffer>) };
}

/// Return a cached device buffer of `len` words, re-creating it when the client
/// or the size changed.
///
/// `upload` seeds a fresh buffer; `None` leaves it uninitialised (scratch).
/// The renderer is single-threaded and the CubeCL client lives for the whole
/// process, so the handle stays valid for every frame after the first. A client
/// address is used as the device identity: a re-created device gets a new
/// `SimBuffers`, so the stale entry is dropped instead of being handed to the
/// new queue.
fn cached_buffer(
    slot: &'static LocalKey<RefCell<Option<CachedBuffer>>>,
    client: &Client,
    len: usize,
    upload: Option<&[u32]>,
) -> Handle {
    slot.with(|cell| {
        let client_id = client as *const Client as usize;
        let mut borrow = cell.borrow_mut();
        let rebuild = match borrow.as_ref() {
            Some(cached) => cached.client != client_id || cached.len != len,
            None => true,
        };
        if rebuild {
            let handle = match upload {
                Some(data) => client.create_from_slice(bytemuck::cast_slice(data)),
                None => client.empty(len * size_of::<u32>()),
            };
            *borrow = Some(CachedBuffer {
                client: client_id,
                len,
                handle,
            });
        }
        // The branch above always leaves an entry behind.
        let cached = borrow.as_ref().expect("cached device buffer");
        cached.handle.clone()
    })
}

/// `true` when the cached `GpuElementProps` upload no longer matches `registry`.
fn element_table_stale(bufs: &SimBuffers, registry: &ElementRegistry) -> bool {
    if registry.elements.len() != bufs.elem_props_host.len() {
        return true;
    }
    registry
        .elements
        .iter()
        .zip(bufs.elem_props_host.iter())
        .any(|(def, gpu)| {
            def.colour != gpu.colour
                || def.properties != gpu.properties
                || def.high_temp.to_bits() != gpu.high_temp.to_bits()
                || def.pixel_mode != gpu.pixel_mode
                || def.fire_colour != gpu.fire_colour
        })
}

/// Frame counter feeding the `SPARK`/`FLARE` flicker.
static FRAME_SEED: AtomicU32 = AtomicU32::new(0x9E37_79B9);

/// Dispatch every render pass on the device, without any host read-back.
///
/// The passes run in TPT's order: background and air overlays, walls, particle
/// base colours, `BLOB`/`GLOW`/`BLUR`, `SPARK`/`FLARE`/`LFLARE`, fire
/// accumulation, the fire glow layer and its composite, fire decay, gravity
/// lensing.
/// Pending host edits are uploaded first, so a paused canvas still renders
/// correctly.
pub fn render_gpu_only(bufs: &mut SimBuffers, registry: &ElementRegistry) {
    let xres = bufs.config.xres;
    let yres = bufs.config.yres;
    if xres == 0 || yres == 0 {
        return;
    }

    // Never render a half-uploaded state: brush edits land on the host shadow first.
    if bufs.host_dirty {
        bufs.sync_host_to_gpu();
    }
    // Element colours/properties live in a GPU table; refresh it when Lua or the
    // UI changed an element definition.
    if element_table_stale(bufs, registry) {
        bufs.refresh_element_tables(registry);
    }

    let xcells = bufs.config.xcells();
    let ycells = bufs.config.ycells();
    let pixels = bufs.config.num_pixels();
    let cells = bufs.config.num_cells();
    if pixels == 0 || cells == 0 {
        return;
    }
    if bufs.config.gravity_zones_enabled {
        // `draw_grav_zones` reads `gravIn.mask`, which TPT refreshes on every
        // wall edit regardless of whether Newtonian gravity is running.
        crate::gravity_kernel::ensure_gravity_mask(bufs);
    }

    let display_mode = bufs.config.display_mode;
    // `Renderer::AdjustHdispLimit` (`Renderer.cpp:1444-1512`): an explicit
    // `heat_display_limit_*` wins, otherwise the auto mode samples the particle
    // temperatures (skipping zero `HeatConduct` elements) and, with the
    // ambient-heat overlay on, the `hv` grid. The two bounds follow the
    // reference's 5 % per-frame shrink and the one-degree minimum gap; the
    // smoothing state lives in `colour_scale`, so a paused canvas keeps the
    // last resolved pair instead of resetting it.
    let (hdisp_min, hdisp_max) = colour_scale::resolve_hdisp_limits(bufs);
    let render_mode = bufs.config.render_mode;
    // `RendererSettings::decorationLevel` (`RendererSettings.h:24-28`). The
    // port's settings only expose the on/off switch (`SimulationConfig::
    // decorations`, TPT's `Renderer.Decorations`), so the anti-clickbait level
    // has no producer yet; the render passes implement its collapse anyway
    // (`Renderer.cpp:300-314`) so a UI option can be wired without touching a
    // kernel.
    let decoration_level = if bufs.config.decorations {
        DECORATION_ENABLED
    } else {
        DECORATION_DISABLED
    };
    let colour_mode = bufs.config.colour_mode;
    let pretty_powder = u32::from(bufs.config.pretty_powder);
    let fire_intensity = bufs.config.fire_intensity;
    let elem_count = bufs.elem_props_host.len();
    let elem_words = elem_count * ELEM_WORDS as usize;
    if elem_count == 0 {
        return;
    }

    // `Simulation::EnableNewtonianGravity(false)` resets `gravOut` to zero
    // (`Simulation.cpp:3975-3983`). `RenderSimulation` runs the lensing pass on
    // that output regardless of the model (`Renderer.cpp:42-47`), so a
    // switched-off model must lens a zero field instead of the previous
    // solve's force; otherwise the fancy view keeps bending with stale data.
    if !bufs.config.newton_grav {
        bufs.zero_gravity_forces();
    }

    // Immutable tables: uploaded once and cached for the life of the device.
    let wall_data = wall_table();
    let alpha_data = fire_alpha_table();
    let wall_handle = cached_buffer(
        &WALL_TABLE_GPU,
        &bufs.client,
        wall_data.len(),
        Some(wall_data),
    );
    let fire_alpha_handle = cached_buffer(
        &FIRE_ALPHA_GPU,
        &bufs.client,
        alpha_data.len(),
        Some(alpha_data),
    );

    let cube_dim = CubeDim::new_1d(RENDER_CUBE_DIM);
    // `CubeCount` is not `Copy`, so build a fresh one per launch.
    let pixel_cubes = || CubeCount::new_1d((pixels as u32).div_ceil(RENDER_CUBE_DIM));
    let cell_cubes = || CubeCount::new_1d((cells as u32).div_ceil(RENDER_CUBE_DIM));
    // The stickman/orbital pass only touches live slots, so it dispatches over
    // the used prefix instead of the whole pool (a 1024x768 canvas reserves
    // 1.57 M slots and uses a few tens of thousands).
    let slot_cubes =
        || CubeCount::new_1d((bufs.used_slot_bound() as u32).div_ceil(RENDER_CUBE_DIM));

    let fire_enabled = (render_mode & (RENDER_FIRE | RENDER_SPRK)) != 0;
    // TPT renders and simulates in lockstep, so its fire buffer advances once
    // per tick. The port presents up to 240 frames per second while stepping at
    // the configured TPS, so the accumulation and the decay are gated on the
    // simulation tick instead of the render call; the bloom composite is a pure
    // read and still runs every frame. While paused the buffer keeps decaying
    // per presented frame, which is what TPT does when its simulation is paused.
    let fire_due = bufs.config.paused || bufs.sim_tick != bufs.fire_tick;
    let glow_enabled = (render_mode & (RENDER_GLOW | RENDER_BLUR | RENDER_BLOB)) != 0;
    let rays_enabled = (render_mode & (RENDER_SPRK | RENDER_EFFE)) != 0;
    let warp_enabled = (display_mode & DISPLAY_WARP) != 0;
    let grav_field_enabled = bufs.config.gravity_field_enabled;
    let grav_zones_enabled = bufs.config.gravity_zones_enabled;
    let emp_decor = bufs.emp_decor;
    // `Renderer::draw_other` early-returns outside the effect modes.
    let emp_enabled = emp_decor > 0 && (render_mode & RENDER_EFFE) != 0;
    // `sim->currentTick` for the `Graphics` rules that blink with the tick
    // (`GRAV.cpp:56-61`, `GLOW.cpp:105`, `SLCN.cpp:125`). TPT reads the
    // simulation tick, not a render counter, so the blink rate follows the
    // configured TPS instead of the present rate.
    let tick = bufs.sim_tick as u32;

    // SAFETY: every `BufferArg` below is created from a device buffer allocated by
    // `SimBuffers` with exactly the element count passed alongside it, and each
    // kernel indexes strictly inside `xres * yres` / `xcells * ycells`.
    unsafe {
        render_bg_kernel::launch_unchecked(
            &bufs.client,
            pixel_cubes(),
            cube_dim,
            BufferArg::from_raw_parts(bufs.gpu_framebuf.clone(), pixels),
            BufferArg::from_raw_parts(bufs.gpu_air_vx.clone(), cells),
            BufferArg::from_raw_parts(bufs.gpu_air_vy.clone(), cells),
            BufferArg::from_raw_parts(bufs.gpu_air_pv.clone(), cells),
            BufferArg::from_raw_parts(bufs.gpu_air_hv.clone(), cells),
            xres,
            yres,
            xcells,
            ycells,
            u32::from(bufs.config.aheat_enable),
            display_mode,
            hdisp_min,
            hdisp_max,
        );

        // `Renderer::RenderSimulation` draws the gravity field before the walls.
        if grav_field_enabled {
            grav_field_kernel::launch_unchecked(
                &bufs.client,
                cell_cubes(),
                cube_dim,
                BufferArg::from_raw_parts(bufs.gpu_framebuf.clone(), pixels),
                BufferArg::from_raw_parts(bufs.gpu_grav_fx.clone(), cells),
                BufferArg::from_raw_parts(bufs.gpu_grav_fy.clone(), cells),
                xres,
                yres,
                xcells,
                ycells,
            );
        }

        render_walls_kernel::launch_unchecked(
            &bufs.client,
            pixel_cubes(),
            cube_dim,
            BufferArg::from_raw_parts(bufs.gpu_framebuf.clone(), pixels),
            BufferArg::from_raw_parts(bufs.gpu_wall_bmap.clone(), cells),
            BufferArg::from_raw_parts(bufs.gpu_wall_emap.clone(), cells),
            BufferArg::from_raw_parts(wall_handle, WL_NUM * WALL_WORDS as usize),
            BufferArg::from_raw_parts(bufs.gpu_fire_r.clone(), cells),
            BufferArg::from_raw_parts(bufs.gpu_fire_g.clone(), cells),
            BufferArg::from_raw_parts(bufs.gpu_fire_b.clone(), cells),
            xres,
            yres,
            xcells,
            render_mode,
        );

        // The particle passes rebuild the per-cell effect flags below; the
        // glow and spark gathers then skip cells that carry neither.
        clear_fx_cells_kernel::launch_unchecked(
            &bufs.client,
            cell_cubes(),
            cube_dim,
            BufferArg::from_raw_parts(bufs.gpu_fx_cell.clone(), cells),
            BufferArg::from_raw_parts(bufs.gpu_fx_axis.clone(), xres as usize + yres as usize),
            xcells,
            ycells,
            xres,
            yres,
        );

        // `Renderer.cpp:527-538`: every attached `SOAP` bubble draws a
        // `BlendLine` to the partner stored in `tmp`, in its own colour and
        // `cola`. TPT draws it inside the particle loop, before the bubble's own
        // pixel mode, so the base particle pass below calls `soap_outline` for
        // every `SOAP` pixel; the bubble's own pixels still cover the bond
        // endpoints. `EFFECT_LINES` only survives the render mask under
        // `RENDER_EFFE` (`ElementGraphics.h:32`), which the pass checks.

        render_particles_base_kernel::launch_unchecked(
            &bufs.client,
            pixel_cubes(),
            cube_dim,
            BufferArg::from_raw_parts(bufs.gpu_photons.clone(), pixels),
            BufferArg::from_raw_parts(bufs.gpu_pmap.clone(), pixels),
            BufferArg::from_raw_parts(bufs.gpu_part_flags.clone(), bufs.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_life.clone(), bufs.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_ctype.clone(), bufs.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_temp.clone(), bufs.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_dcolour.clone(), bufs.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_tmp.clone(), bufs.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_tmp2.clone(), bufs.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_tmp3.clone(), bufs.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_tmp4.clone(), bufs.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_vx.clone(), bufs.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_vy.clone(), bufs.capacity),
            BufferArg::from_raw_parts(bufs.gpu_elem_props.clone(), elem_words),
            BufferArg::from_raw_parts(bufs.gpu_framebuf.clone(), pixels),
            BufferArg::from_raw_parts(bufs.gpu_fx_cell.clone(), cells),
            xres,
            yres,
            render_mode,
            colour_mode,
            decoration_level,
            pretty_powder,
            elem_count as u32,
            hdisp_min,
            hdisp_max,
            tick,
        );

        render_stickmen_and_orbitals_kernel::launch_unchecked(
            &bufs.client,
            slot_cubes(),
            cube_dim,
            BufferArg::from_raw_parts(bufs.gpu_part_ty.clone(), bufs.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_x.clone(), bufs.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_y.clone(), bufs.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_life.clone(), bufs.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_ctype.clone(), bufs.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_tmp.clone(), bufs.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_temp.clone(), bufs.capacity),
            BufferArg::from_raw_parts(bufs.gpu_stkm_legs.clone(), STKM_SLOTS * STKM_LEGS_STRIDE),
            BufferArg::from_raw_parts(bufs.gpu_stkm_meta.clone(), STKM_SLOTS * STKM_META_STRIDE),
            BufferArg::from_raw_parts(bufs.gpu_elem_props.clone(), elem_words),
            BufferArg::from_raw_parts(bufs.gpu_framebuf.clone(), pixels),
            xres,
            yres,
            bufs.capacity as u32,
            elem_count as u32,
            render_mode,
            BufferArg::from_raw_parts(bufs.gpu_pmap.clone(), pixels),
            bufs.config.mouse_x,
            bufs.config.mouse_y,
            u32::from(bufs.config.debug_hud),
            u32::from((bufs.config.display_mode & DISPLAY_PERS) != 0),
        );

        if glow_enabled {
            render_particles_glow_kernel::launch_unchecked(
                &bufs.client,
                pixel_cubes(),
                cube_dim,
                BufferArg::from_raw_parts(bufs.gpu_photons.clone(), pixels),
                BufferArg::from_raw_parts(bufs.gpu_pmap.clone(), pixels),
                BufferArg::from_raw_parts(bufs.gpu_part_flags.clone(), bufs.capacity),
                BufferArg::from_raw_parts(bufs.gpu_part_life.clone(), bufs.capacity),
                BufferArg::from_raw_parts(bufs.gpu_part_ctype.clone(), bufs.capacity),
                BufferArg::from_raw_parts(bufs.gpu_part_temp.clone(), bufs.capacity),
                BufferArg::from_raw_parts(bufs.gpu_part_dcolour.clone(), bufs.capacity),
                BufferArg::from_raw_parts(bufs.gpu_part_tmp.clone(), bufs.capacity),
                BufferArg::from_raw_parts(bufs.gpu_part_tmp2.clone(), bufs.capacity),
                BufferArg::from_raw_parts(bufs.gpu_part_tmp3.clone(), bufs.capacity),
                BufferArg::from_raw_parts(bufs.gpu_part_tmp4.clone(), bufs.capacity),
                BufferArg::from_raw_parts(bufs.gpu_part_vx.clone(), bufs.capacity),
                BufferArg::from_raw_parts(bufs.gpu_part_vy.clone(), bufs.capacity),
                BufferArg::from_raw_parts(bufs.gpu_elem_props.clone(), elem_words),
                BufferArg::from_raw_parts(bufs.gpu_framebuf.clone(), pixels),
                BufferArg::from_raw_parts(bufs.gpu_fx_cell.clone(), cells),
                xres,
                yres,
                render_mode,
                colour_mode,
                decoration_level,
                pretty_powder,
                elem_count as u32,
                hdisp_min,
                hdisp_max,
                tick,
            );
        }

        if rays_enabled {
            build_fx_rows_kernel::launch_unchecked(
                &bufs.client,
                cell_cubes(),
                cube_dim,
                BufferArg::from_raw_parts(bufs.gpu_fx_cell.clone(), cells),
                BufferArg::from_raw_parts(bufs.gpu_fx_axis.clone(), xres as usize + yres as usize),
                xcells,
                ycells,
                xres,
                yres,
            );
            let frame_seed = FRAME_SEED.fetch_add(0x9E37_79B9, Ordering::Relaxed);
            render_particles_rays_kernel::launch_unchecked(
                &bufs.client,
                pixel_cubes(),
                cube_dim,
                BufferArg::from_raw_parts(bufs.gpu_photons.clone(), pixels),
                BufferArg::from_raw_parts(bufs.gpu_pmap.clone(), pixels),
                BufferArg::from_raw_parts(bufs.gpu_part_flags.clone(), bufs.capacity),
                BufferArg::from_raw_parts(bufs.gpu_part_life.clone(), bufs.capacity),
                BufferArg::from_raw_parts(bufs.gpu_part_ctype.clone(), bufs.capacity),
                BufferArg::from_raw_parts(bufs.gpu_part_temp.clone(), bufs.capacity),
                BufferArg::from_raw_parts(bufs.gpu_part_dcolour.clone(), bufs.capacity),
                BufferArg::from_raw_parts(bufs.gpu_part_tmp.clone(), bufs.capacity),
                BufferArg::from_raw_parts(bufs.gpu_part_tmp2.clone(), bufs.capacity),
                BufferArg::from_raw_parts(bufs.gpu_part_tmp3.clone(), bufs.capacity),
                BufferArg::from_raw_parts(bufs.gpu_part_tmp4.clone(), bufs.capacity),
                BufferArg::from_raw_parts(bufs.gpu_part_vx.clone(), bufs.capacity),
                BufferArg::from_raw_parts(bufs.gpu_part_vy.clone(), bufs.capacity),
                BufferArg::from_raw_parts(bufs.gpu_elem_props.clone(), elem_words),
                BufferArg::from_raw_parts(bufs.gpu_framebuf.clone(), pixels),
                BufferArg::from_raw_parts(bufs.gpu_fx_axis.clone(), xres as usize + yres as usize),
                xres,
                yres,
                render_mode,
                colour_mode,
                decoration_level,
                pretty_powder,
                elem_count as u32,
                hdisp_min,
                hdisp_max,
                frame_seed,
                tick,
            );
        }

        if fire_enabled && fire_due {
            render_fire_accum_matter_kernel::launch_unchecked(
                &bufs.client,
                cell_cubes(),
                cube_dim,
                BufferArg::from_raw_parts(bufs.gpu_pmap.clone(), pixels),
                BufferArg::from_raw_parts(bufs.gpu_part_life.clone(), bufs.capacity),
                BufferArg::from_raw_parts(bufs.gpu_part_ctype.clone(), bufs.capacity),
                BufferArg::from_raw_parts(bufs.gpu_part_temp.clone(), bufs.capacity),
                BufferArg::from_raw_parts(bufs.gpu_part_dcolour.clone(), bufs.capacity),
                BufferArg::from_raw_parts(bufs.gpu_part_tmp.clone(), bufs.capacity),
                BufferArg::from_raw_parts(bufs.gpu_part_tmp2.clone(), bufs.capacity),
                BufferArg::from_raw_parts(bufs.gpu_part_tmp3.clone(), bufs.capacity),
                BufferArg::from_raw_parts(bufs.gpu_part_tmp4.clone(), bufs.capacity),
                BufferArg::from_raw_parts(bufs.gpu_part_vx.clone(), bufs.capacity),
                BufferArg::from_raw_parts(bufs.gpu_part_vy.clone(), bufs.capacity),
                BufferArg::from_raw_parts(bufs.gpu_elem_props.clone(), elem_words),
                BufferArg::from_raw_parts(bufs.gpu_fire_r.clone(), cells),
                BufferArg::from_raw_parts(bufs.gpu_fire_g.clone(), cells),
                BufferArg::from_raw_parts(bufs.gpu_fire_b.clone(), cells),
                xres,
                yres,
                xcells,
                ycells,
                render_mode,
                colour_mode,
                decoration_level,
                pretty_powder,
                elem_count as u32,
                hdisp_min,
                hdisp_max,
                tick,
            );

            render_fire_accum_energy_kernel::launch_unchecked(
                &bufs.client,
                cell_cubes(),
                cube_dim,
                BufferArg::from_raw_parts(bufs.gpu_photons.clone(), pixels),
                BufferArg::from_raw_parts(bufs.gpu_part_life.clone(), bufs.capacity),
                BufferArg::from_raw_parts(bufs.gpu_part_ctype.clone(), bufs.capacity),
                BufferArg::from_raw_parts(bufs.gpu_part_temp.clone(), bufs.capacity),
                BufferArg::from_raw_parts(bufs.gpu_part_dcolour.clone(), bufs.capacity),
                BufferArg::from_raw_parts(bufs.gpu_part_tmp.clone(), bufs.capacity),
                BufferArg::from_raw_parts(bufs.gpu_part_tmp2.clone(), bufs.capacity),
                BufferArg::from_raw_parts(bufs.gpu_part_tmp3.clone(), bufs.capacity),
                BufferArg::from_raw_parts(bufs.gpu_part_tmp4.clone(), bufs.capacity),
                BufferArg::from_raw_parts(bufs.gpu_part_vx.clone(), bufs.capacity),
                BufferArg::from_raw_parts(bufs.gpu_part_vy.clone(), bufs.capacity),
                BufferArg::from_raw_parts(bufs.gpu_elem_props.clone(), elem_words),
                BufferArg::from_raw_parts(bufs.gpu_fire_r.clone(), cells),
                BufferArg::from_raw_parts(bufs.gpu_fire_g.clone(), cells),
                BufferArg::from_raw_parts(bufs.gpu_fire_b.clone(), cells),
                xres,
                yres,
                xcells,
                ycells,
                render_mode,
                colour_mode,
                decoration_level,
                pretty_powder,
                elem_count as u32,
                hdisp_min,
                hdisp_max,
                tick,
            );
        }

        // The glow gather only depends on the fire cells, which change at most
        // once per simulation tick, so it is resolved into the layer once per
        // tick and only the one-tap add runs at the present rate.
        if fire_due {
            render_fire_spread_kernel::launch_unchecked(
                &bufs.client,
                pixel_cubes(),
                cube_dim,
                BufferArg::from_raw_parts(bufs.gpu_fire_r.clone(), cells),
                BufferArg::from_raw_parts(bufs.gpu_fire_g.clone(), cells),
                BufferArg::from_raw_parts(bufs.gpu_fire_b.clone(), cells),
                BufferArg::from_raw_parts(fire_alpha_handle, FIRE_ALPHA_DIM * FIRE_ALPHA_DIM),
                BufferArg::from_raw_parts(bufs.gpu_fire_layer.clone(), pixels),
                xres,
                yres,
                xcells,
                ycells,
                fire_intensity,
            );
        }

        if fire_enabled {
            render_fire_add_kernel::launch_unchecked(
                &bufs.client,
                pixel_cubes(),
                cube_dim,
                BufferArg::from_raw_parts(bufs.gpu_framebuf.clone(), pixels),
                BufferArg::from_raw_parts(bufs.gpu_fire_layer.clone(), pixels),
                xres,
                yres,
            );
        }

        // Decay runs even while fire rendering is switched off, so a disabled
        // mode cannot leave a stale bloom behind for the next time it is turned
        // on; it is still tied to the simulation tick, not the present rate.
        if fire_due {
            render_fire_decay_kernel::launch_unchecked(
                &bufs.client,
                cell_cubes(),
                cube_dim,
                BufferArg::from_raw_parts(bufs.gpu_fire_r.clone(), cells),
                BufferArg::from_raw_parts(bufs.gpu_fire_g.clone(), cells),
                BufferArg::from_raw_parts(bufs.gpu_fire_b.clone(), cells),
                xcells,
                ycells,
            );
            bufs.fire_tick = bufs.sim_tick;
        }

        // `Renderer::draw_other` (EMP flash) and `draw_grav_zones`, in TPT's
        // order: after the fire composite, before the lensing pass.
        // The layer is applied above, before the decay, so the frame shows the
        // same pre-decay glow TPT composites.
        if emp_enabled {
            emp_tint_kernel::launch_unchecked(
                &bufs.client,
                pixel_cubes(),
                cube_dim,
                BufferArg::from_raw_parts(bufs.gpu_framebuf.clone(), pixels),
                xres,
                yres,
                emp_decor,
            );
        }

        if grav_zones_enabled {
            grav_zones_kernel::launch_unchecked(
                &bufs.client,
                cell_cubes(),
                cube_dim,
                BufferArg::from_raw_parts(bufs.gpu_framebuf.clone(), pixels),
                BufferArg::from_raw_parts(bufs.gpu_grav_mask.clone(), cells),
                xres,
                yres,
                xcells,
                ycells,
            );
        }

        if warp_enabled {
            let scratch = cached_buffer(&WARP_SCRATCH_GPU, &bufs.client, pixels, None);
            copy_u32_kernel::launch_unchecked(
                &bufs.client,
                pixel_cubes(),
                cube_dim,
                BufferArg::from_raw_parts(bufs.gpu_framebuf.clone(), pixels),
                BufferArg::from_raw_parts(scratch.clone(), pixels),
                pixels as u32,
            );
            render_warp_kernel::launch_unchecked(
                &bufs.client,
                pixel_cubes(),
                cube_dim,
                BufferArg::from_raw_parts(bufs.gpu_framebuf.clone(), pixels),
                BufferArg::from_raw_parts(scratch, pixels),
                BufferArg::from_raw_parts(bufs.gpu_grav_fx.clone(), cells),
                BufferArg::from_raw_parts(bufs.gpu_grav_fy.clone(), cells),
                xres,
                yres,
                xcells,
            );
        }
    }
}

/// Render the whole simulation into the packed framebuffer and return it as RGBA8 bytes.
///
/// Runs [`render_gpu_only`] and then copies the packed framebuffer back into
/// [`SimBuffers::framebuf_rgba`]. The returned slice aliases that host buffer and
/// is ready to be handed to `queue.write_texture` for an `Rgba8Unorm` texture.
pub fn render_simulation_gpu<'a>(bufs: &'a mut SimBuffers, registry: &ElementRegistry) -> &'a [u8] {
    let pixels = bufs.config.num_pixels();
    if pixels == 0 {
        return &[];
    }
    if bufs.framebuf_rgba.len() != pixels {
        bufs.framebuf_rgba.resize(pixels, 0);
    }

    render_gpu_only(bufs, registry);

    // Read the finished image back into the host shadow. Only the framebuffer is
    // pulled back: the glow buffers stay device-resident and `sync_host_to_gpu`
    // never uploads them, so their stale host copies cannot clobber the device.
    let bytes = bufs.client.read_one_unchecked(bufs.gpu_framebuf.clone());
    let raw: &[u8] = &bytes;
    let available = (raw.len() / 4).min(pixels);
    for (dst, chunk) in bufs.framebuf_rgba[..available]
        .iter_mut()
        .zip(raw.as_chunks::<4>().0.iter())
    {
        *dst = u32::from_le_bytes(*chunk);
    }

    bytemuck::cast_slice(&bufs.framebuf_rgba)
}
