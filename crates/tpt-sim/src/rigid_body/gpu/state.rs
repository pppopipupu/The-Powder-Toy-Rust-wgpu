//! Device-resident rigid-body state and host synchronization.

use cubecl::bytes::Bytes;
use cubecl::client::Client;
use cubecl::server::Handle;

use tpt_core::elements::ElementRegistry;
use tpt_core::particle::{RigidBody2D, RigidPixel};

use crate::buffers::SimBuffers;

use super::layout::*;

/// Device-resident state of the rigid-body solver.
///
/// The struct is deliberately *not* part of [`SimBuffers`]: the rigid working
/// set (body records + pixel records) belongs to the rigid subsystem, and the
/// particle / map buffers stay owned by the shared simulation state.
pub struct RigidGpu {
    pub(super) bodies: Option<Handle>,
    pub(super) pixels: Option<Handle>,
    pub(super) claims: Option<Handle>,
    /// Per-body contact manifolds: [`MANIFOLDS`] slots of [`M_STRIDE`] words
    /// each. Every contact thread only touches its own body's row, so the
    /// grouping is race-free; the manifold is flushed into the receiver's
    /// fixed-point reaction words at the end of the pass.
    pub(super) manifolds: Option<Handle>,
    pub(super) owner: Option<Handle>,
    /// One *bit* per device pixel: set when the contact pass must visit it.
    ///
    /// The pass runs one thread per body, so every flag it reads costs a full
    /// memory latency with no other warp to hide it behind. A packed bitmap
    /// lets the scan skip 32 interior pixels per word instead of paying one
    /// latency each: a solid 200x200 body has 800 surface pixels among 40 000,
    /// so the sweep reads ~1 250 words rather than 40 000 words. Interior
    /// pixels, whose eight neighbours are all body pixels, can never meet a
    /// loose particle, so their bit stays clear.
    pub(super) surface: Option<Handle>,
    /// Three `f32` per device pixel: the air pressure / wind force and the
    /// torque contribution `prepare_forces_kernel` computed for it.
    ///
    /// `prepare_forces_kernel` fills the stage and `prepare_finish_kernel`
    /// sums it in pixel order, exactly the order the
    /// former in-line loop used, so the parallel gate stays bit-identical to
    /// the sequential sum while a large body no longer pays one latency-bound
    /// iteration per pixel on a single thread.
    pub(super) force_stage: Option<Handle>,
    pub(super) body_capacity: usize,
    pub(super) pixel_capacity: usize,
    pub(super) owner_capacity: usize,
    pub(super) claims_pixels: usize,
    pub(super) body_count: usize,
    pub(super) pixel_count: usize,
    pub(super) uploaded: bool,
    /// Set by [`Self::note_explicit_sync`] when the engine observed that the
    /// caller read the device back after the previous frame.
    pub(super) explicit_sync: bool,
    /// Per-body signature of the host description last uploaded / read back.
    pub(super) sig: Vec<BodySig>,
}

impl Default for RigidGpu {
    fn default() -> Self {
        Self::new()
    }
}

impl RigidGpu {
    /// Creates an empty device state; device buffers are allocated lazily.
    pub fn new() -> Self {
        Self {
            bodies: None,
            pixels: None,
            claims: None,
            manifolds: None,
            owner: None,
            surface: None,
            force_stage: None,
            body_capacity: 0,
            pixel_capacity: 0,
            owner_capacity: 0,
            claims_pixels: 0,
            body_count: 0,
            pixel_count: 0,
            uploaded: false,
            explicit_sync: false,
            sig: Vec::new(),
        }
    }

    /// Drops every device handle. Called when the canvas or particle pool is
    /// recreated (`resize` / `clear`); the next step re-uploads the host
    /// description.
    pub fn invalidate(&mut self) {
        self.bodies = None;
        self.pixels = None;
        self.claims = None;
        self.manifolds = None;
        self.owner = None;
        self.surface = None;
        self.force_stage = None;
        self.body_capacity = 0;
        self.pixel_capacity = 0;
        self.owner_capacity = 0;
        self.claims_pixels = 0;
        self.body_count = 0;
        self.pixel_count = 0;
        self.uploaded = false;
        self.explicit_sync = false;
        self.sig.clear();
    }

    /// Records that the host shadows were explicitly synchronised after the
    /// previous frame (`ensure_host_current` / save capture / undo snapshot).
    ///
    /// [`crate::SimulationEngine::step`] calls this *before* the kernel passes,
    /// because the passes set `gpu_dirty` again and would otherwise hide the
    /// request. The rigid step then refreshes the host description exactly once.
    pub fn note_explicit_sync(&mut self) {
        self.explicit_sync = true;
    }

    /// `true` when the device holds no rigid state to reconcile.
    pub fn is_idle(&self) -> bool {
        !self.uploaded && self.body_count == 0
    }

    /// Whether the steady-state path currently owns the bodies.
    pub fn is_device_authoritative(&self) -> bool {
        self.uploaded
    }

    /// Pulls the device-resident body description into `bufs.rigid_bodies`.
    ///
    /// Intended for debug / undo / save capture paths that need the live
    /// transforms after calling [`crate::SimBuffers::sync_gpu_to_host`]; the
    /// per-frame pipeline never calls it.
    pub fn sync_host(&mut self, bufs: &mut SimBuffers) {
        if self.uploaded {
            self.refresh_host(bufs);
        }
    }

    /// Runs one rigid-body step.
    ///
    /// * The host description changed (spawn / edit / restore / destroy) or the
    ///   host explicitly synchronised after the previous frame → fall back to
    ///   the host initialiser, which materialises new pixels, sanitises mass and
    ///   reaps orphaned rigid particles. That is the only place this module
    ///   reads particles / maps back.
    /// * Otherwise the device passes run and no buffer crosses the bus.
    pub fn step(&mut self, bufs: &mut SimBuffers, registry: &ElementRegistry, dt: f32) {
        if bufs.rigid_bodies.is_empty() {
            // Every body is gone (fracture, restore, clear): the device passes
            // already detached the pixels, so only the counts need dropping.
            self.uploaded = false;
            self.body_count = 0;
            self.pixel_count = 0;
            self.explicit_sync = false;
            self.sig.clear();
            return;
        }
        // The engine records an explicit host read-back before the kernel
        // passes run, because those passes set `gpu_dirty` again.
        let adopt = self.explicit_sync;
        self.explicit_sync = false;
        // Detect host-side edits (spawn / delete / Lua / save / test mutation)
        // *before* adoption rewrites the host description.
        let host_changed = !self.uploaded || signature(bufs) != self.sig;
        if std::env::var_os("TPT_RIGID_DEBUG").is_some() {
            eprintln!(
                "rigid step adopt={adopt} changed={host_changed} uploaded={} siglen={} bodylen={}",
                self.uploaded,
                self.sig.len(),
                bufs.rigid_bodies.len()
            );
        }
        if adopt && self.uploaded {
            self.refresh_host(bufs);
        }
        if host_changed {
            if bufs.rigid_bodies.is_empty() {
                self.uploaded = false;
                self.body_count = 0;
                self.pixel_count = 0;
                self.sig = signature(bufs);
                return;
            }
            // Body list / description changed: the host initialiser needs
            // authoritative particles, maps and air. This is the only read-back
            // on the rigid path (`spawn`, save restore, Lua edit, destroy).
            bufs.sync_gpu_to_host();
            if self.uploaded && !adopt {
                // Adopt the device pose for every body the host did not touch,
                // so spawning one body does not rewind the others to their
                // upload-time transforms.
                self.refresh_host(bufs);
            }
            super::step_rigid_bodies_host(bufs, registry, dt);
            bufs.sync_host_to_gpu();
            self.upload_from_host(bufs);
            return;
        }

        if self.uploaded && self.body_count > 0 && !bufs.rigid_bodies.is_empty() {
            self.run_passes(bufs, dt);
        }
    }

    fn ensure_capacity(&mut self, client: &Client, bodies: usize, pixels: usize) {
        let mut body_cap = self.body_capacity.max(1);
        while body_cap < bodies {
            body_cap *= 2;
        }
        let mut pixel_cap = self.pixel_capacity.max(1);
        while pixel_cap < pixels {
            pixel_cap *= 2;
        }
        if body_cap != self.body_capacity || self.bodies.is_none() || self.manifolds.is_none() {
            self.body_capacity = body_cap;
            self.bodies = Some(client.empty(body_cap * B_STRIDE * 4));
            self.manifolds = Some(client.empty(body_cap * MANIFOLDS * M_STRIDE * 4));
        }
        if pixel_cap != self.pixel_capacity || self.pixels.is_none() {
            self.pixel_capacity = pixel_cap;
            self.pixels = Some(client.empty(pixel_cap * P_STRIDE * 4));
            // Rewritten wholesale by `upload_from_host`.
            self.surface = Some(client.empty(pixel_cap.div_ceil(32) * 4));
            self.force_stage = Some(client.empty(pixel_cap * 3 * 4));
        }
    }

    pub(super) fn ensure_owner(&mut self, client: &Client, capacity: usize) {
        if self.owner_capacity != capacity || self.owner.is_none() {
            self.owner_capacity = capacity.max(1);
            let handle = client.empty(self.owner_capacity * 4);
            // Slot ownership must start at "free"; the reconcile pass fills in
            // the rigid slots before the first contact pass.
            client.write(&handle, Bytes::from_elems(vec![0u32; self.owner_capacity]));
            self.owner = Some(handle);
        }
    }

    pub(super) fn ensure_claims(&mut self, client: &Client, num_pixels: usize) {
        if self.claims_pixels != num_pixels || self.claims.is_none() {
            self.claims_pixels = num_pixels.max(1);
            let handle = client.empty(self.claims_pixels * 4);
            // The grid must start at `CLAIM_FREE` (fresh device memory is
            // uninitialised) and `clear_claims_kernel` restores that value
            // every frame: the desired-cell `fetch_min` and the fallback
            // compare-exchanges only ever take a cell out of the free state.
            client.write(
                &handle,
                Bytes::from_elems(vec![u32::MAX; self.claims_pixels]),
            );
            self.claims = Some(handle);
        }
    }

    /// Stages the host description into the device body / pixel records.
    fn upload_from_host(&mut self, bufs: &mut SimBuffers) {
        let n = bufs.rigid_bodies.len();
        if n == 0 {
            self.body_count = 0;
            self.pixel_count = 0;
            self.uploaded = false;
            self.sig.clear();
            return;
        }
        let client = bufs.client.clone();
        let total_pixels: usize = bufs.rigid_bodies.iter().map(|b| b.pixels.len()).sum();
        self.ensure_capacity(&client, n, total_pixels.max(1));
        self.ensure_owner(&client, bufs.capacity);
        self.ensure_claims(&client, bufs.config.num_pixels());

        let mut bodies = vec![0u32; self.body_capacity * B_STRIDE];
        let mut pixels = vec![0u32; self.pixel_capacity * P_STRIDE];
        let mut surface = vec![0u32; self.pixel_capacity.div_ceil(32)];

        // Map each body pixel back to the particle slot the host initialiser
        // reserved for it. Matching by cell is exactly what the host sync pass
        // does, and every rasterised pixel sits exactly on its grid coordinate.
        let mut occupancy = super::build_rigid_occupancy(bufs);
        let mut start = 0usize;

        for (bi, body) in bufs.rigid_bodies.iter().enumerate() {
            let bb = bi * B_STRIDE;
            bodies[bb + B_ID] = body.id;
            bodies[bb + B_FLAGS] = if body.active { BF_ACTIVE } else { 0 };
            bodies[bb + B_PIXEL_START] = start as u32;
            bodies[bb + B_PIXEL_COUNT] = body.pixels.len() as u32;
            bodies[bb + B_CM_X] = body.cm_x.to_bits();
            bodies[bb + B_CM_Y] = body.cm_y.to_bits();
            bodies[bb + B_VX] = body.vx.to_bits();
            bodies[bb + B_VY] = body.vy.to_bits();
            bodies[bb + B_ANGLE] = body.angle.to_bits();
            bodies[bb + B_OMEGA] = body.omega.to_bits();
            bodies[bb + B_MASS] = body.mass.to_bits();
            bodies[bb + B_INV_MASS] = body.inv_mass.to_bits();
            bodies[bb + B_INERTIA] = body.inertia.to_bits();
            bodies[bb + B_INV_INERTIA] = body.inv_inertia.to_bits();
            bodies[bb + B_ORIGIN_X] = 0.0f32.to_bits();
            bodies[bb + B_ORIGIN_Y] = 0.0f32.to_bits();
            bodies[bb + B_AX] = 0.0f32.to_bits();
            bodies[bb + B_AY] = 0.0f32.to_bits();
            bodies[bb + B_ANGULAR_ACC] = 0.0f32.to_bits();
            bodies[bb + B_SUBSTEPS] = 1;
            let mut radius2 = 0.0f32;
            for px in &body.pixels {
                let r2 = px.local_x * px.local_x + px.local_y * px.local_y;
                if r2 > radius2 {
                    radius2 = r2;
                }
            }
            bodies[bb + B_RADIUS] = radius2.sqrt().to_bits();
            bodies[bb + B_UNIT_WEIGHT] = (body.mass / body.pixels.len().max(1) as f32)
                .max(1.0)
                .to_bits();
            bodies[bb + B_ALIVE] = body.pixels.len() as u32;
            bodies[bb + B_PREV_CM_X] = body.cm_x.to_bits();
            bodies[bb + B_PREV_CM_Y] = body.cm_y.to_bits();
            bodies[bb + B_PREV_ANGLE] = body.angle.to_bits();
            // The pixel list is new, so the device mass / inertia cache must
            // be rebuilt by the next `prepare_geometry_kernel`.
            bodies[bb + B_GEOM_DIRTY] = 1;

            // Body-local neighbourhood, used by the contact pass to skip the
            // pixels a loose particle can never reach. The indices are stored
            // as `u16`s so a detachment can promote the neighbours; a body
            // above that range keeps every pixel on the surface.
            let tracked = body.pixels.len() <= NO_NEIGHBOR as usize;
            let mut local_index: std::collections::HashMap<(i32, i32), u32> =
                std::collections::HashMap::with_capacity(if tracked {
                    body.pixels.len()
                } else {
                    0
                });
            if tracked {
                for (k, px) in body.pixels.iter().enumerate() {
                    local_index
                        .entry((px.local_x.round() as i32, px.local_y.round() as i32))
                        .or_insert(k as u32);
                }
            }

            for (k, px) in body.pixels.iter().enumerate() {
                let pb = (start + k) * P_STRIDE;
                let pid = super::take_rigid_particle(&mut occupancy, px.grid_x, px.grid_y)
                    .filter(|&id| super::particle_alive(bufs, id))
                    .map_or(NO_PID, |id| id as u32);
                let (interior, neighbor) = pixel_surface_state(
                    &local_index,
                    px.local_x.round() as i32,
                    px.local_y.round() as i32,
                    k,
                );
                if pid != NO_PID && interior == 0 {
                    set_surface_bit(&mut surface, start + k);
                }
                for (slot, word) in [P_NEIGHBORS_0, P_NEIGHBORS_1, P_NEIGHBORS_2, P_NEIGHBORS_3]
                    .into_iter()
                    .enumerate()
                {
                    pixels[pb + word] =
                        neighbor[2 * slot] as u32 | ((neighbor[2 * slot + 1] as u32) << 16);
                }
                pixels[pb + P_LOCAL_X] = px.local_x.to_bits();
                pixels[pb + P_LOCAL_Y] = px.local_y.to_bits();
                pixels[pb + P_ELEM] = px.elem_ty;
                pixels[pb + P_TEMP] = px.temp.to_bits();
                pixels[pb + P_LIFE] = px.life as u32;
                pixels[pb + P_CTYPE] = px.ctype;
                pixels[pb + P_DCOLOUR] = px.dcolour;
                pixels[pb + P_PID] = pid;
                pixels[pb + P_GRID_X] = px.grid_x as u32;
                pixels[pb + P_GRID_Y] = px.grid_y as u32;
                pixels[pb + P_DETACHED] = if pid == NO_PID { 1 } else { 0 };
                pixels[pb + P_BODY] = bi as u32;
            }

            // A pixel that starts out detached (save restore, Lua edit) opens
            // a hole immediately: its neighbours must stay on the surface,
            // exactly like a device-side detachment in `reconcile_kernel`.
            for k in 0..body.pixels.len() {
                let pb = (start + k) * P_STRIDE;
                if pixels[pb + P_DETACHED] == 0 {
                    continue;
                }
                let lo = pixels[pb + P_NEIGHBORS_0];
                let hi = pixels[pb + P_NEIGHBORS_1];
                let third = pixels[pb + P_NEIGHBORS_2];
                let fourth = pixels[pb + P_NEIGHBORS_3];
                for local in [
                    lo & 0xFFFF,
                    lo >> 16,
                    hi & 0xFFFF,
                    hi >> 16,
                    third & 0xFFFF,
                    third >> 16,
                    fourth & 0xFFFF,
                    fourth >> 16,
                ] {
                    if local != NO_NEIGHBOR as u32 {
                        set_surface_bit(&mut surface, start + local as usize);
                    }
                }
            }
            start += body.pixels.len();
        }

        if let Some(handle) = &self.bodies {
            client.write(handle, Bytes::from_elems(bodies));
        }
        if let Some(handle) = &self.pixels {
            client.write(handle, Bytes::from_elems(pixels));
        }
        if let Some(handle) = &self.surface {
            client.write(handle, Bytes::from_elems(surface));
        }
        // The owner map identifies which body a `pmap` slot belongs to. Reset
        // it when the slot assignments can have changed; the reconcile pass
        // repopulates it before the first contact pass.
        if let Some(handle) = &self.owner {
            client.write(handle, Bytes::from_elems(vec![0u32; self.owner_capacity]));
        }
        if std::env::var_os("TPT_RIGID_DEBUG").is_some() {
            eprintln!(
                "upload n={n} pixels={total_pixels} start={start} per-body={:?}",
                bufs.rigid_bodies
                    .iter()
                    .map(|b| b.pixels.len())
                    .collect::<Vec<_>>()
            );
        }
        self.body_count = n;
        self.pixel_count = start;
        self.uploaded = start > 0;
        self.sig = signature(bufs);
    }

    /// Pulls the tiny device body / pixel description into the host bodies.
    ///
    /// Only called when the caller explicitly synchronised after the previous
    /// frame (save capture, undo snapshot, Lua / HUD inspection). The particle
    /// SoA itself is refreshed by [`SimBuffers::ensure_host_current`] or
    /// [`SimBuffers::sync_gpu_to_host`]; this function only restores the
    /// `RigidBody2D` pose and the live pixel list from the device.
    fn refresh_host(&mut self, bufs: &mut SimBuffers) {
        let n = self.body_count;
        let total = self.pixel_count;
        if n == 0 || total == 0 {
            return;
        }
        let Some(bodies_handle) = self.bodies.clone() else {
            return;
        };
        let Some(pixels_handle) = self.pixels.clone() else {
            return;
        };
        let body_trim = ((self.body_capacity - n) * B_STRIDE * 4) as u64;
        let pixel_trim = ((self.pixel_capacity - total) * P_STRIDE * 4) as u64;
        let Ok(readback) = bufs.client.read(vec![
            bodies_handle.offset_end(body_trim),
            pixels_handle.offset_end(pixel_trim),
        ]) else {
            return;
        };
        let body_words = words_of(&readback[0], n * B_STRIDE);
        if std::env::var_os("TPT_RIGID_DEBUG").is_some() {
            for bi in 0..n {
                let bb = bi * B_STRIDE;
                eprintln!(
                    "dev body {bi}: flags={} cm=({}, {}) v=({}, {}) sub={} sup={} scratch={}",
                    body_words[bb + B_FLAGS],
                    f32::from_bits(body_words[bb + B_CM_X]),
                    f32::from_bits(body_words[bb + B_CM_Y]),
                    f32::from_bits(body_words[bb + B_VX]),
                    f32::from_bits(body_words[bb + B_VY]),
                    body_words[bb + B_SUBSTEPS],
                    body_words[bb + B_SUPPORT],
                    body_words[bb + B_SUPPORT_NEXT],
                );
            }
        }
        let pixel_words = words_of(&readback[1], total * P_STRIDE);
        if body_words.len() < n * B_STRIDE || pixel_words.len() < total * P_STRIDE {
            return;
        }

        // Decode the device bodies first so the host list can be matched by id.
        let mut device: Vec<RigidBody2D> = Vec::with_capacity(n);
        for bi in 0..n {
            let bb = bi * B_STRIDE;
            let flags = body_words[bb + B_FLAGS];
            if (flags & BF_ACTIVE) == 0 {
                continue;
            }
            let start = body_words[bb + B_PIXEL_START] as usize;
            let count = body_words[bb + B_PIXEL_COUNT] as usize;
            let origin_x = f32::from_bits(body_words[bb + B_ORIGIN_X]);
            let origin_y = f32::from_bits(body_words[bb + B_ORIGIN_Y]);
            let mut pixel_list: Vec<RigidPixel> = Vec::with_capacity(count);
            for k in 0..count {
                let pi = start + k;
                if pi >= total {
                    break;
                }
                let pb = pi * P_STRIDE;
                if pixel_words[pb + P_DETACHED] != 0 {
                    continue;
                }
                pixel_list.push(RigidPixel {
                    // The device keeps a shifted local origin after a fracture;
                    // the host description is always centre-of-mass relative.
                    local_x: f32::from_bits(pixel_words[pb + P_LOCAL_X]) - origin_x,
                    local_y: f32::from_bits(pixel_words[pb + P_LOCAL_Y]) - origin_y,
                    elem_ty: pixel_words[pb + P_ELEM],
                    temp: f32::from_bits(pixel_words[pb + P_TEMP]),
                    life: pixel_words[pb + P_LIFE] as i32,
                    ctype: pixel_words[pb + P_CTYPE],
                    dcolour: pixel_words[pb + P_DCOLOUR],
                    grid_x: pixel_words[pb + P_GRID_X] as i32,
                    grid_y: pixel_words[pb + P_GRID_Y] as i32,
                });
            }
            if pixel_list.len() < 2 {
                continue;
            }
            device.push(RigidBody2D {
                id: body_words[bb + B_ID],
                active: true,
                cm_x: f32::from_bits(body_words[bb + B_CM_X]),
                cm_y: f32::from_bits(body_words[bb + B_CM_Y]),
                vx: f32::from_bits(body_words[bb + B_VX]),
                vy: f32::from_bits(body_words[bb + B_VY]),
                angle: f32::from_bits(body_words[bb + B_ANGLE]),
                omega: f32::from_bits(body_words[bb + B_OMEGA]),
                mass: f32::from_bits(body_words[bb + B_MASS]),
                inv_mass: f32::from_bits(body_words[bb + B_INV_MASS]),
                inertia: f32::from_bits(body_words[bb + B_INERTIA]),
                inv_inertia: f32::from_bits(body_words[bb + B_INV_INERTIA]),
                pixels: pixel_list,
                materialised: true,
            });
        }

        // Only bodies whose host description still equals the uploaded one are
        // device-authoritative. A spawn (extra entry), a Lua/save edit or a
        // direct test mutation keeps the host description, which the host
        // initialiser then materialises.
        let uploaded = std::mem::take(&mut self.sig);
        let old = std::mem::take(&mut bufs.rigid_bodies);
        let mut kept: Vec<RigidBody2D> = Vec::with_capacity(old.len());
        for (index, mut body) in old.into_iter().enumerate() {
            let untouched = uploaded
                .get(index)
                .is_some_and(|sig| *sig == body_signature(&body));
            match device.iter().find(|dev| dev.id == body.id) {
                Some(dev) if untouched => {
                    // Host description unchanged since the upload: the device is
                    // authoritative for pose and live pixels.
                    kept.push(dev.clone());
                }
                Some(dev) if body.materialised => {
                    // Pose edited by the host (Lua / test / tool) but the body
                    // still owns the device pixels: keep the edited motion state
                    // and adopt the device pixel list / mass properties so the
                    // host initialiser can match every live slot by its current
                    // cell and agrees with the device about the surviving mass.
                    body.pixels = dev.pixels.clone();
                    body.mass = dev.mass;
                    body.inv_mass = dev.inv_mass;
                    body.inertia = dev.inertia;
                    body.inv_inertia = dev.inv_inertia;
                    kept.push(body);
                }
                Some(_) => {
                    // Newly spawned / restored body (`materialised == false`):
                    // the host description is authoritative.
                    kept.push(body);
                }
                None => {
                    if !untouched {
                        kept.push(body);
                    }
                    // Else the body died on the device: drop the host copy.
                }
            }
        }
        bufs.rigid_bodies = kept;
        self.sig = signature(bufs);
    }
}

/// Stable signature of one host-side body description.
///
/// The per-frame steady state leaves the host `RigidBody2D` untouched, so this
/// keeps matching the uploaded signature and no transfer happens. A spawn,
/// delete, Lua/save edit or direct test mutation changes at least one field of
/// the affected body.
#[derive(Clone, PartialEq, Eq)]
pub(super) struct BodySig {
    id: u32,
    active: bool,
    pixels: u32,
    materialised: bool,
    pose: [u32; 7],
    checksum: u32,
}

fn body_signature(body: &RigidBody2D) -> BodySig {
    let mut checksum = 0u32;
    for px in &body.pixels {
        checksum = checksum
            .wrapping_mul(31)
            .wrapping_add(px.grid_x as u32)
            .wrapping_mul(17)
            .wrapping_add(px.grid_y as u32)
            .wrapping_mul(13)
            .wrapping_add(px.elem_ty);
    }
    BodySig {
        id: body.id,
        active: body.active,
        pixels: body.pixels.len() as u32,
        materialised: body.materialised,
        pose: [
            body.cm_x.to_bits(),
            body.cm_y.to_bits(),
            body.angle.to_bits(),
            body.vx.to_bits(),
            body.vy.to_bits(),
            body.omega.to_bits(),
            body.mass.to_bits(),
        ],
        checksum,
    }
}

fn signature(bufs: &SimBuffers) -> Vec<BodySig> {
    bufs.rigid_bodies.iter().map(body_signature).collect()
}

fn words_of(bytes: &[u8], words: usize) -> Vec<u32> {
    let mut out = Vec::with_capacity(words);
    for chunk in bytes.as_chunks::<4>().0.iter().take(words) {
        out.push(u32::from_le_bytes(*chunk));
    }
    out
}

/// Sets the surface bit of device pixel `index` in the packed bitmap.
fn set_surface_bit(surface: &mut [u32], index: usize) {
    surface[index / 32] |= 1u32 << (index % 32);
}

/// Surface classification of one body pixel.
///
/// Returns `(interior, neighbours)`. `interior == 1` means all eight
/// surrounding cells are other pixels of the same body, so the contact pass can
/// skip the pixel; `neighbours` holds the body-local
/// indices in `(-1,-1), (0,-1), (1,-1), (-1,0), (1,0), (-1,1), (0,1), (1,1)`
/// order, [`NO_NEIGHBOR`] where there is no pixel.
fn pixel_surface_state(
    local_index: &std::collections::HashMap<(i32, i32), u32>,
    lx: i32,
    ly: i32,
    index: usize,
) -> (u32, [u16; 8]) {
    let mut neighbor = [NO_NEIGHBOR; 8];
    let mut interior = 1u32;
    let offsets = [
        (-1, -1),
        (0, -1),
        (1, -1),
        (-1, 0),
        (1, 0),
        (-1, 1),
        (0, 1),
        (1, 1),
    ];
    for (slot, (dx, dy)) in offsets.into_iter().enumerate() {
        match local_index.get(&(lx + dx, ly + dy)) {
            Some(&other) if other as usize != index => neighbor[slot] = other as u16,
            _ => interior = 0,
        }
    }
    (interior, neighbor)
}
