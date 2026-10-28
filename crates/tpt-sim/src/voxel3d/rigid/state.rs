//! Device-resident state and tick orchestration of the 3D rigid-body solver.
//!
//! [`Rigid3dState`] owns every buffer the solver touches: the 64 body records,
//! the 64 x 32^3 voxel atlas, the local occupancy masks, the per-body surface
//! lists and the shared fixed-point contact accumulators. Spawning uploads a
//! full snapshot; after that [`Rigid3dState::step`] only launches kernels and
//! never reads a byte back (plan §7.5).
//!
//! # Public data layout
//!
//! * `body_state`: 32 `f32` per body.
//!   `[0..4]` position, `[4..8]` quaternion, `[8..12]` linear velocity,
//!   `[12..16]` angular velocity, `[16]` inverse mass, `[17..20]` inverse
//!   inertia diagonal, `[20..24]` local AABB minimum, `[24..28]` local AABB
//!   maximum, `[28]` quiet-tick accumulator, `[29]` awake flag, `[30]`
//!   bounding radius, `[31]` padding.
//! * `body_desc`: 40 `f32` per body; `[0..16]` column-major `model`, `[16..32]`
//!   column-major `inv_model`, `[32..36]` world AABB minimum and `[36..40]`
//!   world AABB maximum (both xyz plus a padding component). The model maps
//!   local grid coordinates `[0, 32]` to world coordinates.
//! * `contact_accum`: 16 `i32` per body; xyz linear impulse, xyz angular
//!   impulse and xyz position correction in Q12 units, then the contact count.
//! * `body_voxels`: `MAX_BODIES * 32^3` packed local voxels.
//! * `body_masks`: `MAX_BODIES * 16^3` occupancy bits (128 words per body).

use cubecl::bytes::Bytes;
use cubecl::client::Client;
use cubecl::prelude::*;
use cubecl::server::Handle;

use tpt_core::voxel3d::{
    BODY_ATLAS_VOXELS, BODY_VOXELS, BRICK_COUNT, MAX_BODIES, VOXEL_COUNT, blocked_claim_key,
    move_tag,
};

use crate::voxel3d::Voxel3dBuffers;

use super::host::RigidBody3d;
use super::kernels::{
    apply_impulses_kernel, body_block_claim_kernel, displace_loose_kernel, integrate_kernel,
    pair_contacts_kernel, world_contacts_kernel,
};

/// Number of `f32` words in one `body_state` record.
pub const STATE_STRIDE: usize = 32;
/// Number of `f32` words in one `body_desc` record.
pub const DESC_STRIDE: usize = 40;
/// Number of `i32` words in one contact accumulator.
pub const ACCUM_STRIDE: usize = 16;
/// Number of `u32` words in one body's 16^3 leaf mask.
pub const BODY_MASK_WORDS: usize = (16 * 16 * 16) / 32;
/// Number of `u32` words in one bitset of 32,768 bricks.
pub const BRICK_BIT_WORDS: usize = BRICK_COUNT / 32;
/// Number of physics substeps per tick.
pub const SUBSTEPS: usize = 4;
/// Ticks per second the substep length is derived from.
pub const TPS: f32 = 60.0;

// The kernels spell these values as literals; keep the host copies in sync.
const _: () = assert!(STATE_STRIDE == 32);
const _: () = assert!(DESC_STRIDE == 40);
const _: () = assert!(ACCUM_STRIDE == 16);
const _: () = assert!(BODY_MASK_WORDS == 128);
const _: () = assert!(BODY_VOXELS == 32_768);
const _: () = assert!(BODY_ATLAS_VOXELS == MAX_BODIES * BODY_VOXELS);
const _: () = assert!(BRICK_BIT_WORDS == 1_024);

/// Gravity used by the v1 3D sandbox (plan §3.1).
const GRAVITY: [f32; 3] = [0.0, -9.81, 0.0];

/// The complete device state of the 3D rigid-body solver.
pub struct Rigid3dState {
    /// Number of live bodies.
    pub body_count: u32,
    /// 32 `f32` per body, see the module-level layout.
    pub body_state: Handle,
    /// `MAX_BODIES * 32^3` packed local voxels.
    pub body_voxels: Handle,
    /// `MAX_BODIES * 128` local leaf-mask words.
    pub body_masks: Handle,
    /// Shared Q12 contact accumulator, 16 `i32` per body.
    pub contact_accum: Handle,
    /// 40 `f32` render description per body.
    pub body_desc: Handle,
    /// Per-body surface voxel lists, `MAX_BODIES * 32^3` local indices.
    surface: Handle,
    /// Number of valid entries in `surface` for each body.
    surface_count: Handle,
    /// Host-side spawn descriptions; not read by the steady-state path.
    bodies: Vec<RigidBody3d>,
}

impl Rigid3dState {
    /// Allocates the empty device state.
    pub fn new(client: &Client) -> Self {
        Self {
            body_count: 0,
            body_state: zeroed_f32(client, MAX_BODIES * STATE_STRIDE),
            body_voxels: zeroed_u32(client, BODY_ATLAS_VOXELS),
            body_masks: zeroed_u32(client, MAX_BODIES * BODY_MASK_WORDS),
            contact_accum: zeroed_i32(client, MAX_BODIES * ACCUM_STRIDE),
            body_desc: zeroed_f32(client, MAX_BODIES * DESC_STRIDE),
            surface: zeroed_u32(client, BODY_ATLAS_VOXELS),
            surface_count: zeroed_u32(client, MAX_BODIES),
            bodies: Vec::new(),
        }
    }

    /// Materialises and uploads one axis-aligned box body.
    ///
    /// `center` is the world-space centre, `half_extents` the voxel half
    /// extents (`2 * half_extents` solid voxels per axis) and `ty` the Powder
    /// Toy element id. Returns the body slot.
    pub fn spawn_box(
        &mut self,
        client: &Client,
        center: [f32; 3],
        half_extents: [u32; 3],
        ty: u32,
    ) -> usize {
        assert!(
            self.bodies.len() < MAX_BODIES,
            "the 3D rigid-body atlas only holds {MAX_BODIES} bodies"
        );
        // Spawning a body mid-simulation must not rewind the poses of the
        // bodies that are already moving. The device state is authoritative,
        // so snapshot it before the upload and restore the existing rows. This
        // is a spawn-time transfer, not part of the tick path.
        let preserve = if self.bodies.is_empty() {
            (None, None)
        } else {
            let raw_state = client.read_one_unchecked(self.body_state.clone());
            let raw_desc = client.read_one_unchecked(self.body_desc.clone());
            let state: Vec<f32> = bytemuck::cast_slice(&raw_state).to_vec();
            let desc: Vec<f32> = bytemuck::cast_slice(&raw_desc).to_vec();
            (Some(state), Some(desc))
        };

        self.bodies
            .push(RigidBody3d::new_box(center, half_extents, ty));
        self.body_count = self.bodies.len() as u32;
        let new_index = self.bodies.len() - 1;
        self.upload(
            client,
            preserve.0.as_deref(),
            preserve.1.as_deref(),
            new_index,
        );
        new_index
    }

    /// Handle of the render description, for the 3D renderer binding.
    pub fn body_desc_handle(&self) -> Handle {
        self.body_desc.clone()
    }

    /// Number of live bodies.
    pub fn body_count(&self) -> u32 {
        self.body_count
    }

    /// Host spawn description of a spawned body.
    ///
    /// The pose fields are the spawn-time values; once the body has stepped,
    /// read `body_state` instead of this description.
    pub fn body(&self, index: usize) -> Option<&RigidBody3d> {
        self.bodies.get(index)
    }

    /// Runs one simulation tick.
    ///
    /// Four substeps of `integrate -> world_contacts -> pair_contacts ->
    /// apply_impulses` are followed by one `displace_loose` and one
    /// `body_block_claim` pass (plan §4.1, §7.3).
    pub fn step(&mut self, client: &Client, bufs: &Voxel3dBuffers, tick: u64) {
        let count = self.body_count;
        if count == 0 {
            return;
        }
        let substeps = SUBSTEPS as u32;
        let dt = 1.0 / (TPS * substeps_f32());
        let quiet_step = 1.0 / substeps_f32();

        for _ in 0..substeps {
            unsafe {
                integrate_kernel::launch_unchecked(
                    client,
                    CubeCount::Static(1, 1, 1),
                    CubeDim::new_1d(64),
                    BufferArg::from_raw_parts(self.body_state.clone(), MAX_BODIES * STATE_STRIDE),
                    BufferArg::from_raw_parts(self.body_desc.clone(), MAX_BODIES * DESC_STRIDE),
                    count,
                    dt,
                    GRAVITY[0],
                    GRAVITY[1],
                    GRAVITY[2],
                );
                world_contacts_kernel::launch_unchecked(
                    client,
                    CubeCount::Static(count, 1, 1),
                    CubeDim::new_1d(256),
                    BufferArg::from_raw_parts(self.body_state.clone(), MAX_BODIES * STATE_STRIDE),
                    BufferArg::from_raw_parts(self.surface.clone(), BODY_ATLAS_VOXELS),
                    BufferArg::from_raw_parts(self.surface_count.clone(), MAX_BODIES),
                    BufferArg::from_raw_parts(bufs.voxel_a.clone(), VOXEL_COUNT),
                    BufferArg::from_raw_parts(
                        self.contact_accum.clone(),
                        MAX_BODIES * ACCUM_STRIDE,
                    ),
                    count,
                );
                if count >= 2 {
                    pair_contacts_kernel::launch_unchecked(
                        client,
                        CubeCount::Static(count, 1, 1),
                        CubeDim::new_1d(64),
                        BufferArg::from_raw_parts(
                            self.body_state.clone(),
                            MAX_BODIES * STATE_STRIDE,
                        ),
                        BufferArg::from_raw_parts(self.body_desc.clone(), MAX_BODIES * DESC_STRIDE),
                        BufferArg::from_raw_parts(self.body_voxels.clone(), BODY_ATLAS_VOXELS),
                        BufferArg::from_raw_parts(self.surface.clone(), BODY_ATLAS_VOXELS),
                        BufferArg::from_raw_parts(self.surface_count.clone(), MAX_BODIES),
                        BufferArg::from_raw_parts(
                            self.contact_accum.clone(),
                            MAX_BODIES * ACCUM_STRIDE,
                        ),
                        count,
                    );
                }
                apply_impulses_kernel::launch_unchecked(
                    client,
                    CubeCount::Static(1, 1, 1),
                    CubeDim::new_1d(64),
                    BufferArg::from_raw_parts(self.body_state.clone(), MAX_BODIES * STATE_STRIDE),
                    BufferArg::from_raw_parts(self.body_desc.clone(), MAX_BODIES * DESC_STRIDE),
                    BufferArg::from_raw_parts(
                        self.contact_accum.clone(),
                        MAX_BODIES * ACCUM_STRIDE,
                    ),
                    count,
                    quiet_step,
                );
            }
        }

        let blocked_key = blocked_claim_key(move_tag(tick));
        unsafe {
            displace_loose_kernel::launch_unchecked(
                client,
                CubeCount::Static(count, 1, 1),
                CubeDim::new_1d(256),
                BufferArg::from_raw_parts(bufs.voxel_a.clone(), VOXEL_COUNT),
                BufferArg::from_raw_parts(bufs.aux_ctype.clone(), VOXEL_COUNT),
                BufferArg::from_raw_parts(bufs.aux_tmp.clone(), VOXEL_COUNT),
                BufferArg::from_raw_parts(bufs.aux_tmp2.clone(), VOXEL_COUNT),
                BufferArg::from_raw_parts(bufs.aux_tmp3.clone(), VOXEL_COUNT),
                BufferArg::from_raw_parts(self.body_voxels.clone(), BODY_ATLAS_VOXELS),
                BufferArg::from_raw_parts(self.body_state.clone(), MAX_BODIES * STATE_STRIDE),
                BufferArg::from_raw_parts(bufs.brick_occupied.clone(), BRICK_BIT_WORDS),
                BufferArg::from_raw_parts(bufs.dirty_bricks.clone(), BRICK_BIT_WORDS),
                count,
            );
            body_block_claim_kernel::launch_unchecked(
                client,
                CubeCount::Static(count, 1, 1),
                CubeDim::new_1d(256),
                BufferArg::from_raw_parts(self.body_voxels.clone(), BODY_ATLAS_VOXELS),
                BufferArg::from_raw_parts(self.body_state.clone(), MAX_BODIES * STATE_STRIDE),
                BufferArg::from_raw_parts(bufs.claim.clone(), VOXEL_COUNT),
                count,
                blocked_key,
            );
        }
    }

    /// Rebuilds and uploads every host-facing buffer.
    ///
    /// When `preserve_state` / `preserve_desc` are set, they hold the previous
    /// device rows of the bodies that already existed; only `new_index` is
    /// rewritten from the host description. The voxel atlas, masks and surface
    /// lists always come from the host because they are immutable in v1.
    fn upload(
        &self,
        client: &Client,
        preserve_state: Option<&[f32]>,
        preserve_desc: Option<&[f32]>,
        new_index: usize,
    ) {
        let mut state = vec![0.0f32; MAX_BODIES * STATE_STRIDE];
        let mut desc = vec![0.0f32; MAX_BODIES * DESC_STRIDE];
        let preserve = preserve_state.is_some();
        if let Some(previous) = preserve_state {
            let len = previous.len().min(state.len());
            state[..len].copy_from_slice(&previous[..len]);
        }
        if let Some(previous) = preserve_desc {
            let len = previous.len().min(desc.len());
            desc[..len].copy_from_slice(&previous[..len]);
        }
        let mut voxels = vec![0u32; BODY_ATLAS_VOXELS];
        let mut masks = vec![0u32; MAX_BODIES * BODY_MASK_WORDS];
        let mut surface = vec![0u32; BODY_ATLAS_VOXELS];
        let mut surface_counts = vec![0u32; MAX_BODIES];

        for (index, body) in self.bodies.iter().enumerate() {
            let state_base = index * STATE_STRIDE;
            let desc_base = index * DESC_STRIDE;
            let voxel_base = index * BODY_VOXELS;
            let mask_base = index * BODY_MASK_WORDS;
            if !preserve || index == new_index {
                body.write_state(&mut state, state_base);
                body.write_desc(&mut desc, desc_base);
            }
            voxels[voxel_base..voxel_base + BODY_VOXELS].copy_from_slice(&body.local);
            let mask = body.mask_words();
            masks[mask_base..mask_base + BODY_MASK_WORDS].copy_from_slice(&mask);
            let list = body.surface_indices();
            surface[voxel_base..voxel_base + list.len()].copy_from_slice(&list);
            surface_counts[index] = list.len() as u32;
        }

        client.write(&self.body_state, Bytes::from_elems(state));
        client.write(&self.body_desc, Bytes::from_elems(desc));
        client.write(&self.body_voxels, Bytes::from_elems(voxels));
        client.write(&self.body_masks, Bytes::from_elems(masks));
        client.write(&self.surface, Bytes::from_elems(surface));
        client.write(&self.surface_count, Bytes::from_elems(surface_counts));
        client.write(
            &self.contact_accum,
            Bytes::from_elems(vec![0i32; MAX_BODIES * ACCUM_STRIDE]),
        );
    }
}

/// `SUBSTEPS` as `f32`, used to derive the substep length.
fn substeps_f32() -> f32 {
    SUBSTEPS as f32
}

fn zeroed_f32(client: &Client, words: usize) -> Handle {
    let handle = client.empty(words * core::mem::size_of::<f32>());
    client.write(&handle, Bytes::from_elems(vec![0.0f32; words]));
    handle
}

fn zeroed_u32(client: &Client, words: usize) -> Handle {
    let handle = client.empty(words * core::mem::size_of::<u32>());
    client.write(&handle, Bytes::from_elems(vec![0u32; words]));
    handle
}

fn zeroed_i32(client: &Client, words: usize) -> Handle {
    let handle = client.empty(words * core::mem::size_of::<i32>());
    client.write(&handle, Bytes::from_elems(vec![0i32; words]));
    handle
}
