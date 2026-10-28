//! Fixed GPU-resident buffers for the 256³ voxel world (plan §3.3).
//!
//! [`Voxel3dBuffers`] owns the allocations that live for the whole 3D session:
//! the A/B voxel state pair, the movement claim buffer, the leaf and coarse
//! occupancy masks, the per-brick bitsets and the compacted simulation list.
//! Air and rigid-body buffers are separate because their sizes depend on the
//! air mode and the body count.
//!
//! Every buffer is word addressed: the kernels only load and store `u32`, so
//! the byte sizes below are always `words * 4`. The voxel arrays are allocated
//! with [`Client::empty`] because they dominate the footprint, while the
//! masks, bitsets and the indirect dispatch arguments are uploaded from zeroed
//! host slices: a non-zero mask would make the renderer descend into empty
//! bricks, so their initial contents are part of the contract.

use core::mem::size_of;

use cubecl::client::Client;
use cubecl::server::Handle;

use tpt_core::voxel3d::{AIR_COUNT, BRICK_COUNT, COARSE_WORD_COUNT, LEAF_WORD_COUNT, VOXEL_COUNT};

/// Bytes per `u32` word.
const WORD: usize = size_of::<u32>();

/// Words in the coarse occupancy pyramid.
///
/// The six levels hold 4³, 8³, 16³, 32³, 64³ and 128³ block occupancy bits:
/// `8192 + 1024 + 128 + 16 + 2 + 1 = 9363` words, about 36 KiB.
pub const COARSE_MASK_WORDS: usize = 9_363;

/// Words in one bit-per-brick grid (`BRICK_COUNT` bits, 4 KiB).
pub const BRICK_BIT_WORDS: usize = 1_024;

/// Words in the indirect dispatch arguments: `[x, y, z]` workgroup counts.
pub const SIM_COUNT_WORDS: usize = 3;

/// Number of records the creation/mutation queue holds.
pub const MUTATION_CAPACITY: usize = 65_535;
/// `u32` words per mutation record.
pub const MUTATION_STRIDE: usize = 8;
/// Bitset words for the TNT flood visited set (`VOXEL_COUNT` bits).
pub const FLOOD_MARK_WORDS: usize = VOXEL_COUNT / 32;

const _: () = assert!(BRICK_BIT_WORDS * 32 == BRICK_COUNT);
const _: () = assert!(BRICK_COUNT == 32_768);
const _: () = assert!(VOXEL_COUNT == 16_777_216);
const _: () = assert!(LEAF_WORD_COUNT == 65_536);
const _: () = assert!(COARSE_MASK_WORDS == COARSE_WORD_COUNT);
const _: () = assert!(FLOOD_MARK_WORDS * 32 == VOXEL_COUNT);

/// The persistent GPU buffers of one 3D voxel world.
///
/// All voxel arrays are `u32` words (plan §0.1): `voxel_a` is the authoritative
/// state, `voxel_b` the per-tick scratch copy and `claim` the `atomicMax`
/// arbitration buffer. The masks follow §3.4; `sim_count`/`sim_list` are the
/// indirect dispatch pair described in §4.1.
pub struct Voxel3dBuffers {
    /// CubeCL client that allocated the buffers and launches the kernels.
    pub client: Client,
    /// Authoritative voxel state, `VOXEL_COUNT * 4` bytes.
    pub voxel_a: Handle,
    /// Per-tick scratch state, `VOXEL_COUNT * 4` bytes.
    pub voxel_b: Handle,
    /// Movement claims (`owner | tag << 24`), `VOXEL_COUNT * 4` bytes.
    pub claim: Handle,
    /// One bit per 2³ block, `LEAF_WORD_COUNT * 4` bytes (256 KiB).
    pub leaf_masks: Handle,
    /// Occupancy bits for the 4³ … 128³ block levels, 9363 words.
    pub coarse_masks: Handle,
    /// One bit per 8³ brick that contains at least one voxel.
    pub brick_occupied: Handle,
    /// Snapshot of `brick_occupied` taken at the start of the tick.
    pub active_bits: Handle,
    /// Previous tick's `active_bits`, so a brick that just emptied is still
    /// processed once and its A/B copies are both cleared.
    pub prev_active_bits: Handle,
    /// `active_bits | prev_active_bits`: the bricks this tick actually runs.
    pub sim_bits: Handle,
    /// Bricks whose masks must be rebuilt after a direct `voxel_a` write.
    pub dirty_bricks: Handle,
    /// Compacted list of active brick indices, `BRICK_COUNT * 4` bytes.
    pub sim_list: Handle,
    /// Indirect dispatch arguments `[x, y, z]` for the brick passes.
    pub sim_count: Handle,
    /// TPT `ctype` per voxel (`VOXEL_COUNT * 4` bytes).
    pub aux_ctype: Handle,
    /// TPT `tmp` per voxel.
    pub aux_tmp: Handle,
    /// TPT `tmp2` per voxel.
    pub aux_tmp2: Handle,
    /// TPT `tmp3` per voxel.
    pub aux_tmp3: Handle,
    /// `atomicMax` arbitration for the mutation drain (`VOXEL_COUNT` words).
    pub spawn_claim: Handle,
    /// Mutation records, `MUTATION_CAPACITY * MUTATION_STRIDE` words.
    pub mutation_queue: Handle,
    /// Indirect record count `[x, 1, 1]` for the mutation drain.
    pub mutation_count: Handle,
    /// TNT flood visited bitset (`FLOOD_MARK_WORDS` words).
    pub flood_marks: Handle,
    /// TNT flood BFS queue (`VOXEL_COUNT` voxel indices).
    pub flood_queue: Handle,
    /// TNT flood queue head/tail (`[head, tail]`).
    pub flood_state: Handle,
    /// Newtonian-gravity mass per 4³ cell, `i32` fixed point
    /// ([`super::gravity::MASS_SCALE`] units per TPT mass unit).
    pub grav_mass: Handle,
    /// Gravitational potential solved from [`Self::grav_mass`], per 4³ cell.
    pub grav_potential: Handle,
    /// Scratch partner of `grav_potential` for the Jacobi solve.
    pub grav_potential_tmp: Handle,
    /// Newtonian force along x per 4³ cell, in TPT gravity units.
    pub grav_force_x: Handle,
    /// Newtonian force along y per 4³ cell.
    pub grav_force_y: Handle,
    /// Newtonian force along z per 4³ cell.
    pub grav_force_z: Handle,
}

impl Voxel3dBuffers {
    /// Returns the world to its freshly allocated state.
    ///
    /// The three large voxel arrays are zeroed on the device by
    /// [`Self::clear_world`]; the masks, bitsets and dispatch arguments are
    /// small enough to re-upload as zeroed pages. This is a user-triggered
    /// reset, never part of the tick path.
    pub fn reset(&mut self) {
        self.leaf_masks = zeroed_words(&self.client, LEAF_WORD_COUNT);
        self.coarse_masks = zeroed_words(&self.client, COARSE_MASK_WORDS);
        self.brick_occupied = zeroed_words(&self.client, BRICK_BIT_WORDS);
        self.active_bits = zeroed_words(&self.client, BRICK_BIT_WORDS);
        self.prev_active_bits = zeroed_words(&self.client, BRICK_BIT_WORDS);
        self.sim_bits = zeroed_words(&self.client, BRICK_BIT_WORDS);
        self.dirty_bricks = zeroed_words(&self.client, BRICK_BIT_WORDS);
        self.sim_count = zeroed_words(&self.client, SIM_COUNT_WORDS);
        // The voxel-sized arrays keep their handles: `clear_world` below zeroes
        // every one of them that a pass can read, and `mutation_count` /
        // `flood_state` are recreated cleared, so their queues are never
        // indexed. Reallocating them only churned ~450 MiB of device memory.
        self.mutation_count = zeroed_words(&self.client, SIM_COUNT_WORDS);
        self.flood_marks = zeroed_words(&self.client, FLOOD_MARK_WORDS);
        self.flood_state = zeroed_words(&self.client, 2);
        self.grav_mass = zeroed_words(&self.client, AIR_COUNT);
        self.grav_potential = zeroed_words(&self.client, AIR_COUNT);
        self.grav_potential_tmp = zeroed_words(&self.client, AIR_COUNT);
        self.grav_force_x = zeroed_words(&self.client, AIR_COUNT);
        self.grav_force_y = zeroed_words(&self.client, AIR_COUNT);
        self.grav_force_z = zeroed_words(&self.client, AIR_COUNT);
        self.clear_world();
    }

    /// Allocates the full 256³ buffer set on `client`.
    ///
    /// This reserves about 199 MiB of device memory (three 64 MiB voxel arrays
    /// plus the masks, lists and the six 1 MiB gravity fields). Allocation
    /// failures surface on the first kernel launch or read-back instead of
    /// panicking here, which matches how the rest of the simulation allocates
    /// its handles.
    pub fn new(client: Client) -> Self {
        let voxel_bytes = VOXEL_COUNT * WORD;

        let buffers = Self {
            client: client.clone(),
            voxel_a: client.empty(voxel_bytes),
            voxel_b: client.empty(voxel_bytes),
            claim: client.empty(voxel_bytes),
            leaf_masks: zeroed_words(&client, LEAF_WORD_COUNT),
            coarse_masks: zeroed_words(&client, COARSE_MASK_WORDS),
            brick_occupied: zeroed_words(&client, BRICK_BIT_WORDS),
            active_bits: zeroed_words(&client, BRICK_BIT_WORDS),
            prev_active_bits: zeroed_words(&client, BRICK_BIT_WORDS),
            sim_bits: zeroed_words(&client, BRICK_BIT_WORDS),
            dirty_bricks: zeroed_words(&client, BRICK_BIT_WORDS),
            sim_list: client.empty(BRICK_COUNT * WORD),
            sim_count: zeroed_words(&client, SIM_COUNT_WORDS),
            aux_ctype: client.empty(voxel_bytes),
            aux_tmp: client.empty(voxel_bytes),
            aux_tmp2: client.empty(voxel_bytes),
            aux_tmp3: client.empty(voxel_bytes),
            spawn_claim: client.empty(voxel_bytes),
            mutation_queue: client.empty(MUTATION_CAPACITY * MUTATION_STRIDE * WORD),
            mutation_count: zeroed_words(&client, SIM_COUNT_WORDS),
            flood_marks: zeroed_words(&client, FLOOD_MARK_WORDS),
            flood_queue: client.empty(voxel_bytes),
            flood_state: zeroed_words(&client, 2),
            grav_mass: zeroed_words(&client, AIR_COUNT),
            grav_potential: zeroed_words(&client, AIR_COUNT),
            grav_potential_tmp: zeroed_words(&client, AIR_COUNT),
            grav_force_x: zeroed_words(&client, AIR_COUNT),
            grav_force_y: zeroed_words(&client, AIR_COUNT),
            grav_force_z: zeroed_words(&client, AIR_COUNT),
        };
        buffers.clear_world();
        buffers
    }
}

/// Uploads `words` zeroed `u32`s.
///
/// Masks and dispatch arguments must start clear, and [`Client::empty`] does
/// not zero recycled device memory, so these small buffers go through
/// [`Client::create_from_slice`] instead.
fn zeroed_words(client: &Client, words: usize) -> Handle {
    let zeros = vec![0u32; words];
    client.create_from_slice(bytemuck::cast_slice(&zeros))
}
