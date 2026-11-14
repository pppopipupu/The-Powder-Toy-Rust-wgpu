//! Rigid bismuth crystals grown from a cooling melt.
//!
//! A rigid body is a host description that the rigid solver materialises
//! (`rigid_body::step_rigid_bodies`), so the `BISM` reaction kernel cannot
//! create one by itself. The device instead writes a small ring of spawn
//! requests ([`CRYSTAL_REQUEST_CAP`] records) from the pixels where a slow
//! cooling melt starts to crystallise, and this module drains it.
//!
//! The drain is the one host read on the path: a 13-word async slice every
//! [`POLL_INTERVAL`] ticks, polled exactly like the HUD live count
//! ([`crate::live_count`]). It carries only spawn bookkeeping, never
//! simulation or render state, and an empty ring costs one in-flight 52-byte
//! copy. Everything the body needs after the spawn is device-resident again.

use tpt_core::commands::RigidSpawnShape;
use tpt_core::constants::*;

use crate::SimulationEngine;
use crate::live_count::PendingRead;

/// Records the device ring can hold between two drains.
pub(crate) const CRYSTAL_REQUEST_CAP: usize = 3;
/// Words per record: `x, y, radius, temp_bits`.
pub(crate) const CRYSTAL_REQUEST_RECORD: usize = 4;
/// `[write_sequence, gate_tick, record * CRYSTAL_REQUEST_CAP]`.
pub(crate) const CRYSTAL_REQUEST_WORDS: usize = 2 + CRYSTAL_REQUEST_CAP * CRYSTAL_REQUEST_RECORD;
/// Word offset of the first record.
const CRYSTAL_HEADER: usize = 2;

/// Ticks between two async request reads (half a second at 30 fps).
const POLL_INTERVAL: u64 = 15;
/// Upper bound on bodies produced by melt crystallisation, so a large melt
/// cannot flood the rigid solver.
const MAX_CRYSTAL_BODIES: usize = 24;
/// Smallest / largest crystal radius the kernel may request.
const MIN_RADIUS: i32 = 3;
const MAX_RADIUS: i32 = 5;

/// In-flight async read of the device request ring.
pub(crate) type PendingCrystalRead = PendingRead<Vec<u32>>;

impl SimulationEngine {
    /// Polls the bismuth crystal request ring and spawns the requested rigid
    /// bodies.
    ///
    /// Called from [`Self::step`] before the rigid-body pass, so a body the
    /// previous frame's melt queued shows up in this frame's rasterisation.
    pub fn poll_bismuth_crystals(&mut self) {
        if let Some(pending) = self.bufs.pending_crystal_requests.as_ref()
            && let Some(words) = pending.try_take()
        {
            self.bufs.pending_crystal_requests = None;
            self.consume_crystal_requests(&words);
        }
        if !self.frame_count.is_multiple_of(POLL_INTERVAL)
            || self.bufs.pending_crystal_requests.is_some()
        {
            return;
        }
        let client = self.bufs.client.clone();
        let handle = self.bufs.gpu_crystal_requests.clone();
        let pending = PendingRead::spawn(Box::pin(async move {
            let mut words: Vec<u32> = Vec::new();
            if let Ok(readback) = client.read_async(vec![handle]).await
                && let Some(raw) = readback.first()
            {
                words.reserve(raw.len() / 4);
                for chunk in raw.as_chunks::<4>().0 {
                    words.push(u32::from_le_bytes(*chunk));
                }
            }
            words
        }));
        if let Some(words) = pending.try_take() {
            self.consume_crystal_requests(&words);
        } else {
            self.bufs.pending_crystal_requests = Some(pending);
        }
    }

    /// Spawns one rigid body per new request record.
    fn consume_crystal_requests(&mut self, words: &[u32]) {
        if words.len() < CRYSTAL_REQUEST_WORDS {
            return;
        }
        let total = words[0];
        let seen = self.bufs.crystal_seq_seen;
        self.bufs.crystal_seq_seen = total;
        let new = total.wrapping_sub(seen) as usize;
        if new == 0 {
            return;
        }
        // A full ring between two drains loses the oldest records; only the
        // last `CRYSTAL_REQUEST_CAP` are still readable.
        let count = new.min(CRYSTAL_REQUEST_CAP);
        let xres = self.bufs.config.xres as i32;
        let yres = self.bufs.config.yres as i32;
        for step in 0..count {
            if self.bufs.rigid_bodies.len() >= MAX_CRYSTAL_BODIES {
                break;
            }
            let slot = (total.wrapping_sub(1 + step as u32) as usize) % CRYSTAL_REQUEST_CAP;
            let base = CRYSTAL_HEADER + slot * CRYSTAL_REQUEST_RECORD;
            let x = words[base] as i32;
            let y = words[base + 1] as i32;
            let radius = (words[base + 2] as i32).clamp(MIN_RADIUS, MAX_RADIUS);
            let temp = (words[base + 3] as f32).clamp(300.0, 900.0);
            if x < 0 || y < 0 || x >= xres || y >= yres {
                continue;
            }
            let id = self.spawn_rigid_body(
                x as f32,
                y as f32,
                radius,
                radius,
                RigidSpawnShape::Circle,
                PT_BISM,
            );
            // The crystal froze out of the melt, so it starts at the melt's
            // temperature rather than the element default.
            if let Some(body) = self.bufs.rigid_bodies.iter_mut().find(|body| body.id == id) {
                for pixel in &mut body.pixels {
                    pixel.temp = temp;
                }
            }
        }
    }
}
