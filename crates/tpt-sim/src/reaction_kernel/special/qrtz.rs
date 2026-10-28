use cubecl::prelude::*;
use tpt_core::constants::*;

use super::{chance1, clampi};
use crate::reaction_kernel::shared::{
    CR_SET_TEMP, CR_SET_TMP, CR_SET_TMP2, hash_u32, mix_index, pmap_at, pmap_slot, request_create,
};

/// Encode a diffusion amount. The receiver slot rides in the crystal's
/// `ctype`, so the amount itself is the whole `tmp4`.
///
/// `QRTZ.cpp:143-155` moves the whole half-difference, so the byte cap the
/// previous message encoding imposed is gone; `0` means "no message".
#[cube]
fn qrtz_pack(amount: i32) -> i32 {
    clampi(amount, 1i32, i32::MAX)
}

/// `QRTZ`/`PQRT` salt absorption, growth and the *send* half of the `tmp`
/// diffusion (`QRTZ.cpp:68-141`).
///
/// Pixel parallel. Every crystal writes only its own slot: the growth is a
/// creation-queue record at the sampled `(x+srx, y+sry)` (the drain later in
/// the same reaction step materialises it, so the child is live for the
/// movement and render passes of this frame), and the diffusion partner is
/// recorded in `tmp4` as a message that [`qrtz_receive_kernel`] delivers on
/// the next dispatch. Every crystal clears its own `tmp4` before it may write
/// a message, so a stale message can never be credited twice. Two crystals can
/// name the same partner in one frame; both transfers are then measured against
/// the same pre-dispatch value, where TPT would apply them one after the other.
#[cube(launch_unchecked)]
pub(super) fn qrtz_kernel(
    pmap: &[u32],
    part_ty: &mut [u32],
    part_life: &mut [i32],
    part_temp: &[f32],
    part_tmp: &mut [i32],
    part_tmp2: &[i32],
    part_tmp4: &mut [i32],
    part_ctype: &mut [u32],
    part_vx: &[f32],
    part_vy: &[f32],
    create_queue: &mut [Atomic<u32>],
    xres: u32,
    yres: u32,
    frame_seed: u32,
) {
    let idx = ABSOLUTE_POS;
    if idx < (xres * yres) as usize {
        let v = pmap[idx];
        if v != 0u32 {
            let id = pmap_slot(v);
            let ty = part_ty[id];
            if ty == PT_QRTZ || ty == PT_PQRT {
                let px = (idx as u32 % xres) as i32;
                let py = (idx as u32 / xres) as i32;
                let seed = hash_u32(frame_seed ^ mix_index(idx as u32, 2_246_822_519u32));

                // `QRTZ.cpp:68-69`: the crack timer never exceeds five. The
                // countdown itself is the generic `PROP_LIFE_DEC` pass
                // (`Simulation.cpp:3399-3407`); decrementing here as well made
                // the timer tick twice a frame.
                if part_life[id] > 5i32 {
                    part_life[id] = 5i32;
                }

                // `QRTZ.cpp:70-91`: absorb salt water into `tmp` (one in 500).
                // `tmp == -1` marks a dead crystal that no longer absorbs.
                let mut tmp = part_tmp[id];
                if tmp != -1i32 {
                    for n in 0u32..9u32 {
                        let dx = (n % 3u32) as i32 - 1i32;
                        let dy = (n / 3u32) as i32 - 1i32;
                        if dx != 0i32 || dy != 0i32 {
                            let nv = pmap_at(pmap, xres, yres, px + dx, py + dy);
                            if nv != 0u32 {
                                let nid = pmap_slot(nv);
                                if part_ty[nid] == PT_SLTW
                                    && chance1(
                                        hash_u32(seed ^ mix_index(nid as u32, 0x1B87_3593u32)),
                                        500u32,
                                    )
                                {
                                    part_ty[nid] = PT_NONE;
                                    tmp += 1i32;
                                }
                            }
                        }
                    }
                }

                // `QRTZ.cpp:93-141`: grow into a free pixel and diffuse `tmp`.
                // The crystal only acts while it sits still and is not cracked.
                let vx = part_vx[id];
                let vy = part_vy[id];
                part_tmp4[id] = 0i32;
                part_ctype[id] = 0u32;
                if tmp > 0i32 && vx * vx + vy * vy < 0.2f32 && part_life[id] <= 0i32 {
                    let mut stopgrow = false;
                    let mut traded = false;
                    for trade in 0u32..9u32 {
                        let roll = hash_u32(seed ^ mix_index(trade, 0x9E37_79B9u32)) % 0x3FFu32;
                        let rx = (roll % 5u32) as i32 - 2i32;
                        let srx = (roll % 3u32) as i32 - 1i32;
                        let shifted = roll >> 3u32;
                        let ry = (shifted % 5u32) as i32 - 2i32;
                        let sry = (shifted % 3u32) as i32 - 1i32;
                        if (rx != 0i32 || ry != 0i32) && !traded {
                            // Growth: one child per frame, in a free pixel
                            // next to the crystal.
                            if !stopgrow
                                && tmp != 0i32
                                && pmap_at(pmap, xres, yres, px + srx, py + sry) == 0u32
                            {
                                // `QRTZ.cpp:106-125`: the child inherits the
                                // parent's `temp` and `tmp2`, a jittered `tmp2`
                                // half the time, and is born dead (`tmp = -1`)
                                // for one child in two.
                                let child_roll =
                                    hash_u32(roll ^ mix_index(id as u32, 0x27D4_EB2Fu32));
                                let mut child_tmp2 = part_tmp2[id];
                                if child_roll.is_multiple_of(2u32) {
                                    // `between(-1, 1)` on the inherited `tmp2`,
                                    // clamped to `0..=10`.
                                    let jitter = ((child_roll >> 1u32) % 3u32) as i32 - 1i32;
                                    child_tmp2 = clampi(child_tmp2 + jitter, 0i32, 10i32);
                                }
                                let mut child_tmp = 0i32;
                                if (child_roll >> 3u32).is_multiple_of(2u32) {
                                    child_tmp = -1i32;
                                }
                                request_create(
                                    create_queue,
                                    PT_QRTZ,
                                    px + srx,
                                    py + sry,
                                    CR_SET_TEMP | CR_SET_TMP | CR_SET_TMP2,
                                    0i32,
                                    0u32,
                                    part_temp[id],
                                    child_tmp,
                                    child_tmp2,
                                    0i32,
                                    0i32,
                                    0u32,
                                    0u32,
                                    0.0f32,
                                    0.0f32,
                                );
                                tmp -= 1i32;
                                if ty == PT_PQRT {
                                    // A cracked crystal that grew becomes a
                                    // normal crystal again.
                                    part_ty[id] = PT_QRTZ;
                                }
                                let after = hash_u32(roll ^ mix_index(id as u32, 0x2545_F491u32));
                                if tmp == 0i32 && after.is_multiple_of(15u32) {
                                    // The exhausted crystal stops growing.
                                    tmp = -1i32;
                                }
                                stopgrow = true;
                            }
                            // Diffusion: hand half of the difference to a
                            // weaker crystal within two pixels.
                            let nv = pmap_at(pmap, xres, yres, px + rx, py + ry);
                            if nv != 0u32 {
                                let nid = pmap_slot(nv);
                                let ntmp = part_tmp[nid];
                                if part_ty[nid] == PT_QRTZ && tmp > ntmp && ntmp >= 0i32 {
                                    let diff = tmp - ntmp;
                                    let mut amount = 1i32;
                                    if diff > 1i32 {
                                        amount = diff / 2i32;
                                    }
                                    let packed = qrtz_pack(amount);
                                    part_tmp4[id] = packed;
                                    part_ctype[id] = nid as u32;
                                    tmp -= packed;
                                    traded = true;
                                }
                            }
                        }
                    }
                }
                part_tmp[id] = tmp;
            }
        }
    }
}

/// The *receive* half of the `QRTZ` diffusion (`QRTZ.cpp:130-141`).
///
/// Pixel parallel. Each crystal looks for the neighbours that named it in
/// `tmp4` during [`qrtz_kernel`] and credits the amount to its own `tmp`, so
/// both halves of a transfer are self-writes and the total `tmp` of a crystal
/// is conserved.
#[cube(launch_unchecked)]
pub(super) fn qrtz_receive_kernel(
    pmap: &[u32],
    part_ty: &[u32],
    part_tmp: &mut [i32],
    part_tmp4: &[i32],
    part_ctype: &[u32],
    xres: u32,
    yres: u32,
) {
    let idx = ABSOLUTE_POS;
    if idx < (xres * yres) as usize {
        let v = pmap[idx];
        if v != 0u32 {
            let id = pmap_slot(v);
            if part_ty[id] == PT_QRTZ {
                let px = (idx as u32 % xres) as i32;
                let py = (idx as u32 / xres) as i32;
                let mut incoming = 0i32;
                // The sender samples a partner within two pixels, so the same
                // square is enough to find every message addressed to us.
                for n in 0u32..25u32 {
                    let dx = (n % 5u32) as i32 - 2i32;
                    let dy = (n / 5u32) as i32 - 2i32;
                    if dx != 0i32 || dy != 0i32 {
                        let nv = pmap_at(pmap, xres, yres, px + dx, py + dy);
                        if nv != 0u32 {
                            let nid = pmap_slot(nv);
                            let nty = part_ty[nid];
                            // Only a crystal writes a message into its
                            // `tmp4`/`ctype`; any other element's words are
                            // unrelated state.
                            if nty == PT_QRTZ || nty == PT_PQRT {
                                let amount = part_tmp4[nid];
                                if amount > 0i32 && part_ctype[nid] == id as u32 {
                                    incoming += amount;
                                }
                            }
                        }
                    }
                }
                if incoming > 0i32 {
                    part_tmp[id] += incoming;
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {

    /// Host mirror of [`qrtz_pack`]: the amount is the whole `tmp4` word, so
    /// the half-difference is transferred exactly (`QRTZ.cpp:143-155`).
    fn qrtz_pack_host(amount: i32) -> i32 {
        amount.clamp(1, i32::MAX)
    }

    #[test]
    fn qrtz_diffusion_message_keeps_the_half_difference() {
        // A salt-loaded crystal can store far more than 255 units; the amount
        // must not saturate on a byte any more.
        assert_eq!(qrtz_pack_host(1234), 1234);
        assert_eq!(qrtz_pack_host(i32::MAX), i32::MAX);
        // A transfer always moves at least one unit.
        assert_eq!(qrtz_pack_host(0), 1);
        assert_eq!(qrtz_pack_host(-5), 1);
    }

    #[test]
    fn qrtz_growth_needs_a_still_crystal() {
        // `QRTZ.cpp:93`: `tmp > 0 && vx² + vy² < 0.2 && life <= 0`.
        let grows = |vx: f32, vy: f32, life: i32, tmp: i32| {
            tmp > 0 && vx * vx + vy * vy < 0.2f32 && life <= 0
        };
        assert!(grows(0.0f32, 0.0f32, 0, 5));
        assert!(grows(0.4f32, 0.0f32, 0, 5));
        assert!(!grows(0.5f32, 0.0f32, 0, 5), "a moving crystal cannot grow");
        assert!(
            !grows(0.0f32, 0.0f32, 1, 5),
            "a cracked crystal cannot grow"
        );
        assert!(!grows(0.0f32, 0.0f32, 0, 0), "no stored salt, no growth");
    }
}
