// Clippy's rewrite for these lints uses methods the device IR does not have
// (`is_multiple_of`, `clamp`, `Range::contains`, `saturating_sub`), and
// collapsing a nested `if` changes what the CubeCL macro expands, so the plain
// operators are deliberate.
#![allow(clippy::collapsible_if)]

use cubecl::prelude::*;
use tpt_core::constants::*;

use super::{MERC_ABSORB_SCALE, chance, chance1, clampf, clampi};
use crate::reaction_kernel::shared::{hash_u32, mix_index, pmap_at, pmap_slot};

/// `MERC.cpp` target volume: `int(10000/(temp+1))-1`, plus the rounded chance.
///
/// `MERC.cpp:56` divides by the *float* temperature, while the rounding chance
/// at `MERC.cpp:61-62` uses `int(temp)+1`; the two must not be conflated or the
/// target volume is one too high for every temperature with a fractional part.
#[cube]
pub(super) fn merc_maxtmp(temp: f32, seed: u32) -> i32 {
    let mut maxtmp = ((MERC_ABSORB_SCALE as f32) / (temp + 1.0f32)) as i32 - 1i32;
    let mut it1 = (temp as i32) + 1i32;
    if it1 < 1i32 {
        it1 = 1i32;
    }
    let rem = (MERC_ABSORB_SCALE % it1) as u32;
    if chance(seed, rem, it1 as u32) {
        maxtmp += 1i32;
    }
    maxtmp
}

/// `GRAV`, `VIBR`/`BVBR` and `MERC` state updates.
///
/// Slot parallel: every case only needs the particle's own fields, plus the
/// coarse-cell `pv` for `VIBR`'s pressure absorption (`VIBR.cpp:70-80`).
/// `MERC`'s condense/spill branches live in [`special_transform_kernel`], its
/// volume equalisation in [`merc_trade_snapshot_kernel`] / [`merc_trade_kernel`]
/// below, and `TUNG` in [`special_pressure_kernel`], where the spatial map is
/// available.
#[cube(launch_unchecked)]
pub(super) fn special_force_kernel(
    part_x: &[f32],
    part_y: &[f32],
    part_ty: &mut [u32],
    part_life: &mut [i32],
    part_tmp: &mut [i32],
    part_temp: &mut [f32],
    part_vx: &mut [f32],
    part_vy: &mut [f32],
    air_pv: &mut [f32],
    n_slots: u32,
    xcells: u32,
    frame_seed: u32,
) {
    let i = ABSOLUTE_POS;
    if i < n_slots as usize {
        let ty = part_ty[i];
        if ty != PT_NONE {
            let seed = hash_u32(frame_seed ^ mix_index(i as u32, 2_246_822_519u32));

            if ty == PT_GRAV {
                // `GRAV.cpp:44-52`: fast-moving dust remembers a short glow.
                let vx = part_vx[i];
                let vy = part_vy[i];
                if vx * vx + vy * vy >= 0.1f32 && chance1(seed, 512u32) && part_life[i] == 0i32 {
                    part_life[i] = 48i32;
                }
            } else if ty == PT_VIBR || ty == PT_BVBR {
                // `VIBR.cpp:43-84`: absorb heat and pressure into `tmp`.
                if part_life[i] == 0i32 {
                    let mut t = part_temp[i];
                    if t > 274.65f32 {
                        part_tmp[i] += 1i32;
                        t -= 3.0f32;
                    } else if t < 271.65f32 {
                        part_tmp[i] -= 1i32;
                        t += 3.0f32;
                    }
                    part_temp[i] = t;
                    // `VIBR.cpp:70-80`: a pressurised cell charges the crystal
                    // (`tmp += 7`) at the cost of one pressure unit, and a
                    // vacuum drains it (`tmp -= 2`) while raising the cell one
                    // unit. The cell update is a plain read-modify-write like
                    // `TUNG.cpp:78`: two vibranium pixels in one coarse cell can
                    // lose one unit where TPT serialises them.
                    let px = part_x[i] as i32;
                    let py = part_y[i] as i32;
                    if px >= 0i32 && py >= 0i32 {
                        let cidx = (((py as u32) / CELL) * xcells + ((px as u32) / CELL)) as usize;
                        let pv = air_pv[cidx];
                        if pv > 2.5f32 {
                            part_tmp[i] += 7i32;
                            air_pv[cidx] = pv - 1.0f32;
                        } else if pv < -2.5f32 {
                            part_tmp[i] -= 2i32;
                            air_pv[cidx] = pv + 1.0f32;
                        }
                    }
                    // `VIBR.cpp:81-83`: the stored energy arms the countdown.
                    if part_tmp[i] > 1000i32 {
                        part_life[i] = 750i32;
                    }
                }
                // `VIBR.cpp:215`: the reference clamps the charge at the very
                // end of the update.
                part_tmp[i] = clampi(part_tmp[i], 0i32, 1i32 << 15i32);
            } else if ty == PT_MERC {
                // `MERC.cpp:55-146`: the volume target follows the temperature.
                let t = clampf(part_temp[i], MIN_TEMP, MAX_TEMP);
                part_temp[i] = t;
                part_tmp[i] = clampi(part_tmp[i], 0i32, MERC_ABSORB_SCALE);
                // Condensing and spilling live in the transform pass, where the
                // spatial map is available.
            }
        }
    }
}

/// `MERC.cpp:131-141`: the size of one trade, `diff/2` (or one unit).
///
/// TPT transfers `diff/2` from the richer pixel to the poorer one, except that
/// a difference of one transfers exactly one unit.
#[cube]
fn merc_trade_delta(remaining: i32, partner: i32) -> i32 {
    let mut delta = 0i32;
    if remaining > partner && remaining > 0i32 {
        let diff = remaining - partner;
        if diff == 1i32 {
            delta = 1i32;
        } else {
            delta = diff / 2i32;
        }
    }
    delta
}

/// `MERC.cpp:122-123`: `between(-2, 2)` for `rx` and `ry`.
///
/// The draw is keyed by the *sender's* pixel and the frame seed, so both the
/// sender and every receiver replaying its trades agree on the offsets.
#[cube]
fn merc_trade_offset(seed: u32, trade: u32) -> (i32, i32) {
    let roll = hash_u32(seed ^ mix_index(trade, 0x9E37_79B9u32));
    (
        ((roll % 5u32) as i32) - 2i32,
        (((roll >> 8u32) % 5u32) as i32) - 2i32,
    )
}

/// `MERC.cpp:120-145` send half: freeze every mercury pixel's volume.
///
/// The four trades move `tmp` between two particles. A pixel may only write
/// its own slot, so the exchange runs as a QRTZ-style message channel without
/// messages: this pass copies each mercury's post-condense volume into the
/// `intent_target` scratch, and [`merc_trade_kernel`] replays the *senders'*
/// deterministic draws from that frozen copy. `intent_target` is free here:
/// the electrical passes consumed their spark snapshot earlier in the same
/// reaction step, and the movement pass rewrites the whole buffer before it
/// reads it (`reaction_kernel/mod.rs:633-636`).
#[cube(launch_unchecked)]
pub(super) fn merc_trade_snapshot_kernel(
    part_ty: &[u32],
    part_tmp: &[i32],
    snapshot: &mut [u32],
    n_slots: u32,
) {
    let i = ABSOLUTE_POS;
    if i < n_slots as usize {
        if part_ty[i] == PT_MERC {
            snapshot[i] = part_tmp[i] as u32;
        }
    }
}

/// `MERC.cpp:120-145` receive half: apply four random `tmp` trades per pixel.
///
/// Each mercury subtracts the volume it hands to its four sampled partners and
/// adds the volume the mercury pixels in its own 5x5 window hand to it. Every
/// trade is replayed from the frozen snapshot, so a trade's two halves always
/// agree and the total volume is conserved. TPT's four trades run in particle
/// order and each reads the partner's already-updated `tmp`; this pass applies
/// the same trades against the frame-start volumes, so a trade whose partner
/// also traded earlier in TPT's slot order can move a slightly different
/// amount. The fixed point, and therefore the final equilibrium, is the same.
#[cube(launch_unchecked)]
pub(super) fn merc_trade_kernel(
    pmap: &[u32],
    part_ty: &[u32],
    part_tmp: &mut [i32],
    snapshot: &[u32],
    xres: u32,
    yres: u32,
    frame_seed: u32,
) {
    let idx = ABSOLUTE_POS;
    if idx < (xres * yres) as usize {
        let v = pmap[idx];
        if v != 0u32 {
            let id = pmap_slot(v);
            if part_ty[id] == PT_MERC {
                let px = (idx as u32 % xres) as i32;
                let py = (idx as u32 / xres) as i32;
                let seed = hash_u32(frame_seed ^ mix_index(idx as u32, 2_246_822_519u32));
                let mut remaining = snapshot[id] as i32;

                // Outgoing: the four trades of this pixel, in TPT order. The
                // running total mirrors TPT decrementing `parts[i].tmp`.
                for trade in 0u32..4u32 {
                    let (rx, ry) = merc_trade_offset(seed, trade);
                    if rx != 0i32 || ry != 0i32 {
                        let nv = pmap_at(pmap, xres, yres, px + rx, py + ry);
                        if nv != 0u32 {
                            let nid = pmap_slot(nv);
                            if part_ty[nid] == PT_MERC {
                                let delta = merc_trade_delta(remaining, snapshot[nid] as i32);
                                remaining -= delta;
                            }
                        }
                    }
                }

                // Incoming: replay every mercury in the 5x5 window and credit
                // the trades it addressed to this pixel. The sender's pixel
                // seed depends only on its own index and the frame seed, so
                // both sides of a trade compute the same offsets and deltas.
                let mut incoming = 0i32;
                for n in 0u32..25u32 {
                    let dx = (n % 5u32) as i32 - 2i32;
                    let dy = (n / 5u32) as i32 - 2i32;
                    if dx != 0i32 || dy != 0i32 {
                        let nv = pmap_at(pmap, xres, yres, px + dx, py + dy);
                        if nv != 0u32 {
                            let sid = pmap_slot(nv);
                            if part_ty[sid] == PT_MERC {
                                let sx = px + dx;
                                let sy = py + dy;
                                let s_seed = hash_u32(
                                    frame_seed
                                        ^ mix_index(
                                            ((sy as u32) * xres) + sx as u32,
                                            2_246_822_519u32,
                                        ),
                                );
                                let mut s_remaining = snapshot[sid] as i32;
                                for trade in 0u32..4u32 {
                                    let (rx, ry) = merc_trade_offset(s_seed, trade);
                                    if rx != 0i32 || ry != 0i32 {
                                        let pv = pmap_at(pmap, xres, yres, sx + rx, sy + ry);
                                        if pv != 0u32 {
                                            let pid = pmap_slot(pv);
                                            if part_ty[pid] == PT_MERC {
                                                let delta = merc_trade_delta(
                                                    s_remaining,
                                                    snapshot[pid] as i32,
                                                );
                                                s_remaining -= delta;
                                                if pid == id {
                                                    incoming += delta;
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }

                part_tmp[id] = remaining + incoming;
            }
        }
    }
}

#[cfg(test)]
mod tests {

    use super::merc_trade_delta;

    /// Host mirror of the volume target in [`merc_maxtmp`], without the
    /// rounding chance: `MERC.cpp:56` truncates the *float* quotient.
    fn merc_maxtmp_host(temp: f32) -> i32 {
        (10_000.0f32 / (temp + 1.0f32)) as i32 - 1
    }

    #[test]
    fn merc_target_volume_divides_by_the_float_temperature() {
        assert_eq!(merc_maxtmp_host(0.0f32), 9_999);
        assert_eq!(merc_maxtmp_host(99.0f32), 99, "10000/100 - 1");
        assert_eq!(merc_maxtmp_host(293.15f32), 32, "10000/294.15 - 1");
        // Dividing by `int(temp)+1` instead would answer 33 here and 4999 at
        // 1.5 K, i.e. a target volume that is one too high.
        assert_ne!(merc_maxtmp_host(293.15f32), 10_000 / 294 - 1);
        assert_eq!(merc_maxtmp_host(1.5f32), 3_999);
    }

    /// `MERC.cpp:131-141`: only the richer pixel sends, a difference of one
    /// moves one unit and anything larger moves `diff/2`.
    #[test]
    fn merc_trade_moves_half_the_difference() {
        assert_eq!(merc_trade_delta(10, 0), 5);
        assert_eq!(merc_trade_delta(10, 9), 1);
        assert_eq!(merc_trade_delta(10, 8), 1);
        assert_eq!(merc_trade_delta(10, 10), 0);
        assert_eq!(merc_trade_delta(9, 10), 0);
        assert_eq!(merc_trade_delta(0, 0), 0);
        assert_eq!(merc_trade_delta(3, 0), 1);
    }
}
