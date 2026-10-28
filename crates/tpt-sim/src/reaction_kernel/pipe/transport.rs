use cubecl::prelude::*;
use tpt_core::constants::*;

use super::{
    MOVABLE_MASK, NO_SLOT, PFLAG_CAN_CONDUCT, PFLAG_COLORS, PIPE_FLAG_DIR_MASK,
    PIPE_FLAG_DIR_SHIFT, PIPE_FLAG_EAT, PIPE_FLAG_PUSH, PIPE_FLAG_RECV, PIPE_FLAG_RELEASE,
    PIPE_FLAG_STOR, PIPE_FLAG_STOR_TAKEN, PIPE_FLAG_VICTIM, PIPE_TMP_RELEASE_PENDING, PMAP_TYPE,
    PPIP_PAUSED, map_at_both, pipe_offset, sentinel,
};
use crate::reaction_kernel::shared::{
    FLAG_SPAWN_CLONE, WORD_PROPERTIES, elem_word, pmap_at, pmap_slot, soap_detach,
};

/// `BASE` (`1.5f`, `BASE.cpp:33`), the only override in the shipped table.
#[cube]
fn heat_capacity(ty: u32) -> f32 {
    let mut capacity = 1.0f32;
    if ty == PT_BASE {
        capacity = 1.5f32;
    }
    capacity
}

/// `PIPE.cpp:238-242`: the random entrance offset, rotated per frame.
///
/// TPT draws one of the eight offsets from the RNG every frame. The slot id
/// plus the frame seed is a rotation of the eight directions that changes each
/// frame, stays a pure function (so the pipe and the particle it eats agree on
/// the direction) and needs no hash: `shared::hash_u32` overflows in a debug
/// host build, which would make the handshake untestable.
#[cube]
pub(super) fn pipe_entrance_dir(slot: u32, frame_seed: u32) -> u32 {
    ((slot & 7u32) + (frame_seed & 7u32)) & 7u32
}

/// The pipe that eats a particle at `(x, y)`, or [`NO_SLOT`].
///
/// A candidate must be a pipe, coloured, unpaused, empty, not waiting for the
/// host, and its entrance offset must point back at the particle. The lowest
/// slot wins, so exactly one pipe takes the cargo and the particle dies once.
#[cube]
#[allow(clippy::too_many_arguments)]
pub(super) fn pipe_eater_slot(
    pmap: &[u32],
    part_ctype: &[u32],
    part_tmp: &[i32],
    xres: u32,
    yres: u32,
    x: i32,
    y: i32,
    frame_seed: u32,
) -> u32 {
    let own = pmap_at(pmap, xres, yres, x, y);
    let mut best = sentinel(own, NO_SLOT);
    for k in 0u32..8u32 {
        let (dx, dy) = pipe_offset(k);
        let v = pmap_at(pmap, xres, yres, x + dx, y + dy);
        if v != 0u32 {
            let nt = v & PMAP_TYPE;
            if nt == PT_PIPE || nt == PT_PPIP {
                let ns = pmap_slot(v);
                let ntmp = part_tmp[ns];
                if (ntmp & PFLAG_COLORS) != 0i32
                    && (ntmp & PPIP_PAUSED) == 0i32
                    && (ntmp & PIPE_TMP_RELEASE_PENDING) == 0i32
                    && part_ctype[ns] == 0u32
                {
                    // The particle sits at `7 - k` as seen from the pipe: the
                    // offset array is symmetric under `k ↔ 7 - k`.
                    let back = 7u32 - k;
                    if pipe_entrance_dir(ns as u32, frame_seed) == back {
                        let ns32 = ns as u32;
                        if ns32 < best {
                            best = ns32;
                        }
                    }
                }
            }
        }
    }
    best
}

/// `true` when the pipe at `(px, py)` swallows a particle this frame.
///
/// The entrance handshake in one predicate, shared by the pipe (its own pickup)
/// and by the transport (a pipe that eats a particle must not also take a cargo
/// from its upstream: TPT's sequential update fills the pipe first, so the
/// pickup would have found it full, and the sender has to reach the same
/// verdict or the cargo would be dropped by a pipe that never received it).
#[cube]
#[allow(clippy::too_many_arguments)]
pub(super) fn pipe_eats(
    pmap: &[u32],
    photons: &[u32],
    part_ctype: &[u32],
    part_tmp: &[i32],
    elem_props: &[u32],
    xres: u32,
    yres: u32,
    px: i32,
    py: i32,
    slot: u32,
    tmp: i32,
    frame_seed: u32,
) -> bool {
    let mut eats = false;
    if (tmp & PFLAG_COLORS) != 0i32
        && (tmp & PPIP_PAUSED) == 0i32
        && (tmp & PIPE_TMP_RELEASE_PENDING) == 0i32
        && part_ctype[slot as usize] == 0u32
    {
        let dir = pipe_entrance_dir(slot, frame_seed);
        let (dx, dy) = pipe_offset(dir);
        let tx = px + dx;
        let ty = py + dy;
        let nv = map_at_both(pmap, photons, xres, yres, tx, ty);
        if nv != 0u32 {
            let nty = nv & PMAP_TYPE;
            if (elem_word(elem_props, nty, WORD_PROPERTIES) & MOVABLE_MASK) != 0u32
                && pipe_eater_slot(pmap, part_ctype, part_tmp, xres, yres, tx, ty, frame_seed)
                    == slot
            {
                eats = true;
            }
        }
    }
    eats
}

/// (`electronics::stor_pack_tmp`), so the test reads that byte.
#[cube]
#[allow(clippy::too_many_arguments)]
pub(super) fn pipe_stor_source(
    pmap: &[u32],
    part_ctype: &[u32],
    part_tmp: &[i32],
    elem_props: &[u32],
    xres: u32,
    yres: u32,
    sx: i32,
    sy: i32,
    frame_seed: u32,
) -> u32 {
    let sv = pmap_at(pmap, xres, yres, sx, sy);
    let mut best = sentinel(sv, NO_SLOT);
    if sv != 0u32 {
        let ss = pmap_slot(sv);
        let stored_ty = (part_tmp[ss] & 0xFFi32) as u32;
        if stored_ty != 0u32
            && (elem_word(elem_props, stored_ty, WORD_PROPERTIES) & MOVABLE_MASK) != 0u32
        {
            for k in 0u32..8u32 {
                let (dx, dy) = pipe_offset(k);
                let v = pmap_at(pmap, xres, yres, sx + dx, sy + dy);
                if v != 0u32 {
                    let nt = v & PMAP_TYPE;
                    if nt == PT_PIPE || nt == PT_PPIP {
                        let ns = pmap_slot(v);
                        let ntmp = part_tmp[ns];
                        if (ntmp & PFLAG_COLORS) != 0i32
                            && (ntmp & PPIP_PAUSED) == 0i32
                            && (ntmp & PIPE_TMP_RELEASE_PENDING) == 0i32
                            && part_ctype[ns] == 0u32
                            && pipe_entrance_dir(ns as u32, frame_seed) == (7u32 - k)
                        {
                            let ns32 = ns as u32;
                            if ns32 < best {
                                best = ns32;
                            }
                        }
                    }
                }
            }
        }
    }
    best
}

/// `transfer_pipe_to_pipe` (`PIPE.cpp:497-539`), including the `STOR = true`
/// form (`PIPE.cpp:513-519`, `PIPE.cpp:259-263`).
///
/// The sender writes nothing in this pass, so every field the receiver reads is
/// frozen; the clear waits for [`pipe_finish_kernel`]. Only receivers write, and
/// a receiver was empty when the claim was made, so it can never be another
/// receiver's upstream: the write set and the read set are disjoint. A storer is
/// not a pipe either, so it is never a receiver.
#[cube(launch_unchecked)]
pub(super) fn pipe_copy_kernel(
    pmap: &[u32],
    part_ctype: &mut [u32],
    part_tmp: &[i32],
    part_tmp2: &mut [i32],
    part_tmp3: &mut [i32],
    part_tmp4: &mut [i32],
    part_temp: &mut [f32],
    part_flags: &mut [u32],
    xres: u32,
    yres: u32,
) {
    let idx = ABSOLUTE_POS;
    if idx >= (xres * yres) as usize {
        terminate!();
    }
    let v = pmap[idx];
    if v == 0u32 {
        terminate!();
    }
    let slot = pmap_slot(v);
    let flags = part_flags[slot];
    let recv = (flags & PIPE_FLAG_RECV) != 0u32;
    let stor = (flags & PIPE_FLAG_STOR) != 0u32;
    if !recv && !stor {
        terminate!();
    }
    part_flags[slot] = flags & !(PIPE_FLAG_RECV | PIPE_FLAG_STOR | PIPE_FLAG_DIR_MASK);

    let dir = (flags >> PIPE_FLAG_DIR_SHIFT) & 7u32;
    let px = (idx as u32 % xres) as i32;
    let py = (idx as u32 / xres) as i32;
    let (dx, dy) = pipe_offset(dir);
    let uv = pmap_at(pmap, xres, yres, px + dx, py + dy);
    if uv == 0u32 {
        terminate!();
    }
    let us = pmap_slot(uv);

    if stor {
        // `PIPE.cpp:513-519`: `dest->ctype = src->tmp` and `src->tmp = 0`. The
        // port's storer keeps the stored type in the low byte and the stored
        // life in the high byte of its `tmp` (`electronics::stor_pack_tmp`), so
        // the life reaches `tmp2` from the same word; the stored element's own
        // `tmp` / `ctype` are not preserved by the storer and arrive as zero.
        let packed = part_tmp[us];
        let stored = (packed & 0xFFi32) as u32;
        if stored == 0u32 {
            terminate!();
        }
        part_ctype[slot] = stored;
        part_tmp2[slot] = (packed >> 8i32) & 0xFFi32;
        part_tmp3[slot] = 0i32;
        part_tmp4[slot] = 0i32;
        // `PIPE.cpp:523-537`: `c_src` is the heat capacity of the storer's
        // `ctype`, which the port's storer does not keep, so the default 1.0
        // applies.
        let mut temp = part_temp[us];
        if (part_tmp[slot] & PFLAG_CAN_CONDUCT) != 0i32 {
            let c_src = heat_capacity(part_ctype[us]);
            let c_dst = heat_capacity(v & PMAP_TYPE);
            temp = (c_src * part_temp[us] + c_dst * part_temp[slot]) / (c_src + c_dst);
        }
        part_temp[slot] = temp;
    } else {
        let ctype = part_ctype[us];
        if ctype == 0u32 {
            terminate!();
        }

        part_ctype[slot] = ctype;
        part_tmp2[slot] = part_tmp2[us];
        part_tmp3[slot] = part_tmp3[us];
        part_tmp4[slot] = part_tmp4[us];
        // `PIPE.cpp:470-487`: `PFLAG_CAN_CONDUCT` mixes the two temperatures by
        // heat capacity instead of taking the cargo's.
        let mut temp = part_temp[us];
        if (part_tmp[slot] & PFLAG_CAN_CONDUCT) != 0i32 {
            let c_src = heat_capacity(ctype);
            let c_dst = heat_capacity(v & PMAP_TYPE);
            temp = (c_src * part_temp[us] + c_dst * part_temp[slot]) / (c_src + c_dst);
        }
        part_temp[slot] = temp;
    }
}

/// The pipe half of the entrance handshake (`PIPE.cpp:250-258`,
/// `PIPE.cpp:470-495`).
///
/// The claim pass froze [`pipe_eats`] / [`pipe_eater_slot`] into
/// [`PIPE_FLAG_EAT`]; this pass only acts on it, and the particle half is
/// [`pipe_victim_kernel`]. The pipe copies the particle's cargo, and the
/// particle is looked up in `pmap` and then `photons` (`PIPE.cpp:253-254`), so
/// an energy cargo arrives with the same fields a matter one does. The pipe
/// reads the particle's `life` / `temp`, which no thread writes here, and the
/// particle's slot is never a pipe, so the eater is never reading a pipe that is
/// writing its own temperature.
#[cube(launch_unchecked)]
pub(super) fn pipe_entrance_kernel(
    pmap: &[u32],
    photons: &[u32],
    part_ty: &mut [u32],
    part_ctype: &mut [u32],
    part_tmp: &mut [i32],
    part_life: &[i32],
    part_temp: &mut [f32],
    part_tmp2: &mut [i32],
    part_flags: &mut [u32],
    xres: u32,
    yres: u32,
    n_slots: u32,
    frame_seed: u32,
) {
    let idx = ABSOLUTE_POS;
    if idx >= (xres * yres) as usize {
        terminate!();
    }
    let v = pmap[idx];
    if v == 0u32 {
        terminate!();
    }
    let ty = v & PMAP_TYPE;
    if ty != PT_PIPE && ty != PT_PPIP {
        terminate!();
    }
    let slot = pmap_slot(v);
    let flags = part_flags[slot];
    if (flags & PIPE_FLAG_EAT) == 0u32 {
        terminate!();
    }
    part_flags[slot] = flags & !PIPE_FLAG_EAT;

    let px = (idx as u32 % xres) as i32;
    let py = (idx as u32 / xres) as i32;
    let dir = pipe_entrance_dir(slot as u32, frame_seed);
    let (dx, dy) = pipe_offset(dir);
    let nv = map_at_both(pmap, photons, xres, yres, px + dx, py + dy);
    if nv != 0u32 {
        let nty = nv & PMAP_TYPE;
        let tslot = pmap_slot(nv);
        // `PIPE.cpp:255-256`: a soap bubble loses its links before its type,
        // life and temperature are transferred into the pipe.
        if nty == PT_SOAP {
            soap_detach(part_ty, part_ctype, part_tmp, part_tmp2, tslot, n_slots);
        }
        // `PIPE.cpp:470-495` `transfer_part_to_pipe`: the type, the particle's
        // `life` into `tmp2`, and the temperature mix. The particle's own `tmp`
        // and `ctype` (TPT `tmp3` / `tmp4`) are not bound here, so they stay
        // zero; `part_tmp3` / `part_tmp4` would be the tenth and eleventh
        // storage buffer.
        part_ctype[slot] = nty;
        part_tmp2[slot] = part_life[tslot];
        part_temp[slot] =
            pipe_pickup_temp(part_tmp[slot], part_temp[slot], part_temp[tslot], ty, nty);
    }
}

/// The particle half of the entrance handshake (`PIPE.cpp:255-258`).
///
/// Slot parallel, because the eaten particle may live in the `photons` layer
/// where the pixel pass cannot reach it. The claim pass only sets
/// [`PIPE_FLAG_VICTIM`] on a slot [`pipe_eater_slot`] named, so the death is the
/// exact complement of the cargo copy in [`pipe_entrance_kernel`].
#[cube(launch_unchecked)]
pub(super) fn pipe_victim_kernel(part_ty: &mut [u32], part_flags: &mut [u32], n_slots: u32) {
    let i = ABSOLUTE_POS;
    if i < n_slots as usize {
        let flags = part_flags[i];
        if (flags & PIPE_FLAG_VICTIM) != 0u32 {
            part_flags[i] = flags & !PIPE_FLAG_VICTIM;
            part_ty[i] = PT_NONE;
        }
    }
}

/// The transport tail (`PIPE.cpp:497-539`, `PIPE.cpp:233-249`).
///
/// Reads and writes only the caller's own slot, so the clears are ordered
/// against the copy pass by the dispatch boundary alone. The `STOR` branch is
/// the `src->tmp = 0` of `transfer_pipe_to_pipe(..., STOR = true)`
/// (`PIPE.cpp:515-517`), done by the storer's own thread because a pipe may not
/// write it; the storer runs after the copy pass, so the value the pipe read is
/// still there.
#[cube(launch_unchecked)]
pub(super) fn pipe_finish_kernel(
    pmap: &[u32],
    part_ctype: &mut [u32],
    part_tmp: &mut [i32],
    part_tmp2: &mut [i32],
    part_tmp3: &mut [i32],
    part_tmp4: &mut [i32],
    part_flags: &mut [u32],
    xres: u32,
    yres: u32,
) {
    let idx = ABSOLUTE_POS;
    if idx >= (xres * yres) as usize {
        terminate!();
    }
    let v = pmap[idx];
    if v == 0u32 {
        terminate!();
    }
    let ty = v & PMAP_TYPE;
    if ty != PT_PIPE && ty != PT_PPIP {
        if ty == PT_STOR {
            // `PIPE.cpp:259-263` → `PIPE.cpp:515-517`: the storer is emptied.
            let stor = pmap_slot(v);
            let stor_flags = part_flags[stor];
            if (stor_flags & PIPE_FLAG_STOR_TAKEN) != 0u32 {
                part_flags[stor] = stor_flags & !PIPE_FLAG_STOR_TAKEN;
                part_tmp[stor] = 0i32;
            }
        }
        terminate!();
    }
    let slot = pmap_slot(v);
    let flags = part_flags[slot];
    let tmp = part_tmp[slot];

    if (tmp & PIPE_TMP_RELEASE_PENDING) != 0i32 {
        // A release is in flight: the host materialises it from `ctype` after
        // this pass and clears the request bit, so a cleared bit means the
        // particle landed. A failed spawn retries instead of losing the cargo.
        if (flags & FLAG_SPAWN_CLONE) == 0u32 {
            part_ctype[slot] = 0u32;
            part_tmp2[slot] = 0i32;
            part_tmp3[slot] = 0i32;
            part_tmp4[slot] = 0i32;
            part_tmp[slot] = tmp & !PIPE_TMP_RELEASE_PENDING;
        }
    } else if (flags & PIPE_FLAG_PUSH) != 0u32 {
        // The receiver copied the cargo in the previous pass.
        part_ctype[slot] = 0u32;
        part_tmp2[slot] = 0i32;
        part_tmp3[slot] = 0i32;
        part_tmp4[slot] = 0i32;
        part_flags[slot] = flags & !PIPE_FLAG_PUSH;
    } else if (flags & PIPE_FLAG_RELEASE) != 0u32 {
        // `FLAG_SPAWN_CLONE` reads `ctype`, which already holds the cargo type;
        // the request bit goes up only after the field is settled.
        part_flags[slot] = (flags & !PIPE_FLAG_RELEASE) | FLAG_SPAWN_CLONE;
        part_tmp[slot] = tmp | PIPE_TMP_RELEASE_PENDING;
    }
}

/// `PIPE.cpp:470-487`: the pipe's temperature after swallowing a particle.
///
/// Without `PFLAG_CAN_CONDUCT` the cargo's temperature replaces the pipe's;
/// with it the two are mixed by heat capacity.
#[cube]
pub(super) fn pipe_pickup_temp(
    pipe_tmp: i32,
    pipe_temp: f32,
    part_temp: f32,
    pipe_ty: u32,
    part_ty: u32,
) -> f32 {
    let mut temp = part_temp;
    if (pipe_tmp & PFLAG_CAN_CONDUCT) != 0i32 {
        let c_part = heat_capacity(part_ty);
        let c_pipe = heat_capacity(pipe_ty);
        temp = (c_part * part_temp + c_pipe * pipe_temp) / (c_part + c_pipe);
    }
    temp
}

#[cfg(test)]
mod tests {
    use super::super::pack;
    use super::super::{PFLAG_COLOR_GREEN, PFLAG_COLOR_RED};
    use super::*;

    #[test]
    fn entrance_direction_is_per_slot_and_per_frame() {
        // A pure function of (slot, frame): the pipe and the particle it eats
        // must agree, and the entrance must rotate so all eight directions are
        // used over time.
        let mut seen = [false; 8];
        for seed in 0u32..256u32 {
            let d = pipe_entrance_dir(7u32, seed);
            assert!(d < 8u32);
            assert_eq!(d, pipe_entrance_dir(7u32, seed));
            seen[d as usize] = true;
        }
        assert!(
            seen.iter().all(|hit| *hit),
            "entrance never rotated: {seen:?}"
        );
        // Different slots do not all share one entrance in a given frame.
        let dirs: Vec<u32> = (0u32..32u32).map(|s| pipe_entrance_dir(s, 11u32)).collect();
        assert!(dirs.iter().any(|d| *d != dirs[0]));
    }

    #[test]
    fn entrance_handshake_names_exactly_one_eater() {
        // 8x8 canvas: a particle at (4, 4) with two coloured pipes next to it,
        // one west at slot 1 and one south at slot 6. From the west pipe the
        // particle is at offset 6, so its entrance must be 6; from the south
        // pipe it is at offset 4, so its entrance must be 3. `(slot + seed) & 7`
        // reaches both at once when `seed & 7 == 5`, which is why the slots are
        // 1 and 6 rather than 1 and 2.
        let xres = 8u32;
        let yres = 8u32;
        let mut pmap = vec![0u32; (xres * yres) as usize];
        pmap[4 * 8 + 4] = pack(0, PT_WATR);
        pmap[4 * 8 + 3] = pack(1, PT_PIPE);
        pmap[5 * 8 + 4] = pack(6, PT_PPIP);

        let mut ctype = vec![0u32; 7];
        let mut tmp = vec![0i32; 7];
        tmp[1] = PFLAG_COLOR_RED;
        tmp[6] = PFLAG_COLOR_GREEN;

        let both = 5u32;
        assert_eq!(pipe_entrance_dir(1, both), 6);
        assert_eq!(pipe_entrance_dir(6, both), 3);

        // Both pipes point at the particle and both are empty: the lowest slot
        // wins, so exactly one pipe takes the cargo.
        assert_eq!(
            pipe_eater_slot(&pmap, &ctype, &tmp, xres, yres, 4, 4, both),
            1
        );

        // The lower slot is full, so the cargo must come from the other one.
        ctype[1] = PT_DUST;
        assert_eq!(
            pipe_eater_slot(&pmap, &ctype, &tmp, xres, yres, 4, 4, both),
            6
        );

        // A paused pipe never eats.
        tmp[6] |= PPIP_PAUSED;
        assert_eq!(
            pipe_eater_slot(&pmap, &ctype, &tmp, xres, yres, 4, 4, both),
            NO_SLOT
        );
        tmp[6] &= !PPIP_PAUSED;

        // A pipe waiting for the host never eats either.
        tmp[6] |= PIPE_TMP_RELEASE_PENDING;
        assert_eq!(
            pipe_eater_slot(&pmap, &ctype, &tmp, xres, yres, 4, 4, both),
            NO_SLOT
        );
        tmp[6] &= !PIPE_TMP_RELEASE_PENDING;

        // A seed where neither entrance points at the particle: nothing eats.
        let neither = 0u32;
        assert_ne!(pipe_entrance_dir(1, neither), 6);
        assert_ne!(pipe_entrance_dir(6, neither), 3);
        assert_eq!(
            pipe_eater_slot(&pmap, &ctype, &tmp, xres, yres, 4, 4, neither),
            NO_SLOT
        );
    }

    #[test]
    fn pickup_temperature_follows_the_conduction_flag() {
        // `PIPE.cpp:470-487`.
        assert_eq!(
            pipe_pickup_temp(0, 300.0f32, 500.0f32, PT_PIPE, PT_FIRE),
            500.0f32
        );
        // With `PFLAG_CAN_CONDUCT` the capacities mix: both are 1.0 here.
        let mixed = pipe_pickup_temp(PFLAG_CAN_CONDUCT, 300.0f32, 500.0f32, PT_PIPE, PT_FIRE);
        assert!((mixed - 400.0f32).abs() < 1e-5f32);
        // `BASE` is the only element with a different capacity (1.5).
        let base = pipe_pickup_temp(PFLAG_CAN_CONDUCT, 300.0f32, 500.0f32, PT_PIPE, PT_BASE);
        assert!((base - (1.5f32 * 500.0f32 + 300.0f32) / 2.5f32).abs() < 1e-5f32);
    }

    #[test]
    fn storer_hands_its_element_to_the_pipe_that_points_at_it() {
        // `PIPE.cpp:259-263`: an empty pipe whose entrance points at a `STOR`
        // holding a movable element takes it, and both ends agree on the same
        // pipe. The storer's `tmp` packs the type in the low byte and the life
        // in the high byte (`electronics::stor_pack_tmp`).
        let xres = 8u32;
        let yres = 8u32;
        let mut pmap = vec![0u32; (xres * yres) as usize];
        pmap[4 * 8 + 4] = pack(0, PT_STOR);
        pmap[4 * 8 + 3] = pack(1, PT_PIPE);

        let mut ctype = vec![0u32; 2];
        let mut tmp = vec![0i32; 2];
        tmp[0] = (PT_WATR as i32 & 0xFF) | (12i32 << 8);
        tmp[1] = PFLAG_COLOR_RED;
        let mut props = vec![0u32; 32 * 256];
        props[PT_WATR as usize * 32 + WORD_PROPERTIES as usize] = TYPE_LIQUID;

        let seed = 5u32;
        assert_eq!(pipe_entrance_dir(1, seed), 6);
        assert_eq!(
            pipe_stor_source(&pmap, &ctype, &tmp, &props, xres, yres, 4, 4, seed),
            1
        );

        // An empty storer, or one holding a solid, hands nothing over.
        tmp[0] = 0;
        assert_eq!(
            pipe_stor_source(&pmap, &ctype, &tmp, &props, xres, yres, 4, 4, seed),
            NO_SLOT
        );
        tmp[0] = (PT_BRCK as i32 & 0xFF) | (12i32 << 8);
        props[PT_BRCK as usize * 32 + WORD_PROPERTIES as usize] = TYPE_SOLID;
        assert_eq!(
            pipe_stor_source(&pmap, &ctype, &tmp, &props, xres, yres, 4, 4, seed),
            NO_SLOT
        );

        // A pipe that is already full, paused or waiting for the host takes
        // nothing, so the storer keeps its element.
        tmp[0] = (PT_WATR as i32 & 0xFF) | (12i32 << 8);
        ctype[1] = PT_DUST;
        assert_eq!(
            pipe_stor_source(&pmap, &ctype, &tmp, &props, xres, yres, 4, 4, seed),
            NO_SLOT
        );
        ctype[1] = 0;
        tmp[1] |= PPIP_PAUSED;
        assert_eq!(
            pipe_stor_source(&pmap, &ctype, &tmp, &props, xres, yres, 4, 4, seed),
            NO_SLOT
        );
        tmp[1] &= !PPIP_PAUSED;
        // A pipe whose entrance points elsewhere does not reach the storer.
        assert_eq!(
            pipe_stor_source(&pmap, &ctype, &tmp, &props, xres, yres, 4, 4, 0),
            NO_SLOT
        );
    }

    #[test]
    fn an_eating_pipe_suppresses_its_transport_receive() {
        // 8x8 canvas: a water particle at (4, 4) with an empty red pipe west of
        // it (slot 1). The seed points the pipe's entrance at the particle, so
        // the pipe eats — and the sender has to see the same verdict, or it
        // would clear a cargo this pipe never took.
        let xres = 8u32;
        let yres = 8u32;
        let mut pmap = vec![0u32; (xres * yres) as usize];
        pmap[4 * 8 + 4] = pack(0, PT_WATR);
        pmap[4 * 8 + 3] = pack(1, PT_PIPE);

        let mut ctype = vec![0u32; 2];
        let mut tmp = vec![0i32; 2];
        tmp[1] = PFLAG_COLOR_RED;
        let mut props = vec![0u32; 32 * 256];
        props[PT_WATR as usize * 32 + WORD_PROPERTIES as usize] = TYPE_LIQUID;

        let seed = 5u32;
        let photons = vec![0u32; (xres * yres) as usize];
        assert_eq!(pipe_entrance_dir(1, seed), 6);
        assert!(pipe_eats(
            &pmap, &photons, &ctype, &tmp, &props, xres, yres, 3, 4, 1, tmp[1], seed
        ));

        // A pipe that already carries cargo never eats.
        ctype[1] = PT_DUST;
        assert!(!pipe_eats(
            &pmap, &photons, &ctype, &tmp, &props, xres, yres, 3, 4, 1, tmp[1], seed
        ));
        ctype[1] = 0;

        // A solid particle is not movable, so the pipe leaves it alone.
        pmap[4 * 8 + 4] = pack(0, PT_BRCK);
        assert!(!pipe_eats(
            &pmap, &photons, &ctype, &tmp, &props, xres, yres, 3, 4, 1, tmp[1], seed
        ));
    }
}
