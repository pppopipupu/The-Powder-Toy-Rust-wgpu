use cubecl::prelude::*;
use tpt_core::constants::*;

use super::transport::{pipe_eater_slot, pipe_eats, pipe_entrance_dir, pipe_stor_source};
use super::{
    EXIT_FREE, EXIT_NONE, EXIT_PIPE, MOVABLE_MASK, NO_DIR, NO_SLOT, PFLAG_COLORS,
    PFLAG_SINGLE_FORWARD, PIPE_FLAG_DIR_SHIFT, PIPE_FLAG_EAT, PIPE_FLAG_PUSH, PIPE_FLAG_RECV,
    PIPE_FLAG_RELEASE, PIPE_FLAG_STOR, PIPE_FLAG_STOR_TAKEN, PIPE_FLAG_VICTIM,
    PIPE_TMP_RELEASE_PENDING, PMAP_TYPE, PPIP_PAUSED, has_free_neighbour_both, map_at_both,
    next_colour, pipe_offset, prev_colour, sentinel, single_pixel_exit_dir,
};
use crate::reaction_kernel::shared::{WORD_PROPERTIES, elem_word, pmap_at, pmap_slot};

/// makes the choice a pure function of the pre-dispatch state.
#[cube]
#[allow(clippy::too_many_arguments)]
fn pipe_exit(
    pmap: &[u32],
    photons: &[u32],
    part_ctype: &[u32],
    part_tmp: &[i32],
    xres: u32,
    yres: u32,
    px: i32,
    py: i32,
    tmp: i32,
    colour: i32,
) -> (u32, u32) {
    let own = pmap_at(pmap, xres, yres, px, py);
    let mut dir = sentinel(own, NO_DIR);
    let mut kind = sentinel(own, EXIT_NONE);
    if (tmp & PFLAG_SINGLE_FORWARD) != 0i32 {
        // `PIPE.cpp:590-628`: the fixed direction of a one-wide run.
        let d = single_pixel_exit_dir(tmp);
        let (dx, dy) = pipe_offset(d);
        let v = map_at_both(pmap, photons, xres, yres, px + dx, py + dy);
        if v == 0u32 {
            dir = d;
            kind = sentinel(own, EXIT_FREE);
        } else {
            let nt = v & PMAP_TYPE;
            if (nt == PT_PIPE || nt == PT_PPIP) && part_ctype[pmap_slot(v)] == 0u32 {
                dir = d;
                kind = sentinel(own, EXIT_PIPE);
            }
        }
    } else {
        // `PIPE.cpp:552-588`: the transfer target must carry `nextColor` and
        // have no cargo.
        let want = next_colour(colour);
        let mut best = sentinel(own, NO_SLOT);
        for k in 0u32..8u32 {
            let (dx, dy) = pipe_offset(k);
            let v = map_at_both(pmap, photons, xres, yres, px + dx, py + dy);
            if v != 0u32 {
                let nt = v & PMAP_TYPE;
                if nt == PT_PIPE || nt == PT_PPIP {
                    let ns = pmap_slot(v);
                    if (part_tmp[ns] & PFLAG_COLORS) == want && part_ctype[ns] == 0u32 {
                        let ns32 = ns as u32;
                        if ns32 < best {
                            best = ns32;
                            dir = k;
                            kind = sentinel(own, EXIT_PIPE);
                        }
                    }
                }
            }
        }
    }
    (dir, kind)
}

/// The lowest-slot neighbour that may push into a pipe of colour `colour`, as
/// `(slot, direction from the caller to it)`.
///
/// TPT's receiver side is implicit — the sender writes into it — so the gather
/// has to name it: a pipe of colour `colour` accepts from a neighbour of colour
/// `prevColor`, i.e. one whose own `nextColor` is `colour`. Paused pipes and
/// pipes waiting for the host never push.
#[cube]
#[allow(clippy::too_many_arguments)]
fn pipe_upstream(
    pmap: &[u32],
    part_ctype: &[u32],
    part_tmp: &[i32],
    xres: u32,
    yres: u32,
    px: i32,
    py: i32,
    colour: i32,
) -> (u32, u32) {
    let want = prev_colour(colour);
    let own = pmap_at(pmap, xres, yres, px, py);
    let mut best = sentinel(own, NO_SLOT);
    let mut best_dir = sentinel(own, NO_DIR);
    for k in 0u32..8u32 {
        let (dx, dy) = pipe_offset(k);
        let v = pmap_at(pmap, xres, yres, px + dx, py + dy);
        if v != 0u32 {
            let nt = v & PMAP_TYPE;
            if nt == PT_PIPE || nt == PT_PPIP {
                let ns = pmap_slot(v);
                let ntmp = part_tmp[ns];
                if (ntmp & PFLAG_COLORS) == want
                    && (ntmp & PPIP_PAUSED) == 0i32
                    && (ntmp & PIPE_TMP_RELEASE_PENDING) == 0i32
                    && part_ctype[ns] != 0u32
                {
                    let ns32 = ns as u32;
                    if ns32 < best {
                        best = ns32;
                        best_dir = k;
                    }
                }
            }
        }
    }
    (best, best_dir)
}

/// The handshake decisions (`PIPE.cpp:233-263`, `PIPE.cpp:541-629`).
///
/// An edge `p → q` is live when both ends agree, and each end is a read-only
/// predicate over the pre-dispatch state: [`pipe_exit`] names the lowest empty
/// neighbour of colour `nextColor(p)`, [`pipe_upstream`] names the lowest
/// cargo-carrying neighbour of colour `prevColor(q)`. The claim freezes the
/// verdict in `part_flags` so the copy and the clear can run in later passes
/// without re-reading a field that has since changed.
///
/// The pixel is looked up in `pmap` and then `photons` (`PIPE.cpp:244-245`), so
/// an energy particle is a cargo like any other: the pipe claims
/// [`PIPE_FLAG_EAT`] and the particle claims [`PIPE_FLAG_VICTIM`], and the two
/// sides name each other through [`pipe_eater_slot`].
///
/// This pass writes only its own flags, and the fields it reads (`pmap`,
/// `photons`, `ctype`, `tmp`) are written by no other thread here, so it is
/// race-free.
#[cube(launch_unchecked)]
#[allow(clippy::too_many_arguments)]
pub(super) fn pipe_claim_kernel(
    pmap: &[u32],
    photons: &[u32],
    part_ctype: &[u32],
    part_tmp: &[i32],
    elem_props: &[u32],
    part_flags: &mut [u32],
    xres: u32,
    yres: u32,
    frame_seed: u32,
) {
    let idx = ABSOLUTE_POS;
    if idx >= (xres * yres) as usize {
        terminate!();
    }
    let px = (idx as u32 % xres) as i32;
    let py = (idx as u32 / xres) as i32;
    let v = map_at_both(pmap, photons, xres, yres, px, py);
    if v == 0u32 {
        terminate!();
    }
    let ty = v & PMAP_TYPE;
    let slot = pmap_slot(v);
    let mut claim = 0u32;

    if ty == PT_PIPE || ty == PT_PPIP {
        let tmp = part_tmp[slot];
        let colour = tmp & PFLAG_COLORS;
        if colour != 0i32 && (tmp & PPIP_PAUSED) == 0i32 && (tmp & PIPE_TMP_RELEASE_PENDING) == 0i32
        {
            if part_ctype[slot] != 0u32 {
                // Sender half. `pushParticle` (`PIPE.cpp:541-629`) writes the
                // destination; the gather clears its own cargo instead, but
                // only once the destination's own choice names it back.
                let (dir, kind) = pipe_exit(
                    pmap, photons, part_ctype, part_tmp, xres, yres, px, py, tmp, colour,
                );
                let mut pushed = false;
                if kind == EXIT_PIPE {
                    let (dx, dy) = pipe_offset(dir);
                    let tx = px + dx;
                    let ty2 = py + dy;
                    let nv = pmap_at(pmap, xres, yres, tx, ty2);
                    if nv != 0u32 {
                        let ns = pmap_slot(nv);
                        let ntmp = part_tmp[ns];
                        let (up, _) = pipe_upstream(
                            pmap,
                            part_ctype,
                            part_tmp,
                            xres,
                            yres,
                            tx,
                            ty2,
                            ntmp & PFLAG_COLORS,
                        );
                        // A destination that eats a particle this frame takes
                        // nothing: TPT fills the pipe first, so its pickup would
                        // have found it full. The sender has to reach the same
                        // verdict or the cargo would be dropped in the finish
                        // pass by a pipe that never received it.
                        let eats = pipe_eats(
                            pmap, photons, part_ctype, part_tmp, elem_props, xres, yres, tx, ty2,
                            ns as u32, ntmp, frame_seed,
                        );
                        if up == slot as u32 && !eats {
                            claim = PIPE_FLAG_PUSH;
                            pushed = true;
                        }
                    }
                }
                if !pushed {
                    // `PIPE.cpp:233-249` / `PIPE.cpp:590-628`: no pipe to push
                    // into, so the cargo leaves the network through free space.
                    let mut release = kind == EXIT_FREE;
                    if kind == EXIT_NONE
                        && has_free_neighbour_both(pmap, photons, xres, yres, px, py)
                    {
                        release = true;
                    }
                    if release {
                        claim = PIPE_FLAG_RELEASE;
                    }
                }
            } else if pipe_eats(
                pmap,
                photons,
                part_ctype,
                part_tmp,
                elem_props,
                xres,
                yres,
                px,
                py,
                slot as u32,
                tmp,
                frame_seed,
            ) {
                claim = PIPE_FLAG_EAT;
            } else {
                // `PIPE.cpp:259-263`: the same probe that looks for a particle
                // also looks for a `STOR` holding a movable element. Both ends
                // run [`pipe_stor_source`], so the pipe that takes the element
                // is the one the storer names.
                let edir = pipe_entrance_dir(slot as u32, frame_seed);
                let (edx, edy) = pipe_offset(edir);
                let ev = pmap_at(pmap, xres, yres, px + edx, py + edy);
                let mut stor = false;
                if ev != 0u32 && (ev & PMAP_TYPE) == PT_STOR {
                    stor = pipe_stor_source(
                        pmap,
                        part_ctype,
                        part_tmp,
                        elem_props,
                        xres,
                        yres,
                        px + edx,
                        py + edy,
                        frame_seed,
                    ) == slot as u32;
                }
                if stor {
                    claim = PIPE_FLAG_STOR | (edir << PIPE_FLAG_DIR_SHIFT);
                } else {
                    // Receiver half: `transfer_pipe_to_pipe` (`PIPE.cpp:497-539`)
                    // seen from the destination. The direction is recorded so the
                    // copy pass can index the upstream without rescanning
                    // neighbours that have just been filled.
                    let (up, updir) =
                        pipe_upstream(pmap, part_ctype, part_tmp, xres, yres, px, py, colour);
                    if up != NO_SLOT {
                        let (udx, udy) = pipe_offset(updir);
                        let us = up as usize;
                        let utmp = part_tmp[us];
                        let (odir, okind) = pipe_exit(
                            pmap,
                            photons,
                            part_ctype,
                            part_tmp,
                            xres,
                            yres,
                            px + udx,
                            py + udy,
                            utmp,
                            utmp & PFLAG_COLORS,
                        );
                        if okind == EXIT_PIPE && odir == (7u32 - updir) {
                            claim = PIPE_FLAG_RECV | (updir << PIPE_FLAG_DIR_SHIFT);
                        }
                    }
                }
            }
        }
    } else if ty == PT_STOR {
        // `PIPE.cpp:259-263` from the storer's side: the storer empties itself
        // once a pipe takes its element, which the finish pass applies.
        if pipe_stor_source(
            pmap, part_ctype, part_tmp, elem_props, xres, yres, px, py, frame_seed,
        ) != NO_SLOT
        {
            claim = PIPE_FLAG_STOR_TAKEN;
        }
    } else if (elem_word(elem_props, ty, WORD_PROPERTIES) & MOVABLE_MASK) != 0u32
        && pipe_eater_slot(pmap, part_ctype, part_tmp, xres, yres, px, py, frame_seed) != NO_SLOT
    {
        // Particle half: exactly the pipe the eater predicate names takes this
        // slot's cargo, so the death is committed by [`pipe_victim_kernel`].
        claim = PIPE_FLAG_VICTIM;
    }

    if claim != 0u32 {
        part_flags[slot] |= claim;
    }
}

#[cfg(test)]
mod tests {
    use super::super::pack;
    use super::super::{
        PFLAG_COLOR_BLUE, PFLAG_COLOR_GREEN, PFLAG_COLOR_RED, PIPE_FLAG_DIR_MASK,
        PIPE_FLOOD_APPLIED, PIPE_FLOOD_CONDUCT, PIPE_FLOOD_GEN, PIPE_FLOOD_GEN_MOD,
        PIPE_FLOOD_GEN_SHIFT, PIPE_FLOOD_KIND, PIPE_FLOOD_OFF, PIPE_FLOOD_ON, PIPE_FLOOD_REVERSE,
    };
    use super::*;

    /// and pipes are matter.
    #[allow(clippy::too_many_arguments)]
    fn exit_of(
        pmap: &[u32],
        ctype: &[u32],
        tmp: &[i32],
        xres: u32,
        yres: u32,
        px: i32,
        py: i32,
        slot: usize,
    ) -> (u32, u32) {
        let t = tmp[slot];
        let photons = vec![0u32; (xres * yres) as usize];
        pipe_exit(
            pmap,
            &photons,
            ctype,
            tmp,
            xres,
            yres,
            px,
            py,
            t,
            t & PFLAG_COLORS,
        )
    }

    /// [`pipe_upstream`] for the pipe in `slot`, reading that slot's own `tmp`.
    #[allow(clippy::too_many_arguments)]
    fn upstream_of(
        pmap: &[u32],
        ctype: &[u32],
        tmp: &[i32],
        xres: u32,
        yres: u32,
        px: i32,
        py: i32,
        slot: usize,
    ) -> (u32, u32) {
        let t = tmp[slot];
        pipe_upstream(pmap, ctype, tmp, xres, yres, px, py, t & PFLAG_COLORS)
    }

    #[test]
    fn transport_edge_needs_both_ends_to_agree() {
        // Three pipes in a row: red at (1, 2), blue at (2, 2), green at (3, 2).
        // The red pipe's exit is the blue one, and the blue pipe's upstream is
        // the red one, so the edge is live in both directions.
        let xres = 8u32;
        let yres = 8u32;
        let mut pmap = vec![0u32; (xres * yres) as usize];
        pmap[2 * 8 + 1] = pack(0, PT_PIPE);
        pmap[2 * 8 + 2] = pack(1, PT_PIPE);
        pmap[2 * 8 + 3] = pack(2, PT_PIPE);

        let mut ctype = vec![0u32; 3];
        let mut tmp = vec![0i32; 3];
        tmp[0] = PFLAG_COLOR_RED;
        tmp[1] = PFLAG_COLOR_BLUE;
        tmp[2] = PFLAG_COLOR_GREEN;
        ctype[0] = PT_WATR;

        // Red's exit: the blue pipe at offset 6 (`(1, 0)`).
        let (dir, kind) = exit_of(&pmap, &ctype, &tmp, xres, yres, 1, 2, 0);
        assert_eq!((dir, kind), (6, EXIT_PIPE));
        // Blue's upstream: the red pipe at offset 1 (`(-1, 0)`), the reverse of
        // the red pipe's exit offset 6.
        let (up, updir) = upstream_of(&pmap, &ctype, &tmp, xres, yres, 2, 2, 1);
        assert_eq!((up, updir), (0, 1));
        assert_eq!(7u32 - updir, dir);

        // With the red pipe empty the edge disappears from both ends.
        ctype[0] = 0;
        let (up, _) = upstream_of(&pmap, &ctype, &tmp, xres, yres, 2, 2, 1);
        assert_eq!(up, NO_SLOT);

        // A blue pipe with cargo cannot be an exit.
        ctype[1] = PT_DUST;
        let (_, kind) = exit_of(&pmap, &ctype, &tmp, xres, yres, 1, 2, 0);
        assert_eq!(kind, EXIT_NONE);

        // A free pixel at the exit is a release, and a `0x200` pipe takes its
        // fixed direction instead of scanning.
        ctype[1] = 0;
        pmap[2 * 8 + 2] = 0;
        let (_, kind) = exit_of(&pmap, &ctype, &tmp, xres, yres, 1, 2, 0);
        assert_eq!(kind, EXIT_NONE);
        tmp[0] |= PFLAG_SINGLE_FORWARD | (1i32 << 10);
        let (dir, kind) = exit_of(&pmap, &ctype, &tmp, xres, yres, 1, 2, 0);
        assert_eq!((dir, kind), (6, EXIT_FREE));
    }

    #[test]
    fn handshake_flags_round_trip_and_never_touch_a_request_bit() {
        // The transient bits must stay clear of the spawn-request bits
        // (`FLAG_SPAWN_*`, bits 8-13), of the move kernel's bits 0 and 4, and
        // of each other: a collision would either raise a spawn or corrupt a
        // move flag.
        let all = PIPE_FLAG_PUSH
            | PIPE_FLAG_RECV
            | PIPE_FLAG_DIR_MASK
            | PIPE_FLAG_EAT
            | PIPE_FLAG_RELEASE
            | PIPE_FLAG_VICTIM
            | PIPE_FLAG_STOR
            | PIPE_FLAG_STOR_TAKEN;
        let request_mask = 0x3F00u32;
        assert_eq!(all & request_mask, 0u32);
        assert_eq!(all & 0x11u32, 0u32);
        assert_eq!(
            all.count_ones(),
            PIPE_FLAG_PUSH.count_ones()
                + PIPE_FLAG_RECV.count_ones()
                + PIPE_FLAG_DIR_MASK.count_ones()
                + PIPE_FLAG_EAT.count_ones()
                + PIPE_FLAG_RELEASE.count_ones()
                + PIPE_FLAG_VICTIM.count_ones()
                + PIPE_FLAG_STOR.count_ones()
                + PIPE_FLAG_STOR_TAKEN.count_ones()
        );

        // Every direction survives the packing the claim pass writes and the
        // copy pass reads, and the clear leaves nothing of the claim behind.
        for dir in 0u32..8u32 {
            let claim = PIPE_FLAG_RECV | (dir << PIPE_FLAG_DIR_SHIFT);
            assert!((claim & PIPE_FLAG_RECV) != 0u32);
            assert_eq!((claim >> PIPE_FLAG_DIR_SHIFT) & 7u32, dir);
            assert_eq!(claim & PIPE_FLAG_DIR_MASK, dir << PIPE_FLAG_DIR_SHIFT);
            assert_eq!(claim & !(PIPE_FLAG_RECV | PIPE_FLAG_DIR_MASK), 0u32);
        }

        // The flood marks outlive a frame, so they must not collide with
        // anything either, and the four pending flags must be distinct.
        let flood = PIPE_FLOOD_KIND | PIPE_FLOOD_GEN | PIPE_FLOOD_APPLIED;
        assert_eq!(flood & request_mask, 0u32);
        assert_eq!(flood & 0x11u32, 0u32);
        assert_eq!(flood & all, 0u32);
        assert_eq!(PIPE_FLOOD_KIND & PIPE_FLOOD_GEN, 0u32);
        assert_eq!(PIPE_FLOOD_KIND & PIPE_FLOOD_APPLIED, 0u32);
        assert_eq!(PIPE_FLOOD_GEN & PIPE_FLOOD_APPLIED, 0u32);
        let kinds = [
            PIPE_FLOOD_ON,
            PIPE_FLOOD_OFF,
            PIPE_FLOOD_REVERSE,
            PIPE_FLOOD_CONDUCT,
        ];
        assert_eq!(
            PIPE_FLOOD_KIND.count_ones(),
            kinds.iter().map(|k| k.count_ones()).sum::<u32>()
        );
        for kind in kinds {
            assert_eq!(kind & PIPE_FLOOD_KIND, kind);
        }
        // The generation counter only moves forward, so a member never mistakes
        // the flood it already applied for a new one, and one that is up to half
        // a wrap behind still catches up.
        assert_eq!(PIPE_FLOOD_GEN, 0x7000_0000);
        assert_eq!(PIPE_FLOOD_GEN_SHIFT, 28);
        assert_eq!(PIPE_FLOOD_GEN_MOD, 8);
        for epoch in 0u32..PIPE_FLOOD_GEN_MOD {
            let next = (epoch + 1u32) & (PIPE_FLOOD_GEN_MOD - 1u32);
            assert_ne!(next, epoch);
            assert_eq!((next << PIPE_FLOOD_GEN_SHIFT) & !PIPE_FLOOD_GEN, 0u32);
            assert_eq!(
                (next + PIPE_FLOOD_GEN_MOD - epoch) & (PIPE_FLOOD_GEN_MOD - 1u32),
                1u32
            );
            let behind = (epoch + PIPE_FLOOD_GEN_MOD - 4u32) & (PIPE_FLOOD_GEN_MOD - 1u32);
            assert_eq!(
                (epoch + PIPE_FLOOD_GEN_MOD - behind) & (PIPE_FLOOD_GEN_MOD - 1u32),
                4u32
            );
        }
        assert_eq!(PIPE_FLOOD_APPLIED, 0x8000_0000);
        assert_eq!(
            PIPE_FLOOD_KIND | PIPE_FLOOD_GEN | PIPE_FLOOD_APPLIED,
            0xFF00_0000
        );
    }
}
