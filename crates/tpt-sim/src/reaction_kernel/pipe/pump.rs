use cubecl::prelude::*;
use tpt_core::constants::*;

use super::{
    PFLAG_CAN_CONDUCT, PFLAG_COLOR_GREEN, PFLAG_COLOR_RED, PFLAG_COLORS, PFLAG_INITIALIZING,
    PFLAG_SINGLE_FORWARD, PFLAG_SINGLE_PIXEL, PIPE_FLOOD_APPLIED, PIPE_FLOOD_CONDUCT,
    PIPE_FLOOD_GEN, PIPE_FLOOD_GEN_MOD, PIPE_FLOOD_GEN_SHIFT, PIPE_FLOOD_KIND, PIPE_FLOOD_OFF,
    PIPE_FLOOD_ON, PIPE_FLOOD_REVERSE, PMAP_TYPE, PPIP_PAUSED, PPIP_REVERSED, PPIP_TRIGGER_OFF,
    PPIP_TRIGGER_ON, PPIP_TRIGGER_REVERSE, PPIP_TRIGGERS, PT_NUM_U32, next_colour, pipe_offset,
};
use crate::reaction_kernel::shared::{FLAG_SPAWN_CLONE, insulated, pmap_at, pmap_slot};

#[allow(clippy::erasing_op)]
#[cube]
fn sentinel(seed: u32, value: u32) -> u32 {
    (seed & 0u32) | value
}

/// `PIPE.cpp:147-162`: `TRIGGER_REVERSE`.
///
/// Toggles `REVERSED`, swaps red↔blue (green is its own partner, because the
/// mask is `PFLAG_COLOR_GREEN`) and — for a one-wide run — swaps the two
/// four-bit nibbles at bits 9 and 13, i.e. the forward/reverse fixed directions
/// together with their `0x200`/`0x2000` flags.
#[cube]
fn apply_reverse(tmp: i32) -> i32 {
    let mut out = tmp ^ PPIP_REVERSED;
    if (out & PFLAG_COLORS) != PFLAG_COLOR_GREEN {
        out ^= PFLAG_COLOR_GREEN;
    }
    if (out & PFLAG_SINGLE_PIXEL) != 0i32 {
        let lo = ((out as u32) >> 9u32) & 0xFu32;
        let hi = ((out as u32) >> 13u32) & 0xFu32;
        out = (((out as u32) & !0x0001_FE00u32) | (hi << 9u32) | (lo << 13u32)) as i32;
    }
    out
}

/// documented in the module header.
#[cube]
fn has_free_neighbour(pmap: &[u32], xres: u32, yres: u32, x: i32, y: i32) -> bool {
    let mut free = false;
    for k in 0u32..8u32 {
        let (dx, dy) = pipe_offset(k);
        if pmap_at(pmap, xres, yres, x + dx, y + dy) == 0u32 {
            free = true;
        }
    }
    free
}

// Connectivity flood (`PPIP.cpp:64-150`)

/// One round of `Element_PPIP_flood_trigger`.
///
/// The reference walks the whole 8-connected run of same-type pipe slots and
/// ORs `prop` into every member's `tmp` (`PPIP.cpp:96-146`). A dispatch can only
/// move the frontier by one hop, so [`FLOOD_ROUNDS`] rounds per frame carry the
/// mark across the run; the end state is the reference's, and
/// [`PIPE_FLOOD_APPLIED`] makes each member apply the pending flags once.
///
/// Seeding mirrors the reference's three entry points:
///
/// * a `PPIP` next to a spark at `life == 3` whose `ctype` is `PSCN` / `NSCN` /
///   `INST`, with the same insulation test (`SPRK.cpp:264-269`);
/// * a coloured, unpaused pipe at `life == 3` next to `HEAC`, which floods
///   `PFLAG_CAN_CONDUCT` through pipes of its own type (`PIPE.cpp:180-222` →
///   `PPIP.cpp:88`).
///
/// `PPIP.cpp:81` drops a spark whose pending flag is already set; here that is
/// "a member that has not applied the current flood yet". A seed flips the
/// generation tag, which is how the rest of the run knows the mark is new and
/// must be applied again.
///
/// The kernel reads its own and its neighbours' `part_flags` and writes only its
/// own slot; a neighbour read that lands before that neighbour's write can only
/// delay a round, never lose a mark, because the operation is a monotone OR.
#[cube(launch_unchecked)]
pub(super) fn pipe_flood_kernel(
    pmap: &[u32],
    part_tmp: &[i32],
    part_life: &[i32],
    part_ctype: &[u32],
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
        terminate!();
    }
    let slot = pmap_slot(v);
    let px = (idx as u32 % xres) as i32;
    let py = (idx as u32 / xres) as i32;
    let flags = part_flags[slot];
    let mut kind = flags & PIPE_FLOOD_KIND;
    let mut epoch = (flags >> PIPE_FLOOD_GEN_SHIFT) & 7u32;
    let mut applied = (flags & PIPE_FLOOD_APPLIED) != 0u32;

    // `PPIP.cpp:72-88`: the pending flags a seed carries. A seed is taken when
    // the pipe holds no pending flag, or once it has already applied the flags
    // of the flood that is currently marked on it.
    if kind == 0u32 || applied {
        let mut seed = sentinel(flags, 0u32);
        if ty == PT_PPIP {
            // `SPRK.cpp:264-269`: `sender` is the conductor the spark sits on.
            for k in 0u32..8u32 {
                let (dx, dy) = pipe_offset(k);
                let nv = pmap_at(pmap, xres, yres, px + dx, py + dy);
                if nv != 0u32 && (nv & PMAP_TYPE) == PT_SPRK {
                    let ns = pmap_slot(nv);
                    if part_life[ns] == 3i32
                        && !insulated(pmap, xres, yres, px, py, px + dx, py + dy)
                    {
                        let sender = part_ctype[ns];
                        if sender == PT_PSCN {
                            seed |= PIPE_FLOOD_ON;
                        } else if sender == PT_NSCN {
                            seed |= PIPE_FLOOD_OFF;
                        } else if sender == PT_INST {
                            seed |= PIPE_FLOOD_REVERSE;
                        }
                    }
                }
            }
        }
        let own_tmp = part_tmp[slot];
        if (own_tmp & PFLAG_COLORS) != 0i32
            && (own_tmp & PPIP_PAUSED) == 0i32
            && part_life[slot] == 3i32
        {
            // `PIPE.cpp:180-222`: a heat pipe next to the run makes the whole
            // run conduct.
            for k in 0u32..8u32 {
                let (dx, dy) = pipe_offset(k);
                let nv = pmap_at(pmap, xres, yres, px + dx, py + dy);
                if nv != 0u32 && (nv & PMAP_TYPE) == PT_HEAC {
                    seed |= PIPE_FLOOD_CONDUCT;
                }
            }
        }
        if seed != 0u32 {
            // The generation is a counter, so a member can tell a newer flood
            // from the one it has already applied.
            epoch = (epoch + 1u32) & (PIPE_FLOOD_GEN_MOD - 1u32);
            kind = seed;
            applied = false;
        }
    }

    // `PPIP.cpp:118-146`: the mark spreads to every 8-connected neighbour of the
    // same type. A neighbour from a newer generation hands its generation and
    // its flags over; within one generation the flags are ORed, exactly like the
    // reference ORs `prop` into the run.
    for k in 0u32..8u32 {
        let (dx, dy) = pipe_offset(k);
        let nv = pmap_at(pmap, xres, yres, px + dx, py + dy);
        if nv != 0u32 && (nv & PMAP_TYPE) == ty {
            let nf = part_flags[pmap_slot(nv)];
            let nkind = nf & PIPE_FLOOD_KIND;
            if nkind != 0u32 {
                let ngen = (nf >> PIPE_FLOOD_GEN_SHIFT) & 7u32;
                // "Newer" is a forward step of at most half the counter, so a
                // member up to four floods behind still catches up and a member
                // that is ahead never steps back.
                let ahead = (ngen + PIPE_FLOOD_GEN_MOD - epoch) & (PIPE_FLOOD_GEN_MOD - 1u32);
                if ahead != 0u32 && ahead <= (PIPE_FLOOD_GEN_MOD / 2u32) {
                    epoch = ngen;
                    kind = nkind;
                    applied = false;
                } else if ahead == 0u32 && !applied && (nkind & !kind) != 0u32 {
                    kind |= nkind;
                    applied = false;
                }
            }
        }
    }

    // A member drops its mark once every same-type neighbour shares its
    // generation: the whole run has been flooded, so the pending flags are
    // settled. The generation tag stays behind, which is what makes a later
    // flood recognisable as new; the mark itself has to go, or a pipe placed
    // next to the finished run would inherit its flags.
    if kind != 0u32 && applied {
        let mut settled = applied;
        for k in 0u32..8u32 {
            let (dx, dy) = pipe_offset(k);
            let nv = pmap_at(pmap, xres, yres, px + dx, py + dy);
            if nv != 0u32 && (nv & PMAP_TYPE) == ty {
                let nf = part_flags[pmap_slot(nv)];
                if ((nf >> PIPE_FLOOD_GEN_SHIFT) & 7u32) != epoch {
                    settled = false;
                }
            }
        }
        if settled {
            kind = 0u32;
        }
    }

    let mut out = (flags & !(PIPE_FLOOD_KIND | PIPE_FLOOD_GEN | PIPE_FLOOD_APPLIED))
        | kind
        | (epoch << PIPE_FLOOD_GEN_SHIFT);
    if applied {
        out |= PIPE_FLOOD_APPLIED;
    }
    part_flags[slot] = out;
}

/// `PIPE.cpp:165-232`: a coloured pipe at `life == 3` writes `nextColor` into
/// every initialising neighbour.
///
/// The gather asks the initialising pipe instead and takes the *last* such
/// neighbour in TPT's scan order, which is the value TPT's last write leaves
/// behind. When that neighbour is a one-wide run (`0x100`), the reference also
/// seeds the fixed-direction nibbles (`PIPE.cpp:186-199`): the initialising
/// pipe receives `0x200 | (7 - k) << 10` where `k` is the offset from it to the
/// source, and the source ORs `0x2000 | k << 14` into its own `tmp`. `k` here
/// is that same destination-side offset, so both values follow directly from
/// the gather; the two writes touch bits no other thread reads (the source is
/// tested for `PFLAG_COLORS` / pause / life only), so the pass stays
/// race-free.
///
/// The source's pause test uses the bit the pipe ends the frame with — the
/// trigger block (`PIPE.cpp:111-164`) runs before the colour branch, so a pipe
/// paused by this frame's trigger must not seed a cascade and one resumed by it
/// must (`PIPE.cpp:165`). [`with_pending_triggers`] folds the connectivity
/// flood's staged verdict the same way the state pass does, so the flood path
/// lands on the same frame too.
///
/// The answer is staged in `tmp3` — the cargo's `tmp`, which is always zero
/// for an initialising pipe — because the state pass writes its own `tmp` and
/// must not read a neighbour's `tmp` in the same dispatch. The only cross-slot
/// write lands on `tmp` bits 13-16, which neither the colour, pause nor life
/// test reads.
#[cube(launch_unchecked)]
pub(super) fn pipe_cascade_kernel(
    pmap: &[u32],
    part_tmp: &mut [i32],
    part_life: &[i32],
    part_tmp3: &mut [i32],
    part_flags: &[u32],
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
        terminate!();
    }
    let slot = pmap_slot(v);
    let tmp = part_tmp[slot];
    if (tmp & PFLAG_INITIALIZING) == 0i32 {
        terminate!();
    }
    let px = (idx as u32 % xres) as i32;
    let py = (idx as u32 / xres) as i32;

    let mut src = 0i32;
    let mut nibbles = 0i32;
    for k in 0u32..8u32 {
        let (dx, dy) = pipe_offset(k);
        let nv = pmap_at(pmap, xres, yres, px + dx, py + dy);
        if nv != 0u32 {
            let nt = nv & PMAP_TYPE;
            if nt == PT_PIPE || nt == PT_PPIP {
                let ns = pmap_slot(nv);
                // `PIPE.cpp:111-165`: the trigger block runs first, so the
                // colour branch sees the pause bit the pipe ends the frame
                // with. A pipe paused by this frame's trigger does not seed a
                // cascade, and one resumed by it does.
                let ntmp = with_pending_triggers(part_tmp[ns], part_flags[ns]);
                if (ntmp & PFLAG_COLORS) != 0i32
                    && !paused_after_trigger(ntmp)
                    && part_life[ns] == 3i32
                {
                    src = ntmp & PFLAG_COLORS;
                    if (ntmp & PFLAG_SINGLE_PIXEL) != 0i32 {
                        // `PIPE.cpp:186-199`: the destination is the source's
                        // `count - 1` offset; from here that is `7 - k`.
                        nibbles |= PFLAG_SINGLE_FORWARD | (((7u32 - k) as i32) << 10);
                        part_tmp[ns] |= 0x0000_2000i32 | ((k as i32) << 14);
                    }
                }
            }
        }
    }
    if src != 0i32 {
        part_tmp3[slot] = next_colour(src) | nibbles;
    }
}

/// `PIPE.cpp:114-125`: the pause bit the pipe ends the frame with.
///
/// `TRIGGER_ON` clears it and beats `TRIGGER_OFF`, which sets it; with neither
/// pending the bit is what the pipe already carries. The colour branch at
/// `PIPE.cpp:165` reads *this* value, not the pre-trigger one, so a pipe paused
/// this frame does not cascade and a pipe resumed this frame does.
#[cube]
fn paused_after_trigger(tmp: i32) -> bool {
    let mut paused = (tmp & PPIP_PAUSED) != 0i32;
    if (tmp & PPIP_TRIGGER_ON) != 0i32 {
        paused = false;
    } else if (tmp & PPIP_TRIGGER_OFF) != 0i32 {
        paused = true;
    }
    paused
}

/// `Simulation.cpp:3880-3890`: the pipe's `tmp` with the connectivity flood's
/// staged flags folded into the trigger bits it is about to apply.
///
/// The flood leaves its verdict in `part_flags` and [`pipe_state_kernel`] folds
/// it into `tmp`; every pass that needs the reference's pre-`Update` view of the
/// trigger bits folds it the same way. `PIPE_FLOOD_APPLIED` marks the
/// generation the state pass already consumed, so a settled flood is not
/// re-applied.
#[cube]
fn with_pending_triggers(tmp: i32, flags: u32) -> i32 {
    let mut out = tmp;
    let flood = flags & PIPE_FLOOD_KIND;
    if flood != 0u32 && (flags & PIPE_FLOOD_APPLIED) == 0u32 {
        if (flood & PIPE_FLOOD_ON) != 0u32 {
            out |= PPIP_TRIGGER_ON;
        }
        if (flood & PIPE_FLOOD_OFF) != 0u32 {
            out |= PPIP_TRIGGER_OFF;
        }
    }
    out
}

/// `PIPE.cpp:113-140`: the stasis marker a pause flip leaves on the border.
///
/// Returns `-1` when the pipe's pending trigger leaves the pause bit alone, `0`
/// when the pipe ends the frame paused and `1` when it ends the frame running —
/// the two values `PIPE.cpp:137-140` writes into the neighbouring `BRCK`'s
/// `tmp`. `TRIGGER_ON` beats `TRIGGER_OFF` (`PIPE.cpp:114`), and the reference
/// only writes when the bit actually flipped (`pause_changed`,
/// `PIPE.cpp:113-125`), which is what the before/after comparison reproduces.
#[cube]
#[allow(clippy::erasing_op)]
fn pause_marker(tmp: i32) -> i32 {
    let mut marker = (tmp & 0i32) - 1i32;
    let was_paused = (tmp & PPIP_PAUSED) != 0i32;
    let now_paused = paused_after_trigger(tmp);
    if now_paused != was_paused {
        marker = if now_paused { 0i32 } else { 1i32 };
    }
    marker
}

/// `PIPE.cpp:126-145`: mirror a pause flip onto the border bricks.
///
/// When a trigger changes the pause state, the reference writes
/// `parts[ID(r)].tmp = 0` into every `BRCK` of the surrounding 5x5 box if the
/// pipe ends up paused and `1` if it ends up running (`PIPE.cpp:128-141`);
/// `BRCK.cpp:48-59` turns that `1` into the additive glow the renderer draws
/// (`render_kernel/particles/pixel_mode.rs:313`). The box is the whole `rx`/`ry`
/// range, not the eight-neighbour ring, so a brick two pixels away is marked
/// too.
///
/// TPT mutates the brick from the pipe's update; here the brick gathers, so
/// each thread writes only its own slot. The pass runs before
/// [`pipe_state_kernel`], while the pipe still carries the pending trigger bits
/// — the state `pause_changed` is decided from — and folds the connectivity
/// flood's staged flags the way `Simulation.cpp:3880-3890` folds them into
/// `tmp` before `Update` runs. When two pipes cover the same brick, the last
/// one in TPT's `rx`-outer scan order decides, exactly like the reference's
/// last write.
#[cube(launch_unchecked)]
pub(super) fn pipe_brck_kernel(
    pmap: &[u32],
    part_tmp: &mut [i32],
    part_flags: &[u32],
    xres: u32,
    yres: u32,
) {
    let idx = ABSOLUTE_POS;
    if idx >= (xres * yres) as usize {
        terminate!();
    }
    let v = pmap[idx];
    if v == 0u32 || (v & PMAP_TYPE) != PT_BRCK {
        terminate!();
    }
    let slot = pmap_slot(v);
    let px = (idx as u32 % xres) as i32;
    let py = (idx as u32 / xres) as i32;
    let mut marker = sentinel(v, 0xFFFF_FFFFu32) as i32;
    // `k` walks TPT's `rx`-outer / `ry`-inner box, `rx`/`ry` in `-2..=2`.
    for k in 0u32..25u32 {
        let rx = (k / 5u32) as i32 - 2i32;
        let ry = (k % 5u32) as i32 - 2i32;
        if rx != 0i32 || ry != 0i32 {
            let nv = pmap_at(pmap, xres, yres, px + rx, py + ry);
            if nv != 0u32 {
                let nt = nv & PMAP_TYPE;
                if nt == PT_PIPE || nt == PT_PPIP {
                    let ns = pmap_slot(nv);
                    let ntmp = with_pending_triggers(part_tmp[ns], part_flags[ns]);
                    let m = pause_marker(ntmp);
                    if m >= 0i32 {
                        marker = m;
                    }
                }
            }
        }
    }
    if marker >= 0i32 {
        part_tmp[slot] = marker;
    }
}

/// `PIPE_update`'s state half (`PIPE.cpp:105-335`).
///
/// Reads and writes only the pipe's own `ctype` / `tmp` / `life` / payload and
/// its own spawn request; the only neighbour reads are `pmap` probes, so the
/// pass never sees a field another thread of the same dispatch is writing.
#[cube(launch_unchecked)]
pub(super) fn pipe_state_kernel(
    pmap: &[u32],
    part_ctype: &mut [u32],
    part_tmp: &mut [i32],
    part_life: &mut [i32],
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
    let pipe_ty = v & PMAP_TYPE;
    if pipe_ty != PT_PIPE && pipe_ty != PT_PPIP {
        terminate!();
    }

    let slot = pmap_slot(v);
    let px = (idx as u32 % xres) as i32;
    let py = (idx as u32 / xres) as i32;
    let mut tmp = part_tmp[slot];
    let mut life = part_life[slot];
    let mut ctype = part_ctype[slot];
    let mut flags = part_flags[slot];

    // `PIPE.cpp:106-107`: a cargo element that no longer exists is dropped.
    // `ElementDef::enabled` is a host-only flag, so the device test is the id
    // range.
    if ctype >= PT_NUM_U32 {
        ctype = 0u32;
    }

    // `PPIP.cpp:72-88` / `Simulation.cpp:3880-3890`: the connectivity flood
    // leaves its pending flags in `part_flags`; TPT keeps them in `tmp` bits
    // 29-31 and folds them into the active trigger bits before `Update` runs.
    // Each generation is applied once, so a member that the flood marks again
    // in a later frame does not toggle twice.
    let flood = flags & PIPE_FLOOD_KIND;
    if flood != 0u32 && (flags & PIPE_FLOOD_APPLIED) == 0u32 {
        if (flood & PIPE_FLOOD_ON) != 0u32 {
            tmp |= PPIP_TRIGGER_ON;
        }
        if (flood & PIPE_FLOOD_OFF) != 0u32 {
            tmp |= PPIP_TRIGGER_OFF;
        }
        if (flood & PIPE_FLOOD_REVERSE) != 0u32 {
            tmp |= PPIP_TRIGGER_REVERSE;
        }
        if (flood & PIPE_FLOOD_CONDUCT) != 0u32 {
            tmp |= PFLAG_CAN_CONDUCT;
        }
        flags |= PIPE_FLOOD_APPLIED;
    }

    // `PIPE.cpp:111-164`: the `PPIP` trigger bits. `TRIGGER_ON` beats
    // `TRIGGER_OFF`; `TRIGGER_REVERSE` swaps the flow direction.
    if (tmp & PPIP_TRIGGERS) != 0i32 {
        if (tmp & PPIP_TRIGGER_ON) != 0i32 {
            tmp &= !PPIP_PAUSED;
        }
        if (tmp & PPIP_TRIGGER_OFF) != 0i32 && (tmp & PPIP_TRIGGER_ON) == 0i32 {
            tmp |= PPIP_PAUSED;
        }
        if (tmp & PPIP_TRIGGER_REVERSE) != 0i32 {
            tmp = apply_reverse(tmp);
        }
        tmp &= !PPIP_TRIGGERS;
    }

    // `PIPE.cpp:165-232`: the colour cascade, decided by
    // [`pipe_cascade_kernel`] and applied here.
    let staged = part_tmp3[slot];
    if (tmp & PFLAG_INITIALIZING) != 0i32 && staged != 0i32 {
        tmp = (tmp & !(PFLAG_COLORS | PFLAG_INITIALIZING)) | staged;
        life = 6i32;
        ctype = 0u32;
    }

    // `PIPE.cpp:268-300`: while the colour is still unknown, seal the run with
    // `BRCK` and then wait for space. One request per frame, because
    // `FLAG_SPAWN_CLONE` carries a single pixel; the host clears the request bit
    // in the frame it materialises the particle, so a cleared bit means the
    // previous border pixel landed.
    let was_initializing = (tmp & PFLAG_INITIALIZING) != 0i32;
    if (tmp & (PFLAG_COLORS | PFLAG_INITIALIZING)) == 0i32 && life <= 10i32 {
        if life <= 1i32 {
            tmp |= PFLAG_INITIALIZING;
        }
        if (flags & FLAG_SPAWN_CLONE) == 0u32 {
            if ctype == PT_BRCK {
                ctype = 0u32;
            }
            if ctype == 0u32 && has_free_neighbour(pmap, xres, yres, px, py) {
                ctype = PT_BRCK;
                flags |= FLAG_SPAWN_CLONE;
            }
        }
    }

    // `PIPE.cpp:301-334`: the initialisation countdown. TPT's `else if` chain
    // means a pipe that entered the border branch above does not also run the
    // countdown in the same frame, so the decision is taken from the state the
    // pipe entered with.
    if was_initializing {
        if life == 0i32 && has_free_neighbour(pmap, xres, yres, px, py) {
            life = 50i32;
        }
        if life == 5i32 {
            // `PIPE.cpp:309-317`: a pipe with no pipe neighbour starts a
            // one-wide run. The marker the run carries; the fixed-direction
            // nibbles are seeded later, when this pipe's colour is read by an
            // initialising neighbour (`PIPE.cpp:186-199`, in
            // [`pipe_cascade_kernel`]).
            let mut single = life == 5i32;
            for k in 0u32..8u32 {
                let (dx, dy) = pipe_offset(k);
                let nv = pmap_at(pmap, xres, yres, px + dx, py + dy);
                if nv != 0u32 {
                    let nt = nv & PMAP_TYPE;
                    if nt == PT_PIPE || nt == PT_PPIP {
                        single = false;
                    }
                }
            }
            if single {
                tmp |= PFLAG_SINGLE_PIXEL;
            }
        }
        if life == 2i32 {
            tmp = (tmp & !PFLAG_INITIALIZING) | PFLAG_COLOR_RED;
            life = 6i32;
            ctype = 0u32;
        }
    }

    part_tmp[slot] = tmp;
    part_life[slot] = life;
    part_ctype[slot] = ctype;
    part_flags[slot] = flags;
    if ctype == 0u32 {
        // An empty pipe carries no payload, and `tmp3` doubles as the cascade
        // scratch: keeping all three zero means the next pickup starts clean.
        part_tmp2[slot] = 0i32;
        part_tmp3[slot] = 0i32;
        part_tmp4[slot] = 0i32;
    }
}

#[cfg(test)]
mod tests {
    use super::super::{PFLAG_COLOR_BLUE, PFLAG_SINGLE_FORWARD, pack};
    use super::*;

    #[test]
    fn reverse_swaps_colour_and_single_pixel_nibbles() {
        // `PIPE.cpp:147-162`.
        let tmp = PFLAG_SINGLE_PIXEL
            | PFLAG_SINGLE_FORWARD
            | (3i32 << 10)
            | 0x0000_2000
            | (5i32 << 14)
            | PFLAG_COLOR_RED;
        let out = apply_reverse(tmp);
        assert_eq!(out & PPIP_REVERSED, PPIP_REVERSED);
        // red ^ green = blue: the flow cycle now runs the other way.
        assert_eq!(out & PFLAG_COLORS, PFLAG_COLOR_BLUE);
        // The nibble at 9 and the nibble at 13 swapped.
        assert_eq!((out >> 10) & 7, 5);
        assert_eq!((out >> 14) & 7, 3);
        assert_eq!(out & PFLAG_SINGLE_FORWARD, PFLAG_SINGLE_FORWARD);
        assert_eq!(out & 0x0000_2000, 0x0000_2000);

        // Green is its own partner, and a pipe without the single-pixel marker
        // keeps its direction bits.
        let green = apply_reverse(PFLAG_COLOR_GREEN);
        assert_eq!(green & PFLAG_COLORS, PFLAG_COLOR_GREEN);
        let plain = apply_reverse(PFLAG_COLOR_BLUE | (2i32 << 10));
        assert_eq!(plain & PFLAG_COLORS, PFLAG_COLOR_RED);
        assert_eq!((plain >> 10) & 7, 2);
        // Reversing twice restores the colour (the `REVERSED` bit stays set,
        // exactly like TPT's `^=`).
        assert_eq!(
            apply_reverse(apply_reverse(PFLAG_COLOR_RED)) & PFLAG_COLORS,
            PFLAG_COLOR_RED
        );
    }

    #[test]
    fn pause_marker_matches_the_reference_pause_flip() {
        // `PIPE.cpp:113-140`: only a flip of the pause bit writes, and the
        // value is the state the pipe ends the frame in (`0` = paused, `1` =
        // running/glowing).
        assert_eq!(pause_marker(PFLAG_COLOR_RED), -1);
        // `TRIGGER_OFF` on a running pipe pauses it, so the bricks dim.
        assert_eq!(pause_marker(PFLAG_COLOR_RED | PPIP_TRIGGER_OFF), 0);
        // `TRIGGER_OFF` on an already paused pipe changes nothing.
        assert_eq!(
            pause_marker(PFLAG_COLOR_RED | PPIP_PAUSED | PPIP_TRIGGER_OFF),
            -1
        );
        // `TRIGGER_ON` on a paused pipe resumes it, so the bricks glow.
        assert_eq!(
            pause_marker(PFLAG_COLOR_RED | PPIP_PAUSED | PPIP_TRIGGER_ON),
            1
        );
        // `TRIGGER_ON` on a running pipe changes nothing.
        assert_eq!(pause_marker(PFLAG_COLOR_RED | PPIP_TRIGGER_ON), -1);
        // `PIPE.cpp:114`: `TRIGGER_ON` overrides `TRIGGER_OFF`.
        assert_eq!(
            pause_marker(PFLAG_COLOR_RED | PPIP_PAUSED | PPIP_TRIGGER_ON | PPIP_TRIGGER_OFF),
            1
        );
        assert_eq!(
            pause_marker(PFLAG_COLOR_RED | PPIP_TRIGGER_ON | PPIP_TRIGGER_OFF),
            -1
        );
    }

    #[test]
    fn free_space_gate_sees_the_ring_only() {
        let xres = 4u32;
        let yres = 4u32;
        let mut pmap = vec![0u32; (xres * yres) as usize];
        // Fill the whole canvas except the ring around (1, 1).
        for y in 0..yres {
            for x in 0..xres {
                pmap[(y * xres + x) as usize] = pack(0, PT_BRCK);
            }
        }
        assert!(!has_free_neighbour(&pmap, xres, yres, 1, 1));
        pmap[0] = 0;
        assert!(has_free_neighbour(&pmap, xres, yres, 1, 1));
        // A free pixel two steps away is not a neighbour.
        pmap[0] = pack(0, PT_BRCK);
        pmap[(3 * xres + 3) as usize] = 0;
        assert!(!has_free_neighbour(&pmap, xres, yres, 1, 1));
        assert!(has_free_neighbour(&pmap, xres, yres, 2, 2));
    }
}
