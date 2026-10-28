//! Switches, powered devices and liquid crystals.

use cubecl::prelude::*;
use tpt_core::constants::*;

use super::super::shared::{
    CR_SET_CTYPE, CR_SET_VX, CR_SET_VY, FLAG_SPAWN_CLONE, hash_u32, iabs, insulated, pmap_at,
    pmap_slot, request_create,
};
use super::{POWERED_ON_LIFE, SPARK_LIFE, SWCH_OFF_LIFE, SWCH_RELEASE_LIFE, powered_toggle_life};

/// `PCLN.cpp` / `PBCN.cpp`: `PT_LIGH` clones spawn on one frame in 30.
const LIGH_CLONE_CHANCE: u32 = 30;

/// `SWCH.cpp`: returns `true` if pixel `(x, y)` holds red BRAY (`tmp == 2`).
#[cube]
fn is_red_bray(pmap: &[u32], part_tmp: &mut [i32], xres: u32, yres: u32, x: i32, y: i32) -> bool {
    let v = pmap_at(pmap, xres, yres, x, y);
    let mut ok = false;
    if v != 0u32 {
        let ty = v & 0x1FFu32;
        if ty == PT_BRAY {
            let slot = pmap_slot(v);
            if part_tmp[slot] == 2i32 {
                ok = true;
            }
        }
    }
    ok
}

/// `SWCH` / `HSWC` / `PVOD` / `PCLN` / `PBCN` / `BCLN` / `LCRY` updates.
///
/// Pixel parallel over the live state: the chain activations it writes
/// (`life = 10`, `life = 9`, `tmp = 2`) are idempotent, so concurrent writers
/// always agree and the pass needs no snapshot.
///
/// `HSWC` (`HSWC.cpp:66-102`) has two states: a releasing switch counts down,
/// while a latched one (`life == 10`) relays the latch across its radius-2
/// ring and, with `tmp == 1`, deserialises a `FILT`'s `0x10000000 + temp` into
/// its own temperature.
///
/// `BCLN` (`BCLN.cpp:52-100`): the pressure-armed break timer
/// (`life = 80..119`), the air drag while it runs, the `ctype` latch with its
/// own exclusion list over both the particle and the energy layer, the `tmp`
/// copy for a `LIFE` / `LAVA` source, and the per-frame clone request. `PBCN`
/// (`PBCN.cpp:55-70`) shares the drag and its own break timer.
#[cube(launch_unchecked)]
#[allow(clippy::too_many_arguments)] // one argument per device field, no structs in kernels
#[allow(clippy::manual_range_contains)] // `contains` is not available inside kernels
pub(crate) fn switch_device_kernel(
    pmap: &[u32],
    part_ty: &mut [u32],
    part_life: &mut [i32],
    part_temp: &mut [f32],
    part_ctype: &mut [u32],
    part_tmp: &mut [i32],
    part_tmp2: &mut [i32],
    part_flags: &mut [u32],
    air_pv: &mut [f32],
    photons: &[u32],
    part_vx: &mut [f32],
    part_vy: &mut [f32],
    air_vx: &[f32],
    air_vy: &[f32],
    create_queue: &mut [Atomic<u32>],
    xres: u32,
    yres: u32,
    xcells: u32,
    frame_seed: u32,
) {
    let idx = ABSOLUTE_POS;
    if idx < (xres * yres) as usize {
        let v = pmap[idx];
        if v != 0u32 {
            let id = pmap_slot(v);
            let ty = part_ty[id];
            let powered =
                ty == PT_SWCH || ty == PT_HSWC || ty == PT_PVOD || ty == PT_PCLN || ty == PT_PBCN;
            if powered || ty == PT_LCRY || ty == PT_BCLN {
                let px = (idx as u32 % xres) as i32;
                let py = (idx as u32 / xres) as i32;

                if ty == PT_PBCN {
                    // `PBCN.cpp`: pressure damage timer, independent of the
                    // activation state.
                    let mut timer = part_tmp2[id];
                    if timer == 0i32 {
                        let cidx = (((py as u32) / CELL) * xcells + ((px as u32) / CELL)) as usize;
                        if air_pv[cidx] > 4.0f32 {
                            timer = 80i32 + ((hash_u32((id as u32) ^ frame_seed) % 40u32) as i32);
                            part_tmp2[id] = timer;
                        }
                    } else {
                        // `PBCN.cpp:63-64`: the air drags the block while its
                        // timer runs.
                        let cidx = (((py as u32) / CELL) * xcells + ((px as u32) / CELL)) as usize;
                        part_vx[id] += 0.1f32 * air_vx[cidx];
                        part_vy[id] += 0.1f32 * air_vy[cidx];
                        timer -= 1i32;
                        part_tmp2[id] = timer;
                        if timer == 0i32 {
                            part_ty[id] = PT_NONE;
                        }
                    }
                }

                // Set when `BCLN` latched a new element this frame: the
                // reference latches and clones in exclusive branches, so the
                // latch frame produces no copy (`BCLN.cpp:63-99`).
                let mut latched = false;
                if ty == PT_BCLN {
                    // `BCLN.cpp:52-100`. The clone arms its break timer from
                    // the cell pressure, then latches the element it copies
                    // from the first neighbour that is not another clone.
                    let mut life = part_life[id];
                    if life == 0i32 {
                        let cidx = (((py as u32) / CELL) * xcells + ((px as u32) / CELL)) as usize;
                        if air_pv[cidx] > 4.0f32 {
                            // `BCLN.cpp:56-57`: `rng.between(80, 119)`.
                            life = 80i32 + (hash_u32((id as u32) ^ frame_seed) % 40u32) as i32;
                            part_life[id] = life;
                        }
                    }
                    if life != 0i32 {
                        // `BCLN.cpp:59-62`: the air drags the block while its
                        // break timer runs.
                        let cidx = (((py as u32) / CELL) * xcells + ((px as u32) / CELL)) as usize;
                        part_vx[id] += 0.1f32 * air_vx[cidx];
                        part_vy[id] += 0.1f32 * air_vy[cidx];
                    }
                    if part_ctype[id] >= PT_NUM as u32 {
                        part_ctype[id] = PT_NONE;
                    }
                    if part_ctype[id] == PT_NONE {
                        // `BCLN.cpp:63-86`: the exclusion list names the
                        // clones, the stickmen and `PBCN`; unlike `PCLN` it
                        // does not skip sparks or scanners.
                        for n in 0u32..9u32 {
                            let dx = (n % 3u32) as i32 - 1i32;
                            let dy = (n / 3u32) as i32 - 1i32;
                            if dx != 0i32 || dy != 0i32 {
                                // `BCLN.cpp:69-70`: the energy layer wins over
                                // the particle layer at the same pixel.
                                let mut nv = pmap_at(photons, xres, yres, px + dx, py + dy);
                                if nv == 0u32 {
                                    nv = pmap_at(pmap, xres, yres, px + dx, py + dy);
                                }
                                if nv != 0u32 && part_ctype[id] == PT_NONE {
                                    let nid = pmap_slot(nv);
                                    let nty = part_ty[nid];
                                    if nty != PT_NONE
                                        && nty != PT_CLNE
                                        && nty != PT_PCLN
                                        && nty != PT_BCLN
                                        && nty != PT_STKM
                                        && nty != PT_PBCN
                                        && nty != PT_STKM2
                                        && nty < PT_NUM as u32
                                    {
                                        part_ctype[id] = nty;
                                        // `BCLN.cpp:81-82`: a `LIFE` or `LAVA`
                                        // source carries its own `ctype` into
                                        // the clone's `tmp`.
                                        if nty == PT_LIFE || nty == PT_LAVA {
                                            part_tmp[id] = part_ctype[nid] as i32;
                                        }
                                        latched = true;
                                    }
                                }
                            }
                        }
                    }
                } else if powered {
                    // Life countdown; a device at exactly 10 is latched on.
                    let life = part_life[id];
                    if ty == PT_HSWC {
                        // `HSWC.cpp:66-102`: a latched heat switch does not
                        // count down. Its radius-2 ring instead relays the
                        // latch to neighbouring heat switches and — with
                        // `tmp == 1` — reads a `FILT`'s serialised
                        // temperature. The reference has no insulation test
                        // and no diagonal gate on the relay.
                        if life == POWERED_ON_LIFE {
                            let deserialize_temp = part_tmp[id] == 1i32;
                            for n in 0u32..25u32 {
                                let dx = (n % 5u32) as i32 - 2i32;
                                let dy = (n / 5u32) as i32 - 2i32;
                                if dx != 0i32 || dy != 0i32 {
                                    let nx = px + dx;
                                    let ny = py + dy;
                                    // `HSWC.cpp:78-80`: the particle layer
                                    // wins over the energy layer.
                                    let mut nv = pmap_at(pmap, xres, yres, nx, ny);
                                    if nv == 0u32 {
                                        nv = pmap_at(photons, xres, yres, nx, ny);
                                    }
                                    if nv != 0u32 {
                                        let nid = pmap_slot(nv);
                                        if part_ty[nid] == PT_HSWC {
                                            let nlife = part_life[nid];
                                            if nlife < POWERED_ON_LIFE && nlife > 0i32 {
                                                // `HSWC.cpp:71-72`: the
                                                // neighbour is already
                                                // releasing, so this switch
                                                // follows it down.
                                                part_life[id] = SWCH_OFF_LIFE;
                                            } else if nlife == 0i32 {
                                                // `HSWC.cpp:73-74`: a quiet
                                                // neighbour follows the
                                                // latched switch.
                                                part_life[nid] = POWERED_ON_LIFE;
                                            }
                                        }
                                        if deserialize_temp
                                            && part_ty[nid] == PT_FILT
                                            && iabs(dx) <= 1i32
                                            && iabs(dy) <= 1i32
                                        {
                                            // `HSWC.cpp:82-87`: the inner 3x3
                                            // `FILT` carries
                                            // `0x10000000 + temperature`.
                                            let new_temp =
                                                (part_ctype[nid] as i32) - 0x1000_0000i32;
                                            if new_temp >= MIN_TEMP as i32
                                                && new_temp <= MAX_TEMP as i32
                                            {
                                                part_temp[id] = new_temp as f32;
                                            }
                                        }
                                    }
                                }
                            }
                        } else if life > 0i32 {
                            part_life[id] = life - 1i32;
                        }
                    } else if life > 0i32 && life != POWERED_ON_LIFE {
                        part_life[id] = life - 1i32;
                    }

                    // `PCLN.cpp` / `PBCN.cpp`: latch the element to clone from
                    // the first neighbour that is not another clone.
                    if (ty == PT_PCLN || ty == PT_PBCN) && part_ctype[id] >= PT_NUM as u32 {
                        part_ctype[id] = PT_NONE;
                    }
                    if (ty == PT_PCLN || ty == PT_PBCN) && part_ctype[id] == PT_NONE {
                        for n in 0u32..9u32 {
                            let dx = (n % 3u32) as i32 - 1i32;
                            let dy = (n / 3u32) as i32 - 1i32;
                            let nv = pmap_at(pmap, xres, yres, px + dx, py + dy);
                            if nv != 0u32 && part_ctype[id] == PT_NONE {
                                let nty = part_ty[pmap_slot(nv)];
                                if nty != PT_NONE
                                    && nty != PT_CLNE
                                    && nty != PT_PCLN
                                    && nty != PT_BCLN
                                    && nty != PT_SPRK
                                    && nty != PT_NSCN
                                    && nty != PT_PSCN
                                    && nty != PT_STKM
                                    && nty != PT_STKM2
                                    && nty != PT_PBCN
                                    && nty < PT_NUM as u32
                                {
                                    part_ctype[id] = nty;
                                }
                            }
                        }
                    }

                    // Chain activation and spark input, both radius 2.
                    let mut life_now = part_life[id];
                    let mut spark_in = 0u32;
                    let mut touch = 0u32;
                    for n in 0u32..25u32 {
                        let dx = (n % 5u32) as i32 - 2i32;
                        let dy = (n / 5u32) as i32 - 2i32;
                        if dx != 0i32 || dy != 0i32 {
                            let nx = px + dx;
                            let ny = py + dy;
                            let nv = pmap_at(pmap, xres, yres, nx, ny);
                            if nv != 0u32 && !insulated(pmap, xres, yres, nx, ny, px, py) {
                                let nslot = pmap_slot(nv);
                                let nty = part_ty[nslot];
                                let nlife = part_life[nslot];
                                if nty == ty {
                                    if ty == PT_SWCH {
                                        if life_now >= POWERED_ON_LIFE
                                            && nlife < POWERED_ON_LIFE
                                            && nlife > 0i32
                                        {
                                            life_now = SWCH_OFF_LIFE;
                                        } else if life_now == 0i32 && nlife >= POWERED_ON_LIFE {
                                            life_now = nlife;
                                        }
                                    } else if ty != PT_HSWC
                                        && life_now == POWERED_ON_LIFE
                                        && nlife < POWERED_ON_LIFE
                                        && nlife > 0i32
                                    {
                                        life_now = SWCH_OFF_LIFE;
                                    } else if ty != PT_HSWC
                                        && life_now == 0i32
                                        && nlife == POWERED_ON_LIFE
                                    {
                                        part_life[nslot] = POWERED_ON_LIFE;
                                    }
                                } else if nty == PT_SPRK && nlife > 0i32 {
                                    let sender = part_ctype[nslot];
                                    if nlife < SPARK_LIFE {
                                        if ty == PT_SWCH {
                                            // `SPRK.cpp` `case PT_SWCH`
                                            if sender == PT_PSCN && life_now < POWERED_ON_LIFE {
                                                life_now = POWERED_ON_LIFE;
                                            } else if sender == PT_NSCN {
                                                part_ctype[id] = PT_NONE;
                                                life_now = SWCH_OFF_LIFE;
                                            }
                                        } else if spark_in == 0u32 {
                                            // `SPRK.cpp` `case PT_PUMP/GPMP/HSWC/PBCN`
                                            spark_in = sender;
                                        }
                                    }
                                    // Any spark, whatever its age, can trigger
                                    // the `SWCH.cpp` conversion below.
                                    if touch == 0u32 && sender != PT_PSCN && sender != PT_NSCN {
                                        touch = sender;
                                    }
                                }
                            }
                        }
                    }
                    if spark_in != 0u32 {
                        if ty == PT_PVOD || ty == PT_PCLN {
                            // `PVOD.cpp` / `PCLN.cpp` switch unconditionally.
                            if spark_in == PT_PSCN {
                                life_now = POWERED_ON_LIFE;
                            } else if spark_in == PT_NSCN {
                                life_now = SWCH_OFF_LIFE;
                            }
                        } else {
                            life_now = powered_toggle_life(life_now, spark_in);
                        }
                    }
                    if ty == PT_SWCH {
                        // `SWCH.cpp:91-98`: two red BRAYS arriving along a diagonal
                        // beam toggle the switch on/off.
                        let top_left_empty =
                            pmap_at(pmap, xres, yres, px - 1i32, py - 1i32) == 0u32;
                        let top_right_empty =
                            pmap_at(pmap, xres, yres, px + 1i32, py - 1i32) == 0u32;
                        if top_left_empty && top_right_empty {
                            let vert_bray = is_red_bray(pmap, part_tmp, xres, yres, px, py - 1i32)
                                || is_red_bray(pmap, part_tmp, xres, yres, px, py + 1i32);
                            let horiz_bray = is_red_bray(pmap, part_tmp, xres, yres, px - 1i32, py)
                                || is_red_bray(pmap, part_tmp, xres, yres, px + 1i32, py);
                            if vert_bray && horiz_bray {
                                if life_now == POWERED_ON_LIFE {
                                    life_now = SWCH_OFF_LIFE;
                                } else if life_now <= 5i32 {
                                    life_now = SWCH_RELEASE_LIFE;
                                }
                            }
                        }
                    }
                    if part_ty[id] == ty {
                        part_life[id] = life_now;
                    }
                    // A switch that is on turns into a spark when an unrelated
                    // spark touches it (`SWCH.cpp`).
                    if ty == PT_SWCH && life_now == POWERED_ON_LIFE && touch != 0u32 {
                        part_ty[id] = PT_SPRK;
                        part_ctype[id] = PT_SWCH;
                        part_life[id] = SPARK_LIFE;
                    }
                } else {
                    // `LCRY.cpp`: `tmp` is the charge direction, `life` the
                    // charge level (0..10, two steps per frame), `tmp2` the
                    // brightness the renderer uses.
                    let mut tmp = part_tmp[id];
                    let mut life = part_life[id];
                    let mut check = 0i32;
                    let mut setto = 0i32;
                    if tmp == 1i32 {
                        if life <= 0i32 {
                            tmp = 0i32;
                        } else {
                            life -= 2i32;
                            if life < 0i32 {
                                life = 0i32;
                            }
                            part_tmp2[id] = life;
                        }
                    }
                    if tmp == 0i32 || tmp == 1i32 {
                        check = 3i32;
                        setto = 1i32;
                    } else if tmp == 2i32 {
                        if life >= POWERED_ON_LIFE {
                            tmp = 3i32;
                        } else {
                            life += 2i32;
                            if life > POWERED_ON_LIFE {
                                life = POWERED_ON_LIFE;
                            }
                            part_tmp2[id] = life;
                        }
                    }
                    if tmp == 2i32 || tmp == 3i32 {
                        check = 0i32;
                        setto = 2i32;
                    }
                    if tmp < 0i32 || tmp > 3i32 {
                        // `LCRY.cpp` `default:`: a corrupt charge state resets.
                        tmp = 0i32;
                        life = 0i32;
                    }
                    part_tmp[id] = tmp;
                    part_life[id] = life;
                    // `SPRK.cpp` `case PT_LCRY`: a nearby spark sets the charge
                    // direction of an idle crystal.
                    for n in 0u32..25u32 {
                        let dx = (n % 5u32) as i32 - 2i32;
                        let dy = (n / 5u32) as i32 - 2i32;
                        if iabs(dx) < 2i32 && iabs(dy) < 2i32 && (dx != 0i32 || dy != 0i32) {
                            let nv = pmap_at(pmap, xres, yres, px + dx, py + dy);
                            if nv != 0u32 {
                                let nslot = pmap_slot(nv);
                                let nlife = part_life[nslot];
                                if part_ty[nslot] == PT_SPRK && nlife > 0i32 && nlife < SPARK_LIFE {
                                    let sender = part_ctype[nslot];
                                    if sender == PT_PSCN && tmp == 0i32 {
                                        tmp = 2i32;
                                    } else if sender == PT_NSCN && tmp == 3i32 {
                                        tmp = 1i32;
                                    }
                                }
                            }
                        }
                    }
                    part_tmp[id] = tmp;
                    // Charge spreads to neighbours in the opposite direction.
                    for n in 0u32..9u32 {
                        let dx = (n % 3u32) as i32 - 1i32;
                        let dy = (n / 3u32) as i32 - 1i32;
                        if dx != 0i32 || dy != 0i32 {
                            let nv = pmap_at(pmap, xres, yres, px + dx, py + dy);
                            if nv != 0u32 {
                                let nslot = pmap_slot(nv);
                                if part_ty[nslot] == PT_LCRY && part_tmp[nslot] == check {
                                    part_tmp[nslot] = setto;
                                }
                            }
                        }
                    }
                }

                // `PCLN.cpp:113-160` / `PBCN.cpp:117-145` / `BCLN.cpp:96-111`:
                // while on, the clone produces a copy per frame. `PHOT` and
                // `LIFE` payloads have their own creation shapes — eight
                // directed photons or nine genome copies — instead of the
                // single neighbouring clone. `BCLN.cpp:86-99` copies on every
                // frame its `ctype` is valid, with the same one-in-thirty
                // `LIGH` gate.
                let ty_now = part_ty[id];
                let mut spawn_clone = false;
                if ty_now == PT_BCLN {
                    spawn_clone = !latched;
                } else if ty_now == PT_PCLN || ty_now == PT_PBCN {
                    spawn_clone = part_life[id] == POWERED_ON_LIFE;
                }
                if spawn_clone {
                    let ct = part_ctype[id];
                    if ct != PT_NONE && ct < PT_NUM as u32 {
                        let mut generic = true;
                        if ty_now != PT_BCLN && ct == PT_PHOT {
                            // `PCLN.cpp:121-141` / `PBCN.cpp:117-137`: one
                            // photon in each of the eight neighbours, moving
                            // along the offset at three pixels per frame. The
                            // reference also sets `FLAG_SKIPMOVE` when the new
                            // slot is above the clone's; the queue drain has no
                            // slot-order relationship, so the photons here
                            // move on their creation frame.
                            generic = false;
                            for n in 0u32..9u32 {
                                let dx = (n % 3u32) as i32 - 1i32;
                                let dy = (n / 3u32) as i32 - 1i32;
                                if dx != 0i32 || dy != 0i32 {
                                    request_create(
                                        create_queue,
                                        PT_PHOT,
                                        px + dx,
                                        py + dy,
                                        CR_SET_VX | CR_SET_VY,
                                        0i32,
                                        0u32,
                                        0.0f32,
                                        0i32,
                                        0i32,
                                        0i32,
                                        0i32,
                                        0u32,
                                        0u32,
                                        (dx * 3i32) as f32,
                                        (dy * 3i32) as f32,
                                    );
                                }
                            }
                        } else if ct == PT_LIFE && ty_now != PT_BCLN {
                            // `PCLN.cpp:142-152` / `PBCN.cpp:138-148`: every
                            // cell of the 3x3 block — the clone's own included
                            // — receives a `LIFE` whose `ctype` is the clone's
                            // `tmp` (`create_part(-1, ..., PT_LIFE, tmp)`).
                            generic = false;
                            for n in 0u32..9u32 {
                                let dx = (n % 3u32) as i32 - 1i32;
                                let dy = (n / 3u32) as i32 - 1i32;
                                request_create(
                                    create_queue,
                                    PT_LIFE,
                                    px + dx,
                                    py + dy,
                                    CR_SET_CTYPE,
                                    0i32,
                                    part_tmp[id] as u32,
                                    0.0f32,
                                    0i32,
                                    0i32,
                                    0i32,
                                    0i32,
                                    0u32,
                                    0u32,
                                    0.0f32,
                                    0.0f32,
                                );
                            }
                        } else if ct == PT_LIFE {
                            // `BCLN.cpp:99`: a single `LIFE` at a random
                            // offset, carrying the clone's `tmp` as genome.
                            generic = false;
                            let rx = (hash_u32(frame_seed ^ (id as u32) ^ 0x1D2C_3B4Au32) % 3u32)
                                as i32
                                - 1i32;
                            let ry = (hash_u32(frame_seed ^ (id as u32) ^ 0x7F4A_7C15u32) % 3u32)
                                as i32
                                - 1i32;
                            request_create(
                                create_queue,
                                PT_LIFE,
                                px + rx,
                                py + ry,
                                CR_SET_CTYPE,
                                0i32,
                                part_tmp[id] as u32,
                                0.0f32,
                                0i32,
                                0i32,
                                0i32,
                                0i32,
                                0u32,
                                0u32,
                                0.0f32,
                                0.0f32,
                            );
                        }
                        if generic {
                            let mut allowed = true;
                            if ct == PT_LIGH {
                                allowed = hash_u32(frame_seed ^ (id as u32))
                                    .is_multiple_of(LIGH_CLONE_CHANCE);
                            }
                            if allowed {
                                part_flags[id] |= FLAG_SPAWN_CLONE;
                            }
                        }
                    }
                }
            }
        }
    }
}
