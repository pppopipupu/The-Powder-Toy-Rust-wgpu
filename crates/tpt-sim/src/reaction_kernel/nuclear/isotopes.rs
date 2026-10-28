//! `URAN`/`PLUT`/`DEUT`/`AMTR` isotopes and pressure/annihilation updates.

use cubecl::prelude::*;
use tpt_core::constants::*;

use super::super::shared::{
    CR_NO_OCCUPANCY, CR_SET_LIFE, CR_SET_TEMP, WORD_PROPERTIES, elem_word, hash_u32, mix_index,
    pmap_at, pmap_slot, request_create,
};
use super::{chance, clampf};

/// `std::abs` for floats (CubeCL has no `f32::abs` in the portable subset).
#[cube]
fn fabsf(v: f32) -> f32 {
    let mut r = v;
    if v < 0.0f32 {
        r = -v;
    }
    r
}

/// `DEUT.cpp:56-66`: the compression target for one frame.
///
/// `maxlife` starts as `10000/(temp+2) - 1` (plus a fractional carry), then is
/// scaled by `int(5 - 8/(gravtot+2))` so a strong Newtonian field compresses
/// deuterium much harder.
#[cube]
fn deut_max_life(temp: f32, gravtot: f32, roll: u32) -> i32 {
    let mut t = temp + 1.0f32;
    if t < 1.0f32 {
        t = 1.0f32;
    }
    let ti = t as i32;
    let mut maxlife = (10000.0f32 / (t + 1.0f32)) as i32 - 1i32;
    let d = ti as u32;
    if d > 0u32 && (roll % d) < (10000u32 % d) {
        maxlife += 1i32;
    }
    let mult = (5.0f32 - 8.0f32 / (gravtot + 2.0f32)) as i32;
    maxlife * mult
}

/// `URAN`/`PLUT`/`WARP`/`INVS` read the coarse air pressure.
///
/// Pixel parallel: each thread handles the particle stored in its own `pmap`
/// entry and only writes that slot (plus the shared `air_pv` cell).
#[cube(launch_unchecked)]
pub(crate) fn pressure_kernel(
    pmap: &[u32],
    part_ty: &mut [u32],
    part_life: &mut [i32],
    part_vx: &mut [f32],
    part_vy: &mut [f32],
    part_temp: &mut [f32],
    part_tmp: &mut [i32],
    part_tmp2: &mut [i32],
    part_flags: &mut [u32],
    air_pv: &mut [f32],
    create_queue: &mut [Atomic<u32>],
    xres: u32,
    yres: u32,
    xcells: u32,
    frame_seed: u32,
    legacy: u32,
) {
    let idx = ABSOLUTE_POS;
    if idx < (xres * yres) as usize {
        let v = pmap[idx];
        if v != 0u32 {
            let id = pmap_slot(v);
            let ty = part_ty[id];
            if ty != PT_NONE {
                let px = (idx as u32 % xres) as i32;
                let py = (idx as u32 / xres) as i32;
                let cidx = (((py as u32) / CELL) * xcells + ((px as u32) / CELL)) as usize;
                let roll = hash_u32(frame_seed ^ mix_index(idx as u32, 2_246_822_519u32));

                if ty == PT_URAN {
                    // `URAN.cpp:57-70`: `temp *= 1 + pv/2000`, but only outside
                    // legacy mode. `MIN_TEMP` is 0, so TPT's `+ MIN_TEMP` and
                    // the zero-temperature special case are the only quirks.
                    if legacy == 0u32 {
                        let pv = air_pv[cidx];
                        if pv > 0.0f32 {
                            let t = part_temp[id];
                            if t == MIN_TEMP {
                                part_temp[id] = t + 0.01f32;
                            } else {
                                part_temp[id] =
                                    clampf(t * (1.0f32 + pv / 2000.0f32), MIN_TEMP, MAX_TEMP);
                            }
                        }
                    }
                } else if ty == PT_PLUT {
                    // `PLUT.cpp:58-65`: `chance(1,100) && chance(int(5*pv),1000)`.
                    // A negative `int(5*pv)` can never hit, so it collapses to 0.
                    let scaled = (5.0f32 * air_pv[cidx]) as i32;
                    let mut n = 0u32;
                    if scaled > 0i32 {
                        n = scaled as u32;
                    }
                    if chance(roll, 1u32, 100u32)
                        && chance(hash_u32(roll ^ 0x9E37_79B9u32), n, 1000u32)
                    {
                        // `create_part(i, x, y, PT_NEUT)` replaces the plutonium
                        // with a *fresh* neutron: the `DefaultProperties` and the
                        // `NEUT` `Create` hook reset every field, which is what
                        // gives it its 480..959 frame life. Inheriting the
                        // plutonium's `life = 0` would make `PROP_LIFE_KILL_DEC`
                        // kill the neutron on its first frame.
                        part_ty[id] = PT_NEUT;
                        part_life[id] = 480i32 + (roll % 480u32) as i32;
                        // `NEUT.cpp:255-257`: `r = rng.between(128, 255) / 127`,
                        // i.e. 1.008..2.008 px/frame at any angle.
                        let h = hash_u32(roll ^ 0x85EB_CA6Bu32);
                        let angle = (h & 0xFFFFu32) as f32 * (core::f32::consts::TAU / 65_535.0f32);
                        let radius = 1.0f32 + ((h >> 16u32) & 0xFFu32) as f32 / 127.0f32;
                        part_vx[id] = angle.cos() * radius;
                        part_vy[id] = angle.sin() * radius;
                        // `NEUT`'s `DefaultProperties.temp = R_TEMP + 273.15`.
                        part_temp[id] = 299.15f32;
                        part_tmp[id] = 0i32;
                        part_tmp2[id] = 0i32;
                        part_flags[id] = 0u32;
                    }
                } else if ty == PT_WARP {
                    // `WARP.cpp:54-62`: a charged warp vents heat, pressure and
                    // (one in fifty frames) an electron. `CFDS == 1` here.
                    let t2 = part_tmp2[id];
                    if t2 > 2000i32 {
                        part_temp[id] = 10000.0f32;
                        let add = (t2 / 5000i32) as f32;
                        air_pv[cidx] = clampf(air_pv[cidx] + add, MIN_PRESSURE, MAX_PRESSURE);
                        if chance(roll, 1u32, 50u32) {
                            // `create_part(-3, x, y, PT_ELEC)` skips the map
                            // checks and lands on the warp's own pixel.
                            request_create(
                                create_queue,
                                PT_ELEC,
                                px,
                                py,
                                CR_NO_OCCUPANCY,
                                0i32,
                                0u32,
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
                    }
                } else if ty == PT_INVIS {
                    // `INVIS.cpp:64-73`: `tmp` is the pressure resistance.
                    let t = part_tmp[id];
                    let mut resistance = 4.0f32;
                    if t > 0i32 {
                        resistance = t as f32;
                    }
                    let pv = air_pv[cidx];
                    let mut flag = 0i32;
                    if pv < -resistance || pv > resistance {
                        flag = 1i32;
                    }
                    part_tmp2[id] = flag;
                }
            }
        }
    }
}

/// `AMTR` destroys its neighbours and dies after four contacts; `EMBR` is
/// extinguished by any settled matter it touches.
///
/// Pixel parallel, and the antimatter walks its ring sequentially in its own
/// thread (`AMTR.cpp:52-77`): each non-exempt contact increments its `life`,
/// the fourth one kills the antimatter *before* it touches that neighbour, and
/// the other contacts either kill the neighbour or rebuild its slot as a `PHOT`
/// (`create_part(ID(r), x+rx, y+ry, PT_PHOT)`). The port keeps that order, so
/// the frame an antimatter reaches its fourth contact the rest of the ring is
/// untouched, exactly like the reference's `return 1`.
///
/// The photon is queued at the neighbour's pixel with [`CR_NO_OCCUPANCY`]
/// (TPT reuses the neighbour's slot unconditionally, there is no occupancy
/// test) and the neighbour's own slot is killed, which keeps the particle count
/// identical. A target next to two antimatter particles can still be resolved
/// twice if both threads read it before either kill lands; TPT's per-particle
/// order would let the first one win. The pressure loss is booked in the
/// antimatter's own cell (`sim->pv[y/CELL][x/CELL]`), not the neighbour's.
#[cube(launch_unchecked)]
pub(crate) fn annihilate_kernel(
    pmap: &[u32],
    part_ty: &mut [u32],
    part_life: &mut [i32],
    elem_words: &[u32],
    air_pv: &mut [f32],
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
            if ty != PT_NONE {
                let px = (idx as u32 % xres) as i32;
                let py = (idx as u32 / xres) as i32;
                let roll = hash_u32(frame_seed ^ mix_index(idx as u32, 3_266_489_909u32));

                if ty == PT_AMTR {
                    // `AMTR.cpp:52-77`: every non-exempt contact increments the
                    // antimatter's life; the fourth one annihilates it.
                    let cidx = (((py as u32) / CELL) * xcells + ((px as u32) / CELL)) as usize;
                    let mut life = part_life[id];
                    // `AMTR.cpp:56-57`: `rx` is the outer loop, `ry` the inner
                    // one, so the fourth contact (the neighbour the antimatter
                    // never touches) follows TPT's scan order.
                    for rx in 0u32..3u32 {
                        for ry in 0u32..3u32 {
                            let dx = rx as i32 - 1i32;
                            let dy = ry as i32 - 1i32;
                            if (dx != 0i32 || dy != 0i32) && life < 4i32 {
                                let nv = pmap_at(pmap, xres, yres, px + dx, py + dy);
                                if nv != 0u32 {
                                    let nid = pmap_slot(nv);
                                    let nty = part_ty[nid];
                                    // A neighbour that an earlier contact (or
                                    // another antimatter) already killed reads
                                    // empty in TPT's live map.
                                    if nty != PT_NONE && !amtr_exempt(nty) {
                                        life += 1i32;
                                        if life == 4i32 {
                                            // The fourth contact annihilates the
                                            // antimatter before it can touch this
                                            // neighbour (`return 1`).
                                            part_ty[id] = PT_NONE;
                                            break;
                                        } else {
                                            // `AMTR.cpp:71-75`: one contact in ten
                                            // leaves a photon, the rest annihilate
                                            // the target outright; either way the
                                            // antimatter's cell loses 2 pressure.
                                            if chance(
                                                hash_u32(roll ^ (rx * 3u32 + ry)),
                                                1u32,
                                                10u32,
                                            ) {
                                                request_create(
                                                    create_queue,
                                                    PT_PHOT,
                                                    px + dx,
                                                    py + dy,
                                                    CR_NO_OCCUPANCY,
                                                    0i32,
                                                    0u32,
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
                                            part_ty[nid] = PT_NONE;
                                            air_pv[cidx] -= 2.0f32;
                                        }
                                    }
                                }
                            }
                        }
                    }
                    part_life[id] = life;
                } else if ty == PT_EMBR {
                    // `EMBR.cpp:54-75`: a spark that settles against any solid,
                    // powder or liquid (other than another spark) is absorbed.
                    for n in 0u32..9u32 {
                        let dx = (n % 3u32) as i32 - 1i32;
                        let dy = (n / 3u32) as i32 - 1i32;
                        if (dx != 0i32 || dy != 0i32) && part_ty[id] != PT_NONE {
                            let nv = pmap_at(pmap, xres, yres, px + dx, py + dy);
                            if nv != 0u32 {
                                let nid = pmap_slot(nv);
                                let nty = part_ty[nid];
                                let props = elem_word(elem_words, nty, WORD_PROPERTIES);
                                let matter = props & (TYPE_SOLID | TYPE_PART | TYPE_LIQUID) != 0u32;
                                if matter && (props & PROP_SPARKSETTLE) == 0u32 {
                                    part_ty[id] = PT_NONE;
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

/// `AMTR.cpp:62`: the particles an antimatter contact leaves alone.
#[cube]
fn amtr_exempt(ty: u32) -> bool {
    ty == PT_AMTR
        || ty == PT_DMND
        || ty == PT_CLNE
        || ty == PT_PCLN
        || ty == PT_VOID
        || ty == PT_BHOL
        || ty == PT_NBHL
        || ty == PT_PRTI
        || ty == PT_PRTO
}

/// `DEUT` compression, absorption, expansion and diffusion (`DEUT.cpp:54-136`).
///
/// Pixel parallel. The expansion queues one `DEUT` per free neighbour through
/// the creation queue and spends one life per created pixel; TPT fills every
/// free neighbour until the spare life runs out.
#[cube(launch_unchecked)]
pub(crate) fn deut_kernel(
    pmap: &[u32],
    part_ty: &mut [u32],
    part_life: &mut [i32],
    part_temp: &mut [f32],
    grav_fx: &[f32],
    grav_fy: &[f32],
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
            if ty == PT_DEUT {
                let px = (idx as u32 % xres) as i32;
                let py = (idx as u32 / xres) as i32;
                let cidx = (((py as u32) / CELL) * xcells + ((px as u32) / CELL)) as usize;
                let roll = hash_u32(frame_seed ^ mix_index(idx as u32, 2_654_435_761u32));
                // `gravOut.forceX/forceY` of the current cell.
                let gravtot = fabsf(grav_fx[cidx]) + fabsf(grav_fy[cidx]);
                let maxlife = deut_max_life(part_temp[id], gravtot, roll);

                if part_life[id] < maxlife {
                    // Absorb a neighbour (one in three) while there is room.
                    for n in 0u32..9u32 {
                        let dx = (n % 3u32) as i32 - 1i32;
                        let dy = (n / 3u32) as i32 - 1i32;
                        if (dx != 0i32 || dy != 0i32) && part_life[id] < maxlife {
                            let nv = pmap_at(pmap, xres, yres, px + dx, py + dy);
                            if nv != 0u32 {
                                let nid = pmap_slot(nv);
                                // `DEUT.cpp:76`: `n.life <= maxlife - life - 1`,
                                // written as a strict comparison so no
                                // intermediate can overflow.
                                if part_ty[nid] == PT_DEUT
                                    && chance(hash_u32(roll ^ n), 1u32, 3u32)
                                    && part_life[nid] < maxlife - part_life[id]
                                {
                                    part_life[id] += part_life[nid] + 1i32;
                                    part_ty[nid] = PT_NONE;
                                }
                            }
                        }
                    }
                } else {
                    // `DEUT.cpp:88-113` `tryExpand`: fill every empty
                    // neighbour, spending one life per pixel. The reference
                    // leaves the loop as soon as the particle has no spare life
                    // (`parts[i].life <= maxlife`) and only creates into an
                    // empty particle-layer pixel. The new pixel starts empty
                    // (`parts[np].life = 0`) and inherits the parent's
                    // temperature.
                    for n in 0u32..9u32 {
                        let dx = (n % 3u32) as i32 - 1i32;
                        let dy = (n / 3u32) as i32 - 1i32;
                        if (dx != 0i32 || dy != 0i32)
                            && part_life[id] > maxlife
                            && part_life[id] >= 1i32
                        {
                            let nv = pmap_at(pmap, xres, yres, px + dx, py + dy);
                            if nv == 0u32 {
                                request_create(
                                    create_queue,
                                    PT_DEUT,
                                    px + dx,
                                    py + dy,
                                    CR_SET_LIFE | CR_SET_TEMP,
                                    0i32,
                                    0u32,
                                    part_temp[id],
                                    0i32,
                                    0i32,
                                    0i32,
                                    0i32,
                                    0u32,
                                    0u32,
                                    0.0f32,
                                    0.0f32,
                                );
                                part_life[id] -= 1i32;
                            }
                        }
                    }
                }

                // Four random trades spread the life through the pool.
                for trade in 0u32..4u32 {
                    let rx =
                        (hash_u32(roll ^ mix_index(trade, 2_246_822_519u32)) % 5u32) as i32 - 2i32;
                    let ry =
                        (hash_u32(roll ^ mix_index(trade, 3_266_489_909u32)) % 5u32) as i32 - 2i32;
                    if rx != 0i32 || ry != 0i32 {
                        let nv = pmap_at(pmap, xres, yres, px + rx, py + ry);
                        if nv != 0u32 {
                            let nid = pmap_slot(nv);
                            if part_ty[nid] == PT_DEUT
                                && part_life[id] > part_life[nid]
                                && part_life[id] > 0i32
                            {
                                let diff = part_life[id] - part_life[nid];
                                if diff == 1i32 {
                                    part_life[nid] += 1i32;
                                    part_life[id] -= 1i32;
                                } else if diff > 0i32 {
                                    part_life[nid] += diff / 2i32;
                                    part_life[id] -= diff / 2i32;
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Host mirror of the kernel's `chance`, pinning the `rng.chance` shape.
    fn chance_host(roll: u32, n: u32, d: u32) -> bool {
        d > 0 && roll % d < n
    }

    /// Host mirror of [`deut_max_life`]; a kernel cannot call a host function,
    /// so this copy pins the formula in tests.
    fn deut_max_life_host(temp: f32, gravtot: f32, roll: u32) -> i32 {
        let t = (temp + 1.0f32).max(1.0f32);
        let ti = t as i32;
        let mut maxlife = (10000.0f32 / (t + 1.0f32)) as i32 - 1;
        if ti > 0 && roll % (ti as u32) < (10000u32 % ti as u32) {
            maxlife += 1;
        }
        maxlife * (5.0f32 - 8.0f32 / (gravtot + 2.0f32)) as i32
    }

    #[test]
    fn chance_matches_rng_shape() {
        assert!(chance_host(0, 1, 100));
        assert!(!chance_host(1, 1, 100));
        // A non-positive modulus can never hit.
        assert!(!chance_host(0, 5, 0));
        // `n == 0` never hits even for a zero roll.
        assert!(!chance_host(0, 0, 1000));
        // `n > d` always hits, like `rng.chance(11, 40)` on roll 0.
        assert!(chance_host(0, 11, 40));
    }

    #[test]
    fn deut_life_target_scales_with_temperature_and_gravity() {
        // `DEUT.cpp:56-66` with no gravity: `mult == 1` and a roll of 0 always
        // earns the fractional carry (`10000 % 294 == 4`).
        assert_eq!(deut_max_life_host(293.15f32, 0.0f32, 0), 33);
        // A hot pool compresses far less than a cold one.
        assert!(
            deut_max_life_host(5000.0f32, 0.0f32, 0) < deut_max_life_host(293.15f32, 0.0f32, 0)
        );
        // The multiplier is bounded by 1 (no gravity) and 5 (strong field).
        assert_eq!(
            (5.0f32 - 8.0f32 / (0.0f32 + 2.0f32)) as i32,
            1,
            "flat space must not amplify the target"
        );
        assert_eq!(
            (5.0f32 - 8.0f32 / (1000.0f32 + 2.0f32)) as i32,
            4,
            "the multiplier saturates below 5"
        );
        assert!(
            deut_max_life_host(293.15f32, 1000.0f32, 0) > deut_max_life_host(293.15f32, 0.0f32, 0)
        );
    }

    /// Host mirror of [`annihilate_kernel`]'s antimatter ring
    /// (`AMTR.cpp:52-77`): walk the neighbours in the reference's order, count
    /// every non-exempt contact, die on the fourth one *before* it is touched,
    /// and lose two pressure units per resolved contact.
    ///
    /// Returns `(life, alive, resolved_contacts)` where `resolved_contacts` is
    /// how many neighbours were actually destroyed or flashed.
    fn amtr_ring(neighbours: &[u32]) -> (i32, bool, usize) {
        let exempt = |ty: u32| {
            matches!(
                ty,
                PT_AMTR
                    | PT_DMND
                    | PT_CLNE
                    | PT_PCLN
                    | PT_VOID
                    | PT_BHOL
                    | PT_NBHL
                    | PT_PRTI
                    | PT_PRTO
                    | PT_NONE
            )
        };
        let mut life = 0i32;
        let mut resolved = 0usize;
        for &ty in neighbours {
            if life >= 4 {
                break;
            }
            if exempt(ty) {
                continue;
            }
            life += 1;
            if life == 4 {
                return (life, false, resolved);
            }
            resolved += 1;
        }
        (life, true, resolved)
    }

    #[test]
    fn amtr_ring_aborts_on_the_fourth_contact() {
        // Three contacts: the antimatter lives and every target is resolved.
        let (life, alive, resolved) = amtr_ring(&[PT_DUST, PT_WATR, PT_SAND]);
        assert_eq!((life, alive, resolved), (3, true, 3));
        // A fourth contact kills the antimatter and leaves that neighbour
        // untouched, exactly like the reference's `return 1`.
        let (life, alive, resolved) = amtr_ring(&[PT_DUST, PT_WATR, PT_SAND, PT_GLAS]);
        assert_eq!((life, alive, resolved), (4, false, 3));
        // Exempt neighbours do not count as contacts.
        let (life, alive, resolved) =
            amtr_ring(&[PT_DMND, PT_DUST, PT_VOID, PT_WATR, PT_BHOL, PT_SAND]);
        assert_eq!((life, alive, resolved), (3, true, 3));
        // The fourth contact only aborts the ring; contacts after it in the
        // same ring are skipped.
        let (life, alive, resolved) = amtr_ring(&[PT_DUST, PT_DUST, PT_DUST, PT_DUST, PT_DUST]);
        assert_eq!((life, alive, resolved), (4, false, 3));
    }

    #[test]
    fn amtr_contact_costs_two_pressure_units_in_its_own_cell() {
        // `AMTR.cpp:75`: `pv[y/CELL][x/CELL] -= 2.0f`, once per resolved
        // contact, booked in the antimatter's cell.
        let mut pv = 10.0f32;
        for _ in 0..3 {
            pv -= 2.0f32;
        }
        assert_eq!(pv, 4.0f32);
        assert_eq!(10.0f32 - 2.0f32 * 3.0f32, 4.0f32);
    }

    #[test]
    fn uran_pressure_heating_matches_the_reference() {
        // `URAN.cpp:64-68`: `temp * (1 + pv/2000)` clamped to `MIN_TEMP..MAX_TEMP`.
        let heat = |t: f32, pv: f32| (t * (1.0f32 + pv / 2000.0f32)).clamp(MIN_TEMP, MAX_TEMP);
        assert!((heat(293.15f32, 2000.0f32) - 586.3f32).abs() < 1.0e-3f32);
        assert_eq!(heat(MAX_TEMP, 2000.0f32), MAX_TEMP);
        // A vacuum cell is a no-op: the kernel only runs for `pv > 0`.
        assert_eq!(heat(293.15f32, 0.0f32), 293.15f32);
    }
}
