//! Fire burnout, spark discharge and the per-element life-dependent
//! transformations.
//!
//! The `PROP_LIFE_DEC` countdown itself runs once per frame in
//! [`crate::buffers::spatial::life_decay_kernel`]
//! (`Simulation::RecalcFreeParticles`, `Simulation.cpp:3399-3410`) before this
//! pass, so the kernel below only reads the post-decrement `life` and must
//! never decrement it a second time.
//!
//! Slot parallel: it visits every slot of the pool exactly once, whether or not
//! the particle is currently reachable through `pmap` or `photons`, so life
//! timers can never stall on a stale spatial map.

// Clippy's rewrite for these lints uses methods the device IR does not have
// (`is_multiple_of`, `clamp`, `Range::contains`, `saturating_sub`), and
// collapsing a nested `if` changes what the CubeCL macro expands, so the plain
// operators are deliberate.
#![allow(clippy::assign_op_pattern)]

use cubecl::prelude::*;
use tpt_core::constants::*;

use super::shared::{ELEM_SLOTS, WORD_PROPERTIES, elem_word, hash_u32};

/// `R_TEMP + 273.15`: the temperature a discharged `WATR`/`SLTW`/logic pixel
/// returns to (`SPRK.cpp:64`).
const ROOM_TEMP: f32 = 293.15;

/// `FLAG_STASIS` (`0x20`), mirrored for the kernel body.
const FLAG_STASIS_U32: u32 = 0x20;
const _: () = assert!(FLAG_STASIS == 0x20);
/// `WL_STASIS` (`18`) and the `emap < 8` power gate (`Simulation.cpp:2335`).
const WL_STASIS_U32: u32 = 18;
const _: () = assert!(WL_STASIS == 18);

/// The life-dependent reactions a particle can resolve from its own state.
///
/// The countdown and the `PROP_LIFE_KILL` / `PROP_LIFE_KILL_DEC` deaths ran
/// once at the start of the frame in [`crate::buffers::spatial::life_decay_kernel`]
/// (`Simulation.cpp:3399-3410`); this pass only reads the resulting value, so
/// `life <= 1` here means "one tick left" exactly as in
/// `Simulation::UpdateParticles` (`Simulation.cpp:2380-2400`).
///
/// This kernel is *slot parallel*: it visits every slot of the pool exactly
/// once, whether or not the particle is currently reachable through `pmap` or
/// `photons`, so life timers can never stall on a stale spatial map.
#[cube(launch_unchecked)]
pub(super) fn life_and_local_reaction_kernel(
    part_ty: &mut [u32],
    part_life: &mut [i32],
    part_ctype: &mut [u32],
    part_tmp: &mut [i32],
    part_temp: &mut [f32],
    part_flags: &mut [u32],
    part_x: &[f32],
    part_y: &[f32],
    wall_bmap: &[u32],
    wall_emap: &[u32],
    elem_words: &[u32],
    frame_seed: u32,
    n_slots: u32,
    xcells: u32,
) {
    let i = ABSOLUTE_POS;
    if i < n_slots as usize {
        let ty = part_ty[i];
        if ty != PT_NONE {
            // `Simulation.cpp:2335-2338`: a particle inside an unpowered
            // `WL_STASIS` cell is skipped entirely. The flag is refreshed here
            // because this is the first pass that visits every slot, so every
            // later reaction pass can test its own `part_flags` word instead of
            // binding the wall maps.
            let cell = ((i32::cast_from(part_y[i] + 0.5f32) / 4i32) as u32) * xcells
                + ((i32::cast_from(part_x[i] + 0.5f32) / 4i32) as u32);
            let frozen = wall_bmap[cell as usize] == WL_STASIS_U32
                && (wall_emap[cell as usize] & 0xFFu32) < 8u32;
            if frozen {
                part_flags[i] = part_flags[i] | FLAG_STASIS_U32;
                terminate!();
            } else {
                part_flags[i] = part_flags[i] & !FLAG_STASIS_U32;
            }
            let props = elem_word(elem_words, ty, WORD_PROPERTIES);
            let mut life = part_life[i];
            let mut new_ty = ty;
            let mut dead = false;

            // A fire created without an explicit lifetime (the registry default
            // is 0) starts a fresh burn instead of dying on its first frame. A
            // fire that actually burned out is converted to smoke or steam below
            // before it can ever be seen here with `life == 0`.
            if ty == PT_FIRE && life == 0i32 {
                life = 180i32 + ((hash_u32((i as u32) ^ frame_seed) % 80u32) as i32);
            } else if (ty == PT_PLSM || ty == PT_HFLM) && life == 0i32 {
                life = 50i32 + ((hash_u32((i as u32) ^ frame_seed ^ 0xA5A5_u32) % 150u32) as i32);
            } else if ty == PT_TRON && life == 0i32 {
                life = 25i32 + ((hash_u32((i as u32) ^ frame_seed ^ 0x3C3C_u32) % 25u32) as i32);
            }

            // The countdown ran once in `life_decay_kernel` at the start of the
            // frame (`Simulation.cpp:3399-3410`). Decrementing here as well
            // halved every timer and broke the `life == 3` / `life == 1`
            // windows the element updates rely on.

            if ty == PT_FIRE {
                // Fire burns out. `tmp` records which gases fed it: `1` = hydrogen,
                // `2` = oxygen, so `tmp & 3 == 3` means the stoichiometric mix that
                // leaves water vapour behind instead of smoke.
                if life <= 1i32 {
                    let mark = part_tmp[i];
                    if (mark & 3i32) == 3i32 {
                        new_ty = PT_WTRV;
                        life = 0i32;
                        part_ctype[i] = PT_FIRE;
                    } else if part_temp[i] < 625.0f32 {
                        // `FIRE.cpp:91-95`: the smoke gets a random
                        // 250..269 life, not a fixed one.
                        new_ty = PT_SMKE;
                        life = 250i32
                            + ((hash_u32((i as u32) ^ frame_seed ^ 0x5AC_E123u32) % 20u32) as i32);
                        part_tmp[i] = 0i32;
                    } else {
                        dead = true;
                    }
                }
            } else if ty == PT_PLSM {
                // Plasma burns out. `ctype == PT_NBLE` relaxes back to noble gas,
                // stoichiometric H2+O2 produces steam, otherwise plasma dies.
                if life <= 1i32 {
                    let ctype = part_ctype[i];
                    let mark = part_tmp[i];
                    if ctype == PT_NBLE {
                        new_ty = PT_NBLE;
                        life = 0i32;
                        part_ctype[i] = PT_NONE;
                    } else if (mark & 3i32) == 3i32 {
                        new_ty = PT_WTRV;
                        life = 0i32;
                        part_ctype[i] = PT_FIRE;
                    } else {
                        dead = true;
                    }
                }
            } else if ty == PT_SPRK {
                // `SPRK.cpp:62-88`: a spark that ran out of life settles back
                // into the conductor it travelled through. TPT never kills it
                // for a bad `ctype` — it falls back to `METL` — and the
                // water/logic family comes back at room temperature. The
                // restored pixel keeps a refractory cooldown (`life = 4`, so a
                // conductor only conducts again once its own life reached 0),
                // and `RSST` vanishes at the end of its spark cycle.
                if life <= 0i32 {
                    let mut ctype = part_ctype[i];
                    if ctype == PT_WATR
                        || ctype == PT_SLTW
                        || ctype == PT_PSCN
                        || ctype == PT_NSCN
                        || ctype == PT_ETRD
                        || ctype == PT_INWR
                    {
                        part_temp[i] = ROOM_TEMP;
                    }
                    if ctype == PT_NONE || ctype >= ELEM_SLOTS {
                        ctype = PT_METL;
                    }
                    part_ctype[i] = PT_NONE;
                    life = 4i32;
                    if ctype == PT_WATR {
                        life = 64i32;
                    } else if ctype == PT_SLTW {
                        life = 54i32;
                    } else if ctype == PT_SWCH {
                        life = 14i32;
                    } else if ctype == PT_RSST {
                        dead = true;
                    }
                    if !dead {
                        new_ty = ctype;
                    }
                }
            }

            // The generic life kill only applies to the element that declared
            // it; a particle that just transformed inherits its new element's
            // rules on the next frame instead. `PROP_LIFE_KILL_DEC` is not
            // repeated here: it kills only on the transition to zero, which is
            // exactly the `life_decay_kernel` decrement.
            if new_ty == ty && (props & PROP_LIFE_KILL) != 0u32 && life <= 0i32 {
                dead = true;
            }

            if dead {
                part_ty[i] = PT_NONE;
            } else {
                if new_ty != ty {
                    part_ty[i] = new_ty;
                }
                if life != part_life[i] {
                    part_life[i] = life;
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fire_plsm_burnout_host(
        ty: u32,
        life: i32,
        ctype: u32,
        tmp: i32,
        temp: f32,
        roll: u32,
    ) -> Option<(u32, i32, u32)> {
        if ty == PT_FIRE && life <= 1 {
            if (tmp & 3) == 3 {
                Some((PT_WTRV, 0, PT_FIRE))
            } else if temp < 625.0 {
                // `FIRE.cpp:93-95`: `rng.between(250, 269)`.
                Some((PT_SMKE, 250 + (roll % 20) as i32, 0))
            } else {
                None
            }
        } else if ty == PT_PLSM && life <= 1 {
            if ctype == PT_NBLE {
                Some((PT_NBLE, 0, PT_NONE))
            } else if (tmp & 3) == 3 {
                Some((PT_WTRV, 0, PT_FIRE))
            } else {
                None
            }
        } else {
            Some((ty, life, ctype))
        }
    }

    #[test]
    fn fire_burnout_transitions() {
        // H2+O2 fed fire leaves steam.
        assert_eq!(
            fire_plsm_burnout_host(PT_FIRE, 1, 0, 3, 700.0, 0),
            Some((PT_WTRV, 0, PT_FIRE))
        );
        // Cold fire leaves smoke with a random 250..269 life.
        for roll in 0u32..64u32 {
            let (ty, life, ctype) = fire_plsm_burnout_host(PT_FIRE, 0, 0, 0, 500.0, roll).unwrap();
            assert_eq!((ty, ctype), (PT_SMKE, 0));
            assert!(
                (250..=269).contains(&life),
                "smoke life {life} outside FIRE.cpp's range"
            );
        }
        // Hot unmixed fire burns out completely.
        assert_eq!(fire_plsm_burnout_host(PT_FIRE, 1, 0, 0, 800.0, 0), None);
    }

    #[test]
    fn plsm_burnout_transitions() {
        // PLSM(NBLE) returns to NBLE.
        assert_eq!(
            fire_plsm_burnout_host(PT_PLSM, 1, PT_NBLE, 0, 9999.0, 0),
            Some((PT_NBLE, 0, PT_NONE))
        );
        // H2+O2 fed plasma turns into steam.
        assert_eq!(
            fire_plsm_burnout_host(PT_PLSM, 0, 0, 3, 9999.0, 0),
            Some((PT_WTRV, 0, PT_FIRE))
        );
        // Plain plasma dies.
        assert_eq!(fire_plsm_burnout_host(PT_PLSM, 1, 0, 0, 9999.0, 0), None);
    }

    #[test]
    fn default_life_ranges() {
        for seed in [0u32, 42, 100, 999] {
            let fire_l = 180 + (hash_u32(seed) % 80) as i32;
            assert!((180..=259).contains(&fire_l));
            let plsm_l = 50 + (hash_u32(seed ^ 0xA5A5_u32) % 150) as i32;
            assert!((50..=199).contains(&plsm_l));
            let tron_l = 25 + (hash_u32(seed ^ 0x3C3C_u32) % 25) as i32;
            assert!((25..=49).contains(&tron_l));
        }
    }
}
