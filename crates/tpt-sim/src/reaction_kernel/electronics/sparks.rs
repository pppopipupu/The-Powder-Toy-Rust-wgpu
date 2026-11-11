//! Snapshot staging and spark conduction passes.

use cubecl::prelude::*;
use tpt_core::constants::*;

use super::super::shared::{
    WORD_PROPERTIES, elem_word, hash_u32, iabs, insulated, pmap_at, pmap_slot,
};
use super::{SPARK_LIFE, SWCH_OFF_LIFE};

/// Snapshot word: bits `0..8` element id.
pub(super) const STAGE_TY: u32 = 0x0000_00FF;
/// Snapshot word: bits `8..16` `life`, clamped to `0..=255`.
///
/// Every electrical rule only compares `life` against `0` and the small spark /
/// switch constants (`< 4`, `== 10`, `>= 10`, `<= 1`), so a saturating byte is
/// lossless for this pass. `DLAY` delays above 255 frames are read live by
/// [`arc_and_delay_kernel`] and are therefore unaffected.
pub(super) const STAGE_LIFE: u32 = 0x0000_FF00;
/// Snapshot word: bits `16..24` `ctype`.
pub(super) const STAGE_CTYPE: u32 = 0x00FF_0000;
/// Snapshot word: `temp < 173.15` (`QRTZ.cpp` conduction gate).
pub(super) const STAGE_COLD_QUARTZ: u32 = 0x0100_0000;
/// Snapshot word: `temp < 373.0` (`PTCT` gate).
pub(super) const STAGE_COLD: u32 = 0x0200_0000;
/// Snapshot word: `temp > 373.0` (`NTCT` gate).
pub(super) const STAGE_HOT: u32 = 0x0400_0000;

/// `SPRK.cpp` / `NTCT.cpp` / `PTCT.cpp` thermistor threshold.
const THERMISTOR_TEMP: f32 = 373.0;

/// `SPRK.cpp:330`: a sparked `QRTZ` only conducts when the crystal is colder
/// than 173.15 K or the air cell it sits in is above this pressure.
const QRTZ_PRESSURE: f32 = 8.0;

/// Number of `INST` cascade rounds per frame.
///
/// `FloodINST` fills the whole span in one pass (`Simulation.cpp:586-700`); one
/// dispatch moves the frontier a single orthogonal hop, so this is the same
/// bounded-round trade-off as `pipe::FLOOD_ROUNDS`. The cascade re-runs every
/// frame, so a longer run keeps lighting from the sparks on its frontier.
pub(super) const INST_FLOOD_ROUNDS: u32 = 8;

const PT_EMP_U32: u32 = 143;
const _: () = assert!(PT_EMP == 143);
/// Fold the three temperature thresholds used by the conduction rules into the
/// snapshot flag bits.
#[cube]
fn heat_bits(t: f32) -> u32 {
    let mut bits = 0u32;
    if t < 173.15f32 {
        bits |= STAGE_COLD_QUARTZ;
    }
    if t < THERMISTOR_TEMP {
        bits |= STAGE_COLD;
    }
    if t > THERMISTOR_TEMP {
        bits |= STAGE_HOT;
    }
    bits
}

/// Pack one slot of the previous-state snapshot.
///
/// `.clamp()` is not available inside a `#[cube]` function, hence the explicit
/// saturation.
#[cube]
#[allow(clippy::manual_clamp)]
fn stage_pack(ty: u32, life: i32, ctype: u32, bits: u32) -> u32 {
    let mut l = life;
    if l < 0i32 {
        l = 0i32;
    }
    if l > 255i32 {
        l = 255i32;
    }
    (ty & STAGE_TY) | ((l as u32) << 8u32) | ((ctype & 0xFFu32) << 16u32) | bits
}

/// Element id of a snapshot word.
#[cube]
fn stage_ty(word: u32) -> u32 {
    word & STAGE_TY
}

/// `life` of a snapshot word.
#[cube]
fn stage_life(word: u32) -> i32 {
    ((word & STAGE_LIFE) >> 8u32) as i32
}

/// `ctype` of a snapshot word.
#[cube]
fn stage_ctype(word: u32) -> u32 {
    (word & STAGE_CTYPE) >> 16u32
}

/// `true` when the snapshot says the particle is hotter than 373 K.
#[cube]
fn stage_hot(word: u32) -> bool {
    (word & STAGE_HOT) != 0u32
}

/// `true` when the snapshot says the particle is colder than 173.15 K, the
/// `QRTZ` conduction threshold of `SPRK.cpp:330`.
#[cube]
fn stage_cold_quartz(word: u32) -> bool {
    (word & STAGE_COLD_QUARTZ) != 0u32
}

/// `true` when the snapshot says the particle is colder than 373 K.
#[cube]
fn stage_cold(word: u32) -> bool {
    (word & STAGE_COLD) != 0u32
}

/// `SPRK.cpp` `tryConduct`: may a spark whose `ctype` is `sender` conduct into
/// a `receiver` at Manhattan distance `manhattan`?
///
/// `sender_bits` / `receiver_bits` are the snapshot flag words of the spark and
/// the receiver, which is where the `NTCT` / `FNTC` / `PTCT` / `QRTZ`
/// temperature gates read from. `receiver_pressurised` is the other half of the
/// `QRTZ` gate: the air cell under the receiving pixel is above
/// [`QRTZ_PRESSURE`].
///
/// `FNTC` is the fast `NTCT` and shares its gates (`SPRK.cpp:306,313-317,
/// 336-340`); `FPTC` has no sender or receiver case in the reference, so it
/// only appears in the spark's own `ctype` rules below.
///
/// `INST` and `QRTZ` are special receivers even though neither carries
/// `PROP_CONDUCTS` (`SPRK.cpp:290`): `INST` accepts only a `PSCN` spark and
/// then floods its whole wire (`SPRK.cpp:345-348`), and `QRTZ` accepts the
/// logic/metal senders next to a cold or pressurised crystal
/// (`SPRK.cpp:329-332`).
#[cube]
fn spark_conducts(
    sender: u32,
    receiver: u32,
    manhattan: i32,
    sender_bits: u32,
    receiver_bits: u32,
    receiver_pressurised: bool,
) -> bool {
    let mut ok = true;
    // Only a switch conducts across the diagonal of its 5x5 window.
    if manhattan >= 4i32 && sender != PT_SWCH && receiver != PT_SWCH {
        ok = false;
    }
    // A switch does not feed the pure logic levels back into itself.
    // `SPRK.cpp:306` lists `FNTC` next to `NTCT`/`PTCT` here.
    if ok
        && sender == PT_SWCH
        && (receiver == PT_PSCN
            || receiver == PT_NSCN
            || receiver == PT_WATR
            || receiver == PT_SLTW
            || receiver == PT_NTCT
            || receiver == PT_FNTC
            || receiver == PT_PTCT
            || receiver == PT_INWR)
    {
        ok = false;
    }
    // Everything conducts into its own element, so a metal bus carries a spark
    // along itself; this test sits before the sender table in `SPRK.cpp:294`,
    // which exempts `INST` and `QRTZ`.
    if ok && receiver == sender && receiver != PT_INST && receiver != PT_QRTZ {
        ok = true;
    } else if ok && sender == PT_INST {
        // `SPRK.cpp:296-300`: an INST spark only drives an NSCN.
        ok = receiver == PT_NSCN;
    } else if ok && sender == PT_ETRD {
        // Sender cases: elements with a restricted output set.
        ok = receiver == PT_METL
            || receiver == PT_BMTL
            || receiver == PT_BRMT
            || receiver == PT_LRBD
            || receiver == PT_RBDM
            || receiver == PT_PSCN
            || receiver == PT_NSCN;
    } else if ok && sender == PT_NTCT {
        ok = receiver == PT_PSCN || (receiver == PT_NSCN && stage_hot(sender_bits));
    } else if ok && sender == PT_FNTC {
        // `SPRK.cpp:313-317`: `FNTC` is the fast `NTCT`, same output set.
        // `PT_FPTC` is absent from the reference's sender switch (only the
        // `ct` switch and the 473 K heating lists name it), so an `FPTC`
        // spark keeps the unrestricted default.
        ok = receiver == PT_PSCN || (receiver == PT_NSCN && stage_hot(sender_bits));
    } else if ok && sender == PT_PTCT {
        ok = receiver == PT_PSCN || (receiver == PT_NSCN && stage_cold(sender_bits));
    } else if ok && sender == PT_INWR {
        ok = receiver == PT_NSCN || receiver == PT_PSCN;
    } else if ok {
        // Receiver cases: elements with a restricted input set.
        if receiver == PT_QRTZ {
            // `SPRK.cpp:329-332`: the crystal must be below 173.15 K or sit in
            // air above 8 pressure units, and only the logic/metal senders
            // (plus another crystal) charge it.
            ok = (sender == PT_NSCN || sender == PT_METL || sender == PT_PSCN || sender == PT_QRTZ)
                && (stage_cold_quartz(receiver_bits) || receiver_pressurised);
        } else if receiver == PT_INST {
            // `SPRK.cpp:345-348`: only a PSCN charges an INST wire.
            ok = sender == PT_PSCN;
        } else if receiver == PT_NTCT {
            ok = sender == PT_NSCN || (sender == PT_PSCN && stage_hot(receiver_bits));
        } else if receiver == PT_FNTC {
            // `SPRK.cpp:336-340`: fast `FNTC` input set. As in the sender
            // switch, `PT_FPTC` has no case in the reference and therefore
            // accepts any sender.
            ok = sender == PT_NSCN || (sender == PT_PSCN && stage_hot(receiver_bits));
        } else if receiver == PT_PTCT {
            ok = sender == PT_NSCN || (sender == PT_PSCN && stage_cold(receiver_bits));
        } else if receiver == PT_INWR {
            ok = sender == PT_NSCN || sender == PT_PSCN;
        } else if receiver == PT_PSCN {
            ok = sender != PT_NSCN;
        }
    }
    ok
}

/// Copy the electrical part of every slot into the `intent_target` scratch.
///
/// Slot parallel and read-only, so the conduction pass that follows sees one
/// consistent previous frame no matter how the device schedules it.
#[cube(launch_unchecked)]
pub(crate) fn spark_stage_kernel(
    part_ty: &[u32],
    part_life: &[i32],
    part_ctype: &[u32],
    part_temp: &[f32],
    stage: &mut [u32],
    n_slots: u32,
) {
    let i = ABSOLUTE_POS;
    if i < n_slots as usize {
        let ty = part_ty[i];
        let mut word = 0u32;
        if ty != PT_NONE {
            word = stage_pack(ty, part_life[i], part_ctype[i], heat_bits(part_temp[i]));
        }
        stage[i] = word;
    }
}

/// `SPRK.cpp`: spark discharge bookkeeping plus the receiver-side conduction
/// loop.
///
/// Pixel parallel and slot-owned: a thread only ever writes the slot and the
/// `pmap` word of its own pixel. Every neighbour read comes from `stage`.
///
/// `air_pv` / `xcells` are only read for the `QRTZ` pressure gate
/// (`SPRK.cpp:330`), which samples the air cell under the receiving pixel.
#[cube(launch_unchecked)]
pub(crate) fn spark_conduction_kernel(
    pmap: &[u32],
    stage: &[u32],
    part_ty: &mut [u32],
    part_life: &mut [i32],
    part_ctype: &mut [u32],
    part_temp: &mut [f32],
    part_tmp: &mut [i32],
    elem_words: &[u32],
    air_pv: &[f32],
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
            let prev = stage[id];
            let self_ty = stage_ty(prev);
            let self_life = stage_life(prev);
            let self_ctype = stage_ctype(prev);
            let px = (idx as u32 % xres) as i32;
            let py = (idx as u32 / xres) as i32;

            if self_ty == PT_SPRK {
                // `SPRK.cpp` lines 232-239: an NSCN spark extinguishes a spark on SWCH,
                // reverting it to switch in the off state.
                if self_ctype == PT_SWCH {
                    let mut extinguished = false;
                    for n in 0u32..25u32 {
                        if !extinguished {
                            let dx = (n % 5u32) as i32 - 2i32;
                            let dy = (n / 5u32) as i32 - 2i32;
                            if dx != 0i32 || dy != 0i32 {
                                let nx = px + dx;
                                let ny = py + dy;
                                let nv = pmap_at(pmap, xres, yres, nx, ny);
                                if nv != 0u32 && !insulated(pmap, xres, yres, nx, ny, px, py) {
                                    let ns = stage[pmap_slot(nv)];
                                    if stage_ty(ns) == PT_SPRK
                                        && stage_ctype(ns) == PT_NSCN
                                        && stage_life(ns) > 0i32
                                        && stage_life(ns) < SPARK_LIFE
                                    {
                                        part_ty[id] = PT_SWCH;
                                        part_ctype[id] = PT_NONE;
                                        part_life[id] = SWCH_OFF_LIFE;
                                        extinguished = true;
                                    }
                                }
                            }
                        }
                    }
                }

                // `SPRK.cpp` lines 93-96 & `NTCT.cpp`: thermistor spark
                // self-cooling above 295 K. `SPRK.cpp:94-99` routes `FPTC`
                // and `FNTC` through `Element_NTCT_update` too, so the fast
                // thermistors cool by 2.5 K while sparked, not by their own
                // element update's 10 K.
                if self_ctype == PT_NTCT
                    || self_ctype == PT_PTCT
                    || self_ctype == PT_FNTC
                    || self_ctype == PT_FPTC
                {
                    if part_temp[id] > 295.0f32 {
                        part_temp[id] -= 2.5f32;
                    }
                    // `SPRK.cpp` lines 241-246: a METL spark heats an adjacent
                    // sparked NTCT/PTCT/FNTC/FPTC to 473 K.
                    for n in 0u32..25u32 {
                        let dx = (n % 5u32) as i32 - 2i32;
                        let dy = (n / 5u32) as i32 - 2i32;
                        if dx != 0i32 || dy != 0i32 {
                            let nx = px + dx;
                            let ny = py + dy;
                            let nv = pmap_at(pmap, xres, yres, nx, ny);
                            if nv != 0u32 && !insulated(pmap, xres, yres, nx, ny, px, py) {
                                let ns = stage[pmap_slot(nv)];
                                if stage_ty(ns) == PT_SPRK
                                    && stage_ctype(ns) == PT_METL
                                    && stage_life(ns) > 0i32
                                    && stage_life(ns) < SPARK_LIFE
                                {
                                    part_temp[id] = 473.0f32;
                                }
                            }
                        }
                    }
                }

                // `SPRK.cpp` line 192: tungsten heating when sparked.
                if self_ctype == PT_TUNG && part_temp[id] < 3595.0f32 {
                    let roll = (hash_u32(frame_seed ^ ((id as u32) << 8u32) ^ PT_TUNG) % 20u32)
                        as i32
                        - 4i32;
                    part_temp[id] += roll as f32;
                }
                // `SPRK.cpp` `case PT_IRON`: a spark on iron splits adjacent
                // water into oxygen and hydrogen.
                if self_ctype == PT_IRON {
                    for n in 0u32..9u32 {
                        let dx = (n % 3u32) as i32 - 1i32;
                        let dy = (n / 3u32) as i32 - 1i32;
                        if dx != 0i32 || dy != 0i32 {
                            let nv = pmap_at(pmap, xres, yres, px + dx, py + dy);
                            if nv != 0u32 {
                                let nid = pmap_slot(nv);
                                let nty = part_ty[nid];
                                if nty == PT_DSTW || nty == PT_SLTW || nty == PT_WATR {
                                    let roll =
                                        hash_u32(frame_seed ^ ((nid as u32) << 8u32) ^ n) % 100u32;
                                    if roll == 0u32 {
                                        part_ty[nid] = PT_O2;
                                    } else if roll < 3u32 {
                                        part_ty[nid] = PT_H2;
                                    }
                                }
                            }
                        }
                    }
                }

                // `SPRK.cpp`: "ETRD is odd and conducts to others only at life
                // 5" - halfway through its life the arc relays into an idle
                // conductor and the electrode recovers instead of burning out.
                if self_ctype == PT_ETRD && self_life == 5i32 {
                    let mut relayed = false;
                    for n in 0u32..9u32 {
                        if !relayed {
                            let dx = (n % 3u32) as i32 - 1i32;
                            let dy = (n / 3u32) as i32 - 1i32;
                            if dx != 0i32 || dy != 0i32 {
                                let nv = pmap_at(pmap, xres, yres, px + dx, py + dy);
                                if nv != 0u32
                                    && !insulated(pmap, xres, yres, px + dx, py + dy, px, py)
                                {
                                    let nid = pmap_slot(nv);
                                    let nty = part_ty[nid];
                                    if stage_life(stage[nid]) == 0i32
                                        && (nty == PT_METL
                                            || nty == PT_BMTL
                                            || nty == PT_BRMT
                                            || nty == PT_LRBD
                                            || nty == PT_RBDM
                                            || nty == PT_PSCN
                                            || nty == PT_NSCN)
                                    {
                                        part_ty[nid] = PT_SPRK;
                                        part_ctype[nid] = nty;
                                        part_life[nid] = SPARK_LIFE;
                                        part_ty[id] = PT_ETRD;
                                        part_ctype[id] = PT_NONE;
                                        part_life[id] = 20i32;
                                        relayed = true;
                                    }
                                }
                            }
                        }
                    }
                }
            } else if self_life == 0i32
                && (self_ty == PT_INST
                    || self_ty == PT_QRTZ
                    || (elem_word(elem_words, self_ty, WORD_PROPERTIES) & PROP_CONDUCTS) != 0u32)
            {
                // `SPRK.cpp:330`: `QRTZ` is allowed into the receiver loop even
                // without `PROP_CONDUCTS`; its pressure gate samples the air cell
                // under this pixel, not the sender's.
                let cell = (((py as u32) / CELL) * xcells + ((px as u32) / CELL)) as usize;
                let receiver_pressurised = air_pv[cell] > QRTZ_PRESSURE;

                // `SPRK.cpp` lines 269-276: a METL spark heats adjacent idle
                // NTCT, PTCT, FNTC, FPTC or INWR to 473 K.
                if self_ty == PT_NTCT
                    || self_ty == PT_PTCT
                    || self_ty == PT_FNTC
                    || self_ty == PT_FPTC
                    || self_ty == PT_INWR
                {
                    for n in 0u32..25u32 {
                        let dx = (n % 5u32) as i32 - 2i32;
                        let dy = (n / 5u32) as i32 - 2i32;
                        if dx != 0i32 || dy != 0i32 {
                            let nx = px + dx;
                            let ny = py + dy;
                            let nv = pmap_at(pmap, xres, yres, nx, ny);
                            if nv != 0u32 && !insulated(pmap, xres, yres, nx, ny, px, py) {
                                let ns = stage[pmap_slot(nv)];
                                if stage_ty(ns) == PT_SPRK
                                    && stage_ctype(ns) == PT_METL
                                    && stage_life(ns) > 0i32
                                    && stage_life(ns) < SPARK_LIFE
                                {
                                    part_temp[id] = 473.0f32;
                                }
                            }
                        }
                    }
                }

                // Receiver: look for an adjacent spark that conducts into us.
                let mut converted = false;
                for n in 0u32..25u32 {
                    if !converted {
                        let dx = (n % 5u32) as i32 - 2i32;
                        let dy = (n / 5u32) as i32 - 2i32;
                        if dx != 0i32 || dy != 0i32 {
                            let nx = px + dx;
                            let ny = py + dy;
                            let nv = pmap_at(pmap, xres, yres, nx, ny);
                            if nv != 0u32 {
                                let nslot = pmap_slot(nv);
                                let ns = stage[nslot];
                                let nty = stage_ty(ns);
                                let nlife = stage_life(ns);
                                if nty == PT_SPRK && nlife > 0i32 && nlife < SPARK_LIFE {
                                    let sender = stage_ctype(ns);
                                    let conduct = spark_conducts(
                                        sender,
                                        self_ty,
                                        iabs(dx) + iabs(dy),
                                        ns,
                                        prev,
                                        receiver_pressurised,
                                    );
                                    // `NBLE` only conducts before it detonates.
                                    let mut allowed = conduct;
                                    if self_ty == PT_NBLE && (part_tmp[nslot] & 1i32) != 0i32 {
                                        allowed = false;
                                    }
                                    // The water family needs a younger spark
                                    // (`parts[i].life < 3`).
                                    if allowed
                                        && (self_ty == PT_WATR || self_ty == PT_SLTW)
                                        && nlife >= 3i32
                                    {
                                        allowed = false;
                                    }
                                    // `SPRK.cpp:363-373`: an `INST` receiver is
                                    // not converted here — the conduction only
                                    // arms it and `FloodINST` sparks the whole
                                    // connected wire. `flood_inst_kernel` is the
                                    // sole INST path, including its PSCN seed.
                                    if allowed
                                        && self_ty != PT_INST
                                        && !insulated(pmap, xres, yres, nx, ny, px, py)
                                    {
                                        let mut new_life = 4i32;
                                        if self_ty == PT_WATR {
                                            new_life = 6i32;
                                        } else if self_ty == PT_SLTW {
                                            new_life = 5i32;
                                        } else if self_ty == PT_RSST {
                                            // `SPRK.cpp:376-383`: a resist spark
                                            // bucket lives five frames.
                                            new_life = 5i32;
                                        }
                                        part_ty[id] = PT_SPRK;
                                        part_ctype[id] = self_ty;
                                        part_life[id] = new_life;
                                        // `create_part`: the good conductors
                                        // heat up by 10 K while below 673 K.
                                        if (self_ty == PT_METL
                                            || self_ty == PT_BMTL
                                            || self_ty == PT_BRMT
                                            || self_ty == PT_PSCN
                                            || self_ty == PT_NSCN
                                            || self_ty == PT_ETRD
                                            || self_ty == PT_NBLE
                                            || self_ty == PT_IRON)
                                            && part_temp[id] + 10.0f32 < 673.0f32
                                        {
                                            part_temp[id] += 10.0f32;
                                        }
                                        converted = true;
                                    }
                                }
                            }
                        }
                    }
                }
            } else if self_life == 0i32 && self_ty == PT_EMP_U32 {
                // `SPRK.cpp:277-286`: a spark adjacent to idle EMP triggers it.
                let mut triggered = false;
                for n in 0u32..25u32 {
                    if !triggered {
                        let dx = (n % 5u32) as i32 - 2i32;
                        let dy = (n / 5u32) as i32 - 2i32;
                        if dx != 0i32 || dy != 0i32 {
                            let nx = px + dx;
                            let ny = py + dy;
                            let nv = pmap_at(pmap, xres, yres, nx, ny);
                            if nv != 0u32 && !insulated(pmap, xres, yres, nx, ny, px, py) {
                                let ns = stage[pmap_slot(nv)];
                                if stage_ty(ns) == PT_SPRK
                                    && stage_life(ns) > 0i32
                                    && stage_life(ns) < SPARK_LIFE
                                {
                                    part_life[id] = 220i32;
                                    part_tmp[id] = 0i32;
                                    triggered = true;
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

/// `Simulation.cpp:595-597`: `isInst` — a plain `INST` pixel, or the spark it
/// turns into once the cascade reaches it. The wire-crossing tests need the
/// geometry, not the lit state, so this ignores `life`.
#[cube]
fn inst_wire(
    pmap: &[u32],
    part_ty: &mut [u32],
    part_ctype: &mut [u32],
    xres: u32,
    yres: u32,
    x: i32,
    y: i32,
) -> bool {
    let v = pmap_at(pmap, xres, yres, x, y);
    let mut wire = false;
    if v != 0u32 {
        let s = pmap_slot(v);
        if part_ty[s] == PT_INST || (part_ty[s] == PT_SPRK && part_ctype[s] == PT_INST) {
            wire = true;
        }
    }
    wire
}

/// A wire pixel the cascade has already sparked.
#[cube]
fn inst_lit(
    pmap: &[u32],
    part_ty: &mut [u32],
    part_ctype: &mut [u32],
    xres: u32,
    yres: u32,
    x: i32,
    y: i32,
) -> bool {
    let v = pmap_at(pmap, xres, yres, x, y);
    let mut lit = false;
    if v != 0u32 {
        let s = pmap_slot(v);
        if part_ty[s] == PT_SPRK && part_ctype[s] == PT_INST {
            lit = true;
        }
    }
    lit
}

/// `true` when the wire pixel sits inside a horizontal run.
#[cube]
fn inst_interior(
    pmap: &[u32],
    part_ty: &mut [u32],
    part_ctype: &mut [u32],
    xres: u32,
    yres: u32,
    x: i32,
    y: i32,
) -> bool {
    inst_wire(pmap, part_ty, part_ctype, xres, yres, x - 1i32, y)
        && inst_wire(pmap, part_ty, part_ctype, xres, yres, x + 1i32, y)
}

/// `true` when the wire pixel is a one-pixel-wide vertical run (`isInst` false
/// on both horizontal neighbours).
#[cube]
fn inst_single(
    pmap: &[u32],
    part_ty: &mut [u32],
    part_ctype: &mut [u32],
    xres: u32,
    yres: u32,
    x: i32,
    y: i32,
) -> bool {
    inst_wire(pmap, part_ty, part_ctype, xres, yres, x, y)
        && !inst_wire(pmap, part_ty, part_ctype, xres, yres, x - 1i32, y)
        && !inst_wire(pmap, part_ty, part_ctype, xres, yres, x + 1i32, y)
}

/// `Simulation.cpp:648-654` / `Simulation.cpp:672-678`: a horizontal run never
/// connects to a one-pixel vertical run that crosses it. The edge between
/// `(x, y)` and `(x, y + 1)` is such a crossing when one endpoint is inside a
/// horizontal run and the vertical run continues on that endpoint's far side.
#[cube]
fn inst_vertical_blocked(
    pmap: &[u32],
    part_ty: &mut [u32],
    part_ctype: &mut [u32],
    xres: u32,
    yres: u32,
    x: i32,
    y: i32,
) -> bool {
    let above = inst_interior(pmap, part_ty, part_ctype, xres, yres, x, y)
        && inst_single(pmap, part_ty, part_ctype, xres, yres, x, y - 1i32);
    let below = inst_interior(pmap, part_ty, part_ctype, xres, yres, x, y + 1i32)
        && inst_single(pmap, part_ty, part_ctype, xres, yres, x, y + 2i32);
    above || below
}

/// `Simulation.cpp:632-641` / `Simulation.cpp:657-668`: the one-pixel wire
/// crossing. The pixel is a three-wide horizontal run with a one-pixel vertical
/// run touching it from both sides, so the vertical run jumps over it.
#[cube]
fn inst_crossing(
    pmap: &[u32],
    part_ty: &mut [u32],
    part_ctype: &mut [u32],
    xres: u32,
    yres: u32,
    x: i32,
    y: i32,
) -> bool {
    inst_wire(pmap, part_ty, part_ctype, xres, yres, x, y)
        && inst_wire(pmap, part_ty, part_ctype, xres, yres, x - 1i32, y)
        && inst_wire(pmap, part_ty, part_ctype, xres, yres, x + 1i32, y)
        && inst_single(pmap, part_ty, part_ctype, xres, yres, x, y - 1i32)
        && inst_single(pmap, part_ty, part_ctype, xres, yres, x, y + 1i32)
}

/// `SPRK.cpp:340-348`: a `PSCN` spark within the 5x5 window (corners at
/// distance 4 excluded, as everywhere else) starts the wire cascade.
///
/// `SPRK.cpp:363-373` itself never converts the `INST` pixel: it calls
/// `FloodINST`, which sparks the whole connected run. This kernel is the port
/// of that call, so the `PSCN` edge is detected here instead of in the
/// conduction pass.
#[cube]
#[allow(clippy::too_many_arguments)]
fn inst_pscn_reaches(
    pmap: &[u32],
    part_ty: &mut [u32],
    part_ctype: &mut [u32],
    part_life: &mut [i32],
    xres: u32,
    yres: u32,
    x: i32,
    y: i32,
) -> bool {
    let mut reached = false;
    for n in 0u32..25u32 {
        let dx = (n % 5u32) as i32 - 2i32;
        let dy = (n / 5u32) as i32 - 2i32;
        if !reached && (dx != 0i32 || dy != 0i32) && iabs(dx) + iabs(dy) < 4i32 {
            let nx = x + dx;
            let ny = y + dy;
            let v = pmap_at(pmap, xres, yres, nx, ny);
            if v != 0u32 {
                let s = pmap_slot(v);
                if part_ty[s] == PT_SPRK
                    && part_ctype[s] == PT_PSCN
                    && part_life[s] > 0i32
                    && part_life[s] < SPARK_LIFE
                    && !insulated(pmap, xres, yres, nx, ny, x, y)
                {
                    reached = true;
                }
            }
        }
    }
    reached
}

/// Is the idle `INST` at `(x, y)` reached by a spark of the cascade?
///
/// The reference widens each popped pixel left and right unconditionally
/// (`Simulation.cpp:614-622`), steps vertically into a connected neighbour
/// (`Simulation.cpp:644-655` / `Simulation.cpp:669-680`) and jumps a crossing
/// vertical run over the horizontal one (`Simulation.cpp:630-668`).
#[cube]
fn inst_spark_reaches(
    pmap: &[u32],
    part_ty: &mut [u32],
    part_ctype: &mut [u32],
    part_life: &mut [i32],
    xres: u32,
    yres: u32,
    x: i32,
    y: i32,
) -> bool {
    let mut reached = inst_pscn_reaches(pmap, part_ty, part_ctype, part_life, xres, yres, x, y);
    // Horizontal widening has no crossing test.
    if !reached
        && (inst_lit(pmap, part_ty, part_ctype, xres, yres, x - 1i32, y)
            || inst_lit(pmap, part_ty, part_ctype, xres, yres, x + 1i32, y))
    {
        reached = true;
    }
    // A vertical step only connects across a real wire junction.
    if !reached
        && inst_lit(pmap, part_ty, part_ctype, xres, yres, x, y - 1i32)
        && !inst_vertical_blocked(pmap, part_ty, part_ctype, xres, yres, x, y - 1i32)
    {
        reached = true;
    }
    if !reached
        && inst_lit(pmap, part_ty, part_ctype, xres, yres, x, y + 1i32)
        && !inst_vertical_blocked(pmap, part_ty, part_ctype, xres, yres, x, y)
    {
        reached = true;
    }
    // The vertical run continues on the far side of the horizontal wire.
    if !reached
        && inst_lit(pmap, part_ty, part_ctype, xres, yres, x, y - 2i32)
        && inst_crossing(pmap, part_ty, part_ctype, xres, yres, x, y - 1i32)
    {
        reached = true;
    }
    if !reached
        && inst_lit(pmap, part_ty, part_ctype, xres, yres, x, y + 2i32)
        && inst_crossing(pmap, part_ty, part_ctype, xres, yres, x, y + 1i32)
    {
        reached = true;
    }
    reached
}

/// One round of the `INST` wire cascade (`SPRK.cpp:377-381` →
/// `Simulation.cpp:586-700`).
///
/// `FloodINST` sparks a whole connected `INST` run inside the same frame as the
/// triggering spark. A dispatch can only move the frontier one orthogonal hop,
/// so [`INST_FLOOD_ROUNDS`] rounds carry the spark along the wire. The state
/// change is monotone (`INST` → `SPRK`), so a round that reads a neighbour
/// before that neighbour is written only defers the hop to the next round; it
/// can never lose it. The pass re-runs every frame, so a run longer than the
/// cascade keeps lighting from the live sparks on its frontier on the following
/// frames.
///
/// Pixel parallel and slot-owned: a thread writes only its own slot. Every
/// neighbour read is `pmap` geometry plus the neighbour's live type, the same
/// monotone cascade as [`super::super::pipe::pump::pipe_flood_kernel`].
#[cube(launch_unchecked)]
pub(crate) fn flood_inst_kernel(
    pmap: &[u32],
    part_ty: &mut [u32],
    part_life: &mut [i32],
    part_ctype: &mut [u32],
    xres: u32,
    yres: u32,
) {
    let idx = ABSOLUTE_POS;
    if idx < (xres * yres) as usize {
        let v = pmap[idx];
        if v != 0u32 {
            let id = pmap_slot(v);
            // `Simulation.cpp:591-593`: only an idle `INST` can be sparked.
            if part_ty[id] == PT_INST && part_life[id] == 0i32 {
                let px = (idx as u32 % xres) as i32;
                let py = (idx as u32 / xres) as i32;
                if inst_spark_reaches(pmap, part_ty, part_ctype, part_life, xres, yres, px, py) {
                    // `create_part(-1, x, y, PT_SPRK)` (`Simulation.cpp:1867-1870`):
                    // the wire pixel becomes a spark carrying the element it
                    // replaced, with the usual four frames of life.
                    part_ty[id] = PT_SPRK;
                    part_ctype[id] = PT_INST;
                    part_life[id] = SPARK_LIFE;
                }
            }
        }
    }
}
