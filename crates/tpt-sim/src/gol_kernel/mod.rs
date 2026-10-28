//! Game of Life (`LIFE`) — the port of TPT's `Simulation::SimulateGoL`.
//!
//! The reference simulation runs GoL once per frame at the end of `BeforeSim`
//! (`Simulation.cpp`, right before the particle update) and only when at least
//! one `PT_LIFE` particle exists, guarded by `GSPEED` generations per frame
//! (default `1`). [`step_gol_gpu`] is launched from
//! [`crate::SimulationEngine::step`] right after the reaction pass and before
//! the Newtonian gravity / movement passes, so a generation's births exist in
//! the particle pool before the movement pass rebuilds `pmap` for this frame.
//!
//! # Rule encoding (`GOLString.cpp`)
//!
//! A `LIFE` particle stores its rule in `ctype`, exactly like TPT:
//!
//! | bits    | meaning |
//! |---------|---------|
//! | `0..9`  | `S` — survive when the neighbour count is the bit index (`0..=8`) |
//! | `9..18` | `B` — born when `count + 8` is the bit index (`1..=8`) |
//! | `17..21`| `states - 2` (`2` states is encoded as `0`, `17` as `0xF`) |
//!
//! A `ctype` below [`NGOL`] is an index into [`BUILTIN_GOL`] (the menu presets);
//! anything else is a packed ruleset. Every valid ruleset is `>= 0x101`, so the
//! two ranges can never collide — TPT relies on the same invariant.
//!
//! `LIFE.cpp` also keeps per-particle render/state fields that this port
//! reproduces: `dcolour` = the rule's first colour (or the colour copied from a
//! neighbouring cell), `tmp` = the second colour, and `tmp2` = the number of
//! live states (`((rule >> 17) & 0xF) + 1`, i.e. `states - 1`). A cell counts
//! `tmp2` down while it is dying and is killed once it reaches zero, which is
//! what makes the multi-state rules (LOTE, FRG2, STAR, FROG, BRAN) linger for a
//! few generations.
//!
//! # The pass
//!
//! One generation is four dispatches over `xres * yres` (the last one a single
//! thread):
//!
//! * `gol_analyze_kernel` — a **staged read** of the current map. It counts
//!   the *live* `LIFE` neighbours of every pixel (a dying cell does not
//!   contribute, matching `SimulateGoL`'s `tmp2 == states` guard) and runs
//!   TPT's majority vote over the neighbouring rulesets to pick the rule a
//!   birth would use. It writes one packed word per pixel into `gol_meta` and
//!   clears the survivor counter.
//! * `gol_update_kernel` — reads only `gol_meta`, so no thread ever observes
//!   another thread's state change. It advances `tmp2`, kills cells and clears
//!   the birth vote of a stasis-frozen pixel.
//! * `gol_birth_kernel` — turns every surviving birth vote into one
//!   `reaction_kernel::shared::request_create` record.
//! * `gol_settle_kernel` — folds the generation's survivor/birth count into the
//!   device arm word.
//!
//! Deaths are resolved on the device (`part_ty = PT_NONE`); the movement pass
//! rebuilds `pmap` from the pool afterwards, so no map write is needed.
//! Births go through the same creation queue as the reaction passes: the engine
//! calls `reaction_kernel::begin_pending_creates` before this pass and
//! `reaction_kernel::drain_pending_creates` right after it, so the births exist
//! in the particle pool and `pmap` before the movement pass of the same frame.
//! The queue holds `SimBuffers::GPU_CREATE_COUNT` records per tick; a
//! generation that would create more (a whole-canvas birth) drops the surplus,
//! exactly as TPT's `create_part` fails when the pool has no free slot. No
//! particle, map or counter is copied back to the host.
//!
//! The reference gates the call on `elementCount[PT_LIFE] > 0`, a counter it
//! maintains on create/kill. The port keeps that decision on the device:
//! `counters[0]` is an arm word the host sets with a one-word upload when it
//! paints `LIFE`, restores a save or sees [`SimBuffers::active_count`] change,
//! and `gol_settle_kernel` keeps it set while a generation leaves a survivor or
//! queues a birth. Every per-pixel stage reads the word first and terminates
//! immediately when it is clear, so a simulation without `LIFE` pays one atomic
//! load per thread and no host read-back. The periodic detection probe
//! ([`GOL_PROBE_INTERVAL`]) re-arms the word in case a device-side creation was
//! missed by the host signals.
//!
//! `GSPEED` (frames per generation, default `1`) is not exposed by the port, so
//! one generation runs per frame.
//!
//! # Neighbourhood
//!
//! The Life torus is inset by one `CELL` on every side and wraps around, as in
//! `SimulateGoL`:
//!
//! ```text
//! nx = ((x + dx + xres - 3 * CELL) % (xres - 2 * CELL)) + CELL
//! ```
//!
//! `LIFE` pixels outside that area neither evolve nor count as neighbours, but
//! they do keep the pass alive (the survivor counter covers the whole canvas).
mod driver;
mod kernels;
mod rules;

pub use driver::{GolState, paint_life, step_gol_gpu};
pub use rules::{
    BUILTIN_GOL, BuiltinGol, GOL_DEFAULT_RULE, GOL_RULE_MASK, NGOL, gol_colours, gol_max_state,
    gol_rule_label, gol_state_exponent, gol_states, parse_gol_string, resolve_gol_rule,
    serialise_gol_rule,
};

use tpt_core::constants::{CELL, PT_LIFE, WL_STASIS};

/// `CELL` as a `u32`: the dead border of the Life torus.
const CELL_U32: u32 = CELL;
const _: () = assert!(CELL_U32 == 4);

/// `PT_LIFE` as a `u32` (kernel bodies cannot name the host constant).
const PT_LIFE_U32: u32 = PT_LIFE;
const _: () = assert!(PT_LIFE_U32 == 105);

/// `PT_NONE` as a `u32`.
const PT_NONE_U32: u32 = tpt_core::constants::PT_NONE;

/// `WL_STASIS` as a `u32`.
const WL_STASIS_U32: u32 = WL_STASIS as u32;
const _: () = assert!(WL_STASIS_U32 == 18);

/// Low 9 bits of a packed `pmap` word hold the element type.
const PMAP_TYPE_MASK: u32 = 0x1FF;
const _: () = assert!(PMAP_TYPE_MASK == 0x1FF);

/// `[armed, survivor_count]`. The arm word gates every per-pixel stage; the
/// survivor word is the sum the update and birth stages add to and the settle
/// stage folds back into the arm word. Neither is ever read by the host.
const COUNTER_WORDS: usize = 2;
/// Work units per worker cube.
const CUBE_DIM: u32 = 256;
/// Frames between detection probes while the device arm word is clear.
pub const GOL_PROBE_INTERVAL: u32 = 30;

// `gol_meta` word layout: the neighbour count sits in the low nibble, three
// flags follow it and the 23-bit payload starts at bit 7.
const META_COUNT_MASK: u32 = 0xF;
const META_OWN_LIFE: u32 = 1 << 4;
const META_HAS_BIRTH: u32 = 1 << 5;
const META_BORDER: u32 = 1 << 6;
const META_PAYLOAD_SHIFT: u32 = 7;
/// Highest slot / `ctype` the payload can hold (`2^23 - 1`); slot ids are
/// capped at `0x7F_FFFF` by the packed maps and rulesets at `0x1F_FFFF`.
const META_PAYLOAD_MASK: u32 = 0x007F_FFFF;
const _: () = assert!(META_COUNT_MASK & META_OWN_LIFE == 0);
const _: () = assert!(META_COUNT_MASK & META_HAS_BIRTH == 0);
const _: () = assert!(META_COUNT_MASK & META_BORDER == 0);
const _: () = assert!(META_OWN_LIFE < (1 << META_PAYLOAD_SHIFT));
const _: () = assert!(META_HAS_BIRTH < (1 << META_PAYLOAD_SHIFT));
const _: () = assert!(META_BORDER < (1 << META_PAYLOAD_SHIFT));
const _: () = assert!(META_PAYLOAD_MASK == (1 << 23) - 1);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn builtin_rulesets_round_trip_through_their_rule_strings() {
        for (index, builtin) in BUILTIN_GOL.iter().enumerate() {
            let text = serialise_gol_rule(builtin.ruleset);
            let parsed = parse_gol_string(&text)
                .unwrap_or_else(|| panic!("{text} ({}) did not parse", builtin.name));
            assert_eq!(parsed, builtin.ruleset, "{} ({text})", builtin.name);
            // A builtin index resolves to its ruleset, a raw ruleset to itself.
            assert_eq!(resolve_gol_rule(index as u32), builtin.ruleset);
            assert_eq!(resolve_gol_rule(builtin.ruleset), builtin.ruleset);
        }
    }

    #[test]
    fn rulesets_match_the_reference_table() {
        // Spot checks against `SimulationData::builtinGol`.
        assert_eq!(BUILTIN_GOL[0].ruleset, 0x0080C); // GOL: B3/S23
        assert_eq!(BUILTIN_GOL[1].ruleset, 0x0480C); // HLIF: B36/S23
        assert_eq!(BUILTIN_GOL[12].ruleset, 0x00400); // SEED: B2/S
        assert_eq!(BUILTIN_GOL[19].ruleset, 0x48938); // LOTE: B37/S3458/4
        assert_eq!(BUILTIN_GOL[23].ruleset, 0x25440); // BRAN: B246/S6/3
        assert_eq!(BUILTIN_GOL[23].name, "BRAN");
        assert_eq!(BUILTIN_GOL.len(), NGOL);
        // Every builtin ruleset stays above the index range, which is what
        // keeps `ctype < NGOL` unambiguous.
        for builtin in BUILTIN_GOL.iter() {
            assert!(
                builtin.ruleset as usize >= NGOL,
                "{} = {:#x} collides with an index",
                builtin.name,
                builtin.ruleset
            );
        }
    }

    #[test]
    fn rule_fields_decode_like_tpt() {
        // B3/S23: stay bits 2 and 3, begin bit 3, two states.
        let rule = parse_gol_string("B3/S23").unwrap();
        assert_eq!(rule, 0x0080C);
        assert_eq!(rule & 1, 0); // S0 off
        assert_eq!((rule >> 2) & 1, 1); // S2
        assert_eq!((rule >> 3) & 1, 1); // S3
        assert_eq!((rule >> (3 + 8)) & 1, 1); // B3
        assert_eq!((rule >> (2 + 8)) & 1, 0); // B2 off
        assert_eq!(gol_states(rule), 2);
        assert_eq!(gol_max_state(rule), 1);

        let lote = parse_gol_string("B37/S3458/4").unwrap();
        assert_eq!(gol_states(lote), 4);
        assert_eq!(gol_max_state(lote), 3);
        assert_eq!(serialise_gol_rule(lote), "B37/S3458/4");
    }

    #[test]
    fn invalid_rule_strings_are_rejected() {
        for text in [
            "",
            "B",
            "S23",
            "B0/S23",
            "B3S23",
            "B3/",
            "b3/s23",
            "B3/S23/1",
            "B3/S23/18",
            "B3/S23/x",
            "B3/S23/",
            "B9/S23",
            "B3/S9",
        ] {
            assert_eq!(parse_gol_string(text), None, "{text:?} must be rejected");
        }
        // Boundary cases TPT accepts.
        assert!(parse_gol_string("B1/S0/2").is_some());
        assert!(parse_gol_string("B12345678/S012345678/17").is_some());
        assert_eq!(gol_states(parse_gol_string("B1/S/17").unwrap()), 17);
        // An empty `S` section is legal (`stay` stays 0 in `ParseGOLString`).
        assert_eq!(parse_gol_string("B3/S"), Some(0x800));
        assert_eq!(serialise_gol_rule(0x800), "B3/S");
    }

    #[test]
    fn colours_follow_the_create_hook() {
        // Builtin rules carry both palette colours; custom rules carry none.
        assert_eq!(gol_colours(0), (0x0CAC00, 0x0CAC00));
        assert_eq!(gol_colours(19), (0xFF0000, 0xFFFF00));
        assert_eq!(gol_colours(0x80C), (0, 0));
        assert_eq!(gol_rule_label(0), "GOL");
        assert_eq!(gol_rule_label(0x80C), "B3/S23");
    }

    #[test]
    fn meta_layout_has_no_overlapping_fields() {
        // A maximum slot id and a maximum ruleset both survive the payload.
        let slot = META_PAYLOAD_MASK;
        let meta = META_OWN_LIFE | (slot << META_PAYLOAD_SHIFT);
        assert_eq!((meta >> META_PAYLOAD_SHIFT) & META_PAYLOAD_MASK, slot);
        assert_eq!(meta & META_COUNT_MASK, 0);
        let ctype = GOL_RULE_MASK;
        let meta = META_HAS_BIRTH | (ctype << META_PAYLOAD_SHIFT);
        assert_eq!((meta >> META_PAYLOAD_SHIFT) & META_PAYLOAD_MASK, ctype);
        assert_eq!(meta & META_COUNT_MASK, 0);
        assert_eq!(META_OWN_LIFE, 0x10);
        assert_eq!(META_HAS_BIRTH, 0x20);
        assert_eq!(META_BORDER, 0x40);
    }
}
