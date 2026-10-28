//! TPT's built-in Game of Life rules (`SimulationData::builtinGol`) and the
//! `B/S` rule encoding of `GOLString.cpp`.

/// Number of built-in GoL rulesets (`NGOL`).
pub const NGOL: usize = 24;

/// One entry of TPT's `SimulationData::builtinGol` table.
#[derive(Clone, Copy, Debug)]
pub struct BuiltinGol {
    /// Menu label (`"GOL"`, `"HLIF"`, ...).
    pub name: &'static str,
    /// Packed ruleset (see the module docs for the encoding).
    pub ruleset: u32,
    /// Colour of a live cell (`dcolour`), `0x00RRGGBB`.
    pub colour: u32,
    /// Colour of a dying cell (`tmp`), `0x00RRGGBB`.
    pub colour2: u32,
    /// Tooltip text from `SimulationData.cpp`.
    pub description: &'static str,
}

/// TPT's built-in GoL rules, in `NGOL` order (the `ctype` index).
pub static BUILTIN_GOL: [BuiltinGol; NGOL] = [
    BuiltinGol {
        name: "GOL",
        ruleset: 0x0080C,
        colour: 0x0CAC00,
        colour2: 0x0CAC00,
        description: "Game Of Life: Begin 3/Stay 23",
    },
    BuiltinGol {
        name: "HLIF",
        ruleset: 0x0480C,
        colour: 0xFF0000,
        colour2: 0xFF0000,
        description: "High Life: B36/S23",
    },
    BuiltinGol {
        name: "ASIM",
        ruleset: 0x038F0,
        colour: 0x0000FF,
        colour2: 0x0000FF,
        description: "Assimilation: B345/S4567",
    },
    BuiltinGol {
        name: "2X2",
        ruleset: 0x04826,
        colour: 0xFFFF00,
        colour2: 0xFFFF00,
        description: "2X2: B36/S125",
    },
    BuiltinGol {
        name: "DANI",
        ruleset: 0x1C9D8,
        colour: 0x00FFFF,
        colour2: 0x00FFFF,
        description: "Day and Night: B3678/S34678",
    },
    BuiltinGol {
        name: "AMOE",
        ruleset: 0x0A92A,
        colour: 0xFF00FF,
        colour2: 0xFF00FF,
        description: "Amoeba: B357/S1358",
    },
    BuiltinGol {
        name: "MOVE",
        ruleset: 0x14834,
        colour: 0xFFFFFF,
        colour2: 0xFFFFFF,
        description: "'Move' particles. Does not move things.. it is a life type: B368/S245",
    },
    BuiltinGol {
        name: "PGOL",
        ruleset: 0x0A90C,
        colour: 0xE05010,
        colour2: 0xE05010,
        description: "Pseudo Life: B357/S238",
    },
    BuiltinGol {
        name: "DMOE",
        ruleset: 0x1E9E0,
        colour: 0x500000,
        colour2: 0x500000,
        description: "Diamoeba: B35678/S5678",
    },
    BuiltinGol {
        name: "3-4",
        ruleset: 0x01818,
        colour: 0x500050,
        colour2: 0x500050,
        description: "3-4: B34/S34",
    },
    BuiltinGol {
        name: "LLIF",
        ruleset: 0x03820,
        colour: 0x505050,
        colour2: 0x505050,
        description: "Long Life: B345/S5",
    },
    BuiltinGol {
        name: "STAN",
        ruleset: 0x1C9EC,
        colour: 0x5000FF,
        colour2: 0x5000FF,
        description: "Stains: B3678/S235678",
    },
    BuiltinGol {
        name: "SEED",
        ruleset: 0x00400,
        colour: 0xFBEC7D,
        colour2: 0xFBEC7D,
        description: "Seeds: B2/S",
    },
    BuiltinGol {
        name: "MAZE",
        ruleset: 0x0083E,
        colour: 0xA8E4A0,
        colour2: 0xA8E4A0,
        description: "Maze: B3/S12345",
    },
    BuiltinGol {
        name: "COAG",
        ruleset: 0x189EC,
        colour: 0x9ACD32,
        colour2: 0x9ACD32,
        description: "Coagulations: B378/S235678",
    },
    BuiltinGol {
        name: "WALL",
        ruleset: 0x1F03C,
        colour: 0x0047AB,
        colour2: 0x0047AB,
        description: "Walled cities: B45678/S2345",
    },
    BuiltinGol {
        name: "GNAR",
        ruleset: 0x00202,
        colour: 0xE5B73B,
        colour2: 0xE5B73B,
        description: "Gnarl: B1/S1",
    },
    BuiltinGol {
        name: "REPL",
        ruleset: 0x0AAAA,
        colour: 0x259588,
        colour2: 0x259588,
        description: "Replicator: B1357/S1357",
    },
    BuiltinGol {
        name: "MYST",
        ruleset: 0x139E1,
        colour: 0x0C3C00,
        colour2: 0x0C3C00,
        description: "Mystery: B3458/S05678",
    },
    BuiltinGol {
        name: "LOTE",
        ruleset: 0x48938,
        colour: 0xFF0000,
        colour2: 0xFFFF00,
        description: "Living on the Edge: B37/S3458/4",
    },
    BuiltinGol {
        name: "FRG2",
        ruleset: 0x20816,
        colour: 0x006432,
        colour2: 0x00FF5A,
        description: "Like Frogs rule: B3/S124/3",
    },
    BuiltinGol {
        name: "STAR",
        ruleset: 0x98478,
        colour: 0x000040,
        colour2: 0x0000E6,
        description: "Like Star Wars rule: B278/S3456/6",
    },
    BuiltinGol {
        name: "FROG",
        ruleset: 0x21806,
        colour: 0x006400,
        colour2: 0x00FF00,
        description: "Frogs: B34/S12/3",
    },
    BuiltinGol {
        name: "BRAN",
        ruleset: 0x25440,
        colour: 0xFFFF00,
        colour2: 0x969600,
        description: "Brian 6: B246/S6/3",
    },
];

/// Mask covering the 21 bits a packed ruleset occupies.
pub const GOL_RULE_MASK: u32 = 0x001F_FFFF;

/// Default rule of the GOL tool (`ToolGOL.cpp` reads `CustomGOL.Rule`, whose
/// preference default is `B3/S23`).
pub const GOL_DEFAULT_RULE: u32 = 0x0080C;

/// `states - 2` as stored in the packed ruleset.
#[inline]
pub fn gol_state_exponent(rule: u32) -> u32 {
    (rule >> 17) & 0xF
}

/// Number of live states (`2..=17`), as `SerialiseGOLRule` reports it.
#[inline]
pub fn gol_states(rule: u32) -> i32 {
    (gol_state_exponent(rule) + 2) as i32
}

/// Highest `tmp2` value a live cell of this rule holds (`states - 1`).
#[inline]
pub fn gol_max_state(rule: u32) -> i32 {
    gol_state_exponent(rule) as i32 + 1
}

/// Resolves a `LIFE` `ctype` (builtin index or packed ruleset) to a ruleset.
#[inline]
pub fn resolve_gol_rule(ctype: u32) -> u32 {
    match BUILTIN_GOL.get(ctype as usize) {
        Some(builtin) => builtin.ruleset,
        None => ctype & GOL_RULE_MASK,
    }
}

/// `(dcolour, tmp)` a freshly created cell of this `ctype` carries.
///
/// Mirrors `Element_LIFE`'s create hook: a builtin rule resolves to its two
/// palette colours. A custom ruleset has no builtin entry, so its colours come
/// from the engine's `custom_gol` table ([`crate::gol_kernel::paint_life`]).
#[inline]
pub fn gol_colours(ctype: u32) -> (u32, i32) {
    match BUILTIN_GOL.get(ctype as usize) {
        Some(builtin) => (builtin.colour, builtin.colour2 as i32),
        None => (0, 0),
    }
}

/// Parses a `B3/S23` string into a packed ruleset (`GOLString.cpp`).
///
/// The `B` section must list at least one digit in `1..=8`, the `/S` section
/// any digits in `0..=8`, and an optional trailing `/N` selects `N` live states
/// (`2..=17`). Returns `None` for anything TPT would reject.
pub fn parse_gol_string(value: &str) -> Option<u32> {
    let bytes = value.as_bytes();
    if bytes.first() != Some(&b'B') {
        return None;
    }
    let mut pos = 1usize;

    let mut begin = 0u32;
    while let Some(&digit) = bytes.get(pos) {
        if !(b'1'..=b'8').contains(&digit) {
            break;
        }
        begin |= 1u32 << (digit - b'0');
        pos += 1;
    }
    if begin == 0 {
        return None;
    }

    if bytes.get(pos) != Some(&b'/') || bytes.get(pos + 1) != Some(&b'S') {
        return None;
    }
    pos += 2;

    let mut stay = 0u32;
    while let Some(&digit) = bytes.get(pos) {
        if !(b'0'..=b'8').contains(&digit) {
            break;
        }
        stay |= 1u32 << (digit - b'0');
        pos += 1;
    }

    let mut states = 2u32;
    if pos < bytes.len() {
        if bytes[pos] != b'/' {
            return None;
        }
        let tail = &bytes[pos + 1..];
        // `String::toNumber<unsigned int>(true)`: digits only, no sign, at
        // least one digit.
        if tail.is_empty() {
            return None;
        }
        let mut parsed = 0u32;
        for &byte in tail {
            if !byte.is_ascii_digit() {
                return None;
            }
            parsed = parsed
                .saturating_mul(10)
                .saturating_add((byte - b'0') as u32);
        }
        states = parsed;
        if !(2..=17).contains(&states) {
            return None;
        }
    }

    Some(stay | (begin << 8) | ((states - 2) << 17))
}

/// Serialises a packed ruleset back to its canonical `B3/S23` form.
pub fn serialise_gol_rule(rule: u32) -> String {
    let mut out = String::with_capacity(12);
    out.push('B');
    for digit in 1u32..=8u32 {
        if (rule >> (digit + 8)) & 1 != 0 {
            out.push((b'0' + digit as u8) as char);
        }
    }
    out.push_str("/S");
    for digit in 0u32..=8u32 {
        if (rule >> digit) & 1 != 0 {
            out.push((b'0' + digit as u8) as char);
        }
    }
    let extra = gol_state_exponent(rule);
    if extra != 0 {
        out.push('/');
        out.push_str(&(extra + 2).to_string());
    }
    out
}

/// Menu/tooltip label of a `ctype`: the builtin name or the canonical rule.
pub fn gol_rule_label(ctype: u32) -> String {
    match BUILTIN_GOL.get(ctype as usize) {
        Some(builtin) => builtin.name.to_string(),
        None => serialise_gol_rule(ctype & GOL_RULE_MASK),
    }
}
