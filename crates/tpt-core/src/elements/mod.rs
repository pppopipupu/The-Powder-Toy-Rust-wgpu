//! Element definitions and the element registry.
//!
//! Every element carries the same physics knobs as The Powder Toy's original
//! `Element` table (`Advection`, `AirDrag`, `AirLoss`, `Loss`, `Collision`,
//! `Gravity`, `Diffusion`, `HotAir`, `Falldown`, `Flammable`, `Explosive`,
//! `Meltable`, `Hardness`, `HeatConduct`, `Weight`, temperature/pressure
//! transition points, ...), so the GPU kernels can stay data-driven.
//!
//! The built-in table covers TPT's classic element set plus the batch 1 and
//! batch 2 additions (`PT_NONE ..= PT_SEED`, ids `0..BUILTIN_ELEMENT_COUNT`).
//! Slots `BUILTIN_ELEMENT_COUNT..PT_NUM` are kept as disabled definitions and
//! are handed out by [`ElementRegistry::allocate_custom`] for elements created
//! from Lua via `elem.allocate(group, name)`.

use bytemuck::{Pod, Zeroable};
use serde::{Deserialize, Serialize};

use crate::constants::*;

/// Menu tab an element is listed under.
///
/// The discriminants match the ordering used by the UI tab bar; `Walls`,
/// `RigidBodies` and `Tools` are pseudo-categories that have no
/// [`ElementDef`] of their own but share the same tab machinery.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum MenuCategory {
    Walls = 0,
    Electronics = 1,
    Explosives = 2,
    Gases = 3,
    Liquids = 4,
    Powders = 5,
    Solids = 6,
    Radioactive = 7,
    Special = 8,
    RigidBodies = 9,
    Tools = 10,
    /// TPT `SC_POWERED`: elements that need a spark to do their job.
    Powered = 11,
    /// TPT `SC_SENSOR`: elements that observe their surroundings.
    Sensors = 12,
    /// TPT `SC_FORCE`: fields that push particles around.
    Force = 13,
    /// TPT `SC_LIFE`: Game of Life elements and their spawn points.
    Life = 14,
    /// TPT `SC_FAVORITES`: the user-pinned tab (no built-in element uses it).
    Favorites = 15,
    /// TPT `SC_DECO`: decoration-only tab (no built-in element uses it).
    Deco = 16,
}

/// `Element::CtypeDraw` hook (`Element.cpp:252-290`).
///
/// When the brush paints an element onto a pixel that already holds a particle,
/// `create_part(-2, ...)` calls the *occupant's* `CtypeDraw` instead of creating
/// anything: the painted element is recorded in the occupant's `ctype` (or
/// `tmp`), so a `FILT` painted over sand turns that sand into a filter rather
/// than being dropped. `v` is the creation variable the UI packs into the tool
/// id (`ElementTool.cpp:39`).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum CtypeDraw {
    /// The element keeps `create_part`'s default behaviour.
    #[default]
    None,
    /// `Element::basicCtypeDraw`: `ctype = t`.
    Basic,
    /// `Element::ctypeDrawVInCtype`: `ctype = t | (v << PMAPBITS)`.
    ValueInCtype,
    /// `Element::ctypeDrawVInTmp`: `ctype = t`, and `tmp = v` when `t` is `LIFE`.
    ValueInTmp,
    /// `CRAY::ctypeDraw`: `ValueInCtype` plus the `LIGH` power bits and the new
    /// element's spawn temperature.
    Cray,
    /// `STOR::ctypeDraw`: `Basic`, refusing solid targets.
    Stor,
    /// `Element_PCLN_ctypeDraw`: `ValueInTmp`, refusing `PSCN`/`NSCN`/`SPRK`.
    Pcln,
    /// `PSTN::ctypeDraw`: `Basic`, refusing `FRME`.
    Pstn,
}

/// Static description of one element type.
///
/// Values are copied from The Powder Toy's element table (with `CFDS == 1`
/// because this rewrite simulates one particle per pixel, not per 4x4 subcell),
/// so a `DUST` particle behaves like its C++ counterpart: same fall speed,
/// same air drag, same ignition temperature, etc.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ElementDef {
    /// Registry slot, `0..PT_NUM`.
    pub id: u32,
    /// Stable identifier, e.g. `"DEFAULT_PT_DUST"`. Used by saves and Lua.
    pub identifier: String,
    /// Short display name, e.g. `"DUST"`.
    pub name: String,
    /// Human readable description shown in the UI tooltip.
    pub description: String,
    /// Menu tab the element is grouped under.
    pub category: MenuCategory,
    /// `false` for the unused padding slots reserved for custom elements.
    pub enabled: bool,
    /// Base colour, `0x00RRGGBB`.
    pub colour: u32,
    /// Bitwise OR of the `TYPE_*` state flags and `PROP_*` behaviour flags.
    pub properties: u32,
    /// Wavelengths a reflected photon keeps (`Element::PhotonReflectWavelengths`,
    /// 26 bits). `Simulation.cpp:3044-3056` ANDs a bouncing photon's `ctype`
    /// with the mask of the pixel it hit, so the wavelengths this element
    /// absorbs are the ones missing from the mask. TPT's table default is
    /// `0x3FFF_FFFF` (reflect everything).
    pub photon_reflect: u32,
    /// Relative density used by the displacement rules.
    pub weight: i32,
    /// Fraction of the air velocity that is added to the particle each step.
    pub advection: f32,
    /// How strongly the particle is slowed down by air.
    pub air_drag: f32,
    /// Fraction of the particle velocity kept per step while airborne.
    pub air_loss: f32,
    /// Fraction of the particle velocity kept per step.
    pub loss: f32,
    /// Bounciness when hitting a wall or another particle.
    pub collision: f32,
    /// Downward acceleration in pixels per step squared.
    pub gravity: f32,
    /// Random lateral spread per step.
    pub diffusion: f32,
    /// Heat added to the surrounding air per step.
    pub hot_air: f32,
    /// `0` = gas/energy, `1` = powder, `2` = liquid.
    pub falldown: u32,
    /// Chance (0..=1000) to ignite when touching fire.
    pub flammable: i32,
    /// Explosion resistance / detonation class (0 = not explosive).
    pub explosive: i32,
    /// `1` if the element can melt into its `ctype` when heated.
    pub meltable: i32,
    /// Resistance against acid corrosion (0 = instantly dissolved).
    pub hardness: i32,
    /// Heat conductivity, `0..=255`.
    pub heat_conduct: u32,
    /// Spawn temperature in Kelvin.
    pub default_temp: f32,
    /// Spawn life (0 = not used by the element).
    pub default_life: i32,
    /// Spawn `ctype` (element id or an element-specific bit mask).
    pub default_ctype: u32,
    /// Spawn `tmp` (element-specific counter, e.g. GRVT gravity strength).
    pub default_tmp: i32,
    /// Spawn `tmp2`.
    pub default_tmp2: i32,
    /// Spawn `tmp3`.
    pub default_tmp3: i32,
    /// Spawn `tmp4`.
    pub default_tmp4: i32,
    /// `PMODE_*` / `FIRE_*` effect flags used by the fancy renderer.
    pub pixel_mode: u32,
    /// Emissive glow colour as `0xAARRGGBB` (`0` = not emissive).
    pub fire_colour: u32,
    /// `Element::CtypeDraw` hook: what the brush writes into this particle when
    /// another element is painted over it.
    #[serde(default)]
    pub ctype_draw: CtypeDraw,
    /// Pressure below which `low_pressure_transition` is applied.
    pub low_pressure: f32,
    /// Element id to become at `low_pressure`, `NT` for none, `ST` for special.
    pub low_pressure_transition: i32,
    /// Pressure above which `high_pressure_transition` is applied.
    pub high_pressure: f32,
    /// Element id to become at `high_pressure`, `NT` for none, `ST` for special.
    pub high_pressure_transition: i32,
    /// Temperature below which `low_temp_transition` is applied.
    pub low_temp: f32,
    /// Element id to become at `low_temp`, `NT` for none, `ST` for special.
    pub low_temp_transition: i32,
    /// Temperature above which `high_temp_transition` is applied.
    pub high_temp: f32,
    /// Element id to become at `high_temp`, `NT` for none, `ST` for special.
    pub high_temp_transition: i32,
}

impl Default for ElementDef {
    /// Neutral definition matching TPT's `Element()` constructor defaults.
    /// Used both for the unused padding slots and as the base for the
    /// [`elem!`] table below.
    fn default() -> Self {
        Self {
            id: 0,
            identifier: String::from("DEFAULT_INVALID"),
            name: String::new(),
            description: String::from("No description"),
            category: MenuCategory::Special,
            enabled: false,
            colour: 0xFF00FF,
            properties: TYPE_SOLID,
            photon_reflect: 0x3FFF_FFFF,
            weight: 50,
            advection: 0.0,
            air_drag: 0.0,
            air_loss: 1.0,
            loss: 1.0,
            collision: 0.0,
            gravity: 0.0,
            diffusion: 0.0,
            hot_air: 0.0,
            falldown: 0,
            flammable: 0,
            explosive: 0,
            meltable: 0,
            hardness: 30,
            heat_conduct: 128,
            default_temp: R_TEMP + 273.15,
            default_life: 0,
            default_ctype: 0,
            default_tmp: 0,
            default_tmp2: 0,
            default_tmp3: 0,
            default_tmp4: 0,
            pixel_mode: 0,
            fire_colour: 0,
            ctype_draw: CtypeDraw::None,
            low_pressure: IPL,
            low_pressure_transition: NT,
            high_pressure: IPH,
            high_pressure_transition: NT,
            low_temp: ITL,
            low_temp_transition: NT,
            high_temp: ITH,
            high_temp_transition: NT,
        }
    }
}

impl ElementDef {
    /// Applies `Element::defaultGraphics`'s property defaults.
    ///
    /// Only elements without a `Graphics` override run this
    /// ([`has_graphics_override`]); the renderer starts every particle from
    /// `PMODE_FLAT` (`Renderer.cpp:289`) and the function rewrites the mode:
    /// radioactive elements glow, liquids blur, and gases drop the pixel mode in
    /// favour of a half-brightness fire blend with a matching fire colour.
    pub fn apply_default_graphics(&mut self) {
        self.pixel_mode = PMODE_FLAT;
        self.fire_colour = 0;
        if self.properties & PROP_RADIOACTIVE != 0 {
            self.pixel_mode |= PMODE_GLOW;
        }
        if self.properties & TYPE_LIQUID != 0 {
            self.pixel_mode |= PMODE_BLUR;
        }
        if self.properties & TYPE_GAS != 0 {
            self.pixel_mode &= !PMODE;
            self.pixel_mode |= FIRE_BLEND | DECO_FIRE;
            let r = (self.colour >> 16) & 0xFF;
            let g = (self.colour >> 8) & 0xFF;
            let b = self.colour & 0xFF;
            self.fire_colour = (125 << 24) | ((r / 2) << 16) | ((g / 2) << 8) | (b / 2);
        }
    }

    /// `true` for photons, neutrons and electrons (the `TYPE_ENERGY` class).
    #[inline]
    pub fn is_energy(&self) -> bool {
        self.properties & TYPE_ENERGY != 0
    }

    /// `true` for powders (`TYPE_PART`).
    #[inline]
    pub fn is_powder(&self) -> bool {
        self.properties & TYPE_PART != 0
    }

    /// `true` for liquids (`TYPE_LIQUID`).
    #[inline]
    pub fn is_liquid(&self) -> bool {
        self.properties & TYPE_LIQUID != 0
    }

    /// `true` for gases (`TYPE_GAS`).
    #[inline]
    pub fn is_gas(&self) -> bool {
        self.properties & TYPE_GAS != 0
    }

    /// `true` for solids (`TYPE_SOLID`).
    #[inline]
    pub fn is_solid(&self) -> bool {
        self.properties & TYPE_SOLID != 0
    }
}

/// `u32` words per row of [`ElementRegistry::to_gpu_spawn_defaults_u32`].
pub const SPAWN_DEFAULT_WORDS: usize = 8;

/// TPT's `CarriesTypeIn & (1U << FIELD_CTYPE)` table.
///
/// An element listed here keeps a stored element id in its `ctype` across a
/// `create_part` that rebuilds it from another element's `tmp`/`ctype` pair:
/// `NEUT` liquefying `RSSS` (`NEUT.cpp:196-206`) and `PHOT` solidifying `RSST`
/// (`PHOT.cpp:117-133`) both restore `ctype = tmp` only when the stored element
/// carries its type in `ctype`. The C++ table marks 33 elements; `VIRS`, `VRSS`
/// and `VRSG` carry `tmp2` instead and are not part of this set.
pub fn carries_ctype(id: u32) -> bool {
    matches!(
        id,
        PT_BCLN
            | PT_BRMT
            | PT_CLNE
            | PT_CONV
            | PT_CRAY
            | PT_DRAY
            | PT_DTEC
            | PT_EXOT
            | PT_FIGH
            | PT_FIRE
            | PT_FOG
            | PT_FWRK
            | PT_ICEI
            | PT_LAVA
            | PT_LDTC
            | PT_PBCN
            | PT_PCLN
            | PT_PIPE
            | PT_PLSM
            | PT_PPIP
            | PT_PSTN
            | PT_RIME
            | PT_RSSS
            | PT_RSST
            | PT_SNOW
            | PT_SPRK
            | PT_STKM
            | PT_STKM2
            | PT_STOR
            | PT_WTRV
    )
}

/// `true` when the element installs its own `Graphics` function.
///
/// TPT's `Element` constructor starts with `Graphics(&Element::defaultGraphics)`
/// (`Element.cpp:49`), so only an element that assigns `Graphics = &...` (or the
/// explicit `Graphics = nullptr` of `DUST`, `DRAY` and `SAWD`) leaves the
/// property defaults behind. The C++ table marks 90 elements; the port keeps
/// their static mode bits in the data table and their dynamic half in
/// `render_kernel/particles/`.
pub fn has_graphics_override(id: u32) -> bool {
    matches!(id, |PT_PLNT| PT_LCRY
        | PT_STKM
        | PT_SWCH
        | PT_COAL
        | PT_GLOW
        | PT_PCLN
        | PT_HSWC
        | PT_DLAY
        | PT_CBNW
        | PT_STOR
        | PT_PVOD
        | PT_SPNG
        | PT_PUMP
        | PT_PIPE
        | PT_BIZR
        | PT_BIZRG
        | PT_BIZRS
        | PT_THDR
        | PT_HFLM
        | PT_FIRW
        | PT_AMTR
        | PT_LIFE
        | PT_LIGH
        | PT_DEST
        | PT_DEUT
        | PT_WARP
        | PT_GRAV
        | PT_PRTI
        | PT_PRTO
        | PT_VINE
        | PT_INVIS
        | PT_WIFI
        | PT_FILT
        | PT_BRAY
        | PT_STKM2
        | PT_BOMB
        | PT_C5
        | PT_QRTZ
        | PT_PQRT
        | PT_EMP
        | PT_ACEL
        | PT_DCEL
        | PT_GEL
        | PT_TRON
        | PT_EXOT
        | PT_EMBR
        | PT_SOAP
        | PT_PBCN
        | PT_CLST
        | PT_WIRE
        | PT_FIGH
        | PT_PPIP
        | PT_DMG
        | PT_VIBR
        | PT_BVBR
        | PT_PSTN
        | PT_FRME
        | PT_GOLD
        | PT_TUNG
        | PT_PROT
        | PT_VIRS
        | PT_VRSS
        | PT_VRSG
        | PT_DRAY
        | PT_CRMC
        | PT_SAWD
        | PT_POLO
        | PT_SLCN
        | PT_PTNM
        | PT_ROCK
        | PT_LITH
        | PT_BASE
        | PT_SEED
        | PT_SPRK
        | PT_PHOT
        | PT_NEUT
        | PT_ELEC
        | PT_GRVT
        | PT_SMKE
        | PT_PLSM
        | PT_LAVA
        | PT_ACID
        | PT_DUST
        | PT_BCOL
        | PT_WOOD
        | PT_BRCK
        | PT_FIRE
        | PT_GBMB
        | PT_GPMP)
}

/// GPU-uploadable flat struct of element physics properties.
///
/// `#[repr(C)]`, 32 32-bit words (128 bytes) per element, so the whole table is
/// `PT_NUM * 128 == 25 600` bytes and can be uploaded as a single storage buffer.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Pod, Zeroable)]
pub struct GpuElementProps {
    pub colour: u32,
    pub properties: u32,
    pub weight: i32,
    pub falldown: u32,
    pub advection: f32,
    pub air_drag: f32,
    pub air_loss: f32,
    pub loss: f32,
    pub collision: f32,
    pub gravity: f32,
    pub diffusion: f32,
    pub hot_air: f32,
    pub flammable: i32,
    pub explosive: i32,
    pub hardness: i32,
    pub heat_conduct: u32,
    pub default_temp: f32,
    pub default_life: i32,
    pub low_pressure: f32,
    pub low_pressure_transition: i32,
    pub high_pressure: f32,
    pub high_pressure_transition: i32,
    pub low_temp: f32,
    pub low_temp_transition: i32,
    pub high_temp: f32,
    pub high_temp_transition: i32,
    pub meltable: i32,
    pub default_ctype: u32,
    /// `PMODE_*` / `FIRE_*` effect flags (fancy renderer).
    pub pixel_mode: u32,
    /// Emissive glow colour as `0xAARRGGBB`.
    pub fire_colour: u32,
    /// `Graphics` alpha (`cola`, `0..=255`) the renderer's `PMODE_BLEND` and
    /// `PMODE_ADD` stages composite with; `255` is TPT's default.
    pub cola: u32,
    /// `1` when the element restores a stored `ctype` on creation
    /// (`ElementDef::carries_ctype`).
    pub carries_ctype: u32,
}

impl GpuElementProps {
    /// Flattens an [`ElementDef`] into its GPU representation.
    pub fn from_def(def: &ElementDef) -> Self {
        Self {
            colour: def.colour,
            properties: def.properties,
            weight: def.weight,
            falldown: def.falldown,
            advection: def.advection,
            air_drag: def.air_drag,
            air_loss: def.air_loss,
            loss: def.loss,
            collision: def.collision,
            gravity: def.gravity,
            diffusion: def.diffusion,
            hot_air: def.hot_air,
            flammable: def.flammable,
            explosive: def.explosive,
            hardness: def.hardness,
            heat_conduct: def.heat_conduct,
            default_temp: def.default_temp,
            default_life: def.default_life,
            low_pressure: def.low_pressure,
            low_pressure_transition: def.low_pressure_transition,
            high_pressure: def.high_pressure,
            high_pressure_transition: def.high_pressure_transition,
            low_temp: def.low_temp,
            low_temp_transition: def.low_temp_transition,
            high_temp: def.high_temp,
            high_temp_transition: def.high_temp_transition,
            meltable: def.meltable,
            default_ctype: def.default_ctype,
            pixel_mode: def.pixel_mode,
            fire_colour: def.fire_colour,
            cola: 255,
            carries_ctype: u32::from(carries_ctype(def.id)),
        }
    }
}

impl From<&ElementDef> for GpuElementProps {
    fn from(def: &ElementDef) -> Self {
        Self::from_def(def)
    }
}

/// All element definitions plus the precomputed displacement matrix.
#[derive(Clone, Debug)]
pub struct ElementRegistry {
    /// Definitions for every slot, always exactly [`PT_NUM`] entries.
    pub elements: Vec<ElementDef>,
    /// Flat `[PT_NUM * PT_NUM]` matrix indexed by `moving_ty * PT_NUM + dest_ty`.
    /// `0` = blocked, `1` = swap, `2` = share the cell, `3` = special handling.
    pub can_move: Vec<u8>,
}

impl Default for ElementRegistry {
    fn default() -> Self {
        Self::new_default()
    }
}

impl ElementRegistry {
    /// Builds the registry with the `BUILTIN_ELEMENT_COUNT` built-in slots and
    /// the disabled slots above them, which are reserved for Lua-defined
    /// elements.
    pub fn new_default() -> Self {
        let mut elements = vec![ElementDef::default(); PT_NUM];

        for def in builtin_elements() {
            let slot = def.id as usize;
            debug_assert!(slot < PT_NUM, "built-in element id out of range");
            elements[slot] = def;
        }

        let mut registry = Self {
            elements,
            can_move: Vec::new(),
        };
        registry.rebuild_can_move();
        registry
    }

    /// Returns the definition for `id`, falling back to `PT_NONE` for ids
    /// outside the registry instead of panicking on stale save data.
    #[inline]
    pub fn get(&self, id: u32) -> &ElementDef {
        match self.elements.get(id as usize) {
            Some(def) => def,
            None => &self.elements[PT_NONE as usize],
        }
    }

    /// Mutable access to a definition, `None` when `id` is out of range.
    #[inline]
    pub fn get_mut(&mut self, id: u32) -> Option<&mut ElementDef> {
        self.elements.get_mut(id as usize)
    }

    /// Case-insensitive lookup by display name (`"DUST"`) or identifier
    /// (`"DEFAULT_PT_DUST"`).
    pub fn find_by_name(&self, name: &str) -> Option<u32> {
        let needle = name.trim();
        if needle.is_empty() {
            return None;
        }
        self.elements
            .iter()
            .find(|def| {
                !def.name.is_empty()
                    && (def.name.eq_ignore_ascii_case(needle)
                        || def.identifier.eq_ignore_ascii_case(needle))
            })
            .map(|def| def.id)
    }

    /// Allocates a disabled slot for a Lua-defined element.
    ///
    /// `group` and `name` are upper-cased and joined into
    /// `"<GROUP>_PT_<NAME>"`, mirroring TPT's `elem.allocate`. Returns `None`
    /// when the identifier is invalid, already taken, or the table is full.
    pub fn allocate_custom(&mut self, group: &str, name: &str) -> Option<u32> {
        let group = group.trim().to_ascii_uppercase();
        let name = name.trim().to_ascii_uppercase();
        if group.is_empty() || name.is_empty() {
            return None;
        }
        // TPT forbids underscores because they separate the identifier parts.
        if group.contains('_') || name.contains('_') {
            return None;
        }
        // "DEFAULT" is reserved for the built-in elements.
        if group == "DEFAULT" {
            return None;
        }

        let identifier = format!("{group}_PT_{name}");
        if self
            .elements
            .iter()
            .any(|def| def.enabled && def.identifier.eq_ignore_ascii_case(&identifier))
        {
            return None;
        }

        // TPT allocates from the top of the table downwards so that element ids
        // stay inside one byte for as long as possible (better save compatibility).
        let slot = (BUILTIN_ELEMENT_COUNT..PT_NUM)
            .rev()
            .chain((1..BUILTIN_ELEMENT_COUNT).rev())
            .find(|&idx| !self.elements[idx].enabled)?;

        let mut def = ElementDef {
            id: slot as u32,
            identifier,
            name,
            description: String::from("Custom element"),
            category: MenuCategory::Special,
            enabled: true,
            ..ElementDef::default()
        };
        // A Lua element cannot install a `Graphics` function, so it keeps
        // `Element::defaultGraphics` and its property-driven render mode.
        def.apply_default_graphics();
        self.elements[slot] = def;
        self.rebuild_can_move();
        Some(slot as u32)
    }

    /// Recomputes the displacement matrix from the current element table.
    ///
    /// Must be called after element properties change (Lua `elem.property`)
    /// or a custom element is allocated.
    ///
    /// Semantics (matching TPT's `SimulationData::init_can_move`):
    /// - `0` — the mover cannot enter the destination (bounce/block),
    /// - `1` — the mover swaps places with the destination,
    /// - `2` — both particles may occupy the same cell (pass-through),
    /// - `3` — special case evaluated by the movement kernel.
    pub fn rebuild_can_move(&mut self) {
        // Defensive: keep the matrix usable even if a caller built the registry
        // by hand with a short element vector.
        while self.elements.len() < PT_NUM {
            let id = self.elements.len() as u32;
            self.elements.push(ElementDef {
                id,
                ..ElementDef::default()
            });
        }

        let mut table = vec![0u8; PT_NUM * PT_NUM];
        let n = PT_NUM;

        // `SimulationData.cpp:107-113`: a mover swaps by default; `PT_NONE`
        // never moves, and empty cells accept every mover with a plain swap.
        for moving in 1..n {
            for dest in 0..n {
                table[moving * n + dest] = 1;
            }
        }
        // `SimulationData.cpp:115-117`: photons start out passing through
        // everything. The weight rule below clears that again for every
        // element heavier than a photon, and `PROP_PHOTPASS` restores it.
        for dest in 1..n {
            table[PT_PHOT as usize * n + dest] = 2;
        }
        // `SimulationData.cpp:119-143`: weight, neutron and energy rules.
        for moving in 1..n {
            let m = &self.elements[moving];
            let row = moving * n;
            for dest in 1..n {
                let d = &self.elements[dest];
                if m.weight <= d.weight || dest as u32 == PT_GEL {
                    table[row + dest] = 0;
                }
                if moving as u32 == PT_NEUT {
                    if d.properties & PROP_NEUTPASS != 0 {
                        table[row + dest] = 2;
                    }
                    if d.properties & PROP_NEUTABSORB != 0 {
                        table[row + dest] = 1;
                    }
                    if d.properties & PROP_NEUTPENETRATE != 0 {
                        table[row + dest] = 1;
                    }
                }
                if dest as u32 == PT_NEUT && m.properties & PROP_NEUTPENETRATE != 0 {
                    table[row + dest] = 0;
                }
                if m.properties & TYPE_ENERGY != 0 && d.properties & TYPE_ENERGY != 0 {
                    table[row + dest] = 2;
                }
            }
        }
        // `SimulationData.cpp:144-158`: stickmen walk through liquids, gases,
        // empty cells and `PRTO`/`SPAWN`; sparks never move.
        for dest in 0..n {
            let d = &self.elements[dest];
            let mut stkm = 0u8;
            if d.properties & (TYPE_LIQUID | TYPE_GAS) != 0 {
                stkm = 2;
            }
            if dest == 0
                || dest as u32 == PT_PRTO
                || dest as u32 == PT_SPAWN
                || dest as u32 == PT_SPAWN2
            {
                stkm = 2;
            }
            table[PT_STKM as usize * n + dest] = stkm;
            table[PT_STKM2 as usize * n + dest] = stkm;
            table[PT_FIGH as usize * n + dest] = stkm;
            table[PT_SPRK as usize * n + dest] = 0;
        }
        // `SimulationData.cpp:159-187`: shared overrides. Holes eat everything,
        // stickmen are never displaced, `INVIS`/`VOID`/`PVOD` decide at move
        // time (`3`), concrete is immovable, and energy passes through the
        // energy-absorbing solids.
        for moving in 1..n {
            let m = &self.elements[moving];
            let row = moving * n;
            table[row + PT_BHOL as usize] = 1;
            table[row + PT_NBHL as usize] = 1;
            table[row + PT_STKM as usize] = 0;
            table[row + PT_STKM2 as usize] = 0;
            table[row + PT_FIGH as usize] = 0;
            table[row + PT_INVIS as usize] = 3;
            table[row + PT_CNCT as usize] = 0;
            table[row + PT_PVOD as usize] = 3;
            table[row + PT_VOID as usize] = 3;
            table[row + PT_EMBR as usize] = 0;
            table[PT_EMBR as usize * n + moving] = 0;
            if m.properties & TYPE_ENERGY != 0 {
                table[row + PT_VIBR as usize] = 1;
                table[row + PT_BVBR as usize] = 1;
            }
            if m.properties & TYPE_PART != 0 {
                table[row + PT_SAWD as usize] = 0;
            }
        }
        // `SimulationData.cpp:188-199`: photons pass through `PROP_PHOTPASS`
        // destinations; protons and gravitons through everything but the
        // listed blockers.
        for dest in 0..n {
            let d = &self.elements[dest];
            if d.properties & PROP_PHOTPASS != 0 {
                table[PT_PHOT as usize * n + dest] = 2;
            }
            if !matches!(
                dest as u32,
                PT_DMND | PT_INSL | PT_VOID | PT_PVOD | PT_VIBR | PT_BVBR | PT_PRTI | PT_PRTO
            ) {
                table[PT_PROT as usize * n + dest] = 2;
                table[PT_GRVT as usize * n + dest] = 2;
            }
        }
        // `SimulationData.cpp:200-230`: the remaining per-pair exceptions.
        for dest in [PT_DMND, PT_CLNE, PT_PCLN, PT_BCLN, PT_PBCN, PT_ROCK] {
            table[PT_DEST as usize * n + dest as usize] = 0;
        }
        let pair = |moving: u32, dest: u32| (moving as usize) * n + dest as usize;
        table[pair(PT_NEUT, PT_INVIS)] = 2;
        table[pair(PT_ELEC, PT_LCRY)] = 2;
        table[pair(PT_ELEC, PT_EXOT)] = 2;
        table[pair(PT_ELEC, PT_GLOW)] = 2;
        table[pair(PT_PHOT, PT_LCRY)] = 3;
        table[pair(PT_PHOT, PT_GPMP)] = 3;
        table[pair(PT_PHOT, PT_BIZR)] = 2;
        table[pair(PT_ELEC, PT_BIZR)] = 2;
        table[pair(PT_PHOT, PT_BIZRG)] = 2;
        table[pair(PT_ELEC, PT_BIZRG)] = 2;
        table[pair(PT_PHOT, PT_BIZRS)] = 2;
        table[pair(PT_ELEC, PT_BIZRS)] = 2;
        table[pair(PT_BIZR, PT_FILT)] = 2;
        table[pair(PT_BIZRG, PT_FILT)] = 2;
        table[pair(PT_ANAR, PT_WHOL)] = 1;
        table[pair(PT_ANAR, PT_NWHL)] = 1;
        table[pair(PT_ELEC, PT_DEUT)] = 1;
        table[pair(PT_THDR, PT_THDR)] = 2;
        table[pair(PT_EMBR, PT_EMBR)] = 2;
        table[pair(PT_TRON, PT_SWCH)] = 3;
        table[pair(PT_ELEC, PT_RSST)] = 2;
        table[pair(PT_ELEC, PT_RSSS)] = 2;

        self.can_move = table;
    }

    /// Reads one entry of the displacement matrix. Out-of-range ids are
    /// treated as immovable (`0`).
    #[inline]
    pub fn can_move_val(&self, moving_ty: u32, dest_ty: u32) -> u8 {
        let moving = moving_ty as usize;
        let dest = dest_ty as usize;
        if moving >= PT_NUM || dest >= PT_NUM {
            return 0;
        }
        self.can_move
            .get(moving * PT_NUM + dest)
            .copied()
            .unwrap_or(0)
    }

    /// Flattens every definition for upload to the GPU, always [`PT_NUM`] entries.
    pub fn to_gpu_props(&self) -> Vec<GpuElementProps> {
        self.elements
            .iter()
            .map(GpuElementProps::from_def)
            .collect()
    }

    /// Flattens the displacement matrix into `u32`s for the GPU, always
    /// `PT_NUM * PT_NUM` entries.
    pub fn to_gpu_can_move_u32(&self) -> Vec<u32> {
        let mut out = vec![0u32; PT_NUM * PT_NUM];
        for (dst, &src) in out.iter_mut().zip(self.can_move.iter()) {
            *dst = u32::from(src);
        }
        out
    }

    /// TPT's `elements[t].DefaultProperties`, the row `create_part` copies into a
    /// slot before the caller overrides individual fields (`Simulation.cpp:1941`).
    ///
    /// Eight `u32` words per element, always [`PT_NUM`] rows, so a device kernel
    /// can seed a newly created particle exactly as the host allocator does:
    ///
    /// | word | field                       |
    /// |------|-----------------------------|
    /// | 0    | `temp` (`f32` bits)         |
    /// | 1    | `life`                      |
    /// | 2    | `ctype`                     |
    /// | 3    | `tmp`                       |
    /// | 4    | `tmp2`                      |
    /// | 5    | `tmp3`                      |
    /// | 6    | `tmp4`                      |
    /// | 7    | `flags` (`0` for every row) |
    ///
    /// It is a separate table rather than four more columns of
    /// [`GpuElementProps`] because the renderer's kernels mirror the property
    /// row's stride as a literal; widening it would silently mis-align every one
    /// of them.
    pub fn to_gpu_spawn_defaults_u32(&self) -> Vec<u32> {
        let mut out = vec![0u32; PT_NUM * SPAWN_DEFAULT_WORDS];
        for (slot, def) in self.elements.iter().enumerate() {
            let base = slot * SPAWN_DEFAULT_WORDS;
            out[base] = def.default_temp.to_bits();
            out[base + 1] = def.default_life as u32;
            out[base + 2] = def.default_ctype;
            out[base + 3] = def.default_tmp as u32;
            out[base + 4] = def.default_tmp2 as u32;
            out[base + 5] = def.default_tmp3 as u32;
            out[base + 6] = def.default_tmp4 as u32;
            out[base + 7] = 0;
        }
        out
    }
}

/// Shorthand for one row of the built-in element table.
///
/// Only the fields that differ from [`ElementDef::default`] need to be listed.
macro_rules! elem {
    (
        id: $id:expr,
        ident: $ident:expr,
        name: $name:expr,
        cat: $cat:expr,
        // `photon_reflect` is optional and may sit right after `cat` so an
        // element whose wavelength mask differs from TPT's reflect-everything
        // default keeps the mask next to its identity fields.
        $(photon_reflect: $photon_reflect:expr,)?
        colour: $colour:expr,
        props: $props:expr,
        weight: $weight:expr,
        desc: $desc:expr
        $(, $field:ident : $val:expr)* $(,)?
    ) => {
        ElementDef {
            id: $id,
            identifier: $ident.into(),
            name: $name.into(),
            description: $desc.into(),
            category: $cat,
            enabled: true,
            $(photon_reflect: $photon_reflect,)?
            colour: $colour,
            properties: $props,
            weight: $weight,
            $($field: $val,)*
            ..ElementDef::default()
        }
    };
}

/// `PMODE_*` / `FIRE_*` effect flags of the fancy renderer.
///
/// Mirrored from TPT's `src/simulation/ElementGraphics.h`; the render kernel
/// stores them verbatim in [`ElementDef::pixel_mode`] and interprets the bits
/// itself. They are private to this module because the table is the only writer.
///
/// The values below are transcribed from each element's `Graphics` function in
/// `src/simulation/elements/*.cpp`. An element without one keeps the
/// constructor's `Element::defaultGraphics` pointer, whose property defaults
/// [`ElementDef::apply_default_graphics`] reproduces; every other element starts
/// from `PMODE_FLAT`, which is also TPT's default in `Renderer::render_parts`.
/// `fire_colour` is TPT's `firea/firer/fireg/fireb` packed as `0xAARRGGBB`; an
/// entry with a non-zero alpha but a zero RGB means "emit the element's own
/// `Graphics` colour" and is used by the emitters whose colour is computed per
/// particle (FIRE, PLSM, LAVA, PHOT).
const PMODE: u32 = 0x0000_0FFF;
const PMODE_FLAT: u32 = 0x0000_0001;
const PMODE_BLUR: u32 = 0x0000_0004;
const PMODE_GLOW: u32 = 0x0000_0008;
const PMODE_FLARE: u32 = 0x0000_0020;
const PMODE_ADD: u32 = 0x0000_0080;
const PMODE_BLEND: u32 = 0x0000_0100;
const NO_DECO: u32 = 0x0000_1000;
const DECO_FIRE: u32 = 0x0000_2000;
const FIRE_ADD: u32 = 0x0001_0000;
const FIRE_BLEND: u32 = 0x0002_0000;
const FIRE_SPARK: u32 = 0x0004_0000;

/// Built-in element table, split by menu category.
///
/// Declared after [`elem!`] because `macro_rules!` scoping is textual: the
/// macro is only visible to child modules declared below its definition.
mod data;

/// The `BUILTIN_ELEMENT_COUNT` built-in slots: `PT_NONE` plus the simulated
/// elements, in the original table order.
///
/// The definitions live in [`data`], grouped by [`MenuCategory`]. The sections
/// are emitted in the order of the original monolithic table, so every slot
/// keeps the same registry index; the batch 1 additions are emitted last
/// because their ids were appended above the classic ones.
fn builtin_elements() -> Vec<ElementDef> {
    let mut elements = Vec::with_capacity(BUILTIN_ELEMENT_COUNT);

    data::special::push_empty(&mut elements);
    data::powders::push(&mut elements);
    data::liquids::push(&mut elements);
    data::gases::push(&mut elements);
    data::special::push_explosives(&mut elements);
    data::solids::push(&mut elements);
    data::energy::push_electronics(&mut elements);
    data::special::push_specials(&mut elements);
    data::energy::push_radioactive(&mut elements);
    data::special::push_gravity(&mut elements);
    // Batch 1 was appended above the classic ids, so it is emitted after
    // them: its slots are `PT_GOO ..= PT_TSNS` in TPT's element order.
    data::batch1::push(&mut elements);
    // Batch 2 continues the dense numbering with the remaining TPT elements,
    // `PT_PLUT ..= PT_SEED`, again in TPT's element order.
    data::batch2::push(&mut elements);

    // Elements without their own `Graphics` function keep the constructor's
    // `Element::defaultGraphics`, which derives the render mode from the
    // properties instead of the data table.
    for def in &mut elements {
        if !has_graphics_override(def.id) {
            def.apply_default_graphics();
        }
    }

    elements
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn builtin_table_is_complete() {
        let reg = ElementRegistry::new_default();
        assert_eq!(reg.elements.len(), PT_NUM);
        assert_eq!(reg.can_move.len(), PT_NUM * PT_NUM);

        for id in 0..BUILTIN_ELEMENT_COUNT {
            assert_eq!(reg.elements[id].id, id as u32, "slot {id} has a wrong id");
            assert!(!reg.elements[id].name.is_empty(), "slot {id} has no name");
        }
        assert!(
            !reg.elements[PT_NONE as usize].enabled,
            "PT_NONE must not be drawable"
        );
        for id in 1..BUILTIN_ELEMENT_COUNT {
            assert!(reg.elements[id].enabled, "built-in slot {id} is disabled");
        }
        for id in BUILTIN_ELEMENT_COUNT..PT_NUM {
            assert!(!reg.elements[id].enabled, "padding slot {id} is enabled");
        }

        assert_eq!(reg.get(PT_NONE).name, "NONE");
        assert_eq!(reg.get(PT_DUST).name, "DUST");
        assert_eq!(reg.get(PT_LN2).name, "LN2");
        assert_eq!(reg.get(PT_C4).name, "C-4");
        assert_eq!(reg.get(PT_C4).category, MenuCategory::Explosives);
        // Batch 1 and batch 2 head/tail, so an id shift is caught here.
        assert_eq!(reg.get(PT_GOO).name, "GOO");
        assert_eq!(reg.get(PT_TSNS).name, "TSNS");
        assert_eq!(reg.get(PT_CBNW).name, "BUBW");
        assert_eq!(reg.get(PT_BIZRS).category, MenuCategory::Solids);
        assert_eq!(reg.get(PT_PLUT).name, "PLUT");
        assert_eq!(reg.get(PT_CO2).name, "CO2");
        assert_eq!(reg.get(PT_SEED).name, "SEED");
        assert_eq!(reg.get(BUILTIN_ELEMENT_COUNT as u32 - 1).name, "SEED");
        assert_eq!(reg.get(PT_CO2).category, MenuCategory::Gases);
        assert_eq!(reg.get(PT_PSTE).category, MenuCategory::Liquids);
        assert_eq!(reg.get(PT_THRM).category, MenuCategory::Explosives);
    }

    #[test]
    fn default_graphics_matches_tpt_property_defaults() {
        let reg = ElementRegistry::new_default();
        // `Element.cpp:236`: radioactive elements glow.
        assert_ne!(reg.get(PT_PLUT).pixel_mode & PMODE_GLOW, 0);
        assert_ne!(reg.get(PT_URAN).pixel_mode & PMODE_GLOW, 0);
        // `Element.cpp:238-241`: liquids blur.
        assert_eq!(reg.get(PT_WATR).pixel_mode, PMODE_FLAT | PMODE_BLUR);
        // `Element.cpp:243-249`: gases drop the pixel mode for a fire blend
        // with a half-brightness fire colour.
        assert_eq!(reg.get(PT_H2).pixel_mode, FIRE_BLEND | DECO_FIRE);
        assert_eq!(reg.get(PT_H2).fire_colour, 0x7D28_387F);
        assert_eq!(reg.get(PT_O2).pixel_mode, FIRE_BLEND | DECO_FIRE);
        // Elements with their own `Graphics` never run the defaults, even when
        // the function only touches the colour (`AMTR`, `CBNW`) or is null
        // (`DUST`).
        assert_eq!(reg.get(PT_AMTR).pixel_mode, PMODE_FLAT);
        assert_eq!(reg.get(PT_CBNW).pixel_mode, PMODE_FLAT);
        assert_eq!(reg.get(PT_DUST).pixel_mode, PMODE_FLAT);
        assert_eq!(reg.get(PT_PLUT).fire_colour, 0);
    }

    #[test]
    fn graphics_override_table_matches_the_cpp_table() {
        // `Element.cpp:49` installs `defaultGraphics`; `grep Graphics =` over
        // `src/simulation/elements/*.cpp` finds 90 assignments.
        let overrides = (0..PT_NUM as u32)
            .filter(|&id| has_graphics_override(id))
            .count();
        assert_eq!(overrides, 90);
        assert!(!has_graphics_override(PT_NONE));
    }

    #[test]
    fn builtin_ids_are_unique_and_dense() {
        let reg = ElementRegistry::new_default();
        // The table is emitted as a dense sequence: slot `i` holds element `i`.
        for (slot, def) in reg.elements.iter().take(BUILTIN_ELEMENT_COUNT).enumerate() {
            assert_eq!(
                def.id as usize, slot,
                "slot {slot} holds element {}",
                def.id
            );
        }
        // Every name and identifier is unique across the built-in table.
        for (i, a) in reg.elements.iter().enumerate().take(BUILTIN_ELEMENT_COUNT) {
            for b in reg
                .elements
                .iter()
                .skip(i + 1)
                .take(BUILTIN_ELEMENT_COUNT - i - 1)
            {
                assert_ne!(a.name, b.name, "duplicate element name {}", a.name);
                assert_ne!(
                    a.identifier, b.identifier,
                    "duplicate element identifier {}",
                    a.identifier
                );
            }
        }
    }

    /// TPT `ElementNumbers.h` identifiers indexed by TPT id (`195` entries).
    /// Slot 146 is the removed `disabler()` placeholder.
    const TPT_ELEMENT_IDENTIFIERS: [&str; 196] = [
        "DEFAULT_PT_NONE",   // 0
        "DEFAULT_PT_DUST",   // 1
        "DEFAULT_PT_WATR",   // 2
        "DEFAULT_PT_OIL",    // 3
        "DEFAULT_PT_FIRE",   // 4
        "DEFAULT_PT_STNE",   // 5
        "DEFAULT_PT_LAVA",   // 6
        "DEFAULT_PT_GUNP",   // 7
        "DEFAULT_PT_NITR",   // 8
        "DEFAULT_PT_CLNE",   // 9
        "DEFAULT_PT_GAS",    // 10
        "DEFAULT_PT_PLEX",   // 11
        "DEFAULT_PT_GOO",    // 12
        "DEFAULT_PT_ICEI",   // 13
        "DEFAULT_PT_METL",   // 14
        "DEFAULT_PT_SPRK",   // 15
        "DEFAULT_PT_SNOW",   // 16
        "DEFAULT_PT_WOOD",   // 17
        "DEFAULT_PT_NEUT",   // 18
        "DEFAULT_PT_PLUT",   // 19
        "DEFAULT_PT_PLNT",   // 20
        "DEFAULT_PT_ACID",   // 21
        "DEFAULT_PT_VOID",   // 22
        "DEFAULT_PT_WTRV",   // 23
        "DEFAULT_PT_CNCT",   // 24
        "DEFAULT_PT_DSTW",   // 25
        "DEFAULT_PT_SALT",   // 26
        "DEFAULT_PT_SLTW",   // 27
        "DEFAULT_PT_DMND",   // 28
        "DEFAULT_PT_BMTL",   // 29
        "DEFAULT_PT_BRMT",   // 30
        "DEFAULT_PT_PHOT",   // 31
        "DEFAULT_PT_URAN",   // 32
        "DEFAULT_PT_WAX",    // 33
        "DEFAULT_PT_MWAX",   // 34
        "DEFAULT_PT_PSCN",   // 35
        "DEFAULT_PT_NSCN",   // 36
        "DEFAULT_PT_LNTG",   // 37
        "DEFAULT_PT_INSL",   // 38
        "DEFAULT_PT_BHOL",   // 39
        "DEFAULT_PT_WHOL",   // 40
        "DEFAULT_PT_RBDM",   // 41
        "DEFAULT_PT_LRBD",   // 42
        "DEFAULT_PT_NTCT",   // 43
        "DEFAULT_PT_SAND",   // 44
        "DEFAULT_PT_GLAS",   // 45
        "DEFAULT_PT_PTCT",   // 46
        "DEFAULT_PT_BGLA",   // 47
        "DEFAULT_PT_THDR",   // 48
        "DEFAULT_PT_PLSM",   // 49
        "DEFAULT_PT_ETRD",   // 50
        "DEFAULT_PT_NICE",   // 51
        "DEFAULT_PT_NBLE",   // 52
        "DEFAULT_PT_BTRY",   // 53
        "DEFAULT_PT_LCRY",   // 54
        "DEFAULT_PT_STKM",   // 55
        "DEFAULT_PT_SWCH",   // 56
        "DEFAULT_PT_SMKE",   // 57
        "DEFAULT_PT_DESL",   // 58
        "DEFAULT_PT_COAL",   // 59
        "DEFAULT_PT_LO2",    // 60
        "DEFAULT_PT_O2",     // 61
        "DEFAULT_PT_INWR",   // 62
        "DEFAULT_PT_YEST",   // 63
        "DEFAULT_PT_DYST",   // 64
        "DEFAULT_PT_THRM",   // 65
        "DEFAULT_PT_GLOW",   // 66
        "DEFAULT_PT_BRCK",   // 67
        "DEFAULT_PT_HFLM",   // 68
        "DEFAULT_PT_FIRW",   // 69
        "DEFAULT_PT_FUSE",   // 70
        "DEFAULT_PT_FSEP",   // 71
        "DEFAULT_PT_AMTR",   // 72
        "DEFAULT_PT_BCOL",   // 73
        "DEFAULT_PT_PCLN",   // 74
        "DEFAULT_PT_HSWC",   // 75
        "DEFAULT_PT_IRON",   // 76
        "DEFAULT_PT_MORT",   // 77
        "DEFAULT_PT_LIFE",   // 78
        "DEFAULT_PT_DLAY",   // 79
        "DEFAULT_PT_CO2",    // 80
        "DEFAULT_PT_DRIC",   // 81
        "DEFAULT_PT_CBNW",   // 82
        "DEFAULT_PT_STOR",   // 83
        "DEFAULT_PT_PVOD",   // 84
        "DEFAULT_PT_CONV",   // 85
        "DEFAULT_PT_CAUS",   // 86
        "DEFAULT_PT_LIGH",   // 87
        "DEFAULT_PT_TESC",   // 88
        "DEFAULT_PT_DEST",   // 89
        "DEFAULT_PT_SPNG",   // 90
        "DEFAULT_PT_RIME",   // 91
        "DEFAULT_PT_FOG",    // 92
        "DEFAULT_PT_BCLN",   // 93
        "DEFAULT_PT_LOVE",   // 94
        "DEFAULT_PT_DEUT",   // 95
        "DEFAULT_PT_WARP",   // 96
        "DEFAULT_PT_PUMP",   // 97
        "DEFAULT_PT_FWRK",   // 98
        "DEFAULT_PT_PIPE",   // 99
        "DEFAULT_PT_FRZZ",   // 100
        "DEFAULT_PT_FRZW",   // 101
        "DEFAULT_PT_GRAV",   // 102
        "DEFAULT_PT_BIZR",   // 103
        "DEFAULT_PT_BIZRG",  // 104
        "DEFAULT_PT_BIZRS",  // 105
        "DEFAULT_PT_INST",   // 106
        "DEFAULT_PT_ISOZ",   // 107
        "DEFAULT_PT_ISZS",   // 108
        "DEFAULT_PT_PRTI",   // 109
        "DEFAULT_PT_PRTO",   // 110
        "DEFAULT_PT_PSTE",   // 111
        "DEFAULT_PT_PSTS",   // 112
        "DEFAULT_PT_ANAR",   // 113
        "DEFAULT_PT_VINE",   // 114
        "DEFAULT_PT_INVIS",  // 115
        "DEFAULT_PT_116",    // 116
        "DEFAULT_PT_SPAWN2", // 117
        "DEFAULT_PT_SPAWN",  // 118
        "DEFAULT_PT_SHLD1",  // 119
        "DEFAULT_PT_SHLD2",  // 120
        "DEFAULT_PT_SHLD3",  // 121
        "DEFAULT_PT_SHLD4",  // 122
        "DEFAULT_PT_LOLZ",   // 123
        "DEFAULT_PT_WIFI",   // 124
        "DEFAULT_PT_FILT",   // 125
        "DEFAULT_PT_ARAY",   // 126
        "DEFAULT_PT_BRAY",   // 127
        "DEFAULT_PT_STKM2",  // 128
        "DEFAULT_PT_BOMB",   // 129
        "DEFAULT_PT_C5",     // 130
        "DEFAULT_PT_SING",   // 131
        "DEFAULT_PT_QRTZ",   // 132
        "DEFAULT_PT_PQRT",   // 133
        "DEFAULT_PT_EMP",    // 134
        "DEFAULT_PT_BREC",   // 135
        "DEFAULT_PT_ELEC",   // 136
        "DEFAULT_PT_ACEL",   // 137
        "DEFAULT_PT_DCEL",   // 138
        "DEFAULT_PT_BANG",   // 139
        "DEFAULT_PT_IGNT",   // 140
        "DEFAULT_PT_BOYL",   // 141
        "DEFAULT_PT_GEL",    // 142
        "DEFAULT_PT_TRON",   // 143
        "DEFAULT_PT_TTAN",   // 144
        "DEFAULT_PT_EXOT",   // 145
        "",                  // 146: reserved
        "DEFAULT_PT_EMBR",   // 147
        "DEFAULT_PT_H2",     // 148
        "DEFAULT_PT_SOAP",   // 149
        "DEFAULT_PT_NBHL",   // 150
        "DEFAULT_PT_NWHL",   // 151
        "DEFAULT_PT_MERC",   // 152
        "DEFAULT_PT_PBCN",   // 153
        "DEFAULT_PT_GPMP",   // 154
        "DEFAULT_PT_CLST",   // 155
        "DEFAULT_PT_WIRE",   // 156
        "DEFAULT_PT_GBMB",   // 157
        "DEFAULT_PT_FIGH",   // 158
        "DEFAULT_PT_FRAY",   // 159
        "DEFAULT_PT_RPEL",   // 160
        "DEFAULT_PT_PPIP",   // 161
        "DEFAULT_PT_DTEC",   // 162
        "DEFAULT_PT_DMG",    // 163
        "DEFAULT_PT_TSNS",   // 164
        "DEFAULT_PT_VIBR",   // 165
        "DEFAULT_PT_BVBR",   // 166
        "DEFAULT_PT_CRAY",   // 167
        "DEFAULT_PT_PSTN",   // 168
        "DEFAULT_PT_FRME",   // 169
        "DEFAULT_PT_GOLD",   // 170
        "DEFAULT_PT_TUNG",   // 171
        "DEFAULT_PT_PSNS",   // 172
        "DEFAULT_PT_PROT",   // 173
        "DEFAULT_PT_VIRS",   // 174
        "DEFAULT_PT_VRSS",   // 175
        "DEFAULT_PT_VRSG",   // 176
        "DEFAULT_PT_GRVT",   // 177
        "DEFAULT_PT_DRAY",   // 178
        "DEFAULT_PT_CRMC",   // 179
        "DEFAULT_PT_HEAC",   // 180
        "DEFAULT_PT_SAWD",   // 181
        "DEFAULT_PT_POLO",   // 182
        "DEFAULT_PT_RFRG",   // 183
        "DEFAULT_PT_RFGL",   // 184
        "DEFAULT_PT_LSNS",   // 185
        "DEFAULT_PT_LDTC",   // 186
        "DEFAULT_PT_SLCN",   // 187
        "DEFAULT_PT_PTNM",   // 188
        "DEFAULT_PT_VSNS",   // 189
        "DEFAULT_PT_ROCK",   // 190
        "DEFAULT_PT_LITH",   // 191
        "DEFAULT_PT_RSST",   // 192
        "DEFAULT_PT_RSSS",   // 193
        "DEFAULT_PT_BASE",   // 194
        "DEFAULT_PT_SEED",   // 195
    ];

    /// Every TPT element id (`0..=195`, minus the removed `disabler()` slot
    /// 146) resolves to a definition, and no definition claims an unknown TPT
    /// identifier. This keeps the element menu a 1:1 superset of TPT's.
    #[test]
    fn every_tpt_element_id_has_a_definition() {
        let reg = ElementRegistry::new_default();
        for (tpt_id, identifier) in TPT_ELEMENT_IDENTIFIERS.iter().enumerate() {
            let identifier = *identifier;
            if identifier.is_empty() {
                assert_eq!(tpt_id, 146, "only the disabler slot may be empty");
                continue;
            }
            let id = reg
                .find_by_name(identifier)
                .unwrap_or_else(|| panic!("TPT id {tpt_id} ({identifier}) is missing"));
            assert!(
                id < BUILTIN_ELEMENT_COUNT as u32,
                "TPT id {tpt_id} ({identifier}) is a custom slot"
            );
            assert_eq!(reg.get(id).identifier, identifier);
            if tpt_id != 0 {
                assert!(reg.get(id).enabled, "TPT id {tpt_id} is disabled");
            }
        }
        // Every built-in identifier is one of TPT's.
        for def in reg.elements.iter().take(BUILTIN_ELEMENT_COUNT) {
            assert!(
                TPT_ELEMENT_IDENTIFIERS.contains(&def.identifier.as_str()),
                "unexpected identifier {} at slot {}",
                def.identifier,
                def.id
            );
        }
    }

    #[test]
    fn displacement_rules_match_tpt() {
        let reg = ElementRegistry::new_default();

        // `SimulationData::init_can_move` (`SimulationData.cpp:100-230`).
        assert_eq!(reg.can_move_val(PT_NONE, PT_DUST), 0);
        assert_eq!(reg.can_move_val(PT_DUST, PT_NONE), 1);
        assert_eq!(reg.can_move_val(PT_SAND, PT_WATR), 1);
        assert_eq!(reg.can_move_val(PT_WATR, PT_SAND), 0);
        assert_eq!(reg.can_move_val(PT_DUST, PT_DUST), 0);
        assert_eq!(reg.can_move_val(PT_DUST, PT_GEL), 0);

        // Photons pass through `PROP_PHOTPASS` only; everything else blocks.
        assert_eq!(reg.can_move_val(PT_PHOT, PT_NONE), 1);
        assert_eq!(reg.can_move_val(PT_PHOT, PT_GLAS), 2);
        assert_eq!(reg.can_move_val(PT_PHOT, PT_DUST), 0);
        assert_eq!(reg.can_move_val(PT_PHOT, PT_GOLD), 0);
        assert_eq!(reg.can_move_val(PT_PHOT, PT_BMTL), 0);
        assert_eq!(reg.can_move_val(PT_PHOT, PT_LCRY), 3);
        assert_eq!(reg.can_move_val(PT_PHOT, PT_GPMP), 3);

        // Neutrons pass through `PROP_NEUTPASS` and are absorbed by matter.
        assert_eq!(reg.can_move_val(PT_NEUT, PT_GLAS), 2);
        assert_eq!(reg.can_move_val(PT_NEUT, PT_GOLD), 2);

        // Energy shares the cell with energy.
        assert_eq!(reg.can_move_val(PT_ELEC, PT_PHOT), 2);
        assert_eq!(reg.can_move_val(PT_ELEC, PT_VIBR), 1);
        assert_eq!(reg.can_move_val(PT_DUST, PT_VIBR), 0);

        // Runtime-decided entries.
        assert_eq!(reg.can_move_val(PT_DUST, PT_VOID), 3);
        assert_eq!(reg.can_move_val(PT_DUST, PT_PVOD), 3);
        assert_eq!(reg.can_move_val(PT_DUST, PT_INVIS), 3);
        assert_eq!(reg.can_move_val(PT_TRON, PT_SWCH), 3);
        assert_eq!(reg.can_move_val(PT_NEUT, PT_INVIS), 2);

        // Shared overrides.
        assert_eq!(reg.can_move_val(PT_DUST, PT_CNCT), 0);
        assert_eq!(reg.can_move_val(PT_DUST, PT_BHOL), 1);
        assert_eq!(reg.can_move_val(PT_DUST, PT_NBHL), 1);
        assert_eq!(reg.can_move_val(PT_DUST, PT_SAWD), 0);
        assert_eq!(reg.can_move_val(PT_EMBR, PT_DUST), 0);
        assert_eq!(reg.can_move_val(PT_EMBR, PT_EMBR), 2);
        assert_eq!(reg.can_move_val(PT_PROT, PT_DUST), 2);
        assert_eq!(reg.can_move_val(PT_GRVT, PT_VOID), 3);
        assert_eq!(reg.can_move_val(PT_GRVT, PT_INSL), 0);
        assert_eq!(reg.can_move_val(PT_ANAR, PT_WHOL), 1);
        assert_eq!(reg.can_move_val(PT_SPRK, PT_NONE), 0);
        assert_eq!(reg.can_move_val(PT_STKM, PT_WATR), 2);
    }

    #[test]
    fn gpu_tables_have_fixed_stride() {
        let reg = ElementRegistry::new_default();
        let props = reg.to_gpu_props();
        assert_eq!(props.len(), PT_NUM);
        assert_eq!(props[PT_DUST as usize].weight, 85);
        assert_eq!(props[PT_WATR as usize].heat_conduct, 29);
        assert_eq!(
            std::mem::size_of::<GpuElementProps>(),
            32 * std::mem::size_of::<u32>()
        );

        let can_move = reg.to_gpu_can_move_u32();
        assert_eq!(can_move.len(), PT_NUM * PT_NUM);
        assert_eq!(can_move[PT_DUST as usize * PT_NUM + PT_NONE as usize], 1);
    }

    #[test]
    fn custom_elements_take_free_slots() {
        let mut reg = ElementRegistry::new_default();
        // "GOO" is a built-in name since batch 1, so a custom element must not
        // shadow it in `find_by_name`; use a name no built-in claims.
        let id = reg.allocate_custom("test", "glorp").expect("free slot");
        assert!(id as usize >= BUILTIN_ELEMENT_COUNT);
        assert_eq!(reg.get(id).identifier, "TEST_PT_GLORP");
        assert_eq!(reg.get(id).name, "GLORP");
        assert_eq!(reg.find_by_name("test_pt_glorp"), Some(id));
        assert_eq!(reg.find_by_name("GLORP"), Some(id));
        assert_eq!(reg.find_by_name("GOO"), Some(PT_GOO));
        assert_ne!(id, PT_GOO);

        assert!(reg.allocate_custom("TEST", "GLORP").is_none());
        assert!(reg.allocate_custom("DEFAULT", "BAD").is_none());
        assert!(reg.allocate_custom("BAD_GROUP", "X").is_none());
    }
}
