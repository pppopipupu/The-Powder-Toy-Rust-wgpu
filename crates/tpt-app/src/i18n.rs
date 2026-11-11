//! Localized UI strings.
//!
//! Every user-visible label lives in `crates/tpt-app/locales/<language>/<module>.json`
//! as a flat `{ "key": "text" }` object. Code asks for a dotted key
//! (`"top_bar.play"`) and receives the string for the active language. A key
//! that is missing falls back to English and then to the key itself, so a gap
//! shows up in the interface instead of an empty label.
//!
//! Adding a module: create `locales/en/<module>.json` and
//! `locales/zh-CN/<module>.json`, then add one row to [`MODULES`].

use std::cell::Cell;
use std::collections::HashMap;
use std::sync::LazyLock;

/// Interface language. The stored value is the enum's discriminant.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Language {
    #[default]
    En,
    ZhCn,
}

impl Language {
    /// Every language the interface offers, in menu order.
    pub const ALL: [Language; 2] = [Language::En, Language::ZhCn];

    /// Stable identifier used by the preferences file.
    pub fn code(self) -> &'static str {
        match self {
            Language::En => "en",
            Language::ZhCn => "zh-CN",
        }
    }

    /// Native name of the language, shown in the language picker untranslated.
    pub fn label(self) -> &'static str {
        match self {
            Language::En => "English",
            Language::ZhCn => "简体中文",
        }
    }

    /// Parses [`Language::code`]; an unknown code falls back to English.
    pub fn from_code(code: &str) -> Language {
        Language::ALL
            .into_iter()
            .find(|language| language.code() == code)
            .unwrap_or_default()
    }

    fn index(self) -> usize {
        match self {
            Language::En => 0,
            Language::ZhCn => 1,
        }
    }
}

struct Module {
    name: &'static str,
    en: &'static str,
    zh_cn: &'static str,
}

/// One row per locale module. The file stem is the key prefix.
const MODULES: &[Module] = &[
    Module {
        name: "top_bar",
        en: include_str!("../locales/en/top_bar.json"),
        zh_cn: include_str!("../locales/zh-CN/top_bar.json"),
    },
    Module {
        name: "menu",
        en: include_str!("../locales/en/menu.json"),
        zh_cn: include_str!("../locales/zh-CN/menu.json"),
    },
    Module {
        name: "side_bar",
        en: include_str!("../locales/en/side_bar.json"),
        zh_cn: include_str!("../locales/zh-CN/side_bar.json"),
    },
    Module {
        name: "bottom_bar",
        en: include_str!("../locales/en/bottom_bar.json"),
        zh_cn: include_str!("../locales/zh-CN/bottom_bar.json"),
    },
    Module {
        name: "console",
        en: include_str!("../locales/en/console.json"),
        zh_cn: include_str!("../locales/zh-CN/console.json"),
    },
    Module {
        name: "modal",
        en: include_str!("../locales/en/modal.json"),
        zh_cn: include_str!("../locales/zh-CN/modal.json"),
    },
    Module {
        name: "viewport",
        en: include_str!("../locales/en/viewport.json"),
        zh_cn: include_str!("../locales/zh-CN/viewport.json"),
    },
    Module {
        name: "options",
        en: include_str!("../locales/en/options.json"),
        zh_cn: include_str!("../locales/zh-CN/options.json"),
    },
    Module {
        name: "render",
        en: include_str!("../locales/en/render.json"),
        zh_cn: include_str!("../locales/zh-CN/render.json"),
    },
    Module {
        name: "property",
        en: include_str!("../locales/en/property.json"),
        zh_cn: include_str!("../locales/zh-CN/property.json"),
    },
    Module {
        name: "signs",
        en: include_str!("../locales/en/signs.json"),
        zh_cn: include_str!("../locales/zh-CN/signs.json"),
    },
    Module {
        name: "app",
        en: include_str!("../locales/en/app.json"),
        zh_cn: include_str!("../locales/zh-CN/app.json"),
    },
    Module {
        name: "tools",
        en: include_str!("../locales/en/tools.json"),
        zh_cn: include_str!("../locales/zh-CN/tools.json"),
    },
    // Element and wall descriptions copied from the reference element table
    // (`src/simulation/elements/*.cpp` `Description` assignments, surfaced as
    // tooltips by `GameModel.cpp:1851-1852`). Keys are the stable
    // `DEFAULT_PT_*` / `DEFAULT_WL_*` identifiers, so a Lua-defined element
    // simply has no key and falls back to its own `description`.
    Module {
        name: "elements",
        en: include_str!("../locales/en/elements.json"),
        zh_cn: include_str!("../locales/zh-CN/elements.json"),
    },
    Module {
        name: "walls",
        en: include_str!("../locales/en/walls.json"),
        zh_cn: include_str!("../locales/zh-CN/walls.json"),
    },
    Module {
        name: "voxel3d",
        en: include_str!("../locales/en/voxel3d.json"),
        zh_cn: include_str!("../locales/zh-CN/voxel3d.json"),
    },
];

/// Dotted key to text, one table per language, both flattened at first use.
type Table = HashMap<String, String>;

fn parse(module: &Module, source: &'static str) -> Table {
    let parsed: HashMap<String, String> = serde_json::from_str(source)
        .unwrap_or_else(|error| panic!("locale {}.json: {error}", module.name));
    parsed
        .into_iter()
        .map(|(key, text)| (format!("{}.{key}", module.name), text))
        .collect()
}

static TABLES: LazyLock<[Table; 2]> = LazyLock::new(|| {
    [
        MODULES.iter().flat_map(|m| parse(m, m.en)).collect(),
        MODULES.iter().flat_map(|m| parse(m, m.zh_cn)).collect(),
    ]
});

// The initialiser is already a `const` block; clippy 1.99 reports it anyway.
thread_local! {
    /// Interface language of the current thread.
    ///
    /// The app renders the whole UI on one thread, so the language is
    /// process-wide in practice. It is thread-local rather than a global
    /// atomic because the unit tests run in parallel and a test that switches
    /// language must not change what another test's `t()` returns mid-run.
    #[allow(clippy::missing_const_for_thread_local)]
    static LANGUAGE: Cell<u8> = const { Cell::new(Language::En as u8) };
}

/// Language the interface renders in.
pub fn language() -> Language {
    match LANGUAGE.with(Cell::get) {
        1 => Language::ZhCn,
        _ => Language::En,
    }
}

/// Switches the interface language; takes effect on the next frame.
pub fn set_language(language: Language) {
    LANGUAGE.with(|current| current.set(language as u8));
}

/// Text for `key` in the active language, or `None` when it is untranslated.
///
/// Falls back to English; unlike [`t`] a miss stays a miss so callers can
/// supply their own fallback (Lua-defined elements carry a scripted
/// description in `ElementDef`).
pub fn try_t(key: &str) -> Option<&'static str> {
    let tables: &'static [Table; 2] = &TABLES;
    tables[language().index()]
        .get(key)
        .or_else(|| tables[0].get(key))
        .map(String::as_str)
}

/// Text for `key` in the active language.
///
/// Falls back to English, then to `key` itself: an untranslated or mistyped key
/// is visible in the interface instead of silently blank.
pub fn t(key: &str) -> &str {
    match try_t(key) {
        Some(text) => text,
        None => key,
    }
}

/// [`t`] with `{0}`, `{1}`, ... replaced by `args`.
///
/// `format!` needs a literal format string, so parameterized labels go through
/// this helper instead of embedding `{}` in a `format!` call.
pub fn t_fmt(key: &str, args: &[&str]) -> String {
    let mut text = t(key).to_string();
    for (index, arg) in args.iter().enumerate() {
        text = text.replace(&format!("{{{index}}}"), arg);
    }
    text
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every key of every module must exist in every language, otherwise the
    /// interface silently mixes languages.
    #[test]
    fn every_language_defines_every_key() {
        let tables = &*TABLES;
        for key in tables[0].keys() {
            for (index, table) in tables.iter().enumerate() {
                assert!(
                    table.contains_key(key),
                    "language {} misses the key {key}",
                    Language::ALL[index].code()
                );
            }
        }
    }

    #[test]
    fn switching_language_changes_the_lookup() {
        let english = t("options.heat_simulation").to_string();
        set_language(Language::ZhCn);
        let chinese = t("options.heat_simulation").to_string();
        set_language(Language::En);
        assert_ne!(
            english, chinese,
            "both languages define options.heat_simulation"
        );
        assert_eq!(t("options.heat_simulation"), english);
    }

    #[test]
    fn missing_key_renders_as_the_key() {
        assert_eq!(
            t("no_such_module.no_such_key"),
            "no_such_module.no_such_key"
        );
    }

    /// Every built-in element and wall has a description in both languages.
    /// The registry and the wall table are the authoritative key lists; a new
    /// element without a translation must fail here rather than show an English
    /// string inside the Chinese interface.
    #[test]
    fn every_builtin_description_is_translated() {
        use tpt_core::elements::ElementRegistry;
        use tpt_core::walls::default_walls;

        let tables = &*TABLES;
        for definition in &ElementRegistry::new_default().elements {
            if !definition.identifier.starts_with("DEFAULT_PT_") {
                continue;
            }
            let key = format!("elements.{}", definition.identifier);
            for (index, table) in tables.iter().enumerate() {
                assert!(
                    table.contains_key(&key),
                    "language {} misses {key}",
                    Language::ALL[index].code()
                );
            }
        }
        for wall in default_walls() {
            let key = format!("walls.{}", wall.identifier);
            for (index, table) in tables.iter().enumerate() {
                assert!(
                    table.contains_key(&key),
                    "language {} misses {key}",
                    Language::ALL[index].code()
                );
            }
        }
    }

    /// The element descriptions are real translations, not the same English
    /// text copied into both files (except name-like jokes such as `LOLZ`).
    #[test]
    fn element_descriptions_are_localized() {
        let english = t("elements.DEFAULT_PT_DUST").to_string();
        set_language(Language::ZhCn);
        let chinese = t("elements.DEFAULT_PT_DUST").to_string();
        set_language(Language::En);
        assert_eq!(english, "Very light dust. Flammable.");
        assert_ne!(english, chinese);
        assert!(!chinese.is_ascii());
    }

    #[test]
    fn language_round_trips_through_its_code() {
        for language in Language::ALL {
            assert_eq!(Language::from_code(language.code()), language);
        }
        assert_eq!(Language::from_code("klingon"), Language::En);
    }
}
