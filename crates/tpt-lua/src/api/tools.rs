//! The `tools` table (`LuaTools.cpp:1-431`).
//!
//! TPT's tool list is a `std::vector<std::unique_ptr<Tool>>` owned by
//! `GameModel`; scripts index it by number and look identifiers up in
//! `tools.index`. This port keeps the same registry in Lua-side state: element,
//! wall, `SimTool`, decoration and UI tools are seeded from
//! `tpt_core::elements::ElementRegistry` / `tpt_core::walls::default_walls`, and
//! `tools.allocate` appends custom tools. The application palette is
//! element/wall based, so the seeded entries are what scripts can address; the
//! custom-tool callbacks (`Perform`, `Draw`, ...) are stored and
//! [`call_tool_callback`] is the host entry point that fires them.

use std::cell::{Ref, RefCell, RefMut};
use std::rc::Rc;

use mlua::{Function, Lua, MultiValue, RegistryKey, Table, Value};
use tpt_core::elements::{ElementRegistry, MenuCategory};
use tpt_core::walls::default_walls;

use crate::api::{is_truthy, value_to_i64, value_to_string};
use crate::bridge::Bridge;

/// `NUM_TOOLINDICES` (`GameModel.h:16`): primary, secondary, decoration and
/// the fourth tool slot.
const NUM_TOOLINDICES: usize = 4;

/// `SC_TOOL` (`simulation/MenuSection.h:25`).
const MENU_TOOL: i32 = 13;

/// Tool fields a custom script can change through `tools.property`
/// (`LuaTools.cpp:256-285`).
#[derive(Default)]
struct ToolCallbacks {
    perform: Option<RegistryKey>,
    click: Option<RegistryKey>,
    drag: Option<RegistryKey>,
    draw: Option<RegistryKey>,
    draw_line: Option<RegistryKey>,
    draw_rect: Option<RegistryKey>,
    draw_fill: Option<RegistryKey>,
    select: Option<RegistryKey>,
}

/// One entry of the tool list. Mirrors the `Tool` fields the Lua API exposes
/// (`gui/game/tool/Tool.h:22-31`).
pub struct Tool {
    /// `Tool::Identifier`, e.g. `DEFAULT_TOOL_HEAT` or `MYGROUP_TOOL_NAME`.
    pub identifier: String,
    /// `Tool::Name`, shown in the menu.
    pub name: String,
    /// `Tool::Description`, shown in the tooltip.
    pub description: String,
    /// `Tool::Colour`, packed `0x00RRGGBB`.
    pub colour: u32,
    /// `Tool::MenuSection`, an `SC_*` index.
    pub menu_section: i32,
    /// `Tool::MenuVisible`.
    pub menu_visible: bool,
    /// `Tool::MenuSort`.
    pub menu_sort: i32,
    /// `Tool::Strength`.
    pub strength: f32,
    /// True when the tool came from `tools.allocate`.
    pub custom: bool,
    /// Lua callbacks installed by `tools.property`.
    callbacks: ToolCallbacks,
}

impl Tool {
    fn new(identifier: impl Into<String>) -> Self {
        Self {
            identifier: identifier.into(),
            name: String::from("NULL"),
            description: String::from("NULL Tool, does NOTHING"),
            colour: 0x00FF_FFFF,
            menu_section: MENU_TOOL,
            menu_visible: true,
            menu_sort: 0,
            strength: 1.0,
            custom: false,
            callbacks: ToolCallbacks::default(),
        }
    }

    fn with_metadata(
        identifier: impl Into<String>,
        name: impl Into<String>,
        description: impl Into<String>,
        colour: u32,
        menu_section: i32,
    ) -> Self {
        Self {
            identifier: identifier.into(),
            name: name.into(),
            description: description.into(),
            colour: colour & 0x00FF_FFFF,
            menu_section,
            menu_visible: true,
            menu_sort: 0,
            strength: 1.0,
            custom: false,
            callbacks: ToolCallbacks::default(),
        }
    }
}

/// Maps a `tpt_core` element category onto TPT's `SC_*` menu index
/// (`simulation/MenuSection.h:12-27`). The two enums order their variants
/// differently, so the mapping is explicit.
fn menu_section(category: MenuCategory) -> i32 {
    match category {
        MenuCategory::Walls => 0,
        MenuCategory::Electronics => 1,
        MenuCategory::Powered => 2,
        MenuCategory::Sensors => 3,
        MenuCategory::Force => 4,
        MenuCategory::Explosives => 5,
        MenuCategory::Gases => 6,
        MenuCategory::Liquids => 7,
        MenuCategory::Powders => 8,
        MenuCategory::Solids => 9,
        MenuCategory::Radioactive => 10,
        MenuCategory::Special | MenuCategory::RigidBodies => 11,
        MenuCategory::Life => 12,
        MenuCategory::Tools => 13,
        MenuCategory::Favorites => 14,
        MenuCategory::Deco => 15,
    }
}

/// Tool registry plus the four active-tool slots (`GameModel::activeTools`,
/// `GameModel.h:91-92`).
pub struct ToolState {
    /// Indexed like `GameModel::tools`: `None` is a freed slot.
    pub tools: Vec<Option<Tool>>,
    /// `GameModel::activeTools`: identifier per slot.
    pub active: Vec<Option<String>>,
}

impl Default for ToolState {
    fn default() -> Self {
        Self::seeded()
    }
}

impl ToolState {
    /// Builds the default tool list (`GameModel::InitTools`,
    /// `GameModel.cpp:1888-1933`).
    fn seeded() -> Self {
        let mut tools: Vec<Option<Tool>> = Vec::new();

        // Element tools, in element-id order (`GameModel.cpp:1893-1899`).
        let registry = ElementRegistry::new_default();
        for def in &registry.elements {
            if !def.enabled {
                continue;
            }
            tools.push(Some(Tool::with_metadata(
                def.identifier.clone(),
                def.name.clone(),
                def.description.clone(),
                def.colour,
                menu_section(def.category),
            )));
        }

        // Wall tools (`GameModel.cpp:1904-1909`).
        for wall in default_walls() {
            tools.push(Some(Tool::with_metadata(
                wall.identifier.clone(),
                wall.name.clone(),
                wall.description.clone(),
                wall.colour,
                0,
            )));
        }

        // `GetTools()`: the eleven `SimTool`s (`simulation/simtools/*.cpp`).
        for (identifier, name, colour, description) in [
            ("DEFAULT_TOOL_HEAT", "HEAT", 0x00FF_DD00, "Heats the targeted element."),
            ("DEFAULT_TOOL_COOL", "COOL", 0x0000_DDFF, "Cools the targeted element."),
            ("DEFAULT_TOOL_AIR", "AIR", 0x00FF_FFFF, "Air, creates airflow and pressure."),
            ("DEFAULT_TOOL_VAC", "VAC", 0x0030_3030, "Vacuum, reduces air pressure."),
            ("DEFAULT_TOOL_PGRV", "PGRV", 0x00CC_CCFF, "Creates a short-lasting gravity well."),
            (
                "DEFAULT_TOOL_NGRV",
                "NGRV",
                0x00AA_CCFF,
                "Creates a short-lasting negative gravity well.",
            ),
            ("DEFAULT_TOOL_MIX", "MIX", 0x00FF_D090, "Mixes particles."),
            ("DEFAULT_TOOL_CYCL", "CYCL", 0x0013_2F5B, "Cyclone, produces swirling air currents"),
            ("DEFAULT_TOOL_AMBM", "AMBM", 0x0000_DDFF, "Decreases ambient air temperature."),
            ("DEFAULT_TOOL_AMBP", "AMBP", 0x00FF_DD00, "Increases ambient air temperature."),
            ("DEFAULT_TOOL_WIND", "WIND", 0x0040_4040, "Creates air movement."),
        ] {
            tools.push(Some(Tool::with_metadata(
                identifier,
                name,
                description,
                colour,
                MENU_TOOL,
            )));
        }

        // Decoration tools (`GameModel.cpp:1916-1922`).
        for (identifier, name, description) in [
            ("DEFAULT_DECOR_ADD", "ADD", "Colour blending: Add."),
            ("DEFAULT_DECOR_SUB", "SUB", "Colour blending: Subtract."),
            ("DEFAULT_DECOR_MUL", "MUL", "Colour blending: Multiply."),
            ("DEFAULT_DECOR_DIV", "DIV", "Colour blending: Divide."),
            ("DEFAULT_DECOR_SMDG", "SMDG", "Smudge tool, blends surrounding deco together."),
            ("DEFAULT_DECOR_CLR", "CLR", "Erase any set decoration."),
            ("DEFAULT_DECOR_SET", "SET", "Draw decoration (No blending)."),
        ] {
            tools.push(Some(Tool::with_metadata(
                identifier,
                name,
                description,
                0x0000_0000,
                15,
            )));
        }

        // UI tools (`GameModel.cpp:1923-1926`).
        for (identifier, name) in [
            ("DEFAULT_UI_PROPERTY", "PROP"),
            ("DEFAULT_UI_SIGN", "SIGN"),
            ("DEFAULT_UI_SAMPLE", "SMPL"),
            ("DEFAULT_UI_ADDLIFE", "LIFE"),
        ] {
            tools.push(Some(Tool::with_metadata(
                identifier,
                name,
                "UI tool.",
                0x00FE_A900,
                MENU_TOOL,
            )));
        }

        // `SanitizeToolsets` (`GameModel.cpp:1740-1770`) guarantees every
        // active slot holds a tool. The application owns the real selection;
        // the mirror starts on the first element tool.
        let first = tools
            .first()
            .and_then(|slot| slot.as_ref())
            .map(|tool| tool.identifier.clone());
        Self {
            tools,
            active: vec![first; NUM_TOOLINDICES],
        }
    }

    /// `GameModel::GetToolByIndex` (`GameModel.cpp:1785-1790`).
    pub fn get(&self, index: usize) -> Option<&Tool> {
        self.tools.get(index)?.as_ref()
    }

    /// `GameModel::GetToolFromIdentifier`.
    pub fn find_identifier(&self, identifier: &str) -> Option<usize> {
        self.tools
            .iter()
            .position(|tool| tool.as_ref().is_some_and(|t| t.identifier == identifier))
    }

    /// Appends a custom tool (`GameModel::AllocTool`,
    /// `GameModel.cpp:1790-1808`) and returns its index.
    pub fn allocate(&mut self, identifier: String) -> usize {
        let mut tool = Tool::new(identifier);
        tool.custom = true;
        self.tools.push(Some(tool));
        self.tools.len() - 1
    }

    /// Frees a custom tool slot (`GameModel::FreeTool`,
    /// `GameModel.cpp:1810-1821`).
    pub fn free(&mut self, index: usize) {
        if let Some(slot) = self.tools.get_mut(index) {
            *slot = None;
        }
    }
}

/// Shared handle to the tool registry.
#[derive(Clone)]
pub struct ToolHandle(Rc<RefCell<ToolState>>);

impl ToolHandle {
    /// Borrows the registry.
    pub fn borrow(&self) -> Ref<'_, ToolState> {
        self.0.borrow()
    }

    /// Borrows the registry mutably.
    pub fn borrow_mut(&self) -> RefMut<'_, ToolState> {
        self.0.borrow_mut()
    }
}

impl crate::LuaScriptingEngine {
    /// Returns the `tools` registry owned by this Lua state.
    ///
    /// The application can use it to draw the tool palette and to dispatch the
    /// custom-tool callbacks stored by `tools.property`.
    pub fn tools(&self) -> ToolHandle {
        self.lua
            .app_data_ref::<ToolHandle>()
            .expect("the tools table is installed with the Lua state")
            .clone()
    }
}

/// Reads the registry handle from the Lua app data.
pub(crate) fn tools_handle(lua: &Lua) -> mlua::Result<ToolHandle> {
    lua.app_data_ref::<ToolHandle>()
        .map(|handle| handle.clone())
        .ok_or_else(|| mlua::Error::runtime("the tools table is not installed"))
}

/// Reads a callback argument: `nil` or `false` clears it, a function stores it,
/// anything else leaves the callback untouched (`LuaTools.cpp:255-275`).
fn assign_callback(lua: &Lua, slot: &mut Option<RegistryKey>, value: &Value) -> mlua::Result<()> {
    match value {
        Value::Nil | Value::Boolean(false) => {
            *slot = None;
        }
        Value::Function(function) => {
            *slot = Some(lua.create_registry_value(function.clone())?);
        }
        _ => {}
    }
    Ok(())
}

/// `tools.allocate(group, name)` (`LuaTools.cpp:9-46`).
fn allocate(lua: &Lua, args: (String, String)) -> mlua::Result<i64> {
    let handle = tools_handle(lua)?;
    let mut state = handle.borrow_mut();
    let group = args.0.to_uppercase();
    let name = args.1.to_uppercase();
    if name.contains('_') {
        return Err(mlua::Error::runtime("The tool name may not contain '_'."));
    }
    if group.contains('_') {
        return Err(mlua::Error::runtime("The group name may not contain '_'."));
    }
    if group == "DEFAULT" {
        return Err(mlua::Error::runtime(
            "You cannot create tools in the 'DEFAULT' group.",
        ));
    }
    let identifier = format!("{group}_TOOL_{name}");
    if state.find_identifier(&identifier).is_some() {
        return Err(mlua::Error::runtime("Tool identifier already in use."));
    }
    let index = state.allocate(identifier.clone());
    drop(state);

    // `GameModel::BuildMenus` republishes every identifier after an allocation
    // (`LuaTools.cpp:38`); only the new key changes here.
    let tools: Table = lua.globals().get("tools")?;
    let index_table: Table = tools.get("index")?;
    index_table.set(identifier, index as i64)?;
    Ok(index as i64)
}

/// `tools.property(index, name [, value])` (`LuaTools.cpp:246-311`).
fn property(lua: &Lua, args: MultiValue) -> mlua::Result<MultiValue> {
    let handle = tools_handle(lua)?;
    let mut values = args.into_iter();
    let Some(index_value) = values.next() else {
        return Err(mlua::Error::runtime("number expected, got no value"));
    };
    let index = value_to_i64(&index_value)?.max(0) as usize;
    let Some(name_value) = values.next() else {
        return Err(mlua::Error::runtime("string expected, got no value"));
    };
    let name = value_to_string(&name_value)?;
    let new_value = values.next();
    let has_value = new_value.is_some();

    let mut state = handle.borrow_mut();
    let Some(tool) = state.tools.get(index).and_then(|slot| slot.as_ref()) else {
        return Err(mlua::Error::runtime("Invalid tool"));
    };
    let custom = tool.custom;
    if has_value && !custom {
        return Err(mlua::Error::runtime(
            "Can only change properties of custom tools",
        ));
    }

    // Callbacks (`LuaTools.cpp:256-285`).
    let callback = match name.as_str() {
        "Perform" => Some(0),
        "Click" => Some(1),
        "Drag" => Some(2),
        "Draw" => Some(3),
        "DrawLine" => Some(4),
        "DrawRect" => Some(5),
        "DrawFill" => Some(6),
        "Select" => Some(7),
        _ => None,
    };
    if let Some(callback) = callback {
        if let Some(value) = new_value.as_ref() {
            let entry = state.tools[index]
                .as_mut()
                .expect("the tool was validated above");
            let slot = match callback {
                0 => &mut entry.callbacks.perform,
                1 => &mut entry.callbacks.click,
                2 => &mut entry.callbacks.drag,
                3 => &mut entry.callbacks.draw,
                4 => &mut entry.callbacks.draw_line,
                5 => &mut entry.callbacks.draw_rect,
                6 => &mut entry.callbacks.draw_fill,
                _ => &mut entry.callbacks.select,
            };
            assign_callback(lua, slot, value)?;
            return Ok(MultiValue::new());
        }
        return Err(mlua::Error::runtime("Invalid tool property"));
    }

    let entry = state.tools[index]
        .as_mut()
        .expect("the tool was validated above");
    let tool: &mut Tool = entry;
    let mut result = MultiValue::new();
    match name.as_str() {
        "Name" => {
            if let Some(value) = new_value {
                tool.name = value_to_string(&value)?;
            } else {
                result.push_back(Value::String(lua.create_string(&tool.name)?));
            }
        }
        "Description" => {
            if let Some(value) = new_value {
                tool.description = value_to_string(&value)?;
            } else {
                result.push_back(Value::String(lua.create_string(&tool.description)?));
            }
        }
        "Colour" | "Color" => {
            if let Some(value) = new_value {
                tool.colour = value_to_i64(&value)? as u32 & 0x00FF_FFFF;
            } else {
                result.push_back(Value::Integer(tool.colour as i64));
            }
        }
        "MenuSection" => {
            if let Some(value) = new_value {
                tool.menu_section = value_to_i64(&value)? as i32;
            } else {
                result.push_back(Value::Integer(tool.menu_section as i64));
            }
        }
        "MenuVisible" => {
            if let Some(value) = new_value {
                tool.menu_visible = is_truthy(&value);
            } else {
                result.push_back(Value::Boolean(tool.menu_visible));
            }
        }
        "MenuSort" => {
            if let Some(value) = new_value {
                tool.menu_sort = value_to_i64(&value)? as i32;
            } else {
                result.push_back(Value::Integer(tool.menu_sort as i64));
            }
        }
        "Identifier" => {
            // The reference always returns the identifier, even when a value
            // is supplied (`LuaTools.cpp:306-310`).
            result.push_back(Value::String(lua.create_string(&tool.identifier)?));
        }
        _ => return Err(mlua::Error::runtime("Invalid tool property")),
    }
    Ok(result)
}

/// `tools.exists(index)` (`LuaTools.cpp:314-321`).
fn exists(lua: &Lua, index: Value) -> mlua::Result<bool> {
    let index = value_to_i64(&index)?.max(0) as usize;
    Ok(tools_handle(lua)?.borrow().get(index).is_some())
}

/// `tools.isCustom(index)` (`LuaTools.cpp:323-334`).
fn is_custom(lua: &Lua, index: Value) -> mlua::Result<bool> {
    let index = value_to_i64(&index)?.max(0) as usize;
    let handle = tools_handle(lua)?;
    let state = handle.borrow();
    match state.get(index) {
        Some(tool) => Ok(tool.custom),
        None => Err(mlua::Error::runtime("Invalid tool")),
    }
}

/// `tools.free(index)` (`LuaTools.cpp:48-66`).
fn free(lua: &Lua, index: Value) -> mlua::Result<()> {
    let index = value_to_i64(&index)?.max(0) as usize;
    let handle = tools_handle(lua)?;
    let mut state = handle.borrow_mut();
    let Some(tool) = state.get(index) else {
        return Err(mlua::Error::runtime("Invalid tool"));
    };
    if !tool.custom {
        return Err(mlua::Error::runtime("Can only free custom tools"));
    }
    let identifier = tool.identifier.clone();
    state.free(index);
    drop(state);

    let tools: Table = lua.globals().get("tools")?;
    let index_table: Table = tools.get("index")?;
    index_table.set(identifier, Value::Nil)?;
    Ok(())
}

/// `LuaTools::Open` (`LuaTools.cpp:336-360`).
pub(crate) fn build_tools_table(lua: &Lua, _bridge: &Bridge) -> mlua::Result<Table> {
    let handle = ToolHandle(Rc::new(RefCell::new(ToolState::seeded())));
    lua.set_app_data(handle.clone());

    let tools = lua.create_table()?;
    tools.set("allocate", lua.create_function(allocate)?)?;
    tools.set("property", lua.create_function(property)?)?;
    tools.set("exists", lua.create_function(exists)?)?;
    tools.set("isCustom", lua.create_function(is_custom)?)?;
    tools.set("free", lua.create_function(free)?)?;

    let index_table = lua.create_table()?;
    {
        let state = handle.borrow();
        for (index, tool) in state.tools.iter().enumerate() {
            if let Some(tool) = tool {
                index_table.set(tool.identifier.clone(), index as i64)?;
            }
        }
    }
    tools.set("index", index_table)?;
    Ok(tools)
}

/// Invokes a stored custom-tool callback, if the tool has one.
///
/// `LuaTools.cpp:68-245` wires each callback to a native wrapper; the port
/// exposes the same entry point to the host instead of a `SimTool` vtable.
pub fn call_tool_callback(
    lua: &Lua,
    index: usize,
    name: &str,
    args: MultiValue,
) -> mlua::Result<Option<Value>> {
    let handle = tools_handle(lua)?;
    let state = handle.borrow();
    let Some(entry) = state.tools.get(index).and_then(|slot| slot.as_ref()) else {
        return Err(mlua::Error::runtime("Invalid tool"));
    };
    let key = match name {
        "Perform" => &entry.callbacks.perform,
        "Click" => &entry.callbacks.click,
        "Drag" => &entry.callbacks.drag,
        "Draw" => &entry.callbacks.draw,
        "DrawLine" => &entry.callbacks.draw_line,
        "DrawRect" => &entry.callbacks.draw_rect,
        "DrawFill" => &entry.callbacks.draw_fill,
        "Select" => &entry.callbacks.select,
        _ => return Ok(None),
    };
    let Some(key) = key else {
        return Ok(None);
    };
    let function: Function = lua.registry_value(key)?;
    drop(state);
    function.call(args).map(Some)
}
