//! The `interface` table (and the classic `ui` alias) plus the window and
//! component classes: `LuaInterface.cpp`, `LuaWindow.cpp`, `LuaComponent.cpp`,
//! `LuaButton.cpp`, `LuaLabel.cpp`, `LuaTextbox.cpp`, `LuaCheckbox.cpp`,
//! `LuaSlider.cpp` and `LuaProgressBar.cpp`.
//!
//! TPT's classes are Luna userdata that wrap live `ui::Window`/`ui::Component`
//! objects. This port keeps the same object model in [`InterfaceState`]: a list
//! of windows, a flat list of components addressed by id, the text-input grab,
//! the pending dialogs (`beginMessageBox`, `beginThrowError`, `beginConfirm`,
//! `beginInput`) and the mirrored `GameModel` settings the namespace reads and
//! writes. The application draws the window list and services the dialogs; the
//! dismissal helpers on [`InterfaceState`] invoke the Lua callbacks with the
//! same arguments the reference passes.

use std::cell::{Ref, RefCell, RefMut};
use std::rc::Rc;

use mlua::{AnyUserData, Function, Lua, MultiValue, RegistryKey, Table, Value};

use crate::api::{is_truthy, value_to_i64};

/// `NUM_TOOLINDICES` (`GameModel.h:16`).
pub const NUM_TOOLINDICES: usize = 4;

/// `GameController::mouseUpNormal` (`GameController.h:73`).
const MOUSEUP_NORMAL: i64 = 0;
/// `GameController::mouseUpBlur` (`GameController.h:74`).
const MOUSEUP_BLUR: i64 = 1;
/// `GameController::mouseUpDrawEnd` (`GameController.h:75`).
const MOUSEUP_DRAWEND: i64 = 2;

/// Menu sections (`SimulationData.cpp:76-97`).
const MENU_NAMES: [&str; 16] = [
    "Walls",
    "Electronics",
    "Powered Materials",
    "Sensors",
    "Force",
    "Explosives",
    "Gases",
    "Liquids",
    "Powders",
    "Solids",
    "Radioactive",
    "Special",
    "Game Of Life",
    "Tools",
    "Favorites",
    "Decoration tools",
];

/// The six component classes (`LuaComponent.h` subclasses).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ComponentKind {
    /// `ui::Button` (`LuaButton.cpp`).
    Button,
    /// `ui::Label` (`LuaLabel.cpp`).
    Label,
    /// `ui::Textbox` (`LuaTextbox.cpp`).
    Textbox,
    /// `ui::Checkbox` (`LuaCheckbox.cpp`).
    Checkbox,
    /// `ui::Slider` (`LuaSlider.cpp`).
    Slider,
    /// `ui::ProgressBar` (`LuaProgressBar.cpp`).
    ProgressBar,
}

/// One component attached to (or still detached from) a window.
pub struct Component {
    /// Unique component id. The application uses it to report input back to
    /// the right Lua object.
    pub id: u64,
    /// Which component class this is.
    pub kind: ComponentKind,
    /// `Component::Position`.
    pub position: (i32, i32),
    /// `Component::Size`.
    pub size: (i32, i32),
    /// `Component::Visible`.
    pub visible: bool,
    /// `Button::Enabled`.
    pub enabled: bool,
    /// `Button::GetText` / `Label::GetText` / `Textbox::GetText` /
    /// `Checkbox::GetText`.
    pub text: String,
    /// `Button` tooltip.
    pub tooltip: String,
    /// `Checkbox::GetChecked`.
    pub checked: bool,
    /// `Textbox::ReadOnly`.
    pub read_only: bool,
    /// `Slider::GetSteps`.
    pub steps: i32,
    /// `Slider::GetValue`.
    pub value: i32,
    /// `ProgressBar::GetProgress`.
    pub progress: i32,
    /// `ProgressBar::GetStatus`.
    pub status: String,
    /// `Button`/`Checkbox` action callback.
    pub action: Option<RegistryKey>,
    /// `Textbox` text-changed callback.
    pub text_changed: Option<RegistryKey>,
    /// `Slider` value-changed callback.
    pub value_changed: Option<RegistryKey>,
    /// The Lua userdata that owns this component, passed as the single
    /// argument to its callbacks (`LuaButton.cpp:70-83`). Set by
    /// `window.addComponent` (`LuaWindow.cpp:136-143`).
    pub owner: Option<AnyUserData>,
}

impl Component {
    fn new(id: u64, kind: ComponentKind, position: (i32, i32), size: (i32, i32)) -> Self {
        Self {
            id,
            kind,
            position,
            size,
            visible: true,
            enabled: true,
            text: String::new(),
            tooltip: String::new(),
            checked: false,
            read_only: false,
            steps: 10,
            value: 0,
            progress: 0,
            status: String::new(),
            action: None,
            text_changed: None,
            value_changed: None,
            owner: None,
        }
    }
}

/// Callbacks a script can attach to a window (`LuaWindow.cpp:18-36`).
#[derive(Default)]
pub struct WindowCallbacks {
    /// `onInitialized`.
    pub on_initialized: Option<RegistryKey>,
    /// `onExit`.
    pub on_exit: Option<RegistryKey>,
    /// `onTick`.
    pub on_tick: Option<RegistryKey>,
    /// `onDraw`.
    pub on_draw: Option<RegistryKey>,
    /// `onFocus`.
    pub on_focus: Option<RegistryKey>,
    /// `onBlur`.
    pub on_blur: Option<RegistryKey>,
    /// `onTryExit`.
    pub on_try_exit: Option<RegistryKey>,
    /// `onTryOkay`.
    pub on_try_okay: Option<RegistryKey>,
    /// `onMouseMove`.
    pub on_mouse_move: Option<RegistryKey>,
    /// `onMouseDown`.
    pub on_mouse_down: Option<RegistryKey>,
    /// `onMouseUp`.
    pub on_mouse_up: Option<RegistryKey>,
    /// `onMouseWheel`.
    pub on_mouse_wheel: Option<RegistryKey>,
    /// `onKeyPress`.
    pub on_key_press: Option<RegistryKey>,
    /// `onKeyRelease`.
    pub on_key_release: Option<RegistryKey>,
}

impl WindowCallbacks {
    /// Returns the callback registered for `name` (`"onDraw"`, ...).
    pub fn get(&self, name: &str) -> Option<&RegistryKey> {
        match name {
            "onInitialized" => self.on_initialized.as_ref(),
            "onExit" => self.on_exit.as_ref(),
            "onTick" => self.on_tick.as_ref(),
            "onDraw" => self.on_draw.as_ref(),
            "onFocus" => self.on_focus.as_ref(),
            "onBlur" => self.on_blur.as_ref(),
            "onTryExit" => self.on_try_exit.as_ref(),
            "onTryOkay" => self.on_try_okay.as_ref(),
            "onMouseMove" => self.on_mouse_move.as_ref(),
            "onMouseDown" => self.on_mouse_down.as_ref(),
            "onMouseUp" => self.on_mouse_up.as_ref(),
            "onMouseWheel" => self.on_mouse_wheel.as_ref(),
            "onKeyPress" => self.on_key_press.as_ref(),
            "onKeyRelease" => self.on_key_release.as_ref(),
            _ => None,
        }
    }
}

/// One script-created window (`LuaWindow::DrawnWindow`).
pub struct WindowState {
    /// Unique window id.
    pub id: u64,
    /// `Window::Position`.
    pub position: (i32, i32),
    /// `Window::Size`.
    pub size: (i32, i32),
    /// Whether the window is shown (`ui::Engine::ShowWindow`).
    pub visible: bool,
    /// `OnInitialized` has fired.
    pub initialized: bool,
    /// Components attached through `addComponent`, in insertion order.
    pub components: Vec<u64>,
    /// `Window::IsFocused` target (a textbox id).
    pub focused_component: Option<u64>,
    /// Script callbacks.
    pub callbacks: WindowCallbacks,
}

/// Pending `beginMessageBox` dialog (`LuaInterface.cpp:38-66`).
pub struct MessageBox {
    /// Dialog title.
    pub title: String,
    /// Dialog body.
    pub message: String,
    /// `InformationMessage`'s `large` flag.
    pub large: bool,
    /// Invoked with no arguments when the dialog closes.
    pub callback: Option<RegistryKey>,
}

/// Pending `beginThrowError` dialog (`LuaInterface.cpp:68-96`).
pub struct ErrorMessage {
    /// Error text.
    pub message: String,
    /// Invoked with no arguments when the dialog closes.
    pub callback: Option<RegistryKey>,
}

/// Pending `beginConfirm` dialog (`LuaInterface.cpp:134-171`).
pub struct ConfirmDialog {
    /// Dialog title.
    pub title: String,
    /// Dialog body.
    pub message: String,
    /// Confirm button text.
    pub button_text: String,
    /// Invoked with `true`/`false` (`LuaInterface.cpp:145-160`).
    pub callback: Option<RegistryKey>,
}

/// Pending `beginInput` prompt (`LuaInterface.cpp:98-132`).
pub struct InputPrompt {
    /// Dialog title.
    pub title: String,
    /// Prompt line.
    pub prompt: String,
    /// Initial text.
    pub text: String,
    /// Shadow text (`TextPrompt`'s placeholder).
    pub shadow: String,
    /// Invoked with the entered string, or `nil` when cancelled.
    pub callback: Option<RegistryKey>,
}

/// One menu section (`SimulationData::msections`).
pub struct MenuSection {
    /// Section name.
    pub name: String,
    /// `menu_section::doshow`.
    pub enabled: bool,
}

/// The `interface` namespace's backing state.
pub struct InterfaceState {
    /// Windows created by scripts, in creation order.
    pub windows: Vec<WindowState>,
    /// Every component ever created, addressed by [`Component::id`].
    pub components: Vec<Component>,
    /// `ui::Engine::GetWindow()`, restricted to script windows.
    pub shown_window: Option<u64>,
    /// `LuaScriptInterface::textInputRefcount`.
    pub text_input_refcount: i32,
    /// `ui::Engine::TextInputRect`.
    pub text_input_rect: Option<(i32, i32, i32, i32)>,
    /// Pending information dialog.
    pub message_box: Option<MessageBox>,
    /// Pending error dialog.
    pub error_message: Option<ErrorMessage>,
    /// Pending confirmation dialog.
    pub confirm: Option<ConfirmDialog>,
    /// Pending text prompt.
    pub input_prompt: Option<InputPrompt>,
    /// `GameModel::GetActiveMenu`.
    pub active_menu: usize,
    /// Menu sections (`SimulationData::msections`).
    pub menus: Vec<MenuSection>,
    /// `ui::Engine::GetWindow() != gameController->GetView()`.
    pub console_open: bool,
    /// `ui::Engine::GetScale`.
    pub window_scale: i32,
    /// `ui::Engine::GetFullscreen`.
    pub window_fullscreen: bool,
    /// `GameModel::GetBrushID`.
    pub brush_id: usize,
    /// `GameModel::BrushListSize` (three built-in brushes, plus any `.ptb`
    /// brushes the application loads).
    pub brush_list_size: usize,
    /// `GameModel::GetBrush().GetRadius()`.
    pub brush_radius: (i32, i32),
    /// `GameView::GetMousePosition`.
    pub mouse_position: (i32, i32),
    /// `GameModel::GetPerfectCircle`.
    pub perfect_circle: bool,
    next_id: u64,
}

impl Default for InterfaceState {
    fn default() -> Self {
        Self {
            windows: Vec::new(),
            components: Vec::new(),
            shown_window: None,
            text_input_refcount: 0,
            text_input_rect: None,
            message_box: None,
            error_message: None,
            confirm: None,
            input_prompt: None,
            active_menu: 8,
            menus: MENU_NAMES
                .iter()
                .map(|name| MenuSection {
                    name: (*name).to_owned(),
                    enabled: true,
                })
                .collect(),
            console_open: false,
            window_scale: 1,
            window_fullscreen: false,
            brush_id: 0,
            brush_list_size: 3,
            brush_radius: (4, 4),
            mouse_position: (0, 0),
            perfect_circle: false,
            next_id: 1,
        }
    }
}

impl InterfaceState {
    fn next_id(&mut self) -> u64 {
        let id = self.next_id;
        self.next_id += 1;
        id
    }

    /// Finds a window by id.
    pub fn window(&self, id: u64) -> Option<&WindowState> {
        self.windows.iter().find(|window| window.id == id)
    }

    /// Finds a window by id, mutably.
    pub fn window_mut(&mut self, id: u64) -> Option<&mut WindowState> {
        self.windows.iter_mut().find(|window| window.id == id)
    }

    /// Finds a component by id.
    pub fn component(&self, id: u64) -> Option<&Component> {
        self.components.iter().find(|component| component.id == id)
    }

    /// Finds a component by id, mutably.
    pub fn component_mut(&mut self, id: u64) -> Option<&mut Component> {
        self.components.iter_mut().find(|component| component.id == id)
    }

    /// Returns the component's owning window, if it has been added to one.
    pub fn parent_window(&self, component_id: u64) -> Option<u64> {
        self.windows
            .iter()
            .find(|window| window.components.contains(&component_id))
            .map(|window| window.id)
    }
}

/// Calls a stored Lua callback with `args`, doing nothing when absent.
fn call_callback<A: mlua::IntoLuaMulti>(
    lua: &Lua,
    callback: &Option<RegistryKey>,
    args: A,
) -> mlua::Result<()> {
    let Some(key) = callback else {
        return Ok(());
    };
    let function: Function = lua.registry_value(key)?;
    function.call::<()>(args)
}

/// Shared handle to the interface state.
#[derive(Clone)]
pub struct InterfaceHandle(Rc<RefCell<InterfaceState>>);

impl InterfaceHandle {
    /// Borrows the state.
    pub fn borrow(&self) -> Ref<'_, InterfaceState> {
        self.0.borrow()
    }

    /// Borrows the state mutably.
    pub fn borrow_mut(&self) -> RefMut<'_, InterfaceState> {
        self.0.borrow_mut()
    }

    /// Closes the pending message box and runs its callback
    /// (`LuaInterface.cpp:51-64`).
    ///
    /// The dialog is taken out before the callback runs, so the callback may
    /// call back into `interface` without tripping the `RefCell`.
    pub fn dismiss_message_box(&self, lua: &Lua) -> mlua::Result<()> {
        let dialog = { self.borrow_mut().message_box.take() };
        let Some(dialog) = dialog else {
            return Ok(());
        };
        call_callback(lua, &dialog.callback, ())
    }

    /// Closes the pending error dialog and runs its callback
    /// (`LuaInterface.cpp:81-94`).
    pub fn dismiss_error_message(&self, lua: &Lua) -> mlua::Result<()> {
        let dialog = { self.borrow_mut().error_message.take() };
        let Some(dialog) = dialog else {
            return Ok(());
        };
        call_callback(lua, &dialog.callback, ())
    }

    /// Closes the pending confirmation dialog and runs its callback with the
    /// answer (`LuaInterface.cpp:145-160`).
    pub fn dismiss_confirm(&self, lua: &Lua, confirmed: bool) -> mlua::Result<()> {
        let dialog = { self.borrow_mut().confirm.take() };
        let Some(dialog) = dialog else {
            return Ok(());
        };
        call_callback(lua, &dialog.callback, confirmed)
    }

    /// Closes the pending text prompt and runs its callback with the entered
    /// text (`None` for cancel) (`LuaInterface.cpp:116-130`).
    pub fn dismiss_input(&self, lua: &Lua, text: Option<&str>) -> mlua::Result<()> {
        let dialog = { self.borrow_mut().input_prompt.take() };
        let Some(dialog) = dialog else {
            return Ok(());
        };
        call_callback(lua, &dialog.callback, text)
    }

    fn same_state(&self, other: &Self) -> bool {
        Rc::ptr_eq(&self.0, &other.0)
    }
}

impl crate::LuaScriptingEngine {
    /// Returns the `interface` state owned by this Lua state.
    ///
    /// The application draws [`InterfaceState::windows`] and resolves each
    /// [`WindowState::components`] id through [`InterfaceState::component`].
    pub fn interface(&self) -> InterfaceHandle {
        self.lua
            .app_data_ref::<InterfaceHandle>()
            .expect("the interface table is installed with the Lua state")
            .clone()
    }
}

/// Reads the interface handle from the Lua app data.
fn interface_handle(lua: &Lua) -> mlua::Result<InterfaceHandle> {
    lua.app_data_ref::<InterfaceHandle>()
        .map(|handle| handle.clone())
        .ok_or_else(|| mlua::Error::runtime("the interface table is not installed"))
}

/// Queues a host-generated dialog for the application to show.
///
/// `LuaMisc::installScriptManager` opens `InformationMessage`/`ErrorMessage`
/// dialogs directly (`LuaMisc.cpp:78-96`); the port publishes the same state
/// through [`InterfaceState`] because the application owns the window stack.
pub(crate) fn queue_message(
    lua: &Lua,
    title: &str,
    message: &str,
    is_error: bool,
) -> mlua::Result<()> {
    let handle = interface_handle(lua)?;
    let mut state = handle.borrow_mut();
    if is_error {
        state.error_message = Some(ErrorMessage {
            message: message.to_owned(),
            callback: None,
        });
    } else {
        state.message_box = Some(MessageBox {
            title: title.to_owned(),
            message: message.to_owned(),
            large: false,
            callback: None,
        });
    }
    Ok(())
}

fn component_mut(state: &mut InterfaceState, id: u64) -> mlua::Result<&mut Component> {
    state
        .component_mut(id)
        .ok_or_else(|| mlua::Error::runtime("invalid component"))
}

fn window_mut(state: &mut InterfaceState, id: u64) -> mlua::Result<&mut WindowState> {
    state
        .window_mut(id)
        .ok_or_else(|| mlua::Error::runtime("invalid window"))
}

/// `luaL_checkstring`: numbers convert, everything else raises
/// (`LuaCompat.h`'s `tpt_lua_checkString`).
fn check_string(value: &Value) -> mlua::Result<String> {
    match value {
        Value::String(text) => Ok(text.to_string_lossy()),
        Value::Integer(number) => Ok(number.to_string()),
        Value::Number(number) => Ok(number.to_string()),
        other => Err(mlua::Error::runtime(format!(
            "string expected, got {}",
            other.type_name()
        ))),
    }
}

/// Reads an optional `(x, y)` pair: no arguments is a getter, otherwise both
/// numbers are required (`luaL_checktype(L, 1/2, LUA_TNUMBER)`).
fn pair(args: MultiValue) -> mlua::Result<Option<(i32, i32)>> {
    if args.is_empty() {
        return Ok(None);
    }
    let values: Vec<Value> = args.into_vec();
    if values.len() < 2 {
        return Err(mlua::Error::runtime("number expected, got no value"));
    }
    Ok(Some((
        value_to_i64(&values[0])? as i32,
        value_to_i64(&values[1])? as i32,
    )))
}

/// Assigns a component/window callback (`LuaComponentCallback::CheckAndAssignArg1`,
/// `LuaComponent.cpp:8-16`).
fn set_callback(lua: &Lua, slot: &mut Option<RegistryKey>, args: MultiValue) -> mlua::Result<()> {
    let value = args
        .into_iter()
        .next()
        .ok_or_else(|| mlua::Error::runtime("function expected, got no value"))?;
    match value {
        Value::Nil => {
            *slot = None;
            Ok(())
        }
        Value::Function(function) => {
            *slot = Some(lua.create_registry_value(function)?);
            Ok(())
        }
        other => Err(mlua::Error::runtime(format!(
            "function expected, got {}",
            other.type_name()
        ))),
    }
}

/// Trait shared by the six component userdata types.
pub trait ComponentUser {
    /// The interface state the component belongs to.
    fn handle(&self) -> &InterfaceHandle;
    /// The component id.
    fn id(&self) -> u64;
}

macro_rules! component_type {
    ($name:ident, $kind:expr, $extra:ident) => {
        /// Lua userdata for one component class.
        pub struct $name {
            handle: InterfaceHandle,
            id: u64,
        }

        impl ComponentUser for $name {
            fn handle(&self) -> &InterfaceHandle {
                &self.handle
            }

            fn id(&self) -> u64 {
                self.id
            }
        }

        impl $name {
            fn new(handle: InterfaceHandle, id: u64) -> Self {
                Self { handle, id }
            }
        }

        impl mlua::UserData for $name {
            fn add_methods<M: mlua::UserDataMethods<Self>>(methods: &mut M) {
                methods.add_method("position", method_position::<Self>);
                methods.add_method("size", method_size::<Self>);
                methods.add_method("visible", method_visible::<Self>);
                $extra(methods);
            }
        }

        impl $name {
            /// Creates the userdata for a fresh component and returns it.
            fn create(
                lua: &Lua,
                handle: &InterfaceHandle,
                position: (i32, i32),
                size: (i32, i32),
            ) -> mlua::Result<AnyUserData> {
                let id = {
                    let mut state = handle.borrow_mut();
                    let id = state.next_id();
                    state
                        .components
                        .push(Component::new(id, $kind, position, size));
                    id
                };
                lua.create_userdata(Self::new(handle.clone(), id))
            }
        }
    };
}

component_type!(LuaButton, ComponentKind::Button, add_button_methods);
component_type!(LuaLabel, ComponentKind::Label, add_label_methods);
component_type!(LuaTextbox, ComponentKind::Textbox, add_textbox_methods);
component_type!(LuaCheckbox, ComponentKind::Checkbox, add_checkbox_methods);
component_type!(LuaSlider, ComponentKind::Slider, add_slider_methods);
component_type!(LuaProgressBar, ComponentKind::ProgressBar, add_progress_bar_methods);

/// `LuaComponent::position` (`LuaComponent.cpp:29-43`).
fn method_position<T: ComponentUser>(
    _lua: &Lua,
    this: &T,
    args: MultiValue,
) -> mlua::Result<MultiValue> {
    let mut state = this.handle().borrow_mut();
    let component = component_mut(&mut state, this.id())?;
    match pair(args)? {
        None => Ok(MultiValue::from_vec(vec![
            Value::Integer(component.position.0 as i64),
            Value::Integer(component.position.1 as i64),
        ])),
        Some(position) => {
            component.position = position;
            Ok(MultiValue::new())
        }
    }
}

/// `LuaComponent::size` (`LuaComponent.cpp:45-61`).
fn method_size<T: ComponentUser>(
    _lua: &Lua,
    this: &T,
    args: MultiValue,
) -> mlua::Result<MultiValue> {
    let mut state = this.handle().borrow_mut();
    let component = component_mut(&mut state, this.id())?;
    match pair(args)? {
        None => Ok(MultiValue::from_vec(vec![
            Value::Integer(component.size.0 as i64),
            Value::Integer(component.size.1 as i64),
        ])),
        Some(size) => {
            component.size = size;
            Ok(MultiValue::new())
        }
    }
}

/// `LuaComponent::visible` (`LuaComponent.cpp:63-77`).
fn method_visible<T: ComponentUser>(
    _lua: &Lua,
    this: &T,
    args: MultiValue,
) -> mlua::Result<MultiValue> {
    let mut state = this.handle().borrow_mut();
    let component = component_mut(&mut state, this.id())?;
    if args.is_empty() {
        Ok(MultiValue::from_vec(vec![Value::Boolean(component.visible)]))
    } else {
        let values: Vec<Value> = args.into_vec();
        match values.first() {
            Some(Value::Boolean(value)) => {
                component.visible = *value;
                Ok(MultiValue::new())
            }
            Some(other) => Err(mlua::Error::runtime(format!(
                "boolean expected, got {}",
                other.type_name()
            ))),
            None => Err(mlua::Error::runtime("boolean expected, got no value")),
        }
    }
}

/// Shared `text([string])` method.
fn method_text<T: ComponentUser>(
    lua: &Lua,
    this: &T,
    args: MultiValue,
) -> mlua::Result<MultiValue> {
    let mut state = this.handle().borrow_mut();
    let component = component_mut(&mut state, this.id())?;
    if args.is_empty() {
        Ok(MultiValue::from_vec(vec![Value::String(
            lua.create_string(&component.text)?,
        )]))
    } else {
        let values: Vec<Value> = args.into_vec();
        component.text = check_string(
            values
                .first()
                .ok_or_else(|| mlua::Error::runtime("string expected, got no value"))?,
        )?;
        Ok(MultiValue::new())
    }
}

/// Shared `action([function|nil])` method (`LuaButton.cpp:47-53`).
fn method_action<T: ComponentUser>(
    lua: &Lua,
    this: &T,
    args: MultiValue,
) -> mlua::Result<MultiValue> {
    let mut state = this.handle().borrow_mut();
    let component = component_mut(&mut state, this.id())?;
    set_callback(lua, &mut component.action, args)?;
    Ok(MultiValue::new())
}

/// `LuaButton::enabled` (`LuaButton.cpp:26-45`).
fn method_enabled<T: ComponentUser>(
    _lua: &Lua,
    this: &T,
    args: MultiValue,
) -> mlua::Result<MultiValue> {
    let mut state = this.handle().borrow_mut();
    let component = component_mut(&mut state, this.id())?;
    if args.is_empty() {
        Ok(MultiValue::from_vec(vec![Value::Boolean(component.enabled)]))
    } else {
        let values: Vec<Value> = args.into_vec();
        match values.first() {
            Some(Value::Boolean(value)) => {
                component.enabled = *value;
                Ok(MultiValue::new())
            }
            Some(other) => Err(mlua::Error::runtime(format!(
                "boolean expected, got {}",
                other.type_name()
            ))),
            None => Err(mlua::Error::runtime("boolean expected, got no value")),
        }
    }
}

/// `LuaCheckbox::checked` (`LuaCheckbox.cpp:28-45`).
fn method_checked<T: ComponentUser>(
    _lua: &Lua,
    this: &T,
    args: MultiValue,
) -> mlua::Result<MultiValue> {
    let mut state = this.handle().borrow_mut();
    let component = component_mut(&mut state, this.id())?;
    if args.is_empty() {
        Ok(MultiValue::from_vec(vec![Value::Boolean(component.checked)]))
    } else {
        let values: Vec<Value> = args.into_vec();
        match values.first() {
            Some(Value::Boolean(value)) => {
                component.checked = *value;
                Ok(MultiValue::new())
            }
            Some(other) => Err(mlua::Error::runtime(format!(
                "boolean expected, got {}",
                other.type_name()
            ))),
            None => Err(mlua::Error::runtime("boolean expected, got no value")),
        }
    }
}

/// `LuaTextbox::readonly` (`LuaTextbox.cpp:36-49`).
fn method_readonly<T: ComponentUser>(
    _lua: &Lua,
    this: &T,
    args: MultiValue,
) -> mlua::Result<MultiValue> {
    let mut state = this.handle().borrow_mut();
    let component = component_mut(&mut state, this.id())?;
    if args.is_empty() {
        Ok(MultiValue::from_vec(vec![Value::Boolean(
            component.read_only,
        )]))
    } else {
        let values: Vec<Value> = args.into_vec();
        match values.first() {
            Some(Value::Boolean(value)) => {
                component.read_only = *value;
                Ok(MultiValue::new())
            }
            Some(other) => Err(mlua::Error::runtime(format!(
                "boolean expected, got {}",
                other.type_name()
            ))),
            None => Err(mlua::Error::runtime("boolean expected, got no value")),
        }
    }
}

/// `LuaTextbox::focus` (`LuaTextbox.cpp:52-69`).
fn method_focus<T: ComponentUser>(
    _lua: &Lua,
    this: &T,
    args: MultiValue,
) -> mlua::Result<MultiValue> {
    let mut state = this.handle().borrow_mut();
    let id = this.id();
    if args.is_empty() {
        let focused = state
            .parent_window(id)
            .and_then(|parent| state.window(parent))
            .is_some_and(|window| window.focused_component == Some(id));
        return Ok(MultiValue::from_vec(vec![Value::Boolean(focused)]));
    }
    let values: Vec<Value> = args.into_vec();
    let Some(Value::Boolean(focus)) = values.first() else {
        return Err(mlua::Error::runtime("boolean expected"));
    };
    if let Some(parent) = state.parent_window(id)
        && let Some(window) = state.window_mut(parent)
    {
        if *focus {
            window.focused_component = Some(id);
        } else if window.focused_component == Some(id) {
            window.focused_component = None;
        }
    }
    Ok(MultiValue::new())
}

/// Shared `onTextChanged([function|nil])` method.
fn method_on_text_changed<T: ComponentUser>(
    lua: &Lua,
    this: &T,
    args: MultiValue,
) -> mlua::Result<MultiValue> {
    let mut state = this.handle().borrow_mut();
    let component = component_mut(&mut state, this.id())?;
    set_callback(lua, &mut component.text_changed, args)?;
    Ok(MultiValue::new())
}

/// `LuaSlider::value` (`LuaSlider.cpp:48-63`).
fn method_value<T: ComponentUser>(
    _lua: &Lua,
    this: &T,
    args: MultiValue,
) -> mlua::Result<MultiValue> {
    let mut state = this.handle().borrow_mut();
    let component = component_mut(&mut state, this.id())?;
    if args.is_empty() {
        Ok(MultiValue::from_vec(vec![Value::Integer(
            component.value as i64,
        )]))
    } else {
        let values: Vec<Value> = args.into_vec();
        component.value = value_to_i64(
            values
                .first()
                .ok_or_else(|| mlua::Error::runtime("number expected, got no value"))?,
        )? as i32;
        Ok(MultiValue::new())
    }
}

/// `LuaSlider::steps` (`LuaSlider.cpp:31-44`).
fn method_steps<T: ComponentUser>(
    _lua: &Lua,
    this: &T,
    args: MultiValue,
) -> mlua::Result<MultiValue> {
    let mut state = this.handle().borrow_mut();
    let component = component_mut(&mut state, this.id())?;
    if args.is_empty() {
        Ok(MultiValue::from_vec(vec![Value::Integer(
            component.steps as i64,
        )]))
    } else {
        let values: Vec<Value> = args.into_vec();
        component.steps = value_to_i64(
            values
                .first()
                .ok_or_else(|| mlua::Error::runtime("number expected, got no value"))?,
        )? as i32;
        Ok(MultiValue::new())
    }
}

/// Shared `onValueChanged([function|nil])` method.
fn method_on_value_changed<T: ComponentUser>(
    lua: &Lua,
    this: &T,
    args: MultiValue,
) -> mlua::Result<MultiValue> {
    let mut state = this.handle().borrow_mut();
    let component = component_mut(&mut state, this.id())?;
    set_callback(lua, &mut component.value_changed, args)?;
    Ok(MultiValue::new())
}

/// `LuaProgressBar::progress` (`LuaProgressBar.cpp:30-43`).
fn method_progress<T: ComponentUser>(
    _lua: &Lua,
    this: &T,
    args: MultiValue,
) -> mlua::Result<MultiValue> {
    let mut state = this.handle().borrow_mut();
    let component = component_mut(&mut state, this.id())?;
    if args.is_empty() {
        Ok(MultiValue::from_vec(vec![Value::Integer(
            component.progress as i64,
        )]))
    } else {
        let values: Vec<Value> = args.into_vec();
        component.progress = value_to_i64(
            values
                .first()
                .ok_or_else(|| mlua::Error::runtime("number expected, got no value"))?,
        )? as i32;
        Ok(MultiValue::new())
    }
}

/// `LuaProgressBar::status` (`LuaProgressBar.cpp:45-58`).
fn method_status<T: ComponentUser>(
    lua: &Lua,
    this: &T,
    args: MultiValue,
) -> mlua::Result<MultiValue> {
    let mut state = this.handle().borrow_mut();
    let component = component_mut(&mut state, this.id())?;
    if args.is_empty() {
        Ok(MultiValue::from_vec(vec![Value::String(
            lua.create_string(&component.status)?,
        )]))
    } else {
        let values: Vec<Value> = args.into_vec();
        component.status = check_string(
            values
                .first()
                .ok_or_else(|| mlua::Error::runtime("string expected, got no value"))?,
        )?;
        Ok(MultiValue::new())
    }
}

/// Adds the class-specific methods to each userdata type.
fn add_button_methods<M: mlua::UserDataMethods<LuaButton>>(methods: &mut M) {
    methods.add_method("action", method_action::<LuaButton>);
    methods.add_method("text", method_text::<LuaButton>);
    methods.add_method("enabled", method_enabled::<LuaButton>);
}

fn add_label_methods<M: mlua::UserDataMethods<LuaLabel>>(methods: &mut M) {
    methods.add_method("text", method_text::<LuaLabel>);
}

fn add_textbox_methods<M: mlua::UserDataMethods<LuaTextbox>>(methods: &mut M) {
    methods.add_method("text", method_text::<LuaTextbox>);
    methods.add_method("readonly", method_readonly::<LuaTextbox>);
    methods.add_method("focus", method_focus::<LuaTextbox>);
    methods.add_method("onTextChanged", method_on_text_changed::<LuaTextbox>);
}

fn add_checkbox_methods<M: mlua::UserDataMethods<LuaCheckbox>>(methods: &mut M) {
    methods.add_method("action", method_action::<LuaCheckbox>);
    methods.add_method("text", method_text::<LuaCheckbox>);
    methods.add_method("checked", method_checked::<LuaCheckbox>);
}

fn add_slider_methods<M: mlua::UserDataMethods<LuaSlider>>(methods: &mut M) {
    methods.add_method("onValueChanged", method_on_value_changed::<LuaSlider>);
    methods.add_method("value", method_value::<LuaSlider>);
    methods.add_method("steps", method_steps::<LuaSlider>);
}

fn add_progress_bar_methods<M: mlua::UserDataMethods<LuaProgressBar>>(methods: &mut M) {
    methods.add_method("progress", method_progress::<LuaProgressBar>);
    methods.add_method("status", method_status::<LuaProgressBar>);
}

/// `LuaWindow::addComponent` is registered below; the per-class method sets
/// are passed to `component_type!` through the `add_*_methods` functions.
pub struct LuaWindow {
    handle: InterfaceHandle,
    id: u64,
}

impl mlua::UserData for LuaWindow {
    fn add_methods<M: mlua::UserDataMethods<Self>>(methods: &mut M) {
        methods.add_method("position", window_position);
        methods.add_method("size", window_size);
        methods.add_method("addComponent", window_add_component);
        methods.add_method("removeComponent", window_remove_component);
        methods.add_method("onInitialized", |lua, this, args| {
            window_callback(lua, this, args, "onInitialized")
        });
        methods.add_method("onExit", |lua, this, args| {
            window_callback(lua, this, args, "onExit")
        });
        methods.add_method("onTick", |lua, this, args| {
            window_callback(lua, this, args, "onTick")
        });
        methods.add_method("onDraw", |lua, this, args| {
            window_callback(lua, this, args, "onDraw")
        });
        methods.add_method("onFocus", |lua, this, args| {
            window_callback(lua, this, args, "onFocus")
        });
        methods.add_method("onBlur", |lua, this, args| {
            window_callback(lua, this, args, "onBlur")
        });
        methods.add_method("onTryExit", |lua, this, args| {
            window_callback(lua, this, args, "onTryExit")
        });
        methods.add_method("onTryOkay", |lua, this, args| {
            window_callback(lua, this, args, "onTryOkay")
        });
        methods.add_method("onMouseMove", |lua, this, args| {
            window_callback(lua, this, args, "onMouseMove")
        });
        methods.add_method("onMouseDown", |lua, this, args| {
            window_callback(lua, this, args, "onMouseDown")
        });
        methods.add_method("onMouseUp", |lua, this, args| {
            window_callback(lua, this, args, "onMouseUp")
        });
        methods.add_method("onMouseWheel", |lua, this, args| {
            window_callback(lua, this, args, "onMouseWheel")
        });
        methods.add_method("onKeyPress", |lua, this, args| {
            window_callback(lua, this, args, "onKeyPress")
        });
        methods.add_method("onKeyRelease", |lua, this, args| {
            window_callback(lua, this, args, "onKeyRelease")
        });
    }
}

/// `LuaWindow::position` (`LuaWindow.cpp:166-193`).
fn window_position(_lua: &Lua, this: &LuaWindow, args: MultiValue) -> mlua::Result<MultiValue> {
    let mut state = this.handle.borrow_mut();
    let window = window_mut(&mut state, this.id)?;
    match pair(args)? {
        None => Ok(MultiValue::from_vec(vec![
            Value::Integer(window.position.0 as i64),
            Value::Integer(window.position.1 as i64),
        ])),
        Some((x, y)) => {
            if x < 1 || y < 1 {
                return Err(mlua::Error::runtime(format!(
                    "Invalid position: '{x},{y}'"
                )));
            }
            window.position = (x, y);
            Ok(MultiValue::new())
        }
    }
}

/// `LuaWindow::size` (`LuaWindow.cpp:195-222`).
fn window_size(_lua: &Lua, this: &LuaWindow, args: MultiValue) -> mlua::Result<MultiValue> {
    let mut state = this.handle.borrow_mut();
    let window = window_mut(&mut state, this.id)?;
    match pair(args)? {
        None => Ok(MultiValue::from_vec(vec![
            Value::Integer(window.size.0 as i64),
            Value::Integer(window.size.1 as i64),
        ])),
        Some((w, h)) => {
            if w < 10 || h < 10 {
                return Err(mlua::Error::runtime(format!("Invalid size: '{w},{h}'")));
            }
            window.size = (w, h);
            Ok(MultiValue::new())
        }
    }
}

/// Resolves a component argument to `(handle, id)` (`Luna<T>::tryGet` chain,
/// `LuaWindow.cpp:86-120`).
fn component_argument(value: &Value) -> mlua::Result<(InterfaceHandle, u64)> {
    let Value::UserData(userdata) = value else {
        return Err(mlua::Error::runtime("Component expected"));
    };
    macro_rules! try_type {
        ($ty:ty) => {
            if let Ok(component) = userdata.borrow::<$ty>() {
                return Ok((component.handle.clone(), component.id));
            }
        };
    }
    try_type!(LuaButton);
    try_type!(LuaLabel);
    try_type!(LuaTextbox);
    try_type!(LuaCheckbox);
    try_type!(LuaSlider);
    try_type!(LuaProgressBar);
    Err(mlua::Error::runtime("Component expected"))
}

/// `LuaWindow::addComponent` (`LuaWindow.cpp:84-121`).
fn window_add_component(
    _lua: &Lua,
    this: &LuaWindow,
    value: Value,
) -> mlua::Result<MultiValue> {
    let (handle, component_id) = component_argument(&value)?;
    if !this.handle.same_state(&handle) {
        return Err(mlua::Error::runtime(
            "component belongs to a different Lua state",
        ));
    }
    let mut state = this.handle.borrow_mut();
    let window = window_mut(&mut state, this.id)?;
    if !window.components.contains(&component_id) {
        window.components.push(component_id);
    }
    if let Some(component) = state.component_mut(component_id) {
        component.owner = value.as_userdata().cloned();
    }
    Ok(MultiValue::new())
}

/// `LuaWindow::removeComponent` (`LuaWindow.cpp:123-164`).
fn window_remove_component(
    _lua: &Lua,
    this: &LuaWindow,
    value: Value,
) -> mlua::Result<MultiValue> {
    let (handle, component_id) = component_argument(&value)?;
    if !this.handle.same_state(&handle) {
        return Ok(MultiValue::new());
    }
    let mut state = this.handle.borrow_mut();
    if let Some(window) = state.window_mut(this.id) {
        window.components.retain(|id| *id != component_id);
        if window.focused_component == Some(component_id) {
            window.focused_component = None;
        }
    }
    if let Some(component) = state.component_mut(component_id) {
        component.owner = None;
    }
    Ok(MultiValue::new())
}

/// One of the fourteen `LuaWindow::on*` methods
/// (`LuaWindow.cpp:266-328`).
fn window_callback(
    lua: &Lua,
    this: &LuaWindow,
    args: MultiValue,
    name: &str,
) -> mlua::Result<MultiValue> {
    let mut state = this.handle.borrow_mut();
    let window = window_mut(&mut state, this.id)?;
    let slot = match name {
        "onInitialized" => &mut window.callbacks.on_initialized,
        "onExit" => &mut window.callbacks.on_exit,
        "onTick" => &mut window.callbacks.on_tick,
        "onDraw" => &mut window.callbacks.on_draw,
        "onFocus" => &mut window.callbacks.on_focus,
        "onBlur" => &mut window.callbacks.on_blur,
        "onTryExit" => &mut window.callbacks.on_try_exit,
        "onTryOkay" => &mut window.callbacks.on_try_okay,
        "onMouseMove" => &mut window.callbacks.on_mouse_move,
        "onMouseDown" => &mut window.callbacks.on_mouse_down,
        "onMouseUp" => &mut window.callbacks.on_mouse_up,
        "onMouseWheel" => &mut window.callbacks.on_mouse_wheel,
        "onKeyPress" => &mut window.callbacks.on_key_press,
        _ => &mut window.callbacks.on_key_release,
    };
    set_callback(lua, slot, args)?;
    Ok(MultiValue::new())
}

/// Creates a component userdata with the reference's constructor defaults.
fn create_component(
    lua: &Lua,
    handle: &InterfaceHandle,
    kind: ComponentKind,
    args: &MultiValue,
) -> mlua::Result<AnyUserData> {
    let values: Vec<Value> = args.clone().into_vec();
    let int_at = |index: usize, default: i64| -> mlua::Result<i32> {
        match values.get(index) {
            Some(value) => Ok(value_to_i64(value)? as i32),
            None => Ok(default as i32),
        }
    };
    let string_at = |index: usize| -> mlua::Result<String> {
        match values.get(index) {
            Some(value) => check_string(value),
            None => Ok(String::new()),
        }
    };
    let position = (int_at(0, 0)?, int_at(1, 0)?);
    let size = (int_at(2, 10)?, int_at(3, 10)?);
    let userdata = match kind {
        ComponentKind::Button => {
            let data = LuaButton::create(lua, handle, position, size)?;
            let text = string_at(4)?;
            let tooltip = string_at(5)?;
            {
                let mut state = handle.borrow_mut();
                let component = component_mut(&mut state, data.borrow::<LuaButton>()?.id)?;
                component.text = text;
                component.tooltip = tooltip;
            }
            data
        }
        ComponentKind::Label => {
            let data = LuaLabel::create(lua, handle, position, size)?;
            let text = string_at(4)?;
            {
                let mut state = handle.borrow_mut();
                let component = component_mut(&mut state, data.borrow::<LuaLabel>()?.id)?;
                component.text = text;
            }
            data
        }
        ComponentKind::Textbox => {
            let data = LuaTextbox::create(lua, handle, position, size)?;
            let text = string_at(4)?;
            let placeholder = string_at(5)?;
            {
                let mut state = handle.borrow_mut();
                let component = component_mut(&mut state, data.borrow::<LuaTextbox>()?.id)?;
                component.text = text;
                component.tooltip = placeholder;
            }
            data
        }
        ComponentKind::Checkbox => {
            let data = LuaCheckbox::create(lua, handle, position, size)?;
            let text = string_at(4)?;
            {
                let mut state = handle.borrow_mut();
                let component = component_mut(&mut state, data.borrow::<LuaCheckbox>()?.id)?;
                component.text = text;
            }
            data
        }
        ComponentKind::Slider => {
            let data = LuaSlider::create(lua, handle, position, size)?;
            let steps = int_at(4, 10)?;
            {
                let mut state = handle.borrow_mut();
                let component = component_mut(&mut state, data.borrow::<LuaSlider>()?.id)?;
                component.steps = steps;
            }
            data
        }
        ComponentKind::ProgressBar => {
            let data = LuaProgressBar::create(lua, handle, position, size)?;
            let value = int_at(4, 0)?;
            let status = string_at(5)?;
            {
                let mut state = handle.borrow_mut();
                let component = component_mut(&mut state, data.borrow::<LuaProgressBar>()?.id)?;
                component.progress = value;
                component.status = status;
            }
            data
        }
    };
    Ok(userdata)
}

/// Creates a window userdata (`LuaWindow::LuaWindow`, `LuaWindow.cpp:38-82`).
fn create_window(lua: &Lua, handle: &InterfaceHandle, args: MultiValue) -> mlua::Result<AnyUserData> {
    let values: Vec<Value> = args.into_vec();
    let int_at = |index: usize, default: i64| -> mlua::Result<i32> {
        match values.get(index) {
            Some(value) => Ok(value_to_i64(value)? as i32),
            None => Ok(default as i32),
        }
    };
    let mut pos_x = int_at(0, 1)?;
    let mut pos_y = int_at(1, 1)?;
    let mut size_x = int_at(2, 10)?;
    let mut size_y = int_at(3, 10)?;
    if pos_x < 1 && pos_x != -1 {
        pos_x = 1;
    }
    if pos_y < 1 && pos_y != -1 {
        pos_y = 1;
    }
    if size_x < 10 {
        size_x = 10;
    }
    if size_y < 10 {
        size_y = 10;
    }
    let id = {
        let mut state = handle.borrow_mut();
        let id = state.next_id();
        state.windows.push(WindowState {
            id,
            position: (pos_x, pos_y),
            size: (size_x, size_y),
            visible: false,
            initialized: false,
            components: Vec::new(),
            focused_component: None,
            callbacks: WindowCallbacks::default(),
        });
        id
    };
    lua.create_userdata(LuaWindow {
        handle: handle.clone(),
        id,
    })
}

/// `interface.showWindow(window)` (`LuaInterface.cpp:381-390`).
fn show_window(_lua: &Lua, window: AnyUserData) -> mlua::Result<()> {
    let window = window.borrow::<LuaWindow>()?;
    let mut state = window.handle.borrow_mut();
    if let Some(target) = state.window_mut(window.id) {
        target.visible = true;
        target.initialized = true;
    }
    state.shown_window = Some(window.id);
    Ok(())
}

/// `interface.closeWindow(window)` (`LuaInterface.cpp:392-400`).
fn close_window(_lua: &Lua, window: AnyUserData) -> mlua::Result<()> {
    let window = window.borrow::<LuaWindow>()?;
    let mut state = window.handle.borrow_mut();
    if let Some(target) = state.window_mut(window.id) {
        target.visible = false;
    }
    if state.shown_window == Some(window.id) {
        state.shown_window = None;
    }
    Ok(())
}

/// `interface.addComponent(component)` (`LuaInterface.cpp:274-316`): adds to
/// the currently shown script window.
fn interface_add_component(lua: &Lua, value: Value) -> mlua::Result<()> {
    let (handle, component_id) = component_argument(&value)?;
    let _ = lua;
    let mut state = handle.borrow_mut();
    let Some(window_id) = state.shown_window else {
        return Ok(());
    };
    if let Some(window) = state.window_mut(window_id)
        && !window.components.contains(&component_id)
    {
        window.components.push(component_id);
    }
    if let Some(component) = state.component_mut(component_id) {
        component.owner = value.as_userdata().cloned();
    }
    Ok(())
}

/// `interface.removeComponent(component)` (`LuaInterface.cpp:318-365`).
fn interface_remove_component(lua: &Lua, value: Value) -> mlua::Result<()> {
    let (handle, component_id) = component_argument(&value)?;
    let _ = lua;
    let mut state = handle.borrow_mut();
    if let Some(window_id) = state.parent_window(component_id)
        && let Some(window) = state.window_mut(window_id)
    {
        window.components.retain(|id| *id != component_id);
        if window.focused_component == Some(component_id) {
            window.focused_component = None;
        }
    }
    if let Some(component) = state.component_mut(component_id) {
        component.owner = None;
    }
    Ok(())
}

/// `interface.grabTextInput()` (`LuaInterface.cpp:367-374`).
fn grab_text_input(lua: &Lua) -> mlua::Result<()> {
    let handle = interface_handle(lua)?;
    handle.borrow_mut().text_input_refcount += 1;
    Ok(())
}

/// `interface.dropTextInput()` (`LuaInterface.cpp:376-383`).
fn drop_text_input(lua: &Lua) -> mlua::Result<()> {
    let handle = interface_handle(lua)?;
    handle.borrow_mut().text_input_refcount -= 1;
    Ok(())
}

/// `interface.textInputRect(x, y, w, h)` (`LuaInterface.cpp:385-395`).
fn text_input_rect(lua: &Lua, args: (i64, i64, i64, i64)) -> mlua::Result<()> {
    let handle = interface_handle(lua)?;
    handle.borrow_mut().text_input_rect =
        Some((args.0 as i32, args.1 as i32, args.2 as i32, args.3 as i32));
    Ok(())
}

/// Reads the optional string argument at `index`, returning `default` when it
/// is absent or not a string (`PickIfType`, `LuaInterface.cpp:26-45`).
fn pick_string(values: &[Value], index: usize, default: &str) -> String {
    match values.get(index) {
        Some(Value::String(text)) => text.to_string_lossy(),
        _ => default.to_owned(),
    }
}

/// Reads the optional boolean argument at `index` (`PickIfType`).
fn pick_bool(values: &[Value], index: usize, default: bool) -> bool {
    match values.get(index) {
        Some(Value::Boolean(value)) => *value,
        _ => default,
    }
}

/// `cb->Assign(L, lua_gettop(L))`: the callback is always the last argument.
fn last_callback(lua: &Lua, values: &[Value]) -> mlua::Result<Option<RegistryKey>> {
    match values.last() {
        Some(Value::Function(function)) => Ok(Some(lua.create_registry_value(function.clone())?)),
        _ => Ok(None),
    }
}

/// `interface.beginMessageBox([title, message, large, callback])`
/// (`LuaInterface.cpp:38-66`).
fn begin_message_box(lua: &Lua, args: MultiValue) -> mlua::Result<()> {
    let values: Vec<Value> = args.into_vec();
    let callback = last_callback(lua, &values)?;
    let handle = interface_handle(lua)?;
    let mut state = handle.borrow_mut();
    state.message_box = Some(MessageBox {
        title: pick_string(&values, 0, "Title"),
        message: pick_string(&values, 1, "Message"),
        large: pick_bool(&values, 2, false),
        callback,
    });
    Ok(())
}

/// `interface.beginThrowError([message, callback])`
/// (`LuaInterface.cpp:68-96`).
fn begin_throw_error(lua: &Lua, args: MultiValue) -> mlua::Result<()> {
    let values: Vec<Value> = args.into_vec();
    let callback = last_callback(lua, &values)?;
    let handle = interface_handle(lua)?;
    let mut state = handle.borrow_mut();
    state.error_message = Some(ErrorMessage {
        message: pick_string(&values, 0, "Error text"),
        callback,
    });
    Ok(())
}

/// `interface.beginConfirm([title, message, buttonText, callback])`
/// (`LuaInterface.cpp:134-171`).
fn begin_confirm(lua: &Lua, args: MultiValue) -> mlua::Result<()> {
    let values: Vec<Value> = args.into_vec();
    let callback = last_callback(lua, &values)?;
    let handle = interface_handle(lua)?;
    let mut state = handle.borrow_mut();
    state.confirm = Some(ConfirmDialog {
        title: pick_string(&values, 0, "Title"),
        message: pick_string(&values, 1, "Message"),
        button_text: pick_string(&values, 2, "Confirm"),
        callback,
    });
    Ok(())
}

/// `interface.beginInput([title, prompt, text, shadow, callback])`
/// (`LuaInterface.cpp:98-132`).
fn begin_input(lua: &Lua, args: MultiValue) -> mlua::Result<()> {
    let values: Vec<Value> = args.into_vec();
    let callback = last_callback(lua, &values)?;
    let handle = interface_handle(lua)?;
    let mut state = handle.borrow_mut();
    state.input_prompt = Some(InputPrompt {
        title: pick_string(&values, 0, "Title"),
        prompt: pick_string(&values, 1, "Enter some text:"),
        text: pick_string(&values, 2, ""),
        shadow: pick_string(&values, 3, ""),
        callback,
    });
    Ok(())
}

/// `interface.console([show])` (`LuaInterface.cpp:173-191`).
fn console(lua: &Lua, args: MultiValue) -> mlua::Result<MultiValue> {
    let handle = interface_handle(lua)?;
    let mut state = handle.borrow_mut();
    if args.is_empty() {
        return Ok(MultiValue::from_vec(vec![Value::Boolean(
            state.console_open,
        )]));
    }
    let values: Vec<Value> = args.into_vec();
    state.console_open = is_truthy(
        values
            .first()
            .ok_or_else(|| mlua::Error::runtime("boolean expected"))?,
    );
    Ok(MultiValue::new())
}

/// `interface.windowSize([scale, kiosk])` (`LuaInterface.cpp:457-489`).
fn window_size_lua(lua: &Lua, args: MultiValue) -> mlua::Result<MultiValue> {
    let handle = interface_handle(lua)?;
    let mut state = handle.borrow_mut();
    if args.is_empty() {
        return Ok(MultiValue::from_vec(vec![
            Value::Integer(state.window_scale as i64),
            Value::Boolean(state.window_fullscreen),
        ]));
    }
    let values: Vec<Value> = args.into_vec();
    let mut scale = match values.first() {
        Some(value) => value_to_i64(value)? as i32,
        None => 1,
    };
    let kiosk = pick_bool(&values, 1, false);
    if !(1..=10).contains(&scale) {
        scale = 1;
    }
    state.window_scale = scale;
    state.window_fullscreen = kiosk;
    Ok(MultiValue::new())
}

/// `interface.brushID([index])` (`LuaInterface.cpp:202-219`).
fn brush_id(lua: &Lua, args: MultiValue) -> mlua::Result<MultiValue> {
    let handle = interface_handle(lua)?;
    let mut state = handle.borrow_mut();
    if args.is_empty() {
        return Ok(MultiValue::from_vec(vec![Value::Integer(
            state.brush_id as i64,
        )]));
    }
    let values: Vec<Value> = args.into_vec();
    let index = value_to_i64(
        values
            .first()
            .ok_or_else(|| mlua::Error::runtime("number expected"))?,
    )? as i32;
    if index < 0 || index as usize >= state.brush_list_size {
        return Err(mlua::Error::runtime(format!(
            "Invalid brush index {index}"
        )));
    }
    state.brush_id = index as usize;
    Ok(MultiValue::new())
}

/// `interface.brushRadius([rx, ry])` (`LuaInterface.cpp:221-234`).
fn brush_radius(lua: &Lua, args: MultiValue) -> mlua::Result<MultiValue> {
    let handle = interface_handle(lua)?;
    let mut state = handle.borrow_mut();
    if args.is_empty() {
        return Ok(MultiValue::from_vec(vec![
            Value::Number(state.brush_radius.0 as f64),
            Value::Number(state.brush_radius.1 as f64),
        ]));
    }
    let values: Vec<Value> = args.into_vec();
    if values.len() < 2 {
        return Err(mlua::Error::runtime("number expected, got no value"));
    }
    state.brush_radius = (
        value_to_i64(&values[0])? as i32,
        value_to_i64(&values[1])? as i32,
    );
    Ok(MultiValue::new())
}

/// `interface.mousePosition()` (`LuaInterface.cpp:236-244`).
fn mouse_position(lua: &Lua) -> mlua::Result<(i64, i64)> {
    let handle = interface_handle(lua)?;
    let state = handle.borrow();
    Ok((
        state.mouse_position.0 as i64,
        state.mouse_position.1 as i64,
    ))
}

/// `interface.perfectCircleBrush([enabled])` (`LuaInterface.cpp:402-414`).
fn perfect_circle_brush(lua: &Lua, args: MultiValue) -> mlua::Result<MultiValue> {
    let handle = interface_handle(lua)?;
    let mut state = handle.borrow_mut();
    if args.is_empty() {
        return Ok(MultiValue::from_vec(vec![Value::Boolean(
            state.perfect_circle,
        )]));
    }
    let values: Vec<Value> = args.into_vec();
    state.perfect_circle = is_truthy(
        values
            .first()
            .ok_or_else(|| mlua::Error::runtime("boolean expected"))?,
    );
    Ok(MultiValue::new())
}

/// `interface.activeMenu([menu])` (`LuaInterface.cpp:416-433`).
fn active_menu(lua: &Lua, args: MultiValue) -> mlua::Result<MultiValue> {
    let handle = interface_handle(lua)?;
    let mut state = handle.borrow_mut();
    if args.is_empty() {
        return Ok(MultiValue::from_vec(vec![Value::Integer(
            state.active_menu as i64,
        )]));
    }
    let values: Vec<Value> = args.into_vec();
    let menu = value_to_i64(
        values
            .first()
            .ok_or_else(|| mlua::Error::runtime("number expected"))?,
    )?;
    if menu < 0 || menu as usize >= state.menus.len() {
        return Err(mlua::Error::runtime("Invalid menu"));
    }
    state.active_menu = menu as usize;
    Ok(MultiValue::new())
}

/// `interface.menuEnabled(menu [, enabled])` (`LuaInterface.cpp:435-459`).
fn menu_enabled(lua: &Lua, args: MultiValue) -> mlua::Result<MultiValue> {
    let handle = interface_handle(lua)?;
    let mut state = handle.borrow_mut();
    let values: Vec<Value> = args.into_vec();
    let menu = value_to_i64(
        values
            .first()
            .ok_or_else(|| mlua::Error::runtime("number expected"))?,
    )?;
    if menu < 0 || menu as usize >= state.menus.len() {
        return Err(mlua::Error::runtime("Invalid menu"));
    }
    if values.len() == 1 {
        return Ok(MultiValue::from_vec(vec![Value::Boolean(
            state.menus[menu as usize].enabled,
        )]));
    }
    let enabled = match values.get(1) {
        Some(Value::Boolean(value)) => *value,
        Some(other) => {
            return Err(mlua::Error::runtime(format!(
                "boolean expected, got {}",
                other.type_name()
            )));
        }
        None => false,
    };
    state.menus[menu as usize].enabled = enabled;
    Ok(MultiValue::new())
}

/// `interface.numMenus([onlyEnabled])` (`LuaInterface.cpp:461-476`).
fn num_menus(lua: &Lua, args: MultiValue) -> mlua::Result<i64> {
    let handle = interface_handle(lua)?;
    let state = handle.borrow();
    let values: Vec<Value> = args.into_vec();
    let only_enabled = match values.first() {
        Some(Value::Boolean(value)) => *value,
        Some(other) => {
            return Err(mlua::Error::runtime(format!(
                "boolean expected, got {}",
                other.type_name()
            )));
        }
        None => true,
    };
    let count = if only_enabled {
        state.menus.iter().filter(|menu| menu.enabled).count()
    } else {
        state.menus.len()
    };
    Ok(count as i64)
}

/// `interface.activeTool(index [, identifier])` (`LuaInterface.cpp:246-272`).
fn active_tool(lua: &Lua, args: MultiValue) -> mlua::Result<MultiValue> {
    let values: Vec<Value> = args.into_vec();
    let index = value_to_i64(
        values
            .first()
            .ok_or_else(|| mlua::Error::runtime("number expected"))?,
    )?;
    if index < 0 || index as usize >= NUM_TOOLINDICES {
        return Err(mlua::Error::runtime(format!(
            "Invalid tool index {index}"
        )));
    }
    let index = index as usize;
    let tools = crate::api::tools::tools_handle(lua)?;
    if values.len() < 2 {
        let state = tools.borrow();
        let identifier = state.active[index].clone().unwrap_or_default();
        return Ok(MultiValue::from_vec(vec![Value::String(
            lua.create_string(identifier)?,
        )]));
    }
    let identifier = check_string(&values[1])?;
    let mut state = tools.borrow_mut();
    if state.find_identifier(&identifier).is_none() {
        return Err(mlua::Error::runtime(format!(
            "Invalid tool identifier {identifier}"
        )));
    }
    state.active[index] = Some(identifier);
    Ok(MultiValue::new())
}

/// `LuaInterface::Open` (`LuaInterface.cpp:491-540`).
pub(crate) fn build_interface_table(lua: &Lua) -> mlua::Result<Table> {
    let handle = InterfaceHandle(Rc::new(RefCell::new(InterfaceState::default())));
    lua.set_app_data(handle.clone());

    let interface = lua.create_table()?;
    interface.set(
        "showWindow",
        lua.create_function(|lua, window: AnyUserData| show_window(lua, window))?,
    )?;
    interface.set(
        "closeWindow",
        lua.create_function(|lua, window: AnyUserData| close_window(lua, window))?,
    )?;
    interface.set(
        "addComponent",
        lua.create_function(|lua, value: Value| interface_add_component(lua, value))?,
    )?;
    interface.set(
        "removeComponent",
        lua.create_function(|lua, value: Value| interface_remove_component(lua, value))?,
    )?;
    interface.set(
        "grabTextInput",
        lua.create_function(|lua, ()| grab_text_input(lua))?,
    )?;
    interface.set(
        "dropTextInput",
        lua.create_function(|lua, ()| drop_text_input(lua))?,
    )?;
    interface.set("textInputRect", lua.create_function(text_input_rect)?)?;
    interface.set("beginInput", lua.create_function(begin_input)?)?;
    interface.set("beginMessageBox", lua.create_function(begin_message_box)?)?;
    interface.set("beginConfirm", lua.create_function(begin_confirm)?)?;
    interface.set(
        "beginThrowError",
        lua.create_function(begin_throw_error)?,
    )?;
    interface.set("activeMenu", lua.create_function(active_menu)?)?;
    interface.set("menuEnabled", lua.create_function(menu_enabled)?)?;
    interface.set("numMenus", lua.create_function(num_menus)?)?;
    interface.set(
        "perfectCircleBrush",
        lua.create_function(perfect_circle_brush)?,
    )?;
    interface.set("console", lua.create_function(console)?)?;
    interface.set("windowSize", lua.create_function(window_size_lua)?)?;
    interface.set("brushID", lua.create_function(brush_id)?)?;
    interface.set("brushRadius", lua.create_function(brush_radius)?)?;
    interface.set(
        "mousePosition",
        lua.create_function(|lua, ()| mouse_position(lua))?,
    )?;
    interface.set("activeTool", lua.create_function(active_tool)?)?;

    interface.set("MOUSEUP_NORMAL", MOUSEUP_NORMAL)?;
    interface.set("MOUSEUP_BLUR", MOUSEUP_BLUR)?;
    interface.set("MOUSEUP_DRAWEND", MOUSEUP_DRAWEND)?;
    interface.set("NUM_TOOLINDICES", NUM_TOOLINDICES as i64)?;
    add_key_constants(&interface)?;

    // Constructors (`Luna<T>::Register`, `LuaLuna.h:13-58`).
    let constructor = |kind: ComponentKind| {
        let handle = handle.clone();
        lua.create_function(move |lua, args: MultiValue| {
            create_component(lua, &handle, kind, &args)
        })
    };
    interface.set("button", constructor(ComponentKind::Button)?)?;
    interface.set("label", constructor(ComponentKind::Label)?)?;
    interface.set("textbox", constructor(ComponentKind::Textbox)?)?;
    interface.set("checkbox", constructor(ComponentKind::Checkbox)?)?;
    interface.set("slider", constructor(ComponentKind::Slider)?)?;
    interface.set("progressBar", constructor(ComponentKind::ProgressBar)?)?;
    {
        let handle = handle.clone();
        interface.set(
            "window",
            lua.create_function(move |lua, args: MultiValue| create_window(lua, &handle, args))?,
        )?;
    }

    // `LuaWindow`, `LuaButton`, ... metatables are installed by
    // `Lua::create_userdata` on the first instance of each type.
    Ok(interface)
}

/// Adds the SDL modifier and mouse-button constants TPT exposes on `interface`
/// (`LuaSDLKeys.h:495-513`). The scancode table is not reproduced: this port's
/// key events carry names, not SDL scancodes.
fn add_key_constants(interface: &Table) -> mlua::Result<()> {
    interface.set("KMOD_NONE", 0x0000)?;
    interface.set("KMOD_LSHIFT", 0x0001)?;
    interface.set("KMOD_RSHIFT", 0x0002)?;
    interface.set("KMOD_LCTRL", 0x0040)?;
    interface.set("KMOD_RCTRL", 0x0080)?;
    interface.set("KMOD_LALT", 0x0100)?;
    interface.set("KMOD_RALT", 0x0200)?;
    interface.set("KMOD_LGUI", 0x0400)?;
    interface.set("KMOD_RGUI", 0x0800)?;
    interface.set("KMOD_NUM", 0x1000)?;
    interface.set("KMOD_CAPS", 0x2000)?;
    interface.set("KMOD_MODE", 0x4000)?;
    interface.set("KMOD_CTRL", 0x00C0)?;
    interface.set("KMOD_SHIFT", 0x0003)?;
    interface.set("KMOD_ALT", 0x0300)?;
    interface.set("KMOD_GUI", 0x0C00)?;
    interface.set("SDL_BUTTON_LEFT", 1)?;
    interface.set("SDL_BUTTON_MIDDLE", 2)?;
    interface.set("SDL_BUTTON_RIGHT", 3)?;
    interface.set("SDL_BUTTON_X1", 4)?;
    interface.set("SDL_BUTTON_X2", 5)?;
    Ok(())
}
