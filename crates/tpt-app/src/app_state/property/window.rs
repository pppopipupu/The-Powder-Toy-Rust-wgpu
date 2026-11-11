use tpt_core::elements::ElementRegistry;
use tpt_core::particle::GpuParticle;

use crate::app_state::inspector::WallSample;
use crate::app_state::{AppState, SelectedTool, persist_tool_settings};
use crate::prefs::Prefs;

use super::format::{format_value, parse_value};
use super::kind::{PropertyKind, PropertyValue, value_from_prefs, value_of};

/// State of the floating property window (`PropertyWindow`).
#[derive(Clone, Debug)]
pub struct PropertyWindow {
    /// Whether the window is visible.
    pub open: bool,
    /// Property the tool rewrites.
    pub field: PropertyKind,
    /// Text in the value box.
    pub value_text: String,
    /// Parsed value of [`Self::value_text`]; `None` while the text is invalid.
    pub value: Option<PropertyValue>,
    /// Parse error shown under the value box.
    pub error: Option<&'static str>,
    /// Pixel the value was sampled from by `takePropertyFrom`.
    pub seeded_from: Option<(i32, i32)>,
    /// Wall sampled instead of a particle, when the pixel held no particle.
    pub seeded_wall: Option<WallSample>,
    /// `DrawFill`: a click floods the connected particles instead of stamping
    /// the brush (`PropertyTool::DrawFill` / `Simulation::flood_prop`).
    ///
    /// Persisted in `Prop.Fill` so it survives closing the window, switching
    /// tools and restarting the application.
    pub fill: bool,
}

impl PropertyWindow {
    /// Builds the closed window from the persisted `Prop.Type` / `Prop.Value`.
    pub fn new(prefs: &Prefs) -> Self {
        let field = PropertyKind::from_index(prefs.prop_field);
        let value = value_from_prefs(field, prefs);
        Self {
            open: false,
            field,
            value_text: format_value(field, value, prefs.temperature_scale),
            value: Some(value),
            error: None,
            seeded_from: None,
            seeded_wall: None,
            fill: prefs.prop_fill,
        }
    }

    /// Opens the window with the tool's current configuration
    /// (`PropertyTool::Select` / `OpenWindow` without a sample).
    pub fn open(&mut self, prefs: &Prefs) {
        self.field = PropertyKind::from_index(prefs.prop_field);
        let value = value_from_prefs(self.field, prefs);
        self.value_text = format_value(self.field, value, prefs.temperature_scale);
        self.value = Some(value);
        self.error = None;
        self.seeded_from = None;
        self.seeded_wall = None;
        self.open = true;
    }

    /// Opens the window seeded from a sampled particle
    /// (`SampleTool`'s shift behaviour -> `PropertyWindow::TakePropertyFrom`).
    ///
    /// The configured property is kept and only the value box is replaced, so
    /// the tool stays on the field the user was editing.
    pub fn open_from_particle(&mut self, prefs: &Prefs, particle: &GpuParticle, at: (i32, i32)) {
        self.open(prefs);
        let value = value_of(particle, self.field);
        self.value_text = format_value(self.field, value, prefs.temperature_scale);
        self.value = Some(value);
        self.seeded_from = Some(at);
    }

    /// Opens the window after sampling a wall.
    ///
    /// A wall has no particle property to read, so the configured
    /// `Prop.Type` / `Prop.Value` pair is kept and only the wall report is
    /// recorded; the window shows it above the property dropdown.
    pub fn open_from_wall(&mut self, prefs: &Prefs, wall: WallSample) {
        self.open(prefs);
        self.seeded_wall = Some(wall);
    }

    /// Hides the window without touching the tool configuration
    /// (`PropertyWindow::OnTryExit` without OK).
    pub fn close(&mut self) {
        self.open = false;
        self.seeded_from = None;
        self.seeded_wall = None;
    }

    /// Selects a property, re-validating the current text against it like
    /// `PropertyWindow::HandlePropertyChange`.
    pub fn select_field(
        &mut self,
        field: PropertyKind,
        elements: &ElementRegistry,
        temperature_scale: u32,
    ) {
        self.field = field;
        self.revalidate(elements, temperature_scale);
    }

    /// Moves the selection through [`PropertyKind::ALL`], wrapping at the ends
    /// (the reference window's up/down keys).
    pub fn cycle_field(&mut self, delta: i32, elements: &ElementRegistry, temperature_scale: u32) {
        let len = PropertyKind::ALL.len() as i32;
        let current = PropertyKind::ALL
            .iter()
            .position(|kind| *kind == self.field)
            .unwrap_or(0) as i32;
        let next = (current + delta).rem_euclid(len) as usize;
        self.select_field(PropertyKind::ALL[next], elements, temperature_scale);
    }

    /// Re-parses the value box, updating [`Self::value`] and [`Self::error`].
    pub fn revalidate(&mut self, elements: &ElementRegistry, temperature_scale: u32) {
        match parse_value(self.field, &self.value_text, elements, temperature_scale) {
            Ok(value) => {
                self.value = Some(value);
                self.error = None;
            }
            Err(message) => {
                self.value = None;
                self.error = Some(message);
            }
        }
    }

    /// Applies the value box to `Prop.Type` / `Prop.Value` and closes the
    /// window (`PropertyTool::SetConfiguration behind the OK button).
    ///
    /// Returns `false` and keeps the window open when the text does not parse.
    /// The caller writes `prefs.json`; the state machine stays file-free.
    pub fn commit(
        &mut self,
        prefs: &mut Prefs,
        elements: &ElementRegistry,
        temperature_scale: u32,
    ) -> bool {
        self.revalidate(elements, temperature_scale);
        let Some(value) = self.value else {
            return false;
        };
        prefs.prop_field = self.field.index();
        prefs.prop_value = match value {
            PropertyValue::Int(value) => value as f32,
            PropertyValue::Float(value) => value,
        };
        prefs.prop_fill = self.fill;
        self.close();
        true
    }
}

impl AppState {
    /// Opens the property window with the tool's current configuration.
    pub fn open_property_window(&mut self) {
        self.property_window.open(&self.prefs);
    }

    /// `SampleTool`'s shift behaviour: adopt the property tool and open its
    /// window seeded from the particle (or wall) under the cursor.
    ///
    /// `SampleTool::Draw` switches the primary brush to `DEFAULT_UI_PROPERTY`
    /// and calls `PropertyTool::OpenWindow(sim, partId)`. A wall has no
    /// property command, so sampling one keeps the persisted configuration and
    /// shows the wall report in the window instead
    /// (`SimulationSample`'s wall fields).
    pub fn sample_property_window(&mut self, x: i32, y: i32) {
        let particle = self.particle_at(x, y);
        if particle.is_none() && self.wall_at(x, y).is_none() {
            return;
        }
        self.primary_tool = SelectedTool::Property;
        persist_tool_settings(self);
        match particle {
            Some(particle) => {
                self.property_window
                    .open_from_particle(&self.prefs, &particle, (x, y))
            }
            None => {
                let Some(wall) = self.wall_sample_at(x, y) else {
                    return;
                };
                self.property_window.open_from_wall(&self.prefs, wall);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::i18n::t;
    use tpt_core::constants::{PT_DUST, TEMPSCALE_CELSIUS};
    use tpt_core::elements::ElementRegistry;

    fn registry() -> ElementRegistry {
        ElementRegistry::new_default()
    }

    fn celsius() -> u32 {
        TEMPSCALE_CELSIUS
    }

    #[test]
    fn window_opens_with_the_persisted_configuration() {
        let mut prefs = Prefs::default();
        prefs.prop_field = PropertyKind::Life.index();
        prefs.prop_value = 42.0;

        let mut window = PropertyWindow::new(&prefs);
        assert!(!window.open, "the window starts closed");
        window.open(&prefs);
        assert!(window.open);
        assert_eq!(window.field, PropertyKind::Life);
        assert_eq!(window.value_text, "42");
        assert_eq!(window.value, Some(PropertyValue::Int(42)));
        assert_eq!(window.seeded_from, None);
    }

    #[test]
    fn window_samples_the_particle_under_the_cursor() {
        let mut prefs = Prefs::default();
        prefs.prop_field = PropertyKind::Temp.index();
        let particle = GpuParticle {
            ty: PT_DUST,
            life: 9,
            temp: 300.0,
            vx: 1.5,
            ..GpuParticle::default()
        };

        let mut window = PropertyWindow::new(&prefs);
        window.open_from_particle(&prefs, &particle, (12, 34));
        assert!(window.open);
        // `TakePropertyFrom` keeps the configured property and samples its
        // value.
        assert_eq!(window.field, PropertyKind::Temp);
        assert_eq!(window.value, Some(PropertyValue::Float(300.0)));
        // The port's default display scale is Kelvin (`Prefs::default`).
        assert_eq!(window.value_text, "300.00K");
        assert_eq!(window.seeded_from, Some((12, 34)));

        // Sampling again with another configured field reads that field.
        prefs.prop_field = PropertyKind::Vx.index();
        window.open_from_particle(&prefs, &particle, (0, 0));
        assert_eq!(window.field, PropertyKind::Vx);
        assert_eq!(window.value, Some(PropertyValue::Float(1.5)));
        assert_eq!(window.value_text, "1.5");
    }

    #[test]
    fn window_rejects_invalid_text_until_it_parses() {
        let elements = registry();
        let mut prefs = Prefs::default();
        let mut window = PropertyWindow::new(&prefs);
        window.open(&prefs);

        window.value_text = String::from("not a number");
        window.revalidate(&elements, celsius());
        assert_eq!(window.value, None);
        assert_eq!(window.error, Some(t("property.err_invalid_int")));
        // OK is disabled while the text is invalid: nothing is written.
        assert!(!window.commit(&mut prefs, &elements, celsius()));
        assert!(window.open);
        assert_eq!(prefs.prop_field, PropertyKind::Tmp.index());

        window.value_text = String::from("0x10");
        window.revalidate(&elements, celsius());
        assert_eq!(window.value, Some(PropertyValue::Int(16)));
        assert_eq!(window.error, None);
        assert!(window.commit(&mut prefs, &elements, celsius()));
        assert!(!window.open);
        assert_eq!(prefs.prop_field, PropertyKind::Tmp.index());
        assert_eq!(prefs.prop_value, 16.0);
    }

    #[test]
    fn window_selection_cycles_and_wraps() {
        let elements = registry();
        let mut prefs = Prefs::default();
        prefs.prop_field = PropertyKind::ALL[0].index();
        let mut window = PropertyWindow::new(&prefs);
        window.open(&prefs);

        assert_eq!(window.field, PropertyKind::ALL[0]);
        window.cycle_field(1, &elements, celsius());
        assert_eq!(window.field, PropertyKind::ALL[1]);
        window.cycle_field(-1, &elements, celsius());
        assert_eq!(window.field, PropertyKind::ALL[0]);
        // Up from the first entry wraps to the last one.
        window.cycle_field(-1, &elements, celsius());
        assert_eq!(window.field, *PropertyKind::ALL.last().unwrap());
        window.cycle_field(1, &elements, celsius());
        assert_eq!(window.field, PropertyKind::ALL[0]);
    }

    #[test]
    fn window_field_change_revalidates_the_current_text() {
        let elements = registry();
        let prefs = Prefs::default();
        let mut window = PropertyWindow::new(&prefs);
        window.open(&prefs);
        assert_eq!(window.value_text, "0");

        // `0` is not a valid element type (`PT_NONE` is disabled), so switching
        // to `type` reports an error like `PropertyWindow::Update`.
        window.select_field(PropertyKind::Type, &elements, celsius());
        assert_eq!(window.value, None);
        assert_eq!(window.error, Some(t("property.err_unknown_element")));

        window.select_field(PropertyKind::Life, &elements, celsius());
        assert_eq!(window.value, Some(PropertyValue::Int(0)));
    }

    #[test]
    fn window_commit_stores_temperatures_in_kelvin() {
        let elements = registry();
        let mut prefs = Prefs::default();
        prefs.prop_field = PropertyKind::Temp.index();
        prefs.prop_value = 300.0;
        let mut window = PropertyWindow::new(&prefs);
        window.open(&prefs);
        // Kelvin is the default scale; a `C` suffix still parses as Celsius.
        assert_eq!(window.value_text, "300.00K");

        window.value_text = String::from("100C");
        assert!(window.commit(&mut prefs, &elements, celsius()));
        assert_eq!(prefs.prop_field, PropertyKind::Temp.index());
        assert!((prefs.prop_value - 373.15).abs() < 0.01);
    }

    #[test]
    fn window_keeps_the_fill_mode_and_samples_walls() {
        let mut prefs = Prefs::default();
        prefs.prop_field = PropertyKind::Tmp.index();
        let mut window = PropertyWindow::new(&prefs);
        assert!(!window.fill, "`DrawFill` starts off");

        window.fill = true;
        window.open(&prefs);
        // Opening the window must not drop the mode: it belongs to the tool
        // configuration, not to one open/close cycle.
        assert!(window.fill);
        assert_eq!(window.seeded_wall, None);

        let wall = WallSample {
            id: 4,
            name: String::from("CONDUCTOR"),
            colour: 0x00FF_8000,
            emap: 1,
            air_temperature: 300.0,
            pressure: 0.5,
        };
        window.open_from_wall(&prefs, wall.clone());
        assert!(window.open);
        assert_eq!(window.seeded_wall.as_ref(), Some(&wall));
        // The persisted configuration is kept; a wall has no value to sample.
        assert_eq!(window.field, PropertyKind::Tmp);

        window.close();
        assert_eq!(window.seeded_wall, None);
        assert!(window.fill, "closing keeps the tool configuration");
    }
}
