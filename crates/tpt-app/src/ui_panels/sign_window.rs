//! Floating sign text editor (`SignTool`'s `SignWindow`).
//!
//! `SignWindow`: a text box with an OK button, a justification selector, and
//! Move and Delete buttons when editing an existing sign. Move hides the window
//! and makes the sign follow the pointer until the next canvas click.

use egui::RichText;
use tpt_core::sign::{JUSTIFY_LEFT, JUSTIFY_MIDDLE, JUSTIFY_NONE, JUSTIFY_RIGHT};
use tpt_save::MAX_SIGN_TEXT;

use crate::app_state::AppState;
use crate::i18n::{t, t_fmt};

use super::shared::TEXT_DIM;

/// Draws the sign editor when it is open.
pub(super) fn draw_sign_window(ctx: &egui::Context, state: &mut AppState) {
    if !state.sign_editor.open {
        return;
    }

    let mut open = true;
    let mut commit = false;
    let mut cancel = false;
    let mut delete = false;
    let mut move_sign = false;

    let title = if state.sign_editor.index.is_some() {
        t("signs.title_edit")
    } else {
        t("signs.title_new")
    };

    egui::Window::new(title)
        .open(&mut open)
        .resizable(false)
        .default_width(280.0)
        .show(ctx, |ui| {
            let editor = &mut state.sign_editor;

            ui.label(
                RichText::new(t_fmt(
                    "signs.anchor",
                    &[&editor.x.to_string(), &editor.y.to_string()],
                ))
                .small()
                .color(TEXT_DIM),
            );
            ui.add_space(2.0);

            let edit = ui.add(
                egui::TextEdit::singleline(&mut editor.text)
                    .hint_text(t("signs.placeholder"))
                    .char_limit(MAX_SIGN_TEXT)
                    .desired_width(f32::INFINITY),
            );
            // The window is opened by a canvas click, so the text box must
            // take the keyboard immediately; without this the first keystroke
            // would still go to the canvas.
            if editor.focus {
                edit.request_focus();
                editor.focus = false;
            }
            if edit.lost_focus() && ui.input(|input| input.key_pressed(egui::Key::Enter)) {
                commit = true;
            }
            if ui.input(|input| input.key_pressed(egui::Key::Escape)) {
                cancel = true;
            }

            ui.label(
                RichText::new(t_fmt(
                    "signs.char_count",
                    &[
                        &editor.text.chars().count().to_string(),
                        &MAX_SIGN_TEXT.to_string(),
                    ],
                ))
                .small()
                .color(TEXT_DIM),
            );
            ui.add_space(4.0);

            // Justification: `sign::Justification`'s four modes, drawn as a
            // segmented row like the reference window's combo box.
            ui.horizontal(|ui| {
                ui.label(
                    RichText::new(t("signs.justification"))
                        .small()
                        .color(TEXT_DIM),
                );
                for (mode, key) in [
                    (JUSTIFY_LEFT, "signs.justify_left"),
                    (JUSTIFY_MIDDLE, "signs.justify_middle"),
                    (JUSTIFY_RIGHT, "signs.justify_right"),
                    (JUSTIFY_NONE, "signs.justify_none"),
                ] {
                    ui.selectable_value(&mut editor.ju, mode, t(key));
                }
            });
            ui.add_space(4.0);

            ui.horizontal(|ui| {
                if ui
                    .add(egui::Button::new(t("signs.ok")).min_size(egui::vec2(72.0, 22.0)))
                    .clicked()
                {
                    commit = true;
                }
                if ui.button(t("signs.cancel")).clicked() {
                    cancel = true;
                }
                if editor.index.is_some() {
                    // `SignTool.cpp:122-131`: the move handle hides the window
                    // and lets the sign follow the pointer.
                    if ui.button(t("signs.move")).clicked() {
                        move_sign = true;
                    }
                    if ui.button(t("signs.delete")).clicked() {
                        delete = true;
                    }
                }
            });
        });

    if move_sign {
        state.start_sign_move();
    } else if !open || cancel {
        state.close_sign_editor();
    } else if delete {
        state.delete_edited_sign();
    } else if commit {
        state.commit_sign_editor();
    }
}
