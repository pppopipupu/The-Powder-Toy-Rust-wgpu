//! Reusable modal layer: confirm prompt, text prompt and error box.
//!
//! Ports the shared `gui/dialogues` windows of the C++ build:
//! `ConfirmPrompt.cpp` (Cancel/confirm), `TextPrompt.cpp` (one text field
//! whose contents are returned to the caller) and `ErrorMessage.cpp` (title,
//! failing message, dismiss button). Only the newest [`crate::app_state::Modal`]
//! of the stack is drawn, so it owns the keyboard exactly like `MakeActiveWindow`
//! promoted the newest `ui::Window` in `gui/interface/Engine.cpp:70-108`. Escape
//! cancels or dismisses it; the outcome reaches the opening panel through
//! [`AppState::modal_result`].

use egui::{Key, Modifiers, RichText};

use crate::app_state::{AppState, ModalKind, ModalOutcome, ModalResult};
use crate::i18n::t;

use super::shared::TEXT_DIM;

/// Draws the newest dialog of [`AppState::modals`], if any.
///
/// Must run after every other panel so the dialog paints on top and its
/// backdrop swallows pointer input. A closed dialog leaves its outcome in
/// [`AppState::modal_result`] for the panel that opened it.
pub(super) fn draw_modals(ctx: &egui::Context, state: &mut AppState) {
    let Some(modal) = state.modals.top_mut() else {
        return;
    };
    let id = modal.id();
    let focus_first = modal.take_focus_request();
    let mut action: Option<ModalOutcome> = None;

    // `Window::DoKeyPress` (`Window.cpp:375-379`) routes Escape to the cancel
    // button and Return to the okay button of the active window.
    let escape = ctx.input_mut(|input| input.consume_key(Modifiers::NONE, Key::Escape));

    egui::Modal::new(egui::Id::unique(("tpt_modal", id))).show(ctx, |ui| {
        ui.set_min_width(320.0);
        ui.set_max_width(460.0);

        match modal.kind_mut() {
            ModalKind::Confirm {
                title,
                message,
                confirm_label,
            } => {
                ui.label(RichText::new(title.as_str()).strong().size(15.0));
                ui.add_space(4.0);
                ui.label(message.as_str());
                ui.add_space(10.0);
                ui.horizontal(|ui| {
                    let cancel = ui.button(t("modal.cancel"));
                    if focus_first {
                        cancel.request_focus();
                    }
                    if cancel.clicked() {
                        action = Some(ModalOutcome::Cancelled);
                    }
                    let confirm = ui.button(confirm_label.as_str());
                    if confirm.clicked() {
                        action = Some(ModalOutcome::Confirmed);
                    }
                });
            }
            ModalKind::Prompt {
                title,
                message,
                input,
                hint,
                submit_label,
            } => {
                ui.label(RichText::new(title.as_str()).strong().size(15.0));
                ui.add_space(4.0);
                ui.label(message.as_str());
                ui.add_space(6.0);
                // One id per dialog: a fresh prompt starts empty and never
                // inherits the cursor or selection of the previous one.
                let field = ui.add(
                    egui::TextEdit::singleline(input)
                        .id(egui::Id::unique(("tpt_modal_prompt", id)))
                        .desired_width(f32::INFINITY)
                        .hint_text(hint.as_str()),
                );
                if focus_first {
                    field.request_focus();
                }
                // `TextPrompt.cpp` submits on Enter as well as the Okay button.
                if field.lost_focus() && ui.input(|input| input.key_pressed(Key::Enter)) {
                    action = Some(ModalOutcome::Submitted(input.clone()));
                }
                ui.add_space(2.0);
                ui.label(RichText::new(hint.as_str()).small().color(TEXT_DIM));
                ui.add_space(10.0);
                ui.horizontal(|ui| {
                    if ui.button(t("modal.cancel")).clicked() {
                        action = Some(ModalOutcome::Cancelled);
                    }
                    if ui.button(submit_label.as_str()).clicked() {
                        action = Some(ModalOutcome::Submitted(input.clone()));
                    }
                });
            }
            ModalKind::Error { title, message } => {
                ui.label(
                    RichText::new(title.as_str())
                        .strong()
                        .size(15.0)
                        .color(egui::Color32::from_rgb(230, 120, 120)),
                );
                ui.add_space(4.0);
                ui.label(message.as_str());
                ui.add_space(10.0);
                ui.horizontal(|ui| {
                    let dismiss = ui.button(t("modal.dismiss"));
                    if focus_first {
                        dismiss.request_focus();
                    }
                    if dismiss.clicked() {
                        action = Some(ModalOutcome::Dismissed);
                    }
                });
            }
        }
    });

    if action.is_none() {
        // Return is the okay button of the active window. The prompt's text
        // field handles its own Enter first (`field.lost_focus()` above), so
        // this only fires when focus sits on a button.
        let enter = ctx.input_mut(|input| input.consume_key(Modifiers::NONE, Key::Enter));
        if enter {
            action = Some(match modal.kind() {
                ModalKind::Confirm { .. } => ModalOutcome::Confirmed,
                ModalKind::Prompt { input, .. } => ModalOutcome::Submitted(input.clone()),
                ModalKind::Error { .. } => ModalOutcome::Dismissed,
            });
        }
    }

    if action.is_none() && escape {
        // `ConfirmPrompt`/`TextPrompt` close without invoking Okay and
        // `ErrorMessage` dismisses, both through the Escape key.
        action = Some(match modal.kind() {
            ModalKind::Error { .. } => ModalOutcome::Dismissed,
            _ => ModalOutcome::Cancelled,
        });
    }

    if let Some(outcome) = action {
        // `close` only pops the dialog this frame drew, never a newer one.
        if state.modals.close(id) {
            state.modal_result = Some(ModalResult { id, outcome });
        }
    }
}
