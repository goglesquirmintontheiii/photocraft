//! Edit › Generative Fill… (and the selection's right-click menu, below Content-Aware Fill): a
//! dialog like Content-Aware Fill's with a prompt, negative prompt and seed (all may be left
//! empty), the model, and the variations made so far. `+` (or Enter in the prompt) makes another
//! with a local diffusion model; it is shown on the canvas as soon as it lands, and clicking a
//! variation shows that one instead (each is a new layer, one undo step that the next replaces).
//! OK keeps the one shown; with none made yet it fills at once. Cancel takes it back.
//!
//! Edit › Generative Sharpen… (below Generative Fill in the selection's menu) is the same dialog
//! without prompts: how much detail, the seed and the model.

use crate::PhotocraftApp;
use crate::state::DialogKind;
use crate::theme::Tokens;
use photocraft_ml::ModelKind;
use serde_json::{Map, Value, json};

const MARK: &str = "__generativeFill";
pub const COMMAND: &str = "edit.generativeFill";
pub const SHARPEN: &str = "edit.generativeSharpen";

/// Whether the dialog is Generative Sharpen's.
fn sharpen(fields: &Map<String, Value>) -> bool {
    fields.get("mode").and_then(Value::as_str) == Some("sharpen")
}

fn command(fields: &Map<String, Value>) -> &'static str {
    if sharpen(fields) { SHARPEN } else { COMMAND }
}

fn label(fields: &Map<String, Value>) -> &'static str {
    if sharpen(fields) { "Generative Sharpen" } else { "Generative Fill" }
}

pub fn owns(fields: &Map<String, Value>) -> bool {
    fields.contains_key(MARK)
}

fn text(fields: &Map<String, Value>, k: &str) -> String {
    fields.get(k).and_then(Value::as_str).unwrap_or("").to_string()
}

pub fn open(app: &mut PhotocraftApp) -> Result<Value, String> {
    open_mode(app, false)
}

/// Edit › Generative Sharpen….
pub fn open_sharpen(app: &mut PhotocraftApp) -> Result<Value, String> {
    open_mode(app, true)
}

fn open_mode(app: &mut PhotocraftApp, sharpen: bool) -> Result<Value, String> {
    let st = app.session.active().ok_or("no document")?;
    if st.doc.selection.is_none() {
        return Err(if sharpen { "Make a selection first: Generative Sharpen sharpens the selection" } else { "Make a selection first: Generative Fill fills the selection" }.into());
    }
    // A dialog left over from before (it can't be: Cancel and OK clear it) is forgotten.
    let _ = app.run("edit.generativeFill.cancel", json!({}));
    let o = &app.ui.tool_options;
    let mut fields = Map::new();
    fields.insert(MARK.into(), json!(true));
    if sharpen {
        // The Generative Enhance brush and this dialog share their settings.
        fields.insert("mode".into(), json!("sharpen"));
        fields.insert("__label".into(), json!("Generative Sharpen…"));
        fields.insert("seed".into(), json!(o.enhance_seed));
        fields.insert("model".into(), json!(o.enhance_model));
        fields.insert("detail".into(), json!(o.enhance_detail));
    } else {
        // The Infill brush and this dialog share the last prompt.
        fields.insert("__label".into(), json!("Generative Fill…"));
        fields.insert("prompt".into(), json!(o.infill_prompt));
        fields.insert("negative".into(), json!(o.infill_negative));
        fields.insert("seed".into(), json!(o.infill_seed));
        fields.insert("model".into(), json!(o.infill_model));
    }
    let id = app.ui.open_dialog(DialogKind::Command, fields);
    Ok(json!({"dialog": id}))
}

/// The models it can use, best first.
fn models(app: &PhotocraftApp, fields: &Map<String, Value>) -> Vec<(String, String)> {
    let kind = if sharpen(fields) { ModelKind::Enhance } else { ModelKind::Inpainting };
    app.services.models.as_ref().map(|m| m.available(kind).into_iter().map(|i| (i.id, i.name)).collect()).unwrap_or_default()
}

fn detail(fields: &Map<String, Value>) -> f32 {
    fields.get("detail").and_then(Value::as_f64).map_or(0.5, |d| d.clamp(0.0, 1.0) as f32)
}

/// The command's parameters from the dialog's fields.
fn params(app: &PhotocraftApp, fields: &Map<String, Value>, preview: bool) -> Value {
    let model = text(fields, "model");
    let model = if models(app, fields).iter().any(|(id, _)| *id == model) { json!(model) } else { Value::Null };
    let seed = crate::ai_tools_ui::parse_seed(&text(fields, "seed")).map_or(Value::Null, |s| json!(s));
    if sharpen(fields) {
        return json!({"seed": seed, "detail": detail(fields), "model": model, "preview": preview});
    }
    json!({
        "prompt": text(fields, "prompt"),
        "negativePrompt": text(fields, "negative"),
        "seed": seed,
        "model": model,
        "preview": preview,
    })
}

/// The Generative Fill or Sharpen job running on the active document, if any: its progress and
/// message.
fn running(app: &PhotocraftApp) -> Option<(f32, String)> {
    app.session
        .active_job()
        .filter(|j| [COMMAND, SHARPEN].contains(&j.command.as_str()) || ["Generative Fill", "Generative Sharpen"].contains(&j.label.as_str()))
        .map(|j| (j.progress, j.message))
}

/// Makes another variation.
fn generate(app: &mut PhotocraftApp, fields: &mut Map<String, Value>) {
    remember(app, fields);
    let p = params(app, fields, true);
    match app.run(command(fields), p) {
        Ok(_) => {
            fields.remove("error");
        }
        Err(e) => {
            fields.insert("error".into(), json!(e));
        }
    }
}

/// Keeps the prompt for next time (and for the Infill brush).
fn remember(app: &mut PhotocraftApp, fields: &Map<String, Value>) {
    let o = &mut app.ui.tool_options;
    if sharpen(fields) {
        o.enhance_seed = text(fields, "seed");
        o.enhance_model = text(fields, "model");
        o.enhance_detail = detail(fields);
        return;
    }
    o.infill_prompt = text(fields, "prompt");
    o.infill_negative = text(fields, "negative");
    o.infill_seed = text(fields, "seed");
    o.infill_model = text(fields, "model");
}

pub fn body(app: &mut PhotocraftApp, ui: &mut egui::Ui, fields: &mut Map<String, Value>) {
    let t = Tokens::get(ui.ctx());
    let width = 400.0;
    ui.set_min_width(width);
    let available = models(app, fields);
    if available.is_empty() {
        ui.label(
            egui::RichText::new(if app.services.models.is_none() {
                tl!("AI models run in the desktop app.")
            } else if sharpen(fields) {
                tl!("Generative Sharpen needs a model. Download one in Edit › Preferences › AI Models (Generative Enhance).")
            } else {
                tl!("Generative Fill needs a model. Download one in Edit › Preferences › AI Models (Generative Fill & Infill).")
            })
            .color(t.text_dim),
        );
        ui.add_space(6.0);
        if app.services.models.is_some() && crate::widgets::secondary_button(ui, tl!("Get Models…"), 0.0).clicked() {
            crate::prefs_ui::open_preferences(app, "ai");
        }
        return;
    }
    let enter = if sharpen(fields) {
        // Detail and seed.
        let (mut d, mut seed) = (detail(fields), text(fields, "seed"));
        ui.horizontal(|ui| {
            crate::ai_tools_ui::detail_menu(ui, "generative-sharpen-detail", &mut d);
            ui.add_space(8.0);
            crate::ai_tools_ui::seed_field(ui, "generative-sharpen-seed", &mut seed);
        });
        fields.insert("detail".into(), json!(d));
        fields.insert("seed".into(), json!(seed));
        false
    } else {
        // Prompt, negative prompt, seed.
        let (mut prompt, mut negative, mut seed) = (text(fields, "prompt"), text(fields, "negative"), text(fields, "seed"));
        let enter = crate::ai_tools_ui::prompt_fields(ui, "generative-fill", &mut prompt, &mut negative, &mut seed, width);
        fields.insert("prompt".into(), json!(prompt));
        fields.insert("negative".into(), json!(negative));
        fields.insert("seed".into(), json!(seed));
        enter
    };
    ui.add_space(6.0);
    // Model.
    ui.horizontal(|ui| {
        ui.label(egui::RichText::new(tl!("Model:")).color(t.text_dim).size(12.0));
        let mut current = text(fields, "model");
        if !available.iter().any(|(id, _)| *id == current) {
            current = available.first().map(|(id, _)| id.clone()).unwrap_or_default();
        }
        let options: Vec<(String, &str)> = available.iter().map(|(id, name)| (id.clone(), name.as_str())).collect();
        crate::widgets::dropdown(ui, "generative-fill-model", &mut current, &options, 260.0);
        fields.insert("model".into(), json!(current));
        if crate::icons::button(ui, "settings", 22.0, false, tl!("AI Models…")).clicked() {
            crate::prefs_ui::open_preferences(app, "ai");
        }
    });
    ui.add_space(8.0);
    crate::widgets::hairline(ui);
    ui.add_space(6.0);
    // Variations.
    ui.label(egui::RichText::new(tl!("Variations")).font(crate::theme::semibold(13.0)));
    ui.add_space(4.0);
    let previews = app.session.generative_fill_previews();
    let shown = app.session.generative_fill_shown();
    let job = running(app);
    let doc = app.session.active().map(|st| st.doc.id.0).unwrap_or(0);
    let side = 80.0;
    let mut clicked = None;
    let mut plus = false;
    ui.horizontal_wrapped(|ui| {
        ui.spacing_mut().item_spacing = egui::vec2(6.0, 6.0);
        for (i, (seed, (w, h, bytes))) in previews.iter().enumerate() {
            let key = egui::Id::new(("generative-fill-thumb", doc, i, *seed));
            let tex = ui.data(|d| d.get_temp::<egui::TextureHandle>(key)).unwrap_or_else(|| {
                let img = egui::ColorImage::from_rgba_unmultiplied([*w, *h], bytes);
                let tex = ui.ctx().load_texture(format!("generative-fill-{doc}-{i}"), img, egui::TextureOptions::LINEAR);
                ui.data_mut(|d| d.insert_temp(key, tex.clone()));
                tex
            });
            let (rect, resp) = ui.allocate_exact_size(egui::vec2(side, side), egui::Sense::click());
            let scale = (side - 6.0) / (*w).max(*h) as f32;
            let img_rect = egui::Rect::from_center_size(rect.center(), egui::vec2(*w as f32 * scale, *h as f32 * scale));
            ui.painter().rect_filled(rect, 4.0, t.field);
            ui.painter().image(tex.id(), img_rect, egui::Rect::from_min_max(egui::pos2(0.0, 0.0), egui::pos2(1.0, 1.0)), egui::Color32::WHITE);
            let selected = shown == Some(i);
            let stroke = if selected {
                egui::Stroke::new(2.0, t.accent)
            } else if resp.hovered() {
                egui::Stroke::new(1.0, t.text_dim)
            } else {
                egui::Stroke::new(1.0, t.separator)
            };
            ui.painter().rect_stroke(rect, 4.0, stroke, egui::StrokeKind::Inside);
            let resp = resp.on_hover_text(crate::i18n::fmt(tl!("Variation {n} · seed {seed}"), &[("n", &(i + 1).to_string()), ("seed", &seed.to_string())]));
            if resp.clicked() && !selected && job.is_none() {
                clicked = Some(i);
            }
        }
        // The + tile: makes another (a progress ring while one is being made).
        let (rect, resp) = ui.allocate_exact_size(egui::vec2(side, side), egui::Sense::click());
        ui.painter().rect_filled(rect, 4.0, if resp.hovered() && job.is_none() { t.hover } else { t.field });
        ui.painter().rect_stroke(rect, 4.0, egui::Stroke::new(1.0, t.separator), egui::StrokeKind::Inside);
        match &job {
            Some((progress, _)) => {
                ui.painter().text(rect.center(), egui::Align2::CENTER_CENTER, format!("{:.0}%", progress * 100.0), egui::FontId::proportional(14.0), t.text);
                ui.ctx().request_repaint_after(std::time::Duration::from_millis(150));
            }
            None => {
                crate::icons::paint(ui, egui::Rect::from_center_size(rect.center(), egui::vec2(24.0, 24.0)), "plus", 20.0, t.icon);
                if resp.on_hover_text(tl!("Make a variation")).clicked() {
                    plus = true;
                }
            }
        }
    });
    if let Some((_, msg)) = &job {
        ui.add_space(4.0);
        ui.horizontal(|ui| {
            ui.label(egui::RichText::new(msg).color(t.text_dim).size(12.0));
            if crate::widgets::secondary_button(ui, tl!("Stop"), 0.0).clicked()
                && let Some(j) = app.session.active_job()
            {
                app.session.cancel_job(j.id);
            }
        });
    } else if previews.is_empty() {
        ui.add_space(4.0);
        let hint = if sharpen(fields) {
            tl!("Press + to make a variation, or OK to sharpen with one. Seeds change the result with Stable Diffusion models.")
        } else {
            tl!("Press + (or Enter in the prompt) to make a variation, or OK to fill with one.")
        };
        ui.label(egui::RichText::new(hint).color(t.text_dim).size(12.0));
    }
    if let Some(e) = fields.get("error").and_then(Value::as_str) {
        ui.add_space(4.0);
        ui.label(egui::RichText::new(e).color(t.danger).size(12.0));
    }
    if let Some(i) = clicked {
        match app.run("edit.generativeFill.show", json!({"variation": i})) {
            Ok(_) => {
                fields.remove("error");
            }
            Err(e) => {
                fields.insert("error".into(), json!(e));
            }
        }
    }
    if (plus || enter) && job.is_none() {
        generate(app, fields);
    }
}

/// OK can't close the dialog while a variation is being made.
pub fn validate(app: &PhotocraftApp) -> Result<(), String> {
    if running(app).is_some() {
        return Err("Wait for the variation being made, or press Stop".into());
    }
    Ok(())
}

/// OK: keeps the variation shown, or with none made yet, fills at once (in the background).
pub fn confirm(app: &mut PhotocraftApp, fields: &Map<String, Value>) -> Result<Value, String> {
    remember(app, fields);
    let shown = app.run("edit.generativeFill.variations", json!({}))?.get("shown").and_then(Value::as_u64);
    if shown.is_some() {
        return app.run("edit.generativeFill.commit", json!({}));
    }
    let _ = app.run("edit.generativeFill.cancel", json!({}));
    let p = params(app, fields, false);
    app.run(command(fields), p)
}

/// Cancel: stops a variation being made and takes back the one shown.
pub fn cancel(app: &mut PhotocraftApp, fields: &Map<String, Value>) {
    remember(app, fields);
    if let Some(j) = app.session.active_job().filter(|j| j.label == label(fields)) {
        app.session.cancel_job(j.id);
    }
    let _ = app.run("edit.generativeFill.cancel", json!({}));
}

#[cfg(test)]
mod tests {
    use super::*;

    fn app() -> PhotocraftApp {
        let mut app = PhotocraftApp::new(photocraft_engine::Session::new(), crate::Services::default());
        app.session.execute("file.new", json!({"width": 64, "height": 48})).unwrap();
        app
    }

    #[test]
    fn it_needs_a_selection_and_shares_the_prompt_with_infill() {
        let mut app = app();
        assert!(open(&mut app).is_err());
        app.run("select.rect", json!({"x": 4, "y": 4, "width": 20, "height": 20})).unwrap();
        app.ui.tool_options.infill_prompt = "a red kite".into();
        let r = open(&mut app).unwrap();
        let id = r["dialog"].as_u64().unwrap();
        let d = app.ui.dialogs.iter().find(|d| d.id == id).unwrap();
        assert!(owns(&d.fields));
        assert_eq!(text(&d.fields, "prompt"), "a red kite");
        let mut f = d.fields.clone();
        f.insert("prompt".into(), json!("a blue kite"));
        f.insert("seed".into(), json!("12"));
        let p = params(&app, &f, true);
        assert_eq!(p["seed"], 12);
        assert_eq!(p["model"], Value::Null, "no models: the best (none)");
        assert_eq!(p["preview"], true);
        // Cancel keeps what was typed for next time.
        cancel(&mut app, &f);
        assert_eq!(app.ui.tool_options.infill_prompt, "a blue kite");
        assert_eq!(app.ui.tool_options.infill_seed, "12");
        // OK with no model says why.
        assert!(confirm(&mut app, &f).is_err());
    }
}
