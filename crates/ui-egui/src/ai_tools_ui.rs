//! The AI tools in the shell: AI Object Selection and the Generative Brushes (Generative
//! Removal, Reflection Removal).
//!
//! - Gestures: AI Object Selection drags a box or clicks, like Object Selection; a Generative
//!   Brush paints like the Remove Tool and runs its model when the stroke ends.
//! - Options bar: the model menu (installed models enabled for that use, best first; "Get
//!   Models…" opens Edit › Preferences › AI Models when there are none) and the tool's options.
//! - While a brush's model runs, what it is filling stays tinted on the canvas until the result
//!   lands (progress, time taken and time left are in the status bar and progress window).

use photocraft_ml::ModelKind;
use serde_json::{Value, json};

use crate::PhotocraftApp;
use crate::canvas::ViewXform;
use crate::state::Tool;
use crate::theme::Tokens;

/// A Generative Brush stroke whose model is running: its tint stays until the job lands.
pub(crate) struct Pending {
    job: u64,
    trail: crate::stroke_trail::Trail,
}

/// The use a tool's models are for.
pub fn kind_of(tool: Tool) -> Option<ModelKind> {
    match tool {
        Tool::AiObjectSelection => Some(ModelKind::Segmentation),
        Tool::GenerativeRemoval => Some(ModelKind::Removal),
        Tool::ReflectionRemoval => Some(ModelKind::Reflection),
        _ => None,
    }
}

fn chosen_model(app: &mut PhotocraftApp, tool: Tool) -> Option<&mut String> {
    let o = &mut app.ui.tool_options;
    match tool {
        Tool::AiObjectSelection => Some(&mut o.ai_select_model),
        Tool::GenerativeRemoval => Some(&mut o.gen_removal_model),
        Tool::ReflectionRemoval => Some(&mut o.reflection_model),
        _ => None,
    }
}

/// The models `tool` can use, best first: (id, name).
fn models_for(app: &PhotocraftApp, tool: Tool) -> Vec<(String, String)> {
    let (Some(kind), Some(m)) = (kind_of(tool), app.services.models.as_ref()) else { return Vec::new() };
    m.available(kind).into_iter().map(|i| (i.id, i.name)).collect()
}

/// `"model"` for the command: the one picked in the options bar if it's still there.
fn model_param(app: &mut PhotocraftApp, tool: Tool) -> Value {
    let available = models_for(app, tool);
    let Some(picked) = chosen_model(app, tool) else { return Value::Null };
    if available.iter().any(|(id, _)| id == picked) {
        json!(picked.clone())
    } else {
        // Deleted or disabled since: the best one there is.
        picked.clear();
        Value::Null
    }
}

/// Says what's missing when a tool has no model, and offers the AI Models tab.
fn no_models(app: &mut PhotocraftApp, tool: Tool) -> bool {
    if !models_for(app, tool).is_empty() {
        return false;
    }
    app.ui.status = if app.services.models.is_none() {
        tl!("AI models run in the desktop app. Download PhotoCraft for your computer to use them.").to_string()
    } else {
        let kind = kind_of(tool).map(|k| tl!(k.title()).to_string()).unwrap_or_default();
        crate::i18n::fmt(tl!("No {kind} model yet. Download one in Edit › Preferences › AI Models."), &[("kind", &kind)])
    };
    app.ui.status_error = true;
    true
}

/// AI Object Selection: the dragged box, or a click on the object (⇧ adds, ⌥ subtracts, ⇧⌥
/// intersects, otherwise replaces).
pub fn finish_ai_object_selection(app: &mut PhotocraftApp, start: [f64; 2], end: [f64; 2], mods: egui::Modifiers) {
    if no_models(app, Tool::AiObjectSelection) {
        return;
    }
    let (x, y) = (start[0].min(end[0]), start[1].min(end[1]));
    let (w, h) = ((end[0] - start[0]).abs(), (end[1] - start[1]).abs());
    let mode = match (mods.shift, mods.alt) {
        (true, true) => "intersect",
        (true, false) => "add",
        (false, true) => "subtract",
        (false, false) => "replace",
    };
    let mut p = json!({"mode": mode, "sampleAllLayers": app.ui.tool_options.sample_all_layers, "model": model_param(app, Tool::AiObjectSelection)});
    if w < 2.0 || h < 2.0 {
        p["point"] = json!([end[0], end[1]]);
    } else {
        p["rect"] = json!([x.round(), y.round(), w.round(), h.round()]);
    }
    let r = app.run("select.aiObject", p);
    crate::retouch_ui::report_smart_result(app, r, tl!("AI Object Selection found nothing there — try a box around the object"));
}

/// A Generative Brush stroke ended: fill it with the brush's model. Returns false for other tools.
pub fn finish_stroke(app: &mut PhotocraftApp, tool: Tool, points: &[[f64; 3]]) -> bool {
    let cmd = match tool {
        Tool::GenerativeRemoval => "paint.generativeRemoval",
        Tool::ReflectionRemoval => "paint.reflectionRemoval",
        _ => return false,
    };
    if no_models(app, tool) {
        app.trail = None;
        return true;
    }
    let o = &app.ui.tool_options;
    let mut p = json!({"points": points, "sampleAllLayers": o.sample_all_layers, "newLayer": o.gen_new_layer});
    p["model"] = model_param(app, tool);
    p["target"] = crate::canvas::paint_target(app);
    let trail = app.trail.take();
    match app.run(cmd, p) {
        Ok(r) => {
            if let (Some(job), Some(trail)) = (r.get("job").and_then(Value::as_u64), trail) {
                app.ai_pending = Some(Pending { job, trail });
            }
        }
        Err(e) => {
            app.ui.status = e;
            app.ui.status_error = true;
        }
    }
    true
}

/// Keeps the tint of a stroke being filled on the canvas until its job lands.
pub fn draw_pending(app: &mut PhotocraftApp, painter: &egui::Painter, xf: &ViewXform) {
    let Some(p) = app.ai_pending.as_ref() else { return };
    let running = app.session.jobs().iter().any(|j| j.id.0 == p.job && j.state == "running");
    if !running {
        app.ai_pending = None;
        return;
    }
    let Some(bounds) = app.session.active().map(|st| st.doc.bounds()) else { return };
    let doc_rect = xf.doc_rect(bounds);
    let col = Tokens::get(painter.ctx()).accent.gamma_multiply(0.45);
    if let Some(p) = app.ai_pending.as_mut() {
        p.trail.draw(painter, doc_rect, xf.flip, col);
    }
    painter.ctx().request_repaint_after(std::time::Duration::from_millis(100));
}

/// The model menu: the installed models enabled for this use, best first. With none, a button
/// to the AI Models tab instead.
fn model_menu(app: &mut PhotocraftApp, ui: &mut egui::Ui, tool: Tool) {
    let t = Tokens::get(ui.ctx());
    let models = models_for(app, tool);
    ui.label(egui::RichText::new(tl!("Model:")).color(t.text_dim).size(12.0));
    if models.is_empty() {
        if crate::widgets::secondary_button(ui, tl!("Get Models…"), 0.0).on_hover_text(tl!("Download a model in Edit › Preferences › AI Models")).clicked()
        {
            crate::prefs_ui::open_preferences(app, "ai");
        }
        return;
    }
    let Some(picked) = chosen_model(app, tool) else { return };
    // The first is the default ("" = best available); a model that went away falls back to it.
    if !models.iter().any(|(id, _)| id == picked) {
        picked.clear();
    }
    let mut current = if picked.is_empty() { models.first().map(|(id, _)| id.clone()).unwrap_or_default() } else { picked.clone() };
    let options: Vec<(String, &str)> = models.iter().map(|(id, name)| (id.clone(), name.as_str())).collect();
    if crate::widgets::dropdown(ui, "ai-tool-model", &mut current, &options, 180.0) {
        *picked = current;
    }
    if crate::icons::button(ui, "settings", 22.0, false, tl!("AI Models…")).clicked() {
        crate::prefs_ui::open_preferences(app, "ai");
    }
}

/// Options bar for the AI tools. Returns false for other tools.
pub fn options_bar(app: &mut PhotocraftApp, ui: &mut egui::Ui, tool: Tool) -> bool {
    if kind_of(tool).is_none() {
        return false;
    }
    model_menu(app, ui, tool);
    crate::widgets::vline(ui, 22.0);
    let t = Tokens::get(ui.ctx());
    let o = &mut app.ui.tool_options;
    crate::widgets::checkbox(ui, &mut o.sample_all_layers, tl!("Sample All Layers"));
    let hint = match tool {
        Tool::AiObjectSelection => tl!("Drag a box around the object, or click it"),
        Tool::GenerativeRemoval => {
            crate::widgets::checkbox(ui, &mut o.gen_new_layer, tl!("Create a new layer"));
            tl!("Paint over what to remove")
        }
        _ => {
            crate::widgets::checkbox(ui, &mut o.gen_new_layer, tl!("Create a new layer"));
            tl!("Paint over the reflection")
        }
    };
    ui.label(egui::RichText::new(hint).color(t.text_dim).size(12.0));
    if !photocraft_ml::infer::runtime_available() {
        ui.label(egui::RichText::new(tl!("This build can't run AI models")).color(t.warning).size(12.0));
    }
    true
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
    fn each_ai_tool_has_its_use_and_the_others_none() {
        assert_eq!(kind_of(Tool::AiObjectSelection), Some(ModelKind::Segmentation));
        assert_eq!(kind_of(Tool::GenerativeRemoval), Some(ModelKind::Removal));
        assert_eq!(kind_of(Tool::ReflectionRemoval), Some(ModelKind::Reflection));
        assert_eq!(kind_of(Tool::Brush), None);
        assert!(Tool::GenerativeRemoval.is_brushlike() && Tool::ReflectionRemoval.is_brushlike());
        assert!(!Tool::AiObjectSelection.is_brushlike());
    }

    #[test]
    fn without_models_a_stroke_says_where_to_get_one_and_changes_nothing() {
        let mut app = app();
        app.ui.tool = Tool::GenerativeRemoval;
        let before = app.session.active().unwrap().doc.clone();
        assert!(finish_stroke(&mut app, Tool::GenerativeRemoval, &[[10.0, 10.0, 1.0], [20.0, 20.0, 1.0]]));
        assert!(app.ui.status_error);
        assert!(app.ui.status.contains("desktop app") || app.ui.status.contains("AI Models"), "{}", app.ui.status);
        assert!(std::sync::Arc::ptr_eq(&before, &app.session.active().unwrap().doc));
        assert!(!finish_stroke(&mut app, Tool::Brush, &[[1.0, 1.0, 1.0]]), "not an AI brush");
        finish_ai_object_selection(&mut app, [1.0, 1.0], [30.0, 30.0], egui::Modifiers::NONE);
        assert!(app.session.active().unwrap().doc.selection.is_none());
    }

    #[test]
    fn a_picked_model_that_went_away_falls_back_to_the_best() {
        let mut app = app();
        app.ui.tool_options.gen_removal_model = "gone".into();
        assert_eq!(model_param(&mut app, Tool::GenerativeRemoval), Value::Null);
        assert!(app.ui.tool_options.gen_removal_model.is_empty());
    }

    #[test]
    fn time_left_waits_for_enough_progress() {
        use crate::jobs_ui::eta_ms;
        assert_eq!(eta_ms(0.5, 10_000.0), Some(10_000.0));
        assert_eq!(eta_ms(0.01, 10_000.0), None, "too early to tell");
        assert_eq!(eta_ms(0.5, 500.0), None, "under a second in");
        assert_eq!(eta_ms(1.0, 10_000.0), None, "done");
        assert_eq!(eta_ms(f32::NAN, 10_000.0), None);
    }
}
