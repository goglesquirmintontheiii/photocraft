//! Edit › Preferences › AI Models: the local AI models PhotoCraft's generative tools use.
//!
//! A search bar and Import over one collapsible section per use (collapsed by default, opened by
//! a search that matches inside). A model that does several things is listed in each of its
//! sections: its tick box there switches that use on or off, while rename, delete and download act
//! on the model everywhere. Installed models show their size on disk; disabled ones are darker.
//! Recommended models not downloaded yet have a darker, striped row with Download; once
//! downloaded (or imported from elsewhere and recognised) they keep the stripes on the normal
//! background. Downloads and imports run in the background with progress, time taken and an ETA,
//! and continue when the dialog closes. Accounts, at the bottom, holds the Hugging Face and
//! Civitai tokens for models that need a sign-in.
//!
//! Model changes apply at once, like Photoshop's plug-in and preset managers: OK and Cancel are
//! for the dialog's other settings.

use egui::{Align, Color32, CornerRadius, Layout, Rect, RichText, Sense, Stroke, Ui, UiBuilder, pos2, vec2};
use photocraft_ml::manager::CatalogRow;
use photocraft_ml::{InstalledModel, Job, JobKind, JobState, Manager, ModelKind, format_duration, format_size};
use serde_json::{Map, Value, json};

use crate::i18n::fmt;
use crate::theme::Tokens;

/// Dialog field holding this tab's UI state (search text lives in the dialog's own search box).
const STATE: &str = "__ai";
/// Set by the tab when Import › Choose Files… is pressed; the dialog owner shows the file dialog.
pub(crate) const PICK_FILES: &str = "__aiPickFiles";

/// The model file types the import dialog offers.
pub(crate) const MODEL_EXTENSIONS: [&str; 9] = ["onnx", "safetensors", "ckpt", "pt", "pth", "bin", "gguf", "data", "onnx_data"];

const ROW_H: f32 = 46.0;

/// Takes the tab's state out of the dialog fields (a fresh one if missing or not an object);
/// [`body`] puts it back when it's done.
fn take_state(f: &mut Map<String, Value>) -> Map<String, Value> {
    match f.remove(STATE) {
        Some(Value::Object(m)) => m,
        _ => Map::new(),
    }
}

fn get_str(m: &Map<String, Value>, k: &str) -> String {
    m.get(k).and_then(Value::as_str).unwrap_or("").to_string()
}

/// The Import panel's "Use as": the uses picked, and whether they're still the automatic guess.
fn import_kinds(m: &Map<String, Value>) -> Vec<ModelKind> {
    m.get("importKinds").and_then(Value::as_array).into_iter().flatten().filter_map(Value::as_str).filter_map(ModelKind::from_id).collect()
}

fn set_import_kinds(m: &mut Map<String, Value>, kinds: &[ModelKind]) {
    m.insert("importKinds".into(), json!(kinds.iter().map(|k| k.id()).collect::<Vec<_>>()));
}

/// While "Use as" hasn't been touched, it follows what the link or file name gives away.
fn auto_kinds(m: &Map<String, Value>) -> bool {
    m.get("importKindsManual").and_then(Value::as_bool) != Some(true)
}

/// A darker shade of `c` by factor `k` (darker in both themes).
fn shade(c: Color32, k: f32) -> Color32 {
    Color32::from_rgba_unmultiplied((c.r() as f32 * k) as u8, (c.g() as f32 * k) as u8, (c.b() as f32 * k) as u8, c.a())
}

/// Recommended rows not downloaded yet.
fn darker(c: Color32) -> Color32 {
    shade(c, 0.86)
}

/// Installed models switched off in a section: a little darker than enabled ones.
fn disabled_bg(c: Color32) -> Color32 {
    shade(c, 0.9)
}

/// Diagonal stripes across `rect`: what marks a recommended model's row.
fn stripes(ui: &Ui, rect: Rect, color: Color32) {
    let p = ui.painter().with_clip_rect(rect.shrink(1.0));
    let gap = 9.0;
    let mut x = rect.left() - rect.height();
    while x < rect.right() {
        p.line_segment([pos2(x, rect.bottom()), pos2(x + rect.height(), rect.top())], Stroke::new(3.0, color));
        x += gap;
    }
}

fn progress_bar(ui: &mut Ui, fraction: Option<f32>, width: f32) {
    let t = Tokens::get(ui.ctx());
    let (rect, _) = ui.allocate_exact_size(vec2(width, 4.0), Sense::hover());
    ui.painter().rect_filled(rect, CornerRadius::same(2), t.field_border);
    // Unknown size: a moving segment.
    let f = fraction.unwrap_or_else(|| (ui.input(|i| i.time) * 0.8).fract() as f32);
    let fill = if fraction.is_some() {
        Rect::from_min_size(rect.min, vec2(rect.width() * f, rect.height()))
    } else {
        Rect::from_min_size(pos2(rect.left() + rect.width() * f * 0.75, rect.top()), vec2(rect.width() * 0.25, rect.height()))
    };
    ui.painter().rect_filled(fill, CornerRadius::same(2), t.accent);
}

/// "120 MB of 208 MB · 42 s · about 30 s left".
pub(crate) fn job_status(j: &Job) -> String {
    match &j.state {
        JobState::Resolving => tl!("Looking up the model…").to_string(),
        JobState::Running => {
            let mut s = match j.total_bytes {
                Some(t) => fmt(tl!("{done} of {total}"), &[("done", &format_size(j.done_bytes)), ("total", &format_size(t))]),
                None => format_size(j.done_bytes),
            };
            s.push_str(&format!(" · {}", format_duration(j.elapsed())));
            if let Some(eta) = j.eta() {
                s.push_str(" · ");
                s.push_str(&fmt(tl!("about {time} left"), &[("time", &format_duration(eta))]));
            }
            s
        }
        JobState::Done => fmt(tl!("Finished in {time}"), &[("time", &format_duration(j.elapsed()))]),
        JobState::Failed(e) => e.clone(),
        JobState::Cancelled => tl!("Cancelled").to_string(),
    }
}

enum RowLook {
    /// Installed, not from the catalog.
    Plain,
    /// Downloaded from the catalog: normal background, stripes.
    RecommendedInstalled,
    /// Recommended, not downloaded: darker, striped.
    Recommended,
}

/// Paints a row's background and gives a child `Ui` inside it.
fn row<R>(ui: &mut Ui, look: RowLook, disabled: bool, add: impl FnOnce(&mut Ui) -> R) -> R {
    let t = Tokens::get(ui.ctx());
    let w = ui.available_width();
    let (rect, _) = ui.allocate_exact_size(vec2(w, ROW_H), Sense::hover());
    let bg = match look {
        RowLook::Recommended => darker(t.field),
        _ if disabled => disabled_bg(t.field),
        _ => t.field,
    };
    ui.painter().rect_filled(rect, t.radius_sm, bg);
    if !matches!(look, RowLook::Plain) {
        stripes(ui, rect, t.text.gamma_multiply(0.045));
    }
    ui.painter().rect_stroke(rect, t.radius_sm, Stroke::new(1.0, t.field_border), egui::StrokeKind::Inside);
    let mut child = ui.new_child(UiBuilder::new().max_rect(rect.shrink2(vec2(10.0, 5.0))).layout(Layout::right_to_left(Align::Center)));
    add(&mut child)
}

/// Name, then a dim details line, elided to `width`. `dim` for a disabled model.
fn texts(ui: &mut Ui, name: &str, tag: Option<&str>, details: &str, width: f32, dim: bool) {
    let t = Tokens::get(ui.ctx());
    ui.vertical(|ui| {
        ui.spacing_mut().item_spacing.y = 1.0;
        ui.horizontal(|ui| {
            ui.spacing_mut().item_spacing.x = 6.0;
            let color = if dim { t.text_dim } else { t.text };
            ui.add(egui::Label::new(RichText::new(name).font(crate::theme::medium(12.5)).color(color)).truncate().selectable(false)).on_hover_text(name);
            if let Some(tag) = tag {
                ui.label(RichText::new(tag).size(10.5).color(t.accent_text));
            }
        });
        ui.set_max_width(width);
        ui.add(egui::Label::new(RichText::new(details).size(11.0).color(t.text_faint)).truncate().selectable(false)).on_hover_text(details);
    });
}

fn delete_controls(ui: &mut Ui, id: &str, st: &mut Map<String, Value>, m: &Manager, error: &mut Option<String>) {
    let confirming = st.get("confirmDelete").and_then(Value::as_str) == Some(id);
    if confirming {
        if crate::widgets::compact_button(ui, tl!("Keep"), 22.0).clicked() {
            st.remove("confirmDelete");
        }
        if crate::widgets::compact_button(ui, tl!("Delete"), 22.0).clicked() {
            if let Err(e) = m.delete(id) {
                *error = Some(e.to_string());
            }
            st.remove("confirmDelete");
        }
    } else if crate::icons::button(ui, "trash", 24.0, false, tl!("Delete this model")).clicked() {
        st.insert("confirmDelete".into(), json!(id));
    }
}

/// Rename: the name becomes a text field; Enter or leaving it saves, Esc cancels.
fn rename_field(ui: &mut Ui, model: &InstalledModel, st: &mut Map<String, Value>, m: &Manager, error: &mut Option<String>, width: f32) {
    let mut text = st.get("renameText").and_then(Value::as_str).unwrap_or(&model.name).to_string();
    let id = ui.id().with(("ai-rename", &model.id));
    let resp = crate::widgets::text_field(ui, id, &mut text, tl!("Name"), width.clamp(80.0, 320.0), false);
    if st.get("renameText").is_none() {
        resp.request_focus();
    }
    st.insert("renameText".into(), json!(text));
    let escape = ui.input(|i| i.key_pressed(egui::Key::Escape));
    if escape {
        st.remove("renaming");
        st.remove("renameText");
    } else if resp.lost_focus() {
        if let Err(e) = m.rename(&model.id, &text) {
            *error = Some(e.to_string());
        }
        st.remove("renaming");
        st.remove("renameText");
    }
}

fn installed_row(ui: &mut Ui, model: &InstalledModel, kind: ModelKind, st: &mut Map<String, Value>, m: &Manager, error: &mut Option<String>) {
    let t = Tokens::get(ui.ctx());
    let recommended = model.is_recommended();
    let enabled = model.enabled_for(kind);
    let look = if recommended { RowLook::RecommendedInstalled } else { RowLook::Plain };
    // The section matters only for the tick box; renaming shows in one section at a time.
    let renaming = st.get("renaming").and_then(Value::as_str) == Some(&format!("{}/{}", model.id, kind.id()));
    row(ui, look, !enabled, |ui| {
        delete_controls(ui, &model.id, st, m, error);
        if !renaming && crate::icons::button(ui, "pencil", 24.0, false, tl!("Rename this model")).clicked() {
            st.insert("renaming".into(), json!(format!("{}/{}", model.id, kind.id())));
            st.remove("renameText");
        }
        ui.label(RichText::new(format_size(model.size_bytes)).size(11.5).color(t.text_dim).monospace());
        ui.with_layout(Layout::left_to_right(Align::Center), |ui| {
            let (rect, resp) = ui.allocate_exact_size(vec2(14.0, 14.0), Sense::click());
            let tip = if enabled { tl!("Enabled: tools can use this model") } else { tl!("Disabled: no tool offers this model") };
            let resp = resp.on_hover_text(format!("{tip}\n{}", tl!("Ctrl+click to switch it in every section")));
            crate::widgets::check_box(ui.painter(), rect, enabled, resp.hovered(), resp.is_pointer_button_down_on(), &t);
            if resp.clicked() {
                // Ctrl (⌘ on macOS) switches every section the model is in to this one's new state.
                let everywhere = ui.input(|i| i.modifiers.command);
                let r = if everywhere { m.set_enabled_everywhere(&model.id, !enabled) } else { m.set_enabled(&model.id, kind, !enabled) };
                if let Err(e) = r {
                    *error = Some(e.to_string());
                }
            }
            let w = ui.available_width();
            if renaming {
                rename_field(ui, model, st, m, error, w - 8.0);
                return;
            }
            let mut details = model.source.describe();
            if !model.license.is_empty() {
                details = format!("{details} · {}", model.license);
            }
            // Where else it's listed, so a tick here isn't mistaken for switching it off everywhere.
            let others: Vec<&str> = model.kinds.iter().filter(|k| **k != kind).map(|k| tl!(k.title())).collect();
            if !others.is_empty() {
                details = format!("{details} · {}", fmt(tl!("Also in {sections}"), &[("sections", &others.join(", "))]));
            }
            texts(ui, &model.name, recommended.then_some(tl!("Recommended")), &details, w, !enabled);
        });
    });
}

fn recommended_row(ui: &mut Ui, r: &CatalogRow, m: &Manager, error: &mut Option<String>) {
    let t = Tokens::get(ui.ctx());
    row(ui, RowLook::Recommended, false, |ui| {
        let job = r.job.as_ref().filter(|j| j.active() || matches!(j.state, JobState::Failed(_)));
        match job {
            Some(j) if j.active() => {
                if crate::widgets::compact_button(ui, tl!("Cancel"), 22.0).clicked() {
                    m.cancel(j.id);
                }
            }
            _ => {
                if crate::widgets::compact_button(ui, tl!("Download"), 22.0).on_hover_text(r.model.homepage).clicked()
                    && let Err(e) = m.download(r.model.id)
                {
                    *error = Some(e.to_string());
                }
            }
        }
        ui.label(RichText::new(format_size(r.model.size())).size(11.5).color(t.text_dim).monospace());
        ui.with_layout(Layout::left_to_right(Align::Center), |ui| {
            // Lines up with the installed rows' tick boxes.
            ui.add_space(20.0);
            let w = ui.available_width();
            match job {
                Some(j) => {
                    ui.vertical(|ui| {
                        ui.spacing_mut().item_spacing.y = 3.0;
                        ui.add(egui::Label::new(RichText::new(r.model.name).font(crate::theme::medium(12.5)).color(t.text)).truncate().selectable(false));
                        if j.active() {
                            progress_bar(ui, j.fraction(), (w - 8.0).max(40.0));
                        }
                        let color = if matches!(j.state, JobState::Failed(_)) { t.danger } else { t.text_faint };
                        let s = job_status(j);
                        ui.add(egui::Label::new(RichText::new(&s).size(11.0).color(color)).truncate().selectable(false)).on_hover_text(&s);
                    });
                }
                None => {
                    let details = format!("{} · {}", r.model.description, r.model.license);
                    texts(ui, r.model.name, Some(tl!("Recommended")), &details, w, false);
                }
            }
        });
    });
}

/// The section header: chevron, title and counts. Returns whether the section is open.
fn header(ui: &mut Ui, kind: ModelKind, installed: usize, recommended: usize, forced_open: bool, st: &mut Map<String, Value>) -> bool {
    let t = Tokens::get(ui.ctx());
    let key = format!("open.{}", kind.id());
    let open = forced_open || st.get(&key).and_then(Value::as_bool).unwrap_or(false);
    let (rect, resp) = ui.allocate_exact_size(vec2(ui.available_width(), 26.0), Sense::click());
    if resp.hovered() {
        ui.painter().rect_filled(rect, t.radius_sm, t.hover);
    }
    let c = rect.left_center() + vec2(9.0, 0.0);
    let pts = if open {
        vec![c + vec2(-4.0, -2.0), c + vec2(4.0, -2.0), c + vec2(0.0, 3.0)]
    } else {
        vec![c + vec2(-2.0, -4.0), c + vec2(3.0, 0.0), c + vec2(-2.0, 4.0)]
    };
    ui.painter().add(egui::Shape::convex_polygon(pts, t.icon, Stroke::NONE));
    ui.painter().text(rect.left_center() + vec2(20.0, 0.0), egui::Align2::LEFT_CENTER, tl!(kind.title()), crate::theme::semibold(12.5), t.text);
    let mut counts = Vec::new();
    if installed > 0 {
        counts.push(fmt(tl!("{n} installed"), &[("n", &installed.to_string())]));
    }
    if recommended > 0 {
        counts.push(fmt(tl!("{n} to download"), &[("n", &recommended.to_string())]));
    }
    ui.painter().text(rect.right_center() - vec2(8.0, 0.0), egui::Align2::RIGHT_CENTER, counts.join(" · "), crate::theme::medium(11.0), t.text_faint);
    let resp = resp.on_hover_cursor(egui::CursorIcon::PointingHand);
    if resp.clicked() && !forced_open {
        st.insert(key, json!(!open));
    }
    open
}

fn import_panel(ui: &mut Ui, st: &mut Map<String, Value>, m: &Manager, error: &mut Option<String>) {
    let t = Tokens::get(ui.ctx());
    let frame = egui::Frame::new().fill(t.card).stroke(Stroke::new(1.0, t.card_border)).corner_radius(t.radius_sm).inner_margin(10.0);
    frame.show(ui, |ui| {
        ui.set_width((ui.available_width() - 2.0).max(0.0));
        let from_link = st.get("importFrom").and_then(Value::as_str) == Some("link");
        ui.horizontal(|ui| {
            if crate::widgets::pill_tab(ui, tl!("From a File"), !from_link).clicked() {
                st.insert("importFrom".into(), json!("file"));
            }
            if crate::widgets::pill_tab(ui, tl!("From a Link"), from_link).clicked() {
                st.insert("importFrom".into(), json!("link"));
            }
        });
        ui.add_space(6.0);
        // "Use as": one or more uses, guessed from the link as it's typed (or from the files'
        // names when they're chosen) until a use is clicked.
        let mut kinds = import_kinds(st);
        let auto = auto_kinds(st);
        ui.horizontal_wrapped(|ui| {
            ui.label(RichText::new(tl!("Use as")).color(t.text_dim));
            for k in ModelKind::ALL {
                let on = kinds.contains(&k);
                if crate::widgets::pill_tab(ui, k.title(), on).on_hover_text(tl!(k.used_by())).clicked() {
                    if on {
                        kinds.retain(|x| *x != k);
                    } else {
                        kinds.push(k);
                    }
                    kinds = photocraft_ml::kind::normalize(&kinds);
                    set_import_kinds(st, &kinds);
                    st.insert("importKindsManual".into(), json!(true));
                }
            }
        });
        let hint = if kinds.is_empty() {
            tl!("Choose one or more uses, or paste a link to guess them.")
        } else if auto {
            tl!("Guessed from the link. Click a use to change it.")
        } else {
            ""
        };
        if !hint.is_empty() {
            ui.label(RichText::new(hint).size(11.0).color(t.text_faint));
        }
        let mut name = get_str(st, "importName");
        ui.horizontal(|ui| {
            ui.label(RichText::new(tl!("Name")).color(t.text_dim));
            if crate::widgets::text_field(ui, ui.id().with("ai-import-name"), &mut name, tl!("Optional"), 200.0, false).changed() {
                st.insert("importName".into(), json!(name));
            }
        });
        ui.add_space(4.0);
        let name = (!name.trim().is_empty()).then(|| name.trim().to_string());
        if from_link {
            let mut link = get_str(st, "importLink");
            let w = (ui.available_width() - 90.0).max(120.0);
            ui.horizontal(|ui| {
                if crate::widgets::text_field(ui, ui.id().with("ai-import-link"), &mut link, tl!("Hugging Face, Civitai or download link"), w, false).changed()
                {
                    st.insert("importLink".into(), json!(link));
                    if auto {
                        set_import_kinds(st, &photocraft_ml::detect_kinds(&link));
                    }
                }
                if crate::widgets::primary_button(ui, tl!("Import"), 70.0).clicked() {
                    match m.import_link(&link, &import_kinds(st), name.clone()) {
                        Ok(_) => {
                            st.insert("importLink".into(), json!(""));
                            st.insert("importName".into(), json!(""));
                            st.remove("importKinds");
                            st.remove("importKindsManual");
                        }
                        Err(e) => *error = Some(e.to_string()),
                    }
                }
            });
            ui.label(
                RichText::new(tl!("Paste a Hugging Face file or repository (org/name), a Civitai model page, or a direct link to a model file."))
                    .size(11.0)
                    .color(t.text_faint),
            );
        } else {
            ui.horizontal(|ui| {
                if crate::widgets::primary_button(ui, tl!("Choose Files…"), 110.0).clicked() {
                    st.insert(PICK_FILES.into(), json!({"kinds": kinds.iter().map(|k| k.id()).collect::<Vec<_>>(), "auto": auto, "name": name}));
                }
                if crate::widgets::secondary_button(ui, tl!("Choose Folder…"), 110.0)
                    .on_hover_text(tl!("A Stable Diffusion ONNX pipeline folder (model_index.json with unet, text_encoder, vae_encoder and vae_decoder folders)"))
                    .clicked()
                {
                    st.insert(PICK_FILES.into(), json!({"kinds": kinds.iter().map(|k| k.id()).collect::<Vec<_>>(), "auto": auto, "name": name, "folder": true}));
                }
            });
            ui.horizontal(|ui| {
                ui.label(
                    RichText::new(tl!("ONNX, safetensors, checkpoint or GGUF files (pick an ONNX model and its .data file together), or a diffusion pipeline's folder."))
                        .size(11.0)
                        .color(t.text_faint),
                );
            });
        }
    });
}

/// Imports in progress or just finished (downloads of recommended models show in their rows).
fn import_jobs(ui: &mut Ui, jobs: &[Job], m: &Manager) {
    let t = Tokens::get(ui.ctx());
    for j in jobs.iter().filter(|j| j.kind == JobKind::Import) {
        let failed = matches!(j.state, JobState::Failed(_));
        row(ui, RowLook::Plain, false, |ui| {
            if j.active() {
                if crate::widgets::compact_button(ui, tl!("Cancel"), 22.0).clicked() {
                    m.cancel(j.id);
                }
            } else if crate::widgets::compact_button(ui, tl!("Dismiss"), 22.0).clicked() {
                m.dismiss(j.id);
            }
            ui.with_layout(Layout::left_to_right(Align::Center), |ui| {
                ui.add_space(20.0);
                let w = ui.available_width();
                ui.vertical(|ui| {
                    ui.spacing_mut().item_spacing.y = 3.0;
                    let title = fmt(tl!("Importing {name}"), &[("name", &j.label)]);
                    ui.add(egui::Label::new(RichText::new(&title).font(crate::theme::medium(12.5)).color(t.text)).truncate().selectable(false));
                    if j.active() {
                        progress_bar(ui, j.fraction(), (w - 8.0).max(40.0));
                    }
                    let s = job_status(j);
                    ui.add(egui::Label::new(RichText::new(&s).size(11.0).color(if failed { t.danger } else { t.text_faint })).truncate().selectable(false))
                        .on_hover_text(&s);
                });
            });
        });
        ui.add_space(4.0);
    }
}

/// Accounts: Hugging Face and Civitai tokens, for models that need a sign-in. A collapsible
/// section like the others, below them.
fn accounts(ui: &mut Ui, st: &mut Map<String, Value>, m: &Manager, error: &mut Option<String>, query: &str) {
    let t = Tokens::get(ui.ctx());
    let saved = m.tokens();
    let count = [&saved.hugging_face, &saved.civitai].iter().filter(|s| !s.is_empty()).count();
    let open = {
        let key = "open.accounts";
        let open = st.get(key).and_then(Value::as_bool).unwrap_or(false);
        let (rect, resp) = ui.allocate_exact_size(vec2(ui.available_width(), 26.0), Sense::click());
        if resp.hovered() {
            ui.painter().rect_filled(rect, t.radius_sm, t.hover);
        }
        let c = rect.left_center() + vec2(9.0, 0.0);
        let pts = if open {
            vec![c + vec2(-4.0, -2.0), c + vec2(4.0, -2.0), c + vec2(0.0, 3.0)]
        } else {
            vec![c + vec2(-2.0, -4.0), c + vec2(3.0, 0.0), c + vec2(-2.0, 4.0)]
        };
        ui.painter().add(egui::Shape::convex_polygon(pts, t.icon, Stroke::NONE));
        ui.painter().text(rect.left_center() + vec2(20.0, 0.0), egui::Align2::LEFT_CENTER, tl!("Accounts"), crate::theme::semibold(12.5), t.text);
        let summary = if count == 0 { tl!("Not signed in").to_string() } else { fmt(tl!("{n} signed in"), &[("n", &count.to_string())]) };
        ui.painter().text(rect.right_center() - vec2(8.0, 0.0), egui::Align2::RIGHT_CENTER, summary, crate::theme::medium(11.0), t.text_faint);
        if resp.on_hover_cursor(egui::CursorIcon::PointingHand).clicked() {
            st.insert(key.into(), json!(!open));
        }
        open || (!query.is_empty() && matches(query, "accounts token hugging face civitai sign in", ""))
    };
    if !open {
        return;
    }
    ui.indent("ai-accounts", |ui| {
        ui.label(RichText::new(tl!("Some models need you to sign in before they download. Paste an access token from your account; it's kept on this computer and sent only to its own site.")).size(11.0).color(t.text_faint));
        ui.add_space(4.0);
        let rows: [(&str, &str, &str, &str); 2] = [
            ("hf", "Hugging Face", "https://huggingface.co/settings/tokens", "hf_…"),
            ("civitai", "Civitai", "https://civitai.com/user/account", "API key"),
        ];
        for (key, site, page, hint) in rows {
            let stored = if key == "hf" { &saved.hugging_face } else { &saved.civitai };
            let field = format!("token.{key}");
            let mut text = st.get(&field).and_then(Value::as_str).map(str::to_string).unwrap_or_else(|| stored.clone());
            ui.horizontal(|ui| {
                ui.add_sized(vec2(96.0, 22.0), egui::Label::new(RichText::new(site).color(t.text_dim)).selectable(false));
                let w = (ui.available_width() - 150.0).clamp(120.0, 360.0);
                let show = st.get(&format!("show.{key}")).and_then(Value::as_bool).unwrap_or(false);
                let resp = if show {
                    crate::widgets::text_field(ui, ui.id().with(("ai-token", key)), &mut text, tl!(hint), w, true)
                } else {
                    password_field(ui, ui.id().with(("ai-token-hidden", key)), &mut text, tl!(hint), w)
                };
                if resp.changed() {
                    st.insert(field.clone(), json!(text));
                }
                let eye = if show { "eye-off" } else { "eye" };
                if crate::icons::button(ui, eye, 24.0, false, if show { tl!("Hide the token") } else { tl!("Show the token") }).clicked() {
                    st.insert(format!("show.{key}"), json!(!show));
                }
                let edited = photocraft_ml::accounts::clean(&text) != *stored;
                if edited && crate::widgets::compact_button(ui, tl!("Save"), 22.0).clicked() {
                    let mut tokens = saved.clone();
                    if key == "hf" {
                        tokens.hugging_face = text.clone();
                    } else {
                        tokens.civitai = text.clone();
                    }
                    match m.set_tokens(&tokens) {
                        Ok(()) => {
                            st.remove(&field);
                        }
                        Err(e) => *error = Some(e.to_string()),
                    }
                }
                if !edited {
                    ui.hyperlink_to(RichText::new(tl!("Get a token")).size(11.0), page);
                }
            });
        }
    });
}

/// A single-line field that shows dots instead of what's typed.
fn password_field(ui: &mut Ui, id: egui::Id, text: &mut String, hint: &str, width: f32) -> egui::Response {
    let t = Tokens::get(ui.ctx());
    egui::Frame::new()
        .fill(t.field)
        .stroke(Stroke::new(1.0, t.field_border))
        .corner_radius(t.radius_sm)
        .inner_margin(egui::Margin::symmetric(6, 2))
        .show(ui, |ui| ui.add(egui::TextEdit::singleline(text).id(id).password(true).hint_text(hint).frame(egui::Frame::NONE).desired_width(width - 12.0)))
        .inner
}

fn matches(query: &str, name: &str, extra: &str) -> bool {
    query.is_empty() || name.to_lowercase().contains(query) || extra.to_lowercase().contains(query)
}

/// The AI section's content. `search` is the dialog's search text.
pub(crate) fn body(ui: &mut Ui, f: &mut Map<String, Value>, models: Option<&Manager>) {
    let t = Tokens::get(ui.ctx());
    let Some(m) = models else {
        ui.label(RichText::new(tl!("AI models run in the desktop app. Download PhotoCraft for your computer to use them.")).color(t.text_faint));
        return;
    };
    let query = f.get("__search").and_then(Value::as_str).unwrap_or("").trim().to_lowercase();
    let mut state = take_state(f);
    let st = &mut state;
    let mut error: Option<String> = st.get("error").and_then(Value::as_str).map(str::to_string);
    if let Some(w) = m.warning() {
        ui.label(RichText::new(w).size(11.5).color(t.warning));
    }
    // Model search and Import.
    let mut find = get_str(st, "find");
    ui.horizontal(|ui| {
        let w = (ui.available_width() - 96.0).max(120.0);
        if crate::widgets::text_field(ui, ui.id().with("ai-model-search"), &mut find, tl!("Search models by name…"), w, false).changed() {
            st.insert("find".into(), json!(find));
        }
        let open = st.get("importOpen").and_then(Value::as_bool).unwrap_or(false);
        if crate::widgets::secondary_button(ui, tl!("Import…"), 88.0).on_hover_text(tl!("Add a model from a file or a link")).clicked() {
            st.insert("importOpen".into(), json!(!open));
        }
    });
    let find = find.trim().to_lowercase();
    let query = if find.is_empty() { query } else { find };
    ui.add_space(6.0);
    if st.get("importOpen").and_then(Value::as_bool).unwrap_or(false) {
        import_panel(ui, st, m, &mut error);
        ui.add_space(6.0);
    }
    let jobs = m.jobs();
    import_jobs(ui, &jobs, m);
    if let Some(e) = error.clone() {
        ui.horizontal(|ui| {
            ui.add(egui::Label::new(RichText::new(&e).size(11.5).color(t.danger)).wrap());
            if crate::widgets::compact_button(ui, tl!("OK"), 20.0).clicked() {
                error = None;
            }
        });
        ui.add_space(4.0);
    }
    let installed = m.installed();
    let catalog = m.catalog();
    let mut shown_any = false;
    for kind in ModelKind::ALL {
        let rec_installed: Vec<&InstalledModel> =
            installed.iter().filter(|i| i.uses(kind) && i.is_recommended()).filter(|i| matches(&query, &i.name, i.license.as_str())).collect();
        let others: Vec<&InstalledModel> = installed.iter().filter(|i| i.uses(kind) && !i.is_recommended()).filter(|i| matches(&query, &i.name, "")).collect();
        let to_download: Vec<&CatalogRow> = catalog
            .iter()
            .filter(|r| r.model.kinds.contains(&kind) && r.installed.is_none())
            .filter(|r| matches(&query, r.model.name, r.model.description))
            .collect();
        let total_here = rec_installed.len() + others.len() + to_download.len();
        // Every kind a tool uses keeps its section, even empty, so it's clear where an import
        // goes; "Other" appears only when something was imported there. A search hides
        // sections with no match.
        if total_here == 0 && (kind == ModelKind::Other || !query.is_empty()) {
            continue;
        }
        shown_any = true;
        let open = header(ui, kind, rec_installed.len() + others.len(), to_download.len(), !query.is_empty(), st);
        if open {
            ui.indent(("ai-section", kind.id()), |ui| {
                ui.label(RichText::new(tl!(kind.used_by())).size(11.0).color(t.text_faint));
                ui.add_space(2.0);
                for model in others {
                    installed_row(ui, model, kind, st, m, &mut error);
                    ui.add_space(4.0);
                }
                for model in rec_installed {
                    installed_row(ui, model, kind, st, m, &mut error);
                    ui.add_space(4.0);
                }
                for r in to_download {
                    recommended_row(ui, r, m, &mut error);
                    ui.add_space(4.0);
                }
                if total_here == 0 {
                    ui.label(RichText::new(tl!("No models yet. Use Import… to add one.")).size(11.5).color(t.text_dim));
                }
            });
        }
        ui.add_space(2.0);
    }
    if !shown_any {
        ui.label(
            RichText::new(if query.is_empty() { tl!("No models yet. Use Import… to add one.") } else { tl!("No models match your search.") })
                .color(t.text_faint),
        );
    }
    ui.add_space(6.0);
    accounts(ui, st, m, &mut error, &query);
    ui.add_space(8.0);
    ui.label(RichText::new(fmt(tl!("Models are stored in {path}"), &[("path", &m.dir().display().to_string())])).size(11.0).color(t.text_faint));
    match error {
        Some(e) => st.insert("error".into(), json!(e)),
        None => st.remove("error"),
    };
    if m.any_active() {
        ui.ctx().request_repaint_after(std::time::Duration::from_millis(150));
    }
    // Choose Files… is answered by the dialog's owner, which has the file dialog.
    if let Some(pick) = state.remove(PICK_FILES) {
        f.insert(PICK_FILES.into(), pick);
    }
    f.insert(STATE.into(), Value::Object(state));
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn darker_is_darker_and_keeps_alpha() {
        let c = darker(Color32::from_rgba_unmultiplied(200, 100, 50, 255));
        assert!(c.r() < 200 && c.g() < 100 && c.b() < 50);
        assert_eq!(c.a(), 255);
        assert_eq!(darker(Color32::BLACK), Color32::BLACK);
    }

    #[test]
    fn the_tab_state_recovers_from_a_wrong_type() {
        let mut f = Map::new();
        f.insert(STATE.into(), json!(42));
        assert!(take_state(&mut f).is_empty());
        f.insert(STATE.into(), json!({"find": "sam"}));
        assert_eq!(take_state(&mut f).get("find"), Some(&json!("sam")));
    }

    #[test]
    fn search_matches_name_or_details_case_insensitively() {
        assert!(matches("", "x", ""));
        assert!(matches("lama", "Big LaMa", ""));
        assert!(matches("apache", "x", "Apache-2.0"));
        assert!(!matches("sdxl", "LaMa", "removal"));
    }
}
