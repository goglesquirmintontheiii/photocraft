//! The Generative Brushes: Infill (prompted), Generative Removal and Reflection Removal.
//!
//! Paint over something; when the stroke ends, a local AI model fills what was painted. The
//! hole is planned like the Remove Tool's (the stroke's footprint, what a closed stroke encloses,
//! a small margin, limited to the selection). The model sees a square window around it at up to
//! [`WORK`] pixels, so it has context on every side; its result is scaled back and written only
//! inside the hole. It runs as a background job with progress and an ETA, the document unchanged
//! until it lands, in one undo step. With `newLayer` the fill goes to a new layer above.
//!
//! Infill fills the strokes from a prompt, negative prompt and seed (all optional) with a Stable
//! Diffusion inpainting pipeline. Generative Removal uses an object-removal model (LaMa, MI-GAN).
//! Reflection Removal uses a reflection-removal model when one is installed (one image in, the
//! cleaned image out), else an object-removal model: brush over the reflection and it is filled
//! from what surrounds it.

use super::content_aware_move::{Target, replace};
use super::remove::{fill_to_new_layer, hole, plan_for, stroke_bounds};
use super::*;
use photocraft_ml::ModelKind;
use photocraft_ml::infer::{self, Image};

/// Longest side of what the model sees; larger windows are scaled down for it.
pub(super) const WORK: usize = 1024;

/// Context around the hole: the window is this many times the hole's larger side, square.
const CONTEXT: f64 = 2.0;
const MIN_WINDOW: i32 = 128;

/// A square window around `area`, `CONTEXT` times its larger side, moved inside the canvas.
pub(super) fn window_around(area: Rect, canvas: Rect) -> Rect {
    let side = ((f64::from(area.width().max(area.height())) * CONTEXT) as i32).max(MIN_WINDOW);
    let side_w = side.min(canvas.width() as i32);
    let side_h = side.min(canvas.height() as i32);
    let cx = area.x0 + area.width() as i32 / 2;
    let cy = area.y0 + area.height() as i32 / 2;
    let x0 = (cx - side_w / 2).clamp(canvas.x0, canvas.x1 - side_w);
    let y0 = (cy - side_h / 2).clamp(canvas.y0, canvas.y1 - side_h);
    // Never smaller than the stroke itself.
    Rect::new(x0, y0, x0 + side_w, y0 + side_h).union(&area).intersect(&canvas)
}

pub(super) fn generative_removal(s: &mut Session, p: &Value) -> Result<Value> {
    run(s, p, "paint.generativeRemoval", "Generative Removal", ModelKind::Removal)
}

pub(super) fn reflection_removal(s: &mut Session, p: &Value) -> Result<Value> {
    run(s, p, "paint.reflectionRemoval", "Reflection Removal", ModelKind::Reflection)
}

pub(super) fn generative_infill(s: &mut Session, p: &Value) -> Result<Value> {
    run(s, p, "paint.generativeInfill", "Infill", ModelKind::Inpainting)
}

pub(super) fn infill_enabled(s: &Session) -> std::result::Result<(), String> {
    has_pixel_layer(s)?;
    crate::ai::enabled(s, ModelKind::Inpainting)
}

/// The prompt, negative prompt, seed, steps and guidance of a prompted command.
pub(crate) fn prompting(cmd: &str, p: &Value) -> Result<infer::diffusion::Prompting> {
    use infer::diffusion::{MAX_SEED, MAX_STEPS, Prompting};
    let text = |k: &str| -> Result<String> {
        match p.get(k) {
            None | Some(Value::Null) => Ok(String::new()),
            Some(Value::String(t)) if t.chars().count() <= 2000 => Ok(t.clone()),
            Some(_) => Err(bad(cmd, format!("`{k}` is text of up to 2000 characters"))),
        }
    };
    let seed = match p.get("seed") {
        None | Some(Value::Null) => None,
        Some(v) => match v.as_u64().filter(|s| *s <= MAX_SEED) {
            Some(s) => Some(s),
            None => return Err(bad(cmd, format!("`seed` is a whole number from 0 to {MAX_SEED}, or null for a new one"))),
        },
    };
    let d = Prompting::default();
    let steps = match p.get("steps") {
        None | Some(Value::Null) => d.steps,
        Some(v) => v.as_u64().filter(|n| (1..=MAX_STEPS as u64).contains(n)).ok_or_else(|| bad(cmd, format!("`steps` is 1..{MAX_STEPS}")))? as usize,
    };
    let guidance = match p.get("guidance") {
        None | Some(Value::Null) => d.guidance,
        Some(v) => v.as_f64().filter(|g| g.is_finite() && (1.0..=30.0).contains(g)).ok_or_else(|| bad(cmd, "`guidance` is 1..30"))? as f32,
    };
    Ok(Prompting { prompt: text("prompt")?, negative: text("negativePrompt")?, seed, steps, guidance })
}

/// What fills a hole: the model run on the window's image (RGB 0–1) and hole (1 = fill), giving
/// the filled RGB and the seed when the model takes one.
pub(crate) type Filler = Box<dyn FnOnce(&Image, &[f32], &mut dyn FnMut(f32, &str) -> bool) -> photocraft_ml::Result<(Vec<f32>, Option<u64>)> + Send>;

pub(crate) fn filler(model: infer::Model, kind: ModelKind, prompting: Option<infer::diffusion::Prompting>) -> Filler {
    Box::new(move |img, hole, report| match (kind, prompting) {
        (ModelKind::Inpainting, Some(opts)) => infer::diffusion::diffuse(&model, img, hole, &opts, report).map(|g| (g.rgb, Some(g.seed))),
        _ => infer::inpaint(&model, img, hole, report).map(|rgb| (rgb, None)),
    })
}

pub(super) fn removal_enabled(s: &Session) -> std::result::Result<(), String> {
    has_pixel_layer(s)?;
    crate::ai::enabled(s, ModelKind::Removal)
}

pub(super) fn reflection_enabled(s: &Session) -> std::result::Result<(), String> {
    has_pixel_layer(s)?;
    crate::ai::enabled(s, ModelKind::Reflection)
}

fn run(s: &mut Session, p: &Value, cmd: &'static str, label: &'static str, kind: ModelKind) -> Result<Value> {
    let mut plan = plan_for(s, p, cmd)?;
    let model = crate::ai::resolve(s, cmd, kind, p)?;
    let prompts = if kind == ModelKind::Inpainting { Some(prompting(cmd, p)?) } else { None };
    let fill_with = filler(model, kind, prompts);
    let d = s.active().ok_or(EngineError::NoDocument)?;
    let canvas = d.doc.bounds();
    let area = plan.strokes.iter().map(stroke_bounds).reduce(|a, b| a.union(&b)).map(|r| r.intersect(&canvas)).unwrap_or(plan.window);
    plan.window = window_around(area, canvas);
    let doc = d.doc.clone();
    let id = plan.id;
    let new_layer = plan.new_layer;
    let p = p.clone();
    crate::jobs::run(
        s,
        label,
        true,
        {
            let p = p.clone();
            move |ctx| {
                ctx.progress(0.0, label);
                let mut work = (*doc).clone();
                fill(&mut work, &plan, fill_with, &p, ctx, cmd, label)
            }
        },
        move |s, filled| {
            let Some((out, hole, seed)) = filled else { return Ok(json!({"changed": false})) };
            if new_layer {
                let (damage, nid) = s.edit(label, |doc, active| {
                    let r = fill_to_new_layer(doc, id, &out, &hole)?;
                    *active = Some(r.1);
                    Ok(r)
                })?;
                if let Some(st) = s.active_mut() {
                    st.last_damage = Some(damage);
                }
                return Ok(json!({ "damage": damage_json(damage), "layer": nid.0, "seed": seed }));
            }
            let damage = s.edit(label, |doc, _| replace(doc, Target { id, all_layers: false }, &p, &out, &hole))?;
            if let Some(st) = s.active_mut() {
                st.last_damage = Some(damage);
            }
            Ok(json!({ "damage": damage_json(damage), "seed": seed }))
        },
    )
}

/// The filled window, the hole it fills, and the seed when the model took one.
type Filled = (Region, Vec<bool>, Option<u64>);

/// The window filled by the model, and the hole; computed on a snapshot on the job's thread.
fn fill(
    doc: &mut Document,
    plan: &super::remove::Plan,
    fill_with: Filler,
    p: &Value,
    ctx: &crate::jobs::JobCtx,
    cmd: &str,
    label: &str,
) -> Result<Option<Filled>> {
    let hole = hole(plan, doc.selection.as_ref())?;
    if !hole.iter().any(|m| *m) {
        return Err(bad(cmd, "the stroke doesn't cover anything that can change (check the selection)"));
    }
    if hole.iter().all(|m| *m) {
        return Err(bad(cmd, "nothing is left around the stroke to fill it from"));
    }
    ctx.check()?;
    let win = plan.window;
    let (img, fmt) = {
        let (surf, _) = crate::channel_cmds::target_surface(doc, plan.id, p)?;
        let fmt = surf.format();
        let img = if plan.all_layers { composite_region(doc, None, SampleLayers::All, win, fmt) } else { Region::read(surf, win) };
        (img, fmt)
    };
    if !plan.all_layers && alpha_index(&fmt).is_some_and(|a| img.data.chunks_exact(img.ch).all(|px| px.get(a).is_none_or(|v| *v <= 0.0))) {
        return Ok(None);
    }
    let (w, h) = (win.width() as usize, win.height() as usize);
    if img.ch == 0 || img.data.len() != w * h * img.ch {
        return Err(EngineError::Other("internal error: the window's pixels don't match its size".into()));
    }
    // RGB for the model, from any colour mode and depth.
    let rgba: Vec<[f32; 4]> = img.data.chunks_exact(img.ch).map(|px| to_rgba(&fmt, px)).collect();
    let rgb: Vec<f32> = rgba.iter().flat_map(|c| [c[0], c[1], c[2]].map(|v| v.clamp(0.0, 1.0))).collect();
    let holef: Vec<f32> = hole.iter().map(|m| if *m { 1.0 } else { 0.0 }).collect();
    let scale = (WORK as f64 / w.max(h) as f64).min(1.0);
    let (ww, wh) = (((w as f64 * scale).round() as usize).max(1), ((h as f64 * scale).round() as usize).max(1));
    let work_rgb = infer::resize(&rgb, w, h, 3, ww, wh);
    // Scaling down mustn't shrink thin strokes away: any coverage counts.
    let work_hole: Vec<f32> = infer::resize(&holef, w, h, 1, ww, wh).into_iter().map(|v| if v > 0.01 { 1.0 } else { 0.0 }).collect();
    let image = Image::new(ww, wh, work_rgb).map_err(crate::ai::error)?;
    let mut report = crate::ai::reporter(ctx, label, 0.0, 0.95);
    let (result, seed) = fill_with(&image, &work_hole, &mut report).map_err(crate::ai::error)?;
    ctx.check()?;
    let filled = infer::resize(&result, ww, wh, 3, w, h);
    let mut data = img.data.clone();
    for (i, (px, m)) in data.chunks_exact_mut(img.ch).zip(&hole).enumerate() {
        if !*m {
            continue;
        }
        let (Some(c), Some(orig)) = (filled.get(i * 3..i * 3 + 3), rgba.get(i)) else { continue };
        from_rgba_into(&fmt, [c[0], c[1], c[2], orig[3]], px);
    }
    clamp_samples(&fmt, &mut data);
    ctx.progress(1.0, label);
    Ok(Some((Region { rect: win, ch: img.ch, data }, hole, seed)))
}
