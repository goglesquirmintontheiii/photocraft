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
pub(super) const CONTEXT: f64 = 2.0;
const MIN_WINDOW: i32 = 128;

/// Context for detail passes: they look at the area itself, at the highest resolution.
pub(super) const ENHANCE_CONTEXT: f64 = 1.25;

/// Removal models (LaMa) see 512 px: a removal's window is grown to that when the hole allows,
/// and is otherwise only this many times the hole, so the hole loses as little resolution as
/// possible.
const REMOVAL_CONTEXT: f64 = 1.35;
const REMOVAL_NATIVE: f64 = 512.0;

/// The context factor for a hole whose larger side is `side`, for a use.
pub(super) fn context_for(kind: ModelKind, side: u32) -> f64 {
    match kind {
        ModelKind::Enhance => ENHANCE_CONTEXT,
        ModelKind::Removal => {
            let side = f64::from(side.max(1));
            (REMOVAL_NATIVE / side).max(REMOVAL_CONTEXT)
        }
        _ => CONTEXT,
    }
}

/// The largest window side the model sees for a use (larger windows are scaled down for it).
pub(super) fn work_limit(kind: ModelKind) -> usize {
    if kind == ModelKind::Enhance { 2 * WORK } else { WORK }
}

/// A square window around `area`, `context` times its larger side, moved inside the canvas.
pub(super) fn window_around(area: Rect, canvas: Rect, context: f64) -> Rect {
    let side = ((f64::from(area.width().max(area.height())) * context) as i32).max(MIN_WINDOW);
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
    let strength = match p.get("strength") {
        None | Some(Value::Null) => d.strength,
        Some(v) => v.as_f64().filter(|g| g.is_finite() && (0.0..=1.0).contains(g)).ok_or_else(|| bad(cmd, "`strength` is 0..1"))? as f32,
    };
    Ok(Prompting { prompt: text("prompt")?, negative: text("negativePrompt")?, seed, steps, guidance, strength })
}

/// What fills a hole: the model run on the window's image (RGB 0–1) and hole (1 = fill), giving
/// the filled RGB and the seed when the model takes one.
pub(crate) type Filler = Box<dyn FnOnce(&Image, &[f32], &mut dyn FnMut(f32, &str) -> bool) -> photocraft_ml::Result<(Vec<f32>, Option<u64>)> + Send>;

/// What a model is asked to do with the hole.
pub(crate) enum Task {
    /// Fill it from what's around (Generative Removal, Reflection Removal).
    Remove,
    /// Fill it from a prompt (Infill, Generative Fill).
    Prompted(infer::diffusion::Prompting),
    /// Add detail and sharpness (Generative Enhance, Generative Sharpen).
    Enhance(infer::enhance::Enhancing),
}

/// The task for a command of `kind`, from its parameters.
pub(crate) fn task(cmd: &str, kind: ModelKind, p: &Value) -> Result<Task> {
    Ok(match kind {
        ModelKind::Inpainting => Task::Prompted(prompting(cmd, p)?),
        ModelKind::Enhance => {
            let opts = prompting(cmd, p)?;
            let detail = match p.get("detail") {
                None | Some(Value::Null) => 0.5,
                Some(v) => v.as_f64().filter(|d| d.is_finite() && (0.0..=1.0).contains(d)).ok_or_else(|| bad(cmd, "`detail` is 0..1"))? as f32,
            };
            Task::Enhance(infer::enhance::Enhancing { seed: opts.seed, detail })
        }
        _ => Task::Remove,
    })
}

/// The model's fill for a use (`kind`: where its category's models run).
pub(crate) fn filler(mut model: infer::Model, kind: ModelKind, task: Task, prepare: crate::ai::Prepare) -> Filler {
    Box::new(move |img, hole, report| {
        prepare.run(&mut model, report)?;
        infer::for_kind(kind, || match &task {
            Task::Prompted(opts) => infer::diffusion::diffuse(&model, img, hole, opts, report).map(|g| (g.rgb, Some(g.seed))),
            Task::Enhance(opts) => infer::enhance::enhance(&model, img, hole, opts, report),
            Task::Remove => infer::inpaint(&model, img, hole, report).map(|rgb| (rgb, None)),
        })
    })
}

pub(super) fn generative_enhance(s: &mut Session, p: &Value) -> Result<Value> {
    run(s, p, "paint.generativeEnhance", "Generative Enhance", ModelKind::Enhance)
}

pub(super) fn enhance_enabled(s: &Session) -> std::result::Result<(), String> {
    has_pixel_layer(s)?;
    crate::ai::enabled(s, ModelKind::Enhance)
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
    let (model, prepare) = crate::ai::resolve_prepared(s, cmd, kind, p)?;
    let fill_with = filler(model, kind, task(cmd, kind, p)?, prepare);
    let d = s.active().ok_or(EngineError::NoDocument)?;
    let canvas = d.doc.bounds();
    let area = plan.strokes.iter().map(stroke_bounds).reduce(|a, b| a.union(&b)).map(|r| r.intersect(&canvas)).unwrap_or(plan.window);
    plan.window = window_around(area, canvas, context_for(kind, area.width().max(area.height())));
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
                fill(&mut work, &plan, fill_with, &p, ctx, (cmd, label), kind)
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

const DETAIL_SEED: u64 = 0x00de_7a11;

/// Gives a removal's fill the fine texture and grain of its surroundings. The model's result is
/// right in colour and shape but soft (it is drawn at the model's size and scaled up); a
/// patch-based fill of the same hole, made only from the image's own pixels, is crisp. The fine
/// detail (above a scale tied to how much the result was scaled up) comes from the patch fill,
/// the rest from the model, and less of it where the two disagree on colour, so a patch from
/// somewhere unlike the fill adds nothing.
pub(super) fn restore_detail(filled: &mut [f32], texture: &[f32], hole: &[bool], w: usize, h: usize, up: f64) {
    let n = w * h;
    if filled.len() != n * 3 || texture.len() != n * 3 || hole.len() != n {
        return;
    }
    let sigma = (0.75 * up).max(1.2) as f32;
    // Only around the hole.
    let (mut x0, mut y0, mut x1, mut y1) = (w, h, 0, 0);
    for (i, _) in hole.iter().enumerate().filter(|(_, m)| **m) {
        let (x, y) = (i % w, i / w);
        (x0, y0, x1, y1) = (x0.min(x), y0.min(y), x1.max(x + 1), y1.max(y + 1));
    }
    if x0 >= x1 || y0 >= y1 {
        return;
    }
    let pad = (sigma * 3.0).ceil() as usize + 1;
    let (x0, y0, x1, y1) = (x0.saturating_sub(pad), y0.saturating_sub(pad), (x1 + pad).min(w), (y1 + pad).min(h));
    let (cw, chh) = (x1 - x0, y1 - y0);
    let plane = |src: &[f32], c: usize| -> Vec<f32> {
        let mut v = Vec::with_capacity(cw * chh);
        for y in y0..y1 {
            for x in x0..x1 {
                v.push(src.get((y * w + x) * 3 + c).copied().unwrap_or(0.0));
            }
        }
        v
    };
    let blur = |v: &[f32]| photocraft_algo::matting::gaussian_blur(v, cw, chh, sigma);
    let (mut lows, mut highs) = (Vec::with_capacity(3), Vec::with_capacity(3));
    let mut differ = vec![0.0f32; cw * chh];
    for c in 0..3 {
        let (f, t) = (plane(filled, c), plane(texture, c));
        let (fl, tl) = (blur(&f), blur(&t));
        for ((d, a), b) in differ.iter_mut().zip(&fl).zip(&tl) {
            *d += (a - b) * (a - b);
        }
        highs.push(t.iter().zip(&tl).map(|(a, b)| a - b).collect::<Vec<f32>>());
        lows.push(fl);
    }
    // A colour difference of about 0.1 halves the detail taken.
    const AGREE: f32 = 0.12;
    for y in y0..y1 {
        for x in x0..x1 {
            let (i, k) = (y * w + x, (y - y0) * cw + (x - x0));
            if !hole.get(i).copied().unwrap_or(false) {
                continue;
            }
            let trust = (-differ.get(k).copied().unwrap_or(0.0) / (AGREE * AGREE)).exp();
            for c in 0..3 {
                let (Some(lo), Some(hi)) = (lows.get(c).and_then(|v| v.get(k)), highs.get(c).and_then(|v| v.get(k))) else { continue };
                if let (Some(px), Some(orig)) = (filled.get(i * 3 + c).copied(), filled.get_mut(i * 3 + c)) {
                    // Soft fill: its own low frequencies plus the patch's detail; else unchanged.
                    *orig = (trust * (lo + hi) + (1.0 - trust) * px).clamp(0.0, 1.0);
                }
            }
        }
    }
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
    (cmd, label): (&str, &str),
    kind: ModelKind,
) -> Result<Option<Filled>> {
    let limit = work_limit(kind);
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
    let scale = (limit as f64 / w.max(h) as f64).min(1.0);
    let (ww, wh) = (((w as f64 * scale).round() as usize).max(1), ((h as f64 * scale).round() as usize).max(1));
    let work_rgb = infer::resize(&rgb, w, h, 3, ww, wh);
    // Scaling down mustn't shrink thin strokes away: any coverage counts.
    let work_hole: Vec<f32> = infer::resize(&holef, w, h, 1, ww, wh).into_iter().map(|v| if v > 0.01 { 1.0 } else { 0.0 }).collect();
    let image = Image::new(ww, wh, work_rgb).map_err(crate::ai::error)?;
    let restore = kind == ModelKind::Removal;
    let mut report = crate::ai::reporter(ctx, label, 0.0, if restore { 0.8 } else { 0.95 });
    let (result, seed) = fill_with(&image, &work_hole, &mut report).map_err(crate::ai::error)?;
    ctx.check()?;
    let mut filled = infer::resize(&result, ww, wh, 3, w, h);
    if restore {
        // How much larger the model's result was made, counting the model's own size.
        let up = (w.max(h) as f64 / ww.max(wh).min(REMOVAL_NATIVE as usize).max(1) as f64).max(1.0);
        let texture = ctx
            .stage(0.8, 0.97, label, |ctl| photocraft_algo::remove::remove_with(w, h, 3, &rgb, &hole, DETAIL_SEED, ctl))
            .map_err(|_| EngineError::Cancelled)?;
        restore_detail(&mut filled, &texture, &hole, w, h, up);
    }
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

#[cfg(test)]
mod tests {
    use super::*;

    /// Small holes get a window of the removal model's size (no scaling up); large ones get less
    /// context than other fills, so less is lost.
    #[test]
    fn removal_windows_lose_as_little_resolution_as_they_can() {
        let side = |hole: u32| f64::from(hole) * context_for(ModelKind::Removal, hole);
        assert!((side(100) - 512.0).abs() < 1.0, "{}", side(100));
        assert!((side(300) - 512.0).abs() < 1.0, "{}", side(300));
        assert!(side(1000) < 1000.0 * CONTEXT && side(1000) >= 1000.0 * REMOVAL_CONTEXT - 1.0);
        assert_eq!(context_for(ModelKind::Inpainting, 300), CONTEXT);
        assert_eq!(context_for(ModelKind::Enhance, 300), ENHANCE_CONTEXT);
    }

    /// A soft fill takes the crisp texture of a patch fill that agrees with it in colour, and
    /// keeps its colour; where they disagree it stays as it was. Nothing outside the hole changes.
    #[test]
    fn a_soft_fill_gets_the_surroundings_texture() {
        let (w, h) = (64, 64);
        let hole: Vec<bool> = (0..w * h).map(|i| (16..48).contains(&(i % w)) && (16..48).contains(&(i / w))).collect();
        // Texture: fine stripes around 0.5 on the left half, a very different colour on the right.
        let texture: Vec<f32> = (0..w * h)
            .flat_map(|i| {
                let v = if (i % w) < 32 { if (i % w) % 2 == 0 { 0.6 } else { 0.4 } } else { 0.95 };
                [v; 3]
            })
            .collect();
        let soft = vec![0.5f32; w * h * 3];
        let mut out = soft.clone();
        restore_detail(&mut out, &texture, &hole, w, h, 2.0);
        let at = |x: usize, y: usize| out[(y * w + x) * 3];
        // Left of the hole: striped like the texture, same average.
        let (a, b) = (at(20, 30), at(21, 30));
        assert!((a - b).abs() > 0.1, "{a} {b}");
        assert!(((a + b) / 2.0 - 0.5).abs() < 0.03, "{a} {b}");
        // Right: the patch's colour is far off, so the fill is kept.
        assert!((at(44, 30) - 0.5).abs() < 0.02, "{}", at(44, 30));
        // Outside the hole: untouched.
        assert_eq!(at(5, 5), 0.5);
        assert_eq!(at(20, 50), 0.5);
    }
}
