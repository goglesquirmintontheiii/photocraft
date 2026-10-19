//! Edit › Generative Fill: fills the selection from a prompt with a local diffusion model, on a
//! new layer named after the prompt (the selection is its edge, soft where the selection is).
//!
//! `edit.generativeFill` generates and applies in one step. The Generative Fill dialog makes
//! several variations to choose from instead: with `"preview": true` each new variation is kept
//! and shown (applied as the layer, one undo step that the next one replaces), and
//! `edit.generativeFill.show` shows another; `.commit` keeps the one shown, `.cancel` takes it
//! back. Every variation is generated from the document as it was before any was shown, so they
//! never build on each other.

use super::generative::{WORK, filler, prompting, window_around};
use super::patch::{coverage, selection_area};
use super::*;
use photocraft_doc::{DocId, Layer};
use std::sync::Arc;
use photocraft_ml::ModelKind;
use photocraft_ml::infer::Image;

const CMD: &str = "edit.generativeFill";
const LABEL: &str = "Generative Fill";

/// One generated fill: the new layer's pixels over `rect` (with alpha), its seed, and a small
/// picture of the window with it in place (RGBA, 8-bit) for the dialog's list.
#[derive(Clone)]
pub(crate) struct Variation {
    seed: u64,
    region: Region,
    thumb: Thumb,
}

/// A variation's picture: width, height and RGBA bytes.
pub type Thumb = (usize, usize, Arc<Vec<u8>>);

/// Longest side of a variation's picture.
const THUMB: usize = 96;

impl Session {
    /// Edit › Generative Fill's variations on the active document, in order: each one's seed
    /// and picture. Empty when the dialog has none (or the document changed since).
    pub fn generative_fill_previews(&self) -> Vec<(u64, Thumb)> {
        current(self).map(|g| g.variations.iter().map(|v| (v.seed, v.thumb.clone())).collect()).unwrap_or_default()
    }

    /// The Generative Fill variation shown on the active document, if any.
    pub fn generative_fill_shown(&self) -> Option<usize> {
        current(self).and_then(|g| g.shown.as_ref().map(|(i, _)| *i))
    }
}

/// The dialog's variations for one document.
pub(crate) struct State {
    doc: DocId,
    /// The document before any variation was shown: every variation is made from it.
    base: Arc<Document>,
    above: Option<LayerId>,
    name: String,
    variations: Vec<Variation>,
    /// The variation shown and the document it made (its undo step is the last one).
    shown: Option<(usize, Arc<Document>)>,
}

pub(crate) fn enabled(s: &Session) -> std::result::Result<(), String> {
    let st = s.active().ok_or("no document open")?;
    if st.doc.selection.is_none() {
        return Err("make a selection first".into());
    }
    Ok(())
}

/// The layer name: the prompt, shortened, or "Generative Fill".
fn layer_name(prompt: &str) -> String {
    let p = prompt.trim();
    if p.is_empty() {
        return LABEL.to_string();
    }
    let short: String = p.chars().take(40).collect();
    if short.chars().count() < p.chars().count() { format!("{}…", short.trim_end()) } else { short }
}

/// This document's dialog state, if it is still current (nothing else changed the document).
fn current(s: &Session) -> Option<&State> {
    let st = s.active()?;
    let g = s.generative_fill.as_ref()?;
    let fresh = g.doc == st.doc.id && (Arc::ptr_eq(&g.base, &st.doc) || g.shown.as_ref().is_some_and(|(_, d)| Arc::ptr_eq(d, &st.doc)));
    fresh.then_some(g)
}

/// Generates the fill of `doc`'s selection: the layer pixels and the seed.
fn generate(doc: &Document, fill_with: super::generative::Filler, ctx: &crate::jobs::JobCtx) -> Result<Variation> {
    let sel = doc.selection.as_ref().ok_or_else(|| bad(CMD, "make a selection first"))?;
    let canvas = doc.bounds();
    let area = selection_area(sel, canvas);
    if area.is_empty() {
        return Err(bad(CMD, "the selection is empty"));
    }
    let win = window_around(area, canvas);
    let alpha = coverage(sel, win);
    if !alpha.iter().any(|a| *a > 0.0) {
        return Err(bad(CMD, "the selection is empty"));
    }
    if alpha.iter().all(|a| *a >= 1.0) {
        return Err(bad(CMD, "the whole picture is selected: nothing is left around it to fill it from"));
    }
    let base = doc.pixel_format();
    let fmt = PixelFormat::new(base.mode, base.sample, false);
    let img = composite_region(doc, None, SampleLayers::All, win, fmt);
    let (w, h) = (win.width() as usize, win.height() as usize);
    if img.ch == 0 || img.data.len() != w * h * img.ch {
        return Err(EngineError::Other("internal error: the window's pixels don't match its size".into()));
    }
    ctx.check()?;
    let rgb: Vec<f32> = img.data.chunks_exact(img.ch).flat_map(|px| {
        let c = to_rgba(&fmt, px);
        [c[0], c[1], c[2]].map(|v| v.clamp(0.0, 1.0))
    }).collect();
    let hole: Vec<f32> = alpha.iter().map(|a| if *a > 0.01 { 1.0 } else { 0.0 }).collect();
    let scale = (WORK as f64 / w.max(h) as f64).min(1.0);
    let (ww, wh) = (((w as f64 * scale).round() as usize).max(1), ((h as f64 * scale).round() as usize).max(1));
    let image = Image::new(ww, wh, infer_resize(&rgb, w, h, 3, ww, wh)).map_err(crate::ai::error)?;
    let work_hole: Vec<f32> = infer_resize(&hole, w, h, 1, ww, wh).into_iter().map(|v| if v > 0.01 { 1.0 } else { 0.0 }).collect();
    let mut report = crate::ai::reporter(ctx, LABEL, 0.0, 0.97);
    let (filled, seed) = fill_with(&image, &work_hole, &mut report).map_err(crate::ai::error)?;
    ctx.check()?;
    let filled = infer_resize(&filled, ww, wh, 3, w, h);
    let lfmt = PixelFormat::new(base.mode, base.sample, true);
    let n = lfmt.channels();
    let mut data = vec![0.0f32; w * h * n];
    for (i, (px, a)) in data.chunks_exact_mut(n).zip(&alpha).enumerate() {
        if *a <= 0.0 {
            continue;
        }
        let Some(c) = filled.get(i * 3..i * 3 + 3) else { continue };
        from_rgba_into(&lfmt, [c[0], c[1], c[2], *a], px);
    }
    clamp_samples(&lfmt, &mut data);
    // The picture: the fill over what was there, scaled to fit THUMB.
    let mixed: Vec<f32> = alpha
        .iter()
        .enumerate()
        .flat_map(|(i, a)| (0..3).map(move |c| (i, c, *a)))
        .map(|(i, c, a)| {
            let o = rgb.get(i * 3 + c).copied().unwrap_or(0.0);
            let f = filled.get(i * 3 + c).copied().unwrap_or(o);
            f * a + o * (1.0 - a)
        })
        .collect();
    let ts = (THUMB as f64 / w.max(h) as f64).min(1.0);
    let (tw, th) = (((w as f64 * ts).round() as usize).max(1), ((h as f64 * ts).round() as usize).max(1));
    let small = infer_resize(&mixed, w, h, 3, tw, th);
    let bytes: Vec<u8> = small.chunks_exact(3).flat_map(|c| [c[0], c[1], c[2], 1.0].map(|v| (v.clamp(0.0, 1.0) * 255.0).round() as u8)).collect();
    ctx.progress(1.0, LABEL);
    Ok(Variation { seed: seed.unwrap_or(0), region: Region { rect: win, ch: n, data }, thumb: (tw, th, Arc::new(bytes)) })
}

fn infer_resize(src: &[f32], sw: usize, sh: usize, ch: usize, dw: usize, dh: usize) -> Vec<f32> {
    photocraft_ml::infer::resize(src, sw, sh, ch, dw, dh)
}

/// Adds `v` as a new layer above `above`; returns its id and the area it covers.
fn add_layer(doc: &mut Document, above: Option<LayerId>, name: &str, v: &Variation) -> Result<(LayerId, Rect)> {
    let base = doc.pixel_format();
    let lfmt = PixelFormat::new(base.mode, base.sample, true);
    if v.region.ch != lfmt.channels() {
        return Err(EngineError::Other("the document's colour mode changed since the fill was made".into()));
    }
    let mut l = Layer::raster(name.to_string(), lfmt);
    crate::pixels_mut(&mut l)?.write_region(v.region.rect, &v.region.data);
    let id = doc.insert_above(above.filter(|a| doc.layer(*a).is_some()), l);
    Ok((id, v.region.rect))
}

pub(super) fn generative_fill(s: &mut Session, p: &Value) -> Result<Value> {
    let st = s.active().ok_or(EngineError::NoDocument)?;
    if st.doc.selection.is_none() {
        return Err(bad(CMD, "make a selection first"));
    }
    let preview = p.get("preview").and_then(Value::as_bool).unwrap_or(false);
    let opts = prompting(CMD, p)?;
    let model = crate::ai::resolve(s, CMD, ModelKind::Inpainting, p)?;
    let name = layer_name(&opts.prompt);
    let fill_with = filler(model, ModelKind::Inpainting, Some(opts));
    // Variations are all made from the document as it was before the first was shown.
    let (base, above) = match current(s).filter(|_| preview) {
        Some(g) => (g.base.clone(), g.above),
        None => {
            let st = s.active().ok_or(EngineError::NoDocument)?;
            (st.doc.clone(), st.active_layer)
        }
    };
    crate::jobs::run(
        s,
        LABEL,
        true,
        {
            let base = base.clone();
            move |ctx| {
                ctx.progress(0.0, LABEL);
                generate(&base, fill_with, ctx)
            }
        },
        move |s, v| {
            if !preview {
                let (id, rect) = s.edit(LABEL, |doc, active| {
                    let r = add_layer(doc, above, &name, &v)?;
                    *active = Some(r.0);
                    Ok(r)
                })?;
                if let Some(st) = s.active_mut() {
                    st.last_damage = Some(rect);
                }
                return Ok(json!({"layer": id.0, "seed": v.seed, "damage": damage_json(rect)}));
            }
            let doc_id = s.active().map(|st| st.doc.id).ok_or(EngineError::NoDocument)?;
            if current(s).is_none() {
                s.generative_fill = Some(State { doc: doc_id, base, above, name: name.clone(), variations: Vec::new(), shown: None });
            }
            let g = s.generative_fill.as_mut().ok_or_else(|| EngineError::Other("internal error: no Generative Fill state".into()))?;
            g.name = name;
            g.variations.push(v);
            let i = g.variations.len() - 1;
            show_variation(s, i)
        },
    )
}

/// Shows variation `i`: takes back the one shown, then applies this one (one undo step).
fn show_variation(s: &mut Session, i: usize) -> Result<Value> {
    const SHOW: &str = "edit.generativeFill.show";
    let g = current(s).ok_or_else(|| bad(SHOW, "Generative Fill isn't open on this document, or the document changed since"))?;
    let v = g.variations.get(i).cloned().ok_or_else(|| bad(SHOW, format!("there are {} variations; `variation` is 0..{}", g.variations.len(), g.variations.len())))?;
    let (above, name, base) = (g.above, g.name.clone(), g.base.clone());
    let shown_doc = g.shown.as_ref().map(|(_, d)| d.clone());
    if let (Some(d), Some(st)) = (shown_doc, s.active())
        && Arc::ptr_eq(&d, &st.doc)
        && !s.undo()
    {
        return Err(EngineError::Other("the variation shown couldn't be taken back".into()));
    }
    if !s.active().is_some_and(|st| Arc::ptr_eq(&st.doc, &base)) {
        s.generative_fill = None;
        return Err(bad(SHOW, "the document changed since Generative Fill started"));
    }
    let (id, rect) = s.edit(LABEL, |doc, active| {
        let r = add_layer(doc, above, &name, &v)?;
        *active = Some(r.0);
        Ok(r)
    })?;
    let doc = s.active_mut().map(|st| {
        st.last_damage = Some(rect);
        st.doc.clone()
    });
    if let (Some(g), Some(doc)) = (s.generative_fill.as_mut(), doc) {
        g.shown = Some((i, doc));
    }
    Ok(json!({"variation": i, "seed": v.seed, "layer": id.0, "count": current(s).map_or(0, |g| g.variations.len())}))
}

pub(super) fn show(s: &mut Session, p: &Value) -> Result<Value> {
    let i = p.get("variation").and_then(Value::as_u64).ok_or_else(|| bad("edit.generativeFill.show", "`variation` is the variation's index"))?;
    show_variation(s, i as usize)
}

/// The dialog's variations: their seeds and which is shown.
pub(super) fn variations(s: &mut Session, _: &Value) -> Result<Value> {
    Ok(match current(s) {
        Some(g) => json!({
            "open": true,
            "variations": g.variations.iter().map(|v| json!({"seed": v.seed})).collect::<Vec<_>>(),
            "shown": g.shown.as_ref().map(|(i, _)| *i),
        }),
        None => json!({"open": false, "variations": [], "shown": null}),
    })
}

/// OK: keeps the variation shown (its undo step stays) and forgets the others.
pub(super) fn commit(s: &mut Session, _: &Value) -> Result<Value> {
    let shown = current(s).and_then(|g| g.shown.as_ref().map(|(i, _)| *i));
    s.generative_fill = None;
    Ok(json!({"kept": shown}))
}

/// Cancel: takes back the variation shown and forgets them all.
pub(super) fn cancel(s: &mut Session, _: &Value) -> Result<Value> {
    let shown_doc = current(s).and_then(|g| g.shown.as_ref().map(|(_, d)| d.clone()));
    s.generative_fill = None;
    let undone = match (shown_doc, s.active()) {
        (Some(d), Some(st)) if Arc::ptr_eq(&d, &st.doc) => s.undo(),
        _ => false,
    };
    Ok(json!({"undone": undone}))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn layers_are_named_after_the_prompt() {
        assert_eq!(layer_name(""), "Generative Fill");
        assert_eq!(layer_name("  a red kite  "), "a red kite");
        let long = "a very long prompt that goes on and on about the sky and the sea";
        let n = layer_name(long);
        assert!(n.ends_with('…') && n.chars().count() <= 41, "{n}");
    }

    #[test]
    fn without_a_model_or_a_selection_it_says_so() {
        let mut s = Session::new();
        s.execute("file.new", json!({"width": 64, "height": 64})).unwrap();
        let e = s.execute(CMD, json!({"prompt": "a cat"})).unwrap_err().to_string();
        assert!(e.contains("selection"), "{e}");
        s.execute("select.rect", json!({"x": 8, "y": 8, "width": 16, "height": 16})).unwrap();
        let e = s.execute(CMD, json!({"prompt": "a cat"})).unwrap_err().to_string();
        assert!(e.contains("desktop app") || e.contains("model"), "{e}");
        let e = s.execute(CMD, json!({"seed": -1})).unwrap_err().to_string();
        assert!(e.contains("seed"), "{e}");
        assert_eq!(s.execute("edit.generativeFill.variations", json!({})).unwrap()["open"], json!(false));
        assert_eq!(s.execute("edit.generativeFill.cancel", json!({})).unwrap()["undone"], json!(false));
        assert!(s.execute("edit.generativeFill.show", json!({"variation": 0})).is_err());
    }
}
