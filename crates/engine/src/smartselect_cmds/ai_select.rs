//! AI Object Selection: Object Selection with a local AI model (Segment Anything, or a subject
//! matting model). Drag a box around something, click it, or (the Object Selection Brush) paint
//! over it, and the model's mask becomes the selection, combined with the current one by the
//! mode, in one undo step.
//!
//! A box is analysed on its own (with a margin), so a small object gets the model's full
//! resolution; a click analyses the whole image (cached, so further clicks are quick). The
//! model's mask is scaled to the document with smooth, anti-aliased edges.

use super::*;
use photocraft_ml::ModelKind;
use photocraft_ml::infer::{self, Image, Prompt};

const CMD: &str = "select.aiObject";
const LABEL: &str = "AI Object Selection";

/// Longest side of the image the model is given (the encoders work at 1024).
const WORK: usize = 1024;

fn ai_enabled(s: &Session) -> std::result::Result<(), String> {
    crate::ai::enabled(s, ModelKind::Segmentation)
}

enum Ask {
    Box(Rect),
    Point(f32, f32),
    /// Points along a brush stroke: all on the object.
    Stroke(Vec<(f32, f32)>),
}

/// Most points a stroke gives the model (evenly spaced along it).
const STROKE_POINTS: usize = 12;

/// The area analysed: the box with room around it (the model needs context), or the canvas.
fn area_for(ask: &Ask, canvas: Rect) -> Rect {
    match ask {
        Ask::Box(r) => {
            let m = ((r.width().max(r.height()) as f32) * 0.25).max(16.0) as i32;
            r.inflate(m).intersect(&canvas)
        }
        Ask::Point(..) => canvas,
        Ask::Stroke(pts) => {
            let (mut x0, mut y0, mut x1, mut y1) = (f32::MAX, f32::MAX, f32::MIN, f32::MIN);
            for (x, y) in pts {
                (x0, y0, x1, y1) = (x0.min(*x), y0.min(*y), x1.max(*x), y1.max(*y));
            }
            // The object reaches past the stroke: plenty of room around it.
            let side = (x1 - x0).max(y1 - y0);
            let m = (side * 0.75).max(64.0);
            Rect::new((x0 - m).floor() as i32, (y0 - m).floor() as i32, (x1 + m).ceil() as i32 + 1, (y1 + m).ceil() as i32 + 1).intersect(&canvas)
        }
    }
}

/// A stroke's points (`"points": [[x, y], …]`), thinned to at most [`STROKE_POINTS`].
fn stroke_points(p: &Value) -> Result<Option<Vec<(f32, f32)>>> {
    let Some(list) = p.get("points") else { return Ok(None) };
    let arr = list.as_array().ok_or_else(|| bad(CMD, "`points` is [[x, y], …]"))?;
    let mut pts = Vec::new();
    for v in arr.iter().take(100_000) {
        let a = v.as_array().ok_or_else(|| bad(CMD, "`points` is [[x, y], …]"))?;
        let (Some(x), Some(y)) = (a.first().and_then(Value::as_f64), a.get(1).and_then(Value::as_f64)) else {
            return Err(bad(CMD, "`points` is [[x, y], …]"));
        };
        if !x.is_finite() || !y.is_finite() || x.abs() > 1e7 || y.abs() > 1e7 {
            return Err(bad(CMD, "`points` must be finite"));
        }
        pts.push((x as f32, y as f32));
    }
    if pts.is_empty() {
        return Err(bad(CMD, "`points` is empty"));
    }
    if pts.len() > STROKE_POINTS {
        let n = pts.len();
        pts = (0..STROKE_POINTS).filter_map(|i| pts.get(i * (n - 1) / (STROKE_POINTS - 1)).copied()).collect();
    }
    Ok(Some(pts))
}

fn ai_object(s: &mut Session, p: &Value) -> Result<Value> {
    let ask = match (stroke_points(p)?, rect_param(p), point_param(p)) {
        (Some(pts), ..) => Ask::Stroke(pts),
        (_, Some(r), _) if r.width() >= 2 && r.height() >= 2 => Ask::Box(r),
        (_, _, Some((x, y))) => Ask::Point(x, y),
        (_, Some(r), None) => Ask::Point(r.x0 as f32 + 0.5, r.y0 as f32 + 0.5),
        _ => return Err(bad(CMD, "needs \"rect\": [x, y, w, h], \"point\": [x, y] or \"points\": [[x, y], …] (a stroke)")),
    };
    let model = crate::ai::resolve(s, CMD, ModelKind::Segmentation, p)?;
    let all_layers = b(p, "sampleAllLayers", false);
    let m = mode(p, "replace");
    let d = s.active().ok_or(EngineError::NoDocument)?;
    let canvas = d.doc.bounds();
    let probe = match &ask {
        Ask::Box(r) => r.intersect(&canvas),
        Ask::Point(x, y) => Rect::from_xywh(x.floor() as i32, y.floor() as i32, 1, 1).intersect(&canvas),
        Ask::Stroke(pts) => pts
            .iter()
            .map(|(x, y)| Rect::from_xywh(x.floor() as i32, y.floor() as i32, 1, 1))
            .reduce(|a, b| a.union(&b))
            .unwrap_or_default()
            .intersect(&canvas),
    };
    if probe.is_empty() {
        return Err(bad(CMD, "that's outside the canvas"));
    }
    check_sampled_pixels(&d.doc, d.active_layer, all_layers, probe).map_err(|e| match e {
        EngineError::BadParams { msg, .. } => bad(CMD, &msg),
        e => e,
    })?;
    let area = area_for(&ask, canvas);
    let doc = std::sync::Arc::clone(&d.doc);
    let layer = d.active_layer;
    crate::jobs::run(
        s,
        LABEL,
        true,
        move |ctx| {
            ctx.progress(0.0, LABEL);
            find(&doc, layer, all_layers, area, &ask, &model, ctx)
        },
        move |s, region| apply(s, LABEL, region.as_ref(), m),
    )
}

/// Runs the model over `area` and returns the selection it found (`None`: nothing).
fn find(
    doc: &Document,
    layer: Option<LayerId>,
    all_layers: bool,
    area: Rect,
    ask: &Ask,
    model: &infer::Model,
    ctx: &crate::jobs::JobCtx,
) -> Result<Option<Region>> {
    let (aw, ah) = (area.width() as usize, area.height() as usize);
    if aw == 0 || ah == 0 {
        return Ok(None);
    }
    let step = aw.max(ah).div_ceil(WORK).max(1);
    let img = with_doc_sampler(doc, layer, all_layers, |smp, _| smp.rgb_scaled(area, step));
    ctx.check()?;
    let (iw, ih) = (img.w, img.h);
    let rgb: Vec<f32> = img.px.iter().flat_map(|c| c.map(|v| v.clamp(0.0, 1.0))).collect();
    let image = Image::new(iw, ih, rgb).map_err(crate::ai::error)?;
    // Document pixels to the work image's.
    let (sx, sy) = (iw as f32 / aw as f32, ih as f32 / ah as f32);
    let prompt = match ask {
        Ask::Box(r) => Prompt::Box([(r.x0 - area.x0) as f32 * sx, (r.y0 - area.y0) as f32 * sy, (r.x1 - area.x0) as f32 * sx, (r.y1 - area.y0) as f32 * sy]),
        Ask::Point(x, y) => Prompt::Points(vec![([(x - area.x0 as f32) * sx, (y - area.y0 as f32) * sy], true)]),
        Ask::Stroke(pts) => Prompt::Points(pts.iter().map(|(x, y)| ([(x - area.x0 as f32) * sx, (y - area.y0 as f32) * sy], true)).collect()),
    };
    let mut report = crate::ai::reporter(ctx, LABEL, 0.0, 0.9);
    let mask = infer::segment(model, &image, &prompt, &mut report).map_err(crate::ai::error)?;
    ctx.check()?;
    ctx.progress(0.92, LABEL);
    Ok(to_region(&mask, iw, ih, area))
}

/// The work-resolution mask (0–1) as a selection over `area`: bilinear up to document pixels,
/// with an anti-aliased edge around 0.5, cropped to what it covers.
fn to_region(mask: &[f32], iw: usize, ih: usize, area: Rect) -> Option<Region> {
    if iw == 0 || ih == 0 || mask.len() != iw * ih {
        return None;
    }
    let (aw, ah) = (area.width() as usize, area.height() as usize);
    let (sx, sy) = (iw as f32 / aw as f32, ih as f32 / ah as f32);
    // Covered part of the work mask, so only that is scaled up.
    let (mut x0, mut y0, mut x1, mut y1) = (usize::MAX, usize::MAX, 0, 0);
    for (i, v) in mask.iter().enumerate() {
        if *v > 0.4 {
            let (x, y) = (i % iw, i / iw);
            (x0, y0, x1, y1) = (x0.min(x), y0.min(y), x1.max(x), y1.max(y));
        }
    }
    if x0 == usize::MAX {
        return None;
    }
    let to_doc = |v: usize, s: f32, lim: usize| ((v as f32 / s).floor() as usize).min(lim);
    let (dx0, dy0) = (to_doc(x0.saturating_sub(1), sx, aw), to_doc(y0.saturating_sub(1), sy, ah));
    let (dx1, dy1) = (to_doc(x1 + 2, sx, aw).max(dx0 + 1), to_doc(y1 + 2, sy, ah).max(dy0 + 1));
    let (bw, bh) = (dx1 - dx0, dy1 - dy0);
    let sample = |fx: f32, fy: f32| {
        let fx = fx.clamp(0.0, (iw - 1) as f32);
        let fy = fy.clamp(0.0, (ih - 1) as f32);
        let (xa, ya) = (fx.floor() as usize, fy.floor() as usize);
        let (xb, yb) = ((xa + 1).min(iw - 1), (ya + 1).min(ih - 1));
        let (tx, ty) = (fx - xa as f32, fy - ya as f32);
        let g = |x: usize, y: usize| mask.get(y * iw + x).copied().unwrap_or(0.0);
        (g(xa, ya) * (1.0 - tx) + g(xb, ya) * tx) * (1.0 - ty) + (g(xa, yb) * (1.0 - tx) + g(xb, yb) * tx) * ty
    };
    // A soft ramp a little over a work pixel wide, in document pixels.
    let mut out = vec![0u8; bw * bh];
    let mut any = false;
    for y in 0..bh {
        let fy = ((dy0 + y) as f32 + 0.5) * sy - 0.5;
        for x in 0..bw {
            let fx = ((dx0 + x) as f32 + 0.5) * sx - 0.5;
            let v = ((sample(fx, fy) - 0.5) * 6.0 + 0.5).clamp(0.0, 1.0);
            if let Some(o) = out.get_mut(y * bw + x) {
                *o = (v * 255.0).round() as u8;
                any |= *o > 0;
            }
        }
    }
    let bbox = Rect::from_xywh(area.x0 + dx0 as i32, area.y0 + dy0 as i32, bw as u32, bh as u32);
    any.then_some(Region { bbox, mask: out })
}

pub(super) fn specs() -> Vec<CommandSpec> {
    vec![CommandSpec {
        id: CMD,
        label: LABEL,
        menu: &[],
        shortcut: None,
        params: r##"{"rect":[x,y,w,h] | "point":[x,y] | "points":[[x,y],…] (a stroke over the object: the Object Selection Brush),"model":id?=best enabled Object Selection model (see ai.models),"mode":"replace|add|subtract|intersect"="replace","sampleAllLayers":bool=false} → {"selected","changed","bounds"?} (a background job: a local AI model finds the object in the box or under the point)"##,
        enabled: ai_enabled,
        run: ai_object,
        journal: true,
    }]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn masks_scale_up_with_soft_edges_and_crop_to_what_they_cover() {
        // A 10×10 work mask with a 4×4 block, over a 100×100 area at (50, 20).
        let mut m = vec![0.0f32; 100];
        for y in 3..7 {
            for x in 3..7 {
                m[y * 10 + x] = 1.0;
            }
        }
        let r = to_region(&m, 10, 10, Rect::from_xywh(50, 20, 100, 100)).unwrap();
        assert!(r.bbox.x0 >= 50 + 20 && r.bbox.x0 <= 50 + 35, "{:?}", r.bbox);
        assert!(r.bbox.width() < 70);
        let at = |x: i32, y: i32| r.mask[((y - r.bbox.y0) * r.bbox.width() as i32 + (x - r.bbox.x0)) as usize];
        assert_eq!(at(100, 70), 255, "inside");
        assert!(r.mask.iter().any(|v| *v > 0 && *v < 255), "anti-aliased edge");
        assert!(to_region(&[0.0; 100], 10, 10, Rect::from_xywh(0, 0, 10, 10)).is_none());
        assert!(to_region(&[1.0; 5], 10, 10, Rect::from_xywh(0, 0, 10, 10)).is_none(), "wrong size");
    }

    #[test]
    fn boxes_get_room_around_them_and_clicks_the_canvas() {
        let canvas = Rect::from_xywh(0, 0, 1000, 800);
        let a = area_for(&Ask::Box(Rect::from_xywh(100, 100, 200, 100)), canvas);
        assert_eq!(a, Rect::from_xywh(50, 50, 300, 200));
        assert_eq!(area_for(&Ask::Box(Rect::from_xywh(0, 0, 40, 40)), canvas).x0, 0);
        assert_eq!(area_for(&Ask::Point(5.0, 5.0), canvas), canvas);
    }

    #[test]
    fn without_models_it_says_so_instead_of_failing_badly() {
        let mut s = Session::new();
        s.execute("file.new", json!({"width": 64, "height": 64})).unwrap();
        let e = s.execute("select.aiObject", json!({"rect": [4, 4, 20, 20]})).unwrap_err().to_string();
        assert!(e.contains("desktop app") || e.contains("not available"), "{e}");
        for bad in [json!({}), json!({"rect": "x"}), json!({"point": [f64::MAX, 1]}), json!({"model": 3})] {
            assert!(s.execute("select.aiObject", bad).is_err());
        }
    }
}
