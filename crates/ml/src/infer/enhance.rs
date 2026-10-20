//! Detail and sharpness (the Generative Enhance brush and Generative Sharpen), with either kind
//! of model:
//!
//! - a super-resolution network (Real-ESRGAN and the like: an image in, the same image ×2 or ×4
//!   out). The area goes through it at its own resolution (a little smaller when the detail is
//!   strong) and the larger, crisper result is averaged back down to the area's size, so the
//!   network's sharp edges are kept and none of the area's own detail is lost. Seeds don't change
//!   its result.
//! - a Stable Diffusion pipeline, run lightly (an img2img pass at a low strength with a fixed
//!   "sharp, detailed" prompt): the picture keeps its shapes and gains texture; each seed gives a
//!   slightly different result.
//!
//! `detail` (0–1) is how strong the pass is. The caller blends the result into the area.

use super::diffusion::{self, Prompting};
use super::net::{Elem, Loader, Port, Tensor};
use super::{Image, Model, Report, interleave, load, planar, resize, shape_dims, single, step};
use crate::{Error, Result};

#[derive(Clone, Debug, PartialEq)]
pub struct Enhancing {
    pub seed: Option<u64>,
    /// How strong (0–1).
    pub detail: f32,
}

impl Default for Enhancing {
    fn default() -> Self {
        Enhancing { seed: None, detail: 0.5 }
    }
}

const PROMPT: &str = "sharp focus, highly detailed, crisp, high resolution photo, fine texture";
const NEGATIVE: &str = "blurry, out of focus, soft, pixelated, jpeg artifacts, noise, low resolution, smudged";

/// Enhances `img` where `area` (0–1 per pixel) is set; returns the whole image (RGB 0–1) and the
/// seed when the model takes one.
pub fn enhance(model: &Model, img: &Image, area: &[f32], opts: &Enhancing, report: Report<'_>) -> Result<(Vec<f32>, Option<u64>)> {
    enhance_with(super::net::runtime(), model, img, area, opts, report)
}

pub fn enhance_with(loader: &dyn Loader, model: &Model, img: &Image, area: &[f32], opts: &Enhancing, report: Report<'_>) -> Result<(Vec<f32>, Option<u64>)> {
    let detail = if opts.detail.is_finite() { opts.detail.clamp(0.0, 1.0) } else { 0.5 };
    if diffusion::is_pipeline(model) {
        let p = Prompting {
            prompt: PROMPT.into(),
            negative: NEGATIVE.into(),
            seed: opts.seed,
            steps: diffusion::DEFAULT_STEPS,
            guidance: 5.0,
            strength: 0.15 + 0.35 * detail,
        };
        return diffusion::diffuse_with(loader, model, img, area, &p, report).map(|g| (g.rgb, Some(g.seed)));
    }
    let up = upscale(loader, model, img, detail, report)?;
    Ok((up, None))
}

/// The network's input (an image, any or fixed size) and its scale factor, found by running it
/// once on a small picture.
fn probe(net: &dyn super::net::Network) -> Result<(Port, usize, usize)> {
    let ins = net.inputs();
    let port = ins.iter().find(|p| p.shape.len() == 4).or(ins.first()).cloned().ok_or_else(|| Error::Invalid("The model has no image input.".into()))?;
    if ins.len() > 1 {
        return Err(Error::Unsupported("This model takes more than an image, so it can't enhance (choose a super-resolution model or a Stable Diffusion model).".into()));
    }
    let fixed = |i: usize| port.shape.get(i).copied().filter(|v| *v > 0).map(|v| v as usize);
    let (pw, ph) = (fixed(3).unwrap_or(32), fixed(2).unwrap_or(32));
    let out = net.run(vec![(port.name.clone(), Tensor::f32(&[1, 3, ph as i64, pw as i64], vec![0.5; 3 * pw * ph]))])?;
    let t = out.iter().map(|(_, t)| t).find(|t| t.shape.len() == 4).ok_or_else(|| Error::Invalid("The model's result isn't an image.".into()))?;
    let d = shape_dims(t);
    let (ow, oh) = (d.get(3).copied().unwrap_or(0), d.get(2).copied().unwrap_or(0));
    let scale = (ow / pw.max(1)).max(oh / ph.max(1)).max(1);
    if d.get(1) != Some(&3) || ow % pw != 0 || oh % ph != 0 || scale > 8 {
        return Err(Error::Unsupported("This model's result isn't the picture made larger, so it can't enhance. Choose a super-resolution model (×2, ×4) or a Stable Diffusion model.".into()));
    }
    let fixed_w = fixed(3).unwrap_or(0);
    Ok((port, scale, fixed_w))
}

/// How much smaller the area is made before the network redraws it, for a detail (0–1): the
/// network sees it at its own resolution up to medium, and slightly smaller when strong, so blur a
/// few pixels wide reads as low resolution it can redraw. Never by the network's whole factor:
/// that throws away the detail the area has.
pub(crate) fn pre_shrink(detail: f32) -> f32 {
    1.0 + 1.5 * (detail - 0.5).max(0.0)
}

/// How much of the redrawn result replaces the area, for a detail (0–1).
pub(crate) fn mix(detail: f32) -> f32 {
    (0.55 + 0.6 * detail).min(1.0)
}

/// Redraws `img` larger through the network, in overlapping tiles, and brings each tile back to
/// the image's size (averaging, which keeps the network's sharp edges).
fn upscale(loader: &dyn Loader, model: &Model, img: &Image, detail: f32, report: Report<'_>) -> Result<Vec<f32>> {
    let (w, h) = (img.width, img.height);
    step(report, 0.02, "Loading the model…")?;
    let net = load(loader, single(model)?)?;
    let (port, scale, fixed) = probe(net.as_ref())?;
    let pre = pre_shrink(detail).min(scale as f32).max(1.0);
    let (sw, sh) = (((w as f32 / pre).round() as usize).clamp(1, w.max(1)), ((h as f32 / pre).round() as usize).clamp(1, h.max(1)));
    let small = if (sw, sh) == (w, h) { img.rgb.clone() } else { resize(&img.rgb, w, h, 3, sw, sh) };
    let tile = if fixed > 0 { fixed } else { 192 };
    let overlap = if fixed > 0 { (fixed / 8).max(4) } else { 16 };
    let stride = tile.saturating_sub(overlap * 2).max(8);
    // From the small grid to the image's.
    let (kx, ky) = (w as f64 / sw as f64, h as f64 / sh as f64);
    let mut acc = vec![0.0f32; w * h * 3];
    let mut weight = vec![0.0f32; w * h];
    let starts = |len: usize| -> Vec<usize> {
        if len <= tile {
            return vec![0];
        }
        let mut v: Vec<usize> = (0..).map(|i| i * stride).take_while(|s| s + tile < len).collect();
        v.push(len - tile);
        v
    };
    let (xs, ys) = (starts(sw), starts(sh));
    let total = xs.len() * ys.len();
    let mut done = 0;
    for &y0 in &ys {
        for &x0 in &xs {
            step(report, 0.05 + 0.9 * done as f32 / total as f32, &format!("Enhancing… part {} of {total}", done + 1))?;
            let (tw, th) = (tile.min(sw), tile.min(sh));
            // The tile (padded to a fixed input by repeating the edge).
            let (iw, ih) = if fixed > 0 { (tile, tile) } else { (tw, th) };
            let mut t = vec![0.0f32; iw * ih * 3];
            for y in 0..ih {
                for x in 0..iw {
                    let (sx, sy) = ((x0 + x.min(tw - 1)).min(sw - 1), (y0 + y.min(th - 1)).min(sh - 1));
                    for c in 0..3 {
                        if let (Some(d), Some(s)) = (t.get_mut((y * iw + x) * 3 + c), small.get((sy * sw + sx) * 3 + c)) {
                            *d = *s;
                        }
                    }
                }
            }
            let input = planar(&t, iw, ih, 1.0, [0.0; 3], [1.0; 3]);
            let data = if port.elem == Elem::U8 {
                Tensor::u8(&[1, 3, ih as i64, iw as i64], input.iter().map(|v| (v * 255.0).round().clamp(0.0, 255.0) as u8).collect())
            } else {
                Tensor::f32(&[1, 3, ih as i64, iw as i64], input)
            };
            let out = net.run(vec![(port.name.clone(), data)])?;
            let r = out.iter().map(|(_, t)| t).find(|t| t.shape.len() == 4).ok_or_else(|| Error::Invalid("The model's result isn't an image.".into()))?;
            let mut v = r.to_f32();
            if v.iter().any(|x| *x > 1.5) {
                v.iter_mut().for_each(|x| *x /= 255.0);
            }
            let (rw, rh) = (iw * scale, ih * scale);
            if v.len() < rw * rh * 3 {
                return Err(Error::Invalid("The model's result has an odd size.".into()));
            }
            let rgb = interleave(&v, rw, rh);
            // The tile's own part (not the padding), at the image's size.
            let (cw, chh) = (tw * scale, th * scale);
            let mut crop = Vec::with_capacity(cw * chh * 3);
            for y in 0..chh {
                crop.extend_from_slice(rgb.get((y * rw) * 3..(y * rw + cw) * 3).unwrap_or(&[]));
            }
            if crop.len() != cw * chh * 3 {
                return Err(Error::Invalid("The model's result has an odd size.".into()));
            }
            let gx0 = ((x0 as f64 * kx).round() as usize).min(w);
            let gy0 = ((y0 as f64 * ky).round() as usize).min(h);
            let gx1 = if x0 + tw >= sw { w } else { (((x0 + tw) as f64 * kx).round() as usize).min(w) };
            let gy1 = if y0 + th >= sh { h } else { (((y0 + th) as f64 * ky).round() as usize).min(h) };
            let (gw, gh) = (gx1.saturating_sub(gx0), gy1.saturating_sub(gy0));
            if gw == 0 || gh == 0 {
                done += 1;
                continue;
            }
            let back = resize(&crop, cw, chh, 3, gw, gh);
            // Blend the tile in with weights that fade over the overlap.
            let (fade_x, fade_y) = ((overlap as f64 * kx) as f32, (overlap as f64 * ky) as f32);
            let edge = |p: usize, len: usize, fade: f32, at_start: bool, at_end: bool| -> f32 {
                let a = if at_start { 1.0 } else { ((p as f32 + 0.5) / fade.max(1.0)).min(1.0) };
                let b = if at_end { 1.0 } else { (((len - p) as f32 - 0.5) / fade.max(1.0)).min(1.0) };
                a.min(b).max(1e-3)
            };
            for y in 0..gh {
                for x in 0..gw {
                    let wgt = edge(x, gw, fade_x, x0 == 0, x0 + tw >= sw) * edge(y, gh, fade_y, y0 == 0, y0 + th >= sh);
                    let i = (gy0 + y) * w + gx0 + x;
                    for c in 0..3 {
                        if let (Some(a), Some(s)) = (acc.get_mut(i * 3 + c), back.get((y * gw + x) * 3 + c)) {
                            *a += s * wgt;
                        }
                    }
                    if let Some(wv) = weight.get_mut(i) {
                        *wv += wgt;
                    }
                }
            }
            done += 1;
        }
    }
    let m = mix(detail);
    for (i, wv) in weight.iter().enumerate() {
        for c in 0..3 {
            let orig = img.rgb.get(i * 3 + c).copied().unwrap_or(0.0);
            if let Some(a) = acc.get_mut(i * 3 + c) {
                let redrawn = if *wv > 0.0 { *a / wv } else { orig };
                *a = (orig + m * (redrawn - orig)).clamp(0.0, 1.0);
            }
        }
    }
    step(report, 1.0, "")?;
    Ok(acc)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::infer::net::Network;
    use std::path::Path;
    use std::sync::Arc;

    /// A ×2 "super-resolution" network: nearest-neighbour upscaling, made a little brighter.
    struct Up2;

    impl Network for Up2 {
        fn inputs(&self) -> Vec<Port> {
            vec![Port { name: "input".into(), elem: Elem::F32, shape: vec![1, 3, -1, -1] }]
        }
        fn outputs(&self) -> Vec<Port> {
            Vec::new()
        }
        fn backend(&self) -> String {
            "Test".into()
        }
        fn run(&self, ins: Vec<(String, Tensor)>) -> Result<Vec<(String, Tensor)>> {
            let t = &ins[0].1;
            let (h, w) = (t.shape[2] as usize, t.shape[3] as usize);
            let v = t.to_f32();
            let mut out = vec![0.0f32; 3 * 4 * w * h];
            for c in 0..3 {
                for y in 0..2 * h {
                    for x in 0..2 * w {
                        out[(c * 2 * h + y) * 2 * w + x] = (v[(c * h + y / 2) * w + x / 2] + 0.1).min(1.0);
                    }
                }
            }
            Ok(vec![("output".into(), Tensor::f32(&[1, 3, 2 * h as i64, 2 * w as i64], out))])
        }
    }

    struct L;
    impl Loader for L {
        fn load(&self, _: &Path) -> Result<Arc<dyn Network>> {
            Ok(Arc::new(Up2))
        }
    }

    #[test]
    fn a_super_resolution_network_redraws_the_area_at_its_size() {
        let dir = std::env::temp_dir().join(format!("photocraft-enhance-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("up2.onnx"), b"x").unwrap();
        let model = Model { name: "up".into(), files: vec![dir.join("up2.onnx")], catalog_id: None };
        // Large enough to be done in several tiles.
        let (w, h) = (700, 500);
        let img = Image::new(w, h, vec![0.5; w * h * 3]).unwrap();
        let mut parts = 0;
        let (out, seed) = enhance_with(&L, &model, &img, &vec![1.0; w * h], &Enhancing { seed: Some(3), detail: 1.0 }, &mut |_, m| {
            if m.starts_with("Enhancing") {
                parts += 1;
            }
            true
        })
        .unwrap();
        assert_eq!(seed, None, "seeds don't change a network's result");
        assert_eq!(out.len(), w * h * 3);
        assert!(parts > 1, "{parts}");
        // Seamless: every pixel got the network's +0.1.
        assert!(out.iter().all(|v| (v - 0.6).abs() < 0.01), "{:?}", out.iter().fold((1f32, 0f32), |(a, b), v| (a.min(*v), b.max(*v))));
        let _ = std::fs::remove_dir_all(dir);
    }

    /// An identity "super-resolution" network (nearest neighbour ×2, nothing added).
    struct Same2;
    impl Network for Same2 {
        fn inputs(&self) -> Vec<Port> {
            Up2.inputs()
        }
        fn outputs(&self) -> Vec<Port> {
            Vec::new()
        }
        fn backend(&self) -> String {
            "Test".into()
        }
        fn run(&self, ins: Vec<(String, Tensor)>) -> Result<Vec<(String, Tensor)>> {
            let mut out = Up2.run(ins)?;
            if let Some((_, t)) = out.first_mut() {
                let v: Vec<f32> = t.to_f32().into_iter().map(|x| x - 0.1).collect();
                *t = Tensor::f32(&t.shape.clone(), v);
            }
            Ok(out)
        }
    }

    struct LSame;
    impl Loader for LSame {
        fn load(&self, _: &Path) -> Result<Arc<dyn Network>> {
            Ok(Arc::new(Same2))
        }
    }

    /// Passes of a network that adds nothing keep a textured area as it was: the area is never
    /// made smaller than it is at subtle and medium, so its detail isn't lost (it used to be cut
    /// to a quarter, and each pass smoothed a brick wall more).
    #[test]
    fn passes_keep_the_detail_an_area_has() {
        let dir = std::env::temp_dir().join(format!("photocraft-enhance-keep-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("same.onnx"), b"x").unwrap();
        let model = Model { name: "same".into(), files: vec![dir.join("same.onnx")], catalog_id: None };
        let (w, h) = (300, 260);
        // A fine checkerboard of "bricks" 3 px wide.
        let rgb: Vec<f32> = (0..w * h).flat_map(|i| {
            let v = if ((i % w) / 3 + (i / w) / 3) % 2 == 0 { 0.8 } else { 0.2 };
            [v, v * 0.6, v * 0.4]
        }).collect();
        let contrast = |px: &[f32]| -> f32 {
            let mut s = 0.0;
            for y in 0..h {
                for x in 1..w {
                    s += (px[(y * w + x) * 3] - px[(y * w + x - 1) * 3]).abs();
                }
            }
            s
        };
        let before = contrast(&rgb);
        let mut img = Image::new(w, h, rgb).unwrap();
        for _ in 0..3 {
            let (out, _) = enhance_with(&LSame, &model, &img, &vec![1.0; w * h], &Enhancing { seed: None, detail: 0.5 }, &mut |_, _| true).unwrap();
            img = Image::new(w, h, out).unwrap();
        }
        let after = contrast(&img.rgb);
        assert!(after > 0.8 * before, "three passes kept {:.0}% of the texture", 100.0 * after / before);
        assert!(pre_shrink(0.25) == 1.0 && pre_shrink(0.5) == 1.0 && pre_shrink(0.85) < 2.0 && mix(0.85) == 1.0);
        let _ = std::fs::remove_dir_all(dir);
    }
}
