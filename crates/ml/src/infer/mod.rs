//! Running models: what the AI tools call.
//!
//! - [`segment`]: a mask for the object in a box or under a point (AI Object Selection). SAM 2.1
//!   (Hugging Face `onnx-community` exports), Segment Anything / MobileSAM (encoder plus
//!   `sam_mask_decoder`), or a subject-matting model run on the box.
//! - [`matte`]: subject alpha for the whole image (BiRefNet, RMBG).
//! - [`diffusion::diffuse`]: fills a hole from a prompt (the Infill brush, Generative Fill) with a
//!   Stable Diffusion ONNX pipeline.
//! - [`inpaint`]: fills a hole (Generative Removal, Reflection Removal): LaMa-style networks
//!   (image and mask in, 0–1 floats), MI-GAN (8-bit image and mask), or an image-to-image network
//!   with one input (a dedicated reflection-removal model), whose output is used inside the hole.
//!
//! The families are told apart from the model's files and its inputs' names, types and shapes,
//! so imported models of the same kinds work too. Images are interleaved RGB in 0–1, any size; the
//! caller crops and scales (the engine passes a working copy of the brushed area). Every step
//! reports progress and can be cancelled, and nothing here panics on a strange model: it returns
//! [`Error::Unsupported`](crate::Error) or [`Error::Invalid`](crate::Error) saying what didn't fit.

pub mod diffusion;
pub mod enhance;
pub mod net;
pub mod tokenizer;

use crate::{Error, Result};
use net::{Data, Elem, Loader, Network, Port, Tensor};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU8, Ordering};
use std::sync::{Arc, Mutex, PoisonError};

/// Where models run. Each choice falls back to the next one that can run a model: an NPU to
/// the graphics card, the graphics card to the processor.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum Device {
    /// The graphics card, else the processor.
    #[default]
    Auto,
    /// A neural processing unit (Intel AI Boost, AMD XDNA, Qualcomm Hexagon: through DirectML on
    /// Windows), else the graphics card, else the processor.
    Npu,
    /// The processor only.
    Cpu,
}

impl Device {
    pub const ALL: [Device; 3] = [Device::Auto, Device::Npu, Device::Cpu];

    pub fn id(self) -> &'static str {
        match self {
            Device::Auto => "auto",
            Device::Npu => "npu",
            Device::Cpu => "cpu",
        }
    }

    pub fn from_id(id: &str) -> Option<Device> {
        Device::ALL.into_iter().find(|d| d.id() == id)
    }

    pub fn title(self) -> &'static str {
        match self {
            Device::Auto => "Graphics card (GPU)",
            Device::Npu => "NPU (AI accelerator)",
            Device::Cpu => "Processor (CPU)",
        }
    }
}

static DEVICE: AtomicU8 = AtomicU8::new(0);

/// Where models run from now on (networks already loaded are loaded again).
pub fn set_device(d: Device) {
    let v = Device::ALL.iter().position(|x| *x == d).unwrap_or(0) as u8;
    if DEVICE.swap(v, Ordering::Relaxed) != v {
        net::clear_cache();
    }
}

pub fn device() -> Device {
    Device::ALL.get(usize::from(DEVICE.load(Ordering::Relaxed))).copied().unwrap_or_default()
}

/// Whether models may run on the GPU (on by default). `false` is [`Device::Cpu`].
pub fn set_gpu_allowed(on: bool) {
    set_device(if on { Device::Auto } else { Device::Cpu });
}

pub fn gpu_allowed() -> bool {
    device() != Device::Cpu
}

/// Whether this build can run models.
pub fn runtime_available() -> bool {
    net::available()
}

/// An image: interleaved RGB, 0–1.
#[derive(Clone, Debug, PartialEq)]
pub struct Image {
    pub width: usize,
    pub height: usize,
    pub rgb: Vec<f32>,
}

impl Image {
    pub fn new(width: usize, height: usize, rgb: Vec<f32>) -> Result<Image> {
        if width == 0 || height == 0 || width > 16_384 || height > 16_384 || rgb.len() != width.saturating_mul(height).saturating_mul(3) {
            return Err(Error::Invalid(format!("An image of {width}×{height} with {} values can't go to a model.", rgb.len())));
        }
        Ok(Image { width, height, rgb })
    }
}

/// What to select: a box (x0, y0, x1, y1 in image pixels) or points (x, y, inside?).
#[derive(Clone, Debug, PartialEq)]
pub enum Prompt {
    Box([f32; 4]),
    Points(Vec<([f32; 2], bool)>),
}

/// The model to run: its files on disk, and which recommended model it is, if any.
#[derive(Clone, Debug)]
pub struct Model {
    pub name: String,
    pub files: Vec<PathBuf>,
    pub catalog_id: Option<String>,
}

/// Progress (0–1) and a short message; returning `false` cancels.
pub type Report<'a> = &'a mut dyn FnMut(f32, &str) -> bool;

fn step(report: &mut dyn FnMut(f32, &str) -> bool, f: f32, msg: &str) -> Result<()> {
    if report(f.clamp(0.0, 1.0), msg) { Ok(()) } else { Err(Error::Cancelled) }
}

/// Where the last model ran ("DirectML", "WebGPU", "CPU"), for the progress message.
pub fn last_backend() -> Option<String> {
    LAST_BACKEND.lock().unwrap_or_else(PoisonError::into_inner).clone()
}

static LAST_BACKEND: Mutex<Option<String>> = Mutex::new(None);

fn load(loader: &dyn Loader, path: &Path) -> Result<Arc<dyn Network>> {
    let n = net::cached(loader, path)?;
    *LAST_BACKEND.lock().unwrap_or_else(PoisonError::into_inner) = Some(n.backend());
    Ok(n)
}

fn onnx_files(model: &Model) -> Vec<&PathBuf> {
    model.files.iter().filter(|p| p.extension().is_some_and(|e| e.eq_ignore_ascii_case("onnx"))).collect()
}

fn file_name(p: &Path) -> String {
    p.file_name().map(|n| n.to_string_lossy().to_ascii_lowercase()).unwrap_or_default()
}

/// The one network of a single-network model (the largest `.onnx`, so a helper graph shipped
/// beside it doesn't win).
fn single(model: &Model) -> Result<&PathBuf> {
    let onnx = onnx_files(model);
    if onnx.is_empty() {
        return Err(not_onnx(model));
    }
    onnx.into_iter().max_by_key(|p| std::fs::metadata(p).map(|m| m.len()).unwrap_or(0)).ok_or_else(|| not_onnx(model))
}

fn not_onnx(model: &Model) -> Error {
    Error::Unsupported(format!(
        "{} isn't an ONNX model, so it can't run. AI Object Selection and the removal brushes run .onnx models.",
        model.name
    ))
}

// ---------------------------------------------------------------------------------------------
// Pixels

/// Bilinear resize of interleaved `ch`-channel data (pixel centres aligned). Downscaling by more
/// than 2× first averages blocks, so thin details don't alias away.
pub fn resize(src: &[f32], sw: usize, sh: usize, ch: usize, dw: usize, dh: usize) -> Vec<f32> {
    if sw == 0 || sh == 0 || dw == 0 || dh == 0 || ch == 0 || src.len() < sw * sh * ch {
        return vec![0.0; dw.saturating_mul(dh).saturating_mul(ch)];
    }
    if sw == dw && sh == dh {
        return src[..sw * sh * ch].to_vec();
    }
    let fx = sw / dw.max(1);
    let fy = sh / dh.max(1);
    if fx >= 2 || fy >= 2 {
        let (fx, fy) = (fx.max(1), fy.max(1));
        let (bw, bh) = (sw / fx, sh / fy);
        let mut b = vec![0.0f32; bw * bh * ch];
        let n = (fx * fy) as f32;
        for y in 0..bh {
            for x in 0..bw {
                for c in 0..ch {
                    let mut s = 0.0;
                    for yy in 0..fy {
                        let row = (y * fy + yy) * sw;
                        for xx in 0..fx {
                            s += src.get((row + x * fx + xx) * ch + c).copied().unwrap_or(0.0);
                        }
                    }
                    if let Some(v) = b.get_mut((y * bw + x) * ch + c) {
                        *v = s / n;
                    }
                }
            }
        }
        return resize(&b, bw, bh, ch, dw, dh);
    }
    let mut out = vec![0.0f32; dw * dh * ch];
    let sx = sw as f32 / dw as f32;
    let sy = sh as f32 / dh as f32;
    for y in 0..dh {
        let fy = ((y as f32 + 0.5) * sy - 0.5).clamp(0.0, (sh - 1) as f32);
        let y0 = fy.floor() as usize;
        let y1 = (y0 + 1).min(sh - 1);
        let ty = fy - y0 as f32;
        for x in 0..dw {
            let fx = ((x as f32 + 0.5) * sx - 0.5).clamp(0.0, (sw - 1) as f32);
            let x0 = fx.floor() as usize;
            let x1 = (x0 + 1).min(sw - 1);
            let tx = fx - x0 as f32;
            for c in 0..ch {
                let g = |xx: usize, yy: usize| src.get((yy * sw + xx) * ch + c).copied().unwrap_or(0.0);
                let top = g(x0, y0) * (1.0 - tx) + g(x1, y0) * tx;
                let bot = g(x0, y1) * (1.0 - tx) + g(x1, y1) * tx;
                if let Some(v) = out.get_mut((y * dw + x) * ch + c) {
                    *v = top * (1.0 - ty) + bot * ty;
                }
            }
        }
    }
    out
}

/// Interleaved RGB (0–1) to planar CHW, `(v * scale - mean) / std` per channel.
fn planar(rgb: &[f32], w: usize, h: usize, scale: f32, mean: [f32; 3], std: [f32; 3]) -> Vec<f32> {
    let n = w * h;
    let mut out = vec![0.0f32; n * 3];
    for i in 0..n {
        for c in 0..3 {
            let v = rgb.get(i * 3 + c).copied().unwrap_or(0.0);
            if let Some(o) = out.get_mut(c * n + i) {
                *o = (v * scale - mean[c]) / std[c];
            }
        }
    }
    out
}

/// Planar CHW back to interleaved RGB.
fn interleave(chw: &[f32], w: usize, h: usize) -> Vec<f32> {
    let n = w * h;
    let mut out = vec![0.0f32; n * 3];
    for i in 0..n {
        for c in 0..3 {
            if let Some(o) = out.get_mut(i * 3 + c) {
                *o = chw.get(c * n + i).copied().unwrap_or(0.0);
            }
        }
    }
    out
}

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

const IMAGENET_MEAN: [f32; 3] = [0.485, 0.456, 0.406];
const IMAGENET_STD: [f32; 3] = [0.229, 0.224, 0.225];

/// A fixed spatial size from a port's shape `[1, C, H, W]`, or `default` where it's dynamic.
fn spatial(p: &Port, default: usize) -> (usize, usize) {
    let d = |i: usize| p.shape.get(i).copied().filter(|v| *v > 0 && *v <= 8192).map_or(default, |v| v as usize);
    (d(3), d(2))
}

fn find<'a>(ports: &'a [Port], names: &[&str]) -> Option<&'a Port> {
    names.iter().find_map(|n| ports.iter().find(|p| p.name == *n))
}

fn output<'a>(outs: &'a [(String, Tensor)], names: &[&str]) -> Option<&'a Tensor> {
    names.iter().find_map(|n| outs.iter().find(|(k, _)| k == n).map(|(_, t)| t))
}

fn shape_dims(t: &Tensor) -> Vec<usize> {
    t.shape.iter().map(|d| (*d).max(0) as usize).collect()
}

// ---------------------------------------------------------------------------------------------
// Segmentation

/// The image embedding of the last image a selection model encoded, so clicking or dragging again
/// on the same image skips the slow encoder.
struct Embedding {
    encoder: PathBuf,
    key: u64,
    outputs: Vec<(String, Tensor)>,
}

static EMBEDDING: Mutex<Option<Embedding>> = Mutex::new(None);

fn hash_input(t: &[f32], w: usize, h: usize) -> u64 {
    use std::hash::{Hash, Hasher};
    let mut s = std::collections::hash_map::DefaultHasher::new();
    (w, h, t.len()).hash(&mut s);
    for v in t.iter().step_by(3) {
        v.to_bits().hash(&mut s);
    }
    s.finish()
}

/// A mask (0–1, one value per pixel of `img`) for what `prompt` points at.
pub fn segment(model: &Model, img: &Image, prompt: &Prompt, report: Report<'_>) -> Result<Vec<f32>> {
    segment_with(net::runtime(), model, img, prompt, report)
}

pub fn segment_with(loader: &dyn Loader, model: &Model, img: &Image, prompt: &Prompt, report: Report<'_>) -> Result<Vec<f32>> {
    let onnx = onnx_files(model);
    let encoder = onnx.iter().find(|p| {
        let n = file_name(p);
        n.contains("encoder") && !n.contains("decoder") && !n.contains("prompt")
    });
    let decoder = onnx.iter().find(|p| file_name(p).contains("decoder"));
    match (encoder, decoder) {
        (Some(e), Some(d)) => sam(loader, e, d, img, prompt, report),
        _ if onnx.len() == 1 => matte_prompt(loader, model, img, prompt, report),
        _ if onnx.is_empty() => Err(not_onnx(model)),
        _ => Err(Error::Unsupported(format!(
            "{} has several networks but no image encoder and mask decoder, so PhotoCraft can't tell how to run it for selections.",
            model.name
        ))),
    }
}

/// Segment Anything: SAM 2.1 as exported for Transformers.js (`pixel_values` → three
/// `image_embeddings.*` → `input_points` / `input_labels` / `input_boxes`), or SAM 1 and MobileSAM
/// (`input_image` → `image_embeddings` → `point_coords` / `point_labels` / `orig_im_size`).
fn sam(loader: &dyn Loader, enc_path: &Path, dec_path: &Path, img: &Image, prompt: &Prompt, report: Report<'_>) -> Result<Vec<f32>> {
    let (w, h) = (img.width, img.height);
    step(report, 0.02, "Loading the model…")?;
    let enc = load(loader, enc_path)?;
    let dec = load(loader, dec_path)?;
    let dec_in = dec.inputs();
    let sam2 = find(&dec_in, &["input_points"]).is_some();
    let enc_in = enc.inputs();
    let first = enc_in.first().ok_or_else(|| Error::Invalid("The selection model's encoder has no inputs.".into()))?;
    // The encoder's input, and the scale from image pixels to the 1024 frame.
    let (input, sx, sy) = if sam2 {
        let (iw, ih) = spatial(first, 1024);
        let px = resize(&img.rgb, w, h, 3, iw, ih);
        (Tensor::f32(&[1, 3, ih as i64, iw as i64], planar(&px, iw, ih, 1.0, IMAGENET_MEAN, IMAGENET_STD)), iw as f32 / w as f32, ih as f32 / h as f32)
    } else {
        // Longest side to 1024, keeping the aspect ratio.
        let s = 1024.0 / w.max(h) as f32;
        let (rw, rh) = (((w as f32 * s).round() as usize).clamp(1, 1024), ((h as f32 * s).round() as usize).clamp(1, 1024));
        let px = resize(&img.rgb, w, h, 3, rw, rh);
        if first.shape.len() == 3 {
            // Exported with its preprocessing: HWC, 0–255, normalised and padded inside.
            (Tensor::f32(&[rh as i64, rw as i64, 3], px.iter().map(|v| v * 255.0).collect()), s, s)
        } else {
            let mean = [123.675, 116.28, 103.53];
            let std = [58.395, 57.12, 57.375];
            let mut padded = vec![0.0f32; 3 * 1024 * 1024];
            let pl = planar(&px, rw, rh, 255.0, mean, std);
            for c in 0..3 {
                for y in 0..rh {
                    let (src, dst) = ((c * rh + y) * rw, (c * 1024 + y) * 1024);
                    if let (Some(s), Some(d)) = (pl.get(src..src + rw), padded.get_mut(dst..dst + rw)) {
                        d.copy_from_slice(s);
                    }
                }
            }
            (Tensor::f32(&[1, 3, 1024, 1024], padded), s, s)
        }
    };
    let key = match &input.data {
        Data::F32(v) => hash_input(v, w, h),
        _ => 0,
    };
    let cached = {
        let e = EMBEDDING.lock().unwrap_or_else(PoisonError::into_inner);
        e.as_ref().filter(|e| e.encoder == enc_path && e.key == key).map(|e| e.outputs.clone())
    };
    let embeddings = match cached {
        Some(o) => o,
        None => {
            step(report, 0.1, "Analysing the image…")?;
            let out = enc.run(vec![(first.name.clone(), input)])?;
            *EMBEDDING.lock().unwrap_or_else(PoisonError::into_inner) = Some(Embedding { encoder: enc_path.to_path_buf(), key, outputs: out.clone() });
            out
        }
    };
    step(report, 0.8, "Finding the object…")?;
    // Decoder inputs named like encoder outputs are passed straight through.
    let mut inputs: Vec<(String, Tensor)> = embeddings.iter().filter(|(n, _)| dec_in.iter().any(|p| &p.name == n)).cloned().collect();
    if inputs.is_empty() {
        // Exports that call the embedding something else: the decoder's one 4-D float input.
        if let (Some(p), Some((_, t))) = (dec_in.iter().find(|p| p.shape.len() == 4 && p.name.contains("embed")), embeddings.first()) {
            inputs.push((p.name.clone(), t.clone()));
        }
    }
    let mask = if sam2 {
        let (pts, labels, boxes): (Vec<f32>, Vec<i64>, Vec<f32>) = match prompt {
            Prompt::Box(b) => (Vec::new(), Vec::new(), vec![b[0] * sx, b[1] * sy, b[2] * sx, b[3] * sy]),
            Prompt::Points(p) => (p.iter().flat_map(|(q, _)| [q[0] * sx, q[1] * sy]).collect(), p.iter().map(|(_, pos)| i64::from(*pos)).collect(), Vec::new()),
        };
        let n = labels.len() as i64;
        inputs.push(("input_points".into(), Tensor::f32(&[1, 1, n, 2], pts)));
        inputs.push(("input_labels".into(), Tensor::i64(&[1, 1, n], labels)));
        if find(&dec_in, &["input_boxes"]).is_some() {
            let nb = (boxes.len() / 4) as i64;
            inputs.push(("input_boxes".into(), Tensor::f32(&[1, nb, 4], boxes)));
        }
        let out = dec.run(inputs)?;
        let masks = output(&out, &["pred_masks"])
            .or_else(|| out.iter().map(|(_, t)| t).find(|t| t.shape.len() == 5))
            .ok_or_else(|| Error::Invalid("The selection model gave no masks.".into()))?;
        let scores = output(&out, &["iou_scores"]).map(Tensor::to_f32).unwrap_or_default();
        let d = shape_dims(masks);
        let (k, mh, mw) = (d.get(2).copied().unwrap_or(1).max(1), d.get(3).copied().unwrap_or(0), d.get(4).copied().unwrap_or(0));
        let best = (0..k).max_by(|a, b| scores.get(*a).unwrap_or(&0.0).total_cmp(scores.get(*b).unwrap_or(&0.0))).unwrap_or(0);
        let all = masks.to_f32();
        let plane = all.get(best * mh * mw..(best + 1) * mh * mw).ok_or_else(|| Error::Invalid("The selection model's mask has an odd size.".into()))?;
        resize(plane, mw, mh, 1, w, h)
    } else {
        let (mut coords, mut labels): (Vec<f32>, Vec<f32>) = match prompt {
            Prompt::Box(b) => (vec![b[0] * sx, b[1] * sy, b[2] * sx, b[3] * sy], vec![2.0, 3.0]),
            Prompt::Points(p) => {
                (p.iter().flat_map(|(q, _)| [q[0] * sx, q[1] * sy]).collect(), p.iter().map(|(_, pos)| if *pos { 1.0 } else { 0.0 }).collect())
            }
        };
        if matches!(prompt, Prompt::Points(_)) {
            coords.extend([0.0, 0.0]);
            labels.push(-1.0);
        }
        let n = labels.len() as i64;
        inputs.push(("point_coords".into(), Tensor::f32(&[1, n, 2], coords)));
        inputs.push(("point_labels".into(), Tensor::f32(&[1, n], labels)));
        if find(&dec_in, &["mask_input"]).is_some() {
            inputs.push(("mask_input".into(), Tensor::f32(&[1, 1, 256, 256], vec![0.0; 256 * 256])));
        }
        if find(&dec_in, &["has_mask_input"]).is_some() {
            inputs.push(("has_mask_input".into(), Tensor::f32(&[1], vec![0.0])));
        }
        if find(&dec_in, &["orig_im_size"]).is_some() {
            inputs.push(("orig_im_size".into(), Tensor::f32(&[2], vec![h as f32, w as f32])));
        }
        let out = dec.run(inputs)?;
        let masks = output(&out, &["masks"])
            .or_else(|| out.iter().map(|(_, t)| t).find(|t| t.shape.len() == 4))
            .ok_or_else(|| Error::Invalid("The selection model gave no masks.".into()))?;
        let scores = output(&out, &["iou_predictions", "scores"]).map(Tensor::to_f32).unwrap_or_default();
        let d = shape_dims(masks);
        let (k, mh, mw) = (d.get(1).copied().unwrap_or(1).max(1), d.get(2).copied().unwrap_or(0), d.get(3).copied().unwrap_or(0));
        let best = (0..k).max_by(|a, b| scores.get(*a).unwrap_or(&0.0).total_cmp(scores.get(*b).unwrap_or(&0.0))).unwrap_or(0);
        let all = masks.to_f32();
        let plane = all.get(best * mh * mw..(best + 1) * mh * mw).ok_or_else(|| Error::Invalid("The selection model's mask has an odd size.".into()))?;
        if (mw, mh) == (w, h) {
            plane.to_vec()
        } else if mw >= 1024 || mh >= 1024 {
            // Low-res masks in the padded 1024 frame: crop the image's part, then scale.
            let (cw, ch) = (((w as f32 * sx).round() as usize).clamp(1, mw), ((h as f32 * sy).round() as usize).clamp(1, mh));
            let crop: Vec<f32> = (0..ch).flat_map(|y| plane.get(y * mw..y * mw + cw).unwrap_or(&[]).to_vec()).collect();
            resize(&crop, cw, ch, 1, w, h)
        } else {
            resize(plane, mw, mh, 1, w, h)
        }
    };
    step(report, 1.0, "")?;
    Ok(mask.into_iter().map(sigmoid).collect())
}

/// Selection with a matting model: the subject inside the box (with a little room around it), or
/// in the whole image for a click.
fn matte_prompt(loader: &dyn Loader, model: &Model, img: &Image, prompt: &Prompt, report: Report<'_>) -> Result<Vec<f32>> {
    let (w, h) = (img.width, img.height);
    let Prompt::Box(b) = prompt else { return matte_with(loader, model, img, report) };
    let pad = ((b[2] - b[0]).abs().max((b[3] - b[1]).abs()) * 0.08).max(4.0);
    let x0 = (b[0].min(b[2]) - pad).floor().clamp(0.0, w as f32 - 1.0) as usize;
    let y0 = (b[1].min(b[3]) - pad).floor().clamp(0.0, h as f32 - 1.0) as usize;
    let x1 = (b[0].max(b[2]) + pad).ceil().clamp(x0 as f32 + 1.0, w as f32) as usize;
    let y1 = (b[1].max(b[3]) + pad).ceil().clamp(y0 as f32 + 1.0, h as f32) as usize;
    let (cw, ch) = (x1 - x0, y1 - y0);
    let mut crop = Vec::with_capacity(cw * ch * 3);
    for y in y0..y1 {
        crop.extend_from_slice(img.rgb.get((y * w + x0) * 3..(y * w + x1) * 3).unwrap_or(&[]));
    }
    let crop = Image::new(cw, ch, crop)?;
    let alpha = matte_with(loader, model, &crop, report)?;
    let mut out = vec![0.0f32; w * h];
    // Only what's inside the box itself counts.
    let (bx0, by0, bx1, by1) = (b[0].min(b[2]), b[1].min(b[3]), b[0].max(b[2]), b[1].max(b[3]));
    for y in y0..y1 {
        for x in x0..x1 {
            let inside = (x as f32 + 0.5) >= bx0 && (x as f32 + 0.5) <= bx1 && (y as f32 + 0.5) >= by0 && (y as f32 + 0.5) <= by1;
            if inside && let (Some(o), Some(a)) = (out.get_mut(y * w + x), alpha.get((y - y0) * cw + (x - x0))) {
                *o = *a;
            }
        }
    }
    Ok(out)
}

// ---------------------------------------------------------------------------------------------
// Matting

/// Subject alpha (0–1) for every pixel of `img`.
pub fn matte(model: &Model, img: &Image, report: Report<'_>) -> Result<Vec<f32>> {
    matte_with(net::runtime(), model, img, report)
}

pub fn matte_with(loader: &dyn Loader, model: &Model, img: &Image, report: Report<'_>) -> Result<Vec<f32>> {
    let (w, h) = (img.width, img.height);
    step(report, 0.02, "Loading the model…")?;
    let n = load(loader, single(model)?)?;
    let ins = n.inputs();
    let input = ins.iter().find(|p| p.shape.len() == 4).or(ins.first()).ok_or_else(|| Error::Invalid("The matting model has no image input.".into()))?;
    let (iw, ih) = spatial(input, 1024);
    let px = resize(&img.rgb, w, h, 3, iw, ih);
    // BRIA RMBG centres on 0.5 with no scaling; BiRefNet and most others use ImageNet statistics.
    let rmbg = model.catalog_id.as_deref().is_some_and(|c| c.starts_with("rmbg")) || model.name.to_ascii_lowercase().contains("rmbg");
    let data = if rmbg { planar(&px, iw, ih, 1.0, [0.5; 3], [1.0; 3]) } else { planar(&px, iw, ih, 1.0, IMAGENET_MEAN, IMAGENET_STD) };
    step(report, 0.2, "Finding the subject…")?;
    let out = n.run(vec![(input.name.clone(), Tensor::f32(&[1, 3, ih as i64, iw as i64], data))])?;
    // The last output is the finest prediction in multi-output exports.
    let t = out
        .iter()
        .rev()
        .map(|(_, t)| t)
        .find(|t| t.len() == iw * ih)
        .or_else(|| out.last().map(|(_, t)| t))
        .ok_or_else(|| Error::Invalid("The matting model gave no mask.".into()))?;
    let d = shape_dims(t);
    let (mw, mh) = (d.last().copied().unwrap_or(0), d.get(d.len().saturating_sub(2)).copied().unwrap_or(0));
    let mut v = t.to_f32();
    v.truncate(mw * mh);
    if v.len() != mw * mh || v.is_empty() {
        return Err(Error::Invalid("The matting model's mask has an odd size.".into()));
    }
    let (lo, hi) = v.iter().fold((f32::INFINITY, f32::NEG_INFINITY), |(a, b), x| (a.min(*x), b.max(*x)));
    if lo < -0.01 || hi > 1.01 {
        // Logits.
        v.iter_mut().for_each(|x| *x = sigmoid(*x));
    } else if rmbg && hi > lo {
        v.iter_mut().for_each(|x| *x = (*x - lo) / (hi - lo));
    }
    step(report, 1.0, "")?;
    Ok(resize(&v, mw, mh, 1, w, h).into_iter().map(|a| a.clamp(0.0, 1.0)).collect())
}

// ---------------------------------------------------------------------------------------------
// Inpainting

/// Which kind of fill network a model is.
#[derive(Clone, Debug, PartialEq)]
enum Fill {
    /// Image (0–1 floats, CHW) and mask (1 = hole) at a fixed or chosen size; output in 0–1 or
    /// 0–255.
    Lama { image: Port, mask: Port },
    /// MI-GAN: 8-bit image and mask (255 = keep), any size; output 8-bit.
    Migan { image: Port, mask: Port },
    /// One image in, the corrected image out (used inside the hole).
    ImageToImage { image: Port },
}

fn fill_kind(ins: &[Port]) -> Result<Fill> {
    let four: Vec<&Port> = ins.iter().filter(|p| p.shape.len() == 4).collect();
    let mask = four.iter().find(|p| p.name.to_ascii_lowercase().contains("mask")).or_else(|| four.iter().find(|p| p.shape.get(1) == Some(&1)));
    let image = four.iter().find(|p| Some(&p.name) != mask.map(|m| &m.name));
    match (image, mask) {
        (Some(i), Some(m)) if i.elem == Elem::U8 => Ok(Fill::Migan { image: (*i).clone(), mask: (*m).clone() }),
        (Some(i), Some(m)) => Ok(Fill::Lama { image: (*i).clone(), mask: (*m).clone() }),
        (Some(i), None) if four.len() == 1 => Ok(Fill::ImageToImage { image: (*i).clone() }),
        _ => Err(Error::Unsupported("This model's inputs aren't an image and a mask, so it can't fill.".into())),
    }
}

/// Fills `hole` (0–1 per pixel, 1 = replace) in `img`; returns the whole image as RGB 0–1 (the
/// caller blends it in with the hole as the mask).
pub fn inpaint(model: &Model, img: &Image, hole: &[f32], report: Report<'_>) -> Result<Vec<f32>> {
    inpaint_with(net::runtime(), model, img, hole, report)
}

pub fn inpaint_with(loader: &dyn Loader, model: &Model, img: &Image, hole: &[f32], report: Report<'_>) -> Result<Vec<f32>> {
    let (w, h) = (img.width, img.height);
    if hole.len() != w * h {
        return Err(Error::Invalid("The mask doesn't match the image.".into()));
    }
    step(report, 0.02, "Loading the model…")?;
    let n = load(loader, single(model)?)?;
    let kind = fill_kind(&n.inputs())?;
    step(report, 0.15, "Generating…")?;
    let out = match &kind {
        Fill::Migan { image, mask } => {
            let px: Vec<u8> = planar(&img.rgb, w, h, 255.0, [0.0; 3], [1.0; 3]).into_iter().map(|v| v.round().clamp(0.0, 255.0) as u8).collect();
            let m: Vec<u8> = hole.iter().map(|v| if *v >= 0.5 { 0 } else { 255 }).collect();
            let out = n.run(vec![
                (image.name.clone(), Tensor::u8(&[1, 3, h as i64, w as i64], px)),
                (mask.name.clone(), Tensor::u8(&[1, 1, h as i64, w as i64], m)),
            ])?;
            let t =
                out.into_iter().map(|(_, t)| t).find(|t| t.len() == w * h * 3).ok_or_else(|| Error::Invalid("The model's result has an odd size.".into()))?;
            let scale = if matches!(t.data, Data::U8(_)) { 1.0 / 255.0 } else { 1.0 };
            interleave(&t.to_f32().into_iter().map(|v| v * scale).collect::<Vec<_>>(), w, h)
        }
        Fill::Lama { image, mask } => {
            let (iw, ih) = spatial(image, 512);
            let px = resize(&img.rgb, w, h, 3, iw, ih);
            let m: Vec<f32> = resize(hole, w, h, 1, iw, ih).into_iter().map(|v| if v > 0.01 { 1.0 } else { 0.0 }).collect();
            let out = n.run(vec![
                (image.name.clone(), Tensor::f32(&[1, 3, ih as i64, iw as i64], planar(&px, iw, ih, 1.0, [0.0; 3], [1.0; 3]))),
                (mask.name.clone(), Tensor::f32(&[1, 1, ih as i64, iw as i64], m)),
            ])?;
            let t =
                out.into_iter().map(|(_, t)| t).find(|t| t.len() == iw * ih * 3).ok_or_else(|| Error::Invalid("The model's result has an odd size.".into()))?;
            let mut v = t.to_f32();
            if v.iter().any(|x| *x > 1.5) {
                v.iter_mut().for_each(|x| *x /= 255.0);
            }
            resize(&interleave(&v, iw, ih), iw, ih, 3, w, h)
        }
        Fill::ImageToImage { image } => {
            // Dynamic sizes: the crop, rounded to a multiple of 32 and at most 1024 on a side.
            let fit = |v: usize| (v.min(1024) / 32 * 32).max(32);
            let (iw, ih) = match spatial(image, 0) {
                (0, _) | (_, 0) => (fit(w), fit(h)),
                s => s,
            };
            let px = resize(&img.rgb, w, h, 3, iw, ih);
            let out = n.run(vec![(image.name.clone(), Tensor::f32(&[1, 3, ih as i64, iw as i64], planar(&px, iw, ih, 1.0, [0.0; 3], [1.0; 3])))])?;
            // The cleaned image: an output named for it (reflection models also return the
            // reflection layer), else the first three-channel image; it may be smaller than the input.
            let rgb_image = |t: &Tensor| {
                let d = shape_dims(t);
                d.len() == 4 && d.get(1) == Some(&3) && d[2] > 0 && d[3] > 0
            };
            let t = output(&out, &["transmission", "output", "out"])
                .filter(|t| rgb_image(t))
                .or_else(|| out.iter().map(|(_, t)| t).find(|t| rgb_image(t)))
                .ok_or_else(|| Error::Invalid("The model's result isn't an image.".into()))?;
            let d = shape_dims(t);
            let (ow, oh) = (d[3], d[2]);
            let mut v = t.to_f32();
            v.truncate(ow * oh * 3);
            if v.iter().any(|x| *x > 1.5) {
                v.iter_mut().for_each(|x| *x /= 255.0);
            }
            resize(&interleave(&v, ow, oh), ow, oh, 3, w, h)
        }
    };
    step(report, 1.0, "")?;
    Ok(out.into_iter().map(|v| if v.is_finite() { v.clamp(0.0, 1.0) } else { 0.0 }).collect())
}

#[cfg(test)]
mod tests;
