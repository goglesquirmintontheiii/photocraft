//! Prompted inpainting (the Infill brush and Generative Fill) with Stable Diffusion pipelines
//! exported to ONNX in the diffusers layout: `text_encoder/`, `unet/`, `vae_encoder/`,
//! `vae_decoder/` (each a `model.onnx`, external weights beside it), `tokenizer/` and
//! `scheduler/scheduler_config.json`. Flat files named after their part (`unet.onnx` …) work too.
//!
//! The prompt and negative prompt go through the CLIP tokenizer and text encoder; the image with
//! the hole blanked is encoded by the VAE; the UNet denoises seeded noise with classifier-free
//! guidance and a DDIM schedule; the VAE decodes the result. Inpainting UNets (9 input channels:
//! latents, mask, masked-image latents) fill the hole themselves; ordinary text-to-image UNets
//! (4 channels) fill it by keeping the rest of the image at each step's noise level.
//!
//! The picture is worked on at about 512 × 512 (the size these models know), in multiples of 64,
//! keeping its proportions; the caller blends the result into the hole.

use super::net::{Elem, Loader, Network, Port, Tensor};
use super::tokenizer::{ClipTokenizer, LENGTH};
use super::{Image, Model, Report, file_name, interleave, load, planar, resize, step};
use crate::{Error, Result};
use serde_json::Value;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, PoisonError};

/// What to generate.
#[derive(Clone, Debug, PartialEq)]
pub struct Prompting {
    pub prompt: String,
    pub negative: String,
    /// `None` picks a new seed each time (the one used comes back in [`Generated`]).
    pub seed: Option<u64>,
    pub steps: usize,
    /// Classifier-free guidance: how closely to follow the prompt (1 = ignore it).
    pub guidance: f32,
}

impl Default for Prompting {
    fn default() -> Self {
        Prompting { prompt: String::new(), negative: String::new(), seed: None, steps: DEFAULT_STEPS, guidance: 7.5 }
    }
}

pub const DEFAULT_STEPS: usize = 25;
pub const MAX_STEPS: usize = 150;

/// Seeds are shown and typed as whole numbers up to this.
pub const MAX_SEED: u64 = u32::MAX as u64;

/// The filled image (RGB 0–1, the input's size) and the seed it came from.
#[derive(Clone, Debug, PartialEq)]
pub struct Generated {
    pub rgb: Vec<f32>,
    pub seed: u64,
}

/// The files of a diffusion pipeline.
#[derive(Clone, Debug, PartialEq)]
struct Parts {
    text_encoder: PathBuf,
    unet: PathBuf,
    vae_encoder: Option<PathBuf>,
    vae_decoder: PathBuf,
    vocab: PathBuf,
    merges: PathBuf,
    tokenizer_config: Vec<PathBuf>,
    scheduler: Option<PathBuf>,
    vae_config: Option<PathBuf>,
}

fn parent_name(p: &Path) -> String {
    p.parent().and_then(Path::file_name).map(|n| n.to_string_lossy().to_ascii_lowercase()).unwrap_or_default()
}

fn stem(p: &Path) -> String {
    p.file_stem().map(|n| n.to_string_lossy().to_ascii_lowercase()).unwrap_or_default()
}

fn parts(model: &Model) -> Option<Parts> {
    let onnx = |part: &str| {
        model.files.iter().find(|p| p.extension().is_some_and(|e| e.eq_ignore_ascii_case("onnx")) && (parent_name(p) == part || stem(p) == part)).cloned()
    };
    let named = |name: &str| model.files.iter().find(|p| file_name(p) == name).cloned();
    let in_dir = |dir: &str, name: &str| model.files.iter().find(|p| file_name(p) == name && parent_name(p) == dir).cloned();
    Some(Parts {
        text_encoder: onnx("text_encoder")?,
        unet: onnx("unet")?,
        vae_encoder: onnx("vae_encoder"),
        vae_decoder: onnx("vae_decoder")?,
        vocab: named("vocab.json")?,
        merges: named("merges.txt")?,
        tokenizer_config: ["tokenizer_config.json", "special_tokens_map.json"].iter().filter_map(|n| named(n)).collect(),
        scheduler: named("scheduler_config.json"),
        vae_config: in_dir("vae_decoder", "config.json"),
    })
}

/// Whether the model is a diffusion pipeline this can run.
pub fn is_pipeline(model: &Model) -> bool {
    parts(model).is_some()
}

fn not_a_pipeline(model: &Model) -> Error {
    let checkpoint = model.files.iter().any(|p| p.extension().is_some_and(|e| e.eq_ignore_ascii_case("safetensors") || e.eq_ignore_ascii_case("ckpt")));
    Error::Unsupported(if checkpoint {
        format!(
            "{} is a single-file Stable Diffusion checkpoint, which PhotoCraft can't run yet. Download one of the recommended Generative Fill models, or import a Hugging Face repository of an ONNX pipeline (model_index.json with unet, text_encoder, vae_encoder and vae_decoder folders).",
            model.name
        )
    } else {
        format!(
            "{} isn't a complete ONNX diffusion pipeline: it needs text_encoder, unet and vae_decoder networks and the tokenizer's vocab.json and merges.txt.",
            model.name
        )
    })
}

// ---------------------------------------------------------------------------------------------
// Noise

/// SplitMix64: small, fast and the same everywhere, so a seed gives the same picture again.
struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// Uniform in (0, 1].
    fn unit(&mut self) -> f64 {
        ((self.next() >> 11) as f64 + 1.0) / (1u64 << 53) as f64
    }

    /// Standard normal values (Box–Muller).
    fn normals(&mut self, n: usize) -> Vec<f32> {
        let mut out = Vec::with_capacity(n + 1);
        while out.len() < n {
            let (u, v) = (self.unit(), self.unit());
            let r = (-2.0 * u.ln()).sqrt();
            let a = std::f64::consts::TAU * v;
            out.push((r * a.cos()) as f32);
            out.push((r * a.sin()) as f32);
        }
        out.truncate(n);
        out
    }
}

/// A fresh seed when none was given.
pub fn new_seed() -> u64 {
    use std::hash::{BuildHasher, Hasher};
    let mut h = std::collections::hash_map::RandomState::new().build_hasher();
    h.write_u128(std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_nanos()).unwrap_or(0));
    h.finish() % (MAX_SEED + 1)
}

// ---------------------------------------------------------------------------------------------
// Scheduler

/// DDIM (η = 0) over the training noise schedule in `scheduler_config.json`.
#[derive(Clone, Debug, PartialEq)]
struct Schedule {
    alphas_cumprod: Vec<f64>,
    final_alpha: f64,
    timesteps: Vec<usize>,
    step_ratio: usize,
    v_prediction: bool,
}

impl Schedule {
    fn new(config: &Value, steps: usize) -> Schedule {
        let f = |k: &str, d: f64| config.get(k).and_then(Value::as_f64).unwrap_or(d);
        let s = |k: &str, d: &str| config.get(k).and_then(Value::as_str).unwrap_or(d).to_string();
        let train = (f("num_train_timesteps", 1000.0) as usize).clamp(10, 10_000);
        let (b0, b1) = (f("beta_start", 0.000_85), f("beta_end", 0.012));
        let betas: Vec<f64> = (0..train)
            .map(|i| {
                let t = i as f64 / (train - 1) as f64;
                match s("beta_schedule", "scaled_linear").as_str() {
                    "linear" => b0 + (b1 - b0) * t,
                    _ => {
                        let r = b0.sqrt() + (b1.sqrt() - b0.sqrt()) * t;
                        r * r
                    }
                }
            })
            .collect();
        let mut acc = 1.0;
        let alphas_cumprod: Vec<f64> = betas
            .iter()
            .map(|b| {
                acc *= 1.0 - b;
                acc
            })
            .collect();
        let steps = steps.clamp(1, train);
        let offset = f("steps_offset", 0.0).max(0.0) as usize;
        let step_ratio = (train / steps).max(1);
        let timesteps: Vec<usize> = match s("timestep_spacing", "leading").as_str() {
            "trailing" => {
                let r = train as f64 / steps as f64;
                (0..steps).map(|i| ((train as f64 - i as f64 * r).round() as usize).saturating_sub(1)).collect()
            }
            "linspace" => (0..steps).rev().map(|i| ((train - 1) as f64 * i as f64 / (steps - 1).max(1) as f64).round() as usize).collect(),
            _ => (0..steps).rev().map(|i| (i * step_ratio + offset).min(train - 1)).collect(),
        };
        let set_alpha_to_one = config.get("set_alpha_to_one").and_then(Value::as_bool).unwrap_or(false);
        let final_alpha = if set_alpha_to_one { 1.0 } else { alphas_cumprod.first().copied().unwrap_or(1.0) };
        Schedule { alphas_cumprod, final_alpha, timesteps, step_ratio, v_prediction: s("prediction_type", "epsilon") == "v_prediction" }
    }

    fn alpha(&self, t: Option<usize>) -> f64 {
        t.and_then(|t| self.alphas_cumprod.get(t).copied()).unwrap_or(self.final_alpha)
    }

    /// One DDIM step from `x` at `t` with the model's prediction `out`.
    fn step(&self, out: &[f32], t: usize, x: &[f32]) -> Vec<f32> {
        let a = self.alpha(Some(t));
        let prev = t.checked_sub(self.step_ratio);
        let ap = self.alpha(prev);
        let (sa, s1a, sap, s1ap) = (a.sqrt(), (1.0 - a).sqrt(), ap.sqrt(), (1.0 - ap).sqrt());
        x.iter()
            .zip(out)
            .map(|(x, o)| {
                let (x, o) = (f64::from(*x), f64::from(*o));
                let (x0, eps) = if self.v_prediction { (sa * x - s1a * o, sa * o + s1a * x) } else { ((x - s1a * o) / sa, o) };
                (sap * x0 + s1ap * eps) as f32
            })
            .collect()
    }

    /// `x0` noised to the level after step `t` (for keeping the image outside the hole).
    fn noised(&self, x0: &[f32], noise: &[f32], t: Option<usize>) -> Vec<f32> {
        let a = self.alpha(t);
        let (sa, s1a) = (a.sqrt() as f32, (1.0 - a).sqrt() as f32);
        x0.iter().zip(noise).map(|(x, n)| sa * x + s1a * n).collect()
    }
}

// ---------------------------------------------------------------------------------------------
// Running

/// The text embeddings of the last prompts, so more variations of the same prompt skip the
/// text encoder.
struct Embedded {
    encoder: PathBuf,
    texts: (String, String),
    cond: Tensor,
    uncond: Tensor,
}

static EMBEDDED: Mutex<Option<Embedded>> = Mutex::new(None);

fn find_port<'a>(ports: &'a [Port], names: &[&str], dims: usize) -> Option<&'a Port> {
    names.iter().find_map(|n| ports.iter().find(|p| p.name == *n)).or_else(|| ports.iter().find(|p| p.shape.len() == dims))
}

fn embed(text_encoder: &Arc<dyn Network>, tokenizer: &ClipTokenizer, text: &str) -> Result<Tensor> {
    let ins = text_encoder.inputs();
    let input = ins
        .iter()
        .find(|p| p.name == "input_ids")
        .or_else(|| ins.iter().find(|p| matches!(p.elem, Elem::I32 | Elem::I64)))
        .or(ins.first())
        .ok_or_else(|| Error::Invalid("The text encoder has no inputs.".into()))?;
    let ids = tokenizer.encode(text);
    let out = text_encoder.run(vec![(input.name.clone(), Tensor::i64(&[1, LENGTH as i64], ids))])?;
    let t = out
        .iter()
        .find(|(n, _)| n == "last_hidden_state")
        .or_else(|| out.iter().find(|(_, t)| t.shape.len() == 3))
        .map(|(_, t)| t.clone())
        .ok_or_else(|| Error::Invalid("The text encoder gave no text embeddings.".into()))?;
    Ok(Tensor::f32(&t.shape, t.to_f32()))
}

/// The size the model works at: about 512 × 512 in area, multiples of 64, 256–768 a side (or the
/// UNet's own size when it is fixed).
fn work_size(w: usize, h: usize, sample: &Port) -> (usize, usize) {
    let fixed = |i: usize| sample.shape.get(i).copied().filter(|v| *v > 0).map(|v| v as usize * 8);
    if let (Some(fw), Some(fh)) = (fixed(3), fixed(2)) {
        return (fw, fh);
    }
    let s = (512.0 * 512.0 / (w.max(1) * h.max(1)) as f64).sqrt();
    let side = |v: usize| (((v as f64 * s) / 64.0).round() as usize * 64).clamp(256, 768);
    (side(w), side(h))
}

/// Interleaved RGB 0–1 to planar −1…1.
fn to_signed(rgb: &[f32], w: usize, h: usize) -> Vec<f32> {
    planar(rgb, w, h, 2.0, [1.0; 3], [1.0; 3])
}

/// The latents (first four channels: the mean when the encoder returns mean and log-variance).
fn latents_of(out: &[(String, Tensor)], lw: usize, lh: usize) -> Result<Vec<f32>> {
    let t = out
        .iter()
        .find(|(n, _)| n == "latent_sample")
        .or_else(|| out.iter().find(|(_, t)| t.shape.len() == 4))
        .map(|(_, t)| t)
        .ok_or_else(|| Error::Invalid("The image encoder gave no latents.".into()))?;
    let v = t.to_f32();
    let n = lw * lh * 4;
    if v.len() < n || t.shape.get(2).copied() != Some(lh as i64) || t.shape.get(3).copied() != Some(lw as i64) {
        return Err(Error::Invalid("The image encoder's latents have an odd size.".into()));
    }
    Ok(v.get(..n).unwrap_or(&[]).to_vec())
}

/// Fills `hole` (0–1 per pixel, 1 = replace) in `img` from the prompt.
pub fn diffuse(model: &Model, img: &Image, hole: &[f32], opts: &Prompting, report: Report<'_>) -> Result<Generated> {
    diffuse_with(super::net::runtime(), model, img, hole, opts, report)
}

pub fn diffuse_with(loader: &dyn Loader, model: &Model, img: &Image, hole: &[f32], opts: &Prompting, report: Report<'_>) -> Result<Generated> {
    let (w, h) = (img.width, img.height);
    if hole.len() != w * h {
        return Err(Error::Invalid("The mask doesn't match the image.".into()));
    }
    let parts = parts(model).ok_or_else(|| not_a_pipeline(model))?;
    let seed = opts.seed.map_or_else(new_seed, |s| s.min(MAX_SEED));
    let steps = opts.steps.clamp(1, MAX_STEPS);
    step(report, 0.01, "Loading the model…")?;
    // One pipeline in memory at a time: they are gigabytes each.
    let dir = pipeline_dir(&parts.unet);
    super::net::forget_where(|p| is_pipeline_part(p) && !p.starts_with(&dir));
    let tokenizer = ClipTokenizer::load(&parts.vocab, &parts.merges, &parts.tokenizer_config.iter().map(PathBuf::as_path).collect::<Vec<_>>())?;
    let unet = load(loader, &parts.unet)?;
    let u_in = unet.inputs();
    let sample = find_port(&u_in, &["sample"], 4).ok_or_else(|| Error::Unsupported("The UNet has no latent input.".into()))?.clone();
    let hidden = find_port(&u_in, &["encoder_hidden_states"], 3).ok_or_else(|| Error::Unsupported("The UNet has no text input.".into()))?.clone();
    let timestep = u_in
        .iter()
        .find(|p| p.name == "timestep" || p.name == "t")
        .or_else(|| u_in.iter().find(|p| p.name != sample.name && p.name != hidden.name))
        .ok_or_else(|| Error::Unsupported("The UNet has no timestep input.".into()))?
        .clone();
    if u_in.len() > 3 {
        return Err(Error::Unsupported(format!(
            "{}'s UNet takes extra inputs ({}): Stable Diffusion XL and other pipelines like it aren't supported yet. Use a Stable Diffusion 1.5 or 2 pipeline.",
            model.name,
            u_in.iter().map(|p| p.name.as_str()).filter(|n| ![sample.name.as_str(), hidden.name.as_str(), timestep.name.as_str()].contains(n)).collect::<Vec<_>>().join(", ")
        )));
    }
    let channels = sample.shape.get(1).copied().filter(|c| *c > 0).unwrap_or(9);
    let inpainting_unet = match channels {
        9 => true,
        4 => false,
        c => return Err(Error::Unsupported(format!("{}'s UNet takes {c} channels; inpainting needs 9 (or 4 for an ordinary one).", model.name))),
    };
    let batch_two = sample.shape.first().is_none_or(|b| *b != 1);
    let guided = opts.guidance > 1.0;

    // Text.
    step(report, 0.03, "Reading the prompt…")?;
    let texts = (opts.prompt.trim().to_string(), opts.negative.trim().to_string());
    let cached = {
        let e = EMBEDDED.lock().unwrap_or_else(PoisonError::into_inner);
        e.as_ref().filter(|e| e.encoder == parts.text_encoder && e.texts == texts).map(|e| (e.cond.clone(), e.uncond.clone()))
    };
    let (cond, uncond) = match cached {
        Some(c) => c,
        None => {
            let te = load(loader, &parts.text_encoder)?;
            let cond = embed(&te, &tokenizer, &texts.0)?;
            let uncond = embed(&te, &tokenizer, &texts.1)?;
            *EMBEDDED.lock().unwrap_or_else(PoisonError::into_inner) =
                Some(Embedded { encoder: parts.text_encoder.clone(), texts: texts.clone(), cond: cond.clone(), uncond: uncond.clone() });
            (cond, uncond)
        }
    };

    // Image and mask at the working size.
    let (ww, wh) = work_size(w, h, &sample);
    let (lw, lh) = (ww / 8, wh / 8);
    let rgb = resize(&img.rgb, w, h, 3, ww, wh);
    let mask: Vec<f32> = resize(hole, w, h, 1, ww, wh).into_iter().map(|v| if v > 0.25 { 1.0 } else { 0.0 }).collect();
    // The hole at latent size, grown by any coverage so the fill's edges are regenerated too.
    let latent_mask: Vec<f32> = resize(&mask, ww, wh, 1, lw, lh).into_iter().map(|v| if v > 0.05 { 1.0 } else { 0.0 }).collect();
    let scaling = parts
        .vae_config
        .as_ref()
        .and_then(|p| std::fs::read_to_string(p).ok())
        .and_then(|t| serde_json::from_str::<Value>(&t).ok())
        .and_then(|v| v.get("scaling_factor").and_then(Value::as_f64))
        .unwrap_or(0.182_15) as f32;
    step(report, 0.06, "Reading the image…")?;
    let vae_enc_path = parts.vae_encoder.as_ref().ok_or_else(|| Error::Unsupported(format!("{} has no vae_encoder, so it can't see the image.", model.name)))?;
    let vae_enc = load(loader, vae_enc_path)?;
    let enc_in = vae_enc.inputs();
    let enc_port = find_port(&enc_in, &["sample"], 4).ok_or_else(|| Error::Invalid("The image encoder has no image input.".into()))?;
    let encode = |px: Vec<f32>| -> Result<Vec<f32>> {
        let out = vae_enc.run(vec![(enc_port.name.clone(), Tensor::f32(&[1, 3, wh as i64, ww as i64], px))])?;
        Ok(latents_of(&out, lw, lh)?.into_iter().map(|v| v * scaling).collect())
    };
    let signed = to_signed(&rgb, ww, wh);
    let n = ww * wh;
    let (image_latents, masked_latents) = if inpainting_unet {
        let masked: Vec<f32> = signed.iter().enumerate().map(|(i, v)| if mask.get(i % n).copied().unwrap_or(0.0) >= 0.5 { 0.0 } else { *v }).collect();
        (Vec::new(), encode(masked)?)
    } else {
        (encode(signed)?, Vec::new())
    };

    // Denoising.
    let config = parts
        .scheduler
        .as_ref()
        .and_then(|p| std::fs::read_to_string(p).ok())
        .and_then(|t| serde_json::from_str::<Value>(&t).ok())
        .unwrap_or(Value::Null);
    let schedule = Schedule::new(&config, steps);
    let mut rng = Rng(seed);
    let ln = lw * lh * 4;
    let noise = rng.normals(ln);
    let mut latents = noise.clone();
    let lplane = lw * lh;
    let t_shape: Vec<i64> = match timestep.shape.first() {
        None => Vec::new(),
        Some(d) if *d > 0 => vec![*d],
        Some(_) => vec![1],
    };
    let t_count = t_shape.first().copied().unwrap_or(1).max(1) as usize;
    let total = schedule.timesteps.len();
    for (k, &t) in schedule.timesteps.iter().enumerate() {
        step(report, 0.1 + 0.85 * k as f32 / total as f32, &format!("Generating… step {} of {total}", k + 1))?;
        // The UNet's input: latents, then for inpainting UNets the mask and the masked image.
        let mut x = latents.clone();
        if inpainting_unet {
            x.extend_from_slice(&latent_mask);
            x.extend_from_slice(&masked_latents);
        }
        let ts = Tensor::f32(&t_shape, vec![t as f32; t_count]);
        let shape1 = [1, channels, lh as i64, lw as i64];
        let run = |xs: Vec<f32>, emb: Vec<f32>, b: i64| -> Result<Vec<f32>> {
            let mut s = shape1;
            s[0] = b;
            let mut hs = cond.shape.clone();
            if let Some(first) = hs.first_mut() {
                *first = b;
            }
            let out = unet.run(vec![
                (sample.name.clone(), Tensor::f32(&s, xs)),
                (timestep.name.clone(), ts.clone()),
                (hidden.name.clone(), Tensor::f32(&hs, emb)),
            ])?;
            let o = out
                .iter()
                .find(|(n, _)| n == "out_sample")
                .or_else(|| out.iter().find(|(_, t)| t.shape.len() == 4))
                .map(|(_, t)| t.to_f32())
                .ok_or_else(|| Error::Invalid("The UNet gave no result.".into()))?;
            if o.len() < ln * b as usize {
                return Err(Error::Invalid("The UNet's result has an odd size.".into()));
            }
            Ok(o)
        };
        let cond_v = cond.to_f32();
        let pred = if !guided {
            run(x, cond_v, 1)?
        } else {
            let uncond_v = uncond.to_f32();
            let (pu, pc) = if batch_two {
                let mut xs = x.clone();
                xs.extend_from_slice(&x);
                let mut emb = uncond_v;
                emb.extend_from_slice(&cond_v);
                let o = run(xs, emb, 2)?;
                (o.get(..ln).unwrap_or(&[]).to_vec(), o.get(ln..2 * ln).unwrap_or(&[]).to_vec())
            } else {
                (run(x.clone(), uncond_v, 1)?, run(x, cond_v, 1)?)
            };
            let g = opts.guidance;
            pu.iter().zip(&pc).map(|(u, c)| u + g * (c - u)).collect()
        };
        latents = schedule.step(pred.get(..ln).unwrap_or(&[]), t, &latents);
        if !inpainting_unet {
            // Outside the hole, the original image at the next step's noise level.
            let prev = t.checked_sub(schedule.step_ratio).filter(|_| k + 1 < total);
            let kept = if k + 1 < total { schedule.noised(&image_latents, &noise, prev) } else { image_latents.clone() };
            for (i, v) in latents.iter_mut().enumerate() {
                if latent_mask.get(i % lplane).copied().unwrap_or(0.0) < 0.5 {
                    *v = kept.get(i).copied().unwrap_or(*v);
                }
            }
        }
        if latents.iter().any(|v| !v.is_finite()) {
            return Err(Error::Invalid("The model's numbers overflowed (try the full-precision model, or the CPU).".into()));
        }
    }

    // Back to pixels.
    step(report, 0.96, "Finishing…")?;
    let dec = load(loader, &parts.vae_decoder)?;
    let d_in = dec.inputs();
    let d_port = find_port(&d_in, &["latent_sample"], 4).ok_or_else(|| Error::Invalid("The image decoder has no input.".into()))?;
    let z: Vec<f32> = latents.iter().map(|v| v / scaling).collect();
    let out = dec.run(vec![(d_port.name.clone(), Tensor::f32(&[1, 4, lh as i64, lw as i64], z))])?;
    let px = out
        .iter()
        .find(|(n, _)| n == "sample")
        .or_else(|| out.iter().find(|(_, t)| t.shape.len() == 4))
        .map(|(_, t)| t.to_f32())
        .ok_or_else(|| Error::Invalid("The image decoder gave no image.".into()))?;
    if px.len() < n * 3 {
        return Err(Error::Invalid("The image decoder's result has an odd size.".into()));
    }
    let unit: Vec<f32> = px.get(..n * 3).unwrap_or(&[]).iter().map(|v| if v.is_finite() { ((v + 1.0) * 0.5).clamp(0.0, 1.0) } else { 0.0 }).collect();
    let rgb = resize(&interleave(&unit, ww, wh), ww, wh, 3, w, h);
    step(report, 1.0, "")?;
    Ok(Generated { rgb, seed })
}

/// The model folder a pipeline part is in (`<model>/unet/model.onnx` or `<model>/unet.onnx`).
fn pipeline_dir(part: &Path) -> PathBuf {
    let parent = part.parent().map(Path::to_path_buf).unwrap_or_default();
    if ["unet", "text_encoder", "vae_encoder", "vae_decoder"].contains(&parent_name(part).as_str()) {
        parent.parent().map(Path::to_path_buf).unwrap_or(parent)
    } else {
        parent
    }
}

fn is_pipeline_part(p: &Path) -> bool {
    let part = |n: &str| matches!(n, "unet" | "text_encoder" | "vae_encoder" | "vae_decoder");
    part(&parent_name(p)) || part(&stem(p))
}

#[cfg(test)]
mod tests;
