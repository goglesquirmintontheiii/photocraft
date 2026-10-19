//! Stable Diffusion checkpoints (`.safetensors`, the single files Civitai and most model pages
//! share) made into the ONNX pipeline the diffusion runtime runs: the UNet, VAE and text encoder
//! are written as ONNX graphs from the checkpoint's own weights (half precision, beside the
//! graphs), with CLIP's tokenizer and Stable Diffusion's scheduler settings.
//!
//! Stable Diffusion 1.x, 2.x and XL checkpoints (Pony, Illustrious and other SDXL models too)
//! work, inpainting (9-channel UNet) or not. SD 3 and Flux checkpoints are recognised and
//! refused with a reason. A checkpoint without its own VAE borrows one from another installed
//! pipeline of the same family.

mod layers;
pub mod onnx;
pub mod safetensors;
pub mod sd;

use crate::infer::tokenizer::{CLIP_MERGES, clip_vocab};
use crate::{Error, Result};
use layers::Builder;
use safetensors::SafeTensors;
use sd::{Heads, Text, TextOut, UNET, VAE};
use serde_json::json;
use std::path::{Path, PathBuf};

/// What kind of Stable Diffusion a checkpoint is.
#[derive(Clone, Debug, PartialEq)]
pub struct Checkpoint {
    /// 1, 2, or 3 for XL.
    pub version: u8,
    /// UNet input channels: 9 for an inpainting model, 4 for an ordinary one.
    pub in_channels: usize,
    pub has_vae: bool,
    pub text: Text,
    /// SDXL's second text encoder (OpenCLIP bigG).
    pub text_2: Option<Text>,
    pub v_prediction: bool,
}

impl Checkpoint {
    pub fn is_xl(&self) -> bool {
        self.version == 3
    }

    /// The VAE's latent scaling.
    pub fn scaling(&self) -> f32 {
        if self.is_xl() { 0.130_25 } else { 0.182_15 }
    }
}

/// Settings tests change (tiny models have other head counts).
#[derive(Clone, Debug, Default)]
pub struct Options {
    pub unet_heads: Option<Heads>,
    pub text_heads: Option<i64>,
    /// Write single precision instead of half.
    pub full_precision: bool,
}

/// Recognises the checkpoint, or says why it can't be used.
pub fn inspect(ck: &SafeTensors, name: &str) -> Result<Checkpoint> {
    let any = |p: &str| ck.tensors.keys().any(|k| k.starts_with(p));
    if any("model.diffusion_model.joint_blocks.") || any("model.diffusion_model.double_blocks.") || any("double_blocks.") || any("joint_blocks.") {
        return Err(Error::Unsupported(format!("{name} is a Stable Diffusion 3 or Flux model, which PhotoCraft can't run yet.")));
    }
    let conv_in = format!("{UNET}input_blocks.0.0.weight");
    let Some(shape) = ck.shape(&conv_in) else {
        return Err(Error::Unsupported(format!(
            "{name} isn't a Stable Diffusion checkpoint (no UNet in it). LoRAs, embeddings and VAEs on their own can't be used as models."
        )));
    };
    let in_channels = shape.get(1).copied().unwrap_or(4);
    let has_vae = ck.has(&format!("{VAE}decoder.conv_in.weight")) && ck.has(&format!("{VAE}encoder.conv_in.weight"));
    if any("conditioner.embedders.") {
        let l = if any("conditioner.embedders.0.transformer.text_model.") {
            "conditioner.embedders.0.transformer.text_model."
        } else {
            "conditioner.embedders.0.transformer."
        };
        if !any(l) || !any("conditioner.embedders.1.model.transformer.resblocks.") {
            return Err(Error::Unsupported(format!("{name} is an SDXL model without both of its text encoders (a refiner?), which PhotoCraft can't use for fills.")));
        }
        return Ok(Checkpoint {
            version: 3,
            in_channels,
            has_vae,
            text: Text::Clip { prefix: l.into() },
            text_2: Some(Text::OpenClip { prefix: "conditioner.embedders.1.model.".into() }),
            v_prediction: false,
        });
    }
    let text = if any("cond_stage_model.model.transformer.resblocks.") {
        Text::OpenClip { prefix: "cond_stage_model.model.".into() }
    } else if any("cond_stage_model.transformer.text_model.") {
        Text::Clip { prefix: "cond_stage_model.transformer.text_model.".into() }
    } else if any("cond_stage_model.transformer.") {
        Text::Clip { prefix: "cond_stage_model.transformer.".into() }
    } else {
        return Err(Error::Unsupported(format!("{name} has no text encoder, so it can't read prompts.")));
    };
    let version = if matches!(text, Text::OpenClip { .. }) { 2 } else { 1 };
    // SD 2's 768 px models predict velocity; its inpainting and 512 px base models predict noise.
    let lower = name.to_ascii_lowercase();
    let v_prediction = version == 2 && in_channels == 4 && !lower.contains("base") && !lower.contains("512");
    Ok(Checkpoint { version, in_channels, has_vae, text, text_2: None, v_prediction })
}

/// The pipeline's parts as written: `name/model.onnx` and its `model.onnx_data`.
const PARTS: [&str; 5] = ["text_encoder", "text_encoder_2", "unet", "vae_encoder", "vae_decoder"];

/// Makes the ONNX pipeline of `checkpoint` in `out` (created; it should be empty). Returns the
/// files written, relative to `out`. `vae_from` are other pipelines' folders to take the VAE from
/// when the checkpoint has none (one whose latent scaling matches: SDXL's VAE for SDXL). `report` gets progress 0–1 and returns false to cancel.
pub fn convert(checkpoint: &Path, out: &Path, vae_from: &[PathBuf], opts: &Options, report: &mut dyn FnMut(f32) -> bool) -> Result<Vec<String>> {
    let name = checkpoint.file_stem().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default();
    let ck = SafeTensors::open(checkpoint)?;
    let info = inspect(&ck, &name)?;
    let scaling_of = |d: &Path| -> f32 {
        std::fs::read_to_string(d.join("vae_decoder").join("config.json"))
            .ok()
            .and_then(|t| serde_json::from_str::<serde_json::Value>(&t).ok())
            .and_then(|v| v.get("scaling_factor").and_then(serde_json::Value::as_f64))
            .unwrap_or(0.182_15) as f32
    };
    let borrowed = vae_from.iter().find(|d| (scaling_of(d) - info.scaling()).abs() < 1e-3);
    let vae_dir = match (info.has_vae, borrowed) {
        (true, _) => None,
        (false, Some(d)) => Some(d.clone()),
        (false, None) => {
            return Err(Error::Unsupported(format!(
                "{name} has no VAE built in. Download a recommended Generative Fill model of the same kind ({}) first: its VAE is used with this checkpoint.",
                if info.is_xl() { "SDXL" } else { "Stable Diffusion 1.5 or 2" }
            )));
        }
    };
    let elem = if opts.full_precision { onnx::FLOAT } else { onnx::FLOAT16 };
    let total = ck
        .tensors
        .keys()
        .filter(|k| k.starts_with(UNET) || k.starts_with("cond_stage_model.") || k.starts_with("conditioner.") || (info.has_vae && k.starts_with(VAE)))
        .count()
        .max(1);
    let mut done = 0usize;
    let mut cancelled = false;
    let mut tick = || -> Result<()> {
        done += 1;
        if !report((done as f32 / total as f32).min(0.99)) {
            cancelled = true;
            return Err(Error::Cancelled);
        }
        Ok(())
    };
    let mut files = Vec::new();
    for part in PARTS {
        if part == "text_encoder_2" && info.text_2.is_none() {
            continue;
        }
        if part.starts_with("vae") && let Some(src) = &vae_dir {
            for f in ["model.onnx", "model.onnx_data", "weights.pb"] {
                let from = src.join(part).join(f);
                if from.is_file() {
                    std::fs::create_dir_all(out.join(part)).map_err(|e| Error::io("Couldn't create a folder for the model", e))?;
                    std::fs::copy(&from, out.join(part).join(f)).map_err(|e| Error::io("Couldn't copy the VAE", e))?;
                    files.push(format!("{part}/{f}"));
                }
            }
            continue;
        }
        let dir = out.join(part);
        std::fs::create_dir_all(&dir).map_err(|e| Error::io("Couldn't create a folder for the model", e))?;
        let model = dir.join("model.onnx");
        // SDXL's VAE overflows half precision: it stays in single.
        let part_elem = if info.is_xl() && part.starts_with("vae") { onnx::FLOAT } else { elem };
        let g = onnx::Graph::new(&model, part_elem)?;
        let mut b = Builder { g, ck: &ck, tick: &mut tick };
        let built = match part {
            "unet" => sd::unet(&mut b, opts.unet_heads.unwrap_or(if info.version == 1 { Heads::Count(8) } else { Heads::Dim(64) })),
            "text_encoder" | "text_encoder_2" => {
                let text = if part == "text_encoder" { &info.text } else { info.text_2.as_ref().unwrap_or(&info.text) };
                let dim = text_dim(&ck, text)?;
                let out = match (info.is_xl(), part) {
                    (true, "text_encoder") => TextOut::Penultimate { pooled: false },
                    (true, _) => TextOut::Penultimate { pooled: true },
                    _ => TextOut::Final { skip_last: info.version == 2 },
                };
                sd::text_encoder(&mut b, text, opts.text_heads.unwrap_or((dim / 64).max(1) as i64), crate::infer::tokenizer::LENGTH, out)
            }
            "vae_encoder" => sd::vae_encoder(&mut b),
            _ => sd::vae_decoder(&mut b),
        };
        built?;
        b.g.finish(&model)?;
        files.push(format!("{part}/model.onnx"));
        files.push(format!("{part}/model.onnx_data"));
    }
    if cancelled {
        return Err(Error::Cancelled);
    }
    // Tokenizer, scheduler and the pipeline's index.
    let write = |rel: &str, text: String| -> Result<String> {
        let p = out.join(rel);
        if let Some(d) = p.parent() {
            std::fs::create_dir_all(d).map_err(|e| Error::io("Couldn't create a folder for the model", e))?;
        }
        std::fs::write(&p, text).map_err(|e| Error::io(&format!("Couldn't write {rel}"), e))?;
        Ok(rel.to_string())
    };
    files.push(write("tokenizer/vocab.json", serde_json::Value::Object(vocab_map()).to_string())?);
    files.push(write("tokenizer/merges.txt", CLIP_MERGES.to_string())?);
    let pad = if info.version == 2 { "!" } else { "<|endoftext|>" };
    files.push(write("tokenizer/tokenizer_config.json", json!({"pad_token": pad, "model_max_length": 77, "tokenizer_class": "CLIPTokenizer"}).to_string())?);
    if info.text_2.is_some() {
        files.push(write("tokenizer_2/vocab.json", serde_json::Value::Object(vocab_map()).to_string())?);
        files.push(write("tokenizer_2/merges.txt", CLIP_MERGES.to_string())?);
        files.push(write("tokenizer_2/tokenizer_config.json", json!({"pad_token": "!", "model_max_length": 77, "tokenizer_class": "CLIPTokenizer"}).to_string())?);
    }
    files.push(write("vae_decoder/config.json", json!({"scaling_factor": info.scaling()}).to_string())?);
    files.push(write(
        "scheduler/scheduler_config.json",
        json!({
            "_class_name": "DDIMScheduler",
            "beta_schedule": "scaled_linear",
            "beta_start": 0.00085,
            "beta_end": 0.012,
            "num_train_timesteps": 1000,
            "steps_offset": 1,
            "set_alpha_to_one": false,
            "clip_sample": false,
            "timestep_spacing": "leading",
            "prediction_type": if info.v_prediction { "v_prediction" } else { "epsilon" },
        })
        .to_string(),
    )?);
    files.push(write(
        "model_index.json",
        json!({
            "_class_name": match (info.is_xl(), info.in_channels == 9) {
                (true, true) => "ORTStableDiffusionXLInpaintPipeline",
                (true, false) => "ORTStableDiffusionXLPipeline",
                (false, true) => "OnnxStableDiffusionInpaintPipeline",
                (false, false) => "OnnxStableDiffusionPipeline",
            },
            "converted_from": checkpoint.file_name().map(|n| n.to_string_lossy().into_owned())
        })
        .to_string(),
    )?);
    report(1.0);
    Ok(files)
}

fn vocab_map() -> serde_json::Map<String, serde_json::Value> {
    clip_vocab(CLIP_MERGES).into_iter().enumerate().map(|(i, t)| (t, json!(i))).collect()
}

fn text_dim(ck: &SafeTensors, text: &Text) -> Result<usize> {
    let key = match text {
        Text::Clip { prefix } => format!("{prefix}embeddings.token_embedding.weight"),
        Text::OpenClip { prefix } => format!("{prefix}token_embedding.weight"),
    };
    ck.shape(&key).and_then(|s| s.get(1).copied()).ok_or_else(|| Error::Invalid("The checkpoint's text encoder has no token embedding.".into()))
}

/// Whether a model's files are a checkpoint still to be converted (no pipeline beside it).
pub fn checkpoint_to_convert(files: &[PathBuf]) -> Option<&PathBuf> {
    let has_pipeline = files.iter().any(|p| p.file_name().is_some_and(|n| n == "model_index.json"));
    if has_pipeline {
        return None;
    }
    files.iter().filter(|p| p.extension().is_some_and(|e| e.eq_ignore_ascii_case("safetensors"))).max_by_key(|p| std::fs::metadata(p).map(|m| m.len()).unwrap_or(0))
}

#[cfg(test)]
mod tests;
