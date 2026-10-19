//! The real ONNX Runtime path: tiny models with the real models' inputs and outputs
//! (`tests/fixtures/make_fixtures.py`). Built only with a runtime feature; with `onnx-dynamic`
//! it needs PHOTOCRAFT_ONNXRUNTIME to point at libonnxruntime, and is skipped without it.
#![cfg(any(feature = "onnx", feature = "onnx-dynamic"))]

use photocraft_ml::infer::{self, Image, Model};
use std::path::PathBuf;

fn fixture(name: &str) -> Model {
    let p = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures").join(name);
    Model { name: name.into(), files: vec![p], catalog_id: None }
}

fn runtime() -> bool {
    if cfg!(feature = "onnx-dynamic") && std::env::var_os("PHOTOCRAFT_ONNXRUNTIME").is_none() {
        eprintln!("skipped: set PHOTOCRAFT_ONNXRUNTIME to libonnxruntime");
        return false;
    }
    infer::set_gpu_allowed(false);
    true
}

fn go(_: f32, _: &str) -> bool {
    true
}

#[test]
fn a_lama_style_model_runs_through_onnx_runtime() {
    if !runtime() {
        return;
    }
    let img = Image::new(40, 30, vec![0.3; 40 * 30 * 3]).unwrap();
    let mut hole = vec![0.0f32; 40 * 30];
    for y in 10..20 {
        for x in 10..30 {
            hole[y * 40 + x] = 1.0;
        }
    }
    let out = infer::inpaint(&fixture("lama_like.onnx"), &img, &hole, &mut go).unwrap();
    assert!(out[(15 * 40 + 20) * 3] > 0.95, "hole filled white: {}", out[(15 * 40 + 20) * 3]);
    assert!((out[0] - 0.3).abs() < 0.02);
    assert_eq!(infer::last_backend().as_deref(), Some("CPU"));
}

#[test]
fn a_migan_style_model_gets_and_gives_8_bit() {
    if !runtime() {
        return;
    }
    let img = Image::new(17, 9, vec![0.5; 17 * 9 * 3]).unwrap();
    let mut hole = vec![0.0f32; 17 * 9];
    hole[4 * 17 + 8] = 1.0;
    let out = infer::inpaint(&fixture("migan_like.onnx"), &img, &hole, &mut go).unwrap();
    assert!(out[(4 * 17 + 8) * 3] < 0.01, "the fake blacks out the hole");
    assert!((out[0] - 128.0 / 255.0).abs() < 0.01);
}

#[test]
fn a_float16_matting_model_converts_both_ways() {
    if !runtime() {
        return;
    }
    let mut rgb = vec![0.0f32; 32 * 32 * 3];
    for y in 0..32 {
        for x in 16..32 {
            for c in 0..3 {
                rgb[(y * 32 + x) * 3 + c] = 1.0;
            }
        }
    }
    let alpha = infer::matte(&fixture("matte_like_fp16.onnx"), &Image::new(32, 32, rgb).unwrap(), &mut go).unwrap();
    assert!(alpha[5 * 32 + 28] > 0.9 && alpha[5 * 32 + 2] < 0.1, "{} {}", alpha[5 * 32 + 28], alpha[5 * 32 + 2]);
}

#[test]
fn a_file_that_isnt_onnx_is_an_error() {
    if !runtime() {
        return;
    }
    let p = std::env::temp_dir().join(format!("photocraft-not-onnx-{}.onnx", std::process::id()));
    std::fs::write(&p, b"not a model").unwrap();
    let m = Model { name: "junk".into(), files: vec![p.clone()], catalog_id: None };
    let img = Image::new(4, 4, vec![0.0; 48]).unwrap();
    assert!(infer::inpaint(&m, &img, &[1.0; 16], &mut go).is_err());
    let _ = std::fs::remove_file(p);
}

#[test]
fn a_float16_inpainting_pipeline_runs_through_onnx_runtime() {
    if !runtime() {
        return;
    }
    use infer::diffusion::{Prompting, diffuse};
    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/tiny_sd");
    let files = [
        "model_index.json",
        "scheduler/scheduler_config.json",
        "text_encoder/model.onnx",
        "tokenizer/vocab.json",
        "tokenizer/merges.txt",
        "tokenizer/tokenizer_config.json",
        "unet/model.onnx",
        "vae_encoder/model.onnx",
        "vae_decoder/model.onnx",
    ];
    let model = Model { name: "tiny".into(), files: files.iter().map(|f| dir.join(f)).collect(), catalog_id: None };
    assert!(infer::diffusion::is_pipeline(&model));
    let img = Image::new(70, 50, vec![0.6; 70 * 50 * 3]).unwrap();
    let hole: Vec<f32> = (0..70 * 50).map(|i| if (20..50).contains(&(i % 70)) && (10..40).contains(&(i / 70)) { 1.0 } else { 0.0 }).collect();
    let opts = Prompting { prompt: "a cat".into(), negative: "cat".into(), seed: Some(77), steps: 3, guidance: 7.5, strength: 1.0 };
    let a = diffuse(&model, &img, &hole, &opts, &mut go).unwrap();
    assert_eq!(a.rgb.len(), 70 * 50 * 3);
    assert!(a.rgb.iter().all(|v| v.is_finite() && (0.0..=1.0).contains(v)));
    assert_eq!(a.seed, 77);
    let b = diffuse(&model, &img, &hole, &opts, &mut go).unwrap();
    assert_eq!(a.rgb, b.rgb, "a seed gives the same picture");
}

/// The converter's networks compute what diffusers computes with the same checkpoint (tiny
/// SD 1 and SD 2 style checkpoints from `make_tiny_checkpoint.py`, in half precision).
#[test]
fn converted_checkpoints_match_diffusers() {
    if !runtime() {
        return;
    }
    use photocraft_ml::convert::{Options, convert, sd::Heads};
    use photocraft_ml::infer::net::{Tensor, runtime as rt};
    let fixtures = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures");
    for (name, sd2) in [("tiny_sd1_ckpt", false), ("tiny_sd2_ckpt", true)] {
        let expected: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(fixtures.join(format!("{name}.expected.json"))).unwrap()).unwrap();
        let out = std::env::temp_dir().join(format!("photocraft-convert-rt-{name}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&out);
        let opts = Options { unet_heads: Some(if sd2 { Heads::Dim(16) } else { Heads::Count(4) }), text_heads: Some(4), full_precision: false };
        convert(&fixtures.join(format!("{name}.safetensors")), &out, &[], &opts, &mut |_| true).unwrap();
        let f = |v: &serde_json::Value| -> Vec<f32> { v.as_array().unwrap().iter().map(|x| x.as_f64().unwrap() as f32).collect() };
        let close = |what: &str, got: &[f32], want: &[f32]| {
            assert_eq!(got.len(), want.len(), "{name} {what}: size");
            let scale = want.iter().fold(0f32, |m, v| m.max(v.abs())).max(1e-3);
            let worst = got.iter().zip(want).fold(0f32, |m, (a, b)| m.max((a - b).abs()));
            assert!(worst / scale < 0.02, "{name} {what}: off by {worst} (largest value {scale})");
            eprintln!("{name} {what}: off by {:.2e} of {scale:.2}", worst);
        };
        let net = |part: &str| rt().load(&out.join(part).join("model.onnx")).unwrap();
        // UNet.
        let u = &expected["unet"];
        let o = net("unet")
            .run(vec![
                ("sample".into(), Tensor::f32(&[2, 9, 8, 8], f(&u["sample"]))),
                ("timestep".into(), Tensor::f32(&[1], vec![u["timestep"].as_f64().unwrap() as f32])),
                ("encoder_hidden_states".into(), Tensor::f32(&[2, 5, 32], f(&u["ehs"]))),
            ])
            .unwrap();
        close("unet", &o[0].1.to_f32(), &f(&u["out"]));
        // VAE.
        let e = &expected["vae_encoder"];
        let o = net("vae_encoder").run(vec![("sample".into(), Tensor::f32(&[1, 3, 32, 32], f(&e["image"])))]).unwrap();
        close("vae encoder", &o[0].1.to_f32()[..4 * 16], &f(&e["mean"]));
        let d = &expected["vae_decoder"];
        let o = net("vae_decoder").run(vec![("latent_sample".into(), Tensor::f32(&[1, 4, 4, 4], f(&d["z"])))]).unwrap();
        close("vae decoder", &o[0].1.to_f32(), &f(&d["out"]));
        // Text encoder: the reference has 7 tokens; the causal mask makes later padding irrelevant.
        let t = &expected["text"];
        let mut ids: Vec<i64> = t["ids"].as_array().unwrap().iter().map(|v| v.as_i64().unwrap()).collect();
        let n = ids.len();
        ids.resize(77, 1);
        let o = net("text_encoder").run(vec![("input_ids".into(), Tensor::i64(&[1, 77], ids))]).unwrap();
        close("text encoder", &o[0].1.to_f32()[..n * 32], &f(&t["out"]));
        let _ = std::fs::remove_dir_all(out);
    }
}

/// An SDXL-style checkpoint: the UNet with its added embedding, and both text encoders'
/// next-to-last layers and pooled output, as diffusers computes them.
#[test]
fn converted_sdxl_checkpoints_match_diffusers() {
    if !runtime() {
        return;
    }
    use photocraft_ml::convert::{Options, convert, sd::Heads};
    use photocraft_ml::infer::net::{Tensor, runtime as rt};
    let fixtures = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures");
    let expected: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(fixtures.join("tiny_sdxl_ckpt.expected.json")).unwrap()).unwrap();
    let out = std::env::temp_dir().join(format!("photocraft-convert-rt-xl-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&out);
    let opts = Options { unet_heads: Some(Heads::Dim(16)), text_heads: Some(4), full_precision: false };
    let files = convert(&fixtures.join("tiny_sdxl_ckpt.safetensors"), &out, &[], &opts, &mut |_| true).unwrap();
    assert!(files.iter().any(|f| f == "text_encoder_2/model.onnx") && files.iter().any(|f| f == "tokenizer_2/vocab.json"));
    let f = |v: &serde_json::Value| -> Vec<f32> { v.as_array().unwrap().iter().map(|x| x.as_f64().unwrap() as f32).collect() };
    let close = |what: &str, got: &[f32], want: &[f32]| {
        assert_eq!(got.len(), want.len(), "{what}: size");
        let scale = want.iter().fold(0f32, |m, v| m.max(v.abs())).max(1e-3);
        let worst = got.iter().zip(want).fold(0f32, |m, (a, b)| m.max((a - b).abs()));
        eprintln!("sdxl {what}: off by {worst:.2e} of {scale:.2}");
        assert!(worst / scale < 0.02, "{what}: off by {worst} (largest value {scale})");
    };
    let net = |part: &str| rt().load(&out.join(part).join("model.onnx")).unwrap();
    let u = &expected["unet"];
    let o = net("unet")
        .run(vec![
            ("sample".into(), Tensor::f32(&[2, 4, 8, 8], f(&u["sample"]))),
            ("timestep".into(), Tensor::f32(&[1], vec![401.0])),
            ("encoder_hidden_states".into(), Tensor::f32(&[2, 5, 64], f(&u["ehs"]))),
            ("text_embeds".into(), Tensor::f32(&[2, 32], f(&u["text_embeds"]))),
            ("time_ids".into(), Tensor::f32(&[2, 6], f(&u["time_ids"]))),
        ])
        .unwrap();
    close("unet", &o.iter().find(|(n, _)| n == "out_sample").unwrap().1.to_f32(), &f(&u["out"]));
    let t = &expected["text"];
    let mut ids: Vec<i64> = t["ids"].as_array().unwrap().iter().map(|v| v.as_i64().unwrap()).collect();
    let n = ids.len();
    ids.resize(77, 0);
    let o = net("text_encoder").run(vec![("input_ids".into(), Tensor::i64(&[1, 77], ids.clone()))]).unwrap();
    close("text encoder", &o.iter().find(|(n, _)| n == "hidden_states").unwrap().1.to_f32()[..n * 32], &f(&t["l"]));
    let o = net("text_encoder_2").run(vec![("input_ids".into(), Tensor::i64(&[1, 77], ids))]).unwrap();
    close("text encoder 2", &o.iter().find(|(n, _)| n == "hidden_states").unwrap().1.to_f32()[..n * 32], &f(&t["g"]));
    close("pooled text", &o.iter().find(|(n, _)| n == "text_embeds").unwrap().1.to_f32(), &f(&t["pooled"]));
    let _ = std::fs::remove_dir_all(out);
}
