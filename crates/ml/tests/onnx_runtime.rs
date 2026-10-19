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
