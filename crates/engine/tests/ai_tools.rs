//! AI Object Selection and the Generative Brushes, end to end: models imported into a models
//! folder, then the commands run them.
//!
//! Without a model runtime (the default build) they must fail cleanly. With one, run as
//! `PHOTOCRAFT_ONNXRUNTIME=…/libonnxruntime.so cargo test -p photocraft-engine --features
//! photocraft-ml/onnx-dynamic --test ai_tools`; the tiny fixture models from
//! `crates/ml/tests/fixtures` stand in for LaMa and a matting model.

use photocraft_engine::Session;
use photocraft_ml::{Manager, ModelKind};
use serde_json::json;
use std::path::PathBuf;

fn fixture(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../ml/tests/fixtures").join(name)
}

/// A models folder with `file` imported for `kinds`.
fn models(tag: &str, file: &str, kinds: &[ModelKind]) -> Manager {
    let dir = std::env::temp_dir().join(format!("photocraft-ai-tools-{tag}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    let m = Manager::open(&dir).unwrap();
    m.import_files(&[fixture(file)], kinds, Some(tag.into())).unwrap();
    let start = std::time::Instant::now();
    while m.any_active() {
        assert!(start.elapsed().as_secs() < 20, "import took too long");
        std::thread::sleep(std::time::Duration::from_millis(10));
    }
    assert_eq!(m.installed().len(), 1, "{:?}", m.jobs());
    m
}

/// Whether a runtime is there to run the fixtures: a build with one, and (for `onnx-dynamic`) the
/// library named by PHOTOCRAFT_ONNXRUNTIME, or PHOTOCRAFT_AI_TESTS=1 for a linked-in runtime.
fn runtime() -> bool {
    photocraft_ml::infer::set_gpu_allowed(false);
    let asked = std::env::var_os("PHOTOCRAFT_ONNXRUNTIME").is_some() || std::env::var_os("PHOTOCRAFT_AI_TESTS").is_some();
    if !(photocraft_ml::infer::runtime_available() && asked) {
        eprintln!("skipped: no model runtime");
        return false;
    }
    true
}

fn px(s: &Session, x: i32, y: i32) -> [f32; 4] {
    let st = s.active().unwrap();
    st.active_layer.and_then(|id| st.doc.layer(id)).unwrap().surface().unwrap().rgba(x, y)
}

fn selected(s: &Session, x: i32, y: i32) -> f32 {
    let mut p = [0.0f32; 1];
    match s.active().unwrap().doc.selection.as_ref() {
        Some(sel) => {
            sel.read_pixel(x, y, &mut p);
            p[0]
        }
        None => 0.0,
    }
}

/// A white canvas with a black bar from x = 10 to 40, on a pixel layer.
fn doc() -> Session {
    let mut s = Session::new();
    s.execute("file.new", json!({"width": 100, "height": 80})).unwrap();
    s.execute("layer.new.layer", json!({})).unwrap();
    s.execute("edit.fill", json!({"color": "#ffffff"})).unwrap();
    s.execute("paint.stroke", json!({"points": [[25, 0], [25, 80]], "size": 30, "hardness": 100, "color": "#000000"})).unwrap();
    s
}

#[test]
fn without_a_model_or_a_runtime_the_tools_say_why() {
    let mut s = doc();
    for id in ["select.aiObject", "paint.generativeRemoval", "paint.reflectionRemoval"] {
        assert!(!s.is_enabled(id), "{id} needs models");
    }
    let r = s.execute("ai.models", json!({})).unwrap();
    assert_eq!(r["models"]["removal"], json!([]));
    assert!(s.execute("ai.models", json!({"kind": "nonsense"})).is_err());
    s.set_models(Some(models("norun", "lama_like.onnx", &[ModelKind::Removal, ModelKind::Reflection])));
    assert!(s.is_enabled("paint.generativeRemoval") && s.is_enabled("paint.reflectionRemoval"));
    assert!(!s.is_enabled("select.aiObject"), "no selection model");
    let listed = s.execute("ai.models", json!({"kind": "reflection"})).unwrap();
    assert_eq!(listed["models"]["reflection"][0]["name"], "norun");
    for bad in [json!({}), json!({"points": [[1, 1]], "model": "missing"}), json!({"points": [[1, 1]], "model": 7}), json!({"points": "x"})] {
        assert!(s.execute("paint.generativeRemoval", bad).is_err());
    }
    if !photocraft_ml::infer::runtime_available() {
        let e = s.execute("paint.generativeRemoval", json!({"points": [[25, 40]], "size": 20})).unwrap_err().to_string();
        assert!(e.contains("runtime"), "{e}");
        assert_eq!(px(&s, 25, 40)[0], 0.0, "nothing changed");
    }
}

#[test]
fn generative_removal_fills_the_stroke_and_undoes() {
    if !runtime() {
        return;
    }
    let mut s = doc();
    s.set_models(Some(models("lama", "lama_like.onnx", &[ModelKind::Removal])));
    assert_eq!(px(&s, 25, 40)[0], 0.0);
    let r = s.execute("paint.generativeRemoval", json!({"points": [[25, 30], [25, 50]], "size": 34})).unwrap();
    assert!(r.get("damage").is_some(), "{r}");
    // The fixture fills holes with white.
    assert!(px(&s, 25, 40)[0] > 0.9, "{:?}", px(&s, 25, 40));
    assert_eq!(px(&s, 25, 5)[0], 0.0, "outside the stroke, unchanged");
    s.execute("edit.undo", json!({})).unwrap();
    assert_eq!(px(&s, 25, 40)[0], 0.0);
}

#[test]
fn ai_object_selection_selects_what_the_model_finds_in_the_box() {
    if !runtime() {
        return;
    }
    let mut s = doc();
    s.set_models(Some(models("matte", "matte_like_fp16.onnx", &[ModelKind::Segmentation])));
    // The fixture "finds" bright pixels: the white canvas, not the black bar.
    let r = s.execute("select.aiObject", json!({"rect": [0, 10, 100, 60]})).unwrap();
    assert_eq!(r["changed"], true, "{r}");
    assert!(selected(&s, 70, 40) > 0.9);
    assert!(selected(&s, 25, 40) < 0.1, "the bar isn't selected");
    assert!(selected(&s, 70, 2) < 0.1, "outside the box");
}
