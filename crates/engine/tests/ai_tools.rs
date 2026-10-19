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

/// A models folder with the tiny Stable Diffusion pipeline fixture imported for Generative Fill.
fn diffusion_models(tag: &str) -> Manager {
    let dir = std::env::temp_dir().join(format!("photocraft-ai-tools-{tag}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    let m = Manager::open(&dir).unwrap();
    m.import_folder(&fixture("tiny_sd"), &[ModelKind::Inpainting], Some(tag.into())).unwrap();
    let start = std::time::Instant::now();
    while m.any_active() {
        assert!(start.elapsed().as_secs() < 20, "import took too long");
        std::thread::sleep(std::time::Duration::from_millis(10));
    }
    assert_eq!(m.installed().len(), 1, "{:?}", m.jobs());
    m
}

fn layer_count(s: &Session) -> usize {
    s.active().unwrap().doc.layers.len()
}

#[test]
fn infill_and_generative_fill_need_a_model_and_say_so() {
    let mut s = doc();
    assert!(!s.is_enabled("paint.generativeInfill"));
    s.set_models(Some(diffusion_models("sd-norun")));
    assert!(s.is_enabled("paint.generativeInfill"));
    assert!(!s.is_enabled("edit.generativeFill"), "needs a selection");
    s.execute("select.rect", json!({"x": 50, "y": 20, "width": 30, "height": 30})).unwrap();
    assert!(s.is_enabled("edit.generativeFill"));
    for bad in [json!({"seed": "x"}), json!({"seed": 5_000_000_000u64}), json!({"steps": 0}), json!({"guidance": 99}), json!({"prompt": 3})] {
        assert!(s.execute("edit.generativeFill", bad.clone()).is_err(), "{bad}");
    }
    if !photocraft_ml::infer::runtime_available() {
        let e = s.execute("edit.generativeFill", json!({"prompt": "a cat"})).unwrap_err().to_string();
        assert!(e.contains("runtime"), "{e}");
    }
}

#[test]
fn infill_fills_the_stroke_from_the_prompt_and_reports_the_seed() {
    if !runtime() {
        return;
    }
    let mut s = doc();
    s.set_models(Some(diffusion_models("sd-infill")));
    let before = px(&s, 70, 40);
    let r = s.execute("paint.generativeInfill", json!({"points": [[70, 30], [70, 50]], "size": 16, "prompt": "a cat", "seed": 9, "steps": 2})).unwrap();
    assert_eq!(r["seed"], 9, "{r}");
    assert_ne!(px(&s, 70, 40), before, "the stroke was filled");
    assert_eq!(px(&s, 95, 5), [1.0, 1.0, 1.0, 1.0], "outside the stroke, unchanged");
    s.execute("edit.undo", json!({})).unwrap();
    assert_eq!(px(&s, 70, 40), before);
}

#[test]
fn generative_fill_variations_replace_each_other_and_cancel_or_keep() {
    if !runtime() {
        return;
    }
    let mut s = doc();
    s.set_models(Some(diffusion_models("sd-fill")));
    s.execute("select.rect", json!({"x": 50, "y": 20, "width": 30, "height": 30})).unwrap();
    let layers = layer_count(&s);
    let p = json!({"prompt": "a cat", "steps": 2, "preview": true});
    let a = s.execute("edit.generativeFill", p.clone()).unwrap();
    assert_eq!(a["variation"], 0, "{a}");
    assert_eq!(layer_count(&s), layers + 1, "shown as a new layer");
    let top = s.active().unwrap().active_layer.and_then(|id| s.active().unwrap().doc.layer(id)).unwrap().name.clone();
    assert_eq!(top, "a cat");
    let b = s.execute("edit.generativeFill", p.clone()).unwrap();
    assert_eq!(b["variation"], 1);
    assert_eq!(b["count"], 2);
    assert_eq!(layer_count(&s), layers + 1, "the second replaced the first");
    let v = s.execute("edit.generativeFill.variations", json!({})).unwrap();
    assert_eq!(v["shown"], 1);
    assert_eq!(v["variations"].as_array().unwrap().len(), 2);
    let previews = s.generative_fill_previews();
    assert_eq!(previews.len(), 2);
    let (_, (tw, th, bytes)) = &previews[0];
    assert_eq!(bytes.len(), tw * th * 4);
    s.execute("edit.generativeFill.show", json!({"variation": 0})).unwrap();
    assert_eq!(layer_count(&s), layers + 1);
    // In the background, as the dialog runs it: the one shown is replaced when it lands.
    let id = match s.start("edit.generativeFill", p.clone()).unwrap() {
        photocraft_engine::jobs::Started::Job(id) => id,
        photocraft_engine::jobs::Started::Done(v) => panic!("ran inline: {v}"),
    };
    let c = s.wait_job(id).unwrap();
    assert_eq!(c["variation"], 2, "{c}");
    assert_eq!(layer_count(&s), layers + 1);
    assert_eq!(s.execute("edit.generativeFill.variations", json!({})).unwrap()["shown"], 2);
    assert!(s.execute("edit.generativeFill.show", json!({"variation": 5})).is_err());
    // Cancel takes it back.
    assert_eq!(s.execute("edit.generativeFill.cancel", json!({})).unwrap()["undone"], true);
    assert_eq!(layer_count(&s), layers);
    assert!(s.generative_fill_previews().is_empty());
    // OK keeps the one shown, as one step.
    s.execute("edit.generativeFill", p).unwrap();
    assert_eq!(s.execute("edit.generativeFill.commit", json!({})).unwrap()["kept"], 0);
    assert_eq!(layer_count(&s), layers + 1);
    s.execute("edit.undo", json!({})).unwrap();
    assert_eq!(layer_count(&s), layers);
    // Without the dialog it fills at once, on a layer named for the prompt.
    let r = s.execute("edit.generativeFill", json!({"steps": 2, "seed": 4})).unwrap();
    assert_eq!(r["seed"], 4);
    assert_eq!(layer_count(&s), layers + 1);
    let st = s.active().unwrap();
    assert_eq!(st.active_layer.and_then(|id| st.doc.layer(id)).unwrap().name, "Generative Fill");
}

#[test]
fn a_checkpoint_installed_before_it_could_run_is_prepared_on_first_use() {
    if !runtime() {
        return;
    }
    let m = models("ckpt-first-use", "tiny_sd1_ckpt.safetensors", &[ModelKind::Other]);
    let id = m.installed()[0].id.clone();
    m.set_kinds(&id, &[ModelKind::Inpainting]).unwrap();
    assert!(m.installed()[0].files.iter().any(|f| f.ends_with(".safetensors")));
    let mut s = doc();
    s.set_models(Some(m.clone()));
    let before = px(&s, 70, 40);
    let r = s.execute("paint.generativeInfill", json!({"points": [[70, 30], [70, 50]], "size": 16, "prompt": "a cat", "seed": 3, "steps": 2})).unwrap();
    assert_eq!(r["seed"], 3, "{r}");
    assert_ne!(px(&s, 70, 40), before);
    assert!(m.installed()[0].files.iter().any(|f| f == "unet/model.onnx"), "prepared in place");
}

#[test]
fn generative_enhance_and_sharpen_redraw_the_area() {
    if !runtime() {
        return;
    }
    // A super-resolution network: the brush.
    let mut s = doc();
    s.set_models(Some(models("sr", "sr_x2_like.onnx", &[ModelKind::Enhance])));
    assert!(s.is_enabled("paint.generativeEnhance"));
    let r = s.execute("paint.generativeEnhance", json!({"points": [[25, 30], [25, 50]], "size": 20, "detail": 0.8})).unwrap();
    assert!(r.get("damage").is_some(), "{r}");
    assert_eq!(r["seed"], serde_json::Value::Null, "seeds don't apply to a network");
    s.execute("edit.undo", json!({})).unwrap();
    assert!(s.execute("paint.generativeEnhance", json!({"points": [[25, 30]], "detail": 2})).is_err());
    // A diffusion pipeline: Generative Sharpen's variations.
    let mut s = doc();
    let m = diffusion_models("sd-sharpen");
    let id = m.installed()[0].id.clone();
    m.set_kinds(&id, &[ModelKind::Enhance]).unwrap();
    s.set_models(Some(m));
    s.execute("select.rect", json!({"x": 10, "y": 10, "width": 40, "height": 40})).unwrap();
    let layers = layer_count(&s);
    let a = s.execute("edit.generativeSharpen", json!({"steps": 4, "preview": true, "seed": 1})).unwrap();
    assert_eq!(a["variation"], 0, "{a}");
    let b = s.execute("edit.generativeSharpen", json!({"steps": 4, "preview": true, "seed": 2})).unwrap();
    assert_eq!(b["count"], 2);
    assert_eq!(layer_count(&s), layers + 1);
    assert_eq!(s.execute("edit.generativeFill.variations", json!({})).unwrap()["kind"], "sharpen");
    let st = s.active().unwrap();
    assert_eq!(st.active_layer.and_then(|l| st.doc.layer(l)).unwrap().name, "Generative Sharpen");
    s.execute("edit.generativeFill.cancel", json!({})).unwrap();
    assert_eq!(layer_count(&s), layers);
}

#[test]
fn a_stroke_over_an_object_selects_it() {
    let mut s = doc();
    assert!(s.execute("select.aiObject", json!({"points": []})).is_err());
    assert!(s.execute("select.aiObject", json!({"points": [[1, "x"]]})).is_err());
    if !runtime() {
        return;
    }
    s.set_models(Some(models("matte-stroke", "matte_like_fp16.onnx", &[ModelKind::Segmentation])));
    let pts: Vec<serde_json::Value> = (0..40).map(|i| json!([60 + i / 2, 30 + i])).collect();
    let r = s.execute("select.aiObject", json!({"points": pts})).unwrap();
    assert_eq!(r["changed"], true, "{r}");
    // The fixture "finds" bright pixels: the white canvas, not the black bar.
    assert!(selected(&s, 70, 40) > 0.9);
    assert!(selected(&s, 25, 40) < 0.1);
}

#[test]
fn an_sdxl_checkpoint_fills_after_it_is_prepared() {
    if !runtime() {
        return;
    }
    let m = models("sdxl-ckpt", "tiny_sdxl_ckpt.safetensors", &[ModelKind::Inpainting]);
    assert!(m.installed()[0].files.iter().any(|f| f == "text_encoder_2/model.onnx"), "{:?}", m.installed()[0].files);
    let mut s = doc();
    s.set_models(Some(m));
    let before = px(&s, 70, 40);
    let r = s.execute("paint.generativeInfill", json!({"points": [[70, 30], [70, 50]], "size": 16, "prompt": "a cat", "negativePrompt": "a dog", "seed": 5, "steps": 2})).unwrap();
    assert_eq!(r["seed"], 5, "{r}");
    assert_ne!(px(&s, 70, 40), before);
    assert_eq!(px(&s, 95, 5), [1.0, 1.0, 1.0, 1.0], "outside the stroke, unchanged");
}
