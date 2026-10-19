use super::*;
use crate::infer::Image;
use crate::infer::net::Data;
use std::sync::atomic::{AtomicUsize, Ordering};

type Compute = Box<dyn Fn(&[(String, Tensor)]) -> Vec<(String, Tensor)> + Send + Sync>;
/// Each run's input names and shapes.
type Seen = Arc<Mutex<Vec<Vec<(String, Vec<i64>)>>>>;

/// A stand-in network that checks every input's shape against its values.
struct Fake {
    inputs: Vec<Port>,
    f: Compute,
    runs: Arc<AtomicUsize>,
    seen: Seen,
}

impl Network for Fake {
    fn inputs(&self) -> Vec<Port> {
        self.inputs.clone()
    }
    fn outputs(&self) -> Vec<Port> {
        Vec::new()
    }
    fn backend(&self) -> String {
        "Test".into()
    }
    fn run(&self, inputs: Vec<(String, Tensor)>) -> Result<Vec<(String, Tensor)>> {
        self.runs.fetch_add(1, Ordering::Relaxed);
        for p in &self.inputs {
            let t = inputs.iter().find(|(n, _)| *n == p.name).map(|(_, t)| t).ok_or_else(|| Error::Invalid(format!("missing input {}", p.name)))?;
            let expect: i64 = t.shape.iter().product();
            assert_eq!(expect.max(1) as usize, t.len().max(1), "{}: shape {:?} vs {} values", p.name, t.shape, t.len());
        }
        self.seen.lock().unwrap().push(inputs.iter().map(|(n, t)| (n.clone(), t.shape.clone())).collect());
        Ok((self.f)(&inputs))
    }
}

struct Fakes(Vec<(&'static str, Arc<dyn Network>)>);

impl Loader for Fakes {
    fn load(&self, path: &Path) -> Result<Arc<dyn Network>> {
        let n = file_name(path);
        self.0.iter().find(|(k, _)| n == *k).map(|(_, v)| v.clone()).ok_or(Error::NotFound(n))
    }
}

fn port(name: &str, elem: Elem, shape: &[i64]) -> Port {
    Port { name: name.into(), elem, shape: shape.to_vec() }
}

fn get<'a>(ins: &'a [(String, Tensor)], name: &str) -> &'a Tensor {
    &ins.iter().find(|(n, _)| n == name).unwrap().1
}

struct Rig {
    loader: Fakes,
    model: Model,
    unet_runs: Arc<AtomicUsize>,
    text_runs: Arc<AtomicUsize>,
    unet_seen: Seen,
}

/// A pipeline of stand-ins: the text encoder turns ids into numbers, the VAE averages 8 × 8
/// blocks (and the decoder repeats them back), and the UNet predicts no noise at all.
fn rig(tag: &str, channels: i64, batch: i64) -> Rig {
    let dir = std::env::temp_dir().join(format!("photocraft-diffusion-{tag}-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let mut vocab = serde_json::Map::new();
    for (i, t) in ["<|startoftext|>", "<|endoftext|>", "c", "a", "t</w>", "ca", "cat</w>"].iter().enumerate() {
        vocab.insert(t.to_string(), json_i(i as i64));
    }
    std::fs::write(dir.join("vocab.json"), Value::Object(vocab).to_string()).unwrap();
    std::fs::write(dir.join("merges.txt"), "#version: 0.2\nc a\nca t</w>\n").unwrap();
    std::fs::write(dir.join("scheduler_config.json"), r#"{"beta_schedule":"scaled_linear","beta_start":0.00085,"beta_end":0.012,"num_train_timesteps":1000,"steps_offset":1,"timestep_spacing":"leading"}"#).unwrap();
    let names = ["text_encoder.onnx", "unet.onnx", "vae_encoder.onnx", "vae_decoder.onnx", "vocab.json", "merges.txt", "scheduler_config.json"];
    for n in names.iter().filter(|n| n.ends_with(".onnx")) {
        std::fs::write(dir.join(n), b"x").unwrap();
    }
    let model = Model { name: tag.into(), files: names.iter().map(|n| dir.join(n)).collect(), catalog_id: None };
    let text_runs = Arc::new(AtomicUsize::new(0));
    let unet_runs = Arc::new(AtomicUsize::new(0));
    let unet_seen = Arc::new(Mutex::new(Vec::new()));
    let none = || Arc::new(Mutex::new(Vec::new()));
    let text = Fake {
        inputs: vec![port("input_ids", Elem::I32, &[-1, 77])],
        f: Box::new(|ins| {
            let ids = get(ins, "input_ids").to_f32();
            let v: Vec<f32> = ids.iter().flat_map(|i| [i / 10.0; 8]).collect();
            vec![("last_hidden_state".into(), Tensor::f32(&[1, 77, 8], v))]
        }),
        runs: text_runs.clone(),
        seen: none(),
    };
    let unet = Fake {
        inputs: vec![port("sample", Elem::F16, &[batch, channels, -1, -1]), port("timestep", Elem::F16, &[-1]), port("encoder_hidden_states", Elem::F16, &[batch, 77, 8])],
        f: Box::new(|ins| {
            let s = get(ins, "sample");
            let b = s.shape[0];
            let (h, w) = (s.shape[2], s.shape[3]);
            vec![("out_sample".into(), Tensor::f32(&[b, 4, h, w], vec![0.0; (b * 4 * h * w) as usize]))]
        }),
        runs: unet_runs.clone(),
        seen: unet_seen.clone(),
    };
    let vae_enc = Fake {
        inputs: vec![port("sample", Elem::F32, &[1, 3, -1, -1])],
        f: Box::new(|ins| {
            let s = get(ins, "sample");
            let (h, w) = (s.shape[2] as usize, s.shape[3] as usize);
            let v = s.to_f32();
            let (lh, lw) = (h / 8, w / 8);
            let mut out = vec![0.0f32; 4 * lh * lw];
            for c in 0..3 {
                let plane = &v[c * h * w..(c + 1) * h * w];
                let pooled = resize(plane, w, h, 1, lw, lh);
                out[c * lh * lw..(c + 1) * lh * lw].copy_from_slice(&pooled);
            }
            vec![("latent_sample".into(), Tensor::f32(&[1, 4, lh as i64, lw as i64], out))]
        }),
        runs: Arc::new(AtomicUsize::new(0)),
        seen: none(),
    };
    let vae_dec = Fake {
        inputs: vec![port("latent_sample", Elem::F32, &[1, 4, -1, -1])],
        f: Box::new(|ins| {
            let z = get(ins, "latent_sample");
            let (lh, lw) = (z.shape[2] as usize, z.shape[3] as usize);
            let v = z.to_f32();
            let mut out = Vec::with_capacity(3 * 64 * lh * lw);
            for c in 0..3 {
                for y in 0..lh * 8 {
                    for x in 0..lw * 8 {
                        out.push(v[c * lh * lw + (y / 8) * lw + x / 8]);
                    }
                }
            }
            vec![("sample".into(), Tensor::f32(&[1, 3, (lh * 8) as i64, (lw * 8) as i64], out))]
        }),
        runs: Arc::new(AtomicUsize::new(0)),
        seen: none(),
    };
    let loader = Fakes(vec![
        ("text_encoder.onnx", Arc::new(text) as Arc<dyn Network>),
        ("unet.onnx", Arc::new(unet)),
        ("vae_encoder.onnx", Arc::new(vae_enc)),
        ("vae_decoder.onnx", Arc::new(vae_dec)),
    ]);
    Rig { loader, model, unet_runs, text_runs, unet_seen }
}

fn json_i(i: i64) -> Value {
    Value::from(i)
}

fn go(_: f32, _: &str) -> bool {
    true
}

fn hole_in_middle(w: usize, h: usize) -> Vec<f32> {
    (0..w * h).map(|i| if (w / 4..3 * w / 4).contains(&(i % w)) && (h / 4..3 * h / 4).contains(&(i / w)) { 1.0 } else { 0.0 }).collect()
}

#[test]
fn the_schedule_matches_diffusers_ddim() {
    let config: Value = serde_json::from_str(r#"{"beta_schedule":"scaled_linear","beta_start":0.00085,"beta_end":0.012,"num_train_timesteps":1000,"steps_offset":1,"timestep_spacing":"leading"}"#).unwrap();
    let s = Schedule::new(&config, 25);
    assert_eq!(s.timesteps.len(), 25);
    assert_eq!(s.timesteps.first(), Some(&961));
    assert_eq!(s.timesteps.last(), Some(&1));
    assert_eq!(s.step_ratio, 40);
    // ᾱ falls from almost 1 to about 0.0047 (Stable Diffusion's schedule).
    assert!((s.alphas_cumprod[0] - 0.99915).abs() < 1e-5);
    assert!((s.alphas_cumprod[999] - 0.004_66).abs() < 3e-4);
    assert!(s.alphas_cumprod.windows(2).all(|w| w[1] < w[0]));
    let trailing = Schedule::new(&serde_json::json!({"timestep_spacing": "trailing"}), 4);
    assert_eq!(trailing.timesteps, vec![999, 749, 499, 249]);
    let lin = Schedule::new(&serde_json::json!({"timestep_spacing": "linspace"}), 3);
    assert_eq!(lin.timesteps, vec![999, 500, 0]);
}

#[test]
fn ddim_with_a_perfect_noise_prediction_lands_on_the_image() {
    // With ᾱ = 1 after the last step the result is exactly the image (SD's own setting leaves
    // a trace of noise, as diffusers does).
    let s = Schedule::new(&serde_json::json!({"set_alpha_to_one": true}), 30);
    let x0 = vec![0.3f32, -0.7, 0.05, 1.2];
    let noise = Rng(7).normals(4);
    let first = s.timesteps[0];
    let mut x = s.noised(&x0, &noise, Some(first));
    for &t in &s.timesteps {
        let a = s.alphas_cumprod[t];
        let eps: Vec<f32> = x.iter().zip(&x0).map(|(x, x0)| ((f64::from(*x) - a.sqrt() * f64::from(*x0)) / (1.0 - a).sqrt()) as f32).collect();
        x = s.step(&eps, t, &x);
    }
    for (a, b) in x.iter().zip(&x0) {
        assert!((a - b).abs() < 1e-3, "{a} vs {b}");
    }
    // v-prediction gives the same answer for the matching v.
    let mut sv = s.clone();
    sv.v_prediction = true;
    let t = sv.timesteps[3];
    let a = sv.alphas_cumprod[t];
    let xt = sv.noised(&x0, &noise, Some(t));
    let v: Vec<f32> = x0.iter().zip(&noise).map(|(x0, n)| (a.sqrt() * f64::from(*n) - (1.0 - a).sqrt() * f64::from(*x0)) as f32).collect();
    let by_v = sv.step(&v, t, &xt);
    let by_eps = s.step(&noise, t, &xt);
    for (p, q) in by_v.iter().zip(&by_eps) {
        assert!((p - q).abs() < 1e-4);
    }
}

#[test]
fn noise_is_seeded_and_normal() {
    let a = Rng(42).normals(10_000);
    assert_eq!(a, Rng(42).normals(10_000));
    assert_ne!(a, Rng(43).normals(10_000));
    let mean = a.iter().sum::<f32>() / a.len() as f32;
    let var = a.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / a.len() as f32;
    assert!(mean.abs() < 0.05 && (var - 1.0).abs() < 0.05, "{mean} {var}");
    assert!(new_seed() <= MAX_SEED);
}

#[test]
fn an_inpainting_pipeline_runs_guided_in_one_batch_and_repeats_with_a_seed() {
    let r = rig("inpaint", 9, -1);
    let img = Image::new(100, 60, vec![0.5; 100 * 60 * 3]).unwrap();
    let hole = hole_in_middle(100, 60);
    let opts = Prompting { prompt: "a cat".into(), negative: String::new(), seed: Some(1234), steps: 5, guidance: 7.5 };
    let mut msgs = Vec::new();
    let a = diffuse_with(&r.loader, &r.model, &img, &hole, &opts, &mut |f, m| {
        msgs.push((f, m.to_string()));
        true
    })
    .unwrap();
    assert_eq!(a.seed, 1234);
    assert_eq!(a.rgb.len(), 100 * 60 * 3);
    assert!(a.rgb.iter().all(|v| (0.0..=1.0).contains(v)));
    assert_eq!(r.unet_runs.load(Ordering::Relaxed), 5, "one batched run per step");
    assert_eq!(r.text_runs.load(Ordering::Relaxed), 2, "prompt and negative prompt");
    assert!(msgs.iter().any(|(_, m)| m == "Generating… step 5 of 5"));
    assert!(msgs.windows(2).all(|w| w[1].0 >= w[0].0), "progress only goes forward");
    let seen = r.unet_seen.lock().unwrap().clone();
    let shapes: HashMap<String, Vec<i64>> = seen[0].iter().cloned().collect();
    // 100 × 60 works at 640 × 384: latents 80 × 48, nine channels, uncond and cond together.
    assert_eq!(shapes["sample"], vec![2, 9, 48, 80]);
    assert_eq!(shapes["encoder_hidden_states"], vec![2, 77, 8]);
    assert_eq!(shapes["timestep"], vec![1]);
    // Same seed, same picture; another seed, another one.
    let b = diffuse_with(&r.loader, &r.model, &img, &hole, &opts, &mut go).unwrap();
    assert_eq!(a, b);
    let c = diffuse_with(&r.loader, &r.model, &img, &hole, &Prompting { seed: Some(99), ..opts.clone() }, &mut go).unwrap();
    assert_ne!(a.rgb, c.rgb);
    let fresh = diffuse_with(&r.loader, &r.model, &img, &hole, &Prompting { seed: None, ..opts }, &mut go).unwrap();
    assert!(fresh.seed <= MAX_SEED);
}

#[test]
fn a_batch_of_one_runs_twice_a_step_and_no_guidance_once() {
    let r = rig("batch1", 9, 1);
    let img = Image::new(64, 64, vec![0.2; 64 * 64 * 3]).unwrap();
    let hole = hole_in_middle(64, 64);
    let opts = Prompting { prompt: "cat".into(), steps: 3, seed: Some(5), ..Prompting::default() };
    diffuse_with(&r.loader, &r.model, &img, &hole, &opts, &mut go).unwrap();
    assert_eq!(r.unet_runs.load(Ordering::Relaxed), 6);
    let seen = r.unet_seen.lock().unwrap().clone();
    assert!(seen.iter().all(|s| s.iter().any(|(n, sh)| n == "sample" && sh[0] == 1)));
    diffuse_with(&r.loader, &r.model, &img, &hole, &Prompting { guidance: 1.0, ..opts }, &mut go).unwrap();
    assert_eq!(r.unet_runs.load(Ordering::Relaxed), 9);
}

#[test]
fn an_ordinary_unet_keeps_the_image_outside_the_hole() {
    let r = rig("txt2img", 4, -1);
    let (w, h) = (128, 128);
    let img = Image::new(w, h, vec![0.8; w * h * 3]).unwrap();
    let hole = hole_in_middle(w, h);
    let out = diffuse_with(&r.loader, &r.model, &img, &hole, &Prompting { steps: 4, seed: Some(3), ..Prompting::default() }, &mut go).unwrap();
    // A corner, far from the hole: the image itself.
    for c in 0..3 {
        assert!((out.rgb[c] - 0.8).abs() < 0.02, "{}", out.rgb[c]);
    }
    let seen = r.unet_seen.lock().unwrap().clone();
    assert!(seen[0].iter().any(|(n, sh)| n == "sample" && sh[1] == 4));
}

#[test]
fn cancelling_stops_and_a_checkpoint_says_what_to_get() {
    let r = rig("cancel", 9, -1);
    let img = Image::new(64, 64, vec![0.2; 64 * 64 * 3]).unwrap();
    let hole = hole_in_middle(64, 64);
    let mut calls = 0;
    let e = diffuse_with(&r.loader, &r.model, &img, &hole, &Prompting { steps: 10, ..Prompting::default() }, &mut |_, _| {
        calls += 1;
        calls < 6
    });
    assert!(matches!(e, Err(Error::Cancelled)));
    assert!(r.unet_runs.load(Ordering::Relaxed) < 10);
    let ckpt = Model { name: "Inpaint XL".into(), files: vec![PathBuf::from("/m/x/inpaint.safetensors")], catalog_id: None };
    assert!(!is_pipeline(&ckpt));
    let e = diffuse_with(&r.loader, &ckpt, &img, &hole, &Prompting::default(), &mut go).unwrap_err().to_string();
    assert!(e.contains("single-file Stable Diffusion checkpoint"), "{e}");
    assert!(matches!(diffuse_with(&r.loader, &r.model, &img, &[0.0; 3], &Prompting::default(), &mut go), Err(Error::Invalid(_))));
}

#[test]
fn pipeline_parts_are_found_in_folders_or_flat() {
    let m = Model {
        name: "p".into(),
        files: [
            "unet/model.onnx",
            "unet/model.onnx_data",
            "text_encoder/model.onnx",
            "vae_encoder/model.onnx",
            "vae_decoder/model.onnx",
            "vae_decoder/config.json",
            "tokenizer/vocab.json",
            "tokenizer/merges.txt",
            "tokenizer/tokenizer_config.json",
            "scheduler/scheduler_config.json",
        ]
        .iter()
        .map(|f| Path::new("/models/p").join(f))
        .collect(),
        catalog_id: None,
    };
    let p = parts(&m).unwrap();
    assert_eq!(p.unet, Path::new("/models/p/unet/model.onnx"));
    assert_eq!(p.text_encoder, Path::new("/models/p/text_encoder/model.onnx"));
    assert_eq!(p.vae_config.as_deref(), Some(Path::new("/models/p/vae_decoder/config.json")));
    assert_eq!(pipeline_dir(&p.unet), Path::new("/models/p"));
    assert_eq!(pipeline_dir(Path::new("/models/q/unet.onnx")), Path::new("/models/q"));
    assert!(is_pipeline_part(&p.unet) && !is_pipeline_part(Path::new("/models/lama/lama_fp32.onnx")));
    // 1000 × 250 keeps its shape within the limits.
    let any = port("sample", Elem::F32, &[-1, 9, -1, -1]);
    assert_eq!(work_size(1000, 250, &any), (768, 256));
    assert_eq!(work_size(512, 512, &any), (512, 512));
    assert_eq!(work_size(10, 10, &port("sample", Elem::F32, &[1, 9, 64, 64])), (512, 512));
    let _ = Data::F32(Vec::new());
}

use std::collections::HashMap;
