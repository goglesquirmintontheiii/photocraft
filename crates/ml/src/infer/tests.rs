use super::*;
use std::sync::atomic::AtomicUsize;

type Compute = Box<dyn Fn(&[(String, Tensor)]) -> Vec<(String, Tensor)> + Send + Sync>;

/// A stand-in network: fixed ports, and a closure for what it computes.
struct Fake {
    inputs: Vec<Port>,
    outputs: Vec<Port>,
    f: Compute,
    runs: Arc<AtomicUsize>,
}

impl Network for Fake {
    fn inputs(&self) -> Vec<Port> {
        self.inputs.clone()
    }
    fn outputs(&self) -> Vec<Port> {
        self.outputs.clone()
    }
    fn backend(&self) -> String {
        "Test".into()
    }
    fn run(&self, inputs: Vec<(String, Tensor)>) -> Result<Vec<(String, Tensor)>> {
        self.runs.fetch_add(1, Ordering::Relaxed);
        for p in &self.inputs {
            let t = inputs.iter().find(|(n, _)| *n == p.name).map(|(_, t)| t).ok_or_else(|| Error::Invalid(format!("missing input {}", p.name)))?;
            let expect: i64 = t.shape.iter().product();
            assert_eq!(expect as usize, t.len(), "{}: shape {:?} vs {} values", p.name, t.shape, t.len());
        }
        Ok((self.f)(&inputs))
    }
}

/// Loads fakes by file name.
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

fn input<'a>(ins: &'a [(String, Tensor)], name: &str) -> &'a Tensor {
    &ins.iter().find(|(n, _)| n == name).unwrap().1
}

fn temp_model(tag: &str, files: &[&str]) -> Model {
    let dir = std::env::temp_dir().join(format!("photocraft-infer-{tag}-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let files = files
        .iter()
        .map(|f| {
            let p = dir.join(f);
            std::fs::write(&p, b"x").unwrap();
            p
        })
        .collect();
    Model { name: tag.into(), files, catalog_id: None }
}

fn gray(w: usize, h: usize, v: f32) -> Image {
    Image::new(w, h, vec![v; w * h * 3]).unwrap()
}

fn go(_: f32, _: &str) -> bool {
    true
}

#[test]
fn resize_keeps_constants_and_sizes() {
    let src = vec![0.25f32; 10 * 6 * 3];
    let out = resize(&src, 10, 6, 3, 37, 3);
    assert_eq!(out.len(), 37 * 3 * 3);
    assert!(out.iter().all(|v| (v - 0.25).abs() < 1e-6));
    let big = vec![1.0f32; 4000 * 10];
    assert_eq!(resize(&big, 4000, 10, 1, 7, 7).len(), 49);
    assert!(resize(&[], 0, 0, 3, 4, 4).iter().all(|v| *v == 0.0));
    // A left-to-right ramp stays monotone.
    let ramp: Vec<f32> = (0..8).map(|x| x as f32).collect();
    let r = resize(&ramp, 8, 1, 1, 5, 1);
    assert!(r.windows(2).all(|p| p[0] <= p[1]));
}

#[test]
fn images_of_the_wrong_size_are_refused() {
    assert!(Image::new(0, 4, vec![]).is_err());
    assert!(Image::new(2, 2, vec![0.0; 11]).is_err());
    assert!(Image::new(100_000, 1, vec![0.0; 300_000]).is_err());
}

#[test]
fn lama_style_models_get_0_to_1_planes_and_give_back_0_to_255() {
    let runs = Arc::new(AtomicUsize::new(0));
    let lama: Arc<dyn Network> = Arc::new(Fake {
        inputs: vec![port("image", Elem::F32, &[1, 3, 512, 512]), port("mask", Elem::F32, &[1, 1, 512, 512])],
        outputs: vec![port("output", Elem::F32, &[1, 3, 512, 512])],
        f: Box::new(|ins| {
            let img = input(ins, "image").to_f32();
            let mask = input(ins, "mask").to_f32();
            assert!(img.iter().all(|v| (0.0..=1.0).contains(v)));
            assert!(mask.iter().all(|v| *v == 0.0 || *v == 1.0) && mask.contains(&1.0));
            // Fill the hole with white, keep the rest; 0–255 like Carve's export.
            let out: Vec<f32> = (0..3).flat_map(|_| mask.iter().zip(&img).map(|(m, v)| if *m > 0.5 { 255.0 } else { v * 255.0 })).collect();
            vec![("output".into(), Tensor::f32(&[1, 3, 512, 512], out))]
        }),
        runs: runs.clone(),
    });
    let model = temp_model("lama", &["lama_fp32.onnx"]);
    let img = gray(64, 40, 0.2);
    let mut hole = vec![0.0f32; 64 * 40];
    for y in 10..30 {
        for x in 20..40 {
            hole[y * 64 + x] = 1.0;
        }
    }
    let out = inpaint_with(&Fakes(vec![("lama_fp32.onnx", lama)]), &model, &img, &hole, &mut go).unwrap();
    assert_eq!(out.len(), 64 * 40 * 3);
    assert!(out[(20 * 64 + 30) * 3] > 0.9, "the hole is filled");
    assert!((out[(2 * 64 + 2) * 3] - 0.2).abs() < 0.02, "the rest comes back as it was");
    assert_eq!(runs.load(Ordering::Relaxed), 1);
}

#[test]
fn migan_gets_8_bit_and_a_keep_mask() {
    let migan: Arc<dyn Network> = Arc::new(Fake {
        inputs: vec![port("image", Elem::U8, &[1, 3, -1, -1]), port("mask", Elem::U8, &[1, 1, -1, -1])],
        outputs: vec![port("result", Elem::U8, &[1, 3, -1, -1])],
        f: Box::new(|ins| {
            let img = input(ins, "image");
            let m = input(ins, "mask");
            let (Data::U8(px), Data::U8(mask)) = (&img.data, &m.data) else { panic!("u8 inputs") };
            assert!(mask.iter().all(|v| *v == 0 || *v == 255));
            let n = mask.len();
            let out: Vec<u8> = (0..3 * n).map(|i| if mask[i % n] == 0 { 200 } else { px[i] }).collect();
            vec![("result".into(), Tensor::u8(&img.shape, out))]
        }),
        runs: Arc::new(AtomicUsize::new(0)),
    });
    let model = temp_model("migan", &["migan_pipeline_v2.onnx"]);
    let img = gray(30, 20, 0.5);
    let mut hole = vec![0.0f32; 600];
    hole[5 * 30 + 5] = 1.0;
    let out = inpaint_with(&Fakes(vec![("migan_pipeline_v2.onnx", migan)]), &model, &img, &hole, &mut go).unwrap();
    assert!((out[(5 * 30 + 5) * 3] - 200.0 / 255.0).abs() < 1e-4);
    assert!((out[0] - 128.0 / 255.0).abs() < 1e-2);
}

#[test]
fn a_one_input_image_model_is_an_image_to_image_fill() {
    let net: Arc<dyn Network> = Arc::new(Fake {
        inputs: vec![port("input", Elem::F32, &[1, 3, -1, -1])],
        outputs: vec![port("output", Elem::F32, &[1, 3, -1, -1])],
        f: Box::new(|ins| {
            let t = input(ins, "input");
            assert_eq!(t.shape[2] % 32, 0);
            vec![("output".into(), Tensor::f32(&t.shape, t.to_f32().iter().map(|v| 1.0 - v).collect()))]
        }),
        runs: Arc::new(AtomicUsize::new(0)),
    });
    let model = temp_model("dereflect", &["dsrnet.onnx"]);
    let out = inpaint_with(&Fakes(vec![("dsrnet.onnx", net)]), &model, &gray(50, 70, 0.25), &[1.0; 3500], &mut go).unwrap();
    assert!(out.iter().all(|v| (v - 0.75).abs() < 0.02));
}

#[test]
fn sam2_gets_points_or_a_box_in_the_1024_frame_and_the_best_mask_wins() {
    let enc_runs = Arc::new(AtomicUsize::new(0));
    let enc: Arc<dyn Network> = Arc::new(Fake {
        inputs: vec![port("pixel_values", Elem::F32, &[1, 3, 1024, 1024])],
        outputs: vec![port("image_embeddings.0", Elem::F32, &[1, 1, 1, 1]), port("image_embeddings.1", Elem::F32, &[1, 1, 1, 1])],
        f: Box::new(|_| {
            vec![("image_embeddings.0".into(), Tensor::f32(&[1, 1, 1, 1], vec![1.0])), ("image_embeddings.1".into(), Tensor::f32(&[1, 1, 1, 1], vec![2.0]))]
        }),
        runs: enc_runs.clone(),
    });
    let dec: Arc<dyn Network> = Arc::new(Fake {
        inputs: vec![
            port("input_points", Elem::F32, &[1, 1, -1, 2]),
            port("input_labels", Elem::I64, &[1, 1, -1]),
            port("input_boxes", Elem::F32, &[1, -1, 4]),
            port("image_embeddings.0", Elem::F32, &[1, 1, 1, 1]),
            port("image_embeddings.1", Elem::F32, &[1, 1, 1, 1]),
        ],
        outputs: vec![port("iou_scores", Elem::F32, &[1, 1, 3]), port("pred_masks", Elem::F32, &[1, 1, 3, 4, 4])],
        f: Box::new(|ins| {
            let boxes = input(ins, "input_boxes").to_f32();
            if !boxes.is_empty() {
                // A 200×100 image's (100,50)-(200,100) box, in the 1024 frame.
                assert_eq!(boxes, vec![512.0, 512.0, 1024.0, 1024.0]);
            }
            // Mask 1 (best score) covers the right half; the others nothing.
            let mut m = vec![-10.0f32; 3 * 16];
            for y in 0..4 {
                for x in 2..4 {
                    m[16 + y * 4 + x] = 10.0;
                }
            }
            vec![("iou_scores".into(), Tensor::f32(&[1, 1, 3], vec![0.1, 0.9, 0.3])), ("pred_masks".into(), Tensor::f32(&[1, 1, 3, 4, 4], m))]
        }),
        runs: Arc::new(AtomicUsize::new(0)),
    });
    let model = temp_model("sam2", &["vision_encoder.onnx", "vision_encoder.onnx_data", "prompt_encoder_mask_decoder.onnx"]);
    let loader = Fakes(vec![("vision_encoder.onnx", enc), ("prompt_encoder_mask_decoder.onnx", dec)]);
    let img = gray(200, 100, 0.4);
    let mask = segment_with(&loader, &model, &img, &Prompt::Box([100.0, 50.0, 200.0, 100.0]), &mut go).unwrap();
    assert_eq!(mask.len(), 200 * 100);
    assert!(mask[50 * 200 + 190] > 0.99 && mask[50 * 200 + 10] < 0.01);
    // Same image again: the embedding is reused.
    segment_with(&loader, &model, &img, &Prompt::Points(vec![([150.0, 50.0], true)]), &mut go).unwrap();
    assert_eq!(enc_runs.load(Ordering::Relaxed), 1);
}

#[test]
fn mobile_sam_gets_scaled_points_a_padding_point_and_the_image_size() {
    let enc: Arc<dyn Network> = Arc::new(Fake {
        inputs: vec![port("input_image", Elem::F32, &[-1, -1, 3])],
        outputs: vec![port("image_embeddings", Elem::F32, &[1, 256, 64, 64])],
        f: Box::new(|ins| {
            let t = input(ins, "input_image");
            assert_eq!(t.shape, vec![512, 1024, 3], "longest side 1024, HWC");
            assert!(t.to_f32().iter().all(|v| (v - 102.0).abs() < 0.5), "0–255");
            vec![("image_embeddings".into(), Tensor::f32(&[1, 1, 1, 1], vec![0.0]))]
        }),
        runs: Arc::new(AtomicUsize::new(0)),
    });
    let dec: Arc<dyn Network> = Arc::new(Fake {
        inputs: vec![
            port("image_embeddings", Elem::F32, &[1, 256, 64, 64]),
            port("point_coords", Elem::F32, &[1, -1, 2]),
            port("point_labels", Elem::F32, &[1, -1]),
            port("mask_input", Elem::F32, &[1, 1, 256, 256]),
            port("has_mask_input", Elem::F32, &[1]),
            port("orig_im_size", Elem::F32, &[2]),
        ],
        outputs: vec![port("masks", Elem::F32, &[-1, -1, -1, -1]), port("iou_predictions", Elem::F32, &[1, 1])],
        f: Box::new(|ins| {
            assert_eq!(input(ins, "point_coords").to_f32(), vec![512.0, 256.0, 0.0, 0.0]);
            assert_eq!(input(ins, "point_labels").to_f32(), vec![1.0, -1.0]);
            assert_eq!(input(ins, "orig_im_size").to_f32(), vec![300.0, 600.0]);
            vec![("masks".into(), Tensor::f32(&[1, 1, 300, 600], vec![5.0; 180_000])), ("iou_predictions".into(), Tensor::f32(&[1, 1], vec![0.9]))]
        }),
        runs: Arc::new(AtomicUsize::new(0)),
    });
    let model = temp_model("mobilesam", &["mobile_sam_image_encoder.onnx", "sam_mask_decoder_single.onnx"]);
    let loader = Fakes(vec![("mobile_sam_image_encoder.onnx", enc), ("sam_mask_decoder_single.onnx", dec)]);
    let mask = segment_with(&loader, &model, &gray(600, 300, 0.4), &Prompt::Points(vec![([300.0, 150.0], true)]), &mut go).unwrap();
    assert!(mask.iter().all(|v| *v > 0.99));
}

#[test]
fn a_matting_model_selects_inside_the_box_only() {
    let net: Arc<dyn Network> = Arc::new(Fake {
        inputs: vec![port("input_image", Elem::F32, &[1, 3, 1024, 1024])],
        outputs: vec![port("output_image", Elem::F32, &[1, 1, 1024, 1024])],
        f: Box::new(|_| vec![("output_image".into(), Tensor::f32(&[1, 1, 1024, 1024], vec![8.0; 1024 * 1024]))]),
        runs: Arc::new(AtomicUsize::new(0)),
    });
    let model = temp_model("birefnet", &["model.onnx"]);
    let mask = segment_with(&Fakes(vec![("model.onnx", net)]), &model, &gray(100, 100, 0.5), &Prompt::Box([20.0, 20.0, 60.0, 50.0]), &mut go).unwrap();
    assert!(mask[30 * 100 + 30] > 0.99, "inside the box");
    assert_eq!(mask[10 * 100 + 10], 0.0, "outside the box");
    assert_eq!(mask[55 * 100 + 30], 0.0, "in the margin, outside the box");
}

#[test]
fn cancelling_and_odd_models_are_errors_not_panics() {
    let model = temp_model("cancel", &["lama_fp32.onnx"]);
    let loader = Fakes(vec![]);
    let mut stop = |_: f32, _: &str| false;
    assert!(matches!(inpaint_with(&loader, &model, &gray(4, 4, 0.0), &[0.0; 16], &mut stop), Err(Error::Cancelled)));
    assert!(inpaint_with(&loader, &model, &gray(4, 4, 0.0), &[0.0; 3], &mut go).is_err(), "mask size");
    let sd = temp_model("sd", &["sd-v1-5-inpainting.safetensors"]);
    assert!(matches!(inpaint_with(&loader, &sd, &gray(4, 4, 0.0), &[0.0; 16], &mut go), Err(Error::Unsupported(_))));
    assert!(matches!(segment_with(&loader, &sd, &gray(4, 4, 0.0), &Prompt::Points(vec![]), &mut go), Err(Error::Unsupported(_))));
    // A network whose inputs make no sense for filling.
    let odd: Arc<dyn Network> =
        Arc::new(Fake { inputs: vec![port("x", Elem::F32, &[3])], outputs: vec![], f: Box::new(|_| vec![]), runs: Arc::new(AtomicUsize::new(0)) });
    let m = temp_model("odd", &["odd.onnx"]);
    assert!(matches!(inpaint_with(&Fakes(vec![("odd.onnx", odd)]), &m, &gray(4, 4, 0.0), &[1.0; 16], &mut go), Err(Error::Unsupported(_))));
}
