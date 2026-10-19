use super::*;

fn fixture(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures").join(name)
}

pub(crate) fn tiny_options(sd2: bool) -> Options {
    Options { unet_heads: Some(if sd2 { Heads::Dim(16) } else { Heads::Count(4) }), text_heads: Some(4), full_precision: false }
}

#[test]
fn tiny_checkpoints_are_recognised_and_converted() {
    for (file, version) in [("tiny_sd1_ckpt.safetensors", 1), ("tiny_sd2_ckpt.safetensors", 2)] {
        let ck = SafeTensors::open(&fixture(file)).unwrap();
        let info = inspect(&ck, "tiny").unwrap();
        assert_eq!(info.version, version);
        assert_eq!(info.in_channels, 9);
        assert!(info.has_vae);
        assert!(!info.v_prediction, "inpainting models predict noise");
        let out = std::env::temp_dir().join(format!("photocraft-convert-{version}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&out);
        let mut last = 0.0;
        let files = convert(&fixture(file), &out, &[], &tiny_options(version == 2), &mut |f| {
            assert!(f >= last);
            last = f;
            true
        })
        .unwrap();
        assert_eq!(last, 1.0);
        for f in &files {
            assert!(out.join(f).is_file(), "{f}");
            assert!(crate::store::valid_model_path(f), "{f}");
        }
        let model = crate::infer::Model { name: "t".into(), files: files.iter().map(|f| out.join(f)).collect(), catalog_id: None };
        assert!(crate::infer::diffusion::is_pipeline(&model));
        assert!(checkpoint_to_convert(&model.files).is_none());
        let _ = std::fs::remove_dir_all(out);
    }
}

#[test]
fn cancelling_stops_and_other_files_say_what_they_are() {
    let out = std::env::temp_dir().join(format!("photocraft-convert-cancel-{}", std::process::id()));
    let mut n = 0;
    let r = convert(&fixture("tiny_sd1_ckpt.safetensors"), &out, &[], &tiny_options(false), &mut |_| {
        n += 1;
        n < 20
    });
    assert!(matches!(r, Err(Error::Cancelled)));
    let _ = std::fs::remove_dir_all(&out);
    // Not a checkpoint at all.
    assert!(convert(&fixture("lama_like.onnx"), &out, &[], &Options::default(), &mut |_| true).is_err());
    assert_eq!(checkpoint_to_convert(&[PathBuf::from("/m/a.safetensors")]), Some(&PathBuf::from("/m/a.safetensors")));
}
