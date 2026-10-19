//! Recommended models: what the AI tab offers to download in one click.
//!
//! Every entry is a real file on its host with its exact size and, where the host publishes it,
//! its SHA-256; downloads are checked against both. Within a kind, entries are ordered most
//! accurate first, which is the order tools' model dropdowns use.

use crate::ModelKind;
use crate::source::RemoteFile;

#[derive(Debug)]
pub struct CatalogFile {
    pub url: &'static str,
    /// Name inside the model's folder.
    pub name: &'static str,
    pub size: u64,
    /// Lowercase hex; empty when the host doesn't publish one.
    pub sha256: &'static str,
}

impl CatalogFile {
    pub(crate) fn remote(&self) -> RemoteFile {
        RemoteFile {
            url: self.url.to_string(),
            name: self.name.to_string(),
            // The catalog size drives the progress bar; the download itself is checked against
            // the size and SHA-256 the host announces (Hugging Face's X-Linked-Size/-Etag).
            size: None,
            sha256: (!self.sha256.is_empty()).then(|| self.sha256.to_string()),
        }
    }
}

#[derive(Debug)]
pub struct CatalogModel {
    pub id: &'static str,
    pub name: &'static str,
    /// What it's used for; it shows in each of these sections of the AI Models tab.
    pub kinds: &'static [ModelKind],
    /// One line for the AI tab.
    pub description: &'static str,
    /// SPDX-style licence id, as the model's publisher states it.
    pub license: &'static str,
    /// The model's page, for its licence and details.
    pub homepage: &'static str,
    pub files: &'static [CatalogFile],
}

impl CatalogModel {
    pub fn size(&self) -> u64 {
        self.files.iter().map(|f| f.size).fold(0u64, u64::saturating_add)
    }

    /// The main use (the first section it's listed in).
    pub fn kind(&self) -> ModelKind {
        self.kinds.first().copied().unwrap_or(ModelKind::Other)
    }

    /// `org/name` of its Hugging Face repository, lowercase.
    pub fn repo(&self) -> String {
        self.homepage.trim_start_matches("https://huggingface.co/").trim_end_matches('/').to_ascii_lowercase()
    }
}

/// The recommended model a link points at: one of its files, or its repository (by URL or
/// `org/name`). `lower` is the link, lowercase.
pub fn by_link(lower: &str) -> Option<&'static CatalogModel> {
    let l = lower.trim().trim_end_matches('/');
    let l = l.split(['?', '#']).next().unwrap_or(l);
    // A file link names one model exactly ("blob" and "resolve" are the same file).
    let file = l.replace("/blob/", "/resolve/");
    if let Some(m) = RECOMMENDED.iter().find(|m| m.files.iter().any(|f| f.url.to_ascii_lowercase() == file)) {
        return Some(m);
    }
    let repo = l.trim_start_matches("https://").trim_start_matches("http://");
    let repo = repo.trim_start_matches("www.").trim_start_matches("huggingface.co/").trim_start_matches("hf.co/");
    let repo: String = repo.split('/').take(2).collect::<Vec<_>>().join("/");
    // Several recommended models can share a repository (the SD inpainting models); only a
    // repository with one counts.
    let mut found = RECOMMENDED.iter().filter(|m| m.repo() == repo);
    match (found.next(), found.next()) {
        (Some(m), None) => Some(m),
        _ => None,
    }
}

/// The recommended model an installed model is, from its files: every catalog file present
/// with its exact size (`files` are names and sizes on disk). With `link` (where it was
/// imported from) naming the model's repository, names alone are enough.
pub fn identify(files: &[(String, u64)], link: Option<&str>) -> Option<&'static CatalogModel> {
    let by_size = RECOMMENDED.iter().find(|m| m.files.iter().all(|f| files.iter().any(|(n, s)| n == f.name && *s == f.size)));
    by_size.or_else(|| {
        let m = by_link(&link?.to_ascii_lowercase())?;
        m.files.iter().all(|f| files.iter().any(|(n, _)| n == f.name)).then_some(m)
    })
}

pub static RECOMMENDED: &[CatalogModel] = &[
    CatalogModel {
        id: "sam-2-1-base-plus",
        name: "SAM 2.1 Base+",
        kinds: &[ModelKind::Segmentation],
        description: "Meta's Segment Anything 2.1, the most accurate selections",
        license: "Apache-2.0",
        homepage: "https://huggingface.co/onnx-community/sam2.1-hiera-base-plus-ONNX",
        files: &[
            CatalogFile {
                url: "https://huggingface.co/onnx-community/sam2.1-hiera-base-plus-ONNX/resolve/main/onnx/vision_encoder.onnx",
                name: "vision_encoder.onnx",
                size: 702_836,
                sha256: "",
            },
            CatalogFile {
                url: "https://huggingface.co/onnx-community/sam2.1-hiera-base-plus-ONNX/resolve/main/onnx/vision_encoder.onnx_data",
                name: "vision_encoder.onnx_data",
                size: 305_746_496,
                sha256: "",
            },
            CatalogFile {
                url: "https://huggingface.co/onnx-community/sam2.1-hiera-base-plus-ONNX/resolve/main/onnx/prompt_encoder_mask_decoder.onnx",
                name: "prompt_encoder_mask_decoder.onnx",
                size: 213_114,
                sha256: "",
            },
            CatalogFile {
                url: "https://huggingface.co/onnx-community/sam2.1-hiera-base-plus-ONNX/resolve/main/onnx/prompt_encoder_mask_decoder.onnx_data",
                name: "prompt_encoder_mask_decoder.onnx_data",
                size: 20_958_208,
                sha256: "",
            },
        ],
    },
    CatalogModel {
        id: "sam-2-1-small",
        name: "SAM 2.1 Small",
        kinds: &[ModelKind::Segmentation],
        description: "Segment Anything 2.1, balanced speed and accuracy",
        license: "Apache-2.0",
        homepage: "https://huggingface.co/onnx-community/sam2.1-hiera-small-ONNX",
        files: &[
            CatalogFile {
                url: "https://huggingface.co/onnx-community/sam2.1-hiera-small-ONNX/resolve/main/onnx/vision_encoder.onnx",
                name: "vision_encoder.onnx",
                size: 467_440,
                sha256: "",
            },
            CatalogFile {
                url: "https://huggingface.co/onnx-community/sam2.1-hiera-small-ONNX/resolve/main/onnx/vision_encoder.onnx_data",
                name: "vision_encoder.onnx_data",
                size: 162_476_288,
                sha256: "",
            },
            CatalogFile {
                url: "https://huggingface.co/onnx-community/sam2.1-hiera-small-ONNX/resolve/main/onnx/prompt_encoder_mask_decoder.onnx",
                name: "prompt_encoder_mask_decoder.onnx",
                size: 213_114,
                sha256: "",
            },
            CatalogFile {
                url: "https://huggingface.co/onnx-community/sam2.1-hiera-small-ONNX/resolve/main/onnx/prompt_encoder_mask_decoder.onnx_data",
                name: "prompt_encoder_mask_decoder.onnx_data",
                size: 20_958_208,
                sha256: "",
            },
        ],
    },
    CatalogModel {
        id: "sam-2-1-tiny",
        name: "SAM 2.1 Tiny",
        kinds: &[ModelKind::Segmentation],
        description: "Segment Anything 2.1, fastest of the three",
        license: "Apache-2.0",
        homepage: "https://huggingface.co/onnx-community/sam2.1-hiera-tiny-ONNX",
        files: &[
            CatalogFile {
                url: "https://huggingface.co/onnx-community/sam2.1-hiera-tiny-ONNX/resolve/main/onnx/vision_encoder.onnx",
                name: "vision_encoder.onnx",
                size: 354_238,
                sha256: "",
            },
            CatalogFile {
                url: "https://huggingface.co/onnx-community/sam2.1-hiera-tiny-ONNX/resolve/main/onnx/vision_encoder.onnx_data",
                name: "vision_encoder.onnx_data",
                size: 134_084_864,
                sha256: "",
            },
            CatalogFile {
                url: "https://huggingface.co/onnx-community/sam2.1-hiera-tiny-ONNX/resolve/main/onnx/prompt_encoder_mask_decoder.onnx",
                name: "prompt_encoder_mask_decoder.onnx",
                size: 213_114,
                sha256: "",
            },
            CatalogFile {
                url: "https://huggingface.co/onnx-community/sam2.1-hiera-tiny-ONNX/resolve/main/onnx/prompt_encoder_mask_decoder.onnx_data",
                name: "prompt_encoder_mask_decoder.onnx_data",
                size: 20_958_208,
                sha256: "",
            },
        ],
    },
    CatalogModel {
        id: "mobilesam",
        name: "MobileSAM",
        kinds: &[ModelKind::Segmentation],
        description: "Small, fast Segment Anything for older computers",
        license: "MIT",
        homepage: "https://huggingface.co/Acly/MobileSAM",
        files: &[
            CatalogFile {
                url: "https://huggingface.co/Acly/MobileSAM/resolve/main/mobile_sam_image_encoder.onnx",
                name: "mobile_sam_image_encoder.onnx",
                size: 28_157_093,
                sha256: "",
            },
            CatalogFile {
                url: "https://huggingface.co/Acly/MobileSAM/resolve/main/sam_mask_decoder_single.onnx",
                name: "sam_mask_decoder_single.onnx",
                size: 16_501_323,
                sha256: "",
            },
        ],
    },
    CatalogModel {
        id: "birefnet",
        name: "BiRefNet",
        kinds: &[ModelKind::Matting, ModelKind::Segmentation],
        description: "High-resolution subject and background separation",
        license: "MIT",
        homepage: "https://huggingface.co/onnx-community/BiRefNet-ONNX",
        files: &[CatalogFile {
            url: "https://huggingface.co/onnx-community/BiRefNet-ONNX/resolve/main/onnx/model_fp16.onnx",
            name: "model_fp16.onnx",
            size: 489_666_272,
            sha256: "",
        }],
    },
    CatalogModel {
        id: "birefnet-lite",
        name: "BiRefNet Lite",
        kinds: &[ModelKind::Matting, ModelKind::Segmentation],
        description: "Faster subject and background separation",
        license: "MIT",
        homepage: "https://huggingface.co/onnx-community/BiRefNet_lite-ONNX",
        files: &[CatalogFile {
            url: "https://huggingface.co/onnx-community/BiRefNet_lite-ONNX/resolve/main/onnx/model.onnx",
            name: "model.onnx",
            size: 224_005_088,
            sha256: "",
        }],
    },
    CatalogModel {
        id: "rmbg-1-4",
        name: "BRIA RMBG 1.4",
        kinds: &[ModelKind::Matting, ModelKind::Segmentation],
        description: "Background removal; non-commercial use only",
        license: "BRIA RMBG 1.4 (non-commercial)",
        homepage: "https://huggingface.co/briaai/RMBG-1.4",
        files: &[CatalogFile {
            url: "https://huggingface.co/briaai/RMBG-1.4/resolve/main/onnx/model_quantized.onnx",
            name: "model_quantized.onnx",
            size: 44_403_226,
            sha256: "",
        }],
    },
    CatalogModel {
        id: "lama",
        name: "LaMa",
        kinds: &[ModelKind::Removal, ModelKind::Reflection],
        description: "Large-mask inpainting: removes objects and people cleanly",
        license: "Apache-2.0",
        homepage: "https://huggingface.co/Carve/LaMa-ONNX",
        files: &[CatalogFile {
            url: "https://huggingface.co/Carve/LaMa-ONNX/resolve/main/lama_fp32.onnx",
            name: "lama_fp32.onnx",
            size: 208_044_816,
            sha256: "",
        }],
    },
    CatalogModel {
        id: "mi-gan",
        name: "MI-GAN",
        kinds: &[ModelKind::Removal, ModelKind::Reflection],
        description: "Very fast, light object removal",
        license: "MIT",
        homepage: "https://huggingface.co/andraniksargsyan/migan",
        files: &[CatalogFile {
            url: "https://huggingface.co/andraniksargsyan/migan/resolve/main/migan_pipeline_v2.onnx",
            name: "migan_pipeline_v2.onnx",
            size: 28_079_181,
            sha256: "",
        }],
    },
    CatalogModel {
        id: "reflection-removal-dinov3-s",
        name: "Reflection Removal (DINOv3-S)",
        kinds: &[ModelKind::Reflection],
        description: "Removes reflections on glass from the whole painted area, fast (640 px)",
        license: "MIT (DINOv3 backbone: DINOv3 License)",
        homepage: "https://github.com/PINTO0309/reflection-removal",
        files: &[CatalogFile {
            url: "https://github.com/PINTO0309/reflection-removal/releases/download/onnx/dinov3_vits16_disthyper_residual_gennerator_640x640_640x640.onnx",
            name: "dinov3_vits16_disthyper_residual_gennerator_640x640_640x640.onnx",
            size: 89_300_000,
            sha256: "f22232f1cc5b35f3395742bfbb3cf8dfcc9608ab72442835bcdfd62dc04ecc7f",
        }],
    },
    CatalogModel {
        id: "reflection-removal-dinov3-s-fast",
        name: "Reflection Removal (DINOv3-S, fastest)",
        kinds: &[ModelKind::Reflection],
        description: "The same model with a 320 px result, scaled up: quickest, softer",
        license: "MIT (DINOv3 backbone: DINOv3 License)",
        homepage: "https://github.com/PINTO0309/reflection-removal",
        files: &[CatalogFile {
            url: "https://github.com/PINTO0309/reflection-removal/releases/download/onnx/dinov3_vits16_disthyper_residual_gennerator_640x640_320x320.onnx",
            name: "dinov3_vits16_disthyper_residual_gennerator_640x640_320x320.onnx",
            size: 89_200_000,
            sha256: "99b197b7579f2022a38d8570ed94d972037304e5e689c51ba774e03f1f884b48",
        }],
    },
    CatalogModel {
        id: "realistic-vision-5-1-inpainting",
        name: "Realistic Vision 5.1 Inpainting",
        kinds: &[ModelKind::Inpainting],
        description: "Photorealistic prompted fills: people, places, products (Stable Diffusion 1.5)",
        license: "CreativeML OpenRAIL-M",
        homepage: "https://huggingface.co/RanaLLC/Realistic_Vision_V5.1-inpainting-onnx-fp16",
        files: &[
            CatalogFile {
                url: "https://huggingface.co/RanaLLC/Realistic_Vision_V5.1-inpainting-onnx-fp16/resolve/main/model_index.json",
                name: "model_index.json",
                size: 642,
                sha256: "",
            },
            CatalogFile {
                url: "https://huggingface.co/RanaLLC/Realistic_Vision_V5.1-inpainting-onnx-fp16/resolve/main/scheduler/scheduler_config.json",
                name: "scheduler/scheduler_config.json",
                size: 493,
                sha256: "",
            },
            CatalogFile {
                url: "https://huggingface.co/RanaLLC/Realistic_Vision_V5.1-inpainting-onnx-fp16/resolve/main/text_encoder/model.onnx",
                name: "text_encoder/model.onnx",
                size: 246_368_629,
                sha256: "8d0652c077b0877d2ef081ce45440130b67625c428492c85cd3595822845168d",
            },
            CatalogFile {
                url: "https://huggingface.co/RanaLLC/Realistic_Vision_V5.1-inpainting-onnx-fp16/resolve/main/tokenizer/vocab.json",
                name: "tokenizer/vocab.json",
                size: 1_059_962,
                sha256: "",
            },
            CatalogFile {
                url: "https://huggingface.co/RanaLLC/Realistic_Vision_V5.1-inpainting-onnx-fp16/resolve/main/tokenizer/merges.txt",
                name: "tokenizer/merges.txt",
                size: 524_619,
                sha256: "",
            },
            CatalogFile {
                url: "https://huggingface.co/RanaLLC/Realistic_Vision_V5.1-inpainting-onnx-fp16/resolve/main/tokenizer/special_tokens_map.json",
                name: "tokenizer/special_tokens_map.json",
                size: 472,
                sha256: "",
            },
            CatalogFile {
                url: "https://huggingface.co/RanaLLC/Realistic_Vision_V5.1-inpainting-onnx-fp16/resolve/main/tokenizer/tokenizer_config.json",
                name: "tokenizer/tokenizer_config.json",
                size: 734,
                sha256: "",
            },
            CatalogFile {
                url: "https://huggingface.co/RanaLLC/Realistic_Vision_V5.1-inpainting-onnx-fp16/resolve/main/unet/model.onnx",
                name: "unet/model.onnx",
                size: 1_720_173_976,
                sha256: "9824af9d5ac0a35acc6324c7351aab1a9b9e6c5672155b604c2165ea8753573c",
            },
            CatalogFile {
                url: "https://huggingface.co/RanaLLC/Realistic_Vision_V5.1-inpainting-onnx-fp16/resolve/main/vae_decoder/model.onnx",
                name: "vae_decoder/model.onnx",
                size: 99_093_889,
                sha256: "746f3ffce6ac67786cdb60e5a577b00ac62967d6324c07bccd167d10aedaa18f",
            },
            CatalogFile {
                url: "https://huggingface.co/RanaLLC/Realistic_Vision_V5.1-inpainting-onnx-fp16/resolve/main/vae_encoder/model.onnx",
                name: "vae_encoder/model.onnx",
                size: 68_430_178,
                sha256: "f0da9070d007def0d6a4e7c10a21462bb6172e460ef2587c3fe91191397b4cea",
            },
        ],
    },
    CatalogModel {
        id: "dreamshaper-8-inpainting",
        name: "DreamShaper 8 Inpainting",
        kinds: &[ModelKind::Inpainting],
        description: "Versatile prompted fills, photos to illustration (Stable Diffusion 1.5)",
        license: "CreativeML OpenRAIL-M",
        homepage: "https://huggingface.co/RanaLLC/dreamshaper-8-inpainting-onnx-fp16",
        files: &[
            CatalogFile {
                url: "https://huggingface.co/RanaLLC/dreamshaper-8-inpainting-onnx-fp16/resolve/main/model_index.json",
                name: "model_index.json",
                size: 651,
                sha256: "",
            },
            CatalogFile {
                url: "https://huggingface.co/RanaLLC/dreamshaper-8-inpainting-onnx-fp16/resolve/main/scheduler/scheduler_config.json",
                name: "scheduler/scheduler_config.json",
                size: 721,
                sha256: "",
            },
            CatalogFile {
                url: "https://huggingface.co/RanaLLC/dreamshaper-8-inpainting-onnx-fp16/resolve/main/text_encoder/model.onnx",
                name: "text_encoder/model.onnx",
                size: 246_368_629,
                sha256: "1fa485563ad556f9090e5c8dd439219d0c5e34311b4ee1dddb985cb5ac7ce3ab",
            },
            CatalogFile {
                url: "https://huggingface.co/RanaLLC/dreamshaper-8-inpainting-onnx-fp16/resolve/main/tokenizer/vocab.json",
                name: "tokenizer/vocab.json",
                size: 1_059_962,
                sha256: "",
            },
            CatalogFile {
                url: "https://huggingface.co/RanaLLC/dreamshaper-8-inpainting-onnx-fp16/resolve/main/tokenizer/merges.txt",
                name: "tokenizer/merges.txt",
                size: 524_619,
                sha256: "",
            },
            CatalogFile {
                url: "https://huggingface.co/RanaLLC/dreamshaper-8-inpainting-onnx-fp16/resolve/main/tokenizer/special_tokens_map.json",
                name: "tokenizer/special_tokens_map.json",
                size: 472,
                sha256: "",
            },
            CatalogFile {
                url: "https://huggingface.co/RanaLLC/dreamshaper-8-inpainting-onnx-fp16/resolve/main/tokenizer/tokenizer_config.json",
                name: "tokenizer/tokenizer_config.json",
                size: 734,
                sha256: "",
            },
            CatalogFile {
                url: "https://huggingface.co/RanaLLC/dreamshaper-8-inpainting-onnx-fp16/resolve/main/unet/model.onnx",
                name: "unet/model.onnx",
                size: 1_720_173_976,
                sha256: "535555de4991dc38b8f3393587dff1b5aa604d7da62bb5eb0123b317c7a9bb12",
            },
            CatalogFile {
                url: "https://huggingface.co/RanaLLC/dreamshaper-8-inpainting-onnx-fp16/resolve/main/vae_decoder/model.onnx",
                name: "vae_decoder/model.onnx",
                size: 99_093_889,
                sha256: "b992c5beaf92d39f1b2b48b1f03e4a077be3ce6314e1f353f2204f13f1ad5bcd",
            },
            CatalogFile {
                url: "https://huggingface.co/RanaLLC/dreamshaper-8-inpainting-onnx-fp16/resolve/main/vae_encoder/model.onnx",
                name: "vae_encoder/model.onnx",
                size: 68_430_178,
                sha256: "6939389cb60609cf8058e17150bd6bdb8a1592558770446307287c5effd7191b",
            },
        ],
    },
    CatalogModel {
        id: "absolute-reality-inpainting",
        name: "AbsoluteReality 1.6525 Inpainting",
        kinds: &[ModelKind::Inpainting],
        description: "Realistic prompted fills with soft, natural light (Stable Diffusion 1.5)",
        license: "CreativeML OpenRAIL-M",
        homepage: "https://huggingface.co/RanaLLC/absolute-reality-1.6525-inpainting-onnx-fp16",
        files: &[
            CatalogFile {
                url: "https://huggingface.co/RanaLLC/absolute-reality-1.6525-inpainting-onnx-fp16/resolve/main/model_index.json",
                name: "model_index.json",
                size: 651,
                sha256: "",
            },
            CatalogFile {
                url: "https://huggingface.co/RanaLLC/absolute-reality-1.6525-inpainting-onnx-fp16/resolve/main/scheduler/scheduler_config.json",
                name: "scheduler/scheduler_config.json",
                size: 721,
                sha256: "",
            },
            CatalogFile {
                url: "https://huggingface.co/RanaLLC/absolute-reality-1.6525-inpainting-onnx-fp16/resolve/main/text_encoder/model.onnx",
                name: "text_encoder/model.onnx",
                size: 246_368_629,
                sha256: "8cc6c02ba98165e0822a1afa80506d7e3abda0afaab1314d6deaa6d30b59e46b",
            },
            CatalogFile {
                url: "https://huggingface.co/RanaLLC/absolute-reality-1.6525-inpainting-onnx-fp16/resolve/main/tokenizer/vocab.json",
                name: "tokenizer/vocab.json",
                size: 1_059_962,
                sha256: "",
            },
            CatalogFile {
                url: "https://huggingface.co/RanaLLC/absolute-reality-1.6525-inpainting-onnx-fp16/resolve/main/tokenizer/merges.txt",
                name: "tokenizer/merges.txt",
                size: 524_619,
                sha256: "",
            },
            CatalogFile {
                url: "https://huggingface.co/RanaLLC/absolute-reality-1.6525-inpainting-onnx-fp16/resolve/main/tokenizer/special_tokens_map.json",
                name: "tokenizer/special_tokens_map.json",
                size: 472,
                sha256: "",
            },
            CatalogFile {
                url: "https://huggingface.co/RanaLLC/absolute-reality-1.6525-inpainting-onnx-fp16/resolve/main/tokenizer/tokenizer_config.json",
                name: "tokenizer/tokenizer_config.json",
                size: 734,
                sha256: "",
            },
            CatalogFile {
                url: "https://huggingface.co/RanaLLC/absolute-reality-1.6525-inpainting-onnx-fp16/resolve/main/unet/model.onnx",
                name: "unet/model.onnx",
                size: 1_720_173_976,
                sha256: "b84c95e7d9f6bdcc0ce536c68bb727da3593b7afa63978591effc752ec048fcf",
            },
            CatalogFile {
                url: "https://huggingface.co/RanaLLC/absolute-reality-1.6525-inpainting-onnx-fp16/resolve/main/vae_decoder/model.onnx",
                name: "vae_decoder/model.onnx",
                size: 99_093_889,
                sha256: "99e90cbf50fcc39e8701206be456c2e872cee4ff72f5ed63a64307ece88be1d3",
            },
            CatalogFile {
                url: "https://huggingface.co/RanaLLC/absolute-reality-1.6525-inpainting-onnx-fp16/resolve/main/vae_encoder/model.onnx",
                name: "vae_encoder/model.onnx",
                size: 68_430_178,
                sha256: "f0da9070d007def0d6a4e7c10a21462bb6172e460ef2587c3fe91191397b4cea",
            },
        ],
    },
    CatalogModel {
        id: "sd-2-inpainting-onnx",
        name: "Stable Diffusion 2 Inpainting",
        kinds: &[ModelKind::Inpainting],
        description: "Stability AI's general-purpose prompted inpainting",
        license: "OpenRAIL++-M",
        homepage: "https://huggingface.co/RanaLLC/stable-diffusion-2-inpainting-onnx-fp16",
        files: &[
            CatalogFile {
                url: "https://huggingface.co/RanaLLC/stable-diffusion-2-inpainting-onnx-fp16/resolve/main/model_index.json",
                name: "model_index.json",
                size: 594,
                sha256: "",
            },
            CatalogFile {
                url: "https://huggingface.co/RanaLLC/stable-diffusion-2-inpainting-onnx-fp16/resolve/main/scheduler/scheduler_config.json",
                name: "scheduler/scheduler_config.json",
                size: 374,
                sha256: "",
            },
            CatalogFile {
                url: "https://huggingface.co/RanaLLC/stable-diffusion-2-inpainting-onnx-fp16/resolve/main/text_encoder/model.onnx",
                name: "text_encoder/model.onnx",
                size: 681_257_528,
                sha256: "8d52bd737a50e31e965a0d1e18ef22133e4bffec50efa9b14246472e00394e5d",
            },
            CatalogFile {
                url: "https://huggingface.co/RanaLLC/stable-diffusion-2-inpainting-onnx-fp16/resolve/main/tokenizer/vocab.json",
                name: "tokenizer/vocab.json",
                size: 1_059_962,
                sha256: "",
            },
            CatalogFile {
                url: "https://huggingface.co/RanaLLC/stable-diffusion-2-inpainting-onnx-fp16/resolve/main/tokenizer/merges.txt",
                name: "tokenizer/merges.txt",
                size: 524_619,
                sha256: "",
            },
            CatalogFile {
                url: "https://huggingface.co/RanaLLC/stable-diffusion-2-inpainting-onnx-fp16/resolve/main/tokenizer/special_tokens_map.json",
                name: "tokenizer/special_tokens_map.json",
                size: 460,
                sha256: "",
            },
            CatalogFile {
                url: "https://huggingface.co/RanaLLC/stable-diffusion-2-inpainting-onnx-fp16/resolve/main/tokenizer/tokenizer_config.json",
                name: "tokenizer/tokenizer_config.json",
                size: 886,
                sha256: "",
            },
            CatalogFile {
                url: "https://huggingface.co/RanaLLC/stable-diffusion-2-inpainting-onnx-fp16/resolve/main/unet/model.onnx",
                name: "unet/model.onnx",
                size: 1_732_953_907,
                sha256: "a0ebd54f7e4f926b0172b907cb7ee26dde7abc19054085159e2375eac3fbbaa6",
            },
            CatalogFile {
                url: "https://huggingface.co/RanaLLC/stable-diffusion-2-inpainting-onnx-fp16/resolve/main/vae_decoder/model.onnx",
                name: "vae_decoder/model.onnx",
                size: 99_093_889,
                sha256: "99e90cbf50fcc39e8701206be456c2e872cee4ff72f5ed63a64307ece88be1d3",
            },
            CatalogFile {
                url: "https://huggingface.co/RanaLLC/stable-diffusion-2-inpainting-onnx-fp16/resolve/main/vae_encoder/model.onnx",
                name: "vae_encoder/model.onnx",
                size: 68_430_178,
                sha256: "f0da9070d007def0d6a4e7c10a21462bb6172e460ef2587c3fe91191397b4cea",
            },
        ],
    },
    CatalogModel {
        id: "sd-1-5-inpainting-onnx",
        name: "Stable Diffusion 1.5 Inpainting",
        kinds: &[ModelKind::Inpainting],
        description: "The classic prompted inpainting model",
        license: "CreativeML OpenRAIL-M",
        homepage: "https://huggingface.co/RanaLLC/stable-diffusion-v1-5-inpainting-onnx-fp16",
        files: &[
            CatalogFile {
                url: "https://huggingface.co/RanaLLC/stable-diffusion-v1-5-inpainting-onnx-fp16/resolve/main/model_index.json",
                name: "model_index.json",
                size: 640,
                sha256: "",
            },
            CatalogFile {
                url: "https://huggingface.co/RanaLLC/stable-diffusion-v1-5-inpainting-onnx-fp16/resolve/main/scheduler/scheduler_config.json",
                name: "scheduler/scheduler_config.json",
                size: 528,
                sha256: "",
            },
            CatalogFile {
                url: "https://huggingface.co/RanaLLC/stable-diffusion-v1-5-inpainting-onnx-fp16/resolve/main/text_encoder/model.onnx",
                name: "text_encoder/model.onnx",
                size: 246_368_629,
                sha256: "8af516fb184866f8caed192ca3bf0636ef9d40d4f7799c05600cae431d18d8d1",
            },
            CatalogFile {
                url: "https://huggingface.co/RanaLLC/stable-diffusion-v1-5-inpainting-onnx-fp16/resolve/main/tokenizer/vocab.json",
                name: "tokenizer/vocab.json",
                size: 1_059_962,
                sha256: "",
            },
            CatalogFile {
                url: "https://huggingface.co/RanaLLC/stable-diffusion-v1-5-inpainting-onnx-fp16/resolve/main/tokenizer/merges.txt",
                name: "tokenizer/merges.txt",
                size: 524_619,
                sha256: "",
            },
            CatalogFile {
                url: "https://huggingface.co/RanaLLC/stable-diffusion-v1-5-inpainting-onnx-fp16/resolve/main/tokenizer/special_tokens_map.json",
                name: "tokenizer/special_tokens_map.json",
                size: 472,
                sha256: "",
            },
            CatalogFile {
                url: "https://huggingface.co/RanaLLC/stable-diffusion-v1-5-inpainting-onnx-fp16/resolve/main/tokenizer/tokenizer_config.json",
                name: "tokenizer/tokenizer_config.json",
                size: 735,
                sha256: "",
            },
            CatalogFile {
                url: "https://huggingface.co/RanaLLC/stable-diffusion-v1-5-inpainting-onnx-fp16/resolve/main/unet/model.onnx",
                name: "unet/model.onnx",
                size: 1_720_173_976,
                sha256: "a3726e1944ec2fbc9596096cd001520aea789367f329360ee34eb520c751de16",
            },
            CatalogFile {
                url: "https://huggingface.co/RanaLLC/stable-diffusion-v1-5-inpainting-onnx-fp16/resolve/main/vae_decoder/model.onnx",
                name: "vae_decoder/model.onnx",
                size: 99_093_889,
                sha256: "ce0ffca3fcfd0a2729a88d8d90dabf35c66d237404997e0461d52e46ac7a91bc",
            },
            CatalogFile {
                url: "https://huggingface.co/RanaLLC/stable-diffusion-v1-5-inpainting-onnx-fp16/resolve/main/vae_encoder/model.onnx",
                name: "vae_encoder/model.onnx",
                size: 68_430_178,
                sha256: "f0da9070d007def0d6a4e7c10a21462bb6172e460ef2587c3fe91191397b4cea",
            },
        ],
    },
    CatalogModel {
        id: "sd-1-5-inpainting-fp32",
        name: "Stable Diffusion 1.5 Inpainting (full precision)",
        kinds: &[ModelKind::Inpainting],
        description: "Full-precision weights: the one to use without a GPU (slow)",
        license: "CreativeML OpenRAIL-M",
        homepage: "https://huggingface.co/JasonYANG170/sd15-inpainting-onnx-fp32",
        files: &[
            CatalogFile {
                url: "https://huggingface.co/JasonYANG170/sd15-inpainting-onnx-fp32/resolve/main/model_index.json",
                name: "model_index.json",
                size: 548,
                sha256: "",
            },
            CatalogFile {
                url: "https://huggingface.co/JasonYANG170/sd15-inpainting-onnx-fp32/resolve/main/scheduler/scheduler_config.json",
                name: "scheduler/scheduler_config.json",
                size: 389,
                sha256: "",
            },
            CatalogFile {
                url: "https://huggingface.co/JasonYANG170/sd15-inpainting-onnx-fp32/resolve/main/text_encoder/model.onnx",
                name: "text_encoder/model.onnx",
                size: 492_481_988,
                sha256: "f733b4c88a6e45e88460e00d683a465b546ac7b8c307db4a3b33c49f939a9789",
            },
            CatalogFile {
                url: "https://huggingface.co/JasonYANG170/sd15-inpainting-onnx-fp32/resolve/main/tokenizer/vocab.json",
                name: "tokenizer/vocab.json",
                size: 1_109_372,
                sha256: "",
            },
            CatalogFile {
                url: "https://huggingface.co/JasonYANG170/sd15-inpainting-onnx-fp32/resolve/main/tokenizer/merges.txt",
                name: "tokenizer/merges.txt",
                size: 573_514,
                sha256: "",
            },
            CatalogFile {
                url: "https://huggingface.co/JasonYANG170/sd15-inpainting-onnx-fp32/resolve/main/tokenizer/special_tokens_map.json",
                name: "tokenizer/special_tokens_map.json",
                size: 496,
                sha256: "",
            },
            CatalogFile {
                url: "https://huggingface.co/JasonYANG170/sd15-inpainting-onnx-fp32/resolve/main/tokenizer/tokenizer_config.json",
                name: "tokenizer/tokenizer_config.json",
                size: 766,
                sha256: "",
            },
            CatalogFile {
                url: "https://huggingface.co/JasonYANG170/sd15-inpainting-onnx-fp32/resolve/main/unet/model.onnx",
                name: "unet/model.onnx",
                size: 1_973_332,
                sha256: "d6abc46fc2bf96e4fca7dff950beaa67646f9bd0bdd93fcf8621ef122ae8feab",
            },
            CatalogFile {
                url: "https://huggingface.co/JasonYANG170/sd15-inpainting-onnx-fp32/resolve/main/unet/model.onnx_data",
                name: "unet/model.onnx_data",
                size: 3_438_141_440,
                sha256: "df01d68458269b62670c8e287743c2a59f613b21039a5d8937ef2d522c677fe9",
            },
            CatalogFile {
                url: "https://huggingface.co/JasonYANG170/sd15-inpainting-onnx-fp32/resolve/main/vae_decoder/model.onnx",
                name: "vae_decoder/model.onnx",
                size: 265_178_525,
                sha256: "668fa7d9b511fa5ce87280b4f0b0e5177483de7c3ffea7166b47c2b7284a47f2",
            },
            CatalogFile {
                url: "https://huggingface.co/JasonYANG170/sd15-inpainting-onnx-fp32/resolve/main/vae_encoder/model.onnx",
                name: "vae_encoder/model.onnx",
                size: 136_773_617,
                sha256: "8912a7e136a1c3ea21699d170a0e98e89608f8eb8213ba27f7e1a2709aebc517",
            },
        ],
    },
];

#[cfg(test)]
mod tests {
    use super::*;
    use crate::store::{valid_id, valid_model_path};

    #[test]
    fn models_are_recognised_by_link_and_by_files() {
        assert_eq!(by_link("https://huggingface.co/acly/mobilesam").map(|m| m.id), Some("mobilesam"));
        assert_eq!(by_link("acly/mobilesam").map(|m| m.id), Some("mobilesam"));
        assert_eq!(by_link("https://huggingface.co/carve/lama-onnx/blob/main/lama_fp32.onnx").map(|m| m.id), Some("lama"));
        assert_eq!(by_link("ranallc/dreamshaper-8-inpainting-onnx-fp16").map(|m| m.id), Some("dreamshaper-8-inpainting"));
        assert_eq!(
            by_link("https://huggingface.co/ranallc/stable-diffusion-v1-5-inpainting-onnx-fp16/blob/main/unet/model.onnx").map(|m| m.id),
            Some("sd-1-5-inpainting-onnx")
        );
        assert!(by_link("https://example.com/x.onnx").is_none());
        let mobile = vec![("mobile_sam_image_encoder.onnx".to_string(), 28_157_093), ("sam_mask_decoder_single.onnx".to_string(), 16_501_323)];
        assert_eq!(identify(&mobile, None).map(|m| m.id), Some("mobilesam"));
        let resized = vec![("mobile_sam_image_encoder.onnx".to_string(), 1), ("sam_mask_decoder_single.onnx".to_string(), 2)];
        assert!(identify(&resized, None).is_none());
        assert_eq!(identify(&resized, Some("https://huggingface.co/Acly/MobileSAM")).map(|m| m.id), Some("mobilesam"));
        let lama = vec![("lama_fp32.onnx".to_string(), 208_044_816), ("readme.md".to_string(), 10)];
        assert_eq!(identify(&lama, None).map(|m| m.id), Some("lama"));
        assert!(identify(&[], None).is_none());
    }

    #[test]
    fn every_entry_is_well_formed() {
        let mut ids = std::collections::HashSet::new();
        for m in RECOMMENDED {
            assert!(valid_id(m.id) && ids.insert(m.id), "{}", m.id);
            assert!(!m.files.is_empty(), "{}", m.id);
            assert!(!m.kinds.is_empty() && crate::kind::normalize(m.kinds).len() == m.kinds.len(), "{}", m.id);
            assert!(m.homepage.starts_with("https://"), "{}", m.id);
            let mut names = std::collections::HashSet::new();
            for f in m.files {
                assert!(f.url.starts_with("https://"), "{}", f.url);
                assert!(valid_model_path(f.name) && names.insert(f.name), "{}", f.name);
                assert!(f.size > 0, "{}", f.name);
                assert!(f.sha256.is_empty() || crate::source::is_sha256(f.sha256) && f.sha256 == f.sha256.to_ascii_lowercase(), "{}", f.name);
            }
        }
    }
}
