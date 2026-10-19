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
        id: "sam-2-1-large",
        name: "SAM 2.1 Large",
        kinds: &[ModelKind::Segmentation],
        description: "Meta's Segment Anything 2.1 at its largest: the most accurate selections, slower",
        license: "Apache-2.0",
        homepage: "https://huggingface.co/onnx-community/sam2.1-hiera-large-ONNX",
        files: &[
            CatalogFile {
                url: "https://huggingface.co/onnx-community/sam2.1-hiera-large-ONNX/resolve/main/onnx/vision_encoder.onnx",
                name: "vision_encoder.onnx",
                size: 1_350_487,
                sha256: "ad3a8161c6bd6d8beea811300c60e20d626035feaa3c19622ace27b22e25bec8",
            },
            CatalogFile {
                url: "https://huggingface.co/onnx-community/sam2.1-hiera-large-ONNX/resolve/main/onnx/vision_encoder.onnx_data",
                name: "vision_encoder.onnx_data",
                size: 888_587_456,
                sha256: "88c501a65b6aaeac99b65a275f3f559727404283930d107d499fda8714ba7e5a",
            },
            CatalogFile {
                url: "https://huggingface.co/onnx-community/sam2.1-hiera-large-ONNX/resolve/main/onnx/prompt_encoder_mask_decoder.onnx",
                name: "prompt_encoder_mask_decoder.onnx",
                size: 213_114,
                sha256: "a21b3b09f6e5f5838d630d92a150914ca58f8f58b603b8f53fc32b03da982203",
            },
            CatalogFile {
                url: "https://huggingface.co/onnx-community/sam2.1-hiera-large-ONNX/resolve/main/onnx/prompt_encoder_mask_decoder.onnx_data",
                name: "prompt_encoder_mask_decoder.onnx_data",
                size: 20_958_208,
                sha256: "bfdcd4f7ee04298a82fe1aaf1bd7b8edf544d36901e1a886f01892fb915bd63c",
            },
        ],
    },
    CatalogModel {
        id: "sam-2-1-base-plus",
        name: "SAM 2.1 Base+",
        kinds: &[ModelKind::Segmentation],
        description: "Meta's Segment Anything 2.1: accurate selections at a moderate size",
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
        id: "birefnet-hr-matting",
        name: "BiRefNet HR Matting (2K)",
        kinds: &[ModelKind::Matting],
        description: "Best-quality soft cut-outs (hair, fur, glass) at 2048 px; needs a lot of memory",
        license: "MIT",
        homepage: "https://huggingface.co/PinkPixel/birefnet-hr-matting-onnx",
        files: &[
            CatalogFile {
                url: "https://huggingface.co/PinkPixel/birefnet-hr-matting-onnx/resolve/main/model.onnx",
                name: "model.onnx",
                size: 932_150_975,
                sha256: "a548a1f3307250766bef50b174026b3cfa39a695bc1e42ec131c14fe82797ffd",
            },
        ],
    },
    CatalogModel {
        id: "birefnet-matting",
        name: "BiRefNet Matting",
        kinds: &[ModelKind::Matting],
        description: "Soft-edge cut-outs with fine hair detail at 1024 px",
        license: "MIT",
        homepage: "https://huggingface.co/emrikol/birefnet-matting-onnx",
        files: &[
            CatalogFile {
                url: "https://huggingface.co/emrikol/birefnet-matting-onnx/resolve/main/birefnet-matting.onnx",
                name: "birefnet-matting.onnx",
                size: 940_840_787,
                sha256: "f0843e38f6a4e88efc8c5fad4178ad7ed6c818346ce12f82e7b579324fe7e0c5",
            },
        ],
    },
    CatalogModel {
        id: "birefnet-dynamic",
        name: "BiRefNet Dynamic",
        kinds: &[ModelKind::Matting, ModelKind::Segmentation],
        description: "High-accuracy subject selection trained to work at many resolutions",
        license: "MIT",
        homepage: "https://huggingface.co/onnx-community/BiRefNet_dynamic-1024x1024-ONNX",
        files: &[
            CatalogFile {
                url: "https://huggingface.co/onnx-community/BiRefNet_dynamic-1024x1024-ONNX/resolve/main/onnx/model.onnx",
                name: "model.onnx",
                size: 972_495_971,
                sha256: "aa3c17882a27f079fafa7655cf50e83b9e6e067d6c6fa4cbc3524ae441da0bd7",
            },
        ],
    },
    CatalogModel {
        id: "birefnet-hrsod",
        name: "BiRefNet HRSOD",
        kinds: &[ModelKind::Matting, ModelKind::Segmentation],
        description: "Subject selection tuned on high-resolution photos",
        license: "MIT",
        homepage: "https://huggingface.co/onnx-community/BiRefNet-HRSOD_DHU-ONNX",
        files: &[
            CatalogFile {
                url: "https://huggingface.co/onnx-community/BiRefNet-HRSOD_DHU-ONNX/resolve/main/onnx/model.onnx",
                name: "model.onnx",
                size: 972_666_916,
                sha256: "4f5837663194fb88f603b76782eae05a3c29f5749872ca1bfb636bd26e7f6bfc",
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
        id: "realistic-vision-6-inpainting",
        name: "Realistic Vision 6.0 Inpainting",
        kinds: &[ModelKind::Inpainting],
        description: "The most photorealistic prompted fills, full precision (Stable Diffusion 1.5 checkpoint, prepared on first use)",
        license: "CreativeML OpenRAIL-M",
        homepage: "https://huggingface.co/SG161222/Realistic_Vision_V6.0_B1_noVAE",
        files: &[
            CatalogFile {
                url: "https://huggingface.co/SG161222/Realistic_Vision_V6.0_B1_noVAE/resolve/main/Realistic_Vision_V6.0_NV_B1_inpainting.safetensors",
                name: "Realistic_Vision_V6.0_NV_B1_inpainting.safetensors",
                size: 4_265_203_868,
                sha256: "e9c575dbbe237ea889e14c41f953fc9fb311439a88b15bb2e789b0630ae45efe",
            },
        ],
    },
    CatalogModel {
        id: "realistic-vision-6-inpainting-fp16",
        name: "Realistic Vision 6.0 Inpainting (half size)",
        kinds: &[ModelKind::Inpainting],
        description: "Photorealistic prompted fills at half the download (Stable Diffusion 1.5 checkpoint, prepared on first use)",
        license: "CreativeML OpenRAIL-M",
        homepage: "https://huggingface.co/SG161222/Realistic_Vision_V6.0_B1_noVAE",
        files: &[
            CatalogFile {
                url: "https://huggingface.co/SG161222/Realistic_Vision_V6.0_B1_noVAE/resolve/main/Realistic_Vision_V6.0_NV_B1_inpainting_fp16.safetensors",
                name: "Realistic_Vision_V6.0_NV_B1_inpainting_fp16.safetensors",
                size: 2_136_906_350,
                sha256: "d939cbdbce19a9d8837794acd06ef8db1ad4a2f2bcf1418432d897eac98f1afd",
            },
        ],
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
        id: "sd-2-inpainting-full",
        name: "Stable Diffusion 2 Inpainting (full precision)",
        kinds: &[ModelKind::Inpainting],
        description: "The official SD 2 inpainting checkpoint at full precision, prepared on first use",
        license: "CreativeML OpenRAIL++-M",
        homepage: "https://huggingface.co/sd2-community/stable-diffusion-2-inpainting",
        files: &[
            CatalogFile {
                url: "https://huggingface.co/sd2-community/stable-diffusion-2-inpainting/resolve/main/512-inpainting-ema.safetensors",
                name: "512-inpainting-ema.safetensors",
                size: 5_214_662_094,
                sha256: "b29e2ed9a8fe58e76f7e801bda091d23738bd74c1da3f339bcbe2d40922fcb60",
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
    CatalogModel {
        id: "swin2sr-realworld-4x",
        name: "Swin2SR Real-World 4x",
        kinds: &[ModelKind::Enhance],
        description: "Transformer upscaler for real photos: clean, faithful detail; the slowest",
        license: "Apache-2.0",
        homepage: "https://huggingface.co/Xenova/swin2SR-realworld-sr-x4-64-bsrgan-psnr",
        files: &[
            CatalogFile {
                url: "https://huggingface.co/Xenova/swin2SR-realworld-sr-x4-64-bsrgan-psnr/resolve/main/onnx/model.onnx",
                name: "model.onnx",
                size: 52_772_645,
                sha256: "f496dc73dcc01d778b1a12eb4c4038d6b27cd1c0b5bcd4258455ed6d7816c835",
            },
        ],
    },
    CatalogModel {
        id: "nomos2-realplksr-4x",
        name: "4x Nomos2 RealPLKSR",
        kinds: &[ModelKind::Enhance],
        description: "Sharp 4x upscaler for photos that copes with JPEG artifacts and keeps background blur",
        license: "CC-BY-4.0",
        homepage: "https://huggingface.co/Phips/4xNomos2_realplksr_dysample",
        files: &[
            CatalogFile {
                url: "https://huggingface.co/nesaorg/4xNomos2_realplksr_dysample_256_fp32_fullyoptimized/resolve/main/4xNomos2_realplksr_dysample_256_fp32_fullyoptimized.onnx",
                name: "4xNomos2_realplksr_dysample_256_fp32_fullyoptimized.onnx",
                size: 30_298_952,
                sha256: "f4da353c0a380aa2c53485f2479bfd7959cab071734075d96f53df3233976716",
            },
        ],
    },
    CatalogModel {
        id: "realplksr-4x",
        name: "RealPLKSR 4x",
        kinds: &[ModelKind::Enhance],
        description: "Fast, natural-looking 4x upscaler for photographs (from darktable)",
        license: "MIT",
        homepage: "https://huggingface.co/darktable-org/upscale-realplksr-onnx",
        files: &[
            CatalogFile {
                url: "https://huggingface.co/darktable-org/upscale-realplksr-onnx/resolve/main/onnx/model_x4.onnx",
                name: "model_x4.onnx",
                size: 29_711_010,
                sha256: "8ce692652494e2e5c31e1f7c7ab61eb084e9a450cde727a374f5d6746cfe054b",
            },
        ],
    },
    CatalogModel {
        id: "realplksr-2x",
        name: "RealPLKSR 2x",
        kinds: &[ModelKind::Enhance],
        description: "Fast, natural-looking 2x upscaler for lightly soft photos",
        license: "MIT",
        homepage: "https://huggingface.co/darktable-org/upscale-realplksr-onnx",
        files: &[
            CatalogFile {
                url: "https://huggingface.co/darktable-org/upscale-realplksr-onnx/resolve/main/onnx/model_x2.onnx",
                name: "model_x2.onnx",
                size: 29_627_920,
                sha256: "d7abb65092f3808d3aa255ffdab42b1d672883915d90adbdd321204168b9f293",
            },
        ],
    },
    CatalogModel {
        id: "swin2sr-classical-2x",
        name: "Swin2SR Classical 2x",
        kinds: &[ModelKind::Enhance],
        description: "Transformer 2x upscaler for clean, lightly compressed images",
        license: "Apache-2.0",
        homepage: "https://huggingface.co/Xenova/swin2SR-classical-sr-x2-64",
        files: &[
            CatalogFile {
                url: "https://huggingface.co/Xenova/swin2SR-classical-sr-x2-64/resolve/main/onnx/model.onnx",
                name: "model.onnx",
                size: 54_428_699,
                sha256: "6dde3fe2440543ccae7c40d175609f83c18aeaa3d8456745c4329ef97ae744bd",
            },
        ],
    },
    CatalogModel {
        id: "real-esrgan-x4plus",
        name: "Real-ESRGAN x4plus",
        kinds: &[ModelKind::Enhance],
        description: "The classic Real-ESRGAN 4x upscaler for general photos; sharp, sometimes smooth",
        license: "BSD-3-Clause",
        homepage: "https://huggingface.co/bukuroo/RealESRGAN-ONNX",
        files: &[
            CatalogFile {
                url: "https://huggingface.co/bukuroo/RealESRGAN-ONNX/resolve/main/real-esrgan-x4plus-128.onnx",
                name: "real-esrgan-x4plus-128.onnx",
                size: 67_160_311,
                sha256: "6a6f4a3d58553d40fdd443d9e5f4b2deb9b52bef1ec2947700fc2167ac876c7d",
            },
        ],
    },
    CatalogModel {
        id: "real-esrgan-general-v3",
        name: "Real-ESRGAN General v3",
        kinds: &[ModelKind::Enhance],
        description: "Tiny, very fast 4x upscaler for everyday photos",
        license: "BSD-3-Clause",
        homepage: "https://huggingface.co/Heliosoph/realesrgan-onnx",
        files: &[
            CatalogFile {
                url: "https://huggingface.co/Heliosoph/realesrgan-onnx/resolve/main/realesr-general-x4v3.onnx",
                name: "realesr-general-x4v3.onnx",
                size: 4_871_181,
                sha256: "09b757accd747d7e423c1d352b3e8f23e77cc5742d04bae958d4eb8082b76fa4",
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
