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
        id: "sdxl-inpainting",
        name: "SDXL Inpainting 0.1",
        kinds: &[ModelKind::Inpainting],
        description: "Stable Diffusion XL inpainting, best prompted fills",
        license: "OpenRAIL++-M",
        homepage: "https://huggingface.co/aimi-models/inpainting",
        files: &[CatalogFile {
            url: "https://huggingface.co/aimi-models/inpainting/resolve/main/sd_xl_base_1.0_inpainting_0.1.safetensors",
            name: "sd_xl_base_1.0_inpainting_0.1.safetensors",
            size: 6_938_069_944,
            sha256: "",
        }],
    },
    CatalogModel {
        id: "sd-2-inpainting",
        name: "Stable Diffusion 2 Inpainting",
        kinds: &[ModelKind::Inpainting],
        description: "Stable Diffusion 2 inpainting at 512 px",
        license: "OpenRAIL++-M",
        homepage: "https://huggingface.co/sd2-community/stable-diffusion-2-inpainting",
        files: &[CatalogFile {
            url: "https://huggingface.co/sd2-community/stable-diffusion-2-inpainting/resolve/main/512-inpainting-ema.safetensors",
            name: "512-inpainting-ema.safetensors",
            size: 5_214_662_094,
            sha256: "",
        }],
    },
    CatalogModel {
        id: "sd-1-5-inpainting",
        name: "Stable Diffusion 1.5 Inpainting",
        kinds: &[ModelKind::Inpainting],
        description: "The classic, lightest prompted inpainting model",
        license: "CreativeML OpenRAIL-M",
        homepage: "https://huggingface.co/aimi-models/inpainting",
        files: &[CatalogFile {
            url: "https://huggingface.co/aimi-models/inpainting/resolve/main/sd-v1-5-inpainting.safetensors",
            name: "sd-v1-5-inpainting.safetensors",
            size: 4_265_203_873,
            sha256: "",
        }],
    },
];

#[cfg(test)]
mod tests {
    use super::*;
    use crate::store::{valid_file_name, valid_id};

    #[test]
    fn models_are_recognised_by_link_and_by_files() {
        assert_eq!(by_link("https://huggingface.co/acly/mobilesam").map(|m| m.id), Some("mobilesam"));
        assert_eq!(by_link("acly/mobilesam").map(|m| m.id), Some("mobilesam"));
        assert_eq!(by_link("https://huggingface.co/carve/lama-onnx/blob/main/lama_fp32.onnx").map(|m| m.id), Some("lama"));
        // Shared repository: not enough to tell.
        assert!(by_link("https://huggingface.co/aimi-models/inpainting").is_none());
        assert_eq!(
            by_link("https://huggingface.co/aimi-models/inpainting/resolve/main/sd-v1-5-inpainting.safetensors").map(|m| m.id),
            Some("sd-1-5-inpainting")
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
                assert!(valid_file_name(f.name) && names.insert(f.name), "{}", f.name);
                assert!(f.size > 0, "{}", f.name);
                assert!(f.sha256.is_empty() || crate::source::is_sha256(f.sha256) && f.sha256 == f.sha256.to_ascii_lowercase(), "{}", f.name);
            }
        }
    }
}
