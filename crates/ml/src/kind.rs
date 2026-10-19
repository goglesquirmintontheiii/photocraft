//! What a model does, which decides the AI tab's section and which tools can use it.

use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum ModelKind {
    /// Point and box segmentation (SAM family): AI Object Selection.
    Segmentation,
    /// Subject and background matting: selecting people, animals and products.
    Matting,
    /// Prompt-free inpainting: Generative Removal.
    Removal,
    /// Reflection removal brush.
    Reflection,
    /// Prompted diffusion inpainting: the Infill brush and Edit › Generative Fill.
    Inpainting,
    /// Detail and sharpness: super-resolution networks, or a diffusion pipeline used lightly
    /// (the Generative Enhance brush and Generative Sharpen).
    Enhance,
    /// Anything imported that doesn't fit the above (kept, listed, never offered to a tool).
    Other,
}

impl ModelKind {
    /// The AI tab's section order.
    pub const ALL: [ModelKind; 7] = [
        ModelKind::Segmentation,
        ModelKind::Matting,
        ModelKind::Removal,
        ModelKind::Reflection,
        ModelKind::Inpainting,
        ModelKind::Enhance,
        ModelKind::Other,
    ];

    /// Section heading in the AI tab.
    pub fn title(self) -> &'static str {
        match self {
            ModelKind::Segmentation => "Object Selection",
            ModelKind::Matting => "Subject & Background",
            ModelKind::Removal => "Generative Removal",
            ModelKind::Reflection => "Reflection Removal",
            ModelKind::Inpainting => "Generative Fill & Infill",
            ModelKind::Enhance => "Generative Enhance",
            ModelKind::Other => "Other Models",
        }
    }

    /// One line under the heading: which tools use these models.
    pub fn used_by(self) -> &'static str {
        match self {
            ModelKind::Segmentation => "Used by the AI Object Selection tool.",
            ModelKind::Matting => "Used to select people, animals and products in one click.",
            ModelKind::Removal => "Used by the Generative Removal brush.",
            ModelKind::Reflection => "Used by the Reflection Removal brush.",
            ModelKind::Inpainting => "Used by the Infill brush and Generative Fill (prompts).",
            ModelKind::Enhance => "Used by the Generative Enhance brush and Generative Sharpen.",
            ModelKind::Other => "Imported models PhotoCraft's tools don't use yet.",
        }
    }

    /// Stable id for preferences, commands and the manifest.
    pub fn id(self) -> &'static str {
        match self {
            ModelKind::Segmentation => "segmentation",
            ModelKind::Matting => "matting",
            ModelKind::Removal => "removal",
            ModelKind::Reflection => "reflection",
            ModelKind::Inpainting => "inpainting",
            ModelKind::Enhance => "enhance",
            ModelKind::Other => "other",
        }
    }

    pub fn from_id(id: &str) -> Option<ModelKind> {
        ModelKind::ALL.into_iter().find(|k| k.id() == id)
    }
}

/// Guesses what a model is for from its link or file name ("sam2.1-hiera-tiny" → Object
/// Selection, "lama_fp32.onnx" → Generative Removal and Reflection Removal …), for the Import
/// panel's "Use as". Recommended models are recognised by their repository first. Empty when
/// nothing gives it away; the person then picks.
pub fn detect(text: &str) -> Vec<ModelKind> {
    let lower = text.trim().to_ascii_lowercase();
    if lower.is_empty() {
        return Vec::new();
    }
    if let Some(m) = crate::catalog::by_link(&lower) {
        return m.kinds.to_vec();
    }
    // Words split on anything that isn't a letter or digit, so "sam" doesn't match "sample".
    let words: Vec<&str> = lower.split(|c: char| !c.is_ascii_alphanumeric()).filter(|w| !w.is_empty()).collect();
    let word = |w: &str| words.contains(&w);
    let starts = |p: &str| words.iter().any(|w| w.starts_with(p));
    let has = |p: &str| lower.contains(p);
    let mut out = Vec::new();
    if has("esrgan") || has("swinir") || has("upscal") || has("super-res") || has("superres") || word("4x") || word("x4") || word("2x") || word("x2") || has("4x_") || has("_x4") || has("_x2") || has("nomos") || has("ultrasharp") || has("realsr") {
        out.push(ModelKind::Enhance);
    }
    if has("reflection") || starts("dereflect") || word("dsrnet") || word("errnet") || word("rdnet") {
        out.push(ModelKind::Reflection);
    }
    if starts("sam") && !has("sample") || has("segment") || has("efficientvit") || word("hiera") {
        out.push(ModelKind::Segmentation);
    }
    if has("birefnet") || has("rmbg") || has("matting") || has("modnet") || has("u2net") || has("isnet") || has("background") || has("bgremov") {
        out.extend([ModelKind::Matting, ModelKind::Segmentation]);
    }
    let diffusion = has("stable-diffusion")
        || has("stable_diffusion")
        || has("sdxl")
        || has("diffusion")
        || has("flux")
        || has("kandinsky")
        || starts("sd") && (has("inpaint") || has("v1") || has("xl") || word("sd2") || word("sd15"));
    let removal = has("lama") || has("migan") || has("mi-gan") || has("mi_gan") || word("mat") || word("zits");
    if diffusion {
        out.push(ModelKind::Inpainting);
    } else if removal {
        out.extend([ModelKind::Removal, ModelKind::Reflection]);
    } else if has("inpaint") {
        // Any other inpainting model is a prompted (diffusion) one these days.
        out.push(ModelKind::Inpainting);
    }
    let mut seen = Vec::new();
    out.retain(|k| {
        let fresh = !seen.contains(k);
        seen.push(*k);
        fresh
    });
    out
}

/// Kinds in the AI tab's section order, without repeats.
pub fn normalize(kinds: &[ModelKind]) -> Vec<ModelKind> {
    ModelKind::ALL.into_iter().filter(|k| kinds.contains(k)).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn detection_reads_links_and_file_names() {
        use ModelKind::*;
        assert_eq!(detect("https://huggingface.co/Acly/MobileSAM"), vec![Segmentation]);
        assert_eq!(detect("onnx-community/sam2.1-hiera-large-ONNX"), vec![Segmentation]);
        assert_eq!(detect("https://huggingface.co/Carve/LaMa-ONNX/blob/main/lama_fp32.onnx"), vec![Removal, Reflection]);
        assert_eq!(detect("big-lama.safetensors"), vec![Removal, Reflection]);
        assert_eq!(detect("ZhengPeng7/BiRefNet_HR"), vec![Matting, Segmentation]);
        assert_eq!(detect("https://civitai.com/models/1234/epicrealism-inpainting sdxl"), vec![Inpainting]);
        assert_eq!(detect("runwayml/stable-diffusion-inpainting"), vec![Inpainting]);
        assert_eq!(detect("someone/dsrnet-reflection"), vec![Reflection]);
        assert_eq!(detect("RanaLLC/Realistic_Vision_V5.1-inpainting-onnx-fp16"), vec![Inpainting]);
        assert_eq!(detect("https://github.com/PINTO0309/reflection-removal"), vec![Reflection]);
        assert_eq!(detect("RealESRGAN_x4plus.onnx"), vec![Enhance]);
        assert_eq!(detect("4x-UltraSharp.onnx"), vec![Enhance]);
        assert!(detect("sample-model.onnx").is_empty());
        assert!(detect("").is_empty());
        assert!(detect("https://example.com/weights.bin").is_empty());
    }

    #[test]
    fn normalize_orders_and_dedups() {
        use ModelKind::*;
        assert_eq!(normalize(&[Reflection, Removal, Reflection]), vec![Removal, Reflection]);
        assert!(normalize(&[]).is_empty());
    }

    #[test]
    fn ids_round_trip_and_unknown_ids_are_none() {
        for k in ModelKind::ALL {
            assert_eq!(ModelKind::from_id(k.id()), Some(k));
            assert_eq!(serde_json::to_value(k).ok(), Some(serde_json::json!(k.id())));
        }
        assert_eq!(ModelKind::from_id(""), None);
        assert_eq!(ModelKind::from_id("Segmentation"), None);
    }
}
