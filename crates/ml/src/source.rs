//! What an import link points at, and how to turn it into files to download.
//!
//! Accepted links:
//! - a Hugging Face file: `https://huggingface.co/<org>/<repo>/blob|resolve/<rev>/<path>`;
//! - a Hugging Face repository: `https://huggingface.co/<org>/<repo>` (optionally `/tree/<rev>`),
//!   or just `<org>/<repo>`; its model file is found through the Hub API;
//! - a Civitai model or version: `https://civitai.com/models/<id>[?modelVersionId=<v>]`, or a
//!   `https://civitai.com/api/download/models/<v>` download link;
//! - any other `http(s)://` link to a model file.

use crate::store::safe_file_name;
use crate::{Error, Result};
use serde_json::Value;

/// File extensions treated as model weights when picking files from a repository.
pub const WEIGHT_EXTENSIONS: [&str; 9] = ["onnx", "safetensors", "ckpt", "pt", "pth", "bin", "gguf", "onnx_data", "data"];

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Link {
    HfFile { repo: String, rev: String, path: String },
    HfRepo { repo: String, rev: String, subdir: String },
    Civitai { model: Option<u64>, version: Option<u64> },
    Url { url: String },
}

/// A file to fetch for an import.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RemoteFile {
    pub url: String,
    /// Name inside the model's folder.
    pub name: String,
    pub size: Option<u64>,
    /// Lowercase hex, when the source publishes it.
    pub sha256: Option<String>,
}

/// An import link resolved to its files.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Resolved {
    pub name: String,
    pub files: Vec<RemoteFile>,
    pub license: String,
}

impl Link {
    pub fn parse(input: &str) -> Result<Link> {
        let text = input.trim();
        if text.is_empty() {
            return Err(Error::Invalid("Paste a link to a model.".into()));
        }
        let Some((scheme, rest)) = text.split_once("://") else {
            // `org/repo` shorthand for a Hugging Face repository.
            let parts: Vec<&str> = text.split('/').collect();
            if parts.len() == 2 && parts.iter().all(|p| repo_part(p)) {
                return Ok(Link::HfRepo { repo: text.to_string(), rev: "main".into(), subdir: String::new() });
            }
            return Err(Error::Invalid("That isn't a link. Paste an address starting with https://, or a Hugging Face repository like org/name.".into()));
        };
        if !scheme.eq_ignore_ascii_case("https") && !scheme.eq_ignore_ascii_case("http") {
            return Err(Error::Invalid("Only http:// and https:// links can be imported.".into()));
        }
        let (host_path, query) = rest.split_once('?').map_or((rest, ""), |(a, b)| (a, b));
        let host_path = host_path.split('#').next().unwrap_or(host_path);
        let (host, path) = host_path.split_once('/').map_or((host_path, ""), |(h, p)| (h, p));
        let host = host.to_ascii_lowercase();
        if host.is_empty() {
            return Err(Error::Invalid("That link has no website address.".into()));
        }
        let segs: Vec<&str> = path.split('/').filter(|s| !s.is_empty()).collect();
        if host == "huggingface.co" || host == "www.huggingface.co" || host == "hf.co" {
            return hf_link(&segs, text);
        }
        if host == "civitai.com" || host == "www.civitai.com" {
            if let ["models", id, ..] = segs.as_slice() {
                return Ok(Link::Civitai { model: id.parse().ok(), version: query_u64(query, "modelVersionId") });
            }
            if let ["api", "download", "models", v, ..] = segs.as_slice() {
                return Ok(Link::Civitai { model: None, version: v.parse().ok() });
            }
        }
        Ok(Link::Url { url: text.to_string() })
    }

    /// The import's source, for the manifest.
    pub fn source(&self, original: &str) -> crate::ModelSource {
        let url = original.trim().to_string();
        match self {
            Link::HfFile { .. } | Link::HfRepo { .. } => crate::ModelSource::HuggingFace { url },
            Link::Civitai { .. } => crate::ModelSource::Civitai { url },
            Link::Url { .. } => crate::ModelSource::Url { url },
        }
    }

    /// Resolves the link to the files to download, using `get_json` for the Hugging Face and
    /// Civitai APIs.
    pub fn resolve(&self, get_json: &dyn Fn(&str) -> Result<Value>) -> Result<Resolved> {
        match self {
            Link::HfFile { repo, rev, path } => {
                let name = path.rsplit('/').next().unwrap_or(path);
                Ok(Resolved {
                    name: format!("{} ({name})", repo.rsplit('/').next().unwrap_or(repo)),
                    files: vec![RemoteFile { url: hf_url(repo, rev, path), name: safe_file_name(name), size: None, sha256: None }],
                    license: String::new(),
                })
            }
            Link::HfRepo { repo, rev, subdir } => resolve_hf_repo(repo, rev, subdir, get_json),
            Link::Civitai { model, version } => resolve_civitai(*model, *version, get_json),
            Link::Url { url } => {
                let last = url.split(['?', '#']).next().unwrap_or(url).rsplit('/').find(|s| !s.is_empty()).unwrap_or("model.bin");
                let file = safe_file_name(&percent_decode(last));
                Ok(Resolved { name: file.clone(), files: vec![RemoteFile { url: url.clone(), name: file, size: None, sha256: None }], license: String::new() })
            }
        }
    }
}

fn repo_part(p: &str) -> bool {
    !p.is_empty() && p.len() <= 96 && p.bytes().all(|b| b.is_ascii_alphanumeric() || matches!(b, b'-' | b'_' | b'.')) && p != "." && p != ".."
}

fn hf_link(segs: &[&str], original: &str) -> Result<Link> {
    let bad = || Error::Invalid(format!("That Hugging Face link isn't a model repository or file: {original}"));
    // Datasets and Spaces aren't models.
    if matches!(segs.first(), Some(&"datasets") | Some(&"spaces")) {
        return Err(Error::Invalid("That link is a Hugging Face dataset or Space, not a model.".into()));
    }
    let [org, name, rest @ ..] = segs else { return Err(bad()) };
    if !repo_part(org) || !repo_part(name) {
        return Err(bad());
    }
    let repo = format!("{org}/{name}");
    match rest {
        [] => Ok(Link::HfRepo { repo, rev: "main".into(), subdir: String::new() }),
        ["tree", rev, sub @ ..] => Ok(Link::HfRepo { repo, rev: (*rev).to_string(), subdir: sub.join("/") }),
        ["blob" | "resolve", rev, path @ ..] if !path.is_empty() => {
            if path.iter().any(|p| *p == ".." || *p == ".") {
                return Err(bad());
            }
            Ok(Link::HfFile { repo, rev: (*rev).to_string(), path: path.iter().map(|p| percent_decode(p)).collect::<Vec<_>>().join("/") })
        }
        _ => Err(bad()),
    }
}

fn hf_url(repo: &str, rev: &str, path: &str) -> String {
    format!("https://huggingface.co/{repo}/resolve/{rev}/{}", path.split('/').map(percent_encode).collect::<Vec<_>>().join("/"))
}

fn ext(path: &str) -> String {
    path.rsplit('/').next().unwrap_or(path).rsplit_once('.').map(|(_, e)| e.to_ascii_lowercase()).unwrap_or_default()
}

fn resolve_hf_repo(repo: &str, rev: &str, subdir: &str, get_json: &dyn Fn(&str) -> Result<Value>) -> Result<Resolved> {
    let tree = get_json(&format!("https://huggingface.co/api/models/{repo}/tree/{rev}?recursive=true"))?;
    let entries = tree.as_array().ok_or_else(|| Error::Network("Hugging Face sent an unexpected file list.".into()))?;
    let prefix = if subdir.is_empty() { String::new() } else { format!("{}/", subdir.trim_matches('/')) };
    if let Some(r) = diffusion_pipeline(repo, rev, &prefix, entries) {
        let license = get_json(&format!("https://huggingface.co/api/models/{repo}"))
            .ok()
            .and_then(|v| v.pointer("/cardData/license").and_then(Value::as_str).map(str::to_string))
            .unwrap_or_default();
        return Ok(Resolved { license, ..r });
    }
    let mut weights: Vec<(String, Option<u64>, Option<String>)> = entries
        .iter()
        .filter(|e| e.get("type").and_then(Value::as_str) == Some("file"))
        .filter_map(|e| {
            let path = e.get("path")?.as_str()?.to_string();
            if !path.starts_with(&prefix) || !WEIGHT_EXTENSIONS.contains(&ext(&path).as_str()) {
                return None;
            }
            let lfs = e.get("lfs");
            let size = lfs.and_then(|l| l.get("size")).or_else(|| e.get("size")).and_then(Value::as_u64);
            let sha = lfs.and_then(|l| l.get("oid")).and_then(Value::as_str).filter(|s| is_sha256(s)).map(str::to_ascii_lowercase);
            Some((path, size, sha))
        })
        .collect();
    // `.bin` and `.data` are only weights next to a model file of their own family; on their
    // own (tokenizers, optimizer state) they are skipped unless nothing else is there.
    let main: Vec<_> = weights.iter().filter(|(p, ..)| !matches!(ext(p).as_str(), "bin" | "data" | "onnx_data")).cloned().collect();
    let onnx: Vec<_> = weights.iter().filter(|(p, ..)| matches!(ext(p).as_str(), "onnx" | "onnx_data" | "data")).cloned().collect();
    let picked = if !onnx.is_empty() && onnx.iter().filter(|(p, ..)| ext(p) == "onnx").count() <= 3 {
        onnx
    } else if main.len() == 1 {
        main
    } else if main.is_empty() && weights.len() == 1 {
        std::mem::take(&mut weights)
    } else if main.is_empty() && weights.is_empty() {
        return Err(Error::NotFound("No model files were found in that repository.".into()));
    } else {
        return Err(Error::Invalid(format!(
            "That repository has {} model files. Open the one you want on Hugging Face and paste its link instead.",
            main.len().max(weights.len())
        )));
    };
    let license = get_json(&format!("https://huggingface.co/api/models/{repo}"))
        .ok()
        .and_then(|v| v.pointer("/cardData/license").and_then(Value::as_str).map(str::to_string))
        .unwrap_or_default();
    let mut names = std::collections::HashSet::new();
    let mut files = Vec::new();
    for (path, size, sha256) in picked {
        let base = safe_file_name(path.rsplit('/').next().unwrap_or(&path));
        if !names.insert(base.clone()) {
            return Err(Error::Invalid(format!("That repository has two files named {base}; paste the link of the one you want.")));
        }
        files.push(RemoteFile { url: hf_url(repo, rev, &path), name: base, size, sha256 });
    }
    Ok(Resolved { name: repo.rsplit('/').next().unwrap_or(repo).to_string(), files, license })
}

/// The parts of a diffusers ONNX pipeline repository (`model_index.json` beside `unet/`,
/// `text_encoder/`, `vae_encoder/`, `vae_decoder/`, `tokenizer/`, `scheduler/`) that inpainting
/// runs, kept in their folders: the safety checker and feature extractor are left out.
fn diffusion_pipeline(repo: &str, rev: &str, prefix: &str, entries: &[Value]) -> Option<Resolved> {
    const PARTS: [&str; 6] = ["text_encoder", "unet", "vae_encoder", "vae_decoder", "tokenizer", "scheduler"];
    let files: Vec<(String, Option<u64>, Option<String>)> = entries
        .iter()
        .filter(|e| e.get("type").and_then(Value::as_str) == Some("file"))
        .filter_map(|e| {
            let path = e.get("path")?.as_str()?;
            let rel = path.strip_prefix(prefix)?.to_string();
            let lfs = e.get("lfs");
            let size = lfs.and_then(|l| l.get("size")).or_else(|| e.get("size")).and_then(Value::as_u64);
            let sha = lfs.and_then(|l| l.get("oid")).and_then(Value::as_str).filter(|s| is_sha256(s)).map(str::to_ascii_lowercase);
            Some((rel, size, sha))
        })
        .collect();
    let has = |p: &str| files.iter().any(|(r, ..)| r == p);
    let onnx_in = |dir: &str| files.iter().any(|(r, ..)| r.starts_with(&format!("{dir}/")) && ext(r) == "onnx");
    if !has("model_index.json") || !onnx_in("unet") || !onnx_in("text_encoder") || !onnx_in("vae_decoder") {
        return None;
    }
    let mut picked = Vec::new();
    for (rel, size, sha) in files {
        let mut parts = rel.split('/');
        let keep = match (parts.next(), parts.next(), parts.next()) {
            (Some("model_index.json"), None, _) => true,
            (Some(dir), Some(file), None) if PARTS.contains(&dir) => {
                matches!(ext(file).as_str(), "onnx" | "onnx_data" | "pb" | "data" | "json" | "txt") || file == "weights"
            }
            _ => false,
        };
        if keep {
            let name = rel.split('/').map(safe_file_name).collect::<Vec<_>>().join("/");
            picked.push(RemoteFile { url: hf_url(repo, rev, &format!("{prefix}{rel}")), name, size, sha256: sha });
        }
    }
    Some(Resolved { name: repo.rsplit('/').next().unwrap_or(repo).to_string(), files: picked, license: String::new() })
}

/// The files of a diffusion pipeline folder on this computer, with their names inside the model
/// (`unet/model.onnx` …): what [`diffusion_pipeline`] keeps of a repository.
pub(crate) fn pipeline_files_in(dir: &std::path::Path) -> Result<Vec<(std::path::PathBuf, String)>> {
    let mut entries = Vec::new();
    let mut paths = std::collections::HashMap::new();
    let mut visit = |rel: String, p: std::path::PathBuf| {
        let size = std::fs::metadata(&p).map(|m| m.len()).unwrap_or(0);
        entries.push(serde_json::json!({"type": "file", "path": rel.clone(), "size": size}));
        paths.insert(rel, p);
    };
    let read = |d: &std::path::Path| std::fs::read_dir(d).map_err(|e| Error::io(&format!("Couldn't read {}", d.display()), e));
    for e in read(dir)?.flatten() {
        let name = e.file_name().to_string_lossy().into_owned();
        match e.file_type() {
            Ok(t) if t.is_file() => visit(name, e.path()),
            Ok(t) if t.is_dir() => {
                for f in read(&e.path())?.flatten() {
                    if f.file_type().is_ok_and(|t| t.is_file()) {
                        visit(format!("{name}/{}", f.file_name().to_string_lossy()), f.path());
                    }
                }
            }
            _ => {}
        }
    }
    let r = diffusion_pipeline("local/folder", "main", "", &entries).ok_or_else(|| {
        Error::Invalid(
            "That folder isn't an ONNX diffusion pipeline: it needs model_index.json and unet, text_encoder and vae_decoder folders with model.onnx in each.".into(),
        )
    })?;
    Ok(r
        .files
        .into_iter()
        .filter_map(|f| {
            let rel = f.url.rsplit_once("/resolve/main/").map(|(_, r)| percent_decode(r))?;
            Some((paths.get(&rel)?.clone(), f.name))
        })
        .collect())
}

fn resolve_civitai(model: Option<u64>, version: Option<u64>, get_json: &dyn Fn(&str) -> Result<Value>) -> Result<Resolved> {
    let (name, ver) = match (model, version) {
        (Some(m), v) => {
            let info = get_json(&format!("https://civitai.com/api/v1/models/{m}"))?;
            let versions = info.get("modelVersions").and_then(Value::as_array).cloned().unwrap_or_default();
            let ver = match v {
                Some(v) => versions.into_iter().find(|x| x.get("id").and_then(Value::as_u64) == Some(v)),
                None => versions.into_iter().next(),
            }
            .ok_or_else(|| Error::NotFound("That Civitai model has no downloadable version.".into()))?;
            let model_name = info.get("name").and_then(Value::as_str).unwrap_or("Civitai model").to_string();
            let ver_name = ver.get("name").and_then(Value::as_str).unwrap_or("");
            (if ver_name.is_empty() { model_name } else { format!("{model_name} ({ver_name})") }, ver)
        }
        (None, Some(v)) => {
            let ver = get_json(&format!("https://civitai.com/api/v1/model-versions/{v}"))?;
            let model_name = ver.pointer("/model/name").and_then(Value::as_str).unwrap_or("Civitai model");
            let ver_name = ver.get("name").and_then(Value::as_str).unwrap_or("");
            (if ver_name.is_empty() { model_name.to_string() } else { format!("{model_name} ({ver_name})") }, ver)
        }
        (None, None) => return Err(Error::Invalid("That Civitai link doesn't name a model.".into())),
    };
    let files = ver.get("files").and_then(Value::as_array).cloned().unwrap_or_default();
    let file = files
        .iter()
        .find(|f| f.get("primary").and_then(Value::as_bool) == Some(true))
        .or_else(|| files.iter().find(|f| f.get("type").and_then(Value::as_str) == Some("Model")))
        .or_else(|| files.first())
        .ok_or_else(|| Error::NotFound("That Civitai version has no files.".into()))?;
    let url = file
        .get("downloadUrl")
        .and_then(Value::as_str)
        .filter(|u| u.starts_with("https://"))
        .ok_or_else(|| Error::NotFound("Civitai didn't give a download link.".into()))?;
    let fname = safe_file_name(file.get("name").and_then(Value::as_str).unwrap_or("model.safetensors"));
    // Civitai's `sizeKB` is rounded, so it isn't used to check the download; the SHA-256 is.
    let sha256 = file.pointer("/hashes/SHA256").and_then(Value::as_str).filter(|s| is_sha256(s)).map(str::to_ascii_lowercase);
    Ok(Resolved { name, files: vec![RemoteFile { url: url.to_string(), name: fname, size: None, sha256 }], license: String::new() })
}

pub(crate) fn is_sha256(s: &str) -> bool {
    s.len() == 64 && s.bytes().all(|b| b.is_ascii_hexdigit())
}

fn query_u64(query: &str, key: &str) -> Option<u64> {
    query.split('&').find_map(|kv| kv.split_once('=').filter(|(k, _)| *k == key).and_then(|(_, v)| v.parse().ok()))
}

fn percent_encode(seg: &str) -> String {
    seg.bytes()
        .map(|b| if b.is_ascii_alphanumeric() || matches!(b, b'-' | b'_' | b'.' | b'~') { (b as char).to_string() } else { format!("%{b:02X}") })
        .collect()
}

fn percent_decode(s: &str) -> String {
    let b = s.as_bytes();
    let mut out = Vec::with_capacity(b.len());
    let mut i = 0;
    while let Some(&c) = b.get(i) {
        let hex = |j: usize| b.get(j).and_then(|h| (*h as char).to_digit(16));
        if c == b'%'
            && let (Some(h), Some(l)) = (hex(i + 1), hex(i + 2))
        {
            out.push((h * 16 + l) as u8);
            i += 3;
        } else {
            out.push(c);
            i += 1;
        }
    }
    String::from_utf8_lossy(&out).into_owned()
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn hugging_face_links_parse() {
        assert_eq!(
            Link::parse("https://huggingface.co/Carve/LaMa-ONNX/blob/main/lama_fp32.onnx").unwrap(),
            Link::HfFile { repo: "Carve/LaMa-ONNX".into(), rev: "main".into(), path: "lama_fp32.onnx".into() }
        );
        assert_eq!(
            Link::parse("  https://hf.co/org/repo/tree/v2/onnx  ").unwrap(),
            Link::HfRepo { repo: "org/repo".into(), rev: "v2".into(), subdir: "onnx".into() }
        );
        assert_eq!(Link::parse("org/repo").unwrap(), Link::HfRepo { repo: "org/repo".into(), rev: "main".into(), subdir: String::new() });
        assert!(Link::parse("https://huggingface.co/datasets/x/y").is_err());
        assert!(Link::parse("https://huggingface.co/a/b/resolve/main/../../x").is_err());
        assert!(Link::parse("https://huggingface.co/").is_err());
    }

    #[test]
    fn civitai_and_other_links_parse() {
        assert_eq!(
            Link::parse("https://civitai.com/models/4201/realistic?modelVersionId=130072").unwrap(),
            Link::Civitai { model: Some(4201), version: Some(130072) }
        );
        assert_eq!(Link::parse("https://civitai.com/api/download/models/130072").unwrap(), Link::Civitai { model: None, version: Some(130072) });
        assert_eq!(Link::parse("https://example.com/m.onnx").unwrap(), Link::Url { url: "https://example.com/m.onnx".into() });
        for bad in ["", "   ", "ftp://x/y", "file:///etc/passwd", "not a link", "a/b/c", "javascript://x"] {
            assert!(Link::parse(bad).is_err(), "{bad}");
        }
    }

    #[test]
    fn a_plain_url_names_its_file_safely() {
        let r = Link::parse("https://example.com/files/My%20Model.onnx?token=1").unwrap().resolve(&|_| Err(Error::Network("offline".into()))).unwrap();
        assert_eq!(r.files[0].name, "My_Model.onnx");
        let r = Link::parse("https://example.com/").unwrap().resolve(&|_| Err(Error::Network("offline".into()))).unwrap();
        assert_eq!(r.files[0].name, "example.com");
    }

    fn tree(files: &[(&str, u64)]) -> Value {
        Value::Array(
            files
                .iter()
                .map(|(p, s)| json!({"type": "file", "path": p, "size": s, "lfs": {"oid": "a".repeat(64), "size": s}}))
                .chain(std::iter::once(json!({"type": "directory", "path": "onnx"})))
                .collect(),
        )
    }

    #[test]
    fn a_repository_picks_its_onnx_pair_or_its_single_checkpoint() {
        let t = tree(&[("README.md", 1), ("onnx/vision_encoder.onnx", 10), ("onnx/prompt_encoder_mask_decoder.onnx", 5), ("config.json", 1)]);
        let get = |u: &str| if u.contains("/tree/") { Ok(t.clone()) } else { Ok(json!({"cardData": {"license": "apache-2.0"}})) };
        let r = Link::parse("onnx-community/sam").unwrap().resolve(&get).unwrap();
        assert_eq!(r.files.len(), 2);
        assert_eq!(r.license, "apache-2.0");
        assert_eq!(r.files[0].url, "https://huggingface.co/onnx-community/sam/resolve/main/onnx/vision_encoder.onnx");
        assert_eq!(r.files[0].size, Some(10));

        let t = tree(&[("model.safetensors", 100), ("tokenizer.json", 1), ("pytorch_model.bin", 90)]);
        let r = Link::parse("a/b").unwrap().resolve(&|_| Ok(t.clone())).unwrap();
        assert_eq!(r.files.iter().map(|f| f.name.as_str()).collect::<Vec<_>>(), ["model.safetensors"]);

        let t = tree(&[("unet/a.safetensors", 1), ("vae/b.safetensors", 1), ("text_encoder/c.safetensors", 1)]);
        assert!(matches!(Link::parse("a/b").unwrap().resolve(&|_| Ok(t.clone())), Err(Error::Invalid(_))));
        let t = tree(&[("README.md", 1)]);
        assert!(matches!(Link::parse("a/b").unwrap().resolve(&|_| Ok(t.clone())), Err(Error::NotFound(_))));
        assert!(Link::parse("a/b").unwrap().resolve(&|_| Ok(json!({"error": "nope"}))).is_err());
    }

    #[test]
    fn a_diffusion_pipeline_keeps_its_folders_and_skips_the_safety_checker() {
        let t = tree(&[
            ("model_index.json", 640),
            ("README.md", 10),
            ("safety_checker/model.onnx", 608_552_254),
            ("feature_extractor/preprocessor_config.json", 466),
            ("scheduler/scheduler_config.json", 528),
            ("text_encoder/model.onnx", 246_368_629),
            ("tokenizer/vocab.json", 1_059_962),
            ("tokenizer/merges.txt", 524_619),
            ("unet/model.onnx", 1_973_332),
            ("unet/model.onnx_data", 3_438_141_440),
            ("vae_decoder/model.onnx", 99_093_889),
            ("vae_encoder/model.onnx", 68_430_178),
        ]);
        let r = Link::parse("someone/sd-inpainting-onnx").unwrap().resolve(&|u: &str| if u.contains("/tree/") { Ok(t.clone()) } else { Ok(json!({})) }).unwrap();
        let mut names: Vec<&str> = r.files.iter().map(|f| f.name.as_str()).collect();
        names.sort_unstable();
        assert_eq!(
            names,
            [
                "model_index.json",
                "scheduler/scheduler_config.json",
                "text_encoder/model.onnx",
                "tokenizer/merges.txt",
                "tokenizer/vocab.json",
                "unet/model.onnx",
                "unet/model.onnx_data",
                "vae_decoder/model.onnx",
                "vae_encoder/model.onnx"
            ]
        );
        assert!(r.files.iter().all(|f| crate::store::valid_model_path(&f.name)));
        assert!(r.files.iter().any(|f| f.url.ends_with("/resolve/main/unet/model.onnx_data")));
    }

    #[test]
    fn civitai_picks_the_primary_file_of_the_version() {
        let info = json!({"name": "Inpaint XL", "modelVersions": [
            {"id": 1, "name": "v1", "files": [{"name": "a.safetensors", "primary": true, "downloadUrl": "https://civitai.com/api/download/models/1", "hashes": {"SHA256": "B".repeat(64)}}]},
            {"id": 2, "name": "v2", "files": [{"name": "vae.pt", "type": "VAE", "downloadUrl": "https://x/2v"}, {"name": "m.safetensors", "type": "Model", "downloadUrl": "https://x/2m"}]},
        ]});
        let r = Link::parse("https://civitai.com/models/9").unwrap().resolve(&|_| Ok(info.clone())).unwrap();
        assert_eq!(r.name, "Inpaint XL (v1)");
        assert_eq!(r.files[0].sha256, Some("b".repeat(64)));
        let r = Link::parse("https://civitai.com/models/9?modelVersionId=2").unwrap().resolve(&|_| Ok(info.clone())).unwrap();
        assert_eq!(r.files[0].url, "https://x/2m");
        assert!(Link::parse("https://civitai.com/models/9?modelVersionId=3").unwrap().resolve(&|_| Ok(info.clone())).is_err());
        assert!(Link::parse("https://civitai.com/models/9").unwrap().resolve(&|_| Ok(json!({}))).is_err());
    }

    #[test]
    fn percent_coding_round_trips_and_ignores_broken_escapes() {
        assert_eq!(percent_decode("a%20b%zz%"), "a b%zz%");
        assert_eq!(percent_encode("a b+c"), "a%20b%2Bc");
    }
}
