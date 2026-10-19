//! The models installed on this computer: one folder per model under the `Models` folder, plus
//! the `models.json` manifest that records each model's name, kind, files, source, whether it is
//! enabled and its size on disk.

use crate::{Error, ModelKind, Result};
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

/// The manifest's file name inside the models folder.
pub const MANIFEST: &str = "models.json";
/// Where downloads and imports are assembled before they become a model folder. Anything left
/// here by a crash or a closed app is removed when the store opens.
pub const STAGING: &str = ".incoming";

/// Where an installed model came from.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "camelCase")]
pub enum ModelSource {
    /// A recommended model, by catalog id.
    Catalog {
        id: String,
    },
    /// Copied from a file on this computer.
    File {
        path: String,
    },
    HuggingFace {
        url: String,
    },
    Civitai {
        url: String,
    },
    Url {
        url: String,
    },
}

impl ModelSource {
    pub fn describe(&self) -> String {
        match self {
            ModelSource::Catalog { .. } => "Recommended".into(),
            ModelSource::File { path } => format!("Imported from {path}"),
            ModelSource::HuggingFace { .. } => "Imported from Hugging Face".into(),
            ModelSource::Civitai { .. } => "Imported from Civitai".into(),
            ModelSource::Url { url } => format!("Imported from {url}"),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", from = "ManifestModel")]
pub struct InstalledModel {
    /// Folder name and stable id ([a-z0-9-]).
    pub id: String,
    pub name: String,
    /// What it's used for (in section order); it is listed in each of these sections.
    pub kinds: Vec<ModelKind>,
    /// Uses switched off: the model stays on disk and in its other sections, but the tools of
    /// these kinds don't offer it.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub disabled: Vec<ModelKind>,
    /// The recommended model this is: downloaded from the catalog, or imported from elsewhere and
    /// recognised by its files.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub catalog_id: Option<String>,
    /// File names inside the model's folder.
    pub files: Vec<String>,
    /// Size on disk, measured when the store opens and after each install.
    #[serde(default)]
    pub size_bytes: u64,
    pub source: ModelSource,
    #[serde(default)]
    pub license: String,
    /// Seconds since the Unix epoch.
    #[serde(default)]
    pub installed_at: u64,
}

impl InstalledModel {
    /// Listed in `kind`'s section.
    pub fn uses(&self, kind: ModelKind) -> bool {
        self.kinds.contains(&kind)
    }

    /// Offered by `kind`'s tools.
    pub fn enabled_for(&self, kind: ModelKind) -> bool {
        self.uses(kind) && !self.disabled.contains(&kind)
    }

    /// The main use (its first section).
    pub fn kind(&self) -> ModelKind {
        self.kinds.first().copied().unwrap_or(ModelKind::Other)
    }

    pub fn is_recommended(&self) -> bool {
        self.catalog_id.is_some()
    }
}

/// A manifest entry as stored, including what 0.10.0.1 wrote (one `kind`, one `enabled`).
#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct ManifestModel {
    id: String,
    name: String,
    #[serde(default)]
    kinds: Vec<ModelKind>,
    #[serde(default)]
    kind: Option<ModelKind>,
    #[serde(default)]
    disabled: Vec<ModelKind>,
    #[serde(default)]
    enabled: Option<bool>,
    #[serde(default)]
    catalog_id: Option<String>,
    files: Vec<String>,
    #[serde(default)]
    size_bytes: u64,
    source: ModelSource,
    #[serde(default)]
    license: String,
    #[serde(default)]
    installed_at: u64,
}

impl From<ManifestModel> for InstalledModel {
    fn from(m: ManifestModel) -> Self {
        let mut kinds = m.kinds;
        kinds.extend(m.kind);
        let mut kinds = crate::kind::normalize(&kinds);
        if kinds.is_empty() {
            kinds.push(ModelKind::Other);
        }
        let mut disabled = crate::kind::normalize(&m.disabled);
        if m.enabled == Some(false) {
            disabled = kinds.clone();
        }
        disabled.retain(|k| kinds.contains(k));
        let catalog_id = m.catalog_id.or(match &m.source {
            ModelSource::Catalog { id } => Some(id.clone()),
            _ => None,
        });
        InstalledModel {
            id: m.id,
            name: m.name,
            kinds,
            disabled,
            catalog_id,
            files: m.files,
            size_bytes: m.size_bytes,
            source: m.source,
            license: m.license,
            installed_at: m.installed_at,
        }
    }
}

/// Longest model name the AI Models tab accepts.
pub const MAX_NAME: usize = 120;

/// A name as typed, cleaned up: trimmed, no control characters. `None` if nothing is left.
pub fn clean_name(name: &str) -> Option<String> {
    let n: String = name.chars().filter(|c| !c.is_control()).collect::<String>().trim().chars().take(MAX_NAME).collect();
    let n = n.trim().to_string();
    (!n.is_empty()).then_some(n)
}

#[derive(Default, Serialize, Deserialize)]
struct Manifest {
    #[serde(default)]
    version: u32,
    #[serde(default)]
    models: Vec<InstalledModel>,
}

/// The models folder and what is installed in it.
pub struct Store {
    dir: PathBuf,
    models: Vec<InstalledModel>,
    /// Set when the manifest couldn't be read and was set aside (shown in the AI tab).
    pub warning: Option<String>,
}

impl Store {
    /// Opens (creating if needed) the models folder `dir`. A damaged manifest is renamed to
    /// `models.json.damaged` and the store starts empty rather than failing; entries whose folder
    /// is gone are dropped; leftovers of interrupted downloads are deleted.
    pub fn open(dir: &Path) -> Result<Store> {
        std::fs::create_dir_all(dir).map_err(|e| Error::io("Couldn't create the models folder", e))?;
        let path = dir.join(MANIFEST);
        let mut warning = None;
        let manifest = match std::fs::read_to_string(&path) {
            Ok(text) => match serde_json::from_str::<Manifest>(&text) {
                Ok(m) => m,
                Err(e) => {
                    let aside = dir.join(format!("{MANIFEST}.damaged"));
                    let _ = std::fs::rename(&path, &aside);
                    let msg = format!("The model list was damaged ({e}) and was set aside as {}. Models on disk can be imported again.", aside.display());
                    log::warn!("{msg}");
                    warning = Some(msg);
                    Manifest::default()
                }
            },
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Manifest::default(),
            Err(e) => return Err(Error::io("Couldn't read the model list", e)),
        };
        let _ = std::fs::remove_dir_all(dir.join(STAGING));
        let mut store = Store { dir: dir.to_path_buf(), models: Vec::new(), warning };
        let mut seen = std::collections::HashSet::new();
        for mut m in manifest.models {
            if !valid_id(&m.id) || !seen.insert(m.id.clone()) || m.files.iter().any(|f| !valid_model_path(f)) {
                log::warn!("ignoring model entry {:?}: bad id or file name", m.id);
                continue;
            }
            let folder = store.model_dir(&m.id);
            if !folder.is_dir() {
                log::info!("model {} is gone from disk; removing it from the list", m.id);
                continue;
            }
            m.size_bytes = dir_size(&folder);
            store.recognise(&mut m);
            store.models.push(m);
        }
        Ok(store)
    }

    pub fn dir(&self) -> &Path {
        &self.dir
    }

    pub fn models(&self) -> &[InstalledModel] {
        &self.models
    }

    pub fn get(&self, id: &str) -> Option<&InstalledModel> {
        self.models.iter().find(|m| m.id == id)
    }

    pub fn model_dir(&self, id: &str) -> PathBuf {
        self.dir.join(id)
    }

    pub fn staging_dir(&self) -> PathBuf {
        self.dir.join(STAGING)
    }

    /// Paths of a model's files, for the tool that loads it.
    pub fn files_of(&self, id: &str) -> Option<Vec<PathBuf>> {
        let m = self.get(id)?;
        let dir = self.model_dir(id);
        Some(m.files.iter().map(|f| dir.join(f)).collect())
    }

    /// A free id for a new model named `name` ("SAM 2.1 Tiny" → "sam-2-1-tiny", then "-2", …).
    pub fn free_id(&self, name: &str) -> String {
        let base = slug(name);
        let taken = |id: &str| self.get(id).is_some() || self.model_dir(id).exists();
        if !taken(&base) {
            return base;
        }
        (2..10_000).map(|n| format!("{base}-{n}")).find(|id| !taken(id)).unwrap_or_else(|| format!("{base}-{}", now()))
    }

    /// Recognises a recommended model imported from somewhere else (same files and sizes), so
    /// its recommended row shows it as downloaded; it also gets the uses the catalog lists.
    fn recognise(&self, m: &mut InstalledModel) {
        if m.catalog_id.is_none() {
            let dir = self.model_dir(&m.id);
            let files: Vec<(String, u64)> = m.files.iter().map(|f| (f.clone(), std::fs::metadata(dir.join(f)).map(|x| x.len()).unwrap_or(0))).collect();
            let link = match &m.source {
                ModelSource::HuggingFace { url } | ModelSource::Url { url } => Some(url.as_str()),
                _ => None,
            };
            if let Some(c) = crate::catalog::identify(&files, link) {
                m.catalog_id = Some(c.id.to_string());
            }
        }
        if let Some(c) = m.catalog_id.as_deref().and_then(|id| crate::RECOMMENDED.iter().find(|c| c.id == id)) {
            let mut kinds = m.kinds.clone();
            kinds.extend(c.kinds.iter().copied());
            kinds.retain(|k| *k != ModelKind::Other || c.kinds.contains(&ModelKind::Other));
            m.kinds = crate::kind::normalize(&kinds);
            if m.license.is_empty() {
                m.license = c.license.to_string();
            }
        }
    }

    /// The installed copy of the recommended model `catalog_id`, if any.
    pub fn of_catalog(&self, catalog_id: &str) -> Option<&InstalledModel> {
        self.models.iter().find(|m| m.catalog_id.as_deref() == Some(catalog_id))
    }

    /// Records a model whose folder (`model_dir(id)`) is complete, and saves the manifest.
    pub fn add(&mut self, mut model: InstalledModel) -> Result<()> {
        if !valid_id(&model.id) || self.get(&model.id).is_some() {
            return Err(Error::Invalid(format!("A model with the id {:?} already exists.", model.id)));
        }
        model.kinds = crate::kind::normalize(&model.kinds);
        if model.kinds.is_empty() {
            model.kinds.push(ModelKind::Other);
        }
        model.size_bytes = dir_size(&self.model_dir(&model.id));
        if model.installed_at == 0 {
            model.installed_at = now();
        }
        self.recognise(&mut model);
        self.models.push(model);
        self.save()
    }

    fn get_mut(&mut self, id: &str) -> Result<&mut InstalledModel> {
        self.models.iter_mut().find(|m| m.id == id).ok_or_else(|| Error::NotFound(format!("No model {id:?} is installed.")))
    }

    /// Switches one use of a model on or off (the other sections it's listed in are unchanged).
    pub fn set_enabled(&mut self, id: &str, kind: ModelKind, on: bool) -> Result<()> {
        let m = self.get_mut(id)?;
        if !m.uses(kind) {
            return Err(Error::Invalid(format!("{} isn't used for {}.", m.name, kind.title())));
        }
        if m.enabled_for(kind) == on {
            return Ok(());
        }
        if on {
            m.disabled.retain(|k| *k != kind);
        } else {
            m.disabled.push(kind);
            m.disabled = crate::kind::normalize(&m.disabled);
        }
        self.save()
    }

    /// The model's files are now these (a checkpoint made into a pipeline).
    pub fn set_files(&mut self, id: &str, files: Vec<String>) -> Result<()> {
        if files.is_empty() || files.iter().any(|f| !valid_model_path(f)) {
            return Err(Error::Invalid("internal error: unusable model file names".into()));
        }
        self.get_mut(id)?.files = files;
        self.save()
    }

    /// Measures a model's size on disk again.
    pub fn refresh_size(&mut self, id: &str) {
        let size = dir_size(&self.model_dir(id));
        if let Ok(m) = self.get_mut(id) {
            m.size_bytes = size;
        }
    }

    /// Renames a model (it keeps its folder and id).
    pub fn rename(&mut self, id: &str, name: &str) -> Result<()> {
        let name = clean_name(name).ok_or_else(|| Error::Invalid("Type a name for the model.".into()))?;
        let m = self.get_mut(id)?;
        if m.name == name {
            return Ok(());
        }
        m.name = name;
        self.save()
    }

    /// Sets what a model is used for (at least one use).
    pub fn set_kinds(&mut self, id: &str, kinds: &[ModelKind]) -> Result<()> {
        let kinds = crate::kind::normalize(kinds);
        if kinds.is_empty() {
            return Err(Error::Invalid("Choose at least one use for the model.".into()));
        }
        let m = self.get_mut(id)?;
        m.disabled.retain(|k| kinds.contains(k));
        m.kinds = kinds;
        self.save()
    }

    /// Deletes a model's folder and its entry.
    pub fn remove(&mut self, id: &str) -> Result<()> {
        let at = self.models.iter().position(|m| m.id == id).ok_or_else(|| Error::NotFound(format!("No model {id:?} is installed.")))?;
        let folder = self.model_dir(id);
        match std::fs::remove_dir_all(&folder) {
            Ok(()) => {}
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {}
            Err(e) => return Err(Error::io("Couldn't delete the model's files", e)),
        }
        self.models.remove(at);
        self.save()
    }

    /// Writes `models.json` (to a temporary file first, so a crash mid-write keeps the old one).
    pub fn save(&self) -> Result<()> {
        let manifest = Manifest { version: 1, models: self.models.clone() };
        let text = serde_json::to_string_pretty(&manifest).map_err(|e| Error::Invalid(e.to_string()))?;
        let tmp = self.dir.join(format!("{MANIFEST}.tmp"));
        std::fs::write(&tmp, text).map_err(|e| Error::io("Couldn't save the model list", e))?;
        std::fs::rename(&tmp, self.dir.join(MANIFEST)).map_err(|e| Error::io("Couldn't save the model list", e))
    }
}

/// Bytes used by the files under `dir` (not following links; unreadable entries count 0).
pub fn dir_size(dir: &Path) -> u64 {
    fn walk(dir: &Path, depth: u32) -> u64 {
        if depth > 16 {
            return 0;
        }
        let Ok(entries) = std::fs::read_dir(dir) else { return 0 };
        entries
            .flatten()
            .map(|e| match e.file_type() {
                Ok(t) if t.is_dir() => walk(&e.path(), depth + 1),
                Ok(t) if t.is_file() => e.metadata().map(|m| m.len()).unwrap_or(0),
                _ => 0,
            })
            .fold(0u64, u64::saturating_add)
    }
    walk(dir, 0)
}

/// Model ids are folder names: lowercase letters, digits and dashes, 1–80 characters.
pub fn valid_id(id: &str) -> bool {
    !id.is_empty() && id.len() <= 80 && id.bytes().all(|b| b.is_ascii_lowercase() || b.is_ascii_digit() || b == b'-') && !id.starts_with('-')
}

/// A file name inside a model folder: no separators, no `..`, nothing hidden, no NULs.
pub fn valid_file_name(name: &str) -> bool {
    !name.is_empty() && name.len() <= 200 && !name.starts_with('.') && !name.contains(['/', '\\', '\0', ':']) && name.chars().all(|c| !c.is_control())
}

/// A file inside a model folder, possibly in a subfolder of it ("unet/model.onnx", as diffusion
/// pipelines are laid out): one to three valid names joined by `/`.
pub fn valid_model_path(path: &str) -> bool {
    let parts: Vec<&str> = path.split('/').collect();
    (1..=3).contains(&parts.len()) && parts.iter().all(|p| valid_file_name(p))
}

/// Turns any text into a usable file name: separators and odd characters become `_`.
pub fn safe_file_name(name: &str) -> String {
    let cleaned: String = name
        .chars()
        .map(|c| if c.is_ascii_alphanumeric() || matches!(c, '.' | '-' | '_' | '+') { c } else { '_' })
        .collect::<String>()
        .trim_start_matches('.')
        .chars()
        .take(160)
        .collect();
    if cleaned.is_empty() { "model.bin".into() } else { cleaned }
}

fn slug(name: &str) -> String {
    let mut out = String::new();
    for c in name.chars() {
        if c.is_ascii_alphanumeric() {
            out.push(c.to_ascii_lowercase());
        } else if !out.ends_with('-') && !out.is_empty() {
            out.push('-');
        }
        if out.len() >= 60 {
            break;
        }
    }
    let out = out.trim_end_matches('-').to_string();
    if out.is_empty() { "model".into() } else { out }
}

pub(crate) fn now() -> u64 {
    std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;

    pub(crate) fn temp(tag: &str) -> PathBuf {
        let d = std::env::temp_dir().join(format!("photocraft-ml-{tag}-{}-{}", std::process::id(), now()));
        let _ = std::fs::remove_dir_all(&d);
        d
    }

    fn model(store: &Store, name: &str) -> InstalledModel {
        let id = store.free_id(name);
        std::fs::create_dir_all(store.model_dir(&id)).unwrap();
        std::fs::write(store.model_dir(&id).join("weights.onnx"), vec![7u8; 1234]).unwrap();
        InstalledModel {
            id,
            name: name.into(),
            kinds: vec![ModelKind::Removal, ModelKind::Reflection],
            disabled: Vec::new(),
            catalog_id: None,
            files: vec!["weights.onnx".into()],
            size_bytes: 0,
            source: ModelSource::File { path: "/tmp/x.onnx".into() },
            license: String::new(),
            installed_at: 0,
        }
    }

    #[test]
    fn models_survive_a_reopen_with_their_size_and_switch() {
        let dir = temp("reopen");
        let mut s = Store::open(&dir).unwrap();
        let m = model(&s, "LaMa (big)");
        assert_eq!(m.id, "lama-big");
        s.add(m).unwrap();
        s.set_enabled("lama-big", ModelKind::Reflection, false).unwrap();
        s.rename("lama-big", "  My LaMa\n ").unwrap();
        assert!(s.set_enabled("lama-big", ModelKind::Inpainting, false).is_err());
        assert!(s.rename("lama-big", " \t ").is_err());
        let s = Store::open(&dir).unwrap();
        let m = s.get("lama-big").unwrap();
        assert_eq!(m.size_bytes, 1234);
        assert_eq!(m.name, "My LaMa");
        assert!(m.enabled_for(ModelKind::Removal) && !m.enabled_for(ModelKind::Reflection));
        assert!(m.uses(ModelKind::Reflection));
        assert_eq!(s.files_of("lama-big").unwrap(), vec![dir.join("lama-big").join("weights.onnx")]);
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn delete_removes_files_and_entry() {
        let dir = temp("delete");
        let mut s = Store::open(&dir).unwrap();
        let m = model(&s, "MI-GAN");
        s.add(m).unwrap();
        s.remove("mi-gan").unwrap();
        assert!(!dir.join("mi-gan").exists());
        assert!(Store::open(&dir).unwrap().get("mi-gan").is_none());
        assert!(matches!(s.remove("mi-gan"), Err(Error::NotFound(_))));
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn manifests_from_0_10_0_1_load_with_one_kind_and_a_switch() {
        let dir = temp("legacy");
        std::fs::create_dir_all(dir.join("old")).unwrap();
        std::fs::create_dir_all(dir.join("lama")).unwrap();
        std::fs::write(dir.join("lama").join("lama_fp32.onnx"), b"x").unwrap();
        let json = serde_json::json!({"version": 1, "models": [
            {"id": "old", "name": "Old", "kind": "segmentation", "enabled": false, "files": [], "source": {"type": "url", "url": "u"}},
            {"id": "lama", "name": "LaMa", "kind": "removal", "enabled": true, "files": ["lama_fp32.onnx"], "source": {"type": "catalog", "id": "lama"}},
        ]});
        std::fs::write(dir.join(MANIFEST), json.to_string()).unwrap();
        let s = Store::open(&dir).unwrap();
        let old = s.get("old").unwrap();
        assert_eq!(old.kinds, vec![ModelKind::Segmentation]);
        assert!(!old.enabled_for(ModelKind::Segmentation));
        let lama = s.get("lama").unwrap();
        assert_eq!(lama.catalog_id.as_deref(), Some("lama"));
        assert_eq!(lama.kinds, vec![ModelKind::Removal, ModelKind::Reflection], "gets the catalog's uses");
        assert!(lama.enabled_for(ModelKind::Reflection));
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn a_recommended_model_imported_elsewhere_is_recognised() {
        let dir = temp("recognise");
        let mut s = Store::open(&dir).unwrap();
        let id = s.free_id("whatever");
        std::fs::create_dir_all(s.model_dir(&id)).unwrap();
        std::fs::write(s.model_dir(&id).join("mobile_sam_image_encoder.onnx"), b"a").unwrap();
        std::fs::write(s.model_dir(&id).join("sam_mask_decoder_single.onnx"), b"b").unwrap();
        let m = InstalledModel {
            id: id.clone(),
            name: "MobileSAM".into(),
            kinds: vec![ModelKind::Other],
            disabled: Vec::new(),
            catalog_id: None,
            files: vec!["mobile_sam_image_encoder.onnx".into(), "sam_mask_decoder_single.onnx".into()],
            size_bytes: 0,
            source: ModelSource::HuggingFace { url: "Acly/MobileSAM".into() },
            license: String::new(),
            installed_at: 0,
        };
        s.add(m).unwrap();
        let m = s.of_catalog("mobilesam").unwrap();
        assert_eq!(m.id, id);
        assert_eq!(m.kinds, vec![ModelKind::Segmentation], "a wrong guess of Other gives way to the catalog's use");
        assert_eq!(m.license, "MIT");
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn uses_can_be_changed_but_not_emptied() {
        let dir = temp("kinds");
        let mut s = Store::open(&dir).unwrap();
        let m = model(&s, "Thing");
        s.add(m).unwrap();
        s.set_enabled("thing", ModelKind::Reflection, false).unwrap();
        assert!(s.set_kinds("thing", &[]).is_err());
        s.set_kinds("thing", &[ModelKind::Removal, ModelKind::Other]).unwrap();
        let m = s.get("thing").unwrap();
        assert_eq!(m.kinds, vec![ModelKind::Removal, ModelKind::Other]);
        assert!(m.disabled.is_empty());
        assert_eq!(clean_name(&"x".repeat(500)).map(|n| n.len()), Some(MAX_NAME));
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn a_damaged_manifest_is_set_aside_not_fatal() {
        let dir = temp("damaged");
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join(MANIFEST), "{ not json").unwrap();
        let s = Store::open(&dir).unwrap();
        assert!(s.models().is_empty());
        assert!(s.warning.is_some());
        assert!(dir.join("models.json.damaged").exists());
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn hostile_entries_and_missing_folders_are_dropped() {
        let dir = temp("hostile");
        std::fs::create_dir_all(dir.join("ok")).unwrap();
        let json = serde_json::json!({"models": [
            {"id": "../../etc", "name": "x", "kind": "removal", "files": [], "source": {"type": "url", "url": "u"}},
            {"id": "ok", "name": "x", "kind": "removal", "files": ["../../passwd"], "source": {"type": "url", "url": "u"}},
            {"id": "gone", "name": "x", "kind": "removal", "files": ["a.onnx"], "source": {"type": "url", "url": "u"}},
        ]});
        std::fs::write(dir.join(MANIFEST), json.to_string()).unwrap();
        std::fs::create_dir_all(dir.join(STAGING).join("half")).unwrap();
        let s = Store::open(&dir).unwrap();
        assert!(s.models().is_empty());
        assert!(!dir.join(STAGING).exists());
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn names_become_safe_ids_and_file_names() {
        let dir = temp("ids");
        let s = Store::open(&dir).unwrap();
        assert_eq!(s.free_id("SAM 2.1 Hiera Tiny"), "sam-2-1-hiera-tiny");
        assert_eq!(s.free_id("!!!"), "model");
        assert_eq!(s.free_id("日本語"), "model");
        assert_eq!(safe_file_name("../evil/../x.onnx"), "_evil_.._x.onnx");
        assert_eq!(safe_file_name("..."), "model.bin");
        assert!(valid_file_name("lama_fp32.onnx"));
        assert!(!valid_file_name("../x"));
        assert!(!valid_file_name(".hidden"));
        assert!(valid_model_path("unet/model.onnx") && valid_model_path("lama_fp32.onnx"));
        for bad in ["../x", "unet/../x", "/abs", "a//b", "a/b/c/d", "unet/.hidden", "a\\b"] {
            assert!(!valid_model_path(bad), "{bad}");
        }
        let _ = std::fs::remove_dir_all(dir);
    }
}
