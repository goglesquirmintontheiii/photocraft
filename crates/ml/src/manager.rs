//! The model manager the UI drives: the store behind a lock, plus downloads and imports that run
//! on worker threads and report progress through [`Job`]s.

use crate::source::{Link, RemoteFile, Resolved};
use crate::store::{InstalledModel, ModelSource, Store, safe_file_name, valid_model_path};
use crate::{CatalogModel, Error, ModelKind, RECOMMENDED, Result};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, MutexGuard, PoisonError};
use std::time::Instant;

pub type JobId = u64;

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum JobKind {
    /// Downloading a recommended model.
    Download { catalog_id: String },
    /// Importing from a file or a link.
    Import,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum JobState {
    /// Looking up what a link points at.
    Resolving,
    Running,
    Done,
    Failed(String),
    Cancelled,
}

/// A download or import, as the AI tab shows it.
#[derive(Clone, Debug)]
pub struct Job {
    pub id: JobId,
    pub kind: JobKind,
    pub label: String,
    /// What the model will be used for.
    pub model_kinds: Vec<ModelKind>,
    pub state: JobState,
    pub done_bytes: u64,
    pub total_bytes: Option<u64>,
    pub started: Instant,
    /// When the job reached Done, Failed or Cancelled.
    pub finished: Option<Instant>,
    cancel: Arc<AtomicBool>,
}

impl Job {
    pub fn active(&self) -> bool {
        matches!(self.state, JobState::Resolving | JobState::Running)
    }

    /// 0–1, when the total is known.
    pub fn fraction(&self) -> Option<f32> {
        let total = self.total_bytes.filter(|t| *t > 0)?;
        Some((self.done_bytes as f64 / total as f64).clamp(0.0, 1.0) as f32)
    }

    /// Seconds since the job started (to when it finished).
    pub fn elapsed(&self) -> f64 {
        self.finished.unwrap_or_else(Instant::now).duration_since(self.started).as_secs_f64()
    }

    /// Estimated seconds left, from the average speed so far.
    pub fn eta(&self) -> Option<f64> {
        let total = self.total_bytes?;
        let elapsed = self.elapsed();
        if !self.active() || self.done_bytes == 0 || elapsed < 1.0 {
            return None;
        }
        let rate = self.done_bytes as f64 / elapsed;
        Some(total.saturating_sub(self.done_bytes) as f64 / rate)
    }

    /// Bytes per second so far.
    pub fn speed(&self) -> Option<f64> {
        let e = self.elapsed();
        (e >= 1.0 && self.done_bytes > 0).then(|| self.done_bytes as f64 / e)
    }
}

struct Inner {
    store: Store,
    jobs: Vec<Job>,
    next: JobId,
}

/// Shared handle; clones point at the same store and jobs.
#[derive(Clone)]
pub struct Manager {
    inner: Arc<Mutex<Inner>>,
}

/// Things the manager knows about a recommended model, for its row in the AI tab.
#[derive(Clone, Debug)]
pub struct CatalogRow {
    pub model: &'static CatalogModel,
    /// The installed copy, if downloaded.
    pub installed: Option<InstalledModel>,
    /// The download running or last finished for it.
    pub job: Option<Job>,
}

impl Manager {
    /// Opens the models folder `dir` (PhotoCraft's data folder + `Models`).
    pub fn open(dir: &Path) -> Result<Manager> {
        let store = Store::open(dir)?;
        crate::accounts::load(dir);
        Ok(Manager { inner: Arc::new(Mutex::new(Inner { store, jobs: Vec::new(), next: 1 })) })
    }

    fn lock(&self) -> MutexGuard<'_, Inner> {
        self.inner.lock().unwrap_or_else(PoisonError::into_inner)
    }

    pub fn dir(&self) -> PathBuf {
        self.lock().store.dir().to_path_buf()
    }

    pub fn warning(&self) -> Option<String> {
        self.lock().store.warning.clone()
    }

    pub fn installed(&self) -> Vec<InstalledModel> {
        self.lock().store.models().to_vec()
    }

    /// Installed, enabled models of `kind`, best first: what a tool's model dropdown lists.
    pub fn available(&self, kind: ModelKind) -> Vec<InstalledModel> {
        crate::available(self.lock().store.models(), kind)
    }

    /// File paths of an installed model, for loading it.
    pub fn files_of(&self, id: &str) -> Option<Vec<PathBuf>> {
        self.lock().store.files_of(id)
    }

    pub fn jobs(&self) -> Vec<Job> {
        self.lock().jobs.clone()
    }

    pub fn any_active(&self) -> bool {
        self.lock().jobs.iter().any(Job::active)
    }

    /// Every recommended model with its install and download state.
    pub fn catalog(&self) -> Vec<CatalogRow> {
        let inner = self.lock();
        RECOMMENDED
            .iter()
            .map(|model| CatalogRow {
                model,
                installed: inner.store.of_catalog(model.id).cloned(),
                job: inner.jobs.iter().rev().find(|j| j.kind == (JobKind::Download { catalog_id: model.id.to_string() })).cloned(),
            })
            .collect()
    }

    /// Switches one use of a model on or off.
    pub fn set_enabled(&self, id: &str, kind: ModelKind, on: bool) -> Result<()> {
        self.lock().store.set_enabled(id, kind, on)
    }

    /// Switches every use of a model on or off at once.
    pub fn set_enabled_everywhere(&self, id: &str, on: bool) -> Result<()> {
        let mut inner = self.lock();
        let kinds = inner.store.models().iter().find(|m| m.id == id).map(|m| m.kinds.clone()).ok_or_else(|| Error::NotFound("That model isn't installed.".into()))?;
        for k in kinds {
            inner.store.set_enabled(id, k, on)?;
        }
        Ok(())
    }

    pub fn rename(&self, id: &str, name: &str) -> Result<()> {
        self.lock().store.rename(id, name)
    }

    pub fn set_kinds(&self, id: &str, kinds: &[ModelKind]) -> Result<()> {
        self.lock().store.set_kinds(id, kinds)
    }

    /// The saved Hugging Face and Civitai tokens.
    pub fn tokens(&self) -> crate::Tokens {
        crate::accounts::load(self.lock().store.dir())
    }

    pub fn set_tokens(&self, tokens: &crate::Tokens) -> Result<()> {
        crate::accounts::save(self.lock().store.dir(), tokens)
    }

    pub fn delete(&self, id: &str) -> Result<()> {
        self.lock().store.remove(id)
    }

    pub fn cancel(&self, job: JobId) {
        if let Some(j) = self.lock().jobs.iter().find(|j| j.id == job) {
            j.cancel.store(true, Ordering::Relaxed);
        }
    }

    /// Removes a finished job from the list.
    pub fn dismiss(&self, job: JobId) {
        self.lock().jobs.retain(|j| j.id != job || j.active());
    }

    fn new_job(&self, kind: JobKind, label: String, model_kinds: Vec<ModelKind>, total: Option<u64>, state: JobState) -> (JobId, Arc<AtomicBool>) {
        let mut inner = self.lock();
        let id = inner.next;
        inner.next = inner.next.saturating_add(1);
        let cancel = Arc::new(AtomicBool::new(false));
        inner.jobs.push(Job {
            id,
            kind,
            label,
            model_kinds,
            state,
            done_bytes: 0,
            total_bytes: total,
            started: Instant::now(),
            finished: None,
            cancel: cancel.clone(),
        });
        (id, cancel)
    }

    fn update(&self, id: JobId, f: impl FnOnce(&mut Job)) {
        if let Some(j) = self.lock().jobs.iter_mut().find(|j| j.id == id) {
            f(j);
        }
    }

    fn finish(&self, id: JobId, result: Result<()>) {
        if let Err(e) = &result {
            log::warn!("model job {id} failed: {e}");
        }
        self.update(id, |j| {
            j.state = match result {
                Ok(()) => JobState::Done,
                Err(Error::Cancelled) => JobState::Cancelled,
                Err(e) => JobState::Failed(e.to_string()),
            };
            j.finished = Some(Instant::now());
        });
    }

    /// Starts downloading the recommended model `catalog_id`.
    pub fn download(&self, catalog_id: &str) -> Result<JobId> {
        let model = RECOMMENDED.iter().find(|m| m.id == catalog_id).ok_or_else(|| Error::NotFound(format!("No recommended model {catalog_id:?}.")))?;
        {
            let inner = self.lock();
            if inner.store.of_catalog(model.id).is_some() {
                return Err(Error::Invalid(format!("{} is already downloaded.", model.name)));
            }
            if inner.jobs.iter().any(|j| j.active() && j.kind == (JobKind::Download { catalog_id: model.id.to_string() })) {
                return Err(Error::Invalid(format!("{} is already downloading.", model.name)));
            }
        }
        let files: Vec<RemoteFile> = model.files.iter().map(|f| f.remote()).collect();
        let total = model.files.iter().map(|f| f.size).fold(0u64, u64::saturating_add);
        let (id, cancel) =
            self.new_job(JobKind::Download { catalog_id: model.id.to_string() }, model.name.to_string(), model.kinds.to_vec(), Some(total), JobState::Running);
        let this = self.clone();
        let resolved = Resolved { name: model.name.to_string(), files, license: model.license.to_string() };
        let source = ModelSource::Catalog { id: model.id.to_string() };
        let kinds = model.kinds.to_vec();
        spawn(move || {
            let r = this.fetch_and_install(id, &cancel, resolved, &kinds, source);
            this.finish(id, r);
        });
        Ok(id)
    }

    /// Starts importing from a link (Hugging Face, Civitai or any URL) as a model used for
    /// `kinds` (at least one).
    pub fn import_link(&self, link: &str, kinds: &[ModelKind], name: Option<String>) -> Result<JobId> {
        let kinds = uses(kinds)?;
        let parsed = Link::parse(link)?;
        let source = parsed.source(link);
        let label = name.clone().filter(|n| !n.trim().is_empty()).unwrap_or_else(|| link.trim().to_string());
        let (id, cancel) = self.new_job(JobKind::Import, label, kinds.clone(), None, JobState::Resolving);
        let this = self.clone();
        spawn(move || {
            let r = (|| {
                let mut resolved = parsed.resolve(&crate::http::get_json)?;
                if let Some(n) = name.filter(|n| !n.trim().is_empty()) {
                    resolved.name = n.trim().to_string();
                }
                let total = resolved.files.iter().map(|f| f.size).try_fold(0u64, |a, s| s.map(|s| a.saturating_add(s)));
                let label = resolved.name.clone();
                this.update(id, |j| {
                    j.state = JobState::Running;
                    j.total_bytes = total;
                    j.label = label;
                });
                this.fetch_and_install(id, &cancel, resolved, &kinds, source)
            })();
            this.finish(id, r);
        });
        Ok(id)
    }

    /// Starts importing model files from this computer (an ONNX model and its external data
    /// file can be imported together) as one model used for `kinds`.
    pub fn import_files(&self, paths: &[PathBuf], kinds: &[ModelKind], name: Option<String>) -> Result<JobId> {
        let kinds = uses(kinds)?;
        if paths.is_empty() {
            return Err(Error::Invalid("Choose a model file to import.".into()));
        }
        let mut total = 0u64;
        for p in paths {
            let meta = std::fs::metadata(p).map_err(|e| Error::io(&format!("Couldn't read {}", p.display()), e))?;
            if !meta.is_file() {
                return Err(Error::Invalid(format!("{} isn't a file.", p.display())));
            }
            total = total.saturating_add(meta.len());
        }
        let first = paths.first().map(|p| p.file_stem().map(|s| s.to_string_lossy().into_owned()).unwrap_or_default()).unwrap_or_default();
        let display = name
            .filter(|n| !n.trim().is_empty())
            .map(|n| n.trim().to_string())
            .unwrap_or_else(|| if first.is_empty() { "Imported model".into() } else { first });
        let files: Vec<(PathBuf, String)> =
            paths.iter().map(|p| (p.clone(), safe_file_name(&p.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default()))).collect();
        let source = paths.first().map(|p| p.display().to_string()).unwrap_or_default();
        self.start_copy(files, total, display, kinds, source)
    }

    /// Starts importing a Stable Diffusion ONNX pipeline folder from this computer (the
    /// diffusers layout: `model_index.json` beside `unet/`, `text_encoder/`, `vae_encoder/`,
    /// `vae_decoder/`, `tokenizer/`, `scheduler/`), keeping its folders.
    pub fn import_folder(&self, dir: &Path, kinds: &[ModelKind], name: Option<String>) -> Result<JobId> {
        let kinds = uses(kinds)?;
        let files = crate::source::pipeline_files_in(dir)?;
        let total = files.iter().map(|(p, _)| std::fs::metadata(p).map(|m| m.len()).unwrap_or(0)).fold(0u64, u64::saturating_add);
        let folder = dir.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default();
        let display = name
            .filter(|n| !n.trim().is_empty())
            .map(|n| n.trim().to_string())
            .unwrap_or_else(|| if folder.is_empty() { "Imported model".into() } else { folder });
        self.start_copy(files, total, display, kinds, dir.display().to_string())
    }

    fn start_copy(&self, files: Vec<(PathBuf, String)>, total: u64, display: String, kinds: Vec<ModelKind>, source: String) -> Result<JobId> {
        let (id, cancel) = self.new_job(JobKind::Import, display.clone(), kinds.clone(), Some(total), JobState::Running);
        let this = self.clone();
        spawn(move || {
            let r = this.copy_and_install(id, &cancel, &files, display, &kinds, source);
            this.finish(id, r);
        });
        Ok(id)
    }

    /// The staging folder for a new model, and its future id.
    fn stage(&self, name: &str) -> Result<(String, PathBuf)> {
        let inner = self.lock();
        let id = inner.store.free_id(name);
        let staging = inner.store.staging_dir().join(&id);
        drop(inner);
        std::fs::create_dir_all(&staging).map_err(|e| Error::io("Couldn't create a folder for the model", e))?;
        Ok((id, staging))
    }

    /// Moves a complete staging folder into place and records the model.
    fn install(&self, id: &str, staging: &Path, model: InstalledModel) -> Result<()> {
        let mut inner = self.lock();
        let dest = inner.store.model_dir(id);
        if dest.exists() {
            return Err(Error::Invalid(format!("A folder named {id} is already in the models folder.")));
        }
        std::fs::rename(staging, &dest).map_err(|e| Error::io("Couldn't move the model into place", e))?;
        inner.store.add(model)
    }

    fn fetch_and_install(&self, id: JobId, cancel: &AtomicBool, resolved: Resolved, kinds: &[ModelKind], source: ModelSource) -> Result<()> {
        if resolved.files.is_empty() || resolved.files.iter().any(|f| !valid_model_path(&f.name)) {
            return Err(Error::Invalid("The model's files have unusable names.".into()));
        }
        let (model_id, staging) = self.stage(&resolved.name)?;
        let result = (|| {
            let mut before = 0u64;
            for f in &resolved.files {
                let dest = staging.join(&f.name);
                if let Some(parent) = dest.parent() {
                    std::fs::create_dir_all(parent).map_err(|e| Error::io("Couldn't create a folder for the model", e))?;
                }
                let mut last = Instant::now();
                crate::http::download(&f.url, &dest, f.size, f.sha256.as_deref(), cancel, &mut |done, total| {
                    // A few updates a second are enough for the progress bar.
                    if last.elapsed().as_millis() >= 100 || total == Some(done) {
                        last = Instant::now();
                        self.update(id, |j| {
                            j.done_bytes = before.saturating_add(done);
                            if resolved.files.len() == 1 && j.total_bytes.is_none() {
                                j.total_bytes = total;
                            }
                        });
                    }
                })?;
                before = before.saturating_add(std::fs::metadata(&dest).map(|m| m.len()).unwrap_or(0));
            }
            self.install(
                &model_id,
                &staging,
                InstalledModel {
                    id: model_id.clone(),
                    name: resolved.name.clone(),
                    kinds: kinds.to_vec(),
                    disabled: Vec::new(),
                    catalog_id: match &source {
                        ModelSource::Catalog { id } => Some(id.clone()),
                        _ => None,
                    },
                    files: resolved.files.iter().map(|f| f.name.clone()).collect(),
                    size_bytes: 0,
                    source,
                    license: resolved.license.clone(),
                    installed_at: 0,
                },
            )
        })();
        if result.is_err() {
            let _ = std::fs::remove_dir_all(&staging);
        }
        result
    }

    fn copy_and_install(&self, id: JobId, cancel: &AtomicBool, files: &[(PathBuf, String)], name: String, kinds: &[ModelKind], source: String) -> Result<()> {
        let (model_id, staging) = self.stage(&name)?;
        let result = (|| {
            let mut names: Vec<String> = Vec::new();
            let mut before = 0u64;
            for (p, fname) in files {
                if !valid_model_path(fname) {
                    return Err(Error::Invalid(format!("{fname} isn't a usable file name.")));
                }
                if names.contains(fname) {
                    return Err(Error::Invalid(format!("Two of the files are named {fname}.")));
                }
                let mut from = std::fs::File::open(p).map_err(|e| Error::io(&format!("Couldn't open {}", p.display()), e))?;
                let dest = staging.join(fname);
                if let Some(parent) = dest.parent() {
                    std::fs::create_dir_all(parent).map_err(|e| Error::io("Couldn't create a folder for the model", e))?;
                }
                let mut to = std::fs::File::create(&dest).map_err(|e| Error::io("Couldn't create the model file", e))?;
                let mut hasher = <sha2::Sha256 as sha2::Digest>::new();
                let mut last = Instant::now();
                let n = crate::http::pump(&mut from, &mut to, &mut hasher, 0, None, cancel, &mut |done, _| {
                    if last.elapsed().as_millis() >= 100 {
                        last = Instant::now();
                        self.update(id, |j| j.done_bytes = before.saturating_add(done));
                    }
                })?;
                before = before.saturating_add(n);
                self.update(id, |j| j.done_bytes = before);
                names.push(fname.clone());
            }
            let source = ModelSource::File { path: source };
            self.install(
                &model_id,
                &staging,
                InstalledModel {
                    id: model_id.clone(),
                    name,
                    kinds: kinds.to_vec(),
                    disabled: Vec::new(),
                    catalog_id: None,
                    files: names,
                    size_bytes: 0,
                    source,
                    license: String::new(),
                    installed_at: 0,
                },
            )
        })();
        if result.is_err() {
            let _ = std::fs::remove_dir_all(&staging);
        }
        result
    }
}

/// The uses an import asked for, in section order; at least one.
fn uses(kinds: &[ModelKind]) -> Result<Vec<ModelKind>> {
    let kinds = crate::kind::normalize(kinds);
    if kinds.is_empty() {
        return Err(Error::Invalid("Choose what the model is used for.".into()));
    }
    Ok(kinds)
}

/// Runs a job on its own thread; a panic inside it can't take the app down.
fn spawn(f: impl FnOnce() + Send + 'static) {
    let r = std::thread::Builder::new().name("photocraft-models".into()).spawn(move || {
        if std::panic::catch_unwind(std::panic::AssertUnwindSafe(f)).is_err() {
            log::error!("a model download or import stopped unexpectedly");
        }
    });
    if let Err(e) = r {
        log::error!("couldn't start a model download thread: {e}");
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    fn temp(tag: &str) -> PathBuf {
        let d = std::env::temp_dir().join(format!("photocraft-ml-mgr-{tag}-{}-{}", std::process::id(), crate::store::now()));
        let _ = std::fs::remove_dir_all(&d);
        d
    }

    fn wait(m: &Manager, id: JobId) -> Job {
        for _ in 0..500 {
            if let Some(j) = m.jobs().into_iter().find(|j| j.id == id && !j.active()) {
                return j;
            }
            std::thread::sleep(Duration::from_millis(10));
        }
        panic!("job {id} never finished");
    }

    #[test]
    fn importing_files_installs_one_enabled_model() {
        let dir = temp("files");
        let src = dir.join("src");
        std::fs::create_dir_all(&src).unwrap();
        std::fs::write(src.join("model.onnx"), vec![1u8; 3000]).unwrap();
        std::fs::write(src.join("model.onnx.data"), vec![2u8; 5000]).unwrap();
        let m = Manager::open(&dir.join("Models")).unwrap();
        let id = m.import_files(&[src.join("model.onnx"), src.join("model.onnx.data")], &[ModelKind::Reflection, ModelKind::Removal], None).unwrap();
        let job = wait(&m, id);
        assert_eq!(job.state, JobState::Done, "{job:?}");
        assert_eq!(job.done_bytes, 8000);
        let installed = m.installed();
        assert_eq!(installed.len(), 1);
        assert_eq!(installed[0].name, "model");
        assert_eq!(installed[0].size_bytes, 8000);
        assert_eq!(m.available(ModelKind::Removal).len(), 1);
        assert!(m.available(ModelKind::Inpainting).is_empty());
        assert_eq!(m.available(ModelKind::Reflection).len(), 1, "listed in both of its sections");
        m.set_enabled(&installed[0].id, ModelKind::Removal, false).unwrap();
        assert!(m.available(ModelKind::Removal).is_empty());
        assert_eq!(m.available(ModelKind::Reflection).len(), 1, "switching one use off leaves the other");
        m.rename(&installed[0].id, "Renamed").unwrap();
        assert_eq!(m.installed()[0].name, "Renamed");
        m.delete(&installed[0].id).unwrap();
        assert!(m.installed().is_empty());
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn a_pipeline_folder_imports_with_its_folders_and_ctrl_switches_every_use() {
        let dir = temp("pipeline");
        let src = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/tiny_sd");
        let m = Manager::open(&dir.join("Models")).unwrap();
        let id = m.import_folder(&src, &[ModelKind::Inpainting, ModelKind::Other], None).unwrap();
        let job = wait(&m, id);
        assert_eq!(job.state, JobState::Done, "{job:?}");
        let installed = m.installed();
        assert_eq!(installed[0].name, "tiny_sd");
        assert!(installed[0].files.contains(&"unet/model.onnx".to_string()), "{:?}", installed[0].files);
        let files = m.files_of(&installed[0].id).unwrap();
        assert!(files.iter().all(|f| f.is_file()));
        let model = crate::infer::Model { name: "t".into(), files, catalog_id: None };
        assert!(crate::infer::diffusion::is_pipeline(&model));
        // Reopening keeps the folders' files.
        drop(m);
        let m = Manager::open(&dir.join("Models")).unwrap();
        assert_eq!(m.installed()[0].files, installed[0].files);
        m.set_enabled_everywhere(&installed[0].id, false).unwrap();
        assert!(m.available(ModelKind::Inpainting).is_empty() && m.available(ModelKind::Other).is_empty());
        m.set_enabled_everywhere(&installed[0].id, true).unwrap();
        assert_eq!(m.available(ModelKind::Other).len(), 1);
        assert!(m.import_folder(&dir, &[ModelKind::Inpainting], None).is_err(), "not a pipeline");
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn bad_imports_fail_without_leaving_files() {
        let dir = temp("bad");
        let m = Manager::open(&dir).unwrap();
        assert!(m.import_files(&[], &[ModelKind::Other], None).is_err());
        assert!(m.import_files(&[dir.join("missing.onnx")], &[ModelKind::Other], None).is_err());
        assert!(m.import_files(std::slice::from_ref(&dir), &[ModelKind::Other], None).is_err());
        assert!(m.import_link("not a link", &[ModelKind::Other], None).is_err());
        assert!(m.import_link("Acly/MobileSAM", &[], None).is_err(), "an import needs a use");
        assert!(m.download("no-such-model").is_err());
        assert!(!dir.join(crate::store::STAGING).exists() || std::fs::read_dir(dir.join(crate::store::STAGING)).unwrap().next().is_none());
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn eta_comes_from_the_average_speed() {
        let j = Job {
            id: 1,
            kind: JobKind::Import,
            label: String::new(),
            model_kinds: vec![ModelKind::Other],
            state: JobState::Running,
            done_bytes: 50,
            total_bytes: Some(100),
            started: Instant::now() - Duration::from_secs(10),
            finished: None,
            cancel: Arc::default(),
        };
        let eta = j.eta().unwrap();
        assert!((9.0..11.5).contains(&eta), "{eta}");
        assert_eq!(j.fraction(), Some(0.5));
        let done = Job { state: JobState::Done, ..j };
        assert_eq!(done.eta(), None);
    }
}
