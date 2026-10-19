//! Local AI models for commands: which model a command runs, and its progress and errors.
//!
//! The session holds the model manager the app opened ([`Session::set_models`]); a command names
//! a model by its id (`"model"`), or gets the best enabled model for its use. The models
//! themselves run in [`photocraft_ml::infer`], on the job's worker thread.

use crate::{EngineError, Result, Session};
use photocraft_ml::infer::Model;
use photocraft_ml::{Manager, ModelKind};
use serde_json::{Value, json};

impl Session {
    /// Gives the session the local model manager (the desktop app's `Models` folder).
    pub fn set_models(&mut self, models: Option<Manager>) {
        self.models = models;
    }

    pub fn models(&self) -> Option<&Manager> {
        self.models.as_ref()
    }
}

/// Why a command of `kind` can't run, or `Ok` when it has a model to run.
pub(crate) fn enabled(s: &Session, kind: ModelKind) -> std::result::Result<(), String> {
    s.active().ok_or_else(|| "no document open".to_string())?;
    let m = s.models.as_ref().ok_or_else(|| "AI models need the desktop app".to_string())?;
    if m.available(kind).is_empty() {
        return Err(format!("no {} model is installed and enabled (Edit › Preferences › AI Models)", kind.title()));
    }
    Ok(())
}

/// A model that may still have to be prepared (a Stable Diffusion checkpoint made into a
/// pipeline) before it runs, on the job's thread.
pub(crate) struct Prepare(Option<(Manager, String)>);

impl Prepare {
    /// Prepares `model` if it needs it (progress as "Preparing the model…").
    pub(crate) fn run(self, model: &mut Model, report: &mut dyn FnMut(f32, &str) -> bool) -> photocraft_ml::Result<()> {
        let Some((m, id)) = self.0 else { return Ok(()) };
        if photocraft_ml::convert::checkpoint_to_convert(&model.files).is_none() {
            return Ok(());
        }
        model.files = m.prepare(&id, &mut |f| report(f, "Preparing the model for its first use…"))?;
        Ok(())
    }
}

/// [`resolve`], with what prepares the model if it is a checkpoint.
pub(crate) fn resolve_prepared(s: &Session, cmd: &str, kind: ModelKind, p: &Value) -> Result<(Model, Prepare)> {
    let (model, id) = resolve_id(s, cmd, kind, p)?;
    Ok((model, Prepare(s.models.clone().map(|m| (m, id)))))
}

/// The model a command runs: `"model"` (an installed id) if given, else the best enabled model of
/// `kind`.
pub(crate) fn resolve(s: &Session, cmd: &str, kind: ModelKind, p: &Value) -> Result<Model> {
    resolve_id(s, cmd, kind, p).map(|(m, _)| m)
}

fn resolve_id(s: &Session, cmd: &str, kind: ModelKind, p: &Value) -> Result<(Model, String)> {
    let m = s.models.as_ref().ok_or_else(|| EngineError::Other("AI models need the desktop app.".into()))?;
    let installed = m.installed();
    let chosen = match p.get("model") {
        None | Some(Value::Null) => {
            let mut a = photocraft_ml::available(&installed, kind);
            if a.is_empty() {
                return Err(EngineError::Other(format!("No {} model is installed and enabled. Download one in Edit › Preferences › AI Models.", kind.title())));
            }
            a.remove(0)
        }
        Some(Value::String(id)) => {
            let model = installed.into_iter().find(|i| i.id == *id).ok_or_else(|| bad(cmd, &format!("no model {id:?} is installed")))?;
            if !model.enabled_for(kind) {
                return Err(bad(cmd, &format!("{} isn't enabled for {}", model.name, kind.title())));
            }
            model
        }
        Some(_) => return Err(bad(cmd, "`model` is an installed model's id")),
    };
    let files = m.files_of(&chosen.id).ok_or_else(|| EngineError::Other(format!("{}'s files are gone from the models folder.", chosen.name)))?;
    Ok((Model { name: chosen.name, files, catalog_id: chosen.catalog_id }, chosen.id))
}

/// The models a command can run, best first, for its `params` and for agents (`ai.models`).
pub(crate) fn list(s: &Session, kind: ModelKind) -> Value {
    let models = s.models.as_ref().map(|m| m.available(kind)).unwrap_or_default();
    json!(models.iter().map(|m| json!({"id": m.id, "name": m.name, "recommended": m.catalog_id.is_some()})).collect::<Vec<_>>())
}

fn bad(cmd: &str, msg: &str) -> EngineError {
    EngineError::BadParams { cmd: cmd.into(), msg: msg.into() }
}

/// A model error as the command reports it.
pub(crate) fn error(e: photocraft_ml::Error) -> EngineError {
    match e {
        photocraft_ml::Error::Cancelled => EngineError::Cancelled,
        e => EngineError::Other(e.to_string()),
    }
}

/// Progress from the model into the job, between `lo` and `hi`, with the job's label in front
/// and where it runs ("on the GPU") once known. Returns false when the job was cancelled.
pub(crate) fn reporter<'a>(ctx: &'a crate::jobs::JobCtx, label: &'a str, lo: f32, hi: f32) -> impl FnMut(f32, &str) -> bool + 'a {
    move |f, msg| {
        let place = match photocraft_ml::infer::last_backend().as_deref() {
            Some("CPU") => " (CPU)",
            Some("NPU") => " (NPU)",
            Some(_) => " (GPU)",
            None => "",
        };
        let msg = if msg.is_empty() { format!("{label}{place}") } else { format!("{label}{place}: {msg}") };
        ctx.progress(lo + (hi - lo) * f.clamp(0.0, 1.0), &msg);
        !ctx.cancelled()
    }
}

/// Lists the installed, enabled models for each use (what the tools' model menus offer).
fn models_cmd(s: &mut Session, p: &Value) -> Result<Value> {
    let kinds: Vec<ModelKind> = match p.get("kind").and_then(Value::as_str) {
        Some(k) => vec![ModelKind::from_id(k).ok_or_else(|| bad("ai.models", "`kind` is segmentation|matting|removal|reflection|inpainting|enhance|other"))?],
        None => ModelKind::ALL.to_vec(),
    };
    let mut out = serde_json::Map::new();
    for k in kinds {
        out.insert(k.id().into(), list(s, k));
    }
    Ok(json!({ "runtime": photocraft_ml::infer::runtime_available(), "models": out }))
}

pub fn specs() -> Vec<crate::commands::CommandSpec> {
    vec![crate::commands::CommandSpec {
        id: "ai.models",
        label: "AI Models",
        menu: &[],
        shortcut: None,
        params: r#"{"kind":"segmentation|matting|removal|reflection|inpainting|enhance|other"?} → {"runtime":bool (this build can run models),"models":{"<kind>":[{"id","name","recommended"}],…}} (installed and enabled, best first: what "model" in ai.* and the generative brushes takes)"#,
        enabled: |_| Ok(()),
        run: models_cmd,
        journal: false,
    }]
}
