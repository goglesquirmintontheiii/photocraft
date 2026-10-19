//! Local AI models for PhotoCraft's generative and AI-assisted tools.
//!
//! This crate is the model manager behind Edit › Preferences › AI:
//!
//! - [`catalog`]: the recommended models, with where to download each file and its licence.
//! - [`store`]: the models installed on this computer, kept in a `Models` folder beside the
//!   other PhotoCraft files, with a `models.json` manifest (enabled, source, size on disk).
//! - [`source`]: what an import link points at (a Hugging Face file or repository, a Civitai
//!   model, any other URL) and how to turn it into download URLs.
//! - [`accounts`]: Hugging Face and Civitai tokens for gated downloads.
//! - [`manager`]: the thread-safe front the UI drives. Downloads and imports run on worker
//!   threads and report progress, bytes, elapsed time and an ETA.
//!
//! - [`infer`]: running them (selection masks, subject mattes, fills), with ONNX Runtime when the
//!   crate is built with `onnx` or `onnx-dynamic`. [`available`] is what the tools' model
//!   dropdowns list: installed models that are enabled for that use, best first.
//!
//! Never-crash: every operation returns [`Error`]; file contents, manifests, HTTP responses and
//! links are treated as hostile input.

#![forbid(unsafe_code)]
#![deny(clippy::unwrap_used, clippy::expect_used, clippy::panic, clippy::unimplemented, clippy::todo, clippy::unreachable)]

pub mod accounts;
pub mod convert;
pub mod catalog;
mod http;
pub mod infer;
pub mod kind;
pub mod manager;
pub mod source;
pub mod store;

pub use accounts::Tokens;
pub use catalog::{CatalogFile, CatalogModel, RECOMMENDED};
pub use kind::{ModelKind, detect as detect_kinds};
pub use manager::{Job, JobId, JobKind, JobState, Manager};
pub use store::{InstalledModel, ModelSource};

/// Errors from the model manager. Messages are written for the person using the app.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("{0}")]
    Io(String),
    #[error("{0}")]
    Network(String),
    #[error("{0}")]
    Invalid(String),
    #[error("{0}")]
    NotFound(String),
    #[error("Cancelled")]
    Cancelled,
    #[error("{0}")]
    Unsupported(String),
}

impl Error {
    pub(crate) fn io(context: &str, e: std::io::Error) -> Self {
        Error::Io(format!("{context}: {e}"))
    }
}

pub type Result<T> = std::result::Result<T, Error>;

/// Installed, enabled models of `kind`, the most accurate first (catalog order, then imports by
/// name): what a tool's model dropdown lists.
pub fn available(installed: &[InstalledModel], kind: ModelKind) -> Vec<InstalledModel> {
    // Recommended models made for this use first (a reflection model before the object removers
    // that also fill reflections), then the others, each in catalog order; imported ones last.
    let rank = |m: &InstalledModel| match &m.catalog_id {
        Some(id) => match RECOMMENDED.iter().position(|c| c.id == id) {
            Some(i) => (u8::from(RECOMMENDED[i].kind() != kind), i),
            None => (2, 0),
        },
        None => (3, 0),
    };
    let mut out: Vec<InstalledModel> = installed.iter().filter(|m| m.enabled_for(kind)).cloned().collect();
    out.sort_by(|a, b| rank(a).cmp(&rank(b)).then_with(|| a.name.to_lowercase().cmp(&b.name.to_lowercase())));
    out
}

/// "1.7 GB", "312 MB", "48 KB": sizes as the AI tab shows them (decimal units, as file
/// managers on Linux and macOS do).
pub fn format_size(bytes: u64) -> String {
    const UNITS: [(&str, f64); 4] = [("TB", 1e12), ("GB", 1e9), ("MB", 1e6), ("KB", 1e3)];
    let b = bytes as f64;
    for (unit, scale) in UNITS {
        if b >= scale {
            let v = b / scale;
            return if v >= 100.0 { format!("{v:.0} {unit}") } else { format!("{v:.1} {unit}") };
        }
    }
    format!("{bytes} B")
}

/// "4 s", "2 min 05 s", "1 h 12 min": durations for progress and ETA text.
pub fn format_duration(secs: f64) -> String {
    if !secs.is_finite() || secs < 0.0 {
        return "–".into();
    }
    let s = secs.round() as u64;
    match s {
        0..=59 => format!("{s} s"),
        60..=3599 => format!("{} min {:02} s", s / 60, s % 60),
        _ => format!("{} h {:02} min", s / 3600, (s % 3600) / 60),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sizes_read_like_a_file_manager() {
        assert_eq!(format_size(0), "0 B");
        assert_eq!(format_size(999), "999 B");
        assert_eq!(format_size(48_000), "48.0 KB");
        assert_eq!(format_size(312_400_000), "312 MB");
        assert_eq!(format_size(1_700_000_000), "1.7 GB");
        assert_eq!(format_size(u64::MAX), "18446744 TB");
    }

    #[test]
    fn durations_are_short_and_survive_nonsense() {
        assert_eq!(format_duration(4.2), "4 s");
        assert_eq!(format_duration(125.0), "2 min 05 s");
        assert_eq!(format_duration(4320.0), "1 h 12 min");
        assert_eq!(format_duration(f64::NAN), "–");
        assert_eq!(format_duration(-3.0), "–");
        assert_eq!(format_duration(f64::INFINITY), "–");
    }
}
