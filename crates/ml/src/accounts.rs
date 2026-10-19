//! Sign-in tokens for Hugging Face and Civitai, for gated or sign-in-only models.
//!
//! Kept in the models folder as `accounts.json`, readable only by the user on Unix, and never in
//! the preferences (which agents and the control channel can read). A token is sent only to its
//! own site, over HTTPS, and never to a server a download redirects to.

use crate::{Error, Result};
use serde::{Deserialize, Serialize};
use std::path::Path;
use std::sync::{PoisonError, RwLock};

pub const FILE: &str = "accounts.json";

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default, rename_all = "camelCase")]
pub struct Tokens {
    pub hugging_face: String,
    pub civitai: String,
}

/// The tokens downloads use; set when the manager opens and when they're saved.
static CURRENT: RwLock<Tokens> = RwLock::new(Tokens { hugging_face: String::new(), civitai: String::new() });

/// A token as pasted: surrounding space and quotes removed, and only characters a token can have
/// (so a stray newline can't end up in an HTTP header).
pub fn clean(token: &str) -> String {
    token.trim().trim_matches(['"', '\'']).chars().filter(|c| c.is_ascii_graphic()).take(512).collect()
}

/// Reads `accounts.json` in `dir`; missing or unreadable means no tokens.
pub fn load(dir: &Path) -> Tokens {
    let t = std::fs::read_to_string(dir.join(FILE)).ok().and_then(|s| serde_json::from_str::<Tokens>(&s).ok()).unwrap_or_default();
    let t = Tokens { hugging_face: clean(&t.hugging_face), civitai: clean(&t.civitai) };
    set_current(&t);
    t
}

/// Saves the tokens (an empty pair removes the file) and starts using them.
pub fn save(dir: &Path, tokens: &Tokens) -> Result<()> {
    let tokens = Tokens { hugging_face: clean(&tokens.hugging_face), civitai: clean(&tokens.civitai) };
    let path = dir.join(FILE);
    if tokens == Tokens::default() {
        match std::fs::remove_file(&path) {
            Ok(()) => {}
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {}
            Err(e) => return Err(Error::io("Couldn't remove the saved sign-in tokens", e)),
        }
    } else {
        let text = serde_json::to_string_pretty(&tokens).map_err(|e| Error::Invalid(e.to_string()))?;
        let tmp = dir.join(format!("{FILE}.tmp"));
        write_private(&tmp, text.as_bytes()).map_err(|e| Error::io("Couldn't save the sign-in tokens", e))?;
        std::fs::rename(&tmp, &path).map_err(|e| Error::io("Couldn't save the sign-in tokens", e))?;
    }
    set_current(&tokens);
    Ok(())
}

fn set_current(t: &Tokens) {
    *CURRENT.write().unwrap_or_else(PoisonError::into_inner) = t.clone();
}

#[cfg(unix)]
fn write_private(path: &Path, bytes: &[u8]) -> std::io::Result<()> {
    use std::io::Write;
    use std::os::unix::fs::OpenOptionsExt;
    let mut f = std::fs::OpenOptions::new().write(true).create(true).truncate(true).mode(0o600).open(path)?;
    f.write_all(bytes)
}

#[cfg(not(unix))]
fn write_private(path: &Path, bytes: &[u8]) -> std::io::Result<()> {
    std::fs::write(path, bytes)
}

/// The site a URL belongs to, for choosing a token: `Some("huggingface")` or `Some("civitai")`
/// for HTTPS links to those sites (exact host or a subdomain), `None` otherwise.
pub fn site_of(url: &str) -> Option<&'static str> {
    let rest = url.strip_prefix("https://")?;
    let host = rest.split(['/', '?', '#']).next()?.rsplit('@').next()?;
    let host = host.split(':').next()?.to_ascii_lowercase();
    let is = |d: &str| host == d || host.ends_with(&format!(".{d}"));
    if is("huggingface.co") || is("hf.co") {
        Some("huggingface")
    } else if is("civitai.com") {
        Some("civitai")
    } else {
        None
    }
}

/// The `Authorization` header for a request to `url`, when a token for its site is saved.
pub(crate) fn authorization(url: &str) -> Option<String> {
    let site = site_of(url)?;
    let t = CURRENT.read().unwrap_or_else(PoisonError::into_inner);
    let token = match site {
        "huggingface" => &t.hugging_face,
        _ => &t.civitai,
    };
    (!token.is_empty()).then(|| format!("Bearer {token}"))
}

/// Whether a token for `url`'s site is saved (to word sign-in errors).
pub(crate) fn has_token_for(url: &str) -> bool {
    authorization(url).is_some()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tokens_go_only_to_their_own_site_over_https() {
        assert_eq!(site_of("https://huggingface.co/x/y"), Some("huggingface"));
        assert_eq!(site_of("https://cdn-lfs.huggingface.co/x"), Some("huggingface"));
        assert_eq!(site_of("https://civitai.com/api/download/models/1"), Some("civitai"));
        assert_eq!(site_of("http://huggingface.co/x"), None, "never over plain HTTP");
        assert_eq!(site_of("https://huggingface.co.evil.com/x"), None);
        assert_eq!(site_of("https://evilhuggingface.co/x"), None);
        assert_eq!(site_of("https://huggingface.co@evil.com/x"), None);
        assert_eq!(site_of("https://b2.civitai.cdn.example/x"), None);
    }

    #[test]
    fn pasted_tokens_are_cleaned() {
        assert_eq!(clean("  \"hf_abc\"\n"), "hf_abc");
        assert_eq!(clean("hf_a\r\nX-Evil: 1"), "hf_aX-Evil:1");
    }

    #[test]
    fn tokens_save_privately_and_an_empty_pair_removes_the_file() {
        let dir = std::env::temp_dir().join(format!("photocraft-ml-accounts-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        save(&dir, &Tokens { hugging_face: " hf_1 ".into(), civitai: String::new() }).unwrap();
        assert_eq!(load(&dir).hugging_face, "hf_1");
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            assert_eq!(std::fs::metadata(dir.join(FILE)).unwrap().permissions().mode() & 0o777, 0o600);
        }
        save(&dir, &Tokens::default()).unwrap();
        assert!(!dir.join(FILE).exists());
        assert_eq!(load(&dir), Tokens::default());
        let _ = std::fs::remove_dir_all(dir);
    }
}
