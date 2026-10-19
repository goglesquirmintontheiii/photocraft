//! Downloads: HTTPS with resume, redirect handling, cancel, progress and SHA-256 checks.
//! Native only; the browser build has no model manager.

use crate::{Error, Result};
use sha2::{Digest, Sha256};
use std::io::{Read, Write};
use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};

/// Largest API answer read into memory (the Hugging Face file list of a big repository).
const MAX_JSON: u64 = 32 * 1024 * 1024;
const MAX_REDIRECTS: usize = 10;
const CHUNK: usize = 1 << 20;

pub(crate) fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

/// Turns a `Location` header into an absolute URL relative to `base`.
pub(crate) fn absolutize(location: &str, base: &str) -> String {
    let loc = location.trim();
    if loc.starts_with("https://") || loc.starts_with("http://") {
        return loc.to_string();
    }
    let (scheme, rest) = base.split_once("://").unwrap_or(("https", base));
    if let Some(l) = loc.strip_prefix("//") {
        return format!("{scheme}://{l}");
    }
    let host = rest.split(['/', '?', '#']).next().unwrap_or(rest);
    if loc.starts_with('/') {
        return format!("{scheme}://{host}{loc}");
    }
    let path = rest.split(['?', '#']).next().unwrap_or(rest);
    let dir = path.rsplit_once('/').map_or(host, |(d, _)| d);
    format!("{scheme}://{dir}/{loc}")
}

pub(crate) fn status_error(status: u16, url: &str) -> Error {
    let host = url.split_once("://").map_or(url, |(_, r)| r).split('/').next().unwrap_or("The server");
    match status {
        401 | 403 => {
            let site = match crate::accounts::site_of(url) {
                Some("huggingface") => Some("Hugging Face"),
                Some(_) => Some("Civitai"),
                None => None,
            };
            Error::Network(match site {
                Some(name) if crate::accounts::has_token_for(url) => format!(
                    "{host} turned down your {name} token for this model. It may be gated: accept its licence on its page while signed in, or check the token under Accounts at the bottom of AI Models."
                ),
                Some(name) => format!(
                    "{host} asked to sign in before downloading this model. Add your {name} token under Accounts at the bottom of AI Models, or download it in your browser and use Import › From a File."
                ),
                None => format!("{host} asked to sign in before downloading this model. Download it in your browser, then use Import › From a File."),
            })
        }
        404 | 410 => Error::NotFound(format!("Nothing was found at that link on {host}.")),
        429 => Error::Network(format!("{host} is limiting downloads right now. Try again in a few minutes.")),
        s => Error::Network(format!("{host} answered with error {s}.")),
    }
}

/// Hashes a file that is already partly downloaded, so a resumed download can still be checked.
fn hash_existing(path: &Path, hasher: &mut Sha256, cancel: &AtomicBool) -> Result<u64> {
    let mut f = std::fs::File::open(path).map_err(|e| Error::io("Couldn't read the partial download", e))?;
    let mut buf = vec![0u8; CHUNK];
    let mut n = 0u64;
    loop {
        if cancel.load(Ordering::Relaxed) {
            return Err(Error::Cancelled);
        }
        let got = f.read(&mut buf).map_err(|e| Error::io("Couldn't read the partial download", e))?;
        let Some(chunk) = buf.get(..got) else { break };
        if got == 0 {
            break;
        }
        hasher.update(chunk);
        n = n.saturating_add(got as u64);
    }
    Ok(n)
}

/// Copies `from` (a reader of `total` bytes, if known) to `to`, appending, hashing and reporting
/// progress as it goes. Returns the bytes written.
pub(crate) fn pump(
    from: &mut dyn Read,
    to: &mut std::fs::File,
    hasher: &mut Sha256,
    start: u64,
    total: Option<u64>,
    cancel: &AtomicBool,
    progress: &mut dyn FnMut(u64, Option<u64>),
) -> Result<u64> {
    let mut buf = vec![0u8; CHUNK];
    let mut done = start;
    loop {
        if cancel.load(Ordering::Relaxed) {
            return Err(Error::Cancelled);
        }
        let got = match from.read(&mut buf) {
            Ok(n) => n,
            Err(e) if e.kind() == std::io::ErrorKind::Interrupted => continue,
            Err(e) => return Err(Error::Network(format!("The download stopped: {e}"))),
        };
        if got == 0 {
            break;
        }
        let chunk = buf.get(..got).ok_or_else(|| Error::Io("read past the buffer".into()))?;
        to.write_all(chunk).map_err(|e| Error::io("Couldn't write the model file (is the disk full?)", e))?;
        hasher.update(chunk);
        done = done.saturating_add(got as u64);
        progress(done, total);
    }
    to.flush().map_err(|e| Error::io("Couldn't write the model file", e))?;
    Ok(done)
}

/// Checks a finished file against the size and SHA-256 the source published.
pub(crate) fn verify(done: u64, digest: &str, size: Option<u64>, sha256: Option<&str>) -> Result<()> {
    if let Some(s) = size
        && s != done
    {
        return Err(Error::Invalid(format!("The download is incomplete ({done} of {s} bytes). Try again.")));
    }
    if let Some(want) = sha256
        && !want.eq_ignore_ascii_case(digest)
    {
        return Err(Error::Invalid("The downloaded file is damaged (its checksum doesn't match). Try again.".into()));
    }
    Ok(())
}

#[cfg(not(target_arch = "wasm32"))]
mod native {
    use super::*;
    use std::sync::OnceLock;
    use std::time::Duration;

    fn agent() -> &'static ureq::Agent {
        static AGENT: OnceLock<ureq::Agent> = OnceLock::new();
        AGENT.get_or_init(|| {
            ureq::Agent::config_builder()
                .http_status_as_error(false)
                .max_redirects(0)
                .timeout_connect(Some(Duration::from_secs(30)))
                .timeout_recv_response(Some(Duration::from_secs(60)))
                .user_agent(concat!("PhotoCraft/", env!("CARGO_PKG_VERSION")))
                .build()
                .into()
        })
    }

    fn net(e: ureq::Error) -> Error {
        Error::Network(format!("Couldn't reach the server: {e}"))
    }

    /// GETs `url`, following redirects; returns the final response and any SHA-256 / size a
    /// Hugging Face redirect announced (`X-Linked-Etag`, `X-Linked-Size`).
    /// The final response of a GET, its URL after redirects, and the SHA-256 and size a Hugging
    /// Face redirect announced.
    type Fetched = (ureq::http::Response<ureq::Body>, String, Option<String>, Option<u64>);

    fn get(url: &str, range_from: u64) -> Result<Fetched> {
        let mut url = url.to_string();
        let (mut sha, mut size) = (None, None);
        for _ in 0..=MAX_REDIRECTS {
            let mut req = agent().get(&url);
            if range_from > 0 {
                req = req.header("Range", format!("bytes={range_from}-"));
            }
            // Per hop: a redirect to another server (a CDN) never gets the token.
            if let Some(auth) = crate::accounts::authorization(&url) {
                req = req.header("Authorization", auth);
            }
            let resp = req.call().map_err(net)?;
            let h = resp.headers();
            let text = |k: &str| h.get(k).and_then(|v| v.to_str().ok()).map(|s| s.trim().trim_matches('"').to_string());
            if let Some(e) = text("x-linked-etag").filter(|e| crate::source::is_sha256(e)) {
                sha = Some(e.to_ascii_lowercase());
            }
            if let Some(s) = text("x-linked-size").and_then(|s| s.parse().ok()) {
                size = Some(s);
            }
            if resp.status().is_redirection() {
                let loc = text("location").ok_or_else(|| Error::Network("The server redirected without saying where.".into()))?;
                url = absolutize(&loc, &url);
                continue;
            }
            return Ok((resp, url, sha, size));
        }
        Err(Error::Network("The link redirected too many times.".into()))
    }

    pub(crate) fn get_json(url: &str) -> Result<serde_json::Value> {
        let (resp, url, ..) = get(url, 0)?;
        let status = resp.status().as_u16();
        if status != 200 {
            return Err(status_error(status, &url));
        }
        let text = resp.into_body().with_config().limit(MAX_JSON).read_to_string().map_err(net)?;
        serde_json::from_str(&text).map_err(|e| Error::Network(format!("The server sent something unexpected: {e}")))
    }

    /// Downloads `url` to `dest`, resuming a partial `dest` when the server allows it.
    pub(crate) fn download(
        url: &str,
        dest: &Path,
        size: Option<u64>,
        sha256: Option<&str>,
        cancel: &AtomicBool,
        progress: &mut dyn FnMut(u64, Option<u64>),
    ) -> Result<()> {
        let mut hasher = Sha256::new();
        let have = std::fs::metadata(dest).map(|m| m.len()).unwrap_or(0);
        let have = if size.is_some_and(|s| have > s) { 0 } else { have };
        let (resp, final_url, linked_sha, linked_size) = get(url, have)?;
        let status = resp.status().as_u16();
        let size = size.or(linked_size);
        let sha = sha256.map(str::to_ascii_lowercase).or(linked_sha);
        let content_len = resp.headers().get("content-length").and_then(|v| v.to_str().ok()).and_then(|s| s.trim().parse::<u64>().ok());
        let (start, mut file) = match status {
            206 if have > 0 => {
                let start = hash_existing(dest, &mut hasher, cancel)?;
                let f = std::fs::OpenOptions::new().append(true).open(dest).map_err(|e| Error::io("Couldn't write the model file", e))?;
                (start, f)
            }
            200 | 206 => {
                hasher = Sha256::new();
                (0, std::fs::File::create(dest).map_err(|e| Error::io("Couldn't create the model file", e))?)
            }
            416 if have > 0 && size == Some(have) => {
                hash_existing(dest, &mut hasher, cancel)?;
                return verify(have, &hex(&hasher.finalize()), size, sha.as_deref());
            }
            416 => {
                let _ = std::fs::remove_file(dest);
                return download(url, dest, size, sha256, cancel, progress);
            }
            s => return Err(status_error(s, &final_url)),
        };
        let total = size.or_else(|| content_len.map(|l| l.saturating_add(start)));
        progress(start, total);
        let mut reader = resp.into_body().into_reader();
        let done = pump(&mut reader, &mut file, &mut hasher, start, total, cancel, progress)?;
        drop(file);
        let digest = hex(&hasher.finalize());
        if let Err(e) = verify(done, &digest, size, sha.as_deref()) {
            let _ = std::fs::remove_file(dest);
            return Err(e);
        }
        Ok(())
    }
}

#[cfg(not(target_arch = "wasm32"))]
pub(crate) use native::{download, get_json};

#[cfg(target_arch = "wasm32")]
pub(crate) fn get_json(_url: &str) -> Result<serde_json::Value> {
    Err(Error::Unsupported("Model downloads need the desktop app.".into()))
}

#[cfg(target_arch = "wasm32")]
pub(crate) fn download(_: &str, _: &Path, _: Option<u64>, _: Option<&str>, _: &AtomicBool, _: &mut dyn FnMut(u64, Option<u64>)) -> Result<()> {
    Err(Error::Unsupported("Model downloads need the desktop app.".into()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn redirects_resolve_against_the_page() {
        assert_eq!(absolutize("https://cdn.x/f", "https://hf.co/a/b"), "https://cdn.x/f");
        assert_eq!(absolutize("//cdn.x/f", "https://hf.co/a/b"), "https://cdn.x/f");
        assert_eq!(absolutize("/api/f?x=1", "https://hf.co/a/b?q"), "https://hf.co/api/f?x=1");
        assert_eq!(absolutize("c", "https://hf.co/a/b"), "https://hf.co/a/c");
        assert_eq!(absolutize("c", "https://hf.co"), "https://hf.co/c");
    }

    #[test]
    fn verify_checks_size_and_checksum() {
        let empty = hex(&Sha256::digest(b""));
        assert_eq!(empty, "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855");
        assert!(verify(0, &empty, Some(0), Some(&empty.to_uppercase())).is_ok());
        assert!(verify(0, &empty, Some(1), None).is_err());
        assert!(verify(0, &empty, None, Some(&"0".repeat(64))).is_err());
        assert!(verify(5, &empty, None, None).is_ok());
    }

    #[test]
    fn pump_stops_when_cancelled() {
        let dir = std::env::temp_dir().join(format!("photocraft-ml-pump-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let mut f = std::fs::File::create(dir.join("out")).unwrap();
        let cancel = AtomicBool::new(true);
        let mut src: &[u8] = b"hello";
        let r = pump(&mut src, &mut f, &mut Sha256::new(), 0, Some(5), &cancel, &mut |_, _| {});
        assert!(matches!(r, Err(Error::Cancelled)));
        let cancel = AtomicBool::new(false);
        let mut src: &[u8] = b"hello";
        let mut seen = 0;
        assert_eq!(pump(&mut src, &mut f, &mut Sha256::new(), 0, Some(5), &cancel, &mut |d, _| seen = d).unwrap(), 5);
        assert_eq!(seen, 5);
        let _ = std::fs::remove_dir_all(dir);
    }

    /// A one-file HTTP server: `/r` redirects to `/f` announcing the file's SHA-256 the way
    /// Hugging Face does; `/f` serves the body and honours `Range`.
    #[cfg(not(target_arch = "wasm32"))]
    fn serve(body: Vec<u8>, announced_sha: String) -> String {
        use std::io::{BufRead, BufReader};
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        std::thread::spawn(move || {
            for stream in listener.incoming().flatten() {
                let mut reader = BufReader::new(stream.try_clone().unwrap());
                let mut first = String::new();
                reader.read_line(&mut first).unwrap();
                let mut range = 0usize;
                loop {
                    let mut l = String::new();
                    if reader.read_line(&mut l).unwrap() == 0 || l.trim().is_empty() {
                        break;
                    }
                    if let Some(v) = l.to_ascii_lowercase().strip_prefix("range: bytes=") {
                        range = v.trim().trim_end_matches('-').parse().unwrap_or(0);
                    }
                }
                let mut s = stream;
                if first.starts_with("GET /r ") {
                    write!(s, "HTTP/1.1 302 Found\r\nLocation: /f\r\nX-Linked-Etag: \"{announced_sha}\"\r\nContent-Length: 0\r\nConnection: close\r\n\r\n")
                        .unwrap();
                } else if range > 0 {
                    let part = &body[range..];
                    write!(s, "HTTP/1.1 206 Partial Content\r\nContent-Length: {}\r\nConnection: close\r\n\r\n", part.len()).unwrap();
                    s.write_all(part).unwrap();
                } else {
                    write!(s, "HTTP/1.1 200 OK\r\nContent-Length: {}\r\nConnection: close\r\n\r\n", body.len()).unwrap();
                    s.write_all(&body).unwrap();
                }
            }
        });
        format!("http://{addr}")
    }

    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    fn downloads_follow_redirects_resume_and_check_the_announced_checksum() {
        let body: Vec<u8> = (0..3_000_000u32).map(|i| (i % 251) as u8).collect();
        let sha = hex(&Sha256::digest(&body));
        let base = serve(body.clone(), sha.clone());
        let dir = std::env::temp_dir().join(format!("photocraft-ml-dl-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let dest = dir.join("m.onnx");
        let cancel = AtomicBool::new(false);
        let mut last = (0, None);
        download(&format!("{base}/r"), &dest, None, None, &cancel, &mut |d, t| last = (d, t)).unwrap();
        assert_eq!(std::fs::read(&dest).unwrap(), body);
        assert_eq!(last, (3_000_000, Some(3_000_000)));

        // Resume from half a file.
        std::fs::write(&dest, &body[..1_000_000]).unwrap();
        download(&format!("{base}/f"), &dest, Some(3_000_000), Some(&sha), &cancel, &mut |_, _| {}).unwrap();
        assert_eq!(std::fs::read(&dest).unwrap(), body);

        // A wrong announced checksum fails and removes the file.
        let bad = serve(body, "0".repeat(64));
        std::fs::remove_file(&dest).unwrap();
        assert!(matches!(download(&format!("{bad}/r"), &dest, None, None, &cancel, &mut |_, _| {}), Err(Error::Invalid(_))));
        assert!(!dest.exists());
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn errors_name_the_host_and_what_to_do() {
        let e = status_error(401, "https://huggingface.co/x/y/resolve/main/m.safetensors").to_string();
        assert!(e.starts_with("huggingface.co asked to sign in"));
        assert!(matches!(status_error(404, "https://a/b"), Error::NotFound(_)));
    }
}
