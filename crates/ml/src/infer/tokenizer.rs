//! The CLIP tokenizer Stable Diffusion's text encoders read: byte-level BPE over lowercased,
//! whitespace-cleaned text, `<|startoftext|>` … `<|endoftext|>`, padded to 77 tokens. It reads a
//! pipeline's `tokenizer/vocab.json` and `merges.txt` (and the pad token from
//! `tokenizer_config.json` or `special_tokens_map.json`: `<|endoftext|>` for SD 1.x, `!` for SD 2).

use crate::{Error, Result};
use serde_json::Value;
use std::collections::HashMap;
use std::path::Path;

/// Tokens a prompt becomes, start and end tokens included.
pub const LENGTH: usize = 77;

pub struct ClipTokenizer {
    vocab: HashMap<String, i64>,
    ranks: HashMap<(String, String), usize>,
    bytes: [char; 256],
    bos: i64,
    eos: i64,
    pad: i64,
}

/// GPT-2's reversible byte → printable character table.
fn byte_chars() -> [char; 256] {
    let mut table = ['\0'; 256];
    let mut printable: Vec<u32> = (u32::from(b'!')..=u32::from(b'~')).chain(0xA1..=0xAC).chain(0xAE..=0xFF).collect();
    printable.sort_unstable();
    let mut extra = 0u32;
    for b in 0u32..256 {
        let c = if printable.binary_search(&b).is_ok() {
            b
        } else {
            extra += 1;
            255 + extra
        };
        if let Some(slot) = table.get_mut(b as usize) {
            *slot = char::from_u32(c).unwrap_or('?');
        }
    }
    table
}

fn token_of(v: &Value) -> Option<String> {
    match v {
        Value::String(s) => Some(s.clone()),
        Value::Object(o) => o.get("content").and_then(Value::as_str).map(str::to_string),
        _ => None,
    }
}

impl ClipTokenizer {
    pub fn load(vocab: &Path, merges: &Path, config: &[&Path]) -> Result<ClipTokenizer> {
        let read = |p: &Path| std::fs::read_to_string(p).map_err(|e| Error::io(&format!("Couldn't read {}", p.display()), e));
        let vocab: HashMap<String, i64> =
            serde_json::from_str(&read(vocab)?).map_err(|e| Error::Invalid(format!("The model's tokenizer vocabulary is damaged ({e}).")))?;
        let merges = read(merges)?;
        let mut pad = None;
        for c in config {
            if pad.is_some() {
                break;
            }
            if let Ok(text) = std::fs::read_to_string(c)
                && let Ok(v) = serde_json::from_str::<Value>(&text)
            {
                pad = v.get("pad_token").and_then(token_of);
            }
        }
        ClipTokenizer::new(vocab, &merges, pad.as_deref())
    }

    pub fn new(vocab: HashMap<String, i64>, merges: &str, pad_token: Option<&str>) -> Result<ClipTokenizer> {
        let id = |t: &str| vocab.get(t).copied();
        let bos = id("<|startoftext|>").ok_or_else(|| Error::Invalid("The model's tokenizer has no start token.".into()))?;
        let eos = id("<|endoftext|>").ok_or_else(|| Error::Invalid("The model's tokenizer has no end token.".into()))?;
        let pad = pad_token.and_then(id).unwrap_or(eos);
        // CLIP uses the first 49152 − 256 − 2 merges (the header line skipped).
        let ranks = merges
            .lines()
            .filter(|l| !l.starts_with("#version"))
            .take(49_152 - 256 - 2)
            .filter_map(|l| l.split_once(' ').map(|(a, b)| (a.to_string(), b.to_string())))
            .enumerate()
            .map(|(i, pair)| (pair, i))
            .collect();
        Ok(ClipTokenizer { vocab, ranks, bytes: byte_chars(), bos, eos, pad })
    }

    /// Byte-pair merges of one word (its last symbol marked `</w>`), lowest rank first.
    fn bpe(&self, word: &str) -> Vec<String> {
        let mut parts: Vec<String> = word.chars().map(String::from).collect();
        if let Some(last) = parts.last_mut() {
            last.push_str("</w>");
        }
        loop {
            let best = parts
                .windows(2)
                .enumerate()
                .filter_map(|(i, w)| match w {
                    [a, b] => self.ranks.get(&(a.clone(), b.clone())).map(|r| (*r, i)),
                    _ => None,
                })
                .min();
            let Some((_, i)) = best else { break };
            let (Some(a), Some(b)) = (parts.get(i).cloned(), parts.get(i + 1).cloned()) else { break };
            let merged = format!("{a}{b}");
            // Merge every occurrence of this pair, left to right.
            let mut out = Vec::with_capacity(parts.len());
            let mut j = 0;
            while j < parts.len() {
                if parts.get(j) == Some(&a) && parts.get(j + 1) == Some(&b) {
                    out.push(merged.clone());
                    j += 2;
                } else {
                    out.extend(parts.get(j).cloned());
                    j += 1;
                }
            }
            parts = out;
            if parts.len() == 1 {
                break;
            }
        }
        parts
    }

    /// The prompt's token ids: start, up to 75 tokens, end, then padding to [`LENGTH`].
    pub fn encode(&self, text: &str) -> Vec<i64> {
        let mut ids = vec![self.bos];
        'words: for word in split(&clean(text)) {
            let mapped: String = word.bytes().map(|b| self.bytes.get(usize::from(b)).copied().unwrap_or('?')).collect();
            for piece in self.bpe(&mapped) {
                if ids.len() >= LENGTH - 1 {
                    break 'words;
                }
                ids.push(self.vocab.get(&piece).copied().unwrap_or(self.eos));
            }
        }
        ids.push(self.eos);
        ids.resize(LENGTH, self.pad);
        ids
    }
}

/// Lowercase, control characters dropped, runs of whitespace as one space, trimmed.
fn clean(text: &str) -> String {
    let lower: String = text.chars().filter(|c| !c.is_control() || c.is_whitespace()).collect::<String>().to_lowercase();
    lower.split_whitespace().collect::<Vec<_>>().join(" ")
}

/// CLIP's pre-tokenizer: the special tokens, English contractions, runs of letters, single
/// digits, and runs of anything else that isn't a space.
fn split(text: &str) -> Vec<String> {
    let chars: Vec<char> = text.chars().collect();
    let mut out = Vec::new();
    let mut i = 0;
    let rest = |i: usize| chars.get(i..).map(|s| s.iter().collect::<String>()).unwrap_or_default();
    while i < chars.len() {
        let Some(&c) = chars.get(i) else { break };
        if c.is_whitespace() {
            i += 1;
            continue;
        }
        if c == '<' {
            let r = rest(i);
            if let Some(t) = ["<|startoftext|>", "<|endoftext|>"].into_iter().find(|t| r.starts_with(t)) {
                out.push(t.to_string());
                i += t.chars().count();
                continue;
            }
        }
        if c == '\'' {
            let r = rest(i);
            if let Some(t) = ["'s", "'t", "'re", "'ve", "'m", "'ll", "'d"].into_iter().find(|t| r.starts_with(t)) {
                out.push(t.to_string());
                i += t.chars().count();
                continue;
            }
        }
        if c.is_alphabetic() {
            let start = i;
            while chars.get(i).is_some_and(|c| c.is_alphabetic()) {
                i += 1;
            }
            out.push(chars.get(start..i).unwrap_or(&[]).iter().collect());
        } else if c.is_numeric() {
            out.push(c.to_string());
            i += 1;
        } else {
            let start = i;
            while chars.get(i).is_some_and(|c| !c.is_whitespace() && !c.is_alphabetic() && !c.is_numeric()) {
                // A contraction or special token starts a new piece.
                if i > start && matches!(chars.get(i), Some('\'') | Some('<')) {
                    break;
                }
                i += 1;
            }
            out.push(chars.get(start..i).unwrap_or(&[]).iter().collect());
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tiny() -> ClipTokenizer {
        let mut vocab = HashMap::new();
        for (i, t) in ["<|startoftext|>", "<|endoftext|>", "!", "c", "a", "t</w>", "ca", "cat</w>", "s</w>", "a</w>", ",</w>", "1</w>", "2</w>"].iter().enumerate() {
            vocab.insert(t.to_string(), i as i64 + 10);
        }
        ClipTokenizer::new(vocab, "#version: 0.2\nc a\nca t</w>\n", Some("!")).unwrap()
    }

    #[test]
    fn prompts_become_77_ids_with_start_end_and_padding() {
        let t = tiny();
        let ids = t.encode("  A CAT, 12 cats ");
        assert_eq!(ids.len(), LENGTH);
        // start, "a", "cat", ",", "1", "2", "cats" (ca + t, which isn't in the vocabulary, + s),
        // end, then "!" padding.
        assert_eq!(ids.get(..11).unwrap(), &[10, 19, 17, 20, 21, 22, 16, 11, 18, 11, 12]);
        assert!(ids.get(11..).unwrap().iter().all(|i| *i == 12));
        assert_eq!(t.encode("").get(..3).unwrap(), &[10, 11, 12]);
    }

    #[test]
    fn long_prompts_are_cut_and_still_end() {
        let t = tiny();
        let ids = t.encode(&"cat ".repeat(200));
        assert_eq!(ids.len(), LENGTH);
        assert_eq!(ids.last(), Some(&11));
        assert_eq!(ids.first(), Some(&10));
    }

    #[test]
    fn splitting_follows_clip() {
        assert_eq!(split("it's a <|endoftext|>dog!!"), vec!["it", "'s", "a", "<|endoftext|>", "dog", "!!"]);
        assert_eq!(split("x1y"), vec!["x", "1", "y"]);
        assert_eq!(clean("Hello\tWORLD\u{7}"), "hello world");
        let b = byte_chars();
        assert_eq!(b[usize::from(b'a')], 'a');
        assert_eq!(b[usize::from(b' ')], 'Ġ');
    }
}
