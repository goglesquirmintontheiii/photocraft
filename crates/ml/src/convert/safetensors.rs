//! Reads tensors from a `.safetensors` file one at a time (checkpoints are gigabytes): an 8-byte
//! little-endian header length, a JSON header naming each tensor's type, shape and byte range,
//! then the data.

use crate::{Error, Result};
use serde_json::Value;
use std::collections::HashMap;
use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::Path;
use std::sync::Mutex;

#[derive(Clone, Debug, PartialEq)]
pub struct Info {
    pub dtype: String,
    pub shape: Vec<usize>,
    start: u64,
    end: u64,
}

impl Info {
    pub fn len(&self) -> usize {
        self.shape.iter().product()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

pub struct SafeTensors {
    file: Mutex<File>,
    data_start: u64,
    pub tensors: HashMap<String, Info>,
}

/// Header size limit: real headers are a few hundred kilobytes.
const MAX_HEADER: u64 = 100 << 20;

impl SafeTensors {
    pub fn open(path: &Path) -> Result<SafeTensors> {
        let mut file = File::open(path).map_err(|e| Error::io(&format!("Couldn't open {}", path.display()), e))?;
        let size = file.metadata().map(|m| m.len()).unwrap_or(0);
        let bad = || Error::Invalid(format!("{} isn't a readable .safetensors file.", path.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default()));
        let mut len = [0u8; 8];
        file.read_exact(&mut len).map_err(|_| bad())?;
        let n = u64::from_le_bytes(len);
        if n == 0 || n > MAX_HEADER || n + 8 > size {
            return Err(bad());
        }
        let mut header = vec![0u8; n as usize];
        file.read_exact(&mut header).map_err(|_| bad())?;
        let json: Value = serde_json::from_slice(&header).map_err(|_| bad())?;
        let obj = json.as_object().ok_or_else(bad)?;
        let data_start = 8 + n;
        let mut tensors = HashMap::new();
        for (name, v) in obj {
            if name == "__metadata__" {
                continue;
            }
            let dtype = v.get("dtype").and_then(Value::as_str).ok_or_else(bad)?.to_string();
            let shape: Vec<usize> = v.get("shape").and_then(Value::as_array).ok_or_else(bad)?.iter().map(|d| d.as_u64().map(|d| d as usize)).collect::<Option<_>>().ok_or_else(bad)?;
            let offs = v.get("data_offsets").and_then(Value::as_array).ok_or_else(bad)?;
            let (Some(start), Some(end)) = (offs.first().and_then(Value::as_u64), offs.get(1).and_then(Value::as_u64)) else { return Err(bad()) };
            let elem = elem_size(&dtype);
            let count: usize = shape.iter().product();
            if end < start || data_start + end > size || elem.is_some_and(|e| (end - start) as usize != count * e) {
                return Err(bad());
            }
            tensors.insert(name.clone(), Info { dtype, shape, start, end });
        }
        Ok(SafeTensors { file: Mutex::new(file), data_start, tensors })
    }

    pub fn has(&self, name: &str) -> bool {
        self.tensors.contains_key(name)
    }

    pub fn shape(&self, name: &str) -> Option<&[usize]> {
        self.tensors.get(name).map(|i| i.shape.as_slice())
    }

    /// A tensor's values as `f32`, whatever they're stored as.
    pub fn read(&self, name: &str) -> Result<(Vec<usize>, Vec<f32>)> {
        let info = self.tensors.get(name).ok_or_else(|| Error::Invalid(format!("The checkpoint has no {name}.")))?;
        let mut raw = vec![0u8; (info.end - info.start) as usize];
        {
            let mut f = self.file.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
            f.seek(SeekFrom::Start(self.data_start + info.start)).and_then(|_| f.read_exact(&mut raw)).map_err(|e| Error::io("Couldn't read the checkpoint", e))?;
        }
        let values: Vec<f32> = match info.dtype.as_str() {
            "F32" => raw.chunks_exact(4).map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]])).collect(),
            "F16" => raw.chunks_exact(2).map(|b| half::f16::from_le_bytes([b[0], b[1]]).to_f32()).collect(),
            "BF16" => raw.chunks_exact(2).map(|b| half::bf16::from_le_bytes([b[0], b[1]]).to_f32()).collect(),
            "F64" => raw.chunks_exact(8).map(|b| f64::from_le_bytes([b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7]]) as f32).collect(),
            "I64" => raw.chunks_exact(8).map(|b| i64::from_le_bytes([b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7]]) as f32).collect(),
            other => return Err(Error::Unsupported(format!("The checkpoint stores {name} as {other}, which PhotoCraft can't read."))),
        };
        Ok((info.shape.clone(), values))
    }
}

fn elem_size(dtype: &str) -> Option<usize> {
    match dtype {
        "F32" | "I32" | "U32" => Some(4),
        "F16" | "BF16" | "I16" | "U16" => Some(2),
        "F64" | "I64" | "U64" => Some(8),
        "I8" | "U8" | "BOOL" => Some(1),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reads_types_and_refuses_damage() {
        let dir = std::env::temp_dir().join(format!("photocraft-st-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let header = r#"{"a":{"dtype":"F16","shape":[2],"data_offsets":[0,4]},"b":{"dtype":"BF16","shape":[1],"data_offsets":[4,6]},"c":{"dtype":"F32","shape":[1,1],"data_offsets":[6,10]},"__metadata__":{"x":"y"}}"#;
        let mut bytes = (header.len() as u64).to_le_bytes().to_vec();
        bytes.extend_from_slice(header.as_bytes());
        bytes.extend_from_slice(&half::f16::from_f32(1.5).to_le_bytes());
        bytes.extend_from_slice(&half::f16::from_f32(-2.0).to_le_bytes());
        bytes.extend_from_slice(&half::bf16::from_f32(3.0).to_le_bytes());
        bytes.extend_from_slice(&0.25f32.to_le_bytes());
        let p = dir.join("t.safetensors");
        std::fs::write(&p, &bytes).unwrap();
        let st = SafeTensors::open(&p).unwrap();
        assert_eq!(st.read("a").unwrap(), (vec![2], vec![1.5, -2.0]));
        assert_eq!(st.read("b").unwrap().1, vec![3.0]);
        assert_eq!(st.read("c").unwrap(), (vec![1, 1], vec![0.25]));
        assert!(st.read("missing").is_err());
        // A truncated file and junk are errors, not panics.
        std::fs::write(&p, &bytes[..bytes.len() - 3]).unwrap();
        assert!(SafeTensors::open(&p).is_err());
        std::fs::write(&p, b"not a checkpoint at all").unwrap();
        assert!(SafeTensors::open(&p).is_err());
        let _ = std::fs::remove_dir_all(dir);
    }
}
