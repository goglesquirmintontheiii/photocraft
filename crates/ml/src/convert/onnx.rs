//! Writes ONNX models: just enough of the protobuf encoding for a graph of nodes, its inputs and
//! outputs, small constants inline, and weights in an external data file beside the model (so
//! models of any size load, and are written without holding the weights in memory).

use crate::{Error, Result};
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::Path;

pub const FLOAT: i32 = 1;
pub const INT64: i32 = 7;
pub const INT32: i32 = 6;
pub const FLOAT16: i32 = 10;

const IR_VERSION: i64 = 8;
pub const OPSET: i64 = 17;

// ---------------------------------------------------------------------------------------------
// Protobuf

fn varint(out: &mut Vec<u8>, mut v: u64) {
    loop {
        let b = (v & 0x7f) as u8;
        v >>= 7;
        if v == 0 {
            out.push(b);
            return;
        }
        out.push(b | 0x80);
    }
}

fn tag(out: &mut Vec<u8>, field: u32, wire: u8) {
    varint(out, (u64::from(field) << 3) | u64::from(wire));
}

fn int(out: &mut Vec<u8>, field: u32, v: i64) {
    tag(out, field, 0);
    varint(out, v as u64);
}

fn bytes(out: &mut Vec<u8>, field: u32, b: &[u8]) {
    tag(out, field, 2);
    varint(out, b.len() as u64);
    out.extend_from_slice(b);
}

fn string(out: &mut Vec<u8>, field: u32, s: &str) {
    bytes(out, field, s.as_bytes());
}

fn float(out: &mut Vec<u8>, field: u32, v: f32) {
    tag(out, field, 5);
    out.extend_from_slice(&v.to_le_bytes());
}

// ---------------------------------------------------------------------------------------------
// Model

#[derive(Clone, Debug, PartialEq)]
pub enum Attr {
    Int(i64),
    Float(f32),
    Ints(Vec<i64>),
    Str(String),
}

#[derive(Clone, Debug, PartialEq)]
pub enum Dim {
    Fixed(i64),
    Named(String),
}

struct Node {
    op: &'static str,
    inputs: Vec<String>,
    outputs: Vec<String>,
    attrs: Vec<(&'static str, Attr)>,
}

enum Data {
    Raw(Vec<u8>),
    External { offset: u64, length: u64 },
}

struct Init {
    name: String,
    dims: Vec<i64>,
    dtype: i32,
    data: Data,
}

struct Io {
    name: String,
    elem: i32,
    dims: Vec<Dim>,
}

/// A graph being built: nodes, constants and weights. Weights go straight to `<model>_data`.
pub struct Graph {
    nodes: Vec<Node>,
    inits: Vec<Init>,
    inputs: Vec<Io>,
    outputs: Vec<Io>,
    data: BufWriter<File>,
    data_name: String,
    offset: u64,
    counter: usize,
    /// Element type of weights and activations (FLOAT16, or FLOAT).
    pub elem: i32,
}

impl Graph {
    /// A graph whose model will be written to `model` (weights to `model` + `_data`).
    pub fn new(model: &Path, elem: i32) -> Result<Graph> {
        let data_name = format!("{}_data", model.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_else(|| "model.onnx".into()));
        let data_path = model.with_file_name(&data_name);
        let file = File::create(&data_path).map_err(|e| Error::io(&format!("Couldn't create {}", data_path.display()), e))?;
        Ok(Graph { nodes: Vec::new(), inits: Vec::new(), inputs: Vec::new(), outputs: Vec::new(), data: BufWriter::with_capacity(1 << 20, file), data_name, offset: 0, counter: 0, elem })
    }

    fn fresh(&mut self, hint: &str) -> String {
        self.counter += 1;
        format!("{hint}_{}", self.counter)
    }

    pub fn input(&mut self, name: &str, elem: i32, dims: Vec<Dim>) -> String {
        self.inputs.push(Io { name: name.into(), elem, dims });
        name.into()
    }

    pub fn output(&mut self, value: &str, name: &str, elem: i32, dims: Vec<Dim>) {
        self.node("Identity", &[value], vec![], Some(name));
        self.outputs.push(Io { name: name.into(), elem, dims });
    }

    fn node(&mut self, op: &'static str, inputs: &[&str], attrs: Vec<(&'static str, Attr)>, out: Option<&str>) -> String {
        let o = out.map_or_else(|| self.fresh(op), str::to_string);
        self.nodes.push(Node { op, inputs: inputs.iter().map(|s| s.to_string()).collect(), outputs: vec![o.clone()], attrs });
        o
    }

    /// One node with one output.
    pub fn op(&mut self, op: &'static str, inputs: &[&str], attrs: Vec<(&'static str, Attr)>) -> String {
        self.node(op, inputs, attrs, None)
    }

    /// A small constant, stored in the model.
    pub fn constant_i64(&mut self, values: &[i64]) -> String {
        let name = self.fresh("const");
        let raw = values.iter().flat_map(|v| v.to_le_bytes()).collect();
        self.inits.push(Init { name: name.clone(), dims: vec![values.len() as i64], dtype: INT64, data: Data::Raw(raw) });
        name
    }

    pub fn constant_f32(&mut self, dims: &[i64], values: &[f32]) -> String {
        let name = self.fresh("const");
        let raw = values.iter().flat_map(|v| v.to_le_bytes()).collect();
        self.inits.push(Init { name: name.clone(), dims: dims.to_vec(), dtype: FLOAT, data: Data::Raw(raw) });
        name
    }

    /// A constant in the graph's element type (small, or large: then it goes to the data file).
    pub fn constant(&mut self, dims: &[i64], values: &[f32]) -> Result<String> {
        if values.len() > 64 {
            return self.weight_values(dims, values);
        }
        let name = self.fresh("const");
        let raw = encode(self.elem, values);
        self.inits.push(Init { name: name.clone(), dims: dims.to_vec(), dtype: self.elem, data: Data::Raw(raw) });
        Ok(name)
    }

    pub fn scalar(&mut self, v: f32) -> Result<String> {
        self.constant(&[], &[v])
    }

    /// A weight, written to the data file in the graph's element type.
    pub fn weight_values(&mut self, dims: &[i64], values: &[f32]) -> Result<String> {
        let expect: i64 = dims.iter().product();
        if expect.max(1) as usize != values.len().max(1) {
            return Err(Error::Invalid(format!("internal error: a weight of shape {dims:?} has {} values", values.len())));
        }
        let name = self.fresh("w");
        let raw = encode(self.elem, values);
        // Aligned to 64 bytes, as ONNX Runtime prefers for mapping the file.
        let pad = (64 - self.offset % 64) % 64;
        if pad > 0 {
            self.data.write_all(&vec![0u8; pad as usize]).map_err(|e| Error::io("Couldn't write the model's weights", e))?;
            self.offset += pad;
        }
        self.data.write_all(&raw).map_err(|e| Error::io("Couldn't write the model's weights", e))?;
        let length = raw.len() as u64;
        self.inits.push(Init { name: name.clone(), dims: dims.to_vec(), dtype: self.elem, data: Data::External { offset: self.offset, length } });
        self.offset += length;
        Ok(name)
    }

    /// Writes the model file (and finishes the data file).
    pub fn finish(mut self, model: &Path) -> Result<()> {
        self.data.flush().map_err(|e| Error::io("Couldn't write the model's weights", e))?;
        let mut graph = Vec::new();
        for n in &self.nodes {
            let mut b = Vec::new();
            for i in &n.inputs {
                string(&mut b, 1, i);
            }
            for o in &n.outputs {
                string(&mut b, 2, o);
            }
            string(&mut b, 3, n.outputs.first().map_or("", String::as_str));
            string(&mut b, 4, n.op);
            for (name, a) in &n.attrs {
                let mut ab = Vec::new();
                string(&mut ab, 1, name);
                match a {
                    Attr::Float(f) => {
                        float(&mut ab, 2, *f);
                        int(&mut ab, 20, 1);
                    }
                    Attr::Int(i) => {
                        int(&mut ab, 3, *i);
                        int(&mut ab, 20, 2);
                    }
                    Attr::Str(s) => {
                        string(&mut ab, 4, s);
                        int(&mut ab, 20, 3);
                    }
                    Attr::Ints(v) => {
                        for i in v {
                            int(&mut ab, 8, *i);
                        }
                        int(&mut ab, 20, 7);
                    }
                }
                bytes(&mut b, 5, &ab);
            }
            bytes(&mut graph, 1, &b);
        }
        string(&mut graph, 2, "photocraft");
        for t in &self.inits {
            let mut b = Vec::new();
            for d in &t.dims {
                int(&mut b, 1, *d);
            }
            int(&mut b, 2, i64::from(t.dtype));
            string(&mut b, 8, &t.name);
            match &t.data {
                Data::Raw(raw) => bytes(&mut b, 9, raw),
                Data::External { offset, length } => {
                    for (k, v) in [("location", self.data_name.clone()), ("offset", offset.to_string()), ("length", length.to_string())] {
                        let mut e = Vec::new();
                        string(&mut e, 1, k);
                        string(&mut e, 2, &v);
                        bytes(&mut b, 13, &e);
                    }
                    int(&mut b, 14, 1);
                }
            }
            bytes(&mut graph, 5, &b);
        }
        for (field, list) in [(11u32, &self.inputs), (12u32, &self.outputs)] {
            for io in list {
                let mut shape = Vec::new();
                for d in &io.dims {
                    let mut db = Vec::new();
                    match d {
                        Dim::Fixed(v) => int(&mut db, 1, *v),
                        Dim::Named(n) => string(&mut db, 2, n),
                    }
                    bytes(&mut shape, 1, &db);
                }
                let mut tensor = Vec::new();
                int(&mut tensor, 1, i64::from(io.elem));
                bytes(&mut tensor, 2, &shape);
                let mut ty = Vec::new();
                bytes(&mut ty, 1, &tensor);
                let mut vi = Vec::new();
                string(&mut vi, 1, &io.name);
                bytes(&mut vi, 2, &ty);
                bytes(&mut graph, field, &vi);
            }
        }
        let mut m = Vec::new();
        int(&mut m, 1, IR_VERSION);
        string(&mut m, 2, "PhotoCraft");
        bytes(&mut m, 7, &graph);
        let mut opset = Vec::new();
        string(&mut opset, 1, "");
        int(&mut opset, 2, OPSET);
        bytes(&mut m, 8, &opset);
        std::fs::write(model, m).map_err(|e| Error::io(&format!("Couldn't write {}", model.display()), e))
    }
}

/// Values as little-endian bytes of `elem` (FLOAT16 or FLOAT).
fn encode(elem: i32, values: &[f32]) -> Vec<u8> {
    if elem == FLOAT16 {
        values.iter().flat_map(|v| half::f16::from_f32(*v).to_le_bytes()).collect()
    } else {
        values.iter().flat_map(|v| v.to_le_bytes()).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn varints_encode_like_protobuf() {
        let mut v = Vec::new();
        varint(&mut v, 300);
        assert_eq!(v, [0xac, 0x02]);
        let mut v = Vec::new();
        int(&mut v, 1, -1);
        assert_eq!(v.len(), 11, "negative numbers take ten bytes");
    }
}
