//! Building blocks of the Stable Diffusion networks as ONNX nodes, with their weights read from
//! the checkpoint by name: linear and convolution layers, group and layer norms, activations and
//! attention.

use super::onnx::{Attr, Graph};
use super::safetensors::SafeTensors;
use crate::{Error, Result};

pub struct Builder<'a> {
    pub g: Graph,
    pub ck: &'a SafeTensors,
    /// Called after each weight is written (progress, cancelling).
    pub tick: &'a mut dyn FnMut() -> Result<()>,
}

fn dims(shape: &[usize]) -> Vec<i64> {
    shape.iter().map(|d| *d as i64).collect()
}

impl Builder<'_> {
    pub fn has(&self, key: &str) -> bool {
        self.ck.has(key)
    }

    pub fn shape(&self, key: &str) -> Result<Vec<usize>> {
        self.ck.shape(key).map(<[usize]>::to_vec).ok_or_else(|| Error::Invalid(format!("The checkpoint has no {key}.")))
    }

    /// A checkpoint tensor as a weight, unchanged.
    pub fn weight(&mut self, key: &str) -> Result<String> {
        let (shape, v) = self.ck.read(key)?;
        let w = self.g.weight_values(&dims(&shape), &v)?;
        (self.tick)()?;
        Ok(w)
    }

    /// A checkpoint tensor with another shape (same values).
    pub fn weight_as(&mut self, key: &str, shape: &[usize]) -> Result<String> {
        let (_, v) = self.ck.read(key)?;
        let w = self.g.weight_values(&dims(shape), &v)?;
        (self.tick)()?;
        Ok(w)
    }

    /// Values computed elsewhere, as a weight.
    pub fn values(&mut self, shape: &[usize], v: &[f32]) -> Result<String> {
        let w = self.g.weight_values(&dims(shape), v)?;
        (self.tick)()?;
        Ok(w)
    }

    pub fn op(&mut self, op: &'static str, inputs: &[&str]) -> String {
        self.g.op(op, inputs, vec![])
    }

    pub fn op_a(&mut self, op: &'static str, inputs: &[&str], attrs: Vec<(&'static str, Attr)>) -> String {
        self.g.op(op, inputs, attrs)
    }

    pub fn i64s(&mut self, v: &[i64]) -> String {
        self.g.constant_i64(v)
    }

    /// `x · Wᵀ (+ b)` for a weight `[out, in]` (also a 1×1 convolution's `[out, in, 1, 1]`).
    pub fn linear_w(&mut self, x: &str, w: (&[usize], Vec<f32>), b: Option<&str>) -> Result<String> {
        let (shape, v) = w;
        let (out, inp) = (shape.first().copied().unwrap_or(0), shape.get(1).copied().unwrap_or(0));
        if out * inp != v.len() {
            return Err(Error::Invalid("internal error: a linear weight isn't two-dimensional".into()));
        }
        let mut t = vec![0.0f32; v.len()];
        for o in 0..out {
            for i in 0..inp {
                if let (Some(d), Some(s)) = (t.get_mut(i * out + o), v.get(o * inp + i)) {
                    *d = *s;
                }
            }
        }
        let wt = self.values(&[inp, out], &t)?;
        let y = self.op("MatMul", &[x, &wt]);
        Ok(match b {
            Some(b) => {
                let bias = self.weight(b)?;
                self.op("Add", &[&y, &bias])
            }
            None => y,
        })
    }

    pub fn linear(&mut self, x: &str, prefix: &str) -> Result<String> {
        let (shape, v) = self.ck.read(&format!("{prefix}.weight"))?;
        let b = format!("{prefix}.bias");
        let b = self.has(&b).then_some(b);
        self.linear_w(x, (&shape, v), b.as_deref())
    }

    pub fn conv(&mut self, x: &str, prefix: &str, stride: i64, pads: [i64; 4]) -> Result<String> {
        let w = self.weight(&format!("{prefix}.weight"))?;
        let b = self.weight(&format!("{prefix}.bias"))?;
        Ok(self.op_a("Conv", &[x, &w, &b], vec![("strides", Attr::Ints(vec![stride, stride])), ("pads", Attr::Ints(pads.to_vec()))]))
    }

    /// Group normalization (as a reshape, instance norm and reshape, which every runtime has).
    pub fn group_norm(&mut self, x: &str, prefix: &str, groups: i64, eps: f32) -> Result<String> {
        let c = self.shape(&format!("{prefix}.weight"))?.first().copied().unwrap_or(0);
        let shape = self.op("Shape", &[x]);
        let grouped = self.i64s(&[0, groups, -1]);
        let r = self.op("Reshape", &[x, &grouped]);
        let ones = self.g.constant(&[groups], &vec![1.0; groups as usize])?;
        let zeros = self.g.constant(&[groups], &vec![0.0; groups as usize])?;
        let n = self.op_a("InstanceNormalization", &[&r, &ones, &zeros], vec![("epsilon", Attr::Float(eps))]);
        let y = self.op("Reshape", &[&n, &shape]);
        let gamma = self.weight_as(&format!("{prefix}.weight"), &[c, 1, 1])?;
        let beta = self.weight_as(&format!("{prefix}.bias"), &[c, 1, 1])?;
        let y = self.op("Mul", &[&y, &gamma]);
        Ok(self.op("Add", &[&y, &beta]))
    }

    pub fn layer_norm(&mut self, x: &str, prefix: &str, eps: f32) -> Result<String> {
        let g = self.weight(&format!("{prefix}.weight"))?;
        let b = self.weight(&format!("{prefix}.bias"))?;
        Ok(self.op_a("LayerNormalization", &[x, &g, &b], vec![("axis", Attr::Int(-1)), ("epsilon", Attr::Float(eps))]))
    }

    pub fn silu(&mut self, x: &str) -> String {
        let s = self.op("Sigmoid", &[x]);
        self.op("Mul", &[x, &s])
    }

    /// Exact GELU: x · ½(1 + erf(x/√2)).
    pub fn gelu(&mut self, x: &str) -> Result<String> {
        let r = self.g.scalar(std::f32::consts::FRAC_1_SQRT_2)?;
        let half = self.g.scalar(0.5)?;
        let one = self.g.scalar(1.0)?;
        let s = self.op("Mul", &[x, &r]);
        let e = self.op("Erf", &[&s]);
        let e = self.op("Add", &[&e, &one]);
        let h = self.op("Mul", &[x, &half]);
        Ok(self.op("Mul", &[&h, &e]))
    }

    /// CLIP's quick GELU: x · σ(1.702 x).
    pub fn quick_gelu(&mut self, x: &str) -> Result<String> {
        let k = self.g.scalar(1.702)?;
        let s = self.op("Mul", &[x, &k]);
        let s = self.op("Sigmoid", &[&s]);
        Ok(self.op("Mul", &[x, &s]))
    }

    /// `[B, C, H, W]` → `[B, H·W, C]`.
    pub fn tokens_of(&mut self, x: &str) -> String {
        let flat = self.i64s(&[0, 0, -1]);
        let r = self.op("Reshape", &[x, &flat]);
        self.op_a("Transpose", &[&r], vec![("perm", Attr::Ints(vec![0, 2, 1]))])
    }

    /// `[B, H·W, C]` → the shape `shape` of a `[B, C, H, W]` tensor.
    pub fn image_of_tokens(&mut self, x: &str, shape: &str) -> String {
        let t = self.op_a("Transpose", &[x], vec![("perm", Attr::Ints(vec![0, 2, 1]))]);
        self.op("Reshape", &[&t, shape])
    }

    /// Scaled dot-product attention of `q` `[B, N, heads·d]` over `k`, `v` `[B, M, heads·d]`, with
    /// an optional additive mask `[N, M]`.
    pub fn attention(&mut self, q: &str, k: &str, v: &str, heads: i64, dim: i64, mask: Option<&str>) -> Result<String> {
        let d = dim / heads.max(1);
        let split = self.i64s(&[0, 0, heads, d]);
        let q = self.op("Reshape", &[q, &split]);
        let q = self.op_a("Transpose", &[&q], vec![("perm", Attr::Ints(vec![0, 2, 1, 3]))]);
        let scale = self.g.scalar(1.0 / (d as f32).sqrt())?;
        let q = self.op("Mul", &[&q, &scale]);
        let k = self.op("Reshape", &[k, &split]);
        let k = self.op_a("Transpose", &[&k], vec![("perm", Attr::Ints(vec![0, 2, 3, 1]))]);
        let v = self.op("Reshape", &[v, &split]);
        let v = self.op_a("Transpose", &[&v], vec![("perm", Attr::Ints(vec![0, 2, 1, 3]))]);
        let mut s = self.op("MatMul", &[&q, &k]);
        if let Some(m) = mask {
            s = self.op("Add", &[&s, m]);
        }
        let p = self.op_a("Softmax", &[&s], vec![("axis", Attr::Int(-1))]);
        let o = self.op("MatMul", &[&p, &v]);
        let o = self.op_a("Transpose", &[&o], vec![("perm", Attr::Ints(vec![0, 2, 1, 3]))]);
        let merge = self.i64s(&[0, 0, -1]);
        Ok(self.op("Reshape", &[&o, &merge]))
    }

    /// Nearest-neighbour ×2 upscaling of `[B, C, H, W]`.
    pub fn upsample2(&mut self, x: &str) -> String {
        let scales = self.g.constant_f32(&[4], &[1.0, 1.0, 2.0, 2.0]);
        self.op_a(
            "Resize",
            &[x, "", &scales],
            vec![
                ("mode", Attr::Str("nearest".into())),
                ("coordinate_transformation_mode", Attr::Str("asymmetric".into())),
                ("nearest_mode", Attr::Str("floor".into())),
            ],
        )
    }
}
