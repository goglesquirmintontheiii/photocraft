//! The Stable Diffusion 1.x / 2.x networks, read from a checkpoint's original (LDM) names:
//! the UNet (`model.diffusion_model.*`), the VAE (`first_stage_model.*`) and the text encoder
//! (`cond_stage_model.transformer.*` for CLIP, `cond_stage_model.model.*` for OpenCLIP). The
//! structure (levels, blocks, channels) is read from which weights exist and their shapes.

use super::layers::Builder;
use super::onnx::{Attr, Dim, INT32};
use crate::Result;

pub const UNET: &str = "model.diffusion_model.";
pub const VAE: &str = "first_stage_model.";

/// Attention heads: a fixed count (SD 1.x: 8) or a fixed size per head (SD 2.x: 64).
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Heads {
    Count(i64),
    Dim(i64),
}

impl Heads {
    fn for_channels(self, c: i64) -> i64 {
        match self {
            Heads::Count(n) => n,
            Heads::Dim(d) => (c / d.max(1)).max(1),
        }
    }
}

fn dyn_dim(n: &str) -> Dim {
    Dim::Named(n.into())
}

// ---------------------------------------------------------------------------------------------
// UNet

fn resblock(b: &mut Builder, x: &str, p: &str, emb: &str) -> Result<String> {
    let h = b.group_norm(x, &format!("{p}.in_layers.0"), 32, 1e-5)?;
    let h = b.silu(&h);
    let h = b.conv(&h, &format!("{p}.in_layers.2"), 1, [1, 1, 1, 1])?;
    let e = b.silu(emb);
    let e = b.linear(&e, &format!("{p}.emb_layers.1"))?;
    let axes = b.i64s(&[2, 3]);
    let e = b.op("Unsqueeze", &[&e, &axes]);
    let h = b.op("Add", &[&h, &e]);
    let h = b.group_norm(&h, &format!("{p}.out_layers.0"), 32, 1e-5)?;
    let h = b.silu(&h);
    let h = b.conv(&h, &format!("{p}.out_layers.3"), 1, [1, 1, 1, 1])?;
    let skip = if b.has(&format!("{p}.skip_connection.weight")) { b.conv(x, &format!("{p}.skip_connection"), 1, [0, 0, 0, 0])? } else { x.to_string() };
    Ok(b.op("Add", &[&skip, &h]))
}

fn cross_attention(b: &mut Builder, x: &str, ctx: &str, p: &str, heads: Heads) -> Result<String> {
    let inner = b.shape(&format!("{p}.to_q.weight"))?.first().copied().unwrap_or(0) as i64;
    let q = b.linear(x, &format!("{p}.to_q"))?;
    let k = b.linear(ctx, &format!("{p}.to_k"))?;
    let v = b.linear(ctx, &format!("{p}.to_v"))?;
    let o = b.attention(&q, &k, &v, heads.for_channels(inner), inner, None)?;
    b.linear(&o, &format!("{p}.to_out.0"))
}

fn transformer(b: &mut Builder, x: &str, p: &str, ctx: &str, heads: Heads) -> Result<String> {
    let shape = b.op("Shape", &[x]);
    let h = b.group_norm(x, &format!("{p}.norm"), 32, 1e-6)?;
    let linear_proj = b.shape(&format!("{p}.proj_in.weight"))?.len() == 2;
    let mut h = if linear_proj {
        let t = b.tokens_of(&h);
        b.linear(&t, &format!("{p}.proj_in"))?
    } else {
        let c = b.conv(&h, &format!("{p}.proj_in"), 1, [0, 0, 0, 0])?;
        b.tokens_of(&c)
    };
    let mut k = 0;
    while b.has(&format!("{p}.transformer_blocks.{k}.norm1.weight")) {
        let t = format!("{p}.transformer_blocks.{k}");
        let n = b.layer_norm(&h, &format!("{t}.norm1"), 1e-5)?;
        let a = cross_attention(b, &n, &n, &format!("{t}.attn1"), heads)?;
        h = b.op("Add", &[&h, &a]);
        let n = b.layer_norm(&h, &format!("{t}.norm2"), 1e-5)?;
        let a = cross_attention(b, &n, ctx, &format!("{t}.attn2"), heads)?;
        h = b.op("Add", &[&h, &a]);
        let n = b.layer_norm(&h, &format!("{t}.norm3"), 1e-5)?;
        // GEGLU: one projection to twice the width; the first half gated by GELU of the second.
        let inner = b.shape(&format!("{t}.ff.net.0.proj.weight"))?.first().copied().unwrap_or(0) as i64 / 2;
        let f = b.linear(&n, &format!("{t}.ff.net.0.proj"))?;
        let axes = b.i64s(&[-1]);
        let (s0, s1, s2) = (b.i64s(&[0]), b.i64s(&[inner]), b.i64s(&[inner * 2]));
        let x1 = b.op("Slice", &[&f, &s0, &s1, &axes]);
        let gate = b.op("Slice", &[&f, &s1, &s2, &axes]);
        let gate = b.gelu(&gate)?;
        let f = b.op("Mul", &[&x1, &gate]);
        let f = b.linear(&f, &format!("{t}.ff.net.2"))?;
        h = b.op("Add", &[&h, &f]);
        k += 1;
    }
    let out = if linear_proj {
        let l = b.linear(&h, &format!("{p}.proj_out"))?;
        b.image_of_tokens(&l, &shape)
    } else {
        let s = b.image_of_tokens(&h, &shape);
        b.conv(&s, &format!("{p}.proj_out"), 1, [0, 0, 0, 0])?
    };
    Ok(b.op("Add", &[&out, x]))
}

/// The pieces of one input or output block.
struct Parts {
    res: bool,
    attn: bool,
    down: bool,
    /// The index of the upsampling layer inside an output block.
    up: Option<usize>,
}

fn parts(b: &Builder, block: &str) -> Parts {
    let up = (1..=2).find(|i| b.has(&format!("{block}.{i}.conv.weight")));
    Parts {
        res: b.has(&format!("{block}.0.in_layers.0.weight")),
        attn: b.has(&format!("{block}.1.norm.weight")),
        down: b.has(&format!("{block}.0.op.weight")),
        up,
    }
}

/// The UNet: `sample` `[B, C, H, W]`, `timestep` `[T]` (T = 1 or B), `encoder_hidden_states`
/// `[B, S, D]` → `out_sample`.
pub fn unet(b: &mut Builder, heads: Heads) -> Result<()> {
    let p = |s: &str| format!("{UNET}{s}");
    let cin = b.shape(&p("input_blocks.0.0.weight"))?.get(1).copied().unwrap_or(4) as i64;
    let sample = b.g.input("sample", b.g.elem, vec![dyn_dim("batch"), Dim::Fixed(cin), dyn_dim("height"), dyn_dim("width")]);
    let timestep = b.g.input("timestep", b.g.elem, vec![dyn_dim("steps")]);
    let ctx_dim = b.shape(&p("middle_block.1.transformer_blocks.0.attn2.to_k.weight")).ok().and_then(|s| s.get(1).copied()).unwrap_or(768) as i64;
    let ctx = b.g.input("encoder_hidden_states", b.g.elem, vec![dyn_dim("batch"), dyn_dim("sequence"), Dim::Fixed(ctx_dim)]);

    // Timestep embedding: cos/sin of the timestep at log-spaced frequencies, then an MLP.
    let mc = b.shape(&p("time_embed.0.weight"))?.get(1).copied().unwrap_or(320);
    let half = mc / 2;
    let freqs: Vec<f32> = (0..half).map(|i| (-(10000f64.ln()) * i as f64 / half as f64).exp() as f32).collect();
    let t32 = b.op_a("Cast", &[&timestep], vec![("to", Attr::Int(1))]);
    let ax1 = b.i64s(&[1]);
    let t2 = b.op("Unsqueeze", &[&t32, &ax1]);
    let f = b.g.constant_f32(&[1, half as i64], &freqs);
    let m = b.op("Mul", &[&t2, &f]);
    let (cos, sin) = (b.op("Cos", &[&m]), b.op("Sin", &[&m]));
    let tf = b.op_a("Concat", &[&cos, &sin], vec![("axis", Attr::Int(1))]);
    let tf = b.op_a("Cast", &[&tf], vec![("to", Attr::Int(i64::from(b.g.elem)))]);
    let e = b.linear(&tf, &p("time_embed.0"))?;
    let e = b.silu(&e);
    let mut emb = b.linear(&e, &p("time_embed.2"))?;
    // SDXL: the pooled text and the picture's size and crop (`time_ids`), each sinusoidally
    // embedded, add to the timestep's embedding.
    if b.has(&p("label_emb.0.0.weight")) {
        let width = b.shape(&p("label_emb.0.0.weight"))?.get(1).copied().unwrap_or(2816);
        let pooled = b.shape("conditioner.embedders.1.model.text_projection").ok().and_then(|s| s.get(1).copied()).unwrap_or_else(|| width.saturating_sub(6 * 256));
        let tdim = width.saturating_sub(pooled) / 6;
        let text_embeds = b.g.input("text_embeds", b.g.elem, vec![dyn_dim("batch"), Dim::Fixed(pooled as i64)]);
        let time_ids = b.g.input("time_ids", b.g.elem, vec![dyn_dim("batch"), Dim::Fixed(6)]);
        let half = tdim / 2;
        let freqs: Vec<f32> = (0..half).map(|i| (-(10000f64.ln()) * i as f64 / half as f64).exp() as f32).collect();
        let flat = b.i64s(&[-1, 1]);
        let t = b.op("Reshape", &[&time_ids, &flat]);
        let t = b.op_a("Cast", &[&t], vec![("to", Attr::Int(1))]);
        let f = b.g.constant_f32(&[1, half as i64], &freqs);
        let m = b.op("Mul", &[&t, &f]);
        let (cos, sin) = (b.op("Cos", &[&m]), b.op("Sin", &[&m]));
        let te = b.op_a("Concat", &[&cos, &sin], vec![("axis", Attr::Int(1))]);
        let rows = b.i64s(&[-1, (6 * tdim) as i64]);
        let te = b.op("Reshape", &[&te, &rows]);
        let te = b.op_a("Cast", &[&te], vec![("to", Attr::Int(i64::from(b.g.elem)))]);
        let add = b.op_a("Concat", &[&text_embeds, &te], vec![("axis", Attr::Int(1))]);
        let add = b.linear(&add, &p("label_emb.0.0"))?;
        let add = b.silu(&add);
        let add = b.linear(&add, &p("label_emb.0.2"))?;
        emb = b.op("Add", &[&emb, &add]);
    }

    let mut h = b.conv(&sample, &p("input_blocks.0.0"), 1, [1, 1, 1, 1])?;
    let mut skips = vec![h.clone()];
    let mut i = 1;
    loop {
        let block = p(&format!("input_blocks.{i}"));
        let parts = parts(b, &block);
        if !parts.res && !parts.down {
            break;
        }
        if parts.down {
            h = b.conv(&h, &format!("{block}.0.op"), 2, [1, 1, 1, 1])?;
        } else {
            h = resblock(b, &h, &format!("{block}.0"), &emb)?;
            if parts.attn {
                h = transformer(b, &h, &format!("{block}.1"), &ctx, heads)?;
            }
        }
        skips.push(h.clone());
        i += 1;
    }
    h = resblock(b, &h, &p("middle_block.0"), &emb)?;
    if b.has(&p("middle_block.1.norm.weight")) {
        h = transformer(b, &h, &p("middle_block.1"), &ctx, heads)?;
    }
    h = resblock(b, &h, &p("middle_block.2"), &emb)?;
    let mut o = 0;
    loop {
        let block = p(&format!("output_blocks.{o}"));
        let parts = parts(b, &block);
        if !parts.res {
            break;
        }
        let skip = skips.pop().ok_or_else(|| crate::Error::Invalid("The checkpoint's UNet has more output blocks than input blocks.".into()))?;
        h = b.op_a("Concat", &[&h, &skip], vec![("axis", Attr::Int(1))]);
        h = resblock(b, &h, &format!("{block}.0"), &emb)?;
        if parts.attn {
            h = transformer(b, &h, &format!("{block}.1"), &ctx, heads)?;
        }
        if let Some(u) = parts.up {
            h = b.upsample2(&h);
            h = b.conv(&h, &format!("{block}.{u}.conv"), 1, [1, 1, 1, 1])?;
        }
        o += 1;
    }
    h = b.group_norm(&h, &p("out.0"), 32, 1e-5)?;
    h = b.silu(&h);
    h = b.conv(&h, &p("out.2"), 1, [1, 1, 1, 1])?;
    let cout = b.shape(&p("out.2.weight"))?.first().copied().unwrap_or(4) as i64;
    b.g.output(&h, "out_sample", b.g.elem, vec![dyn_dim("batch"), Dim::Fixed(cout), dyn_dim("height"), dyn_dim("width")]);
    Ok(())
}

// ---------------------------------------------------------------------------------------------
// VAE

fn vae_res(b: &mut Builder, x: &str, p: &str) -> Result<String> {
    let h = b.group_norm(x, &format!("{p}.norm1"), 32, 1e-6)?;
    let h = b.silu(&h);
    let h = b.conv(&h, &format!("{p}.conv1"), 1, [1, 1, 1, 1])?;
    let h = b.group_norm(&h, &format!("{p}.norm2"), 32, 1e-6)?;
    let h = b.silu(&h);
    let h = b.conv(&h, &format!("{p}.conv2"), 1, [1, 1, 1, 1])?;
    let skip = if b.has(&format!("{p}.nin_shortcut.weight")) { b.conv(x, &format!("{p}.nin_shortcut"), 1, [0, 0, 0, 0])? } else { x.to_string() };
    Ok(b.op("Add", &[&skip, &h]))
}

/// The VAE's single-head attention over all positions (1×1 convolutions or linear layers).
fn vae_attn(b: &mut Builder, x: &str, p: &str) -> Result<String> {
    let shape = b.op("Shape", &[x]);
    let h = b.group_norm(x, &format!("{p}.norm"), 32, 1e-6)?;
    let t = b.tokens_of(&h);
    let lin = |b: &mut Builder, x: &str, name: &str| -> Result<String> {
        let (s, v) = b.ck.read(&format!("{p}.{name}.weight"))?;
        let s2 = [s.first().copied().unwrap_or(0), s.get(1).copied().unwrap_or(0)];
        b.linear_w(x, (&s2, v), Some(&format!("{p}.{name}.bias")))
    };
    let c = b.shape(&format!("{p}.q.weight"))?.first().copied().unwrap_or(0) as i64;
    let q = lin(b, &t, "q")?;
    let k = lin(b, &t, "k")?;
    let v = lin(b, &t, "v")?;
    let o = b.attention(&q, &k, &v, 1, c, None)?;
    let o = lin(b, &o, "proj_out")?;
    let o = b.image_of_tokens(&o, &shape);
    Ok(b.op("Add", &[x, &o]))
}

fn vae_mid(b: &mut Builder, x: &str, p: &str) -> Result<String> {
    let h = vae_res(b, x, &format!("{p}.mid.block_1"))?;
    let h = vae_attn(b, &h, &format!("{p}.mid.attn_1"))?;
    vae_res(b, &h, &format!("{p}.mid.block_2"))
}

/// The encoder: `sample` (RGB −1…1) → `latent_sample`: the mean and log-variance (8 channels).
pub fn vae_encoder(b: &mut Builder) -> Result<()> {
    let p = |s: &str| format!("{VAE}encoder.{s}");
    let x = b.g.input("sample", b.g.elem, vec![dyn_dim("batch"), Dim::Fixed(3), dyn_dim("height"), dyn_dim("width")]);
    let mut h = b.conv(&x, &p("conv_in"), 1, [1, 1, 1, 1])?;
    let mut i = 0;
    while b.has(&p(&format!("down.{i}.block.0.norm1.weight"))) {
        let mut j = 0;
        while b.has(&p(&format!("down.{i}.block.{j}.norm1.weight"))) {
            h = vae_res(b, &h, &p(&format!("down.{i}.block.{j}")))?;
            j += 1;
        }
        if b.has(&p(&format!("down.{i}.downsample.conv.weight"))) {
            // Padded on the right and bottom only, then a stride-2 convolution.
            h = b.conv(&h, &p(&format!("down.{i}.downsample.conv")), 2, [0, 0, 1, 1])?;
        }
        i += 1;
    }
    h = vae_mid(b, &h, &format!("{VAE}encoder"))?;
    h = b.group_norm(&h, &p("norm_out"), 32, 1e-6)?;
    h = b.silu(&h);
    h = b.conv(&h, &p("conv_out"), 1, [1, 1, 1, 1])?;
    h = b.conv(&h, &format!("{VAE}quant_conv"), 1, [0, 0, 0, 0])?;
    let c = b.shape(&format!("{VAE}quant_conv.weight"))?.first().copied().unwrap_or(8) as i64;
    b.g.output(&h, "latent_sample", b.g.elem, vec![dyn_dim("batch"), Dim::Fixed(c), dyn_dim("lh"), dyn_dim("lw")]);
    Ok(())
}

/// The decoder: `latent_sample` (latents ÷ scaling factor) → `sample` (RGB −1…1).
pub fn vae_decoder(b: &mut Builder) -> Result<()> {
    let p = |s: &str| format!("{VAE}decoder.{s}");
    let c = b.shape(&format!("{VAE}post_quant_conv.weight"))?.get(1).copied().unwrap_or(4) as i64;
    let z = b.g.input("latent_sample", b.g.elem, vec![dyn_dim("batch"), Dim::Fixed(c), dyn_dim("lh"), dyn_dim("lw")]);
    let mut h = b.conv(&z, &format!("{VAE}post_quant_conv"), 1, [0, 0, 0, 0])?;
    h = b.conv(&h, &p("conv_in"), 1, [1, 1, 1, 1])?;
    h = vae_mid(b, &h, &format!("{VAE}decoder"))?;
    let mut levels = 0;
    while b.has(&p(&format!("up.{levels}.block.0.norm1.weight"))) {
        levels += 1;
    }
    for i in (0..levels).rev() {
        let mut j = 0;
        while b.has(&p(&format!("up.{i}.block.{j}.norm1.weight"))) {
            h = vae_res(b, &h, &p(&format!("up.{i}.block.{j}")))?;
            j += 1;
        }
        if b.has(&p(&format!("up.{i}.upsample.conv.weight"))) {
            h = b.upsample2(&h);
            h = b.conv(&h, &p(&format!("up.{i}.upsample.conv")), 1, [1, 1, 1, 1])?;
        }
    }
    h = b.group_norm(&h, &p("norm_out"), 32, 1e-6)?;
    h = b.silu(&h);
    h = b.conv(&h, &p("conv_out"), 1, [1, 1, 1, 1])?;
    b.g.output(&h, "sample", b.g.elem, vec![dyn_dim("batch"), Dim::Fixed(3), dyn_dim("height"), dyn_dim("width")]);
    Ok(())
}

// ---------------------------------------------------------------------------------------------
// Text encoder

/// Which text encoder the checkpoint has.
#[derive(Clone, Debug, PartialEq)]
pub enum Text {
    /// OpenAI CLIP in Hugging Face names under this prefix (SD 1.x), quick-GELU.
    Clip { prefix: String },
    /// OpenCLIP (SD 2.x): its last layer is left out, GELU.
    OpenClip { prefix: String },
}

/// What a text encoder gives.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum TextOut {
    /// `last_hidden_state`: the layers (all, or all but the last: SD 2), then the final norm.
    Final { skip_last: bool },
    /// SDXL: `hidden_states`, the next-to-last layer's output (no final norm); with `pooled`,
    /// also `text_embeds`: the end token's final, normed state through the text projection.
    Penultimate { pooled: bool },
}

/// The text encoder: `input_ids` `[B, 77]` (int32) → `last_hidden_state` `[B, 77, D]` (or the
/// SDXL outputs of [`TextOut::Penultimate`]).
pub fn text_encoder(b: &mut Builder, text: &Text, heads: i64, length: usize, out: TextOut) -> Result<()> {
    let ids = b.g.input("input_ids", INT32, vec![dyn_dim("batch"), Dim::Fixed(length as i64)]);
    let ids64 = b.op_a("Cast", &[&ids], vec![("to", Attr::Int(7))]);
    let (tok, pos, layers, final_ln, open) = match text {
        Text::Clip { prefix } => (
            format!("{prefix}embeddings.token_embedding.weight"),
            format!("{prefix}embeddings.position_embedding.weight"),
            format!("{prefix}encoder.layers."),
            format!("{prefix}final_layer_norm"),
            false,
        ),
        Text::OpenClip { prefix } => {
            (format!("{prefix}token_embedding.weight"), format!("{prefix}positional_embedding"), format!("{prefix}transformer.resblocks."), format!("{prefix}ln_final"), true)
        }
    };
    let dim = b.shape(&tok)?.get(1).copied().unwrap_or(768);
    let table = b.weight(&tok)?;
    let mut h = b.op_a("Gather", &[&table, &ids64], vec![("axis", Attr::Int(0))]);
    let (_, posv) = b.ck.read(&pos)?;
    let posv: Vec<f32> = posv.into_iter().take(length * dim).collect();
    let pos = b.values(&[length, dim], &posv)?;
    h = b.op("Add", &[&h, &pos]);
    // Causal mask: a token sees itself and those before it.
    let mask: Vec<f32> = (0..length * length).map(|i| if i % length > i / length { -65504.0 } else { 0.0 }).collect();
    let mask = b.values(&[length, length], &mask)?;
    let mut n = 0usize;
    while b.has(&format!("{layers}{n}.{}", if open { "ln_1.weight" } else { "layer_norm1.weight" })) {
        n += 1;
    }
    let used = match out {
        // SD 2 takes OpenCLIP's next-to-last layer.
        TextOut::Final { skip_last } => if skip_last { n.saturating_sub(1) } else { n },
        TextOut::Penultimate { pooled } => if pooled { n } else { n.saturating_sub(1) },
    };
    let mut penultimate = None;
    for l in 0..used {
        if l + 1 == n {
            penultimate = Some(h.clone());
        }
        let p = format!("{layers}{l}");
        let r = h.clone();
        let x = b.layer_norm(&h, &format!("{p}.{}", if open { "ln_1" } else { "layer_norm1" }), 1e-5)?;
        let (q, k, v) = if open {
            let (s, w) = b.ck.read(&format!("{p}.attn.in_proj_weight"))?;
            let (_, bias) = b.ck.read(&format!("{p}.attn.in_proj_bias"))?;
            let c = s.get(1).copied().unwrap_or(dim);
            let mut out = Vec::new();
            for part in 0..3 {
                let wv = w.get(part * c * c..(part + 1) * c * c).unwrap_or(&[]).to_vec();
                let bv = bias.get(part * c..(part + 1) * c).unwrap_or(&[]).to_vec();
                let y = b.linear_w(&x, (&[c, c], wv), None)?;
                let bb = b.values(&[c], &bv)?;
                out.push(b.op("Add", &[&y, &bb]));
            }
            let mut it = out.into_iter();
            (it.next().unwrap_or_default(), it.next().unwrap_or_default(), it.next().unwrap_or_default())
        } else {
            (b.linear(&x, &format!("{p}.self_attn.q_proj"))?, b.linear(&x, &format!("{p}.self_attn.k_proj"))?, b.linear(&x, &format!("{p}.self_attn.v_proj"))?)
        };
        let a = b.attention(&q, &k, &v, heads, dim as i64, Some(&mask))?;
        let a = b.linear(&a, &format!("{p}.{}", if open { "attn.out_proj" } else { "self_attn.out_proj" }))?;
        h = b.op("Add", &[&r, &a]);
        let r = h.clone();
        let x = b.layer_norm(&h, &format!("{p}.{}", if open { "ln_2" } else { "layer_norm2" }), 1e-5)?;
        let x = b.linear(&x, &format!("{p}.{}", if open { "mlp.c_fc" } else { "mlp.fc1" }))?;
        let x = if open { b.gelu(&x)? } else { b.quick_gelu(&x)? };
        let x = b.linear(&x, &format!("{p}.{}", if open { "mlp.c_proj" } else { "mlp.fc2" }))?;
        h = b.op("Add", &[&r, &x]);
    }
    let dims = vec![dyn_dim("batch"), Dim::Fixed(length as i64), Dim::Fixed(dim as i64)];
    match out {
        TextOut::Final { .. } => {
            h = b.layer_norm(&h, &final_ln, 1e-5)?;
            b.g.output(&h, "last_hidden_state", b.g.elem, dims);
        }
        TextOut::Penultimate { pooled: false } => b.g.output(&h, "hidden_states", b.g.elem, dims),
        TextOut::Penultimate { pooled: true } => {
            let pen = penultimate.unwrap_or_else(|| h.clone());
            b.g.output(&pen, "hidden_states", b.g.elem, dims);
            // The end token's state: its one-hot row picks it (the padding is never the end token).
            let fin = b.layer_norm(&h, &final_ln, 1e-5)?;
            let eos = b.i64s(&[49_407]);
            let hit = b.op("Equal", &[&ids64, &eos]);
            let hit = b.op_a("Cast", &[&hit], vec![("to", Attr::Int(i64::from(b.g.elem)))]);
            let ax = b.i64s(&[-1]);
            let hit = b.op("Unsqueeze", &[&hit, &ax]);
            let picked = b.op("Mul", &[&fin, &hit]);
            let ax1 = b.i64s(&[1]);
            let pooled = b.op_a("ReduceSum", &[&picked, &ax1], vec![("keepdims", Attr::Int(0))]);
            let prefix = match text {
                Text::OpenClip { prefix } | Text::Clip { prefix } => prefix.clone(),
            };
            let proj_key = format!("{prefix}text_projection");
            let (shape, v) = b.ck.read(&proj_key)?;
            // OpenCLIP stores the projection as [width, projection]: x · P.
            let p = b.values(&shape, &v)?;
            let emb = b.op("MatMul", &[&pooled, &p]);
            let pd = shape.get(1).copied().unwrap_or(dim) as i64;
            b.g.output(&emb, "text_embeds", b.g.elem, vec![dyn_dim("batch"), Dim::Fixed(pd)]);
        }
    }
    Ok(())
}
