"""Writes tiny Stable Diffusion checkpoints in the original (LDM) single-file layout, and what
diffusers computes with them, for the checkpoint converter's tests (crates/ml/src/convert).

Needs torch, diffusers and transformers (run from this folder: python3 make_tiny_checkpoint.py).
Each model is made in diffusers with random weights, saved under the original names (the names
are checked by loading them back through diffusers' own single-file converters), and run on fixed
inputs; the outputs go to <name>.expected.json beside <name>.safetensors.
"""
import json
import torch
from safetensors.torch import save_file
from diffusers import UNet2DConditionModel, AutoencoderKL
from diffusers.loaders import single_file_utils as sfu
from transformers import CLIPTextConfig, CLIPTextModel, CLIPTextModelWithProjection

torch.manual_seed(0)


def randomize(model):
    # Norm weights away from 1 and 0 so every scale and shift is exercised.
    with torch.no_grad():
        for n, p in model.named_parameters():
            p.copy_(torch.randn_like(p) * (0.2 if p.dim() > 1 else 0.1) + (1.0 if n.endswith("norm.weight") or ".norm" in n and n.endswith("weight") else 0.0))
    return model


def unet_to_ldm(sd, levels, layers, attn_down, attn_up):
    """diffusers UNet names -> LDM names (the inverse of convert_ldm_unet_checkpoint)."""
    out = {}
    res = [("norm1", "in_layers.0"), ("conv1", "in_layers.2"), ("norm2", "out_layers.0"), ("conv2", "out_layers.3"),
           ("time_emb_proj", "emb_layers.1"), ("conv_shortcut", "skip_connection")]
    def put(prefix_new, prefix_old, key, v, resnet):
        rest = key[len(prefix_old):]
        if resnet:
            for a, b in res:
                if rest.startswith(a + "."):
                    rest = b + rest[len(a):]
        out[prefix_new + rest] = v
    for k, v in sd.items():
        if k.startswith("add_embedding.linear_1."):
            out["label_emb.0.0." + k.split(".")[-1]] = v
        elif k.startswith("add_embedding.linear_2."):
            out["label_emb.0.2." + k.split(".")[-1]] = v
        elif k.startswith("time_embedding.linear_1."):
            out["time_embed.0." + k.split(".")[-1]] = v
        elif k.startswith("time_embedding.linear_2."):
            out["time_embed.2." + k.split(".")[-1]] = v
        elif k.startswith("conv_in."):
            out["input_blocks.0.0." + k.split(".")[-1]] = v
        elif k.startswith("conv_norm_out."):
            out["out.0." + k.split(".")[-1]] = v
        elif k.startswith("conv_out."):
            out["out.2." + k.split(".")[-1]] = v
        elif k.startswith("down_blocks."):
            p = k.split(".")
            i = int(p[1])
            if p[2] == "resnets":
                j = int(p[3]); put(f"input_blocks.{(layers+1)*i+j+1}.0.", f"down_blocks.{i}.resnets.{j}.", k, v, True)
            elif p[2] == "attentions":
                j = int(p[3]); put(f"input_blocks.{(layers+1)*i+j+1}.1.", f"down_blocks.{i}.attentions.{j}.", k, v, False)
            elif p[2] == "downsamplers":
                out[f"input_blocks.{(layers+1)*(i+1)}.0.op." + p[-1]] = v
        elif k.startswith("mid_block."):
            p = k.split(".")
            j = int(p[2])
            if p[1] == "resnets":
                put(f"middle_block.{2*j}.", f"mid_block.resnets.{j}.", k, v, True)
            else:
                put("middle_block.1.", "mid_block.attentions.0.", k, v, False)
        elif k.startswith("up_blocks."):
            p = k.split(".")
            i = int(p[1])
            if p[2] == "resnets":
                j = int(p[3]); put(f"output_blocks.{(layers+1)*i+j}.0.", f"up_blocks.{i}.resnets.{j}.", k, v, True)
            elif p[2] == "attentions":
                j = int(p[3]); put(f"output_blocks.{(layers+1)*i+j}.1.", f"up_blocks.{i}.attentions.{j}.", k, v, False)
            elif p[2] == "upsamplers":
                out[f"output_blocks.{(layers+1)*i+layers}.{2 if attn_up[i] else 1}.conv." + p[-1]] = v
        else:
            raise KeyError(k)
    return {"model.diffusion_model." + k: v for k, v in out.items()}


def vae_to_ldm(sd, levels):
    out = {}
    def res(rest):
        return rest.replace("conv_shortcut", "nin_shortcut")
    for k, v in sd.items():
        p = k.split(".")
        if k.startswith("encoder.down_blocks."):
            i = int(p[2])
            if p[3] == "resnets":
                out[f"encoder.down.{i}.block.{p[4]}." + res(".".join(p[5:]))] = v
            else:
                out[f"encoder.down.{i}.downsample.conv." + p[-1]] = v
        elif k.startswith("decoder.up_blocks."):
            i = int(p[2]); li = levels - 1 - i
            if p[3] == "resnets":
                out[f"decoder.up.{li}.block.{p[4]}." + res(".".join(p[5:]))] = v
            else:
                out[f"decoder.up.{li}.upsample.conv." + p[-1]] = v
        elif ".mid_block.resnets." in k:
            which = p[0]; j = int(p[3])
            out[f"{which}.mid.block_{j+1}." + res(".".join(p[4:]))] = v
        elif ".mid_block.attentions.0." in k:
            which = p[0]; rest = ".".join(p[4:])
            m = {"group_norm": "norm", "to_q": "q", "to_k": "k", "to_v": "v", "to_out.0": "proj_out"}
            for a, b in m.items():
                if rest.startswith(a + "."):
                    rest = b + rest[len(a):]
            if v.dim() == 2:
                v = v[:, :, None, None]  # LDM's attention projections are 1x1 convolutions
            out[f"{which}.mid.attn_1." + rest] = v
        else:
            out[k.replace("conv_norm_out", "norm_out")] = v
    return {"first_stage_model." + k: v for k, v in out.items()}


def clip_to_openclip(sd, extra_layer, projection=None):
    """HF CLIP names -> OpenCLIP (SD 2) names, with one more transformer layer that SD 2 leaves out."""
    out = {}
    layers = {}
    for k, v in sd.items():
        if k == "text_model.embeddings.token_embedding.weight":
            out["token_embedding.weight"] = v
        elif k == "text_model.embeddings.position_embedding.weight":
            out["positional_embedding"] = v
        elif k.startswith("text_model.final_layer_norm."):
            out["ln_final." + k.split(".")[-1]] = v
        elif k.startswith("text_model.encoder.layers."):
            p = k.split(".")
            i = p[3]; rest = ".".join(p[4:])
            layers.setdefault(i, {})[rest] = v
    for i, l in layers.items():
        pre = f"transformer.resblocks.{i}."
        out[pre + "attn.in_proj_weight"] = torch.cat([l["self_attn.q_proj.weight"], l["self_attn.k_proj.weight"], l["self_attn.v_proj.weight"]])
        out[pre + "attn.in_proj_bias"] = torch.cat([l["self_attn.q_proj.bias"], l["self_attn.k_proj.bias"], l["self_attn.v_proj.bias"]])
        out[pre + "attn.out_proj.weight"] = l["self_attn.out_proj.weight"]
        out[pre + "attn.out_proj.bias"] = l["self_attn.out_proj.bias"]
        for a, b in [("layer_norm1", "ln_1"), ("layer_norm2", "ln_2"), ("mlp.fc1", "mlp.c_fc"), ("mlp.fc2", "mlp.c_proj")]:
            out[pre + f"{b}.weight"] = l[f"{a}.weight"]
            out[pre + f"{b}.bias"] = l[f"{a}.bias"]
    n = len(layers)
    for k, v in extra_layer.items():
        out[f"transformer.resblocks.{n}." + k] = v
    out["text_projection"] = torch.randn(32, 32) if projection is None else projection
    return out


def flat(t):
    return [round(float(x), 6) for x in t.detach().double().flatten().tolist()]


def make(name, sd2):
    heads = (2, 4) if sd2 else 4
    unet = randomize(UNet2DConditionModel(
        sample_size=8, in_channels=9, out_channels=4, block_out_channels=(32, 64), layers_per_block=2,
        down_block_types=("CrossAttnDownBlock2D", "DownBlock2D"), up_block_types=("UpBlock2D", "CrossAttnUpBlock2D"),
        cross_attention_dim=32, attention_head_dim=heads, use_linear_projection=sd2, norm_num_groups=32)).eval()
    vae = randomize(AutoencoderKL(
        in_channels=3, out_channels=3, block_out_channels=(32, 32, 32, 64), layers_per_block=1, latent_channels=4,
        down_block_types=("DownEncoderBlock2D",) * 4, up_block_types=("UpDecoderBlock2D",) * 4, norm_num_groups=32)).eval()
    clip = randomize(CLIPTextModel(CLIPTextConfig(
        # SD 1's has CLIP's whole vocabulary, so real prompts can go through it end to end.
        vocab_size=1000 if sd2 else 49408, hidden_size=32, intermediate_size=64, num_hidden_layers=2, num_attention_heads=4,
        max_position_embeddings=77, hidden_act="gelu" if sd2 else "quick_gelu"))).eval()

    ckpt = {}
    ckpt.update(unet_to_ldm(unet.state_dict(), 2, 2, (True, False), (False, True)))
    ckpt.update(vae_to_ldm(vae.state_dict(), 4))
    if sd2:
        extra = {k: torch.randn(v.shape) for k, v in clip_to_openclip(clip.state_dict(), {}).items() if ".resblocks.0." in k}
        extra = {k.split(".resblocks.0.")[1]: v for k, v in extra.items()}
        ckpt.update({"cond_stage_model.model." + k: v for k, v in clip_to_openclip(clip.state_dict(), extra).items()})
    else:
        ckpt.update({"cond_stage_model.transformer." + k: v for k, v in clip.state_dict().items() if "position_ids" not in k})
    # Half precision, as checkpoints usually are (one block in single, to read both).
    ckpt = {k: v.contiguous().to(torch.float32 if "time_embed" in k else torch.float16) for k, v in ckpt.items()}

    # The names are right: diffusers' own converters give back the same weights.
    full = {k: v.float() for k, v in ckpt.items()}
    back = sfu.convert_ldm_unet_checkpoint(full, dict(unet.config))
    for k, v in unet.state_dict().items():
        assert torch.allclose(back[k].float(), v, atol=1e-2), k
    vback = sfu.convert_ldm_vae_checkpoint({k[len("first_stage_model."):]: v for k, v in full.items() if k.startswith("first_stage_model.")}, dict(vae.config))
    for k, v in vae.state_dict().items():
        assert torch.allclose(vback[k].float().reshape(v.shape), v, atol=1e-2), k
    if sd2:
        cback = sfu.convert_open_clip_checkpoint(clip, {k: v for k, v in full.items() if ".resblocks.2." not in k})
        for k, v in clip.state_dict().items():
            if "position_ids" in k:
                continue
            assert torch.allclose(cback[k].float(), v, atol=1e-2), k
    # The models in diffusers now hold exactly the checkpoint's (rounded) weights.
    with torch.no_grad():
        for k, v in unet.state_dict().items():
            v.copy_(back[k].float())
        for k, v in vae.state_dict().items():
            v.copy_(vback[k].float().reshape(v.shape))
        if sd2:
            for k, v in clip.state_dict().items():
                if "position_ids" not in k:
                    v.copy_(cback[k].float())
        else:
            for k, v in clip.state_dict().items():
                if "position_ids" not in k:
                    v.copy_(full["cond_stage_model.transformer." + k])

    save_file(ckpt, f"{name}.safetensors")

    g = torch.Generator().manual_seed(1)
    sample = torch.randn(2, 9, 8, 8, generator=g)
    t = torch.tensor([401.0])
    ehs = torch.randn(2, 5, 32, generator=g)
    img = torch.rand(1, 3, 32, 32, generator=g) * 2 - 1
    z = torch.randn(1, 4, 4, 4, generator=g)
    ids = torch.tensor([[0, 5, 17, 999, 1, 1, 1]])
    with torch.no_grad():
        u = unet(sample, t, encoder_hidden_states=ehs).sample
        moments = vae.quant_conv(vae.encoder(img))
        dec = vae.decode(z).sample
        hid = clip(ids).last_hidden_state
    json.dump({
        "sd2": sd2,
        "heads": list(heads) if sd2 else [heads, heads],
        "unet": {"sample": flat(sample), "timestep": 401.0, "ehs": flat(ehs), "out": flat(u)},
        "vae_encoder": {"image": flat(img), "mean": flat(moments[:, :4])},
        "vae_decoder": {"z": flat(z), "out": flat(dec)},
        "text": {"ids": ids.flatten().tolist(), "out": flat(hid)},
    }, open(f"{name}.expected.json", "w"))
    print(name, "ok:", len(ckpt), "tensors")


def make_xl(name):
    """An SDXL-style checkpoint: two text encoders (CLIP and OpenCLIP with a projection), the
    UNet's added time and pooled-text embedding, deeper transformers, no attention on top."""
    unet = randomize(UNet2DConditionModel(
        sample_size=8, in_channels=4, out_channels=4, block_out_channels=(32, 64), layers_per_block=2,
        down_block_types=("DownBlock2D", "CrossAttnDownBlock2D"), up_block_types=("CrossAttnUpBlock2D", "UpBlock2D"),
        cross_attention_dim=64, attention_head_dim=(2, 4), transformer_layers_per_block=(1, 2), use_linear_projection=True,
        addition_embed_type="text_time", addition_time_embed_dim=8, projection_class_embeddings_input_dim=32 + 6 * 8,
        norm_num_groups=32)).eval()
    vae = randomize(AutoencoderKL(
        in_channels=3, out_channels=3, block_out_channels=(32, 32, 32, 64), layers_per_block=1, latent_channels=4,
        down_block_types=("DownEncoderBlock2D",) * 4, up_block_types=("UpDecoderBlock2D",) * 4, norm_num_groups=32)).eval()
    cfg = dict(vocab_size=49408, hidden_size=32, intermediate_size=64, num_hidden_layers=3, num_attention_heads=4, max_position_embeddings=77)
    clip_l = randomize(CLIPTextModel(CLIPTextConfig(hidden_act="quick_gelu", **cfg))).eval()
    clip_g = randomize(CLIPTextModelWithProjection(CLIPTextConfig(hidden_act="gelu", projection_dim=32, **cfg))).eval()

    ckpt = {}
    ckpt.update(unet_to_ldm(unet.state_dict(), 2, 2, (False, True), (True, False)))
    ckpt.update(vae_to_ldm(vae.state_dict(), 4))
    ckpt.update({"conditioner.embedders.0.transformer." + k: v for k, v in clip_l.state_dict().items() if "position_ids" not in k})
    g = {k: v for k, v in clip_g.state_dict().items() if not k.startswith("text_projection")}
    ckpt.update({"conditioner.embedders.1.model." + k: v for k, v in clip_to_openclip(g, {}, clip_g.text_projection.weight.T.contiguous()).items()})
    ckpt = {k: v.contiguous().to(torch.float16) for k, v in ckpt.items()}
    full = {k: v.float() for k, v in ckpt.items()}
    back = sfu.convert_ldm_unet_checkpoint(full, dict(unet.config))
    vback = sfu.convert_ldm_vae_checkpoint({k[len("first_stage_model."):]: v for k, v in full.items() if k.startswith("first_stage_model.")}, dict(vae.config))
    lback = sfu.convert_ldm_clip_checkpoint(full, remove_prefix="conditioner.embedders.0.transformer.")
    gback = sfu.convert_open_clip_checkpoint(clip_g, full, prefix="conditioner.embedders.1.model.")
    with torch.no_grad():
        for model, b in [(unet, back), (vae, vback), (clip_l, lback), (clip_g, gback)]:
            for k, v in model.state_dict().items():
                if "position_ids" in k:
                    continue
                assert k in b, k
                assert torch.allclose(b[k].float().reshape(v.shape), v, atol=1e-2), k
                v.copy_(b[k].float().reshape(v.shape))
    save_file(ckpt, f"{name}.safetensors")

    gen = torch.Generator().manual_seed(2)
    sample = torch.randn(2, 4, 8, 8, generator=gen)
    ehs = torch.randn(2, 5, 64, generator=gen)
    text_embeds = torch.randn(2, 32, generator=gen)
    time_ids = torch.tensor([[512.0, 768.0, 0.0, 0.0, 512.0, 768.0]] * 2)
    ids = torch.tensor([[49406, 320, 1125, 49407, 0, 0, 0]])
    with torch.no_grad():
        u = unet(sample, torch.tensor([401.0]), encoder_hidden_states=ehs, added_cond_kwargs={"text_embeds": text_embeds, "time_ids": time_ids}).sample
        hl = clip_l(ids, output_hidden_states=True).hidden_states[-2]
        og = clip_g(ids, output_hidden_states=True)
    json.dump({
        "unet": {"sample": flat(sample), "timestep": 401.0, "ehs": flat(ehs), "text_embeds": flat(text_embeds), "time_ids": flat(time_ids), "out": flat(u)},
        "text": {"ids": ids.flatten().tolist(), "l": flat(hl), "g": flat(og.hidden_states[-2]), "pooled": flat(og.text_embeds)},
    }, open(f"{name}.expected.json", "w"))
    print(name, "ok:", len(ckpt), "tensors")


make("tiny_sd1_ckpt", False)
make("tiny_sd2_ckpt", True)
make_xl("tiny_sdxl_ckpt")
