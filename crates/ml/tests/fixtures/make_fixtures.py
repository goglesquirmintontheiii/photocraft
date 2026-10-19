"""Writes the tiny ONNX models the runtime tests load (run from this folder: python3 make_fixtures.py).

They have the inputs and outputs of the real models PhotoCraft runs, but compute something trivial
and checkable, so the tests exercise the ONNX Runtime glue (names, element types, shapes, f16)
without downloading hundreds of megabytes.
"""
import onnx
from onnx import TensorProto, helper

def save(graph, name):
    m = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    m.ir_version = 8
    onnx.checker.check_model(m)
    onnx.save(m, name)

# LaMa-like: image [1,3,64,64] 0-1, mask [1,1,64,64] (1 = hole) -> output 0-255, the hole white.
image = helper.make_tensor_value_info("image", TensorProto.FLOAT, [1, 3, 64, 64])
mask = helper.make_tensor_value_info("mask", TensorProto.FLOAT, [1, 1, 64, 64])
out = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 3, 64, 64])
one = helper.make_tensor("one", TensorProto.FLOAT, [], [1.0])
k255 = helper.make_tensor("k255", TensorProto.FLOAT, [], [255.0])
nodes = [
    helper.make_node("Sub", ["one", "mask"], ["keep"]),
    helper.make_node("Mul", ["image", "keep"], ["kept"]),
    helper.make_node("Add", ["kept", "mask"], ["filled"]),
    helper.make_node("Mul", ["filled", "k255"], ["output"]),
]
save(helper.make_graph(nodes, "lama_like", [image, mask], [out], [one, k255]), "lama_like.onnx")

# MI-GAN-like: u8 image and mask (255 = keep), any size -> u8 result: kept pixels, holes black.
image = helper.make_tensor_value_info("image", TensorProto.UINT8, [1, 3, "h", "w"])
mask = helper.make_tensor_value_info("mask", TensorProto.UINT8, [1, 1, "h", "w"])
out = helper.make_tensor_value_info("result", TensorProto.UINT8, [1, 3, "h", "w"])
k = helper.make_tensor("k", TensorProto.FLOAT, [], [1.0 / 255.0])
nodes = [
    helper.make_node("Cast", ["image"], ["fi"], to=TensorProto.FLOAT),
    helper.make_node("Cast", ["mask"], ["fm"], to=TensorProto.FLOAT),
    helper.make_node("Mul", ["fm", "k"], ["m01"]),
    helper.make_node("Mul", ["fi", "m01"], ["f"]),
    helper.make_node("Cast", ["f"], ["result"], to=TensorProto.UINT8),
]
save(helper.make_graph(nodes, "migan_like", [image, mask], [out], [k]), "migan_like.onnx")

# Matting-like in float16: input [1,3,32,32] -> logits [1,1,32,32] = 20 * (mean channel - 0).
x = helper.make_tensor_value_info("input_image", TensorProto.FLOAT16, [1, 3, 32, 32])
y = helper.make_tensor_value_info("output_image", TensorProto.FLOAT16, [1, 1, 32, 32])
k = helper.make_tensor("k", TensorProto.FLOAT16, [], [20.0])
nodes = [
    helper.make_node("ReduceMean", ["input_image"], ["m"], axes=[1], keepdims=1),
    helper.make_node("Mul", ["m", "k"], ["output_image"]),
]
g = helper.make_graph(nodes, "matte_like", [x], [y], [k])
m = helper.make_model(g, opset_imports=[helper.make_opsetid("", 17)])
m.ir_version = 8
onnx.checker.check_model(m)
onnx.save(m, "matte_like_fp16.onnx")

# A tiny Stable Diffusion inpainting pipeline in the diffusers ONNX layout, float16 like the
# recommended ones: int32 token ids, a float16 timestep, 9-channel UNet. It computes nothing
# meaningful; it checks the pipeline's names, types and shapes through the real runtime.
import json, os
F16 = TensorProto.FLOAT16
os.makedirs("tiny_sd/text_encoder", exist_ok=True)
for d in ["unet", "vae_encoder", "vae_decoder", "tokenizer", "scheduler"]:
    os.makedirs("tiny_sd/" + d, exist_ok=True)

ids = helper.make_tensor_value_info("input_ids", TensorProto.INT32, ["batch", "sequence"])
hid = helper.make_tensor_value_info("last_hidden_state", F16, ["batch", "sequence", 8])
ones = helper.make_tensor("ones", F16, [1, 1, 8], [0.001] * 8)
ax = helper.make_tensor("ax", TensorProto.INT64, [1], [2])
nodes = [
    helper.make_node("Cast", ["input_ids"], ["f"], to=F16),
    helper.make_node("Unsqueeze", ["f", "ax"], ["u"]),
    helper.make_node("Mul", ["u", "ones"], ["last_hidden_state"]),
]
save(helper.make_graph(nodes, "text_encoder", [ids], [hid], [ones, ax]), "tiny_sd/text_encoder/model.onnx")

sample = helper.make_tensor_value_info("sample", F16, ["batch", 9, "height", "width"])
ts = helper.make_tensor_value_info("timestep", F16, ["steps"])
ehs = helper.make_tensor_value_info("encoder_hidden_states", F16, ["batch", "sequence", 8])
out = helper.make_tensor_value_info("out_sample", F16, ["batch", 4, "height", "width"])
starts = helper.make_tensor("starts", TensorProto.INT64, [1], [0])
ends = helper.make_tensor("ends", TensorProto.INT64, [1], [4])
axes = helper.make_tensor("axes", TensorProto.INT64, [1], [1])
k = helper.make_tensor("k", F16, [], [0.1])
z = helper.make_tensor("z", F16, [], [0.0])
shp = helper.make_tensor("shp", TensorProto.INT64, [4], [-1, 1, 1, 1])
nodes = [
    helper.make_node("Slice", ["sample", "starts", "ends", "axes"], ["lat"]),
    helper.make_node("Mul", ["lat", "k"], ["a"]),
    helper.make_node("ReduceMean", ["encoder_hidden_states"], ["e"], axes=[1, 2], keepdims=1),
    helper.make_node("Unsqueeze", ["e", "axes"], ["e4"]),
    helper.make_node("Mul", ["e4", "z"], ["e0"]),
    helper.make_node("Reshape", ["timestep", "shp"], ["t4"]),
    helper.make_node("Mul", ["t4", "z"], ["t0"]),
    helper.make_node("Add", ["a", "e0"], ["b"]),
    helper.make_node("Add", ["b", "t0"], ["out_sample"]),
]
save(helper.make_graph(nodes, "unet", [sample, ts, ehs], [out], [starts, ends, axes, k, z, shp]), "tiny_sd/unet/model.onnx")

img = helper.make_tensor_value_info("sample", F16, [1, 3, "height", "width"])
lat = helper.make_tensor_value_info("latent_sample", F16, [1, 4, "lh", "lw"])
s0 = helper.make_tensor("s0", TensorProto.INT64, [1], [0])
e1 = helper.make_tensor("e1", TensorProto.INT64, [1], [1])
a1 = helper.make_tensor("a1", TensorProto.INT64, [1], [1])
nodes = [
    helper.make_node("AveragePool", ["sample"], ["p"], kernel_shape=[8, 8], strides=[8, 8]),
    helper.make_node("Slice", ["p", "s0", "e1", "a1"], ["c0"]),
    helper.make_node("Concat", ["p", "c0"], ["latent_sample"], axis=1),
]
save(helper.make_graph(nodes, "vae_encoder", [img], [lat], [s0, e1, a1]), "tiny_sd/vae_encoder/model.onnx")

lat = helper.make_tensor_value_info("latent_sample", F16, [1, 4, "lh", "lw"])
img = helper.make_tensor_value_info("sample", F16, [1, 3, "height", "width"])
e3 = helper.make_tensor("e3", TensorProto.INT64, [1], [3])
scales = helper.make_tensor("scales", TensorProto.FLOAT, [4], [1.0, 1.0, 8.0, 8.0])
nodes = [
    helper.make_node("Slice", ["latent_sample", "s0", "e3", "a1"], ["rgb"]),
    helper.make_node("Resize", ["rgb", "", "scales"], ["sample"], mode="nearest"),
]
save(helper.make_graph(nodes, "vae_decoder", [lat], [img], [s0, e3, a1, scales]), "tiny_sd/vae_decoder/model.onnx")

vocab = {t: i for i, t in enumerate(["<|startoftext|>", "<|endoftext|>", "c", "a", "t</w>", "ca", "cat</w>"])}
json.dump(vocab, open("tiny_sd/tokenizer/vocab.json", "w"))
open("tiny_sd/tokenizer/merges.txt", "w").write("#version: 0.2\nc a\nca t</w>\n")
json.dump({"pad_token": "<|endoftext|>"}, open("tiny_sd/tokenizer/tokenizer_config.json", "w"))
json.dump({"_class_name": "DDIMScheduler", "beta_schedule": "scaled_linear", "beta_start": 0.00085, "beta_end": 0.012,
           "num_train_timesteps": 1000, "steps_offset": 1, "prediction_type": "epsilon", "set_alpha_to_one": False}, open("tiny_sd/scheduler/scheduler_config.json", "w"))
json.dump({"_class_name": "OnnxStableDiffusionInpaintPipeline"}, open("tiny_sd/model_index.json", "w"))
