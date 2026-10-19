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
