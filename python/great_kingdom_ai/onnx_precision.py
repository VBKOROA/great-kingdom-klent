"""FP16 conversion with float32 public inputs and outputs."""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any


def convert_onnx_to_fp16_keep_io(path: Path) -> None:
    try:
        import onnx
        from onnxconverter_common import float16
    except ModuleNotFoundError as exc:
        raise RuntimeError("onnxconverter-common is required for FP16 ONNX export") from exc

    model = onnx.load(path)
    _isolate_shared_outputs(model.graph, onnx)
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=r"the float32 number .* will be truncated to .*",
            category=UserWarning,
            module=r"onnxconverter_common\.float16",
        )
        converted = float16.convert_float_to_float16(model, keep_io_types=True)
    _repair_float_casts(converted.graph, onnx)
    # The ordinary checker does not detect all inferred tensor type conflicts.
    # Validate before replacing the FP32 source, even when parity is disabled.
    onnx.checker.check_model(converted, full_check=True)
    onnx.save(converted, path)


def _isolate_shared_outputs(graph: Any, onnx: Any) -> None:
    """Keep public output casts out of internal consumers such as KLENT value."""
    used_names = {
        name for node in graph.node for name in (*node.input, *node.output, node.name)
    }
    for output in graph.output:
        if output.type.tensor_type.elem_type != onnx.TensorProto.FLOAT:
            continue
        if not any(output.name in node.input for node in graph.node):
            continue
        original = output.name
        internal = f"{original}_fp16_internal"
        while internal in used_names:
            internal += "_"
        used_names.add(internal)
        for node in graph.node:
            for names in (node.input, node.output):
                for index, name in enumerate(names):
                    if name == original:
                        names[index] = internal
        for value in graph.value_info:
            if value.name == original:
                value.name = internal
        graph.node.append(onnx.helper.make_node("Identity", [internal], [original]))


def _repair_float_casts(graph: Any, onnx: Any) -> None:
    """Some converter versions change value_info but leave Cast(to=FLOAT)."""
    types = {
        value.name: value.type.tensor_type.elem_type
        for value in (*graph.input, *graph.value_info, *graph.output)
    }
    for node in graph.node:
        if node.op_type == "Cast" and types.get(node.output[0]) == onnx.TensorProto.FLOAT16:
            for attribute in node.attribute:
                if attribute.name == "to" and attribute.i == onnx.TensorProto.FLOAT:
                    attribute.i = onnx.TensorProto.FLOAT16
        for attribute in node.attribute:
            if attribute.type == onnx.AttributeProto.GRAPH:
                _repair_float_casts(attribute.g, onnx)
            elif attribute.type == onnx.AttributeProto.GRAPHS:
                for subgraph in attribute.graphs:
                    _repair_float_casts(subgraph, onnx)
