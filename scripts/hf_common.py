#!/usr/bin/env python3
"""Shared helpers for downloading + converting HuggingFace models to ONNX blobs.

Each model has its own thin convert script (e.g. scripts/convert.py for
sentiment, scripts/convert_irony.py for irony) that calls these helpers. The
download, ONNX export + quantization, and tokenizer build are identical across
the RoBERTa-based models this repo embeds, so they live here rather than being
duplicated per model.
"""
import io
import os

import torch
from huggingface_hub import snapshot_download


def download_model(repo_id, hf_dir, patterns=None):
    """Download the repo files needed for a model into hf_dir.

    Uses huggingface_hub.snapshot_download, which provides caching, resume,
    retries, and progress -- replacing the bespoke urllib downloader. Returns
    the local dir so transformers can load with local_files_only=True.
    """
    snapshot_download(
        repo_id,
        local_dir=hf_dir,
        allow_patterns=patterns,
    )
    return hf_dir


def export_quantize(model, out_int8):
    """Export a torch model to fp32 ONNX and dynamic-quantize it to int8.

    Writes only out_int8 to disk. The fp32 export is required as the input to
    onnxruntime's dynamic quantization, but it is produced in memory (BytesIO)
    and parsed straight into an onnx.ModelProto, so the ~500MB fp32 model never
    touches disk. The ONNX graph has exactly two inputs -- input_ids and
    attention_mask -- because RoBERTa does not use token_type_ids and the
    forward is traced with token_type_ids=None.
    """
    buf = io.BytesIO()
    dummy_ids = torch.ones(1, 16, dtype=torch.int64)
    dummy_mask = torch.ones(1, 16, dtype=torch.int64)
    torch.onnx.export(
        model,
        (dummy_ids, dummy_mask),
        buf,
        input_names=["input_ids", "attention_mask"],
        output_names=["logits"],
        dynamic_axes={
            "input_ids": {0: "batch_size", 1: "sequence_length"},
            "attention_mask": {0: "batch_size", 1: "sequence_length"},
            "logits": {0: "batch_size"},
        },
        opset_version=17,
        dynamo=False,  # legacy TorchScript exporter: shape-inference clean
    )
    import onnx
    fp32 = onnx.ModelProto()
    fp32.ParseFromString(buf.getvalue())
    print(f"exported fp32 onnx in memory: {len(buf.getvalue())} bytes")

    from onnxruntime.quantization import QuantType, quantize_dynamic
    quantize_dynamic(fp32, out_int8, weight_type=QuantType.QInt8)
    print(f"quantized int8 -> {out_int8}: {os.path.getsize(out_int8)} bytes")


def build_tokenizer(hf_dir, out):
    """Build a tokenizers-format tokenizer.json from vocab.json + merges.txt.

    The daulet/tokenizers Go binding reads the standard `tokenizers` library
    serialization. We construct it directly so the build does not depend on the
    Python tokenizers library fetching a tokenizer.json (the HF repo has none).
    """
    import json
    vocab = json.load(open(os.path.join(hf_dir, "vocab.json")))
    merges = []
    for line in open(os.path.join(hf_dir, "merges.txt")):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        merges.append(line.split())

    def special(content, lid, lstrip):
        return {"id": lid, "content": content, "single_word": False,
                "lstrip": lstrip, "rstrip": False, "normalized": not lstrip,
                "special": True}

    added = [
        special("<s>", 0, False),
        special("<pad>", 1, False),
        special("</s>", 2, False),
        special("<unk>", 3, False),
        special("<mask>", 50264, True),
    ]
    data = {
        "version": "1.0",
        "truncation": None,
        "padding": None,
        "added_tokens": added,
        "normalizer": None,
        "pre_tokenizer": {"type": "ByteLevel", "add_prefix_space": False,
                          "trim_offsets": True, "use_regex": True},
        "post_processor": {"type": "RobertaProcessing", "sep": ["</s>", 2],
                           "cls": ["<s>", 0], "trim_offsets": True,
                           "add_prefix_space": False},
        "decoder": {"type": "ByteLevel", "add_prefix_space": True,
                    "trim_offsets": True, "use_regex": True},
        "model": {"type": "BPE", "dropout": None, "unk_token": None,
                  "continuing_subword_prefix": "", "end_of_word_suffix": "",
                  "fuse_unk": False, "byte_fallback": False,
                  "vocab": vocab, "merges": merges},
    }
    with open(out, "w") as f:
        json.dump(data, f)
