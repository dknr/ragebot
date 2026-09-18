#!/usr/bin/env python3
"""Download the hate model from HuggingFace and export to ONNX.

Produces one file in the build directory:
  hate_int8.onnx   the fp32 model (int8 quantization destroys this model's accuracy)

The model is cardiffnlp/twitter-roberta-base-hate-latest (RobertaForSequenceClassification),
a RoBERTa-base binary classifier (NOT-HATE / HATE). Its tokenizer is identical to the
other models, so this script reuses the tokenizer.json built by scripts/build_tokenizer.py
and only downloads the hate weights + config. Unlike the other models, this one is exported
as fp32 (not int8) because dynamic int8 quantization destroys its binary classification
accuracy.
"""
import argparse
import io
import os

import torch
from transformers import RobertaForSequenceClassification
from huggingface_hub import snapshot_download

MODEL_ID = "cardiffnlp/twitter-roberta-base-hate-latest"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", default="work", help="working dir for downloaded HF files")
    ap.add_argument("--out", default=".", help="dir for the produced onnx file")
    ap.add_argument("--model", default=MODEL_ID, help="huggingface model id")
    args = ap.parse_args()

    hf = os.path.join(args.work, "hf-hate")
    snapshot_download(
        args.model,
        local_dir=hf,
        allow_patterns=[
            "config.json", "pytorch_model.bin",
            "special_tokens_map.json", "tokenizer_config.json", "tokenizer.json",
        ],
    )

    model = RobertaForSequenceClassification.from_pretrained(hf, local_files_only=True)
    model.eval()

    out = os.path.join(args.out, "hate_int8.onnx")
    buf = io.BytesIO()
    torch.onnx.export(
        model,
        (torch.ones(1, 16, dtype=torch.int64), torch.ones(1, 16, dtype=torch.int64)),
        buf,
        input_names=["input_ids", "attention_mask"],
        output_names=["logits"],
        dynamic_axes={
            "input_ids": {0: "batch_size", 1: "sequence_length"},
            "attention_mask": {0: "batch_size", 1: "sequence_length"},
            "logits": {0: "batch_size"},
        },
        opset_version=17,
        dynamo=False,
    )
    with open(out, "wb") as f:
        f.write(buf.getvalue())
    print(f"{out}: {os.path.getsize(out)} bytes (fp32)")


if __name__ == "__main__":
    main()
