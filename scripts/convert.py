#!/usr/bin/env python3
"""Download the sentiment model from HuggingFace and export + quantize it to ONNX.

Produces two files in the build directory:
  model_int8.onnx   the dynamic-quantized int8 model (what the Go binary embeds)
  tokenizer.json    the tokenizers-format tokenizer for the Go tokenizer binding

The model is cardiffnlp/twitter-roberta-base-sentiment-latest (RobertaForSequenceClassification).
The fp32 ONNX export is an in-memory intermediate (never written to disk); only the int8
quantization is kept. Download and conversion use the shared helpers in hf_common.py.
"""
import argparse
import os

import torch
from transformers import RobertaForSequenceClassification

import hf_common

MODEL_ID = "cardiffnlp/twitter-roberta-base-sentiment-latest"
FILES = ["config.json", "pytorch_model.bin", "vocab.json", "merges.txt", "special_tokens_map.json"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", default="work", help="working dir for downloaded HF files")
    ap.add_argument("--out", default=".", help="dir for the produced onnx/tokenizer files")
    ap.add_argument("--model", default=MODEL_ID, help="huggingface model id")
    args = ap.parse_args()

    hf = os.path.join(args.work, "hf")
    hf_common.download_model(args.model, hf, patterns=FILES)

    model = RobertaForSequenceClassification.from_pretrained(hf, local_files_only=True)
    model.eval()

    int8 = os.path.join(args.out, "model_int8.onnx")
    tok_path = os.path.join(args.out, "tokenizer.json")

    if not os.path.exists(int8):
        hf_common.export_quantize(model, int8)

    if not os.path.exists(tok_path):
        print("building tokenizer.json")
        hf_common.build_tokenizer(hf, tok_path)

    for p in (int8, tok_path):
        print(f"{p}: {os.path.getsize(p)} bytes")


if __name__ == "__main__":
    main()
