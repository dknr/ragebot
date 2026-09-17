#!/usr/bin/env python3
"""Download the irony model from HuggingFace and export + quantize it to ONNX.

Produces one file in the build directory:
  irony_int8.onnx   the dynamic-quantized int8 model (what the Go binary embeds)

The model is cardiffnlp/twitter-roberta-base-irony (RobertaForSequenceClassification),
the same RoBERTa-base architecture as the sentiment model. Its tokenizer is identical,
so this script reuses the tokenizer.json built by scripts/build_tokenizer.py and only downloads
the irony weights + config. The fp32 ONNX export is an in-memory intermediate (never
written to disk); only the int8 quantization is kept.
"""
import argparse
import os

import torch
from transformers import RobertaForSequenceClassification

import hf_common

MODEL_ID = "cardiffnlp/twitter-roberta-base-irony"
FILES = ["config.json", "pytorch_model.bin"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", default="work", help="working dir for downloaded HF files")
    ap.add_argument("--out", default=".", help="dir for the produced onnx file")
    ap.add_argument("--model", default=MODEL_ID, help="huggingface model id")
    args = ap.parse_args()

    hf = os.path.join(args.work, "hf-irony")
    hf_common.download_model(args.model, hf, patterns=FILES)

    model = RobertaForSequenceClassification.from_pretrained(hf, local_files_only=True)
    model.eval()

    int8 = os.path.join(args.out, "irony_int8.onnx")
    hf_common.export_quantize(model, int8)
    print(f"{int8}: {os.path.getsize(int8)} bytes")


if __name__ == "__main__":
    main()
