#!/usr/bin/env python3
"""Download the emotion model from HuggingFace and export + quantize it to ONNX.

Produces one file in the build directory:
  emotion_int8.onnx   the dynamic-quantized int8 model (what the Go binary embeds)

The model is cardiffnlp/twitter-roberta-large-emotion-latest (RobertaForSequenceClassification),
a RoBERTa-large model with 11 emotion classes (multilabel via sigmoid). Its tokenizer is
identical to the sentiment/irony models, so this script reuses the tokenizer.json built by
scripts/build_tokenizer.py and only downloads the emotion weights + config. The fp32 ONNX
export is an in-memory intermediate (never written to disk); only the int8 quantization is kept.
"""
import argparse
import os

import torch
from transformers import RobertaForSequenceClassification

import hf_common

MODEL_ID = "cardiffnlp/twitter-roberta-large-emotion-latest"
FILES = ["config.json", "model.safetensors"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", default="work", help="working dir for downloaded HF files")
    ap.add_argument("--out", default=".", help="dir for the produced onnx file")
    ap.add_argument("--model", default=MODEL_ID, help="huggingface model id")
    args = ap.parse_args()

    hf = os.path.join(args.work, "hf-emotion")
    hf_common.download_model(args.model, hf, patterns=FILES)

    model = RobertaForSequenceClassification.from_pretrained(hf, local_files_only=True)
    model.eval()

    int8 = os.path.join(args.out, "emotion_int8.onnx")
    hf_common.export_quantize(model, int8)
    print(f"{int8}: {os.path.getsize(int8)} bytes")


if __name__ == "__main__":
    main()
