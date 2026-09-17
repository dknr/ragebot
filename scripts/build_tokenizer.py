#!/usr/bin/env python3
"""Download the RoBERTa tokenizer files and build tokenizer.json.

Produces one file in the build directory:
  tokenizer.json   the tokenizers-format tokenizer for the Go tokenizer binding

Both embedded models share the same RoBERTa-base tokenizer, so this is built
once from the sentiment model's repo (vocab.json + merges.txt) and reused by
the sentiment and irony ONNX models. Download and building use the shared
helpers in hf_common.py.
"""
import argparse
import os

import hf_common

MODEL_ID = "cardiffnlp/twitter-roberta-base-sentiment-latest"
FILES = ["vocab.json", "merges.txt"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", default="work", help="working dir for downloaded HF files")
    ap.add_argument("--out", default=".", help="dir for the produced tokenizer file")
    ap.add_argument("--model", default=MODEL_ID, help="huggingface model id")
    args = ap.parse_args()

    hf = os.path.join(args.work, "hf-tok")
    hf_common.download_model(args.model, hf, patterns=FILES)

    tok_path = os.path.join(args.out, "tokenizer.json")
    hf_common.build_tokenizer(hf, tok_path)
    print(f"{tok_path}: {os.path.getsize(tok_path)} bytes")


if __name__ == "__main__":
    main()
