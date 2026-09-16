#!/usr/bin/env python3
"""Download the sentiment model from HuggingFace and export + quantize it to ONNX.

Produces three files in the build directory:
  model_fp32.onnx   the fp32 ONNX export (intermediate)
  model_int8.onnx   the dynamic-quantized int8 model (what the Go binary embeds)
  tokenizer.json    the tokenizers-format tokenizer for the Go tokenizer binding

The model is cardiffnlp/twitter-roberta-base-sentiment-latest (RobertaForSequenceClassification).
The ONNX export has exactly two inputs -- input_ids and attention_mask -- because RoBERTa does not
use token_type_ids and torch.onnx.export traces the forward with token_type_ids=None.
"""
import argparse
import os
import sys
import urllib.request

MODEL_ID = "cardiffnlp/twitter-roberta-base-sentiment-latest"
FILES = ["config.json", "pytorch_model.bin", "vocab.json", "merges.txt", "special_tokens_map.json"]
BASE_URL = "https://huggingface.co/{model}/resolve/main/{file}"


def download(url, path):
    if os.path.exists(path) and os.path.getsize(path) > 0:
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".part"
    for attempt in range(5):
        print(f"downloading {url} (attempt {attempt + 1})")
        with urllib.request.urlopen(url) as r:
            want = int(r.headers.get("Content-Length") or 0)
            got = 0
            with open(tmp, "wb") as f:
                while True:
                    chunk = r.read(1 << 20)
                    if not chunk:
                        break
                    f.write(chunk)
                    got += len(chunk)
        if want and got == want:
            os.replace(tmp, path)
            return
        print(f"  truncated: got {got}, expected {want}; retrying")
    raise RuntimeError(f"failed to download {url}")


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", default="work", help="working dir for downloaded HF files")
    ap.add_argument("--out", default=".", help="dir for the produced onnx/tokenizer files")
    ap.add_argument("--model", default=MODEL_ID, help="huggingface model id")
    args = ap.parse_args()

    hf = os.path.join(args.work, "hf")
    for f in FILES:
        download(BASE_URL.format(model=args.model, file=f), os.path.join(hf, f))

    import torch
    from transformers import RobertaForSequenceClassification

    model = RobertaForSequenceClassification.from_pretrained(hf, local_files_only=True)
    model.eval()

    fp32 = os.path.join(args.out, "model_fp32.onnx")
    int8 = os.path.join(args.out, "model_int8.onnx")
    tok_path = os.path.join(args.out, "tokenizer.json")

    if not os.path.exists(fp32):
        print("exporting fp32 onnx")
        dummy_ids = torch.ones(1, 16, dtype=torch.int64)
        dummy_mask = torch.ones(1, 16, dtype=torch.int64)
        torch.onnx.export(
            model,
            (dummy_ids, dummy_mask),
            fp32,
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

    if not os.path.exists(int8):
        print("quantizing to int8 (dynamic)")
        # onnxruntime.quantization is the standard path; onnx.quantize is the
        # legacy API. Prefer the former.
        from onnxruntime.quantization import QuantType, quantize_dynamic
        quantize_dynamic(fp32, int8, weight_type=QuantType.QInt8)

    if not os.path.exists(tok_path):
        print("building tokenizer.json")
        build_tokenizer(hf, tok_path)

    for p in (fp32, int8, tok_path):
        print(f"{p}: {os.path.getsize(p)} bytes")


if __name__ == "__main__":
    main()
