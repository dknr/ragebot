# Builds ragebot (a single Go binary) plus the embedded sentiment and irony
# models and ONNX Runtime shared library. The model blobs are regenerated from
# HuggingFace, so a clean checkout needs Go (1.21+ with cgo), Rust + cargo (to
# build libtokenizers.a from github.com/daulet/tokenizers), and Python 3 (to
# download + convert the models and fetch ORT).

ARCH      := $(shell uname -m)
ORT_VER   := 1.30.0
VENV      := .venv
PY        := $(VENV)/bin/python
BIN       := ragebot
BLOBS     := blobs
WORK      := work

# libtokenizers.a is built from the daulet/tokenizers Go module (a Rust crate).
# If it already exists (e.g. a prebuilt copy is checked in) make skips this.
TOK_MOD   := $(shell go env GOMODCACHE)/github.com/daulet/tokenizers@v1.27.0

# mautrix-go ships two olm implementations: the C libolm (needs the olm library)
# and the pure Go goolm. The goolm tag avoids the cgo dependency for olm; the
# tokenizers binding still needs cgo and libtokenizers.a on the link path.
.PHONY: all build run model irony ort vet clean

all: build

build: $(BLOBS)/model.onnx.zst $(BLOBS)/irony.onnx.zst $(BLOBS)/tokenizer.json.zst $(BLOBS)/ort.so.zst libtokenizers.a
	CGO_LDFLAGS="-L$(CURDIR)" go build -tags goolm -o $(BIN) .

run: build
	./$(BIN) -config config.json

## Download + convert the sentiment model: int8 quantization, tokenizer.json.
model: $(WORK)/model_int8.onnx $(WORK)/tokenizer.json

$(WORK)/model_int8.onnx $(WORK)/tokenizer.json: $(PY) scripts/convert.py
	$(PY) scripts/convert.py --work $(WORK) --out $(WORK)

## Download + convert the irony model: int8 quantization, reuses tokenizer.json.
irony: $(WORK)/irony_int8.onnx

$(WORK)/irony_int8.onnx: $(PY) scripts/convert_irony.py $(WORK)/tokenizer.json
	$(PY) scripts/convert_irony.py --work $(WORK) --out $(WORK)

## Fetch the ONNX Runtime shared library for the current arch.
ort: $(WORK)/ort.so

$(WORK)/ort.so: $(PY) scripts/fetch_ort.py
	$(PY) scripts/fetch_ort.py --out $(WORK) --cache $(WORK)

## Compress build artifacts into blobs/ for go:embed.
$(BLOBS)/model.onnx.zst: $(WORK)/model_int8.onnx
	mkdir -p $(BLOBS)
	zstd -f -19 -q $< -o $@

$(BLOBS)/irony.onnx.zst: $(WORK)/irony_int8.onnx
	mkdir -p $(BLOBS)
	zstd -f -19 -q $< -o $@

$(BLOBS)/tokenizer.json.zst: $(WORK)/tokenizer.json
	mkdir -p $(BLOBS)
	zstd -f -19 -q $< -o $@

$(BLOBS)/ort.so.zst: $(WORK)/ort.so
	mkdir -p $(BLOBS)
	zstd -f -19 -q $< -o $@

## Build libtokenizers.a from the daulet/tokenizers Rust crate.
libtokenizers.a:
	go mod download github.com/daulet/tokenizers
	chmod -R u+w $(WORK)/tokenizers-src 2>/dev/null || true
	rm -rf $(WORK)/tokenizers-src
	cp -r $(TOK_MOD) $(WORK)/tokenizers-src
	chmod -R u+w $(WORK)/tokenizers-src
	cd $(WORK)/tokenizers-src && cargo build --release -p tokenizers-ffi && \
		cp target/release/libtokenizers_ffi.a $(CURDIR)/libtokenizers.a

## Python venv with the conversion tooling.
$(PY):
	python3 -m venv $(VENV)
	$(VENV)/bin/pip install --quiet torch transformers tokenizers onnxruntime onnx onnxscript

vet:
	go vet -tags goolm ./...

clean:
	rm -rf $(BIN) $(BLOBS) $(WORK) $(VENV) libtokenizers.a
