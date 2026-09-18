package main

import (
	"errors"
	"fmt"
	"io"
	"math"
	"os"
	"runtime"
	"strings"
	"sync"
	"time"

	_ "embed"

	tok "github.com/daulet/tokenizers"
	"github.com/klauspost/compress/zstd"
	ort "github.com/yalue/onnxruntime_go"
	"golang.org/x/sys/unix"
)

// The sentiment model, tokenizer, and ONNX Runtime shared library are embedded
// as compressed blobs. go:embed requires them to live next to this file, so
// blobs/ holds them. Everything is regenerable via the Makefile.
//
//go:embed blobs/model.onnx.zst
var modelZst []byte

//go:embed blobs/irony.onnx.zst
var ironyZst []byte

//go:embed blobs/emotion_int8.onnx.zst
var emotionZst []byte

//go:embed blobs/hate_int8.onnx.zst
var hateZst []byte

//go:embed blobs/tokenizer.json.zst
var tokenizerZst []byte

//go:embed blobs/ort.so.zst
var ortSoZst []byte

// labels matches the model's logits column order (negative, neutral, positive).
var labels = []string{"negative", "neutral", "positive"}

// ironyLabels matches the irony model's logits column order (non_irony, irony).
var ironyLabels = []string{"non_irony", "irony"}

// emotionLabels matches the emotion model's logits column order (11 classes).
var emotionLabels = []string{"anger", "anticipation", "disgust", "fear", "joy", "love", "optimism", "pessimism", "sadness", "surprise", "trust"}

var hateLabels = []string{"NOT-HATE", "HATE"}

// sentiment is the result of a single classification run.
type sentiment struct {
	Label      string    `json:"label"`
	Confidence float32   `json:"confidence"`
	Scores     []float32 `json:"scores"` // label order
	Tokens     int       `json:"tokens"`
	ElapsedMS  float64   `json:"elapsed_ms"`
}

// analyzer runs tokenize + inference. The tokenizers binding states no
// concurrency guarantee, so one lock covers encode+run.
type analyzer struct {
	mu          sync.Mutex
	tk          *tok.Tokenizer
	sess        *ort.DynamicAdvancedSession
	ironySess   *ort.DynamicAdvancedSession
	emotionSess *ort.DynamicAdvancedSession
	hateSess    *ort.DynamicAdvancedSession
	sessOpts    *ort.SessionOptions
	maxTokens   int
}

func decompress(z []byte) []byte {
	r, err := zstd.NewReader(strings.NewReader(string(z)))
	if err != nil {
		panic(err)
	}
	defer r.Close()
	out, err := io.ReadAll(r)
	if err != nil {
		panic(err)
	}
	return out
}

// memfdWrite puts data in an anonymous in-RAM file and returns a path dlopen
// can use. The os.File is intentionally never closed (runtime.KeepAlive):
// closing it unlinks the memfd and invalidates the mapping ORT created.
func memfdWrite(name string, data []byte) (string, error) {
	fd, err := unix.MemfdCreate(name, 0)
	if err != nil {
		return "", err
	}
	f := os.NewFile(uintptr(fd), name)
	if _, err := f.Write(data); err != nil {
		f.Close()
		return "", err
	}
	path := fmt.Sprintf("/proc/self/fd/%d", fd)
	runtime.KeepAlive(f)
	return path, nil
}

func softmax(in []float32) []float32 {
	mx := in[0]
	for _, v := range in[1:] {
		if v > mx {
			mx = v
		}
	}
	out := make([]float32, len(in))
	sum := float32(0)
	for i, v := range in {
		e := float32(math.Exp(float64(v - mx)))
		out[i] = e
		sum += e
	}
	for i := range out {
		out[i] /= sum
	}
	return out
}

// newAnalyzer loads the embedded model and runtime. It must be called once at
// startup; the returned analyzer owns the ORT environment and tokenizer.
func newAnalyzer(maxTokens int) (*analyzer, error) {
	// The ORT shared library is dlopen'd, never linked, so it must live at a
	// path. Decompress the embedded blob into an anonymous memfd and hand that
	// path to dlopen.
	soPath, err := memfdWrite("libonnxruntime", decompress(ortSoZst))
	if err != nil {
		return nil, err
	}
	ort.SetSharedLibraryPath(soPath)
	if err := ort.InitializeEnvironment(); err != nil {
		return nil, err
	}

	tk, err := tok.FromBytes(decompress(tokenizerZst))
	if err != nil {
		return nil, err
	}

	sessOpts, err := ort.NewSessionOptions()
	if err != nil {
		return nil, err
	}

	sess, err := ort.NewDynamicAdvancedSessionWithONNXData(decompress(modelZst),
		[]string{"input_ids", "attention_mask"}, []string{"logits"}, sessOpts)
	if err != nil {
		return nil, err
	}

	ironySess, err := ort.NewDynamicAdvancedSessionWithONNXData(decompress(ironyZst),
		[]string{"input_ids", "attention_mask"}, []string{"logits"}, sessOpts)
	if err != nil {
		return nil, err
	}

	emotionSess, err := ort.NewDynamicAdvancedSessionWithONNXData(decompress(emotionZst),
		[]string{"input_ids", "attention_mask"}, []string{"logits"}, sessOpts)
	if err != nil {
		return nil, err
	}

	hateSess, err := ort.NewDynamicAdvancedSessionWithONNXData(decompress(hateZst),
		[]string{"input_ids", "attention_mask"}, []string{"logits"}, sessOpts)
	if err != nil {
		return nil, err
	}

	return &analyzer{
		tk:          tk,
		sess:        sess,
		ironySess:   ironySess,
		emotionSess: emotionSess,
		hateSess:    hateSess,
		sessOpts:    sessOpts,
		maxTokens:   maxTokens,
	}, nil
}

func (a *analyzer) classify(text string) (sentiment, error) {
	return a.run(text, labels, a.sess)
}

// classifyIrony runs the same tokenize + inference pipeline against the irony
// model. The returned Scores[1] is the irony probability (logits column order
// is non_irony, irony).
func (a *analyzer) classifyIrony(text string) (sentiment, error) {
	return a.run(text, ironyLabels, a.ironySess)
}

// classifyEmotions runs the emotion model and applies sigmoid to the 11 raw
// logits (multilabel). Returns the per-label sigmoid probabilities.
func (a *analyzer) classifyEmotions(text string) (sentiment, error) {
	return a.runMultilabel(text, emotionLabels, a.emotionSess)
}

// classifyHate runs the hate model. Scores[1] is the hate probability (label
// order is NOT-HATE, HATE).
func (a *analyzer) classifyHate(text string) (sentiment, error) {
	return a.run(text, hateLabels, a.hateSess)
}

// sigmoid converts a raw logit to a probability.
func sigmoid(x float32) float32 {
	if x >= 0 {
		e := float32(math.Exp(-float64(x)))
		return 1.0 / (1.0 + e)
	}
	e := float32(math.Exp(float64(x)))
	return e / (1.0 + e)
}

func (a *analyzer) run(text string, labels []string, sess *ort.DynamicAdvancedSession) (sentiment, error) {
	a.mu.Lock()
	defer a.mu.Unlock()

	ids, _, err := a.tk.EncodeErr(text, true)
	if err != nil {
		return sentiment{}, err
	}
	if a.maxTokens > 0 && len(ids) > a.maxTokens {
		ids = ids[:a.maxTokens]
	}
	seq := len(ids)
	if seq == 0 {
		return sentiment{}, errors.New("tokenizer produced no tokens")
	}

	inputIDs := make([]int64, seq)
	mask := make([]int64, seq)
	for i, id := range ids {
		inputIDs[i] = int64(id)
		mask[i] = 1
	}

	inT, err := ort.NewTensor(ort.NewShape(1, int64(seq)), inputIDs)
	if err != nil {
		return sentiment{}, err
	}
	defer inT.Destroy()
	maT, err := ort.NewTensor(ort.NewShape(1, int64(seq)), mask)
	if err != nil {
		return sentiment{}, err
	}
	defer maT.Destroy()
	outT, err := ort.NewEmptyTensor[float32](ort.NewShape(1, int64(len(labels))))
	if err != nil {
		return sentiment{}, err
	}
	defer outT.Destroy()

	t0 := time.Now()
	if err := sess.Run([]ort.Value{inT, maT}, []ort.Value{outT}); err != nil {
		return sentiment{}, err
	}
	probs := softmax(append([]float32(nil), outT.GetData()...))

	best := 0
	for i, p := range probs {
		if p > probs[best] {
			best = i
		}
	}
	return sentiment{
		Label:      labels[best],
		Confidence: probs[best],
		Scores:     probs,
		Tokens:     seq,
		ElapsedMS:  float64(time.Since(t0).Microseconds()) / 1000.0,
	}, nil
}

// runMultilabel runs inference and applies sigmoid to each logit independently
// (multilabel classification). Returns the per-label sigmoid probabilities.
func (a *analyzer) runMultilabel(text string, labels []string, sess *ort.DynamicAdvancedSession) (sentiment, error) {
	a.mu.Lock()
	defer a.mu.Unlock()

	ids, _, err := a.tk.EncodeErr(text, true)
	if err != nil {
		return sentiment{}, err
	}
	if a.maxTokens > 0 && len(ids) > a.maxTokens {
		ids = ids[:a.maxTokens]
	}
	seq := len(ids)
	if seq == 0 {
		return sentiment{}, errors.New("tokenizer produced no tokens")
	}

	inputIDs := make([]int64, seq)
	mask := make([]int64, seq)
	for i, id := range ids {
		inputIDs[i] = int64(id)
		mask[i] = 1
	}

	inT, err := ort.NewTensor(ort.NewShape(1, int64(seq)), inputIDs)
	if err != nil {
		return sentiment{}, err
	}
	defer inT.Destroy()
	maT, err := ort.NewTensor(ort.NewShape(1, int64(seq)), mask)
	if err != nil {
		return sentiment{}, err
	}
	defer maT.Destroy()
	outT, err := ort.NewEmptyTensor[float32](ort.NewShape(1, int64(len(labels))))
	if err != nil {
		return sentiment{}, err
	}
	defer outT.Destroy()

	t0 := time.Now()
	if err := sess.Run([]ort.Value{inT, maT}, []ort.Value{outT}); err != nil {
		return sentiment{}, err
	}
	logits := append([]float32(nil), outT.GetData()...)

	probs := make([]float32, len(logits))
	for i, l := range logits {
		probs[i] = sigmoid(l)
	}

	best := 0
	for i, p := range probs {
		if p > probs[best] {
			best = i
		}
	}
	return sentiment{
		Label:      labels[best],
		Confidence: probs[best],
		Scores:     probs,
		Tokens:     seq,
		ElapsedMS:  float64(time.Since(t0).Microseconds()) / 1000.0,
	}, nil
}

// emojiThresholds maps each label to descending (threshold, emoji) pairs,
// ported from the Python predecessor (refs/ragebot-python/defaults.py). The
// first threshold the confidence meets wins; below the lowest threshold no
// reaction is sent. "irony" is checked first, so irony takes priority over
// the sentiment label.
var emojiThresholds = map[string][]struct {
	threshold float32
	emoji     string
}{
	"irony": {
		{0.95, "😏"},
	},
	"negative": {
		{0.95, "🤬"},
		{0.90, "😡"},
		{0.80, "🙈"},
	},
	"neutral": {
		{0.90, "😐"},
	},
	"positive": {
		{0.95, "🎉"},
		{0.90, "👏"},
		{0.80, "👍"},
	},
}

// emojiForSentiment returns the reaction emoji for a classification result, or
// "" when the confidence is below every threshold for that label. Irony is
// evaluated first (highest priority); if it clears an irony threshold it wins,
// otherwise the sentiment label's thresholds apply.
func emojiForSentiment(label string, confidence float32, irony float32) string {
	if thresholds, ok := emojiThresholds["irony"]; ok {
		for _, t := range thresholds {
			if irony >= t.threshold {
				return t.emoji
			}
		}
	}
	thresholds, ok := emojiThresholds[label]
	if !ok {
		return ""
	}
	for _, t := range thresholds {
		if confidence >= t.threshold {
			return t.emoji
		}
	}
	return ""
}

// emojiEmotionThresholds maps each emotion label to descending
// (threshold, emoji) pairs. The first threshold the emotion probability meets
// wins; below the lowest threshold no reaction is sent.
var emojiEmotionThresholds = map[string][]struct {
	threshold float32
	emoji     string
}{
	"anger": {
		{0.95, "🤬"},
		{0.90, "😡"},
		{0.80, "😠"},
	},
	"anticipation": {
		{0.95, "👀"},
		{0.90, "🤔"},
		{0.80, "🫣"},
	},
	"disgust": {
		{0.90, "🤢"},
		{0.80, "🙄"},
	},
	"fear": {
		{0.95, "😱"},
		{0.90, "😨"},
		{0.80, "😰"},
	},
	"joy": {
		{0.95, "😂"},
		{0.90, "🥳"},
		{0.80, "😊"},
	},
	"love": {
		{0.95, "🥰"},
		{0.90, "😍"},
		{0.80, "💕"},
	},
	"optimism": {
		{0.95, "🌟"},
		{0.90, "💪"},
		{0.80, "✨"},
	},
	"pessimism": {
		{0.95, "😞"},
		{0.90, "😒"},
		{0.80, "🙁"},
	},
	"sadness": {
		{0.95, "😭"},
		{0.90, "😢"},
		{0.80, "💔"},
	},
	"surprise": {
		{0.95, "🤯"},
		{0.80, "😲"},
	},
	"trust": {
		{0.95, "🤝"},
		{0.90, "💜"},
		{0.80, "🫶"},
	},
}

// emojiForEmotion returns the reaction emoji for an emotion label and its
// probability, or "" when the probability is below every threshold for that
// emotion.
func emojiForEmotion(label string, probability float32) string {
	thresholds, ok := emojiEmotionThresholds[label]
	if !ok {
		return ""
	}
	for _, t := range thresholds {
		if probability >= t.threshold {
			return t.emoji
		}
	}
	return ""
}

// emojiHateThresholds maps the hate label to descending (threshold, emoji) pairs.
var emojiHateThresholds = map[string][]struct {
	threshold float32
	emoji     string
}{
	"HATE": {
		{0.95, "🚫"},
		{0.90, "⚠️"},
		{0.80, "🔇"},
	},
}

// emojiForHate returns the reaction emoji for a hate classification, or "" when
// the hate probability is below the threshold or the label is NOT-HATE.
func emojiForHate(label string, hateProb float32) string {
	if label != "HATE" {
		return ""
	}
	thresholds, ok := emojiHateThresholds[label]
	if !ok {
		return ""
	}
	for _, t := range thresholds {
		if hateProb >= t.threshold {
			return t.emoji
		}
	}
	return ""
}
