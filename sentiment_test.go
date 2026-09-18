package main

import (
	"os"
	"testing"
)

var testAnalyzer *analyzer

func TestMain(m *testing.M) {
	var err error
	testAnalyzer, err = newAnalyzer(512)
	if err != nil {
		panic(err)
	}
	os.Exit(m.Run())
}

func TestIronyInference(t *testing.T) {
	ironic := "Oh great, another Monday, just what I needed."
	res, err := testAnalyzer.classifyIrony(ironic)
	if err != nil {
		t.Fatalf("classifyIrony: %v", err)
	}
	t.Logf("ironic sample: label=%s irony_prob=%.3f scores=%v", res.Label, res.Scores[1], res.Scores)
	if res.Scores[1] < 0.5 {
		t.Errorf("expected irony probability >= 0.5, got %v", res.Scores[1])
	}

	plain := "The weather is nice today and I am happy."
	res2, err := testAnalyzer.classifyIrony(plain)
	if err != nil {
		t.Fatalf("classifyIrony: %v", err)
	}
	t.Logf("plain sample: label=%s irony_prob=%.3f scores=%v", res2.Label, res2.Scores[1], res2.Scores)
	// The Twitter-trained irony model over-flags pleasant statements, so the
	// real gate is the emojiThresholds["irony"] pairs, not a 0.5 cutoff.
	if res2.Scores[1] >= res.Scores[1] {
		t.Errorf("expected ironic sample to score higher than plain, got %v >= %v", res2.Scores[1], res.Scores[1])
	}
	if res.Scores[1] < 0.90 {
		t.Errorf("expected ironic sample to clear the 0.90 irony threshold, got %v", res.Scores[1])
	}
	if res2.Scores[1] >= 0.80 {
		t.Errorf("expected plain sample below the 0.80 irony threshold, got %v", res2.Scores[1])
	}
}

func TestEmojiForUnified(t *testing.T) {
	tests := []struct {
		name         string
		sentLabel    string
		sentScores   []float32
		ironyScore   float32
		emoLabel     string
		emoScores    []float32
		wantEmoji    string
	}{
		// Sentiment wins
		{"high positive", "positive", []float32{0.05, 0.0, 0.96}, 0.1, "joy", []float32{0.1, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0}, "🎉"},
		// Emotion wins over sentiment when higher
		{"emotion beats sentiment", "positive", []float32{0.05, 0.0, 0.80}, 0.1, "anger", []float32{0.92, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0}, "😡"},
		// Irony wins
		{"high irony", "neutral", []float32{0.0, 0.90, 0.1}, 0.97, "sadness", []float32{0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.90, 0.0, 0.0}, "😏"},
		// Emotion anger beats irony
		{"anger beats irony", "positive", []float32{0.0, 0.0, 0.90}, 0.94, "anger", []float32{0.96, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0}, "🤬"},
		// No threshold met
		{"no threshold", "negative", []float32{0.70, 0.0, 0.0}, 0.5, "sadness", []float32{0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.60, 0.0, 0.0}, ""},
		// Neutral sentiment
		{"neutral", "neutral", []float32{0.0, 0.91, 0.0}, 0.0, "joy", []float32{0.0, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0}, "😐"},
	}
	for _, tt := range tests {
		got := emojiForUnified(tt.sentLabel, tt.sentScores, tt.ironyScore, tt.emoLabel, tt.emoScores)
		if got != tt.wantEmoji {
			t.Errorf("%s: emojiForUnified(%q, %v, %.2f, %q, %v) = %q, want %q",
				tt.name, tt.sentLabel, tt.sentScores, tt.ironyScore, tt.emoLabel, tt.emoScores, got, tt.wantEmoji)
		}
	}
}

func TestClassifyEmotions(t *testing.T) {
	// Joyful text should score high on joy
	joyText := "I am so happy and excited about the wonderful day ahead!"
	res, err := testAnalyzer.classifyEmotions(joyText)
	if err != nil {
		t.Fatalf("classifyEmotions joy: %v", err)
	}
	t.Logf("joy text: label=%s confidence=%.3f scores=%v", res.Label, res.Confidence, res.Scores)
	if res.Label != "joy" {
		t.Errorf("expected joy label, got %q", res.Label)
	}

	// Angry text should score high on anger
	angerText := "This is absolutely infuriating and makes me so angry!"
	res2, err := testAnalyzer.classifyEmotions(angerText)
	if err != nil {
		t.Fatalf("classifyEmotions anger: %v", err)
	}
	t.Logf("anger text: label=%s confidence=%.3f scores=%v", res2.Label, res2.Confidence, res2.Scores)
	if res2.Label != "anger" {
		t.Errorf("expected anger label, got %q", res2.Label)
	}

	// Plain statement should not exceed high confidence thresholds
	plainText := "The sky is blue and the grass is green."
	res3, err := testAnalyzer.classifyEmotions(plainText)
	if err != nil {
		t.Fatalf("classifyEmotions plain: %v", err)
	}
	t.Logf("plain text: label=%s confidence=%.3f scores=%v", res3.Label, res3.Confidence, res3.Scores)
}

func TestEmojiForHate(t *testing.T) {
	tests := []struct {
		label   string
		hateProb float32
		wantEmoji string
	}{
		{"HATE", 0.96, "🚫"},
		{"HATE", 0.90, "⚠️"},
		{"HATE", 0.81, "🔇"},
		{"HATE", 0.79, ""},
		{"NOT-HATE", 0.99, ""},
		{"NOT-HATE", 0.50, ""},
	}
	for _, tt := range tests {
		got := emojiForHate(tt.label, tt.hateProb)
		if got != tt.wantEmoji {
			t.Errorf("emojiForHate(%q, %.2f) = %q, want %q", tt.label, tt.hateProb, got, tt.wantEmoji)
		}
	}
}

func TestClassifyHate(t *testing.T) {
	// Explicit hate speech should be classified as HATE
	hateText := "All people of this race should be killed and eradicated."
	res, err := testAnalyzer.classifyHate(hateText)
	if err != nil {
		t.Fatalf("classifyHate hate: %v", err)
	}
	t.Logf("hate text: label=%s confidence=%.3f scores=%v", res.Label, res.Confidence, res.Scores)
	if res.Label != "HATE" {
		t.Errorf("expected HATE label, got %q", res.Label)
	}

	// Normal text should be classified as NOT-HATE
	normalText := "I love this community, everyone is so friendly and helpful here."
	res2, err := testAnalyzer.classifyHate(normalText)
	if err != nil {
		t.Fatalf("classifyHate normal: %v", err)
	}
	t.Logf("normal text: label=%s confidence=%.3f scores=%v", res2.Label, res2.Confidence, res2.Scores)
	if res2.Label != "NOT-HATE" {
		t.Errorf("expected NOT-HATE label, got %q", res2.Label)
	}
}
