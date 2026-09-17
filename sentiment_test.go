package main

import "testing"

func TestIronyInference(t *testing.T) {
	a, err := newAnalyzer(512)
	if err != nil {
		t.Fatalf("newAnalyzer: %v", err)
	}
	ironic := "Oh great, another Monday, just what I needed."
	res, err := a.classifyIrony(ironic)
	if err != nil {
		t.Fatalf("classifyIrony: %v", err)
	}
	t.Logf("ironic sample: label=%s irony_prob=%.3f scores=%v", res.Label, res.Scores[1], res.Scores)
	if res.Scores[1] < 0.5 {
		t.Errorf("expected irony probability >= 0.5, got %v", res.Scores[1])
	}

	plain := "The weather is nice today and I am happy."
	res2, err := a.classifyIrony(plain)
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
