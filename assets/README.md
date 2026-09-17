This directory holds the NEAT! reaction GIFs (`neat-*.gif`). They are not
committed; drop them here locally and the binary will embed them at build
time via `go:embed assets`. This file is a sentinel so the embed always has
at least one file to include, letting a clean checkout build with zero GIFs
(which disables the feature gracefully at runtime).
