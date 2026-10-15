# Deterministic visual-mono layer v3

The active mono mechanism is a 100-point additive scorecard. Every component is an explicit foreground/background relation; all integer weights are non-negative and sum to 100. Collection names, model names, absolute hue, BCP and YouGov are excluded.

Training uses only the immutable `mono-tests-v5` snapshot (839 precomputed 32x32 examples; 127 positive). Grouped validation: quantized AUC 0.8716 on unseen models and 0.8722 on held-out collections. The continuous and integer versions differ by less than 0.0002 AUC.

The previous spatial hierarchical model remains available only for disagreement diagnostics. Unknown production pairs are calculated from local lossless assets with the same deterministic equations; a marked spatial-v2 fallback is used only if an asset is unavailable.
