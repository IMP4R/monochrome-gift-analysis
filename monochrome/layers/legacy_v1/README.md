# Frozen legacy monochrome layer

Immutable snapshot of the complete pre-spatial monochrome mechanism at production v2.26.

- Model: `mono-logreg-v1`
- Inputs: 23 hand-engineered palette/tone/perceptual features
- Explicit exclusions: BCP, YouGov, color family, absolute hue
- Original label sources: append-only `test_votes_v2.jsonl` and early v4 snapshot recorded in the model artifact
- Production combination before v2: `0.80 * legacy mono + 0.20 * BCP color score`

This directory is a reference layer. New training code must not overwrite these files.
