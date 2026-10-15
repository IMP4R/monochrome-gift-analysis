# Visual mono layer v2

Active layer trained from 1,335 majority-collapsed human swipe labels (323 positive).

## Inputs
- immutable collection and model identities with shrinkage;
- model palette contrast, entropy/effective color count, hue dispersion and variegation;
- lossless model + backdrop reduced to 32×32;
- every foreground pixel classified as same, same-hue/different-tone, achromatic pair, soft accent, or conflict;
- perceptual ΔE quantiles, region-level closeness/conflict, connected conflict components, shape/symmetry, edge density, colorfulness, entropy and compression complexity;
- no BCP, YouGov, color family or absolute hue in mono prediction.

## Model choice
A regularized hierarchical logistic model trained from scratch won the honest tests: random-background CV AUC 0.865, model-group AUC 0.826, leave-one-collection-out AUC 0.821. A tiny 32×32 CNN was worse (0.849 / 0.803). NIMA, LAION/CLIP, CLIP-IQA and DINOv2 target photographic aesthetics, semantics or generic quality rather than foreground/background color coherence, so they were not deployed.

The complete previous scorer is frozen under `layers/legacy_v1`; old data and artifacts are never overwritten. New critiques go only to `test_votes_v5.jsonl`.

Current production lookup: 8,411 / 8,411 candidates, zero extraction errors; unknown future pairs use an explicit legacy fallback until the lookup is refreshed.
