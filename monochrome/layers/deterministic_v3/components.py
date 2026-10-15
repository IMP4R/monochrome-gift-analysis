"""Pure visual abstractions. No identities, hue preference, BCP, or YouGov."""
from __future__ import annotations
import math
COMPONENT_NAMES = [
    "exact_anchor", "near_color_mass", "broad_color_mass", "best_tail_fit",
    "body_tail_fit", "spatial_mean", "spatial_floor", "perceptual_fit",
    "tone_fit", "hue_tone_mass", "conflict_mass_clean",
    "conflict_cluster_clean", "regional_clean", "soft_accent_clean",
    "neutral_compatibility",
]
REGION_CLOSE = [f"region_{x}_closeness" for x in ("center","border","top","bottom","left","right")]
REGION_CONFLICT = [f"region_{x}_conflict" for x in ("center","border","top","bottom","left","right")]
def clamp(v): return max(0.0, min(1.0, float(v)))
def component_values(tab):
    g=lambda n: float(tab.get(n,0.0) or 0.0)
    close=[g(n) for n in REGION_CLOSE]; conflict=[g(n) for n in REGION_CONFLICT]
    ach=clamp(2*g("pixel_class_achromatic_pair"))
    values={
        "exact_anchor": math.sqrt(clamp(g("pixel_match_lt_12")/.10)),
        "near_color_mass": math.sqrt(clamp(g("pixel_match_lt_18")/.30)),
        "broad_color_mass": math.sqrt(clamp(g("pixel_match_lt_25")/.55)),
        "best_tail_fit": clamp((.42-g("delta_e_q10"))/.34),
        "body_tail_fit": clamp((.58-g("delta_e_q25"))/.43),
        "spatial_mean": clamp((sum(close)/len(close)-.06)/.36),
        "spatial_floor": clamp((min(close)-.03)/.27),
        "perceptual_fit": clamp(g("weighted_perceptual_fit")),
        "tone_fit": clamp(g("weighted_tone_fit")),
        "hue_tone_mass": math.sqrt(clamp((g("pixel_class_same")+g("pixel_class_same_hue_tone"))/.65)),
        "conflict_mass_clean": clamp(1-g("pixel_class_conflict")),
        "conflict_cluster_clean": clamp(1-g("conflict_largest_component")),
        "regional_clean": clamp(1-sum(conflict)/len(conflict)),
        "soft_accent_clean": clamp(1-g("pixel_class_soft_accent")),
        "neutral_compatibility": clamp(1-g("neutral_share")*(1-ach)*(.35+.65*g("bg_sat"))),
    }
    return {n: values[n] for n in COMPONENT_NAMES}
def quantized_score(components, integer_weights):
    return sum(int(integer_weights.get(n,0))*clamp(components.get(n,0)) for n in COMPONENT_NAMES)
def probability(score, intercept, scale):
    z=max(-30.,min(30.,float(intercept)+float(scale)*float(score)))
    return 1/(1+math.exp(-z))
