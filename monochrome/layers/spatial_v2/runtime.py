"""Runtime for visual-mono layer v2. Scores are precomputed from 32x32 features."""
import json, math, threading
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
ARTIFACT_PATH = ROOT / "models" / "mono_spatial_hier_v2.json"
SCORES_PATH = ROOT / "models" / "mono_spatial_hier_v2_scores.json"
_lock = threading.Lock(); _state = {"sig": None, "artifact": {}, "scores": {}}

def _load():
    sig = tuple((p.stat().st_mtime_ns, p.stat().st_size) for p in (ARTIFACT_PATH, SCORES_PATH))
    if _state["sig"] == sig: return _state
    with _lock:
        if _state["sig"] != sig:
            artifact = json.loads(ARTIFACT_PATH.read_text())
            payload = json.loads(SCORES_PATH.read_text())
            if payload.get("model_version") != artifact.get("model_version"):
                raise RuntimeError("spatial model/score version mismatch")
            _state.update(sig=sig, artifact=artifact, scores=payload.get("scores", {}))
    return _state

def model_info(): return dict(_load()["artifact"])
def model_version(): return _load()["artifact"].get("model_version", "spatial-fallback")
def decision_threshold(): return float(_load()["artifact"].get("calibration", {}).get("decision_threshold", .17))
def score_probability(short, model, backdrop):
    return _load()["scores"].get("|".join((str(short).lower(), str(model), str(backdrop))))

def apply_pair(d, p):
    short = d["collection"]["short_name"].lower(); key = "|".join((short, p["model"], p["backdrop"]))
    if "legacy_mono_probability" not in p:
        p["legacy_mono_probability"] = p.get("mono_probability")
        p["legacy_mono_score"] = p.get("mono_score")
        p["legacy_mono_points"] = p.get("mono_points")
        p["legacy_mono_model_version"] = p.get("mono_model_version")
    q = score_probability(short, p["model"], p["backdrop"])
    source = "precomputed_32x32" if q is not None else "legacy_fallback"
    if q is None: q = float(p.get("legacy_mono_probability") or 0)
    q = min(1., max(0., float(q))); mono_score = round(100*q, 2)
    color_score = float(p.get("color_score") or 50.); mono_points = round(.6*mono_score, 3); color_points = round(.4*color_score, 3)
    p.update(item_key=key, spatial_mono_probability=round(q, 7), spatial_mono_score=mono_score,
             spatial_mono_source=source, spatial_mono_model_version=model_version(),
             mono_probability=round(q, 7), mono_score=mono_score, mono_points=mono_points,
             mono_model_version=model_version(), mono_model_threshold=decision_threshold(),
             color_points=color_points, score=round(.6*mono_score+.4*color_score, 2),
             score_formula="0.60*spatial_mono_score + 0.40*BCP_color_score")
    p["mono_layers"] = {
        "legacy": {"version": p.get("legacy_mono_model_version"), "probability": p.get("legacy_mono_probability"), "score": p.get("legacy_mono_score")},
        "spatial": {"version": model_version(), "probability": round(q, 7), "score": mono_score, "source": source, "pixel_size": 32},
    }
    return p

def apply_constructor(d):
    if not isinstance(d, dict) or not d.get("collection"): return d
    for p in d.get("monochrome_pairs", []): apply_pair(d, p)
    d["algo"] = "v2.27"
    method = d.setdefault("scoring_method", {})
    method.update(name="Spatial visual mono + BCP-32 color, additive",
                  formula="0.60*spatial_mono_score + 0.40*BCP_color_score",
                  mono_model=model_version(), legacy_mono_model="mono-logreg-v1",
                  layer_architecture="legacy_v1 frozen + spatial_v2 active",
                  spatial_input="32x32 per-pixel perceptual classes + palette/regions + collection/model identities",
                  pretrained=False, preference_inside_mono=False)
    return d
