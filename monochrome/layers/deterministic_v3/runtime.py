"""Active deterministic v3 layer: integer weights and lazily expanded explanations."""
from __future__ import annotations
import json,threading
from pathlib import Path
from .components import COMPONENT_NAMES,component_values,quantized_score,probability
from .pixel_runtime import extract_needed
ROOT=Path(__file__).resolve().parents[2];CACHE=ROOT/'cache';ARTIFACT_PATH=ROOT/'models/mono_deterministic_scorecard_v3.json';SCORES_PATH=ROOT/'models/mono_deterministic_scorecard_v3_scores.json'
_lock=threading.Lock();_state={'sig':None,'artifact':{},'scores':{}}
def _load():
    sig=tuple((p.stat().st_mtime_ns,p.stat().st_size) for p in (ARTIFACT_PATH,SCORES_PATH))
    if _state['sig']==sig:return _state
    with _lock:
        if _state['sig']!=sig:
            a=json.loads(ARTIFACT_PATH.read_text());s=json.loads(SCORES_PATH.read_text())
            if s.get('model_version')!=a.get('model_version'):raise RuntimeError('deterministic artifact/lookup mismatch')
            _state.update(sig=sig,artifact=a,scores=s.get('scores',{}))
    return _state
def model_info():return dict(_load()['artifact'])
def model_version():return _load()['artifact'].get('model_version','deterministic-fallback')
def decision_score():return float(_load()['artifact'].get('quantized',{}).get('decision_score',65.))
def _entry(short,model,backdrop,p,allow_runtime=True):
    key='|'.join((str(short).lower(),str(model),str(backdrop)));e=_load()['scores'].get(key)
    if e is not None:return e,'precomputed_quantized_32x32'
    if allow_runtime:
        try:
            tab=extract_needed(CACHE,short,model,backdrop,p.get('mono_features') or {});c=component_values(tab);a=_load()['artifact'];q=a['quantized'];s=quantized_score(c,q['integer_weights']);pr=probability(s,q['logit_intercept'],q['logit_scale_per_score_point']);return {'s':round(s,5),'p':round(pr,7),'c':[round(c[n],6) for n in COMPONENT_NAMES]},'runtime_quantized_32x32'
        except Exception:
            pass
    s=float(p.get('spatial_mono_score') or p.get('legacy_mono_score') or 0);pr=float(p.get('spatial_mono_probability') or p.get('legacy_mono_probability') or s/100);return {'s':round(s,5),'p':round(pr,7),'c':None},'spatial_v2_fallback'
def _set_details(p,e):
    c=e.get('c');comps={n:float(v) for n,v in zip(COMPONENT_NAMES,c)} if c else {};weights=_load()['artifact']['quantized']['integer_weights'];points={n:round(weights[n]*comps[n],3) for n in comps};active=[n for n in comps if weights.get(n,0)>0];strong=sorted(active,key=lambda n:points[n],reverse=True)[:3];weak=sorted(active,key=lambda n:(comps[n],-weights[n]))[:3]
    p.update(deterministic_mono_components=comps,deterministic_component_points=points,deterministic_reason_codes={'strongest':strong,'weakest':weak});return p
def attach_details(d,p):
    short=d['collection']['short_name'].lower();e,_=_entry(short,p['model'],p['backdrop'],p,allow_runtime=True);return _set_details(p,e)
def apply_pair(d,p,details=False,allow_runtime=False):
    short=d['collection']['short_name'].lower();key='|'.join((short,p['model'],p['backdrop']));e,source=_entry(short,p['model'],p['backdrop'],p,allow_runtime=allow_runtime);a=_load()['artifact'];q=a['quantized'];score_pub=round(max(0.,min(100.,float(e['s']))),3);pr=max(0.,min(1.,float(e['p'])));color=float(p.get('color_score') or 50.);mp=round(.6*score_pub,3);cp=round(.4*color,3);threshold_probability=probability(q['decision_score'],q['logit_intercept'],q['logit_scale_per_score_point'])
    p.update(item_key=key,deterministic_mono_score=score_pub,deterministic_mono_probability=round(pr,7),deterministic_mono_source=source,deterministic_mono_model_version=model_version(),deterministic_decision_score=decision_score(),mono_score=score_pub,mono_probability=round(pr,7),mono_points=mp,mono_model_version=model_version(),mono_model_threshold=round(threshold_probability,7),color_points=cp,score=round(.6*score_pub+.4*color,2),score_formula='0.60*deterministic_mono_score + 0.40*BCP_color_score')
    layers=p.setdefault('mono_layers',{});layers['deterministic']={'version':model_version(),'probability':round(pr,7),'score':score_pub,'source':source,'quantized_weight_total':100,'decision_score':decision_score()}
    if details:_set_details(p,e)
    else:
        for k in ('deterministic_mono_components','deterministic_component_points','deterministic_reason_codes'):p.pop(k,None)
    return p
def apply_constructor(d):
    if not isinstance(d,dict) or not d.get('collection'):return d
    for p in d.get('monochrome_pairs',[]):apply_pair(d,p,details=False)
    d['algo']='v2.29';m=d.setdefault('scoring_method',{});m.update(name='Deterministic quantized visual mono + BCP-32 color',formula='0.60*deterministic_mono_score + 0.40*BCP_color_score',mono_model=model_version(),legacy_mono_model='mono-logreg-v1',comparison_mono_model='mono-spatial-hier-v2',layer_architecture='legacy_v1 frozen + spatial_v2 diagnostic + deterministic_v3 active',deterministic_input='32x32 relational color abstractions; 15 components; integer weights sum to 100',uses_collection=False,uses_model_identity=False,uses_absolute_hue=False,pretrained=False,preference_inside_mono=False,quantized_weights=_load()['artifact']['quantized']['integer_weights'])
    return d
