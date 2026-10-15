#!/usr/bin/env python3
"""Train an interpretable ridge-logistic monochrome model from append-only swipes."""
import argparse, collections, hashlib, json, math, time
from pathlib import Path
import numpy as np
import colors


def file_md5(path):
    h=hashlib.md5()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(1<<20),b''):h.update(chunk)
    return h.hexdigest()


def load_examples(cache_dir):
    grouped=collections.defaultdict(list);sources={}
    for path in sorted(cache_dir.glob('test_votes_v*.jsonl')):
        sources[path.name]={'md5':file_md5(path),'bytes':path.stat().st_size}
        for line in path.read_text(encoding='utf-8',errors='ignore').splitlines():
            try:r=json.loads(line)
            except Exception:continue
            if r.get('task')!='swipe_monochrome' or r.get('choice') not in ('monochrome','not_monochrome'):continue
            a=r.get('a') or {};key=a.get('item_key')
            if key:grouped[key].append(r)
    examples=[];ties=0
    for key,rows in grouped.items():
        yes=sum(r['choice']=='monochrome' for r in rows)
        if 2*yes==len(rows):ties+=1;continue
        a=max(rows,key=lambda r:r.get('ts',0)).get('a') or {}
        h=(a.get('backdrop_hex') or '').lstrip('#')
        if len(h)!=6:continue
        try:bg=tuple(int(h[i:i+2],16) for i in (0,2,4))
        except ValueError:continue
        model_info={'palette':a.get('model_palette') or [],'neutral':a.get('model_neutral'),
                    'center':a.get('model_center') or {},'achromatic':a.get('model_achromatic'),
                    'rainbow':a.get('rainbow') or 0.0,'rgb':None}
        f=colors.mono_features(bg,model_info,a.get('base_score'),a.get('classic_score'),a.get('backdrop'))
        examples.append((key,f,int(yes>len(rows)/2)))
    return examples,sources,ties


def fit(X,y,lam):
    mean=X.mean(0);scale=X.std(0);scale[scale<1e-8]=1.0
    Z=np.c_[np.ones(len(X)),(X-mean)/scale];b=np.zeros(Z.shape[1]);pen=np.r_[0.0,np.ones(Z.shape[1]-1)]
    for _ in range(120):
        p=1/(1+np.exp(-np.clip(Z@b,-30,30)));g=Z.T@(y-p)-lam*pen*b
        H=Z.T@((p*(1-p))[:,None]*Z)+lam*np.diag(pen)+1e-9*np.eye(Z.shape[1])
        step=np.linalg.solve(H,g);b+=step
        if np.max(np.abs(step))<1e-9:break
    return mean,scale,b


def predict(model,X):
    mean,scale,b=model;Z=np.c_[np.ones(len(X)),(X-mean)/scale]
    return 1/(1+np.exp(-np.clip(Z@b,-30,30)))


def auc(p,y):
    pos=p[y==1];neg=p[y==0]
    return sum(1 if a>b else .5 if a==b else 0 for a in pos for b in neg)/(len(pos)*len(neg))


def best_threshold(p,y):
    best=None
    for t in sorted(set(float(x) for x in p)):
        pred=p>=t;tp=int(np.sum(pred&(y==1)));tn=int(np.sum((~pred)&(y==0)));fp=int(np.sum(pred&(y==0)));fn=int(np.sum((~pred)&(y==1)))
        recall=tp/max(1,tp+fn);specificity=tn/max(1,tn+fp);ba=(recall+specificity)/2
        if best is None or ba>best[0]:best=(ba,t,recall,specificity)
    return best


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--cache',default='cache');ap.add_argument('--output',default='models/mono_logreg_v1.json');ap.add_argument('--version',default='mono-logreg-v1');ap.add_argument('--l2',type=float,default=3.0);args=ap.parse_args()
    examples,sources,ties=load_examples(Path(args.cache));names=list(colors.MONO_FEATURE_NAMES)
    X=np.asarray([[f[n] for n in names] for _,f,_ in examples],float);y=np.asarray([v for _,_,v in examples],int)
    if len(y)<30 or y.sum()<5:raise SystemExit('not enough labels')
    # Five-fold item-level OOF: repeated cards were majority-collapsed above.
    rng=np.random.default_rng(20260811);pos=rng.permutation(np.where(y==1)[0]);neg=rng.permutation(np.where(y==0)[0]);folds=[np.r_[pos[i::5],neg[i::5]] for i in range(5)];oof=np.zeros(len(y))
    for test in folds:
        train=np.setdiff1d(np.arange(len(y)),test);m=fit(X[train],y[train],args.l2);oof[test]=predict(m,X[test])
    cv_auc=float(auc(oof,y));cv_log_loss=float(-np.mean(y*np.log(np.clip(oof,1e-8,1))+(1-y)*np.log(np.clip(1-oof,1e-8,1))))
    ba,threshold,recall,specificity=best_threshold(oof,y);model=fit(X,y,args.l2);mean,scale,b=model
    artifact={'model_version':args.version,'model_type':'ridge_logistic_regression','trained_at':time.time(),'training_items':int(len(y)),'positive_items':int(y.sum()),'negative_items':int(len(y)-y.sum()),'majority_ties_excluded':int(ties),'feature_names':names,'mean':[float(x) for x in mean],'scale':[float(x) for x in scale],'coefficients':[float(x) for x in b[1:]],'intercept':float(b[0]),'l2':args.l2,'cv_auc':round(cv_auc,6),'cv_log_loss':round(cv_log_loss,6),'decision_threshold':round(float(threshold),6),'threshold_balanced_accuracy':round(float(ba),6),'threshold_recall':round(float(recall),6),'threshold_specificity':round(float(specificity),6),'training_sources':sources,'feature_coefficients':{n:round(float(c),6) for n,c in zip(names,b[1:])},'notes':'No BCP, YouGov, color family, or absolute hue features. Append-only votes were read, never modified.'}
    out=Path(args.output);out.parent.mkdir(parents=True,exist_ok=True);tmp=out.with_suffix(out.suffix+'.tmp');tmp.write_text(json.dumps(artifact,ensure_ascii=False,indent=2));tmp.replace(out)
    print(json.dumps({k:artifact[k] for k in ('model_version','training_items','positive_items','cv_auc','cv_log_loss','decision_threshold','threshold_balanced_accuracy','threshold_recall','threshold_specificity')},indent=2))

if __name__=='__main__':main()
