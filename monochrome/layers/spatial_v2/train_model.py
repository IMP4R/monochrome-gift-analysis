import json
from pathlib import Path
import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score,average_precision_score,log_loss,brier_score_loss
from sklearn.model_selection import StratifiedKFold,StratifiedGroupKFold,LeaveOneGroupOut
ROOT=Path(__file__).resolve().parents[1];snap=ROOT/'analysis/snapshot_20260811';d=np.load(snap/'spatial_dataset_v2.npz');y=d['y'].astype(int);tab=d['tabular'].astype(float);pix=d['pixels'].astype(np.float32);cols=d['collection'].astype(str);mods=d['model'].astype(str)
pooled=pix.reshape(len(y),12,8,4,8,4).mean((3,5)).reshape(len(y),-1)
splits={'random5':list(StratifiedKFold(5,shuffle=True,random_state=20260811).split(tab,y)),'model_group5':list(StratifiedGroupKFold(5,shuffle=True,random_state=20260811).split(tab,y,groups=mods)),'collection_logo':list(LeaveOneGroupOut().split(tab,y,groups=cols))}
def logit(p):return np.log(np.clip(p,1e-5,1-1e-5)/np.clip(1-p,1e-5,1-1e-5))
def target_features(tr,te):
 prior=float(y[tr].mean());out=[]
 for values,strength in ((cols,30.),(mods,10.)):
  stats={}
  for v in set(values[tr]):
   z=y[tr][values[tr]==v];stats[v]=(float((z.sum()+strength*prior)/(len(z)+strength)),len(z))
  for ix in (tr,te):
   arr=[]
   for v in values[ix]:
    p,n=stats.get(v,(prior,0));arr.append((logit(p)-logit(prior),np.log1p(n)/5))
   out.append(np.asarray(arr,float))
 return np.c_[out[0],out[2]],np.c_[out[1],out[3]]
def run(splitlist,C,pca_n=0,use_id=True):
 oof=np.full(len(y),np.nan)
 for tr,te in splitlist:
  mu=tab[tr].mean(0);sd=tab[tr].std(0);sd[sd<1e-6]=1;A=(tab[tr]-mu)/sd;B=(tab[te]-mu)/sd
  if pca_n:
   pm=pooled[tr].mean(0);ps=pooled[tr].std(0);ps[ps<1e-6]=1;p=PCA(pca_n,whiten=True,random_state=20260811).fit((pooled[tr]-pm)/ps);A=np.c_[A,p.transform((pooled[tr]-pm)/ps)];B=np.c_[B,p.transform((pooled[te]-pm)/ps)]
  if use_id:
   ia,ib=target_features(tr,te);A=np.c_[A,ia];B=np.c_[B,ib]
  m=LogisticRegression(C=C,solver='liblinear',max_iter=3000,random_state=20260811).fit(A,y[tr]);oof[te]=m.predict_proba(B)[:,1]
 q=np.isfinite(oof);return {'auc':float(roc_auc_score(y[q],oof[q])),'ap':float(average_precision_score(y[q],oof[q])),'logloss':float(log_loss(y[q],np.clip(oof[q],1e-6,1-1e-6))),'brier':float(brier_score_loss(y[q],oof[q])),'oof':oof}
res={};configs=[('tab96',0,False),('tab96_identity',0,True),('tab96_identity_pca8',8,True),('tab96_identity_pca16',16,True),('tab96_identity_pca32',32,True)]
for name,pn,ui in configs:
 for sn,sp in splits.items():
  best=None
  for C in (.003,.01,.03,.1,.3):
   r=run(sp,C,pn,ui)
   if best is None or r['auc']>best[1]['auc']:best=(C,r)
  C,r=best;res[name+'|'+sn]={'C':C,**{k:v for k,v in r.items() if k!='oof'}};print(f'{name:26s} {sn:16s} C={C:<5} AUC={r["auc"]:.4f} AP={r["ap"]:.4f} LL={r["logloss"]:.4f}',flush=True)
# Conservative selected model: tabular + hierarchical identity. Spatial information is already aggregated in tab96.
fixed={}
for sn,sp in splits.items():
 r=run(sp,.03,0,True);fixed[sn]={k:v for k,v in r.items() if k!='oof'};np.save(snap/f'hierarchical_{sn}_oof.npy',r['oof']);print('FIXED',sn,{k:round(v,5) for k,v in fixed[sn].items()},flush=True)
(snap/'hierarchical_report_v2.json').write_text(json.dumps({'results':res,'selected':'tab96_identity','C':.03,'fixed_metrics':fixed,'identity_encoding':{'collection_prior_strength':30,'model_prior_strength':10,'fold_local':True},'notes':['Pixel classifications and region aggregates are in tab96','Raw spatial PCA retained only if group CV improves','Absolute hue and color preference excluded']},indent=2))
