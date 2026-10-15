import json,hashlib,collections
from pathlib import Path
import numpy as np
from spatial_features import AssetStore,extract_pair,PIXEL_CHANNEL_NAMES
ROOT=Path(__file__).resolve().parents[1];snap=ROOT/'analysis/snapshot_20260811';assets=snap/'server/cache';paths=[snap/'test_votes_v2.jsonl',snap/'test_votes_v4.jsonl']
rows=[];sources={}
for p in paths:
 sources[p.name]={'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'bytes':p.stat().st_size}
 for line in p.read_text().splitlines():
  try:r=json.loads(line)
  except:continue
  if r.get('task')=='swipe_monochrome' and r.get('choice') in ('monochrome','not_monochrome') and (r.get('a') or {}).get('item_key'):rows.append(r)
g=collections.defaultdict(list)
for r in rows:g[r['a']['item_key']].append(r)
examples=[];ties=[]
for key,rr in g.items():
 yes=sum(r['choice']=='monochrome' for r in rr)
 if yes*2==len(rr):ties.append(key);continue
 latest=max(rr,key=lambda r:r.get('ts',0));examples.append((key,latest['a'],int(yes*2>len(rr)),latest.get('ts',0),latest.get('test_schema'),len(rr)))
store=AssetStore(assets);Xp=[];tabs=[];meta=[];missing=[];names=None
for idx,(key,a,y,ts,source,nvotes) in enumerate(examples):
 try:pix,tab=extract_pair(a,store)
 except Exception as e:missing.append({'key':key,'error':repr(e)});continue
 if names is None:names=list(tab)
 if list(tab)!=names:raise RuntimeError('feature order changed')
 Xp.append(pix);tabs.append([tab[n] for n in names]);meta.append((key,y,a.get('collection',''),a.get('short',''),a.get('model',''),a.get('backdrop',''),ts,source or '',a.get('swipe_sampling_strategy') or 'legacy',nvotes,float(a.get('mono_probability') or 0),float(a.get('base_score') or 0)))
 if (idx+1)%200==0:print('built',idx+1,flush=True)
arr=np.array(meta,dtype=object);out=snap/'spatial_dataset_v2.npz';np.savez_compressed(out,pixels=np.stack(Xp).astype(np.float16),tabular=np.asarray(tabs,np.float32),y=arr[:,1].astype(np.int8),item_key=arr[:,0].astype(str),collection=arr[:,2].astype(str),short=arr[:,3].astype(str),model=arr[:,4].astype(str),backdrop=arr[:,5].astype(str),ts=arr[:,6].astype(float),source=arr[:,7].astype(str),strategy=arr[:,8].astype(str),vote_count=arr[:,9].astype(np.int16),baseline_probability=arr[:,10].astype(np.float32),baseline_score=arr[:,11].astype(np.float32))
report={'dataset_version':'spatial-dataset-v2','rows':len(meta),'positive':int(arr[:,1].astype(int).sum()),'negative':int(len(meta)-arr[:,1].astype(int).sum()),'ties_excluded':len(ties),'missing':missing,'pixel_size':32,'pixel_channels':PIXEL_CHANNEL_NAMES,'tabular_feature_names':names,'sources':sources,'principles':['No BCP or YouGov','No absolute hue feature','Pixel classes computed before model fitting','Append-only labels copied, never modified']}
(snap/'spatial_dataset_v2.meta.json').write_text(json.dumps(report,ensure_ascii=False,indent=2));print(json.dumps({k:report[k] for k in ('rows','positive','negative','ties_excluded','missing')},indent=2));print('pixels',np.stack(Xp).shape,'tabular',np.asarray(tabs).shape,'features',len(names),'output_mb',round(out.stat().st_size/1e6,2))
