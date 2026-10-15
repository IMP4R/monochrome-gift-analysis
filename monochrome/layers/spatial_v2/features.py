"""Task-specific 32x32 perceptual features for monochrome judgement.
No BCP/YouGov or absolute hue is used: this layer models visual coherence only.
"""
from __future__ import annotations
from pathlib import Path
from urllib.parse import quote
import math,zlib
import numpy as np
from PIL import Image
from scipy import ndimage

SIZE=32
BASE_FEATURE_NAMES=[
 'base','classic','rainbow','neutral_share','matched_share','full_share','mismatch_share',
 'weighted_hue_fit','min_hue_dist','weighted_perceptual_fit','min_delta_e','weighted_delta_e',
 'weighted_tone_fit','min_lstar_gap','center_delta_e','center_lstar_gap','dominant_share',
 'palette_entropy','bg_lstar','bg_chroma','bg_sat','model_achromatic','bg_achromatic']
PIXEL_CHANNEL_NAMES=[
 'alpha','alpha_edge','signed_lstar_delta','absolute_lstar_delta','chroma_delta',
 'hue_closeness','perceptual_delta','pixel_same','pixel_same_hue_tone',
 'pixel_achromatic_pair','pixel_soft_accent','pixel_conflict']


def srgb_to_lab(rgb):
 x=np.asarray(rgb,dtype=np.float32)
 x=np.where(x<=.04045,x/12.92,((x+.055)/1.055)**2.4)
 M=np.array([[.4124564,.3575761,.1804375],[.2126729,.7151522,.0721750],[.0193339,.1191920,.9503041]],np.float32)
 xyz=x@M.T; xyz=xyz/np.array([.95047,1.,1.08883],np.float32)
 d=6/29
 f=np.where(xyz>d**3,np.cbrt(xyz),xyz/(3*d*d)+4/29)
 return np.stack([116*f[...,1]-16,500*(f[...,0]-f[...,1]),200*(f[...,1]-f[...,2])],-1)


def hex_rgb(value):
 h=str(value or '').lstrip('#')
 if len(h)!=6:return None
 try:return np.array([int(h[i:i+2],16)/255 for i in (0,2,4)],np.float32)
 except:return None


def wentropy(w):
 w=np.asarray(w,float);w=w[w>1e-12]
 if not len(w):return 0.
 w=w/w.sum();return float(-(w*np.log(w)).sum())


def wstats(x,w):
 x=np.asarray(x,float).ravel();w=np.asarray(w,float).ravel();m=np.isfinite(x)&np.isfinite(w)&(w>1e-6)
 if not m.any():return (0.,0.,0.,0.,0.,0.,0.)
 x=x[m];w=w[m];w=w/w.sum();order=np.argsort(x);xs=x[order];ws=w[order];cw=np.cumsum(ws)
 q=[float(xs[min(len(xs)-1,np.searchsorted(cw,v))]) for v in (.1,.25,.5,.75,.9)]
 mean=float((x*w).sum());std=float(np.sqrt(((x-mean)**2*w).sum()))
 return (mean,std,*q)


def hasler_colorfulness(rgb,weights=None):
 x=np.asarray(rgb,float)
 if weights is not None:
  m=np.asarray(weights)>0.05
  if not m.any():return 0.
  x=x[m]
 r,g,b=x[...,0]*255,x[...,1]*255,x[...,2]*255
 rg=r-g;yb=.5*(r+g)-b
 return float(np.hypot(rg.std(),yb.std())+.3*np.hypot(rg.mean(),yb.mean()))/100.


def image_entropy(rgb,mask=None):
 x=np.asarray(rgb,float)
 if mask is not None:x=x[np.asarray(mask)>0.05]
 else:x=x.reshape(-1,3)
 if not len(x):return 0.
 # Perceptual-ish 4-bit RGB bins; normalized Shannon entropy.
 q=np.clip((x*15).astype(np.int32),0,15);ids=q[:,0]*256+q[:,1]*16+q[:,2]
 c=np.bincount(ids,minlength=4096);p=c[c>0]/len(ids)
 return float(-(p*np.log(p)).sum()/math.log(min(len(ids),4096))) if len(ids)>1 else 0.


def palette_features(palette):
 rows=[]
 for p in palette or []:
  rgb=hex_rgb(p.get('hex'))
  if rgb is None:continue
  share=float(p.get('share',p.get('share_all',0)) or 0)
  rows.append((rgb,max(0.,share)))
 if not rows:return {k:0. for k in PALETTE_NAMES}
 rgbs=np.stack([r for r,_ in rows]);w=np.array([x for _,x in rows],float)
 if w.sum()<=0:w[:]=1
 w/=w.sum();lab=srgb_to_lab(rgbs);L=lab[:,0];C=np.linalg.norm(lab[:,1:],axis=1)
 pair=[];pairw=[];hd=[]
 hues=np.degrees(np.arctan2(lab[:,2],lab[:,1]))%360
 for i in range(len(rows)):
  for j in range(i+1,len(rows)):
   pair.append(float(np.linalg.norm(lab[i]-lab[j])));pairw.append(float(w[i]*w[j]));hd.append(float(abs((hues[i]-hues[j]+180)%360-180)))
 if pair:
  pw=np.array(pairw);pw/=pw.sum();pa=np.array(pair);de_mean=float((pa*pw).sum());de_std=float(np.sqrt(((pa-de_mean)**2*pw).sum()));de_min=float(pa.min());de_max=float(pa.max());hspan=max(hd)
 else:de_mean=de_std=de_min=de_max=hspan=0.
 hvec=np.sum(w*np.exp(1j*np.radians(hues)));hdisp=float(1-abs(hvec))
 ent=wentropy(w);ent_norm=ent/math.log(max(2,len(w)))
 lm=float((L*w).sum());lstd=float(np.sqrt(((L-lm)**2*w).sum()));cm=float((C*w).sum());cstd=float(np.sqrt(((C-cm)**2*w).sum()))
 contrast=.6*min(1,de_max/100)+.4*min(1,(float(L.max()-L.min()) if len(L) else 0)/100)
 variegation=.4*min(1,de_mean/80)+.3*ent_norm+.3*min(1,hspan/180)
 vals=[len(rows),ent_norm,math.exp(ent),float((w*w).sum()),de_mean/100,de_std/100,de_min/100,de_max/100,lstd/50,(float(L.max()-L.min()) if len(L) else 0)/100,cm/100,cstd/50,(float(C.max()-C.min()) if len(C) else 0)/100,hdisp,hspan/180,float(w[C<12].sum()),float(w.min()),contrast,variegation]
 return dict(zip(PALETTE_NAMES,map(float,vals)))

PALETTE_NAMES=['palette_color_count','palette_entropy_exact','palette_effective_colors','palette_hhi','palette_pair_delta_mean','palette_pair_delta_std','palette_pair_delta_min','palette_pair_delta_max','palette_lstar_std','palette_lstar_range','palette_chroma_mean','palette_chroma_std','palette_chroma_range','palette_hue_dispersion','palette_hue_span','palette_neutral_share_exact','palette_smallest_accent_share','palette_contrast','palette_variegation']

class AssetStore:
 def __init__(self,root):self.root=Path(root);self.models={};self.bgs={}
 def model(self,short,name):
  k=(short,name)
  if k not in self.models:
   p=self.root/(f'modelwebp_lossless1_{short.lower()}_{quote(str(name),safe="")}.webp')
   if not p.exists():raise FileNotFoundError(p)
   self.models[k]=np.asarray(Image.open(p).convert('RGBA').resize((SIZE,SIZE),Image.Resampling.LANCZOS),np.float32)/255
  return self.models[k]
 def bg(self,short,name):
  k=(short,name)
  if k not in self.bgs:
   p=self.root/(f'bg_{short}_{quote(str(name),safe="")}.png')
   if not p.exists():raise FileNotFoundError(p)
   self.bgs[k]=np.asarray(Image.open(p).convert('RGB').resize((SIZE,SIZE),Image.Resampling.LANCZOS),np.float32)/255
  return self.bgs[k]


def extract_pair(a,store):
 rgba=store.model(a['short'],a['model']);bg=store.bg(a['short'],a['backdrop']);mrgb=rgba[...,:3];alpha=rgba[...,3]
 mlab=srgb_to_lab(mrgb);blab=srgb_to_lab(bg);diff=mlab-blab;de=np.linalg.norm(diff,axis=-1);mc=np.linalg.norm(mlab[...,1:],axis=-1);bc=np.linalg.norm(blab[...,1:],axis=-1)
 mh=np.degrees(np.arctan2(mlab[...,2],mlab[...,1]))%360;bh=np.degrees(np.arctan2(blab[...,2],blab[...,1]))%360;hd=np.abs((mh-bh+180)%360-180)
 fg=alpha>=.05;same=fg&(de<12);tone=fg&~same&(hd<24)&(np.abs(diff[...,0])<42);ach=fg&~same&~tone&(mc<12)&(bc<12);soft=fg&~same&~tone&~ach&(de<40);conf=fg&~same&~tone&~ach&~soft
 gy,gx=np.gradient(alpha);edge=np.sqrt(gx*gx+gy*gy)
 pix=np.stack([alpha,np.clip(edge*4,0,1),np.clip(diff[...,0]/60,-1,1)*alpha,np.clip(np.abs(diff[...,0])/60,0,1)*alpha,np.clip((mc-bc)/80,-1,1)*alpha,(1-hd/180)*alpha,np.clip(de/100,0,1.5)*alpha,same.astype(float),tone.astype(float),ach.astype(float),soft.astype(float),conf.astype(float)],0).astype(np.float32)
 tab={n:float((a.get('mono_features') or {}).get(n,0) or 0) for n in BASE_FEATURE_NAMES};tab.update(palette_features(a.get('model_palette') or []))
 w=alpha;dmean,dstd,dq10,dq25,dq50,dq75,dq90=wstats(de,w)
 tab.update({'pixel_occupancy':float(alpha.mean()),'pixel_hard_occupancy':float((alpha>.5).mean()),'alpha_edge_density':float(edge.mean()),'delta_e_mean':dmean/100,'delta_e_std':dstd/100,'delta_e_q10':dq10/100,'delta_e_q25':dq25/100,'delta_e_q50':dq50/100,'delta_e_q75':dq75/100,'delta_e_q90':dq90/100})
 den=max(1e-6,float(w.sum()))
 for t in (8,12,18,25,35,50):tab[f'pixel_match_lt_{t}']=float((w*(de<t)).sum()/den)
 for n,m in [('same',same),('same_hue_tone',tone),('achromatic_pair',ach),('soft_accent',soft),('conflict',conf)]:tab['pixel_class_'+n]=float((w*m).sum()/den)
 yy,xx=np.mgrid[-1:1:complex(SIZE),-1:1:complex(SIZE)];rad=np.sqrt(xx*xx+yy*yy)
 regions={'center':rad<.52,'border':rad>.82,'top':yy<0,'bottom':yy>=0,'left':xx<0,'right':xx>=0,'tl':(xx<0)&(yy<0),'tr':(xx>=0)&(yy<0),'bl':(xx<0)&(yy>=0),'br':(xx>=0)&(yy>=0)}
 close=np.exp(-de/24)
 for n,m in regions.items():
  rw=w*m;rd=max(1e-6,float(rw.sum()));tab['region_'+n+'_closeness']=float((rw*close).sum()/rd);tab['region_'+n+'_conflict']=float((rw*conf).sum()/rd)
 cx=float((w*xx).sum()/den);cy=float((w*yy).sum()/den);tab.update({'shape_centroid_x':cx,'shape_centroid_y':cy,'shape_spread':float(np.sqrt((w*((xx-cx)**2+(yy-cy)**2)).sum()/den)),'shape_h_symmetry':float(1-np.mean(np.abs(alpha-alpha[:,::-1]))),'shape_v_symmetry':float(1-np.mean(np.abs(alpha-alpha[::-1,:])))})
 lbl,nc=ndimage.label(conf&(alpha>.25));sizes=np.bincount(lbl.ravel())[1:] if nc else np.array([]);tab['conflict_components']=min(1.,nc/10);tab['conflict_largest_component']=float(sizes.max()/max(1,(alpha>.25).sum())) if len(sizes) else 0.
 comp=mrgb*alpha[...,None]+bg*(1-alpha[...,None]);tab['model_colorfulness']=hasler_colorfulness(mrgb,alpha);tab['composite_colorfulness']=hasler_colorfulness(comp);tab['foreground_color_entropy']=image_entropy(mrgb,alpha);tab['composite_color_entropy']=image_entropy(comp)
 cg=np.mean(np.sqrt(np.sum(np.diff(comp,axis=0)**2,axis=-1)))+np.mean(np.sqrt(np.sum(np.diff(comp,axis=1)**2,axis=-1)));tab['composite_gradient_energy']=float(cg/2)
 raw=np.clip(comp*255,0,255).astype(np.uint8).tobytes();tab['composite_compression_complexity']=len(zlib.compress(raw,9))/len(raw)
 return pix,tab

# Stable feature order, discovered once from a synthetic empty record at import is avoided;
# builder records the exact ordered keys returned by the first real sample.
