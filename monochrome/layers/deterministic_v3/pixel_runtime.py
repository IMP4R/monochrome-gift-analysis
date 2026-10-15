"""Small pure NumPy/Pillow fallback extractor for unseen production pairs."""
from __future__ import annotations
from functools import lru_cache
from pathlib import Path
from urllib.parse import quote
import numpy as np
from PIL import Image
SIZE=32
def srgb_to_lab(rgb):
    x=np.asarray(rgb,dtype=np.float32);x=np.where(x<=.04045,x/12.92,((x+.055)/1.055)**2.4)
    M=np.array([[.4124564,.3575761,.1804375],[.2126729,.7151522,.0721750],[.0193339,.1191920,.9503041]],np.float32)
    xyz=x@M.T/np.array([.95047,1.,1.08883],np.float32);d=6/29;f=np.where(xyz>d**3,np.cbrt(xyz),xyz/(3*d*d)+4/29)
    return np.stack([116*f[...,1]-16,500*(f[...,0]-f[...,1]),200*(f[...,1]-f[...,2])],-1)
def wquantile(x,w,q):
    x=np.asarray(x,float).ravel();w=np.asarray(w,float).ravel();m=(w>1e-6)&np.isfinite(x);x=x[m];w=w[m]
    if not len(x): return 0.
    o=np.argsort(x);x=x[o];w=w[o];cw=np.cumsum(w)/w.sum()
    return float(x[min(len(x)-1,np.searchsorted(cw,q))])
def largest_component_share(mask,den):
    a=np.asarray(mask,bool);seen=np.zeros(a.shape,bool);best=0
    for yy,xx in zip(*np.where(a&~seen)):
        if seen[yy,xx]:continue
        stack=[(int(yy),int(xx))];seen[yy,xx]=1;n=0
        while stack:
            y,x=stack.pop();n+=1
            for dy,dx in ((1,0),(-1,0),(0,1),(0,-1)):
                ny,nx=y+dy,x+dx
                if 0<=ny<a.shape[0] and 0<=nx<a.shape[1] and a[ny,nx] and not seen[ny,nx]:seen[ny,nx]=1;stack.append((ny,nx))
        best=max(best,n)
    return best/max(1,den)
@lru_cache(maxsize=2048)
def _assets(root,short,model,backdrop):
    root=Path(root);mp=root/f'modelwebp_lossless1_{short.lower()}_{quote(str(model),safe="")}.webp';bp=root/f'bg_{short.lower()}_{quote(str(backdrop),safe="")}.png'
    rgba=np.asarray(Image.open(mp).convert('RGBA').resize((SIZE,SIZE),Image.Resampling.LANCZOS),np.float32)/255
    bg=np.asarray(Image.open(bp).convert('RGB').resize((SIZE,SIZE),Image.Resampling.LANCZOS),np.float32)/255
    return rgba,bg
def extract_needed(cache_root,short,model,backdrop,mono_features):
    rgba,bg=_assets(str(cache_root),str(short).lower(),str(model),str(backdrop));mrgb=rgba[...,:3];alpha=rgba[...,3];mlab=srgb_to_lab(mrgb);blab=srgb_to_lab(bg);diff=mlab-blab;de=np.linalg.norm(diff,axis=-1);mc=np.linalg.norm(mlab[...,1:],axis=-1);bc=np.linalg.norm(blab[...,1:],axis=-1)
    mh=np.degrees(np.arctan2(mlab[...,2],mlab[...,1]))%360;bh=np.degrees(np.arctan2(blab[...,2],blab[...,1]))%360;hd=np.abs((mh-bh+180)%360-180);fg=alpha>=.05;same=fg&(de<12);tone=fg&~same&(hd<24)&(np.abs(diff[...,0])<42);ach=fg&~same&~tone&(mc<12)&(bc<12);soft=fg&~same&~tone&~ach&(de<40);conf=fg&~same&~tone&~ach&~soft;w=alpha;den=max(1e-6,float(w.sum()))
    tab={k:float((mono_features or {}).get(k,0) or 0) for k in ("weighted_perceptual_fit","weighted_tone_fit","neutral_share","bg_sat")}
    tab.update(delta_e_q10=wquantile(de,w,.10)/100,delta_e_q25=wquantile(de,w,.25)/100,pixel_match_lt_12=float((w*(de<12)).sum()/den),pixel_match_lt_18=float((w*(de<18)).sum()/den),pixel_match_lt_25=float((w*(de<25)).sum()/den),pixel_class_same=float((w*same).sum()/den),pixel_class_same_hue_tone=float((w*tone).sum()/den),pixel_class_achromatic_pair=float((w*ach).sum()/den),pixel_class_soft_accent=float((w*soft).sum()/den),pixel_class_conflict=float((w*conf).sum()/den),conflict_largest_component=float(largest_component_share(conf&(alpha>.25),int((alpha>.25).sum()))))
    yy,xx=np.mgrid[-1:1:complex(SIZE),-1:1:complex(SIZE)];rad=np.sqrt(xx*xx+yy*yy);regions={'center':rad<.52,'border':rad>.82,'top':yy<0,'bottom':yy>=0,'left':xx<0,'right':xx>=0};close=np.exp(-de/24)
    for n,m in regions.items():
        rw=w*m;rd=max(1e-6,float(rw.sum()));tab[f'region_{n}_closeness']=float((rw*close).sum()/rd);tab[f'region_{n}_conflict']=float((rw*conf).sum()/rd)
    return tab
