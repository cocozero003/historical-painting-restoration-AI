
#!/usr/bin/env python3
import argparse, math, random
from pathlib import Path
import numpy as np
from PIL import Image, ImageFilter, ImageDraw
def set_seed(s): random.seed(s); np.random.seed(s)
def random_path(draw,w,h,ctrl=4,th=2):
    pts=[(random.randint(0,w-1),random.randint(0,h-1)) for _ in range(ctrl)]
    for i in range(len(pts)-1):
        a,b=pts[i],pts[i+1]
        for t in np.linspace(0,1,30):
            x=int(a[0]*(1-t)+b[0]*t); y=int(a[1]*(1-t)+b[1]*t)
            if 0<=x<w and 0<=y<h: draw.ellipse((x-th,y-th,x+th,y+th), fill=255)
def loss_mask(H,W,n=6):
    img=Image.new("L",(W,H),0); d=ImageDraw.Draw(img)
    for _ in range(n):
        r=random.randint(20,120); x=random.randint(0,W-1); y=random.randint(0,H-1)
        d.ellipse((x-r,y-r,x+r,y+r), fill=255)
    return (np.array(img.filter(ImageFilter.GaussianBlur(2)))>64).astype(np.uint8)
def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--input_dir", required=True); ap.add_argument("--output_dir", required=True)
    ap.add_argument("--num_per_image", type=int, default=2); ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max_size", type=int, default=1600); a=ap.parse_args()
    set_seed(a.seed)
    inp=Path(a.input_dir); out=Path(a.output_dir)
    out_img=out/"synthetic_damaged"; out_msk=out/"synthetic_masks"
    out_img.mkdir(parents=True, exist_ok=True); out_msk.mkdir(parents=True, exist_ok=True)
    exts={".jpg",".jpeg",".png",".bmp",".tif",".tiff"}
    paths=[p for p in inp.rglob("*") if p.suffix.lower() in exts]
    cnt=0
    for p in paths:
        try: img=Image.open(p).convert("RGB")
        except Exception: continue
        if a.max_size and max(img.size)>a.max_size:
            w,h=img.size; s=a.max_size/float(max(w,h)); img=img.resize((int(w*s),int(h*s)), Image.LANCZOS)
        arr=np.array(img); H,W=arr.shape[:2]
        for k in range(a.num_per_image):
            m=np.zeros((H,W), np.uint8); random_path(ImageDraw.Draw(Image.fromarray(m*255)), W, H, ctrl=5, th=2)
            m |= loss_mask(H,W,n=4)
            dimg=arr.copy(); dimg[m==1]=0
            stem=f"{p.stem}_syn_{k:02d}"
            Image.fromarray(dimg).save(out_img/f"{stem}.png")
            Image.fromarray((m*255).astype(np.uint8)).save(out_msk/f"{stem}.png"); cnt+=1
    print("Wrote", cnt, "synthetic pairs")
if __name__=="__main__": main()
