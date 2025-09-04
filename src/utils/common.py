
import cv2, numpy as np
from pathlib import Path
def imread_rgb(path):
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None: raise FileNotFoundError(f"Cannot read image: {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
def imwrite_rgb(path, img_rgb):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR))
def overlay_mask(img_rgb, mask, alpha=0.45):
    red = np.zeros_like(img_rgb); red[...,0]=255
    blend = (img_rgb*(1-alpha) + red*alpha).astype(np.uint8)
    return np.where(mask[...,None].astype(bool), blend, img_rgb)
