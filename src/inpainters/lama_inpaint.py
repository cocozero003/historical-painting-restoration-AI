
import os, numpy as np
class LamaInpainter:
    def __init__(self, weights_dir="models/lama", device=None):
        try:
            import torch  # noqa
        except Exception:
            self.model=None; self.ok=False; return
        import torch
        self.torch=torch; self.device=device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model=None; self.ok=False
        mp=os.path.join(weights_dir,"big-lama.pt")
        if os.path.exists(mp):
            try:
                self.model=torch.jit.load(mp, map_location=self.device); self.model.eval(); self.ok=True
            except Exception:
                self.model=None; self.ok=False
    def available(self): return self.ok and (self.model is not None)
    def inpaint(self, img_rgb, mask):
        if not self.available(): raise RuntimeError("LaMa weights missing")
        t=self.torch
        x=t.from_numpy(img_rgb.astype(np.float32)/255.).permute(2,0,1).unsqueeze(0).to(self.device)
        m=t.from_numpy(mask.astype(np.float32)).unsqueeze(0).unsqueeze(0).to(self.device)
        with t.no_grad():
            y=self.model(x,m) if callable(self.model) else x
        return (y.squeeze(0).permute(1,2,0).cpu().numpy()*255).clip(0,255).astype(np.uint8)
