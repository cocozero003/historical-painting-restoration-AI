
import os, numpy as np
def available(weights_path="models/unet/damage_mask_unet.pth"):
    try:
        import torch  # noqa
    except Exception:
        return False
    return os.path.exists(weights_path)
def detect_damage_mask(img_rgb, weights_path="models/unet/damage_mask_unet.pth", threshold=0.5):
    import torch, torch.nn.functional as F
    from src.models.unet import UNet
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = UNet().to(device)
    state = torch.load(weights_path, map_location=device)
    model.load_state_dict(state); model.eval()
    arr = img_rgb.astype(np.float32)/255.0
    t = torch.from_numpy(arr).permute(2,0,1).unsqueeze(0).to(device)
    with torch.no_grad():
        m = torch.sigmoid(model(t))[0,0].cpu().numpy()
    return (m>threshold).astype(np.uint8)
