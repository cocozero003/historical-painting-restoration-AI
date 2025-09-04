
import argparse
from pathlib import Path
from src.utils.common import imread_rgb, imwrite_rgb, overlay_mask
from src.inpainters.opencv_inpaint import inpaint_opencv
from src.detectors.opencv_crack import detect_damage_mask as opencv_mask
def run_pipeline(input_path, output_path, detector="opencv", use_lama=False, sensitivity=0.6):
    img = imread_rgb(input_path)
    if detector=="unet":
        try:
            from src.detectors.unet_detector import detect_damage_mask as unet_mask, available as unet_ok
            mask = unet_mask(img) if unet_ok() else opencv_mask(img, sensitivity=sensitivity)
        except Exception:
            mask = opencv_mask(img, sensitivity=sensitivity)
    else:
        mask = opencv_mask(img, sensitivity=sensitivity)
    restored=None
    if use_lama:
        try:
            from src.inpainters.lama_inpaint import LamaInpainter
            lama=LamaInpainter()
            if lama.available(): restored=lama.inpaint(img, mask)
        except Exception: restored=None
    if restored is None:
        restored = inpaint_opencv(img, mask, method="telea", radius=3)
    out = Path(output_path); out.parent.mkdir(parents=True, exist_ok=True)
    imwrite_rgb(out, restored)
    imwrite_rgb(out.parent / (out.stem + "_overlay.png"), overlay_mask(img, mask, alpha=0.45))
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True); ap.add_argument("--output", required=True)
    ap.add_argument("--detector", choices=["opencv","unet"], default="opencv")
    ap.add_argument("--use_lama", action="store_true"); ap.add_argument("--sensitivity", type=float, default=0.6)
    a = ap.parse_args(); run_pipeline(a.input, a.output, detector=a.detector, use_lama=a.use_lama, sensitivity=a.sensitivity)
