
import gradio as gr
from src.utils.common import overlay_mask
from src.detectors.opencv_crack import detect_damage_mask as opencv_mask
from src.inpainters.opencv_inpaint import inpaint_opencv
def unet_ok():
    try:
        from src.detectors.unet_detector import available as ok
        return ok()
    except Exception:
        return False
def try_lama(img_rgb, mask):
    try:
        from src.inpainters.lama_inpaint import LamaInpainter
        lama=LamaInpainter()
        if lama.available(): return lama.inpaint(img_rgb, mask)
    except Exception: pass
    return None
def restore(img, detector, use_lama, sensitivity, method, radius):
    if img is None: return None, None, None
    if detector=="U-Net" and unet_ok():
        try:
            from src.detectors.unet_detector import detect_damage_mask as um
            mask=um(img)
        except Exception:
            mask=opencv_mask(img, sensitivity=sensitivity)
    else:
        mask=opencv_mask(img, sensitivity=sensitivity)
    overlay=overlay_mask(img, mask, 0.45)
    restored = try_lama(img, mask) if use_lama else None
    if restored is None: restored=inpaint_opencv(img, mask, method=method, radius=radius)
    return img, overlay, restored
with gr.Blocks(title="Historic Painting Restoration") as demo:
    gr.Markdown("# Historic Painting Restoration")
    with gr.Row():
        with gr.Column():
            inp=gr.Image(type="numpy", label="Upload painting")
            det=gr.Radio(["OpenCV","U-Net"], value="OpenCV", label="Damage detector")
            use_lama=gr.Checkbox(label="Use LaMa (if weights present)", value=False)
            sens=gr.Slider(0.2,0.95,value=0.6,step=0.05,label="OpenCV detector sensitivity")
            method=gr.Radio(["telea","ns"], value="telea", label="OpenCV inpaint method")
            rad=gr.Slider(1,9,value=3,step=1,label="Inpaint radius")
            btn=gr.Button("Restore")
        with gr.Column():
            o1=gr.Image(type="numpy", label="Original")
            o2=gr.Image(type="numpy", label="Damage overlay")
            o3=gr.Image(type="numpy", label="Restored")
    btn.click(fn=restore, inputs=[inp, det, use_lama, sens, method, rad], outputs=[o1,o2,o3])
if __name__=="__main__": demo.launch()
