
import cv2, numpy as np
def inpaint_opencv(img_rgb, mask, method="telea", radius=3):
    mask255 = (mask.astype(np.uint8))*255
    bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    if method=="telea":
        out = cv2.inpaint(bgr, mask255, radius, cv2.INPAINT_TELEA)
    else:
        out = cv2.inpaint(bgr, mask255, radius, cv2.INPAINT_NS)
    return cv2.cvtColor(out, cv2.COLOR_BGR2RGB)
