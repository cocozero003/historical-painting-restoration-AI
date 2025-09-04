
import cv2, numpy as np
def detect_damage_mask(img_rgb, sensitivity=0.6):
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    clahe = cv2.createCLAHE(2.0, (8,8)).apply(gray)
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    tophat = cv2.morphologyEx(clahe, cv2.MORPH_TOPHAT, k)
    blackhat = cv2.morphologyEx(clahe, cv2.MORPH_BLACKHAT, k)
    enhance = cv2.addWeighted(tophat, 0.6, blackhat, 0.6, 0)
    edges = cv2.Canny(enhance, 50, 150)
    k2 = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    edges = cv2.dilate(edges, k2, 1)
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, k2, 1)
    thr = cv2.adaptiveThreshold(enhance, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 15, 2)
    mask = cv2.bitwise_or(edges, thr)
    mask = cv2.medianBlur(mask, 3)
    if sensitivity < 1.0:
        k3 = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
        mask = cv2.erode(mask, k3, int((1.0-sensitivity)*2))
    return (mask>0).astype(np.uint8)
