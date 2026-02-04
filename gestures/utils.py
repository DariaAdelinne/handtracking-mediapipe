import cv2
import numpy as np
from pathlib import Path

# gestures/utils.py  -> proiectul e cu un nivel mai sus
ROOT_DIR = Path(__file__).resolve().parents[1]
ASSETS_DIR = ROOT_DIR / "assets"

_IMAGE_CACHE = {}  # path -> image


def load_image(filename: str):
    """
    Încarcă o imagine (png/jpg) din folderul assets/.
    Cache-uită ca să nu o reîncarce în fiecare frame.
    """
    path = (ASSETS_DIR / filename).resolve()
    key = str(path)

    if key in _IMAGE_CACHE:
        return _IMAGE_CACHE[key]

    img = cv2.imread(key, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"Nu găsesc imaginea: {path}")
    if img.ndim != 3 or img.shape[2] not in (3, 4):
        raise ValueError(f"Imagine invalidă (trebuie BGR/BGRA): {path}")

    _IMAGE_CACHE[key] = img
    return img


def overlay_image(bg_bgr, overlay_img, x, y, alpha_fallback=0.85):
    H, W = bg_bgr.shape[:2]
    h, w = overlay_img.shape[:2]

    if x >= W or y >= H:
        return bg_bgr

    w = min(w, W - x)
    h = min(h, H - y)
    if w <= 0 or h <= 0:
        return bg_bgr

    roi = bg_bgr[y:y+h, x:x+w]

    if overlay_img.shape[2] == 4:
        ov = overlay_img[:h, :w]
        a = ov[:, :, 3:4] / 255.0
        fg = ov[:, :, :3].astype(np.float32)
        bg = roi.astype(np.float32)
        roi[:] = (fg * a + bg * (1 - a)).astype(np.uint8)
    else:
        ov = overlay_img[:h, :w].astype(np.float32)
        bg = roi.astype(np.float32)
        roi[:] = (ov * alpha_fallback + bg * (1 - alpha_fallback)).astype(np.uint8)

    bg_bgr[y:y+h, x:x+w] = roi
    return bg_bgr
