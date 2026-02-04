import cv2
import numpy as np
from pathlib import Path

# gestures/utils.py -> project root is one level above this file
ROOT_DIR = Path(__file__).resolve().parents[1]
ASSETS_DIR = ROOT_DIR / "assets"

# Cache for loaded images to avoid reloading every frame
_IMAGE_CACHE = {}  # path -> image


def load_image(filename: str):
    """
    Loads an image (PNG/JPG) from the assets/ directory.
    Images are cached to avoid reloading them every frame.
    """
    path = (ASSETS_DIR / filename).resolve()
    key = str(path)

    if key in _IMAGE_CACHE:
        return _IMAGE_CACHE[key]

    img = cv2.imread(key, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"Image not found: {path}")
    if img.ndim != 3 or img.shape[2] not in (3, 4):
        raise ValueError(f"Invalid image format (expected BGR/BGRA): {path}")

    _IMAGE_CACHE[key] = img
    return img


def overlay_image(bg_bgr, overlay_img, x, y, alpha_fallback=0.85):
    """
    Overlays an image on top of a background frame at position (x, y).

    Supports:
    - BGRA images with alpha channel
    - BGR images using a fallback alpha value
    """
    H, W = bg_bgr.shape[:2]
    h, w = overlay_img.shape[:2]

    # Completely outside the frame
    if x >= W or y >= H:
        return bg_bgr

    # Clip overlay size to fit inside the frame
    w = min(w, W - x)
    h = min(h, H - y)
    if w <= 0 or h <= 0:
        return bg_bgr

    roi = bg_bgr[y:y+h, x:x+w]

    if overlay_img.shape[2] == 4:
        # Overlay has an alpha channel
        ov = overlay_img[:h, :w]
        a = ov[:, :, 3:4] / 255.0
        fg = ov[:, :, :3].astype(np.float32)
        bg = roi.astype(np.float32)
        roi[:] = (fg * a + bg * (1 - a)).astype(np.uint8)
    else:
        # No alpha channel: use fallback transparency
        ov = overlay_img[:h, :w].astype(np.float32)
        bg = roi.astype(np.float32)
        roi[:] = (ov * alpha_fallback + bg * (1 - alpha_fallback)).astype(np.uint8)

    bg_bgr[y:y+h, x:x+w] = roi
    return bg_bgr
