"""
Synthetic degradation generator.

Takes clean images and produces labeled degraded variants so we have a
training/evaluation set without needing a large pre-labeled dataset.
Each function returns a new BGR uint8 image; none mutate the input.
"""

import cv2
import numpy as np

# Severity presets. Widened from the original values after evaluation showed
# adjacent severities (esp. corruption) weren't separable in feature space --
# see EVALUATION.md for the before/after comparison.
BLUR_SIGMAS = {"low": 1.0, "medium": 3.2, "high": 7.0}
EXPOSURE_GAMMAS_OVER = {"low": 1.7, "medium": 2.6, "high": 3.8}
EXPOSURE_GAMMAS_UNDER = {"low": 0.6, "medium": 0.38, "high": 0.18}
NOISE_SIGMAS = {"low": 10, "medium": 25, "high": 55}
JPEG_QUALITIES = {"low": 35, "medium": 15, "high": 5}
CORRUPTION_BLOCKS = {"low": 2, "medium": 6, "high": 14}


def add_blur(img: np.ndarray, sigma: float) -> np.ndarray:
    k = max(3, int(sigma * 4) | 1)  # odd kernel size scaled to sigma
    return cv2.GaussianBlur(img, (k, k), sigmaX=sigma)


def add_exposure(img: np.ndarray, gamma: float) -> np.ndarray:
    """gamma > 1 -> brighter (use for overexposure), gamma < 1 -> darker
    (use for underexposure). Standard gamma LUT on the 0-255 range."""
    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in range(256)]).astype(np.uint8)
    return cv2.LUT(img, table)


def add_noise(img: np.ndarray, sigma: float) -> np.ndarray:
    noise = np.random.normal(0, sigma, img.shape).astype(np.float32)
    noisy = img.astype(np.float32) + noise
    return np.clip(noisy, 0, 255).astype(np.uint8)


def add_corruption(img: np.ndarray, jpeg_quality: int, n_blocks: int = 6) -> np.ndarray:
    """Simulates severe degradation/corruption: heavy JPEG re-compression
    plus a handful of randomly placed corrupted (noise-filled) blocks."""
    ok, enc = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])
    corrupted = cv2.imdecode(enc, cv2.IMREAD_COLOR) if ok else img.copy()

    h, w = corrupted.shape[:2]
    for _ in range(n_blocks):
        bw, bh = np.random.randint(w // 12, w // 5), np.random.randint(h // 12, h // 5)
        x, y = np.random.randint(0, max(1, w - bw)), np.random.randint(0, max(1, h - bh))
        corrupted[y:y + bh, x:x + bw] = np.random.randint(0, 255, (bh, bw, 3), dtype=np.uint8)
    return corrupted


def generate_labeled_variant(img: np.ndarray, degradation: str, severity: str) -> np.ndarray:
    """degradation in {'blur','overexposure','underexposure','noise','corruption'}"""
    if degradation == "blur":
        return add_blur(img, BLUR_SIGMAS[severity])
    if degradation == "overexposure":
        return add_exposure(img, EXPOSURE_GAMMAS_OVER[severity])
    if degradation == "underexposure":
        return add_exposure(img, EXPOSURE_GAMMAS_UNDER[severity])
    if degradation == "noise":
        return add_noise(img, NOISE_SIGMAS[severity])
    if degradation == "corruption":
        return add_corruption(img, JPEG_QUALITIES[severity], n_blocks=CORRUPTION_BLOCKS[severity])
    raise ValueError(f"unknown degradation type: {degradation}")


DEGRADATION_TYPES = ["blur", "overexposure", "underexposure", "noise", "corruption"]
SEVERITIES = ["low", "medium", "high"]
