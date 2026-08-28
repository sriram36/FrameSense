"""
Engineered image-quality feature extraction.

All functions take a BGR uint8 image (as loaded by cv2.imread) and return
plain Python floats so results are JSON-serializable without extra work.
"""

import cv2
import numpy as np

FEATURE_NAMES = ["sharpness", "brightness", "contrast", "noise_estimate", "saturation", "blockiness", "edge_density"]


def compute_sharpness(gray: np.ndarray) -> float:
    """Variance of the Laplacian. Higher = sharper. Blurry images collapse
    toward 0 because a blurred image has few strong edges."""
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def compute_brightness(gray: np.ndarray) -> float:
    """Mean luminance, normalized to [0, 1]. Near 0 = underexposed,
    near 1 = overexposed."""
    return float(gray.mean() / 255.0)


def compute_contrast(gray: np.ndarray) -> float:
    """Std dev of luminance, normalized to [0, 1]. Low contrast often
    accompanies over/underexposure or heavy compression loss."""
    return float(gray.std() / 255.0)


def compute_noise(gray: np.ndarray) -> float:
    """Fast noise-sigma estimator (Immerkaer's method): convolve with a
    Laplacian-like kernel that cancels smooth structure, leaving mostly
    noise energy behind, then normalize by image area.
    Reference: J. Immerkaer, 'Fast Noise Variance Estimation', 1996.
    """
    h, w = gray.shape
    kernel = np.array([[1, -2, 1], [-2, 4, -2], [1, -2, 1]], dtype=np.float64)
    conv = cv2.filter2D(gray.astype(np.float64), -1, kernel)
    sigma = np.sum(np.abs(conv)) * np.sqrt(0.5 * np.pi) / (6 * (w - 2) * (h - 2))
    return float(sigma / 255.0)


def compute_saturation(bgr: np.ndarray) -> float:
    """Mean saturation channel from HSV, normalized to [0, 1]."""
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    return float(hsv[:, :, 1].mean() / 255.0)


def compute_blockiness(gray: np.ndarray) -> float:
    """JPEG-style blocking-artifact score: ratio of pixel discontinuity at
    8x8 block boundaries vs. discontinuity elsewhere. Heavy re-compression
    and block corruption both inflate this well above 1.0; natural images
    sit close to 1.0. Aimed at separating corruption severities, which the
    other features alone don't discriminate well (see EVALUATION.md)."""
    g = gray.astype(np.float64)
    h, w = g.shape
    col_diffs = np.abs(np.diff(g, axis=1))  # h x (w-1)
    boundary_cols = np.arange(7, w - 1, 8)
    if len(boundary_cols) == 0:
        return 1.0
    boundary_mean = col_diffs[:, boundary_cols].mean()
    all_mean = col_diffs.mean()
    nonboundary_mean = (all_mean * col_diffs.shape[1] - boundary_mean * len(boundary_cols)) / max(
        1, col_diffs.shape[1] - len(boundary_cols)
    )
    return float(boundary_mean / (nonboundary_mean + 1e-6))


def compute_edge_density(gray: np.ndarray) -> float:
    """Fraction of pixels that are Canny edges. A second, independent
    sharpness-adjacent signal (Laplacian variance is sensitive to outliers;
    edge density is a more stable, bounded complement)."""
    edges = cv2.Canny(gray, 50, 150)
    return float(np.mean(edges > 0))


def extract_features(bgr: np.ndarray) -> dict:
    """Extract the full engineered feature set from a BGR image.
    Returns a dict keyed by FEATURE_NAMES (order matters for model input)."""
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    return {
        "sharpness": compute_sharpness(gray),
        "brightness": compute_brightness(gray),
        "contrast": compute_contrast(gray),
        "noise_estimate": compute_noise(gray),
        "saturation": compute_saturation(bgr),
        "blockiness": compute_blockiness(gray),
        "edge_density": compute_edge_density(gray),
    }


def feature_vector(bgr: np.ndarray) -> np.ndarray:
    """Same as extract_features but returns a fixed-order numpy vector,
    for feeding directly into a scikit-learn model."""
    feats = extract_features(bgr)
    return np.array([feats[name] for name in FEATURE_NAMES], dtype=np.float64)
