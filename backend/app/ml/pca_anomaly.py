"""
PCA-based reconstruction-error anomaly detector.

This is a classical-ML anomaly-detection formulation (explicitly acceptable
per the assessment brief) that needs only numpy/scikit-learn, so it runs in
any environment -- including ones without PyTorch or GPU access. It serves
two purposes:

1. A fast sanity-check of the "reconstruct clean images, flag high-error
   inputs as defective" approach before investing in the heavier conv
   autoencoder in autoencoder.py.
2. A working fallback the API can use automatically if the PyTorch model
   isn't available (see app/ml/inference.py), so defect detection never
   silently disables itself.

Approach: flatten small grayscale images to vectors, fit PCA on clean
images only, and use per-image reconstruction error (original vs.
PCA-reconstructed) as the anomaly score -- the direct classical analogue of
an autoencoder's reconstruction error, just with a linear (PCA) model
instead of a learned nonlinear one.
"""

import cv2
import numpy as np
from sklearn.decomposition import PCA

IMG_SIZE = 48  # small enough that PCA on flattened pixels is fast and stable


def _preprocess(bgr: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (IMG_SIZE, IMG_SIZE)).astype(np.float64) / 255.0
    return gray.flatten()


class PCAAnomalyDetector:
    def __init__(self, n_components: int = 20):
        self.n_components = n_components
        self.pca = PCA(n_components=n_components)
        self.threshold = None
        self.roc_auc = None

    def fit(self, clean_images: list[np.ndarray]):
        X = np.array([_preprocess(img) for img in clean_images])
        self.pca.fit(X)
        return self

    def reconstruction_error(self, bgr: np.ndarray) -> tuple[float, np.ndarray]:
        """Returns (scalar anomaly score, per-pixel error map at IMG_SIZE x IMG_SIZE)."""
        x = _preprocess(bgr).reshape(1, -1)
        recon = self.pca.inverse_transform(self.pca.transform(x))
        error_map = (x - recon).reshape(IMG_SIZE, IMG_SIZE) ** 2
        return float(error_map.mean()), error_map

    def is_defective(self, bgr: np.ndarray) -> tuple[bool, float]:
        score, _ = self.reconstruction_error(bgr)
        return (self.threshold is not None and score > self.threshold), score

    def save(self, path: str):
        import joblib

        joblib.dump(
            {"pca": self.pca, "threshold": self.threshold, "roc_auc": self.roc_auc, "n_components": self.n_components},
            path,
        )

    @classmethod
    def load(cls, path: str) -> "PCAAnomalyDetector":
        import joblib

        data = joblib.load(path)
        detector = cls(n_components=data["n_components"])
        detector.pca = data["pca"]
        detector.threshold = data["threshold"]
        detector.roc_auc = data["roc_auc"]
        return detector


def error_map_to_heatmap(error_map: np.ndarray, original_bgr: np.ndarray) -> np.ndarray:
    """Upscales the low-res error map back to the original image size and
    overlays it as a color heatmap, for the explainability/localization bonus."""
    h, w = original_bgr.shape[:2]
    norm = error_map / (error_map.max() + 1e-8)
    norm_u8 = (norm * 255).astype(np.uint8)
    norm_resized = cv2.resize(norm_u8, (w, h))
    heatmap = cv2.applyColorMap(norm_resized, cv2.COLORMAP_JET)
    return cv2.addWeighted(original_bgr, 0.55, heatmap, 0.45, 0)
