"""
Ties feature extraction + classifier + autoencoder together into one
scoring pipeline, loaded once at API startup (see app/main.py).
"""

import os

import joblib
import numpy as np

from .features import feature_vector

MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")

# Increased from an earlier {10, 20, 35} after evaluation showed a
# medium-confidence corruption detection could still score 89/ACCEPTABLE --
# too lenient for a defect detector, where under-flagging real problems is
# the costly failure mode (better to occasionally over-flag a borderline
# image than let a corrupted one through as ACCEPTABLE).
SEVERITY_PENALTY = {"low": 15, "medium": 32, "high": 55}


class QualityPipeline:
    def __init__(self, models_dir: str = MODELS_DIR):
        self.classifier_bundle = None
        self.autoencoder = None
        self.ae_threshold = None
        self.ae_device = "cpu"
        self.pca_detector = None
        self._load(models_dir)

    def _load(self, models_dir: str):
        clf_path = os.path.join(models_dir, "classifier.joblib")
        if os.path.exists(clf_path):
            self.classifier_bundle = joblib.load(clf_path)
        else:
            print(f"WARNING: no classifier found at {clf_path} -- run train_classifier.py first")

        ae_path = os.path.join(models_dir, "autoencoder.pt")
        if os.path.exists(ae_path):
            try:
                import torch

                from .autoencoder import ConvAutoencoder

                checkpoint = torch.load(ae_path, map_location="cpu")
                model = ConvAutoencoder()
                model.load_state_dict(checkpoint["state_dict"])
                model.eval()
                self.autoencoder = model
                self.ae_threshold = checkpoint["threshold"]
            except ImportError:
                print("WARNING: torch not installed -- will try the PCA anomaly detector fallback")
        else:
            print(f"WARNING: no autoencoder found at {ae_path} -- run train_autoencoder.py, or rely on the PCA fallback")

        # Fallback anomaly detector: no torch dependency, so defect detection
        # still works even without the conv autoencoder. See
        # app/ml/pca_anomaly.py and train/train_pca_anomaly.py.
        if self.autoencoder is None:
            pca_path = os.path.join(models_dir, "pca_anomaly.joblib")
            if os.path.exists(pca_path):
                from .pca_anomaly import PCAAnomalyDetector

                self.pca_detector = PCAAnomalyDetector.load(pca_path)
                print(f"loaded PCA anomaly detector fallback (ROC-AUC={self.pca_detector.roc_auc:.3f})")
            else:
                print(f"WARNING: no anomaly detector available at all (checked {ae_path} and {pca_path}) "
                      "-- defect detection disabled, run train_autoencoder.py or train_pca_anomaly.py")

    @property
    def is_ready(self) -> bool:
        return self.classifier_bundle is not None

    def _classify(self, bgr: np.ndarray, feats: dict) -> list[dict]:
        vec = np.array([feats[name] for name in self.classifier_bundle["feature_names"]]).reshape(1, -1)
        model = self.classifier_bundle["model"]
        proba = model.predict_proba(vec)[0]
        classes = model.classes_
        top_idx = int(np.argmax(proba))
        label, confidence = classes[top_idx], float(proba[top_idx])

        if label == "clean":
            return []
        issue_type, severity = label.rsplit("_", 1)
        return [{"type": issue_type, "severity": severity, "confidence": round(confidence, 3)}]

    def _anomaly_score(self, bgr: np.ndarray) -> tuple[float | None, bool]:
        if self.autoencoder is not None:
            from .autoencoder import reconstruction_error

            score, _error_map = reconstruction_error(self.autoencoder, bgr)
            return score, score > self.ae_threshold

        if self.pca_detector is not None:
            is_defect, score = self.pca_detector.is_defective(bgr)
            return score, is_defect

        return None, False

    def analyze(self, bgr: np.ndarray) -> dict:
        feats = {}
        from .features import extract_features

        feats = extract_features(bgr)

        issues = self._classify(bgr, feats) if self.classifier_bundle else []
        anomaly_score, is_defect = self._anomaly_score(bgr)

        score = 100
        for issue in issues:
            score -= SEVERITY_PENALTY.get(issue["severity"], 15) * issue["confidence"]
        score = max(0, min(100, round(score)))

        if is_defect:
            label = "DEFECTIVE"
            issues.append({"type": "potential_defect", "severity": "high", "confidence": 0.8})
        elif score >= 80:
            label = "ACCEPTABLE"
        elif score >= 50:
            label = "DEGRADED"
        else:
            label = "DEFECTIVE"

        return {
            "quality_score": score,
            "quality_label": label,
            "issues": issues,
            "features": {k: round(v, 4) for k, v in feats.items()},
            "anomaly_score": anomaly_score,
        }


# Single shared instance, loaded once at import time (i.e. at API startup).
pipeline = QualityPipeline()
