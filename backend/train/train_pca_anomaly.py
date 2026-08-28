"""
Trains the PCA-based anomaly detector (see app/ml/pca_anomaly.py) on clean
images only, then picks a reconstruction-error threshold using a held-out
set of clean + degraded images.

Usage:
    python train_pca_anomaly.py --images-dir /path/to/clean/images --out ../app/ml/models

Unlike train_autoencoder.py, this has no torch dependency and runs anywhere
numpy/scikit-learn are available.
"""

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from app.ml.pca_anomaly import PCAAnomalyDetector  # noqa: E402
from app.ml.synth_degrade import generate_labeled_variant  # noqa: E402
from train_classifier import load_clean_images  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--images-dir", default=None)
    parser.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "..", "app", "ml", "models"))
    parser.add_argument("--n-components", type=int, default=20)
    args = parser.parse_args()
    os.makedirs(args.out, exist_ok=True)

    clean_images = load_clean_images(args.images_dir)
    split = int(len(clean_images) * 0.8)
    train_images, holdout_clean = clean_images[:split], clean_images[split:]

    detector = PCAAnomalyDetector(n_components=args.n_components)
    detector.fit(train_images)

    rng = np.random.default_rng(123)
    errors, labels = [], []
    # Scoped based on a full per-degradation-type diagnostic (all 5 types x
    # 3 severities against the clean baseline -- see EVALUATION.md for the
    # full table). Findings: underexposure produces the STRONGEST signal
    # (up to 2.36x clean error at high severity) -- PCA's linear basis
    # struggles badly with the heavy pixel-value crushing of underexposure.
    # Noise and corruption give a moderate, severity-scaling signal. Blur
    # and overexposure sit at or below the clean baseline (PCA's low-pass
    # reconstruction handles smoothed/brightened images easily) -- including
    # them as "defect" would dilute the signal exactly as the first version
    # of this script demonstrated (ROC-AUC dropped to ~0.47, i.e. random).
    ANOMALY_DEGRADATIONS = ["underexposure", "noise", "corruption"]
    for img in holdout_clean:
        err, _ = detector.reconstruction_error(img)
        errors.append(err)
        labels.append(0)
        degradation = rng.choice(ANOMALY_DEGRADATIONS)
        degraded = generate_labeled_variant(img, degradation, "high")
        err_d, _ = detector.reconstruction_error(degraded)
        errors.append(err_d)
        labels.append(1)

    from app.ml.anomaly_common import choose_threshold

    stats = choose_threshold(np.array(errors), np.array(labels))
    detector.threshold = stats["threshold"]
    detector.roc_auc = stats["roc_auc"]

    print(f"n_components={args.n_components}  threshold={stats['threshold']:.5f}  "
          f"ROC-AUC={stats['roc_auc']:.3f}  TPR={stats['tpr']:.2f}  FPR={stats['fpr']:.2f}")

    model_path = os.path.join(args.out, "pca_anomaly.joblib")
    detector.save(model_path)
    print(f"saved model -> {model_path}")


if __name__ == "__main__":
    main()
