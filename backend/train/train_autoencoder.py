"""
Trains the anomaly-detection autoencoder on clean images only, then picks
a reconstruction-error threshold using a held-out set of clean + degraded
images (reusing the synthetic degradation generator for the degraded half).

Usage:
    python train_autoencoder.py --images-dir /path/to/clean/images --out ../app/ml/models

Requires torch (see requirements.txt) -- not installed in this sandbox,
so this script is provided as correct, ready-to-run code rather than
something already executed here.
"""

import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from app.ml.autoencoder import ConvAutoencoder, choose_threshold, reconstruction_error, train  # noqa: E402
from app.ml.synth_degrade import DEGRADATION_TYPES, generate_labeled_variant  # noqa: E402
from train_classifier import load_clean_images  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--images-dir", default=None)
    parser.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "..", "app", "ml", "models"))
    parser.add_argument("--epochs", type=int, default=25)
    args = parser.parse_args()
    os.makedirs(args.out, exist_ok=True)

    clean_images = load_clean_images(args.images_dir)
    split = int(len(clean_images) * 0.8)
    train_images, holdout_clean = clean_images[:split], clean_images[split:]

    model = ConvAutoencoder()
    train(model, train_images, epochs=args.epochs)

    # Build a labeled validation set: held-out clean images (label 0) plus
    # degraded versions of them (label 1), to pick a defensible threshold.
    errors, labels = [], []
    for img in holdout_clean:
        err, _ = reconstruction_error(model, img)
        errors.append(err)
        labels.append(0)
        # Scoped to {underexposure, noise, corruption} based on EVALUATION.md diagnosis
        ANOMALY_DEGRADATIONS = ["underexposure", "noise", "corruption"]
        degraded = generate_labeled_variant(img, np.random.choice(ANOMALY_DEGRADATIONS), "high")
        err_d, _ = reconstruction_error(model, degraded)
        errors.append(err_d)
        labels.append(1)

    stats = choose_threshold(np.array(errors), np.array(labels))
    print(f"chosen threshold={stats['threshold']:.5f}  ROC-AUC={stats['roc_auc']:.3f}  "
          f"TPR={stats['tpr']:.2f}  FPR={stats['fpr']:.2f}")
    print("-> record these numbers in EVALUATION.md")

    model_path = os.path.join(args.out, "autoencoder.pt")
    torch.save({"state_dict": model.state_dict(), "threshold": stats["threshold"], "roc_auc": stats["roc_auc"]}, model_path)
    print(f"saved model -> {model_path}")


if __name__ == "__main__":
    main()
