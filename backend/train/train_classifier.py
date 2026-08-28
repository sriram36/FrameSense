"""
Trains the engineered-feature classifier that detects blur, exposure,
noise, and corruption issues.

Usage:
    python train_classifier.py --images-dir /path/to/clean/images --out ../app/ml/models

If --images-dir is omitted, a small set of synthetic "clean" placeholder
images is generated instead, purely so this script is runnable end to end
without a real dataset on hand. Replace with real photos before submission
-- see the README's "Data Sourcing" section for where to get some.
"""

import argparse
import glob
import os
import sys

import cv2
import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV, train_test_split

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from app.ml.features import FEATURE_NAMES, feature_vector  # noqa: E402
from app.ml.synth_degrade import (  # noqa: E402
    DEGRADATION_TYPES,
    SEVERITIES,
    generate_labeled_variant,
)


def _generate_textured_scene(rng: np.random.Generator, size: int = 256) -> np.ndarray:
    """Generates one textured synthetic scene: gradient sky/ground, a
    textured 'building' facade with windows, a few extra shapes, and fine
    speckle so sharpness/noise features have real signal to measure.
    Exposed as a standalone function so other scripts (e.g.
    generate_samples.py) can draw from the exact same distribution used
    for training -- using a different generator for train vs. eval images
    is a real way to fool yourself about generalization."""
    img = np.zeros((size, size, 3), dtype=np.uint8)
    horizon = rng.integers(size // 3, 2 * size // 3)
    sky_top = rng.integers(140, 220, size=3)
    sky_bottom = rng.integers(90, 160, size=3)
    for y in range(horizon):
        t = y / max(1, horizon)
        img[y, :] = (sky_top * (1 - t) + sky_bottom * t).astype(np.uint8)
    ground_color = rng.integers(50, 140, size=3)
    img[horizon:, :] = ground_color

    bw, bh = rng.integers(size // 4, size // 2), rng.integers(size // 4, size // 2)
    bx, by = rng.integers(0, size - bw), rng.integers(max(0, horizon - bh), horizon)
    facade_color = tuple(int(c) for c in rng.integers(60, 130, size=3))
    cv2.rectangle(img, (bx, by), (bx + bw, by + bh), facade_color, -1)
    window_color = tuple(int(c) for c in rng.integers(150, 230, size=3))
    step = rng.integers(18, 30)
    for wy in range(by + 10, by + bh - 10, step):
        for wx in range(bx + 10, bx + bw - 10, step):
            cv2.rectangle(img, (wx, wy), (wx + step - 8, wy + step - 12), window_color, -1)

    for _ in range(rng.integers(2, 5)):
        color = tuple(int(c) for c in rng.integers(40, 220, size=3))
        pt1 = (int(rng.integers(0, size)), int(rng.integers(0, size)))
        radius = int(rng.integers(8, size // 6))
        cv2.circle(img, pt1, radius, color, -1)

    speckle = rng.integers(-10, 10, img.shape, dtype=np.int16)
    return np.clip(img.astype(np.int16) + speckle, 0, 255).astype(np.uint8)


def load_clean_images(images_dir: str | None, n_synthetic: int = 100, size: int = 256):
    """Loads real images from disk if a directory is given, otherwise
    generates textured synthetic scenes via _generate_textured_scene().
    Replace this whole function with real photos before submission --
    see the README's 'Data Sourcing' section for where to get some."""
    images = []
    if images_dir:
        paths = sorted(
            glob.glob(os.path.join(images_dir, "*.jpg"))
            + glob.glob(os.path.join(images_dir, "*.jpeg"))
            + glob.glob(os.path.join(images_dir, "*.png"))
        )
        for p in paths:
            img = cv2.imread(p, cv2.IMREAD_COLOR)
            if img is not None:
                images.append(cv2.resize(img, (size, size)))
        if images:
            print(f"loaded {len(images)} real images from {images_dir}")
            return images
        print(f"no readable images found in {images_dir}, falling back to synthetic")

    rng = np.random.default_rng(42)
    images = [_generate_textured_scene(rng, size) for _ in range(n_synthetic)]
    print(f"generated {len(images)} textured synthetic scenes (no --images-dir given)")
    return images


def build_dataset(clean_images):
    X, y = [], []
    for img in clean_images:
        X.append(feature_vector(img))
        y.append("clean")
        for degradation in DEGRADATION_TYPES:
            for severity in SEVERITIES:
                variant = generate_labeled_variant(img, degradation, severity)
                X.append(feature_vector(variant))
                y.append(f"{degradation}_{severity}")
    return np.array(X), np.array(y)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--images-dir", default=None, help="folder of clean source images")
    parser.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "..", "app", "ml", "models"))
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)

    clean_images = load_clean_images(args.images_dir)
    X, y = build_dataset(clean_images)
    print(f"dataset: {X.shape[0]} samples, {X.shape[1]} features, {len(set(y))} classes")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    # Small hyperparameter search, cross-validated on the training split only
    # (test split stays untouched until final evaluation, avoiding leakage).
    param_grid = {
        "n_estimators": [200, 400],
        "max_depth": [12, 20, None],
        "min_samples_leaf": [1, 2],
    }
    search = GridSearchCV(
        RandomForestClassifier(random_state=42, class_weight="balanced"),
        param_grid,
        cv=4,
        scoring="accuracy",
        n_jobs=-1,
    )
    search.fit(X_train, y_train)
    clf = search.best_estimator_
    print(f"\nbest params: {search.best_params_}  (cv accuracy on train split: {search.best_score_:.3f})")

    y_pred = clf.predict(X_test)
    print("\n=== classification report (held-out test split) ===")
    print(classification_report(y_test, y_pred, zero_division=0))

    labels_sorted = sorted(set(y))
    cm = confusion_matrix(y_test, y_pred, labels=labels_sorted)
    print("=== confusion matrix ===")
    print("labels:", labels_sorted)
    print(cm)

    model_path = os.path.join(args.out, "classifier.joblib")
    joblib.dump({"model": clf, "feature_names": FEATURE_NAMES, "classes": list(clf.classes_)}, model_path)
    print(f"\nsaved model -> {model_path}")


if __name__ == "__main__":
    main()
