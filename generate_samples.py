"""
Generates the samples/ folder required by the submission: one image per
quality condition, built from the same synth_degrade functions AND the same
synthetic scene generator used in training, so these are genuinely
out-of-sample (different RNG seed) rather than drawn from a different
distribution entirely -- using a different generator for eval vs. training
images is a good way to fool yourself about how well a model generalizes.

Usage: python generate_samples.py
"""

import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "backend"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "backend", "train"))
from app.ml.synth_degrade import add_blur, add_corruption, add_exposure, add_noise  # noqa: E402
from train_classifier import _generate_textured_scene  # noqa: E402

OUT_DIR = os.path.join(os.path.dirname(__file__), "samples")

# Different seed than train_classifier.py's (42), so this is a genuinely
# unseen scene, not one already used in training or PCA fitting.
SAMPLE_SEED = 999


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    rng = np.random.default_rng(SAMPLE_SEED)
    base = _generate_textured_scene(rng, size=256)

    samples = {
        "acceptable.jpg": base,
        "blurry.jpg": add_blur(base, sigma=5.0),
        "overexposed.jpg": add_exposure(base, gamma=3.0),
        "underexposed.jpg": add_exposure(base, gamma=0.28),
        "noisy.jpg": add_noise(base, sigma=40),
        "corrupted.jpg": add_corruption(base, jpeg_quality=8, n_blocks=10),
    }

    for name, img in samples.items():
        path = os.path.join(OUT_DIR, name)
        cv2.imwrite(path, img)
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
