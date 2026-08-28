import os
import sys
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from app.ml.autoencoder import ConvAutoencoder, reconstruction_error
from app.ml.synth_degrade import generate_labeled_variant, DEGRADATION_TYPES, SEVERITIES
from train_classifier import load_clean_images

def main():
    clean_images = load_clean_images(None) # Use synthetic if none provided
    split = int(len(clean_images) * 0.8)
    holdout_clean = clean_images[split:]

    models_dir = os.path.join(os.path.dirname(__file__), "..", "app", "ml", "models")
    ae_path = os.path.join(models_dir, "autoencoder.pt")
    
    if not os.path.exists(ae_path):
        print("Autoencoder not found. Train it first.")
        return

    checkpoint = torch.load(ae_path, map_location="cpu")
    model = ConvAutoencoder()
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    # Calculate clean baseline errors
    clean_errors = []
    for img in holdout_clean:
        err, _ = reconstruction_error(model, img)
        clean_errors.append(err)
    
    mean_clean_error = np.mean(clean_errors)
    print(f"Mean clean error (baseline): {mean_clean_error:.5f}")

    print("Reconstruction error ratio (vs clean baseline):")
    print("| Degradation | low | medium | high |")
    print("|---|---|---|---|")

    for deg in DEGRADATION_TYPES:
        row = [f"| {deg}"]
        for sev in SEVERITIES:
            deg_errors = []
            for img in holdout_clean:
                degraded = generate_labeled_variant(img, deg, sev)
                err, _ = reconstruction_error(model, degraded)
                deg_errors.append(err)
            ratio = np.mean(deg_errors) / mean_clean_error
            row.append(f"{ratio:.2f}x")
        print(" | ".join(row) + " |")

if __name__ == "__main__":
    main()
