"""
Lightweight convolutional autoencoder for anomaly-style defect detection.

Trained ONLY on clean/acceptable images. At inference time, an image that
reconstructs poorly (high pixel error) is flagged as a "potential visual
defect" -- this is independent of the labeled blur/exposure/noise/corruption
classes handled by the RandomForest classifier in features.py + train_classifier.py.

Not executed in this sandbox (torch isn't installed here / no network to add
it), but this follows standard, well-tested PyTorch patterns. Install torch
locally (see requirements.txt) and run train_autoencoder.py to produce
app/ml/models/autoencoder.pt.
"""

import numpy as np
import torch
import torch.nn as nn

IMG_SIZE = 128  # resize target; keep small so CPU training stays fast


class ConvAutoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, 3, stride=2, padding=1), nn.ReLU(True),   # 128 -> 64
            nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.ReLU(True),  # 64 -> 32
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.ReLU(True),  # 32 -> 16
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1), nn.ReLU(True),
            nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=1), nn.ReLU(True),
            nn.ConvTranspose2d(16, 3, 3, stride=2, padding=1, output_padding=1), nn.Sigmoid(),
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))


def preprocess(bgr: np.ndarray) -> torch.Tensor:
    """BGR uint8 HxWx3 -> normalized float tensor 1x3xHxW in [0,1]."""
    import cv2

    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    rgb = cv2.resize(rgb, (IMG_SIZE, IMG_SIZE)).astype(np.float32) / 255.0
    tensor = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0)
    return tensor


def train(model: ConvAutoencoder, clean_images: list, epochs: int = 25, lr: float = 1e-3, device: str = "cpu"):
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    batch = torch.cat([preprocess(img) for img in clean_images], dim=0).to(device)

    model.train()
    for epoch in range(epochs):
        opt.zero_grad()
        recon = model(batch)
        loss = loss_fn(recon, batch)
        loss.backward()
        opt.step()
        if epoch % 5 == 0 or epoch == epochs - 1:
            print(f"epoch {epoch:3d}  reconstruction MSE {loss.item():.5f}")
    return model


@torch.no_grad()
def reconstruction_error(model: ConvAutoencoder, bgr: np.ndarray, device: str = "cpu") -> tuple[float, np.ndarray]:
    """Returns (scalar anomaly score, per-pixel error map HxW) for one image."""
    model.eval()
    x = preprocess(bgr).to(device)
    recon = model(x)
    error_map = (x - recon).pow(2).mean(dim=1).squeeze(0).cpu().numpy()  # HxW
    return float(error_map.mean()), error_map


def choose_threshold(errors: np.ndarray, is_defect: np.ndarray) -> dict:
    """Picks an anomaly threshold from the ROC curve on a labeled
    clean(0)/degraded(1) validation set. Returns the chosen threshold plus
    the ROC-AUC so it can be reported in EVALUATION.md."""
    from .anomaly_common import choose_threshold as _choose_threshold

    return _choose_threshold(errors, is_defect)
