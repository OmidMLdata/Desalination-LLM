from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd

from ..rendering import render_pore
from .base import Prediction


class CNNMLPPredictor:
    """Paper-matched ResNet50 + residual-MLP inference backend.

    The four original checkpoints and the training-label CSV are required. They were
    referenced by the uploaded project manifest but were not present in the uploaded
    ``gemini.tar.xz`` archive, so they are intentionally not fabricated here.
    """

    REQUIRED_FILES = (
        "resnet_smoothl1_all2_flux.ckpt",
        "mlp_smoothl1_all2_flux.ckpt",
        "resnet_smoothl1_all2_rej.ckpt",
        "mlp_smoothl1_all2_rej.ckpt",
    )

    def __init__(self, xy: np.ndarray, model_dir: str | Path, labels_csv: str | Path, device: str = "auto"):
        try:
            import torch
            from torch import nn
            from torchvision import models, transforms
            from PIL import Image
        except ImportError as exc:
            raise RuntimeError("Install the optional torch dependencies: pip install -e '.[torch]'") from exc

        self.torch = torch
        self.nn = nn
        self.models = models
        self.transforms = transforms
        self.Image = Image
        self.xy = np.asarray(xy, dtype=float)
        self.model_dir = Path(model_dir)
        missing = [name for name in self.REQUIRED_FILES if not (self.model_dir / name).is_file()]
        if missing:
            raise FileNotFoundError(f"Missing CNN/MLP checkpoints in {self.model_dir}: {missing}")
        labels_path = Path(labels_csv)
        if not labels_path.is_file():
            raise FileNotFoundError(f"Training-label CSV not found: {labels_path}")
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        class ResidualBlock(nn.Module):
            def __init__(self, in_dim: int):
                super().__init__()
                self.main = nn.Sequential(
                    nn.Linear(in_dim, in_dim, bias=False),
                    nn.ReLU(),
                    nn.BatchNorm1d(in_dim),
                    nn.Linear(in_dim, in_dim, bias=False),
                )

            def forward(self, value):
                return value + self.main(value)

        class MLP(nn.Module):
            def __init__(self, in_size: int):
                super().__init__()
                self.model = nn.Sequential(
                    nn.Linear(in_size, 256),
                    nn.ReLU(),
                    ResidualBlock(256),
                    nn.Linear(256, 64),
                    nn.ReLU(),
                    ResidualBlock(64),
                    nn.ReLU(),
                )
                self.output = nn.Linear(64, 1)

            def forward(self, value):
                features = self.model(value)
                return self.output(features), features

        self._MLP = MLP
        self.flux_resnet = self._load(models.resnet50(weights=None), self.model_dir / self.REQUIRED_FILES[0])
        self.flux_mlp = self._load(MLP(1000), self.model_dir / self.REQUIRED_FILES[1])
        self.rej_resnet = self._load(models.resnet50(weights=None), self.model_dir / self.REQUIRED_FILES[2])
        self.rej_mlp = self._load(MLP(1000), self.model_dir / self.REQUIRED_FILES[3])
        frame = pd.read_csv(labels_path)
        self.flux_mean = float(frame["flux"].mean())
        self.flux_std = float(frame["flux"].std())
        self.rej_mean = float(frame["rejection"].mean())
        self.rej_std = float(frame["rejection"].std())
        self.transform = transforms.Compose([
            transforms.CenterCrop((350, 350)),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])

    def _load(self, model, path: Path):
        try:
            state = self.torch.load(path, map_location=self.device, weights_only=True)
        except TypeError:  # PyTorch versions before weights_only support
            state = self.torch.load(path, map_location=self.device)
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        cleaned = {}
        for key, value in state.items():
            for prefix in ("module.", "model."):
                if key.startswith(prefix):
                    key = key[len(prefix):]
            cleaned[key] = value
        model.load_state_dict(cleaned, strict=False)
        return model.to(self.device).eval()

    def predict(self, removed_indices: set[int]) -> Prediction:
        if not removed_indices:
            return Prediction(0.0, 1.0)
        with TemporaryDirectory() as directory:
            image_path = render_pore(self.xy, removed_indices, Path(directory) / "pore.png", dpi=100)
            image = self.Image.open(image_path).convert("RGB")
            tensor = self.transform(image).unsqueeze(0).to(self.device)
            with self.torch.no_grad():
                flux_standard, _ = self.flux_mlp(self.flux_resnet(tensor))
                rejection_standard, _ = self.rej_mlp(self.rej_resnet(tensor))
            flux = float(flux_standard.item() * self.flux_std + self.flux_mean)
            rejection = float(rejection_standard.item() * self.rej_std + self.rej_mean)
            return Prediction(max(0.0, flux), float(np.clip(rejection, 0.0, 1.0)))
