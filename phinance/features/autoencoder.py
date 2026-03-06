"""Lightweight NumPy autoencoder for unsupervised feature learning."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass
class AutoencoderConfig:
    """Hyperparameters for autoencoder training."""

    input_dim: int
    latent_dim: int = 8
    hidden_dim: int = 32
    learning_rate: float = 1e-3
    epochs: int = 25
    batch_size: int = 64
    random_seed: int = 7


class MarketAutoencoder:
    """Single hidden-layer autoencoder with tanh activations."""

    def __init__(self, config: AutoencoderConfig) -> None:
        self.config = config
        rng = np.random.default_rng(config.random_seed)

        def _weight(fan_in: int, fan_out: int) -> np.ndarray:
            scale = np.sqrt(2.0 / max(1, fan_in + fan_out))
            return rng.normal(0.0, scale, size=(fan_in, fan_out)).astype(np.float32)

        self.w1 = _weight(config.input_dim, config.hidden_dim)
        self.b1 = np.zeros(config.hidden_dim, dtype=np.float32)
        self.w2 = _weight(config.hidden_dim, config.latent_dim)
        self.b2 = np.zeros(config.latent_dim, dtype=np.float32)
        self.w3 = _weight(config.latent_dim, config.hidden_dim)
        self.b3 = np.zeros(config.hidden_dim, dtype=np.float32)
        self.w4 = _weight(config.hidden_dim, config.input_dim)
        self.b4 = np.zeros(config.input_dim, dtype=np.float32)

        self.feature_mean = np.zeros(config.input_dim, dtype=np.float32)
        self.feature_std = np.ones(config.input_dim, dtype=np.float32)

    @staticmethod
    def _tanh(x: np.ndarray) -> np.ndarray:
        return np.tanh(x)

    @staticmethod
    def _tanh_grad(a: np.ndarray) -> np.ndarray:
        return 1.0 - (a * a)

    def _normalize(self, x: np.ndarray) -> np.ndarray:
        return (x - self.feature_mean) / np.maximum(self.feature_std, 1e-8)

    def _forward(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        h1_pre = x @ self.w1 + self.b1
        h1 = self._tanh(h1_pre)
        z_pre = h1 @ self.w2 + self.b2
        z = self._tanh(z_pre)
        h2_pre = z @ self.w3 + self.b3
        h2 = self._tanh(h2_pre)
        x_hat = h2 @ self.w4 + self.b4
        return h1, z, h2, x_hat, h1_pre, h2_pre

    def fit(self, features: np.ndarray) -> dict[str, list[float]]:
        x_raw = np.asarray(features, dtype=np.float32)
        if x_raw.ndim != 2 or x_raw.shape[1] != self.config.input_dim:
            raise ValueError(f"Expected shape (n_samples, {self.config.input_dim}), got {x_raw.shape}")

        self.feature_mean = x_raw.mean(axis=0).astype(np.float32)
        self.feature_std = (x_raw.std(axis=0) + 1e-6).astype(np.float32)
        x = self._normalize(x_raw)

        history = {"reconstruction_loss": []}
        rng = np.random.default_rng(self.config.random_seed)
        n = len(x)

        for _ in range(self.config.epochs):
            order = rng.permutation(n)
            epoch_loss = 0.0
            for start in range(0, n, self.config.batch_size):
                idx = order[start : start + self.config.batch_size]
                xb = x[idx]
                bs = max(1, len(xb))

                h1, z, h2, x_hat, _, _ = self._forward(xb)
                diff = x_hat - xb
                loss = float(np.mean(diff * diff))
                epoch_loss += loss * bs

                dx_hat = (2.0 / bs) * diff
                dw4 = h2.T @ dx_hat
                db4 = dx_hat.sum(axis=0)

                dh2 = dx_hat @ self.w4.T
                dh2_pre = dh2 * self._tanh_grad(h2)
                dw3 = z.T @ dh2_pre
                db3 = dh2_pre.sum(axis=0)

                dz = dh2_pre @ self.w3.T
                dz_pre = dz * self._tanh_grad(z)
                dw2 = h1.T @ dz_pre
                db2 = dz_pre.sum(axis=0)

                dh1 = dz_pre @ self.w2.T
                dh1_pre = dh1 * self._tanh_grad(h1)
                dw1 = xb.T @ dh1_pre
                db1 = dh1_pre.sum(axis=0)

                lr = self.config.learning_rate
                self.w4 -= lr * dw4
                self.b4 -= lr * db4
                self.w3 -= lr * dw3
                self.b3 -= lr * db3
                self.w2 -= lr * dw2
                self.b2 -= lr * db2
                self.w1 -= lr * dw1
                self.b1 -= lr * db1

            history["reconstruction_loss"].append(epoch_loss / max(1, n))

        return history

    def encode(self, features: np.ndarray) -> np.ndarray:
        x = np.asarray(features, dtype=np.float32)
        if x.ndim == 1:
            x = x.reshape(1, -1)
        x_norm = self._normalize(x)
        h1 = self._tanh(x_norm @ self.w1 + self.b1)
        z = self._tanh(h1 @ self.w2 + self.b2)
        return z.astype(np.float32)

    def reconstruct(self, features: np.ndarray) -> np.ndarray:
        x = np.asarray(features, dtype=np.float32)
        if x.ndim == 1:
            x = x.reshape(1, -1)
        x_norm = self._normalize(x)
        h1, z, h2, x_hat, _, _ = self._forward(x_norm)
        return x_hat.astype(np.float32)

    def save(self, path: str | Path) -> Path:
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        payload: dict[str, Any] = {
            "config": asdict(self.config),
            "feature_mean": self.feature_mean,
            "feature_std": self.feature_std,
            "w1": self.w1,
            "b1": self.b1,
            "w2": self.w2,
            "b2": self.b2,
            "w3": self.w3,
            "b3": self.b3,
            "w4": self.w4,
            "b4": self.b4,
        }
        np.savez_compressed(out, **payload)
        return out

    @classmethod
    def load(cls, path: str | Path) -> "MarketAutoencoder":
        payload = np.load(Path(path), allow_pickle=True)
        config = AutoencoderConfig(**payload["config"].item())
        model = cls(config)
        for name in ["feature_mean", "feature_std", "w1", "b1", "w2", "b2", "w3", "b3", "w4", "b4"]:
            setattr(model, name, payload[name].astype(np.float32))
        return model
