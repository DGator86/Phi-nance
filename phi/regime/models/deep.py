"""Deep-learning regime detection models and detector wrapper."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from phi.regime.base import RegimeDetector
from phi.regime.data import prepare_data_for_training
from phi.regime.utils import extract_features

try:
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset
except Exception:  # pragma: no cover
    torch = None
    nn = None

_NNBase = nn.Module if nn is not None else object


class LSTMClassifier(_NNBase):
    """LSTM sequence classifier for discrete regime labels."""

    def __init__(self, input_size: int, hidden_size: int, num_layers: int, num_classes: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])


class TransformerClassifier(_NNBase):
    """Minimal Transformer encoder classifier for regime labels."""

    def __init__(self, input_size: int, hidden_size: int, num_layers: int, num_classes: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.input_proj = nn.Linear(input_size, hidden_size)
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=max(1, hidden_size // 16),
            dim_feedforward=hidden_size * 4,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.input_proj(x)
        out = self.encoder(z)
        return self.fc(out[:, -1, :])


class DeepRegimeDetector(RegimeDetector):
    """Trainable deep-learning detector implementing the RegimeDetector interface."""

    def __init__(self, model_type: str = "lstm", **model_params: Any) -> None:
        self.model_type = model_type
        self.model_params = model_params
        self.model: nn.Module | None = None
        self.scaler: Any = None
        self.feature_columns: list[str] = []
        self.label_names: list[str] = []
        self.metadata: dict[str, Any] = {
            "detector_class": self.__class__.__name__,
            "params": {"model_type": model_type, **model_params},
        }

    def _require_torch(self) -> None:
        if torch is None or nn is None:
            raise ImportError("PyTorch is required for DeepRegimeDetector")

    def _build_model(self, input_size: int, num_classes: int) -> nn.Module:
        hidden_size = int(self.model_params.get("hidden_size", 64))
        num_layers = int(self.model_params.get("num_layers", 2))
        dropout = float(self.model_params.get("dropout", 0.1))
        if self.model_type == "transformer":
            return TransformerClassifier(input_size, hidden_size, num_layers, num_classes, dropout)
        return LSTMClassifier(input_size, hidden_size, num_layers, num_classes, dropout)

    def fit(self, ohlcv: pd.DataFrame, labels: pd.Series | None = None, **kwargs: Any) -> DeepRegimeDetector:
        self._require_torch()
        if labels is None:
            raise ValueError("DeepRegimeDetector.fit requires labels for supervised training")

        params = {**self.model_params, **kwargs}
        prepared = prepare_data_for_training(ohlcv, params, labels=labels)
        x_train = prepared["X_train"]
        y_train = prepared["y_train"]
        x_val = prepared["X_val"]
        y_val = prepared["y_val"]

        self.scaler = prepared["scaler"]
        self.feature_columns = prepared["feature_columns"]
        self.label_names = prepared["labels"]

        input_size = x_train.shape[-1]
        num_classes = len(self.label_names)
        self.model = self._build_model(input_size=input_size, num_classes=num_classes)

        device = torch.device(str(params.get("device", "cpu")))
        self.model.to(device)

        dataset = TensorDataset(torch.from_numpy(x_train).float(), torch.from_numpy(y_train).long())
        loader = DataLoader(dataset, batch_size=int(params.get("batch_size", 32)), shuffle=True)

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=float(params.get("lr", 1e-3)))
        epochs = int(params.get("epochs", 30))
        patience = int(params.get("patience", 5))

        best_state: dict[str, Any] | None = None
        best_val = float("inf")
        no_improve = 0

        for _ in range(epochs):
            self.model.train()
            for xb, yb in loader:
                xb = xb.to(device)
                yb = yb.to(device)
                optimizer.zero_grad()
                logits = self.model(xb)
                loss = criterion(logits, yb)
                loss.backward()
                optimizer.step()

            if len(x_val) == 0:
                continue

            self.model.eval()
            with torch.no_grad():
                xv = torch.from_numpy(x_val).float().to(device)
                yv = torch.from_numpy(y_val).long().to(device)
                val_loss = float(criterion(self.model(xv), yv).cpu().item())

            if val_loss < best_val:
                best_val = val_loss
                no_improve = 0
                best_state = {k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()}
            else:
                no_improve += 1
                if no_improve >= patience:
                    break

        if best_state:
            self.model.load_state_dict(best_state)

        self.metadata.update(
            {
                "feature_columns": self.feature_columns,
                "label_names": self.label_names,
                "seq_length": int(params.get("seq_length", 20)),
                "window": int(params.get("window", 20)),
                "training_period": {
                    "start": str(pd.to_datetime(ohlcv.index.min()).date()),
                    "end": str(pd.to_datetime(ohlcv.index.max()).date()),
                },
            }
        )
        return self

    def predict(self, ohlcv: pd.DataFrame) -> pd.Series:
        self._require_torch()
        if self.model is None or self.scaler is None:
            raise ValueError("DeepRegimeDetector must be fit before predict")

        seq_length = int(self.metadata.get("seq_length", self.model_params.get("seq_length", 20)))
        window = int(self.metadata.get("window", self.model_params.get("window", 20)))
        features = extract_features(ohlcv, window=window)
        features = features[self.feature_columns]
        scaled = self.scaler.transform(features.to_numpy(dtype=np.float32))

        x = []
        idx = []
        for end in range(seq_length - 1, len(scaled)):
            x.append(scaled[end - seq_length + 1 : end + 1])
            idx.append(features.index[end])

        result = pd.Series(np.nan, index=ohlcv.index, name="regime", dtype=object)
        if not x:
            return result

        self.model.eval()
        with torch.no_grad():
            tensor = torch.from_numpy(np.asarray(x, dtype=np.float32))
            logits = self.model(tensor)
            pred = logits.argmax(dim=1).cpu().numpy()

        labels = [self.label_names[int(i)] if self.label_names else f"state_{int(i)}" for i in pred]
        result.loc[idx] = labels
        return result

    def save(self, path: str | Path) -> None:
        self._require_torch()
        if self.model is None or self.scaler is None:
            raise ValueError("Cannot save an unfitted DeepRegimeDetector")
        model_path = Path(path)
        model_path.parent.mkdir(parents=True, exist_ok=True)

        torch.save(
            {
                "model_state": self.model.state_dict(),
                "model_type": self.model_type,
                "model_params": self.model_params,
                "feature_columns": self.feature_columns,
                "label_names": self.label_names,
                "metadata": self.metadata,
            },
            model_path,
        )
        joblib.dump(self.scaler, model_path.with_suffix(".scaler.pkl"))
        model_path.with_suffix(".json").write_text(json.dumps(self.metadata, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> DeepRegimeDetector:
        if torch is None:
            raise ImportError("PyTorch is required for DeepRegimeDetector")
        payload = torch.load(Path(path), map_location="cpu")
        inst = cls(model_type=str(payload["model_type"]), **dict(payload.get("model_params", {})))
        inst.feature_columns = list(payload.get("feature_columns", []))
        inst.label_names = list(payload.get("label_names", []))
        inst.metadata = dict(payload.get("metadata", {}))
        inst.scaler = joblib.load(Path(path).with_suffix(".scaler.pkl"))

        input_size = len(inst.feature_columns)
        num_classes = len(inst.label_names)
        inst.model = inst._build_model(input_size=input_size, num_classes=num_classes)
        inst.model.load_state_dict(payload["model_state"])
        inst.model.eval()
        return inst
