"""Train transformer on historical market data for next-day return prediction."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import torch
import yaml
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, random_split

from phinance.ml.data import MarketSequenceDataset
from phinance.ml.transformer import MarketTransformer, MarketTransformerConfig


def _load_symbol_frame(symbol: str, data_dir: Path) -> pd.DataFrame:
    path = data_dir / f"{symbol}.csv"
    if not path.exists():
        raise FileNotFoundError(f"Expected historical CSV at {path}")
    return pd.read_csv(path)


def train(config_path: str | Path) -> Path:
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    symbols = cfg["data"]["symbols"]
    data_dir = Path(cfg["data"]["data_dir"])
    frames = [_load_symbol_frame(symbol, data_dir) for symbol in symbols]

    dataset = MarketSequenceDataset(
        frames,
        sequence_length=int(cfg["data"]["sequence_length"]),
        fit_scaler=True,
    )

    val_ratio = float(cfg["training"].get("validation_ratio", 0.2))
    val_size = max(1, int(len(dataset) * val_ratio))
    train_size = len(dataset) - val_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=int(cfg["training"]["batch_size"]), shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=int(cfg["training"]["batch_size"]), shuffle=False)

    model_cfg = MarketTransformerConfig(
        input_dim=len(dataset.feature_columns),
        d_model=int(cfg["model"]["d_model"]),
        num_layers=int(cfg["model"]["num_layers"]),
        num_heads=int(cfg["model"]["num_heads"]),
        dropout=float(cfg["model"]["dropout"]),
        dim_feedforward=int(cfg["model"]["dim_feedforward"]),
        max_seq_len=int(cfg["data"]["sequence_length"]),
    )
    model = MarketTransformer(model_cfg)

    criterion = nn.MSELoss()
    optimizer = AdamW(model.parameters(), lr=float(cfg["training"]["learning_rate"]))

    best_val = float("inf")
    patience = int(cfg["training"].get("early_stopping_patience", 5))
    bad_epochs = 0

    for _epoch in range(int(cfg["training"]["epochs"])):
        model.train()
        for x, y in train_loader:
            pred, _ = model(x)
            loss = criterion(pred, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        model.eval()
        running = 0.0
        count = 0
        with torch.no_grad():
            for x, y in val_loader:
                pred, _ = model(x)
                running += float(criterion(pred, y).item()) * len(y)
                count += len(y)
        val_loss = running / max(count, 1)

        if val_loss < best_val:
            best_val = val_loss
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                break

    out_path = Path(cfg["training"]["checkpoint_path"])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "model_config": model_cfg.to_dict(),
            "feature_columns": dataset.feature_columns,
            "sequence_length": dataset.sequence_length,
            "scaler": dataset.scaler.to_dict(),
            "metadata": {
                "symbols": symbols,
                "best_val_loss": best_val,
                "task": "next_day_return_regression",
            },
        },
        out_path,
    )
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/transformer_config.yaml")
    args = parser.parse_args()
    out = train(args.config)
    print(f"Saved transformer checkpoint -> {out}")


if __name__ == "__main__":
    main()
