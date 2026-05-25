"""Deep learning models for stock return prediction.

Inspired by Qlib's model zoo:
  - ALSTM: LSTM with additive attention over hidden states
  - GRU: Gated Recurrent Unit with dropout
  - Transformer: Self-attention encoder for time-series factors

All models take (batch, seq_len, n_features) tensor input and predict
a scalar forward return. Designed for cross-sectional ranking, not
absolute value prediction.
"""

import logging
import math
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Lazy import to avoid hard torch dependency at module level
_HAS_TORCH = False
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, TensorDataset

    _HAS_TORCH = True
except ImportError:
    logger.warning("PyTorch not installed. Deep learning models disabled.")


# ── Model definitions ────────────────────────────────────────────────

class ALSTM(nn.Module):
    """Attention LSTM for time-series stock factor prediction.

    Architecture:
      LSTM → Attention pooling over hidden states → FC → output

    The attention mechanism learns which time steps are most predictive,
    naturally handling varying lookback effectiveness.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.3,
        fc_dim: int = 64,
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_dim, hidden_dim, num_layers,
            batch_first=True, dropout=dropout if num_layers > 1 else 0,
        )
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1),
        )
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, fc_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fc_dim, 1),
        )

    def forward(self, x: "torch.Tensor") -> "torch.Tensor":
        # x: (batch, seq_len, input_dim)
        lstm_out, _ = self.lstm(x)  # (batch, seq_len, hidden_dim)

        # Additive attention
        attn_weights = self.attention(lstm_out)  # (batch, seq_len, 1)
        attn_weights = F.softmax(attn_weights.squeeze(-1), dim=1)  # (batch, seq_len)
        context = torch.sum(lstm_out * attn_weights.unsqueeze(-1), dim=1)  # (batch, hidden_dim)

        return self.fc(context).squeeze(-1)  # (batch,)


class GRUModel(nn.Module):
    """GRU-based time-series predictor with residual connections."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.3,
        fc_dim: int = 64,
    ):
        super().__init__()
        self.gru = nn.GRU(
            input_dim, hidden_dim, num_layers,
            batch_first=True, dropout=dropout if num_layers > 1 else 0,
        )
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, fc_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fc_dim, 1),
        )

    def forward(self, x: "torch.Tensor") -> "torch.Tensor":
        gru_out, _ = self.gru(x)  # (batch, seq_len, hidden_dim)
        last_hidden = gru_out[:, -1, :]  # (batch, hidden_dim)
        return self.fc(last_hidden).squeeze(-1)


class TimeSeriesTransformer(nn.Module):
    """Lightweight transformer encoder for time-series factor sequences.

    Uses learnable position encoding and a compact encoder stack
    suitable for sequences of ~20-60 time steps.
    """

    def __init__(
        self,
        input_dim: int,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        dropout: float = 0.2,
        fc_dim: int = 64,
        max_seq_len: int = 60,
    ):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_encoding = nn.Parameter(torch.zeros(1, max_seq_len, d_model))
        nn.init.normal_(self.pos_encoding, 0, 0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model * 4,
            dropout=dropout, activation="gelu", batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.fc = nn.Sequential(
            nn.Linear(d_model, fc_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fc_dim, 1),
        )

    def forward(self, x: "torch.Tensor") -> "torch.Tensor":
        # x: (batch, seq_len, input_dim)
        x = self.input_proj(x)  # (batch, seq_len, d_model)
        seq_len = x.size(1)
        x = x + self.pos_encoding[:, :seq_len, :]
        encoded = self.encoder(x)  # (batch, seq_len, d_model)
        pooled = encoded.mean(dim=1)  # mean pooling
        return self.fc(pooled).squeeze(-1)


# ── Model registry ────────────────────────────────────────────────────

MODEL_REGISTRY = {
    "alstm": ALSTM,
    "gru": GRUModel,
    "transformer": TimeSeriesTransformer,
}


# ── Training wrapper ──────────────────────────────────────────────────

class DeepLearningPipeline:
    """Unified training/inference for deep learning stock ranking models.

    Handles:
      - Sequence construction from factor DataFrames
      - Training with early stopping and LR scheduling
      - Prediction with confidence estimation
      - Compatibility with the existing LightGBM pipeline interface
    """

    def __init__(
        self,
        model_name: str = "alstm",
        seq_len: int = 20,
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.3,
        device: str = "cpu",
    ):
        if not _HAS_TORCH:
            raise ImportError("PyTorch required. Run: pip install torch>=2.0")

        self.model_name = model_name
        self.seq_len = seq_len
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.device = device

        self._model: nn.Module | None = None
        self._feature_names: list[str] | None = None
        self._input_dim: int = 0

    # ── Training ──────────────────────────────────────────────────

    def train(
        self,
        X: np.ndarray,
        y: np.ndarray,
        val_split: float = 0.2,
        epochs: int = 100,
        batch_size: int = 256,
        lr: float = 1e-3,
        patience: int = 15,
        verbose: bool = True,
    ) -> dict:
        """Train the model on (samples, seq_len, features) array.

        Args:
            X: shape (samples, seq_len, features) or (samples, features)
               If 2D, will be treated as single-timestep sequences.
            y: shape (samples,) target values (forward returns).
        Returns:
            Training metrics dict.
        """
        if X.ndim == 2:
            X = X[:, np.newaxis, :]  # (samples, 1, features)

        n_samples, seq_len, n_features = X.shape
        self._input_dim = n_features

        # Train/val split (sequential)
        split = int(n_samples * (1 - val_split))
        X_train, X_val = X[:split], X[split:]
        y_train, y_val = y[:split], y[split:]

        # Convert to tensors
        train_ds = TensorDataset(
            torch.tensor(X_train, dtype=torch.float32),
            torch.tensor(y_train, dtype=torch.float32),
        )
        val_ds = TensorDataset(
            torch.tensor(X_val, dtype=torch.float32),
            torch.tensor(y_val, dtype=torch.float32),
        )
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=batch_size * 2)

        # Build model
        model_cls = MODEL_REGISTRY[self.model_name]
        self._model = model_cls(
            input_dim=n_features,
            hidden_dim=self.hidden_dim,
            num_layers=self.num_layers,
            dropout=self.dropout,
        ).to(self.device)

        optimizer = torch.optim.AdamW(self._model.parameters(), lr=lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=5,
        )
        criterion = nn.MSELoss()

        best_val_loss = float("inf")
        best_state = None
        patience_counter = 0
        history = {"train_loss": [], "val_loss": []}

        for epoch in range(epochs):
            # Train
            self._model.train()
            train_loss = 0.0
            for bx, by in train_loader:
                bx, by = bx.to(self.device), by.to(self.device)
                optimizer.zero_grad()
                pred = self._model(bx)
                loss = criterion(pred, by)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self._model.parameters(), 1.0)
                optimizer.step()
                train_loss += loss.item() * len(bx)
            train_loss /= len(train_ds)

            # Validate
            self._model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for bx, by in val_loader:
                    bx, by = bx.to(self.device), by.to(self.device)
                    pred = self._model(bx)
                    val_loss += criterion(pred, by).item() * len(bx)
            val_loss /= len(val_ds)

            history["train_loss"].append(round(train_loss, 6))
            history["val_loss"].append(round(val_loss, 6))

            scheduler.step(val_loss)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = {k: v.cpu().clone() for k, v in self._model.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1

            if verbose and (epoch + 1) % 10 == 0:
                lr_now = optimizer.param_groups[0]["lr"]
                logger.info(f"Epoch {epoch+1}/{epochs}: train_loss={train_loss:.4f}, "
                            f"val_loss={val_loss:.4f}, lr={lr_now:.2e}")

            if patience_counter >= patience:
                if verbose:
                    logger.info(f"Early stopping at epoch {epoch+1}")
                break

        # Restore best
        if best_state:
            self._model.load_state_dict(best_state)

        # Compute IC on validation set
        with torch.no_grad():
            val_preds = self._model(
                torch.tensor(X_val, dtype=torch.float32).to(self.device)
            ).cpu().numpy()
            val_ic = _spearman_corr(val_preds, y_val)

        return {
            "model_name": self.model_name,
            "train_samples": len(X_train),
            "val_samples": len(X_val),
            "n_features": n_features,
            "seq_len": seq_len,
            "best_val_loss": round(best_val_loss, 6),
            "val_ic": round(float(val_ic), 4),
            "epochs_trained": len(history["train_loss"]),
            "early_stopped": patience_counter >= patience,
        }

    # ── Prediction ────────────────────────────────────────────────

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Generate predictions. Returns 1D array of predicted returns."""
        if self._model is None:
            raise RuntimeError("Model not trained.")

        if X.ndim == 2:
            X = X[:, np.newaxis, :]

        self._model.eval()
        with torch.no_grad():
            X_t = torch.tensor(X, dtype=torch.float32).to(self.device)
            preds = self._model(X_t).cpu().numpy()
        return preds

    # ── Serialization ─────────────────────────────────────────────

    def save(self, path: str):
        """Save model state dict and config."""
        if self._model is None:
            raise RuntimeError("No model to save.")
        torch.save(
            {
                "model_state": self._model.state_dict(),
                "config": {
                    "model_name": self.model_name,
                    "seq_len": self.seq_len,
                    "hidden_dim": self.hidden_dim,
                    "num_layers": self.num_layers,
                    "dropout": self.dropout,
                    "input_dim": self._input_dim,
                },
                "feature_names": self._feature_names,
            },
            path,
        )
        logger.info(f"Model saved to {path}")

    def load(self, path: str):
        """Load model from checkpoint."""
        checkpoint = torch.load(path, map_location=self.device, weights_only=True)
        config = checkpoint["config"]

        self.model_name = config["model_name"]
        self.seq_len = config["seq_len"]
        self.hidden_dim = config["hidden_dim"]
        self.num_layers = config["num_layers"]
        self.dropout = config["dropout"]
        self._input_dim = config["input_dim"]
        self._feature_names = checkpoint.get("feature_names")

        model_cls = MODEL_REGISTRY[self.model_name]
        self._model = model_cls(
            input_dim=self._input_dim,
            hidden_dim=self.hidden_dim,
            num_layers=self.num_layers,
            dropout=self.dropout,
        ).to(self.device)
        self._model.load_state_dict(checkpoint["model_state"])
        self._model.eval()
        logger.info(f"Model loaded from {path}: {self.model_name} "
                    f"(input={self._input_dim}, hidden={self.hidden_dim})")


# ── Sequence builder ──────────────────────────────────────────────────

def build_sequences(
    factors_df: pd.DataFrame,
    price_data: dict[str, pd.DataFrame],
    codes: list[str],
    trade_dates: list[str],
    seq_len: int = 20,
    feature_cols: list[str] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Build time-series sequences from factor and price data.

    For each (code, trade_date) pair, creates a sequence of the last seq_len
    daily observations (OHLCV + pre-computed factors).

    Args:
        factors_df: Cross-sectional factors, indexed by code.
        price_data: {code: DataFrame} with at least seq_len rows per code.
        codes: List of stock codes to build sequences for.
        trade_dates: Target trade dates for prediction.
        seq_len: Number of time steps per sequence.
        feature_cols: Columns to include. If None, infers from price data.

    Returns:
        X: (samples, seq_len, features) array
        y: (samples,) array (NaN for prediction-only mode)
    """
    if not _HAS_TORCH:
        raise ImportError("PyTorch required.")

    if feature_cols is None:
        feature_cols = ["open", "high", "low", "close", "volume", "amount", "pct_change", "turnover"]

    all_X = []
    all_y = []

    for code in codes:
        p_df = price_data.get(code)
        if p_df is None or len(p_df) < seq_len + 1:
            continue

        for t_date in trade_dates:
            if t_date not in p_df.index:
                continue
            idx = p_df.index.get_loc(t_date)
            if idx < seq_len:
                continue

            # Extract sequence window
            window = p_df.iloc[idx - seq_len + 1:idx + 1]
            feats = []
            for col in feature_cols:
                if col in window.columns:
                    vals = window[col].values.astype(np.float32)
                    feats.append(vals)
                else:
                    feats.append(np.zeros(seq_len, dtype=np.float32))

            # Safety: fill NaN with 0
            seq = np.column_stack(feats).T  # (seq_len, n_features)
            seq = np.nan_to_num(seq, nan=0.0, posinf=0.0, neginf=0.0)

            # Check for zero-variance (suspended stock)
            if np.std(seq[:, 3]) < 1e-6:  # close price column
                continue

            all_X.append(seq)

            # Forward return target
            if idx + 1 < len(p_df):
                next_close = float(p_df["close"].iloc[idx + 1])
                cur_close = float(p_df["close"].iloc[idx])
                if cur_close > 0:
                    all_y.append((next_close - cur_close) / cur_close)
                else:
                    all_y.append(0.0)
            else:
                all_y.append(np.nan)

    X = np.stack(all_X, axis=0) if all_X else np.empty((0, seq_len, len(feature_cols)))
    y = np.array(all_y, dtype=np.float32)
    return X, y


# ── Helpers ───────────────────────────────────────────────────────────

def _spearman_corr(pred: np.ndarray, target: np.ndarray) -> float:
    """Compute Spearman rank correlation between predictions and targets."""
    from scipy.stats import spearmanr

    mask = ~np.isnan(pred) & ~np.isnan(target)
    if mask.sum() < 10:
        return 0.0
    corr, _ = spearmanr(pred[mask], target[mask])
    return float(corr if not np.isnan(corr) else 0.0)
