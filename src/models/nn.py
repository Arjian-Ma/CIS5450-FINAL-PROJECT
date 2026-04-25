"""
src/models/nn.py
────────────────
Neural Network models for predicting log_copies_sold using PyTorch.

Five architectures are defined, each using ReLU activations with varying depth
and width. All share the same NeuralNetModel interface used by the project's
other model classes (fit / predict / evaluate).

Architectures
─────────────
  NN1 – Shallow     : 2 hidden layers  [in → 256 → 128 → 1]
  NN2 – Medium      : 3 hidden layers  [in → 512 → 256 → 128 → 1]
  NN3 – Deep        : 4 hidden layers  [in → 512 → 256 → 128 → 64 → 1]
  NN4 – Wide        : 3 hidden layers  [in → 1024 → 512 → 256 → 1]
  NN5 – Deep+Drop   : 5 hidden layers + Dropout
                      [in → 512 → 256 → 128 → 64 → 32 → 1]

Usage
─────
    from src.models.nn import build_all_nn_models, run_nn_models

    # Quick single model
    from src.models.nn import NeuralNetModel, NN_CONFIGS
    model = NeuralNetModel(**NN_CONFIGS["NN3_Deep"])
    model.fit(data["X_train"], data["y_train"],
              X_val=data["X_val"], y_val=data["y_val"])
    metrics = model.evaluate(data["X_val"], data["y_val"], split_name="val")

    # Run all five and get a summary table
    summary_df = run_nn_models(data)
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ── Optional PyTorch import ────────────────────────────────────────────────────
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logger.warning(
        "PyTorch is not installed. Install it with:\n"
        "  pip install torch --break-system-packages\n"
        "Neural network models will not be available until then."
    )

from src.evaluation.metrics import evaluate_predictions


# ─────────────────────────────────────────────────────────────────────────────
# PyTorch Module definitions
# ─────────────────────────────────────────────────────────────────────────────

def _build_mlp(
    input_dim: int,
    hidden_layers: List[int],
    dropout_rate: float = 0.0,
    batch_norm: bool = False,
) -> "nn.Sequential":
    """
    Build a fully-connected ReLU MLP for regression.

    Parameters
    ----------
    input_dim     : Number of input features.
    hidden_layers : List of hidden layer sizes, e.g. [256, 128].
    dropout_rate  : Dropout probability after each hidden layer (0 = disabled).
    batch_norm    : If True, add BatchNorm1d after each Linear (before ReLU).

    Returns
    -------
    nn.Sequential that maps (batch, input_dim) → (batch, 1).
    """
    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch is required to build neural network models.")

    layers: List[nn.Module] = []
    in_size = input_dim

    for out_size in hidden_layers:
        layers.append(nn.Linear(in_size, out_size))
        if batch_norm:
            layers.append(nn.BatchNorm1d(out_size))
        layers.append(nn.ReLU())
        if dropout_rate > 0.0:
            layers.append(nn.Dropout(p=dropout_rate))
        in_size = out_size

    # Final output: single continuous value (log_copies_sold)
    layers.append(nn.Linear(in_size, 1))

    return nn.Sequential(*layers)


# ─────────────────────────────────────────────────────────────────────────────
# NeuralNetModel — common interface wrapper
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class NeuralNetModel:
    """
    Wrapper around a PyTorch MLP that follows the project's model interface.

    Parameters
    ----------
    name          : Display name (e.g. "NN3_Deep").
    hidden_layers : List of hidden layer widths.
    dropout_rate  : Dropout probability (0 = no dropout).
    batch_norm    : Whether to add BatchNorm after each hidden layer.
    lr            : Adam learning rate.
    batch_size    : Mini-batch size for SGD.
    max_epochs    : Maximum training epochs.
    patience      : Early-stopping patience (epochs without val improvement).
    device        : 'cpu', 'cuda', or 'mps' (auto-detected if None).
    use_log_target: If False, targets are raw copiesSold. The model internally
                    scales y by y_scale for numerical stability and denormalises
                    predictions at inference time.
    """

    name: str = "NeuralNet"
    hidden_layers: List[int] = field(default_factory=lambda: [256, 128])
    dropout_rate: float = 0.0
    batch_norm: bool = False
    lr: float = 1e-3
    batch_size: int = 512
    max_epochs: int = 100
    patience: int = 10
    device: Optional[str] = None
    use_log_target: bool = False   # set True if y is log1p(copiesSold)

    # Populated after fitting
    train_metrics: dict = field(default_factory=dict)
    val_metrics: dict = field(default_factory=dict)
    test_metrics: dict = field(default_factory=dict)
    history: dict = field(default_factory=dict)

    # Internal — not part of the public API
    _net: Optional["nn.Module"] = field(default=None, repr=False, init=False)
    _input_dim: Optional[int] = field(default=None, repr=False, init=False)
    _y_scale: float = field(default=1.0, repr=False, init=False)  # internal scaling factor

    # ── Device resolution ─────────────────────────────────────────────────────

    def _resolve_device(self) -> "torch.device":
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required.")
        if self.device is not None:
            return torch.device(self.device)
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    # ── Fit ───────────────────────────────────────────────────────────────────

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        verbose: bool = True,
    ) -> "NeuralNetModel":
        """
        Train the MLP.

        Parameters
        ----------
        X_train, y_train : Training features and log-transformed targets.
        X_val,   y_val   : Optional validation data for early stopping and
                           loss curves. If None, training runs for max_epochs.
        verbose          : Print epoch summaries every 10 epochs.
        """
        if not TORCH_AVAILABLE:
            raise ImportError("Install PyTorch before calling .fit().")

        dev = self._resolve_device()
        self._input_dim = X_train.shape[1]

        # ── Internal target scaling for raw copiesSold ─────────────────────
        # Raw copiesSold reaches 343M — dividing by the training mean keeps
        # values near 1.0 so MSE loss and gradients stay numerically stable.
        y_arr = np.asarray(y_train, dtype=np.float64)
        if not self.use_log_target:
            self._y_scale = float(np.mean(y_arr[y_arr > 0])) or 1.0
        else:
            self._y_scale = 1.0

        def _scale(y):
            return np.asarray(y, dtype=np.float64) / self._y_scale

        # Build network
        self._net = _build_mlp(
            input_dim=self._input_dim,
            hidden_layers=self.hidden_layers,
            dropout_rate=self.dropout_rate,
            batch_norm=self.batch_norm,
        ).to(dev)

        # Convert to tensors (scale y if using raw target)
        def _to_tensor(X, y):
            Xt = torch.tensor(np.asarray(X), dtype=torch.float32).to(dev)
            yt = torch.tensor(_scale(y), dtype=torch.float32).view(-1, 1).to(dev)
            return Xt, yt

        Xt_train, yt_train = _to_tensor(X_train, y_train)
        has_val = X_val is not None and y_val is not None
        if has_val:
            Xt_val, yt_val = _to_tensor(X_val, y_val)

        loader = DataLoader(
            TensorDataset(Xt_train, yt_train),
            batch_size=self.batch_size,
            shuffle=True,
        )

        optimizer = optim.Adam(self._net.parameters(), lr=self.lr)
        criterion = nn.MSELoss()

        train_losses: List[float] = []
        val_losses: List[float] = []
        best_val_loss = float("inf")
        best_state = None
        no_improve = 0

        t0 = time.time()

        for epoch in range(1, self.max_epochs + 1):
            # ── Training pass ─────────────────────────────────────────────────
            self._net.train()
            batch_losses: List[float] = []
            for Xb, yb in loader:
                optimizer.zero_grad()
                pred = self._net(Xb)
                loss = criterion(pred, yb)
                loss.backward()
                # Gradient clipping — prevents exploding gradients on raw scale
                torch.nn.utils.clip_grad_norm_(self._net.parameters(), max_norm=1.0)
                optimizer.step()
                batch_losses.append(loss.item())

            train_loss = float(np.mean(batch_losses))
            train_losses.append(train_loss)

            # ── Validation pass ───────────────────────────────────────────────
            if has_val:
                self._net.eval()
                with torch.no_grad():
                    val_pred = self._net(Xt_val)
                    val_loss = criterion(val_pred, yt_val).item()
                val_losses.append(val_loss)

                # Early stopping
                if val_loss < best_val_loss - 1e-6:
                    best_val_loss = val_loss
                    best_state = {
                        k: v.cpu().clone()
                        for k, v in self._net.state_dict().items()
                    }
                    no_improve = 0
                else:
                    no_improve += 1

                if no_improve >= self.patience:
                    if verbose:
                        logger.info(
                            "[%s] Early stopping at epoch %d  "
                            "(val_loss=%.5f, best=%.5f)",
                            self.name, epoch, val_loss, best_val_loss,
                        )
                    break

            if verbose and epoch % 10 == 0:
                msg = f"[{self.name}] Epoch {epoch:>4d}/{self.max_epochs}"
                msg += f"  train_loss={train_loss:.5f}"
                if has_val:
                    msg += f"  val_loss={val_losses[-1]:.5f}"
                msg += f"  ({time.time()-t0:.1f}s)"
                logger.info(msg)

        # Restore best weights if early stopping fired
        if best_state is not None:
            self._net.load_state_dict(
                {k: v.to(dev) for k, v in best_state.items()}
            )

        self.history = {
            "train_loss": train_losses,
            "val_loss": val_losses,
        }

        if verbose:
            logger.info(
                "[%s] Training complete — %d epochs, %.1fs",
                self.name, len(train_losses), time.time() - t0,
            )

        return self

    # ── Predict ───────────────────────────────────────────────────────────────

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Return predictions in the original target scale (copies sold or
        log_copies_sold depending on use_log_target).
        """
        if self._net is None:
            raise RuntimeError(f"Model '{self.name}' has not been fitted yet.")
        dev = self._resolve_device()
        self._net.eval()
        Xt = torch.tensor(np.asarray(X), dtype=torch.float32).to(dev)
        with torch.no_grad():
            preds = self._net(Xt).cpu().numpy().flatten()
        # Denormalise back to original scale
        preds = preds * self._y_scale
        # Clip negatives — copies sold can't be negative
        if not self.use_log_target:
            preds = np.clip(preds, 0, None)
        return preds

    # ── Evaluate ──────────────────────────────────────────────────────────────

    def evaluate(
        self,
        X: np.ndarray,
        y: np.ndarray | pd.Series,
        split_name: str = "val",
    ) -> dict:
        """
        Compute RMSE / MAE / R² on the target scale.

        When use_log_target=False (raw copiesSold), metrics are computed
        directly on raw scale without any log transformation.
        When use_log_target=True, delegates to evaluate_predictions which
        also returns inverse-transformed raw-scale metrics.

        Parameters
        ----------
        X          : Feature matrix.
        y          : Ground-truth target values (raw or log depending on flag).
        split_name : One of 'train', 'val', 'test' — stored on the model.
        """
        from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

        y_pred = self.predict(X)
        y_true = np.asarray(y, dtype=np.float64)

        if self.use_log_target:
            # y is log1p(copiesSold) — use the shared evaluate_predictions
            metrics = evaluate_predictions(y_true, y_pred, model_name=self.name)
        else:
            # y is raw copiesSold — compute metrics directly on raw scale
            rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
            mae  = float(mean_absolute_error(y_true, y_pred))
            r2   = float(r2_score(y_true, y_pred))
            metrics = dict(
                rmse_log=rmse,   # key name kept for compatibility with runner
                mae_log=mae,
                r2_log=r2,
                rmse_raw=rmse,
                mae_raw=mae,
            )
            logger.debug(
                "[%s]  RMSE=%.0f  MAE=%.0f  R²=%.4f",
                self.name, rmse, mae, r2,
            )

        if split_name == "train":
            self.train_metrics = metrics
        elif split_name == "val":
            self.val_metrics = metrics
        elif split_name == "test":
            self.test_metrics = metrics
        return metrics

    # ── Summary ───────────────────────────────────────────────────────────────

    def summary(self) -> str:
        """Return a human-readable architecture summary."""
        if not TORCH_AVAILABLE:
            return f"{self.name}: PyTorch not installed."
        if self._net is None:
            layers_str = " → ".join(
                ["input"] + [str(h) for h in self.hidden_layers] + ["1"]
            )
            return f"{self.name} (not yet fitted): {layers_str}"
        total_params = sum(p.numel() for p in self._net.parameters())
        layers_str = " → ".join(
            ["input"] + [str(h) for h in self.hidden_layers] + ["1"]
        )
        return (
            f"{self.name}  |  {layers_str}"
            f"  |  dropout={self.dropout_rate}"
            f"  |  {total_params:,} params"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Five architecture configurations
# ─────────────────────────────────────────────────────────────────────────────

NN_CONFIGS: dict[str, dict] = {
    # ── NN1: Shallow — 2 hidden layers ───────────────────────────────────────
    "NN1_Shallow": dict(
        name="NN1_Shallow",
        hidden_layers=[256, 128],
        dropout_rate=0.0,
        batch_norm=False,
        lr=1e-3,
        batch_size=512,
        max_epochs=100,
        patience=10,
        use_log_target=False,
    ),
    # ── NN2: Medium — 3 hidden layers ────────────────────────────────────────
    "NN2_Medium": dict(
        name="NN2_Medium",
        hidden_layers=[512, 256, 128],
        dropout_rate=0.0,
        batch_norm=False,
        lr=1e-3,
        batch_size=512,
        max_epochs=100,
        patience=10,
        use_log_target=False,
    ),
    # ── NN3: Deep — 4 hidden layers ──────────────────────────────────────────
    "NN3_Deep": dict(
        name="NN3_Deep",
        hidden_layers=[512, 256, 128, 64],
        dropout_rate=0.0,
        batch_norm=False,
        lr=5e-4,
        batch_size=512,
        max_epochs=150,
        patience=15,
        use_log_target=False,
    ),
    # ── NN4: Wide — 3 wide hidden layers ─────────────────────────────────────
    "NN4_Wide": dict(
        name="NN4_Wide",
        hidden_layers=[1024, 512, 256],
        dropout_rate=0.0,
        batch_norm=False,
        lr=1e-3,
        batch_size=512,
        max_epochs=100,
        patience=10,
        use_log_target=False,
    ),
    # ── NN5: Deep + Dropout — 5 hidden layers with regularisation ────────────
    "NN5_DeepDrop": dict(
        name="NN5_DeepDrop",
        hidden_layers=[512, 256, 128, 64, 32],
        dropout_rate=0.3,
        batch_norm=True,
        lr=5e-4,
        batch_size=256,
        max_epochs=200,
        patience=20,
        use_log_target=False,
    ),
}


# ─────────────────────────────────────────────────────────────────────────────
# Convenience builders
# ─────────────────────────────────────────────────────────────────────────────

def build_all_nn_models() -> list[NeuralNetModel]:
    """Return one un-fitted NeuralNetModel for each of the five configurations."""
    return [NeuralNetModel(**cfg) for cfg in NN_CONFIGS.values()]


# ─────────────────────────────────────────────────────────────────────────────
# Convenience runner
# ─────────────────────────────────────────────────────────────────────────────

def run_nn_models(
    data: dict,
    *,
    verbose: bool = True,
    configs: Optional[list[str]] = None,
) -> pd.DataFrame:
    """
    Train all five neural network architectures and return a comparison table.

    Parameters
    ----------
    data    : dict from src.features.engineer.prepare_features()
    verbose : Print training progress.
    configs : Optional list of config keys to run (e.g. ["NN1_Shallow"]).
              If None, all five are run.

    Returns
    -------
    summary_df : DataFrame indexed by model name with RMSE / MAE / R² columns.
    """
    X_train = data["X_train"]
    X_val   = data["X_val"]
    X_test  = data["X_test"]
    y_train = data["y_train"]
    y_val   = data["y_val"]
    y_test  = data["y_test"]

    selected = configs if configs else list(NN_CONFIGS.keys())
    records: list[dict] = []

    for key in selected:
        if key not in NN_CONFIGS:
            logger.warning("Unknown NN config '%s', skipping.", key)
            continue

        model = NeuralNetModel(**NN_CONFIGS[key])
        logger.info("=" * 60)
        logger.info("Training %s", model.summary())
        logger.info("=" * 60)

        model.fit(X_train, y_train, X_val=X_val, y_val=y_val, verbose=verbose)

        tr  = model.evaluate(X_train, y_train, split_name="train")
        val = model.evaluate(X_val,   y_val,   split_name="val")
        tst = model.evaluate(X_test,  y_test,  split_name="test")

        records.append(
            {
                "model":          model.name,
                "architecture":   " → ".join(
                    str(h) for h in model.hidden_layers
                ),
                "n_params":       (
                    sum(p.numel() for p in model._net.parameters())
                    if model._net else 0
                ),
                "dropout":        model.dropout_rate,
                "train_RMSE_log": round(tr["rmse_log"],  4),
                "val_RMSE_log":   round(val["rmse_log"], 4),
                "test_RMSE_log":  round(tst["rmse_log"], 4),
                "val_MAE_log":    round(val["mae_log"],  4),
                "val_R2_log":     round(val["r2_log"],   4),
                "val_RMSE_raw":   round(val["rmse_raw"], 0),
            }
        )

        logger.info(
            "  %-20s  val RMSE(log)=%.4f  val R²=%.4f",
            model.name, val["rmse_log"], val["r2_log"],
        )

    return pd.DataFrame(records).set_index("model")
