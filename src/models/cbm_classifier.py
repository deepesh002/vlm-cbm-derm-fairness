from __future__ import annotations

import copy
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import joblib
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler

from ..utils import get_device
from .concept_predictor import CONCEPT_IDS

logger = logging.getLogger(__name__)


# CBM-LR: sparse logistic regression
class CBMLogisticRegression:

    def __init__(
        self,
        C: float = 1.0,
        penalty: str = "l1",
        solver: str = "liblinear",
        max_iter: int = 2000,
        class_weight: Optional[str] = "balanced",
        standardize: bool = True,
        concept_names: Sequence[str] = tuple(CONCEPT_IDS),
        random_state: int = 42,
    ) -> None:
        self.C = C
        self.penalty = penalty
        self.solver = solver
        self.max_iter = max_iter
        self.class_weight = class_weight
        self.standardize = standardize
        self.concept_names = list(concept_names)
        self.random_state = random_state

        self.scaler: Optional[StandardScaler] = None
        self.model: Optional[LogisticRegression] = None

    # .....................................................................
    def fit(self, X: np.ndarray, y: np.ndarray) -> "CBMLogisticRegression":
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y).astype(int)
        if self.standardize:
            self.scaler = StandardScaler().fit(X)
            X = self.scaler.transform(X)
        self.model = LogisticRegression(
            C=self.C, penalty=self.penalty, solver=self.solver,
            max_iter=self.max_iter, class_weight=self.class_weight,
            random_state=self.random_state,
        ).fit(X, y)
        return self

    def _preprocess(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float32)
        return self.scaler.transform(X) if self.scaler is not None else X

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(self._preprocess(X))

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(self._preprocess(X))

    # .....................................................................
    def cross_validate(
        self,
        X: np.ndarray, y: np.ndarray,
        C_grid: Sequence[float] = (0.01, 0.1, 1.0, 10.0),
        cv: int = 5,
        scoring: str = "roc_auc",
    ) -> Dict[str, Any]:
        y = np.asarray(y).astype(int)
        X_proc = self._preprocess(X) if self.scaler is not None else StandardScaler().fit_transform(X) if self.standardize else X
        if self.scaler is None and self.standardize:
            self.scaler = StandardScaler().fit(X)
            X_proc = self.scaler.transform(X)
        base = LogisticRegression(
            penalty=self.penalty, solver=self.solver, max_iter=self.max_iter,
            class_weight=self.class_weight, random_state=self.random_state,
        )
        grid = GridSearchCV(
            base, param_grid={"C": list(C_grid)},
            cv=StratifiedKFold(n_splits=cv, shuffle=True,
                               random_state=self.random_state),
            scoring=scoring, n_jobs=-1,
        ).fit(X_proc, y)
        self.C = grid.best_params_["C"]
        self.model = grid.best_estimator_
        return {
            "best_C": self.C,
            "best_score": grid.best_score_,
            "cv_results": grid.cv_results_,
        }

    # .....................................................................
    def concept_weights(self) -> Dict[str, float]:
        assert self.model is not None, "Call fit() first"
        # Binary LR: model.coef_ has shape (1, n_concepts). Positive weights
        # => evidence for the positive class (melanoma).
        coef = self.model.coef_.ravel()
        return dict(zip(self.concept_names, coef.tolist()))

    def intercept(self) -> float:
        return float(self.model.intercept_[0]) if self.model is not None else 0.0

    # .....................................................................
    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"scaler": self.scaler, "model": self.model,
                     "concept_names": self.concept_names,
                     "C": self.C, "penalty": self.penalty}, path)

    @classmethod
    def load(cls, path: str | Path) -> "CBMLogisticRegression":
        state = joblib.load(path)
        obj = cls(C=state.get("C", 1.0), penalty=state.get("penalty", "l1"),
                  concept_names=state.get("concept_names", CONCEPT_IDS))
        obj.scaler = state["scaler"]
        obj.model = state["model"]
        return obj


# CBM-MLP
class _ConceptMLPModule(nn.Module):
    def __init__(self, n_concepts: int = 9, hidden_dim: int = 32,
                 dropout: float = 0.3, n_classes: int = 2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_concepts, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class CBMMLP:

    def __init__(
        self,
        n_concepts: int = 9,
        hidden_dim: int = 32,
        dropout: float = 0.3,
        n_classes: int = 2,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        epochs: int = 100,
        batch_size: int = 64,
        early_stopping_patience: int = 10,
        standardize: bool = True,
        device: Optional[str | torch.device] = None,
        random_state: int = 42,
    ) -> None:
        self.n_concepts = n_concepts
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.n_classes = n_classes
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.epochs = epochs
        self.batch_size = batch_size
        self.early_stopping_patience = early_stopping_patience
        self.standardize = standardize
        self.device = torch.device(device) if device is not None else get_device()
        self.random_state = random_state

        self.scaler: Optional[StandardScaler] = None
        self.model: Optional[_ConceptMLPModule] = None
        self.history: List[Dict[str, float]] = []

    # .....................................................................
    def fit(
        self,
        X: np.ndarray, y: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
    ) -> "CBMMLP":
        torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)

        if self.standardize:
            self.scaler = StandardScaler().fit(X)
            X = self.scaler.transform(X)
            if X_val is not None:
                X_val = self.scaler.transform(X_val)

        X_t = torch.as_tensor(X, dtype=torch.float32, device=self.device)
        y_t = torch.as_tensor(y, dtype=torch.long, device=self.device)

        self.model = _ConceptMLPModule(
            n_concepts=self.n_concepts, hidden_dim=self.hidden_dim,
            dropout=self.dropout, n_classes=self.n_classes,
        ).to(self.device)
        opt = torch.optim.AdamW(self.model.parameters(), lr=self.learning_rate,
                                weight_decay=self.weight_decay)
        # Class-balanced CE
        counts = np.bincount(np.asarray(y).astype(int), minlength=self.n_classes)
        weights = torch.tensor((counts.sum() / (counts + 1e-6)), dtype=torch.float32,
                               device=self.device)
        weights = weights / weights.mean()
        criterion = nn.CrossEntropyLoss(weight=weights)

        n = X_t.shape[0]
        idx = torch.arange(n, device=self.device)
        best_val, best_state, patience = float("inf"), None, 0

        for epoch in range(self.epochs):
            self.model.train()
            perm = idx[torch.randperm(n, device=self.device)]
            total = 0.0
            for start in range(0, n, self.batch_size):
                batch_idx = perm[start:start + self.batch_size]
                logits = self.model(X_t[batch_idx])
                loss = criterion(logits, y_t[batch_idx])
                opt.zero_grad()
                loss.backward()
                opt.step()
                total += loss.item() * batch_idx.numel()
            train_loss = total / n

            # Validation
            if X_val is not None and y_val is not None:
                with torch.no_grad():
                    self.model.eval()
                    Xv = torch.as_tensor(X_val, dtype=torch.float32, device=self.device)
                    yv = torch.as_tensor(y_val, dtype=torch.long, device=self.device)
                    val_loss = F.cross_entropy(self.model(Xv), yv).item()
                self.history.append({"epoch": epoch, "train": train_loss, "val": val_loss})
                if val_loss < best_val - 1e-4:
                    best_val, patience = val_loss, 0
                    best_state = copy.deepcopy(self.model.state_dict())
                else:
                    patience += 1
                    if patience >= self.early_stopping_patience:
                        break
            else:
                self.history.append({"epoch": epoch, "train": train_loss})

        if best_state is not None:
            self.model.load_state_dict(best_state)
        return self

    # .....................................................................
    def _preprocess(self, X: np.ndarray) -> torch.Tensor:
        if self.scaler is not None:
            X = self.scaler.transform(X)
        return torch.as_tensor(np.asarray(X, dtype=np.float32), device=self.device)

    @torch.no_grad()
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        self.model.eval()
        logits = self.model(self._preprocess(X))
        return F.softmax(logits, dim=-1).cpu().numpy()

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.argmax(self.predict_proba(X), axis=-1)

    # .....................................................................
    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "state_dict": self.model.state_dict() if self.model is not None else None,
            "scaler": self.scaler,
            "config": {
                "n_concepts": self.n_concepts,
                "hidden_dim": self.hidden_dim,
                "dropout": self.dropout,
                "n_classes": self.n_classes,
            },
        }, path)

    @classmethod
    def load(cls, path: str | Path, device: Optional[str | torch.device] = None) -> "CBMMLP":
        device = torch.device(device) if device is not None else get_device()
        state = torch.load(path, map_location=device)
        cfg = state["config"]
        obj = cls(device=device, **cfg)
        obj.model = _ConceptMLPModule(**cfg).to(device)
        obj.model.load_state_dict(state["state_dict"])
        obj.scaler = state["scaler"]
        obj.model.eval()
        return obj


# Cross-validation driver used in 03_cbm_training.ipynb
@dataclass
class CVResult:
    fold_metrics: List[Dict[str, float]] = field(default_factory=list)
    mean: Dict[str, float] = field(default_factory=dict)
    std: Dict[str, float] = field(default_factory=dict)


def stratified_cv(
    model_factory,
    X: np.ndarray, y: np.ndarray,
    n_folds: int = 5,
    metric_fn=None,
    random_state: int = 42,
) -> CVResult:
    from sklearn.metrics import roc_auc_score, average_precision_score, balanced_accuracy_score

    def default_metrics(y_true, proba, pred):
        return {
            "auroc": roc_auc_score(y_true, proba),
            "auprc": average_precision_score(y_true, proba),
            "balanced_acc": balanced_accuracy_score(y_true, pred),
        }
    metric_fn = metric_fn or default_metrics

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    out = CVResult()
    for fold, (tr, te) in enumerate(skf.split(X, y)):
        model = model_factory()
        model.fit(X[tr], y[tr])
        proba = model.predict_proba(X[te])[:, 1]
        pred = (proba >= 0.5).astype(int)
        m = metric_fn(y[te], proba, pred)
        m["fold"] = fold
        out.fold_metrics.append(m)

    keys = [k for k in out.fold_metrics[0].keys() if k != "fold"]
    out.mean = {k: float(np.mean([f[k] for f in out.fold_metrics])) for k in keys}
    out.std = {k: float(np.std([f[k] for f in out.fold_metrics])) for k in keys}
    return out
