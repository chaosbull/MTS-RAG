# Copyright 2026 chaosbull
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Author: chaosbull
# https://github.com/chaosbull

"""Meta Ridge fusion of block predictions and optional Raman soft sensors."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler

from .config import META_RIDGE_ALPHA, PROCESS_BLOCKS


def _pretty_source(col: str) -> str:
    if col.startswith("pred_"):
        bid = col[len("pred_") :]
        return PROCESS_BLOCKS.get(bid, {}).get("label", bid.replace("_", " ").title())
    if col.startswith("soft_"):
        return f"Raman · {col[len('soft_'):]}"
    return col


@dataclass
class MetaModelResult:
    block_weights: Dict[str, float]
    intercept: float
    r2_train: float
    r2_test: float
    mae_test: float
    weight_table: pd.DataFrame = field(default_factory=pd.DataFrame)
    extra_features: List[str] = field(default_factory=list)
    y_pred_test: Optional[np.ndarray] = None
    y_true_test: Optional[np.ndarray] = None


def fit_meta_model(
    train_pred: pd.DataFrame,
    test_pred: pd.DataFrame,
    pred_cols: Optional[List[str]] = None,
    extra_train: Optional[pd.DataFrame] = None,
    extra_test: Optional[pd.DataFrame] = None,
    alpha: float = META_RIDGE_ALPHA,
) -> MetaModelResult:
    if pred_cols is None:
        pred_cols = [c for c in train_pred.columns if c.startswith("pred_")]

    def _pack(base, extra):
        X = base[pred_cols].copy()
        extras = []
        if extra is not None and len(extra):
            for c in extra.columns:
                X[c] = extra[c].values
                extras.append(c)
        y = base["y_true"].values
        mask = np.isfinite(X.to_numpy()).all(axis=1) & np.isfinite(y)
        return X.loc[mask], y[mask], extras

    Xtr, ytr, extras = _pack(train_pred, extra_train)
    Xte, yte, _ = _pack(test_pred, extra_test)
    if len(Xtr) < 50:
        raise ValueError("Insufficient rows for meta-model")

    scaler = StandardScaler()
    Xtr_s = scaler.fit_transform(Xtr)
    Xte_s = scaler.transform(Xte)
    model = Ridge(alpha=alpha, fit_intercept=True)
    model.fit(Xtr_s, ytr)
    ptr = model.predict(Xtr_s)
    pte = model.predict(Xte_s)

    weights = {c: float(w) for c, w in zip(Xtr.columns, model.coef_)}
    abs_sum = sum(abs(v) for v in weights.values()) + 1e-12
    table = pd.DataFrame(
        {
            "source_id": list(weights.keys()),
            "source": [_pretty_source(c) for c in weights.keys()],
            "standardized_weight": list(weights.values()),
            "abs_share": [abs(v) / abs_sum for v in weights.values()],
        }
    ).sort_values("abs_share", ascending=False)

    return MetaModelResult(
        block_weights=weights,
        intercept=float(model.intercept_),
        r2_train=float(r2_score(ytr, ptr)),
        r2_test=float(r2_score(yte, pte)),
        mae_test=float(mean_absolute_error(yte, pte)),
        weight_table=table.reset_index(drop=True),
        extra_features=extras,
        y_pred_test=pte,
        y_true_test=yte,
    )
