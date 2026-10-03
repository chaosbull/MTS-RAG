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

"""Disjoint block Ridge models for process groups."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler

from .config import (
    BLOCK_RIDGE_ALPHA,
    PROCESS_BLOCKS,
    VAR_BATCH,
    VAR_ENGLISH,
    VAR_TARGET,
    VAR_TIME,
)
from .data_loader import get_process_columns


def _en(name: str) -> str:
    return VAR_ENGLISH.get(name, name)


def match_block_columns(df: pd.DataFrame, patterns: List[str]) -> List[str]:
    cols = get_process_columns(df)
    hit = []
    for c in cols:
        if c == VAR_TARGET:
            continue
        if any(p in str(c) for p in patterns):
            hit.append(c)
    return hit


def assign_disjoint_blocks(df: pd.DataFrame) -> Dict[str, List[str]]:
    claimed: set[str] = set()
    out: Dict[str, List[str]] = {}
    for bid, meta in PROCESS_BLOCKS.items():
        cols = []
        for c in match_block_columns(df, meta["patterns"]):
            if c not in claimed:
                cols.append(c)
                claimed.add(c)
        out[bid] = cols
    return out


@dataclass
class BlockModelResult:
    block_id: str
    block_label: str
    features: List[str]
    coefficients: Dict[str, float]
    intercept: float
    r2_train: float
    r2_test: float
    mae_test: float
    n_train: int
    n_test: int
    feature_importance: pd.DataFrame = field(default_factory=pd.DataFrame)
    scaler: Optional[StandardScaler] = None
    model: Optional[Ridge] = None


def fit_block_model(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    block_id: str,
    feature_cols: Optional[List[str]] = None,
    target: str = VAR_TARGET,
    alpha: float = BLOCK_RIDGE_ALPHA,
) -> Optional[BlockModelResult]:
    meta = PROCESS_BLOCKS[block_id]
    feats = list(feature_cols) if feature_cols is not None else match_block_columns(train_df, meta["patterns"])
    if not feats:
        return None

    # One shared stabilizer can help weak blocks without claiming exclusive membership.
    stabilizers = [c for c in ("总糖", "粘度", "氨基氮") if c in train_df.columns and c not in feats]
    use_cols = list(feats) + stabilizers[:1]
    if VAR_TIME in train_df.columns and VAR_TIME not in use_cols:
        use_cols.append(VAR_TIME)
    seen: set[str] = set()
    use_cols = [c for c in use_cols if not (c in seen or seen.add(c))]

    def _xy(frame: pd.DataFrame):
        d = frame[use_cols + [target]].dropna()
        return d[use_cols], d[target]

    Xtr, ytr = _xy(train_df)
    Xte, yte = _xy(test_df)
    if len(Xtr) < 80 or len(Xte) < 20:
        return None

    scaler = StandardScaler()
    Xtr_s = scaler.fit_transform(Xtr)
    Xte_s = scaler.transform(Xte)

    try:
        model = RidgeCV(alphas=np.logspace(-1, 2, 8), fit_intercept=True)
        model.fit(Xtr_s, ytr)
    except Exception:
        n_proc = max(1, len(feats))
        alpha_use = float(alpha if n_proc >= 3 else max(1.0, alpha * 0.5))
        model = Ridge(alpha=alpha_use, fit_intercept=True)
        model.fit(Xtr_s, ytr)

    pred_tr = model.predict(Xtr_s)
    pred_te = model.predict(Xte_s)
    coefs = {_en(c): float(w) for c, w in zip(use_cols, model.coef_)}
    roles = ["block" if (c in feats or c == VAR_TIME) else "stabilizer" for c in use_cols]
    imp = (
        pd.DataFrame(
            {
                "feature": [_en(c) for c in use_cols],
                "feature_cn": use_cols,
                "role": roles,
                "standardized_coef": model.coef_,
                "abs_coef": np.abs(model.coef_),
            }
        )
        .sort_values("abs_coef", ascending=False)
        .reset_index(drop=True)
    )

    return BlockModelResult(
        block_id=block_id,
        block_label=meta["label"],
        features=use_cols,
        coefficients=coefs,
        intercept=float(model.intercept_),
        r2_train=float(r2_score(ytr, pred_tr)),
        r2_test=float(r2_score(yte, pred_te)),
        mae_test=float(mean_absolute_error(yte, pred_te)),
        n_train=len(Xtr),
        n_test=len(Xte),
        feature_importance=imp,
        scaler=scaler,
        model=model,
    )


def fit_all_blocks(train_df: pd.DataFrame, test_df: pd.DataFrame) -> Dict[str, BlockModelResult]:
    assignment = assign_disjoint_blocks(train_df)
    out = {}
    for bid in PROCESS_BLOCKS:
        res = fit_block_model(train_df, test_df, bid, feature_cols=assignment.get(bid, []))
        if res is not None:
            out[bid] = res
    return out


def block_predictions(
    df: pd.DataFrame,
    blocks: Dict[str, BlockModelResult],
    target: str = VAR_TARGET,
) -> pd.DataFrame:
    preds = pd.DataFrame(index=df.index)
    preds[VAR_BATCH] = df[VAR_BATCH].values
    preds[VAR_TIME] = df[VAR_TIME].values
    preds["y_true"] = df[target].values

    for bid, br in blocks.items():
        cols = [c for c in br.features if c in df.columns]
        if len(cols) != len(br.features):
            preds[f"pred_{bid}"] = np.nan
            continue
        X = df[cols].copy()
        if br.scaler is not None and br.model is not None:
            fill = {c: float(br.scaler.mean_[i]) for i, c in enumerate(cols)}
            X = X.fillna(fill)
            preds[f"pred_{bid}"] = br.model.predict(br.scaler.transform(X))
        else:
            X = X.fillna(X.median(numeric_only=True))
            Xs = StandardScaler().fit_transform(X)
            w = np.array([br.coefficients[_en(c)] for c in cols], dtype=float)
            preds[f"pred_{bid}"] = Xs @ w + br.intercept
    return preds
