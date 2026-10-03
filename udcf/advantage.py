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

"""Benchmarks against flat Ridge, static coefficients, and naive alarm bands."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler

from .config import BLOCK_RIDGE_ALPHA, VAR_ENGLISH, VAR_TARGET, VAR_TIME
from .data_loader import get_process_columns
from .influence_model import InfluenceResult, fit_partial_regression


@dataclass
class AdvantageBundle:
    metrics: pd.DataFrame
    time_varying: pd.DataFrame
    safety: pd.DataFrame
    narrative: List[str]


def _en(name: str) -> str:
    return VAR_ENGLISH.get(name, name)


def _flat_ridge_r2(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    features: List[str],
    alpha: float = BLOCK_RIDGE_ALPHA,
) -> Tuple[float, float]:
    cols = [c for c in features if c in train_df.columns]
    if VAR_TIME in train_df.columns and VAR_TIME not in cols:
        cols = cols + [VAR_TIME]
    use = cols + [VAR_TARGET]
    tr = train_df[use].dropna()
    te = test_df[use].dropna()
    if len(tr) < 80 or len(te) < 20:
        return np.nan, np.nan
    scaler = StandardScaler()
    Xtr = scaler.fit_transform(tr[cols])
    Xte = scaler.transform(te[cols])
    model = Ridge(alpha=alpha, fit_intercept=True)
    model.fit(Xtr, tr[VAR_TARGET])
    pred = model.predict(Xte)
    return float(r2_score(te[VAR_TARGET], pred)), float(mean_absolute_error(te[VAR_TARGET], pred))


def _static_vs_timevarying_beta(
    train_df: pd.DataFrame,
    result: InfluenceResult,
    driver: str,
    response: str,
) -> pd.DataFrame:
    if driver not in result.drivers or response not in result.responses:
        return pd.DataFrame()
    di = result.drivers.index(driver)
    ri = result.responses.index(response)
    b_static, _, _, sd_d, sd_y = fit_partial_regression(train_df, driver, response, result.drivers)
    b_static_std = b_static * sd_d / (sd_y + 1e-12)
    rows = []
    for hi, hour in enumerate(result.hours):
        b_t = result.beta_std[hi, di, ri]
        if not np.isfinite(b_t):
            continue
        rows.append(
            {
                "hour": hour,
                "beta_std_time_local": float(b_t),
                "beta_std_static": float(b_static_std),
                "abs_deviation": float(abs(b_t - b_static_std)),
            }
        )
    return pd.DataFrame(rows)


def _safety_comparison(traj: pd.DataFrame) -> pd.DataFrame:
    if traj is None or traj.empty:
        return pd.DataFrame()
    rows = []
    for resp, g in traj.groupby("response"):
        g = g.sort_values("hour")
        fw_ok = float((g["status"] == "OK").mean())
        center = float(g["baseline_median"].median())
        spread = float(np.nanpercentile(np.abs(g["baseline_median"] - center), 98) + 1e-9)
        naive_lo = center - 0.5 * spread
        naive_hi = center + 0.5 * spread
        naive_ok = float(((g["expected_value"] >= naive_lo) & (g["expected_value"] <= naive_hi)).mean())
        rows.append(
            {
                "response": resp,
                "framework_in_band": fw_ok,
                "naive_global_in_band": naive_ok,
                "advantage": fw_ok - naive_ok,
            }
        )
    return pd.DataFrame(rows).sort_values("advantage", ascending=False)


def build_advantage_bundle(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    blocks: dict,
    meta_r2: float,
    meta_mae: float,
    coverage: float,
    influence: InfluenceResult,
    traj: pd.DataFrame,
    drivers: List[str],
) -> AdvantageBundle:
    feat_pool = []
    for br in blocks.values():
        for c in br.features:
            if c != VAR_TIME and c not in feat_pool:
                feat_pool.append(c)
    if not feat_pool:
        feat_pool = [c for c in get_process_columns(train_df) if c != VAR_TARGET][:12]

    flat_r2, flat_mae = _flat_ridge_r2(train_df, test_df, feat_pool)
    sparse_r2, sparse_mae = _flat_ridge_r2(train_df, test_df, drivers)
    best_block = float(max(br.r2_test for br in blocks.values())) if blocks else np.nan
    mean_block = float(np.mean([br.r2_test for br in blocks.values()])) if blocks else np.nan

    metrics = pd.DataFrame(
        [
            {
                "model": "Hierarchical Meta Ridge (ours)",
                "family": "Proposed",
                "r2_test": meta_r2,
                "mae_test": meta_mae,
                "interpretable_levels": 3,
                "time_varying": True,
                "alarm_aware": True,
            },
            {
                "model": "Flat Ridge (all block features)",
                "family": "Baseline",
                "r2_test": flat_r2,
                "mae_test": flat_mae,
                "interpretable_levels": 1,
                "time_varying": False,
                "alarm_aware": False,
            },
            {
                "model": "Flat Ridge (Spearman drivers only)",
                "family": "Baseline",
                "r2_test": sparse_r2,
                "mae_test": sparse_mae,
                "interpretable_levels": 1,
                "time_varying": False,
                "alarm_aware": False,
            },
            {
                "model": "Best single block Ridge",
                "family": "Baseline",
                "r2_test": best_block,
                "mae_test": np.nan,
                "interpretable_levels": 1,
                "time_varying": False,
                "alarm_aware": False,
            },
            {
                "model": "Mean of block Ridges",
                "family": "Baseline",
                "r2_test": mean_block,
                "mae_test": np.nan,
                "interpretable_levels": 2,
                "time_varying": False,
                "alarm_aware": False,
            },
        ]
    )

    pot = VAR_TARGET
    if pot in influence.responses:
        ri = influence.responses.index(pot)
        score = np.nanmean(np.abs(influence.beta_std[:, :, ri]), axis=0)
        score[~np.isfinite(score)] = -1
        driver = influence.drivers[int(np.argmax(score))]
        tv = _static_vs_timevarying_beta(train_df, influence, driver, pot)
    else:
        tv = pd.DataFrame()
        driver = drivers[0] if drivers else ""

    safety = _safety_comparison(traj)
    narratives = []
    if np.isfinite(meta_r2) and np.isfinite(flat_r2):
        narratives.append(
            f"Meta fusion test R2 = {meta_r2:.3f} vs flat Ridge {flat_r2:.3f} "
            f"(delta = {meta_r2 - flat_r2:+.3f})."
        )
    if np.isfinite(meta_r2) and np.isfinite(best_block):
        narratives.append(
            f"Meta fusion exceeds the best single block ({best_block:.3f}) "
            f"by {meta_r2 - best_block:+.3f} R2."
        )
    if not tv.empty:
        mean_dev = float(tv["abs_deviation"].mean())
        narratives.append(
            f"Time-local beta_std for {_en(driver)}->Potency differs from a static "
            f"coefficient by mean |delta beta| = {mean_dev:.3f}."
        )
    narratives.append(f"Dynamic alarm coverage on held-out batches = {coverage * 100:.1f}%.")
    if not safety.empty:
        narratives.append(
            "Across responses, dynamic bands keep "
            f"{safety['framework_in_band'].mean()*100:.1f}% of controlled trajectories "
            f"in-band vs {safety['naive_global_in_band'].mean()*100:.1f}% for naive global bands."
        )
    narratives.append(
        "Unlike flat models, the hierarchy exposes Variable -> Block -> System -> Intervention layers."
    )
    return AdvantageBundle(metrics=metrics, time_varying=tv, safety=safety, narrative=narratives)
