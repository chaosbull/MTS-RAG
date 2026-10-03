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

"""Time-local partial regression, alarm bands, and full-horizon propagation.

Heatmaps use standardized coefficients:
    beta_std = beta_raw * sigma_driver / sigma_response
Control / propagation always use beta_raw in native units.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from .config import (
    ALARM_BAND_WIDTH_FACTOR,
    ALARM_QUANTILE_HIGH,
    ALARM_QUANTILE_LOW,
    COEVOLUTION_GRID,
    HOUR_STRIDE,
    LOCAL_RIDGE_ALPHA,
    MIN_INFLUENCE_ABS,
    MIN_SAMPLES_PER_HOUR,
    PERTURBATION_SIGMA,
    TIME_WINDOW_HOURS,
    TOP_N_DRIVERS,
    VAR_BATCH,
    VAR_ENGLISH,
    VAR_TARGET,
    VAR_TIME,
)
from .data_loader import get_process_columns


@dataclass
class InfluenceResult:
    drivers: List[str]
    responses: List[str]
    hours: List[int]
    beta_raw: np.ndarray
    beta_std: np.ndarray
    propagation_matrix: np.ndarray
    alarm_low: np.ndarray
    alarm_high: np.ndarray
    r2_validation: Dict[str, float] = field(default_factory=dict)
    coverage_rate: float = 0.0
    driver_sigma: Dict[str, float] = field(default_factory=dict)
    response_sigma: Dict[str, float] = field(default_factory=dict)

    @property
    def beta_matrix(self) -> np.ndarray:
        return self.beta_std


def english(name: str) -> str:
    return VAR_ENGLISH.get(name, name)


def select_top_drivers(df: pd.DataFrame, n: int = TOP_N_DRIVERS) -> List[str]:
    cols = [c for c in get_process_columns(df) if c != VAR_TARGET]
    corrs = {}
    y = df[VAR_TARGET]
    for c in cols:
        s = df[c]
        if s.nunique(dropna=True) < 5:
            continue
        rho = s.corr(y, method="spearman")
        if pd.notna(rho):
            corrs[c] = abs(float(rho))
    return sorted(corrs, key=corrs.get, reverse=True)[:n]


def time_local_data(df: pd.DataFrame, hour: int, window: int = TIME_WINDOW_HOURS) -> pd.DataFrame:
    return df[(df[VAR_TIME] >= hour - window) & (df[VAR_TIME] <= hour + window)].copy()


def shrink_alarm_band(
    center: float, lo: float, hi: float, factor: float = ALARM_BAND_WIDTH_FACTOR
) -> Tuple[float, float]:
    factor = float(np.clip(factor, 0.05, 1.0))
    lo2 = center - factor * (center - lo)
    hi2 = center + factor * (hi - center)
    if lo2 > hi2:
        lo2, hi2 = hi2, lo2
    return float(lo2), float(hi2)


def fit_partial_regression(
    sub: pd.DataFrame,
    driver: str,
    response: str,
    controls: List[str],
) -> Tuple[float, np.ndarray, float, float, float]:
    use = [driver, response] + [c for c in controls if c not in (driver, response)]
    use = [c for c in use if c in sub.columns]
    data = sub[use].dropna()
    if len(data) < MIN_SAMPLES_PER_HOUR:
        return 0.0, np.array([]), 0.0, 1.0, 1.0

    y = data[response].values.astype(float)
    x_cols = [driver] + [c for c in use if c not in (driver, response)]
    X = data[x_cols].values.astype(float)
    sd_d = float(np.std(X[:, 0]))
    sd_y = float(np.std(y))
    if sd_d < 1e-12 or sd_y < 1e-12:
        return 0.0, np.array([]), 0.0, 1.0, 1.0

    model = Ridge(alpha=LOCAL_RIDGE_ALPHA, fit_intercept=True)
    model.fit(X, y)
    pred = model.predict(X)
    resid = y - pred
    r2 = 1.0 - float(np.sum(resid ** 2)) / (float(np.sum((y - y.mean()) ** 2)) + 1e-12)
    return float(model.coef_[0]), resid, r2, sd_d, sd_y


def compute_influence_model(
    df: pd.DataFrame,
    drivers: Optional[List[str]] = None,
    responses: Optional[List[str]] = None,
    hour_range: Optional[Tuple[int, int]] = None,
    hour_stride: int = HOUR_STRIDE,
) -> InfluenceResult:
    if drivers is None:
        drivers = select_top_drivers(df)
    if responses is None:
        seen = set()
        responses = []
        for r in drivers + [VAR_TARGET]:
            if r not in seen:
                seen.add(r)
                responses.append(r)

    t_min = int(df[VAR_TIME].min())
    t_max = int(df[VAR_TIME].max())
    if hour_range:
        t_min, t_max = hour_range
    hours = list(range(t_min, t_max + 1, max(1, hour_stride)))

    n_h, n_d, n_r = len(hours), len(drivers), len(responses)
    beta_raw = np.full((n_h, n_d, n_r), np.nan)
    beta_std = np.full((n_h, n_d, n_r), np.nan)
    prop = np.full((n_h, n_d, n_r), np.nan)
    a_lo = np.full((n_h, n_d, n_r), np.nan)
    a_hi = np.full((n_h, n_d, n_r), np.nan)

    driver_sigma = {d: float(df[d].std(skipna=True) or 1.0) for d in drivers}
    response_sigma = {r: float(df[r].std(skipna=True) or 1.0) for r in responses}

    print(f"  Drivers: {[english(d) for d in drivers]}")
    print(f"  Time points: {n_h} hours ({t_min}-{t_max}, stride={hour_stride})")

    r2_acc: Dict[str, list] = {}
    for hi, hour in enumerate(hours):
        sub = time_local_data(df, hour)
        if len(sub) < MIN_SAMPLES_PER_HOUR:
            continue
        for di, driver in enumerate(drivers):
            for ri, response in enumerate(responses):
                if driver == response:
                    continue
                b, resid, r2, sd_d, sd_y = fit_partial_regression(sub, driver, response, drivers)
                if len(resid) == 0:
                    continue
                b_std = b * sd_d / (sd_y + 1e-12)
                beta_raw[hi, di, ri] = b
                beta_std[hi, di, ri] = b_std
                prop[hi, di, ri] = b * driver_sigma[driver] * PERTURBATION_SIGMA
                center = float(sub[response].median())
                if len(resid) >= 15:
                    lo0 = center + float(np.quantile(resid, ALARM_QUANTILE_LOW))
                    hi0 = center + float(np.quantile(resid, ALARM_QUANTILE_HIGH))
                else:
                    vals = sub[response].dropna()
                    lo0 = float(np.quantile(vals, ALARM_QUANTILE_LOW))
                    hi0 = float(np.quantile(vals, ALARM_QUANTILE_HIGH))
                a_lo[hi, di, ri], a_hi[hi, di, ri] = shrink_alarm_band(center, lo0, hi0)
                r2_acc.setdefault(f"{english(driver)}->{english(response)}", []).append(r2)

    n_sig = int(np.sum(np.isfinite(beta_std) & (np.abs(beta_std) >= MIN_INFLUENCE_ABS)))
    print(f"  Significant standardized influences: {n_sig}")

    return InfluenceResult(
        drivers=drivers,
        responses=responses,
        hours=hours,
        beta_raw=beta_raw,
        beta_std=beta_std,
        propagation_matrix=prop,
        alarm_low=a_lo,
        alarm_high=a_hi,
        r2_validation={k: float(np.nanmean(v)) for k, v in r2_acc.items()},
        driver_sigma=driver_sigma,
        response_sigma=response_sigma,
    )


def validate_alarm_coverage(
    result: InfluenceResult,
    df: pd.DataFrame,
    test_batches: Optional[List[str]] = None,
    max_records: int = 300_000,
) -> Tuple[float, pd.DataFrame]:
    data = df if test_batches is None else df[df[VAR_BATCH].isin(test_batches)]
    rows = []
    for hi, hour in enumerate(result.hours):
        sub = data[data[VAR_TIME] == hour]
        if sub.empty:
            continue
        for ri, response in enumerate(result.responses):
            lo_vals = result.alarm_low[hi, :, ri]
            hi_vals = result.alarm_high[hi, :, ri]
            finite = np.isfinite(lo_vals) & np.isfinite(hi_vals)
            if not finite.any():
                continue
            lo = float(np.nanmedian(lo_vals[finite]))
            hi_b = float(np.nanmedian(hi_vals[finite]))
            di = int(np.where(finite)[0][0])
            driver = result.drivers[di]
            obs = sub[response].dropna().values
            for o in obs:
                rows.append(
                    {
                        "hour": hour,
                        "driver": english(driver),
                        "response": english(response),
                        "observed": float(o),
                        "alarm_low": lo,
                        "alarm_high": hi_b,
                        "in_band": bool(lo <= o <= hi_b),
                    }
                )
    val_df = pd.DataFrame(rows)
    if val_df.empty:
        return 0.0, val_df
    if len(val_df) > max_records:
        val_df = val_df.sample(n=max_records, random_state=42).reset_index(drop=True)
    coverage = float(val_df["in_band"].mean())
    result.coverage_rate = coverage
    return coverage, val_df


def build_dynamic_control_table(
    result: InfluenceResult,
    df: pd.DataFrame,
    driver: str,
    target_hour: int,
    delta_driver: Optional[float] = None,
) -> pd.DataFrame:
    if driver not in result.drivers:
        raise ValueError(f"Driver {driver} not in model")
    if target_hour not in result.hours:
        target_hour = min(result.hours, key=lambda h: abs(h - target_hour))
    hi = result.hours.index(target_hour)
    di = result.drivers.index(driver)
    if delta_driver is None:
        delta_driver = result.driver_sigma.get(driver, 1.0) * PERTURBATION_SIGMA

    sub = time_local_data(df, target_hour)
    rows = []
    for ri, response in enumerate(result.responses):
        if response == driver:
            continue
        b = result.beta_raw[hi, di, ri]
        if not np.isfinite(b):
            continue
        baseline = float(sub[response].median())
        expected_change = b * delta_driver
        expected_value = baseline + expected_change
        lo = result.alarm_low[hi, di, ri]
        hi_b = result.alarm_high[hi, di, ri]
        status = "OK"
        if np.isfinite(lo) and expected_value < lo:
            status = "LOW ALARM"
        elif np.isfinite(hi_b) and expected_value > hi_b:
            status = "HIGH ALARM"
        rows.append(
            {
                "hour": target_hour,
                "driver": english(driver),
                "driver_delta": delta_driver,
                "response": english(response),
                "baseline_median": baseline,
                "expected_change": expected_change,
                "expected_value": expected_value,
                "beta_raw": b,
                "beta_std": result.beta_std[hi, di, ri],
                "alarm_low": lo,
                "alarm_high": hi_b,
                "status": status,
            }
        )
    return pd.DataFrame(rows)


def build_full_horizon_trajectory(
    result: InfluenceResult,
    df: pd.DataFrame,
    driver: str,
    delta_driver: Optional[float] = None,
    hour_max: Optional[int] = None,
) -> pd.DataFrame:
    if driver not in result.drivers:
        raise ValueError(f"Driver {driver} not in model")
    di = result.drivers.index(driver)
    if delta_driver is None:
        delta_driver = result.driver_sigma.get(driver, 1.0) * PERTURBATION_SIGMA

    rows = []
    for hi, hour in enumerate(result.hours):
        if hour_max is not None and hour > hour_max:
            continue
        sub = time_local_data(df, hour)
        for ri, response in enumerate(result.responses):
            if response == driver:
                continue
            b = result.beta_raw[hi, di, ri]
            if not np.isfinite(b):
                continue
            if response not in sub.columns or sub[response].dropna().empty:
                continue
            baseline = float(sub[response].median())
            dlt = float(b * delta_driver)
            expected = baseline + dlt
            lo = result.alarm_low[hi, di, ri]
            hi_b = result.alarm_high[hi, di, ri]
            status = "OK"
            if np.isfinite(lo) and expected < lo:
                status = "LOW ALARM"
            elif np.isfinite(hi_b) and expected > hi_b:
                status = "HIGH ALARM"
            std = result.beta_std[hi, di, ri]
            rows.append(
                {
                    "hour": hour,
                    "driver": english(driver),
                    "driver_delta": float(delta_driver),
                    "response": english(response),
                    "baseline_median": baseline,
                    "expected_change": dlt,
                    "expected_value": expected,
                    "beta_raw": float(b),
                    "beta_std": float(std) if np.isfinite(std) else np.nan,
                    "alarm_low": lo,
                    "alarm_high": hi_b,
                    "status": status,
                }
            )
    return pd.DataFrame(rows)


def horizon_delta_matrix(traj: pd.DataFrame) -> pd.DataFrame:
    if traj.empty:
        return pd.DataFrame()
    return traj.pivot_table(
        index="hour", columns="response", values="expected_change", aggfunc="mean"
    ).sort_index()


def summarize_influence(result: InfluenceResult) -> pd.DataFrame:
    rows = []
    for hi, hour in enumerate(result.hours):
        for di, driver in enumerate(result.drivers):
            for ri, response in enumerate(result.responses):
                b = result.beta_std[hi, di, ri]
                if not np.isfinite(b):
                    continue
                rows.append(
                    {
                        "hour": hour,
                        "driver": english(driver),
                        "response": english(response),
                        "partial_beta_raw": result.beta_raw[hi, di, ri],
                        "partial_beta_std": b,
                        "propagation_1sigma": result.propagation_matrix[hi, di, ri],
                        "alarm_low": result.alarm_low[hi, di, ri],
                        "alarm_high": result.alarm_high[hi, di, ri],
                    }
                )
    return pd.DataFrame(rows)


def pairwise_time_heatmap_data(
    df: pd.DataFrame,
    var_x: str,
    var_y: str,
    target: str,
    hour: int,
    grid_res: int = COEVOLUTION_GRID,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    from scipy.interpolate import griddata
    from scipy.ndimage import gaussian_filter

    sub = time_local_data(df, hour)[[var_x, var_y, target]].dropna()
    if len(sub) < MIN_SAMPLES_PER_HOUR:
        return np.array([]), np.array([]), np.array([])

    x_lo, x_hi = np.percentile(sub[var_x], [5, 95])
    y_lo, y_hi = np.percentile(sub[var_y], [5, 95])
    if x_hi <= x_lo or y_hi <= y_lo:
        return np.array([]), np.array([]), np.array([])
    m = (
        (sub[var_x] >= x_lo)
        & (sub[var_x] <= x_hi)
        & (sub[var_y] >= y_lo)
        & (sub[var_y] <= y_hi)
    )
    sub = sub.loc[m]
    if len(sub) < 30:
        return np.array([]), np.array([]), np.array([])
    if len(sub) > 4000:
        sub = sub.sample(n=4000, random_state=42)

    pts = sub[[var_x, var_y]].to_numpy(dtype=float)
    vals = sub[target].to_numpy(dtype=float)
    xi = np.linspace(x_lo, x_hi, grid_res)
    yi = np.linspace(y_lo, y_hi, grid_res)
    xx, yy = np.meshgrid(xi, yi)
    z = griddata(pts, vals, (xx, yy), method="linear")
    z_near = griddata(pts, vals, (xx, yy), method="nearest")
    miss = ~np.isfinite(z)
    z[miss] = z_near[miss]
    z = gaussian_filter(z.astype(float), sigma=2.8, mode="nearest")
    return xx, yy, z
