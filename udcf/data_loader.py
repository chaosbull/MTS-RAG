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

"""Load plant batch Excel files and build an hourly modeling table."""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.interpolate import PchipInterpolator
from tqdm import tqdm

from .config import (
    CACHE_DIR,
    ERYTHROMYCIN_DATA_FOLDER,
    NET_CHANGE_VARS,
    POTENCY_MAX_VALID,
    POTENCY_MIN_VALID,
    TIME_COL_START,
    VAR_BATCH,
    VAR_CUM_TARGET,
    VAR_TARGET,
    VAR_TIME,
    VARIABLE_COL_INDEX,
    VISCOSITY_SPIKE_RATIO,
)


def clean_variable_name(name) -> str:
    if isinstance(name, str):
        name = name.strip()
        name = re.sub(r"\s*\([^)]*\)", "", name)
        name = re.sub(r"\s+", "", name)
    return name


def is_viscosity_var(var_name) -> bool:
    return "粘度" in str(var_name)


def fix_viscosity_spikes(values: np.ndarray) -> np.ndarray:
    v = np.asarray(values, dtype=float).copy()
    n = len(v)
    for i in range(n - 2):
        left, right, right_right = v[i], v[i + 1], v[i + 2]
        if np.isnan(left) or np.isnan(right) or right <= 0:
            continue
        if left > VISCOSITY_SPIKE_RATIO * right:
            v[i + 1] = (left + right_right) / 2.0 if not np.isnan(right_right) else np.nan
    return v


def smooth_interp_at_hours(times, values, hour_grid) -> np.ndarray:
    times = np.asarray(times, dtype=float)
    values = np.asarray(values, dtype=float)
    valid = ~np.isnan(times) & ~np.isnan(values)
    times, values = times[valid], values[valid]
    if len(times) == 0:
        return np.full(len(hour_grid), np.nan)
    if len(times) == 1:
        return np.full(len(hour_grid), values[0])
    tmp = pd.DataFrame({"t": times, "v": values}).groupby("t", as_index=False)["v"].mean()
    times, values = tmp["t"].values, tmp["v"].values
    order = np.argsort(times)
    times, values = times[order], values[order]
    return PchipInterpolator(times, values, extrapolate=True)(hour_grid)


def preprocess_batch_hourly(time_row, var_names, data_matrix) -> pd.DataFrame:
    t_max = int(np.ceil(np.nanmax(time_row)))
    hours = np.arange(0, t_max + 1, dtype=float)
    out = {VAR_TIME: hours}
    for v_idx, v_name in enumerate(var_names):
        vals = data_matrix[v_idx, :].astype(float).copy()
        if is_viscosity_var(v_name):
            vals = fix_viscosity_spikes(vals)
        out[v_name] = smooth_interp_at_hours(time_row, vals, hours)
    return pd.DataFrame(out)


def load_single_excel(file_path: Path) -> pd.DataFrame | None:
    try:
        df_raw = pd.read_excel(file_path, sheet_name=0, header=None)
        time_row = df_raw.iloc[0, TIME_COL_START:].values.astype(float)
        var_names = df_raw.iloc[1:, VARIABLE_COL_INDEX].apply(clean_variable_name).values
        data_matrix = df_raw.iloc[1:, TIME_COL_START:].values.astype(float)
        hourly = preprocess_batch_hourly(time_row, var_names, data_matrix)
        hourly[VAR_BATCH] = Path(file_path).stem
        return hourly
    except Exception as exc:
        print(f"Failed to read {file_path}: {exc}")
        return None


def load_all_data(folder_path: Path | None = None) -> tuple[pd.DataFrame, int, int]:
    folder = Path(folder_path) if folder_path else ERYTHROMYCIN_DATA_FOLDER
    if not folder.exists():
        raise FileNotFoundError(
            f"Batch folder not found: {folder}\n"
            "Set UDCF_DATA_DIR to the directory of plant Excel files, "
            "or place files under data/erythromycin/."
        )
    all_files = list(folder.glob("*.xlsx"))
    print(f"Found {len(all_files)} Excel files")
    df_list = []
    for path in tqdm(all_files, desc="Loading batches"):
        df_temp = load_single_excel(path)
        if df_temp is not None and not df_temp.empty:
            df_list.append(df_temp)
    if not df_list:
        raise ValueError("No valid data loaded")
    df_wide = pd.concat(df_list, ignore_index=True)
    df_wide = df_wide.sort_values([VAR_BATCH, VAR_TIME]).reset_index(drop=True)
    return df_wide, len(all_files), len(df_list)


def get_process_columns(df: pd.DataFrame) -> list[str]:
    skip = {VAR_BATCH, VAR_TIME, VAR_CUM_TARGET}
    return [c for c in df.select_dtypes(include=[np.number]).columns if c not in skip]


def prepare_model_table(df_wide: pd.DataFrame) -> pd.DataFrame:
    if VAR_TARGET not in df_wide.columns:
        raise ValueError(f"Target column {VAR_TARGET} not found: {df_wide.columns.tolist()}")

    df = df_wide.copy().sort_values([VAR_BATCH, VAR_TIME]).reset_index(drop=True)
    for col in df.select_dtypes(include=[np.number]).columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.replace([np.inf, -np.inf], np.nan)

    valid_mask = (df[VAR_TARGET] >= POTENCY_MIN_VALID) & (df[VAR_TARGET] <= POTENCY_MAX_VALID)
    df = df[valid_mask].copy()

    obs_cols = [c for c in df.columns if c not in NET_CHANGE_VARS and c not in (VAR_BATCH, VAR_TIME)]
    for col in obs_cols:
        if df[col].isna().mean() > 0.3:
            df[col] = df.groupby(VAR_BATCH)[col].transform(lambda s: s.ffill().bfill())
        else:
            df[col] = df[col].fillna(0)

    for col in NET_CHANGE_VARS:
        if col in df.columns:
            df[col] = df[col].fillna(0)

    df[VAR_CUM_TARGET] = df.groupby(VAR_BATCH, sort=False)[VAR_TARGET].cumsum()
    df[VAR_TIME] = df[VAR_TIME].round().astype(int)
    return df


def load_or_cache(force_reload: bool = False) -> pd.DataFrame:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = CACHE_DIR / "erythromycin_hourly.pkl"
    if cache_path.exists() and not force_reload:
        print(f"Loading cached table: {cache_path}")
        return pd.read_pickle(cache_path)
    df_wide, n_files, n_ok = load_all_data()
    df = prepare_model_table(df_wide)
    print(f"Prepared {len(df):,} hourly records from {n_ok}/{n_files} batches")
    try:
        df.to_pickle(cache_path)
        print(f"Cached -> {cache_path}")
    except Exception as exc:
        print(f"Cache write skipped: {exc}")
    return df
