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

"""Raman PLS soft sensors (real files or a labeled synthetic demo)."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

from .config import RAMAN_DIR, RAMAN_SOFT_TARGETS, VAR_BATCH, VAR_ENGLISH, VAR_TARGET, VAR_TIME


@dataclass
class RamanSoftSensor:
    target_cn: str
    target_en: str
    n_components: int
    r2_test: float
    wavenumbers: np.ndarray
    loading_importance: pd.DataFrame = field(default_factory=pd.DataFrame)


@dataclass
class RamanBundle:
    sensors: Dict[str, RamanSoftSensor]
    soft_predictions: pd.DataFrame
    status: str  # "real" | "synthetic_demo" | "unavailable"


def snv(X: np.ndarray) -> np.ndarray:
    mu = X.mean(axis=1, keepdims=True)
    sd = X.std(axis=1, keepdims=True) + 1e-8
    return (X - mu) / sd


def savgol_deriv1(X: np.ndarray, window: int = 11) -> np.ndarray:
    if window % 2 == 0:
        window += 1
    kernel = np.ones(window) / window
    xs = np.apply_along_axis(lambda r: np.convolve(r, kernel, mode="same"), 1, X)
    d = np.zeros_like(xs)
    d[:, 1:-1] = (xs[:, 2:] - xs[:, :-2]) / 2.0
    d[:, 0] = xs[:, 1] - xs[:, 0]
    d[:, -1] = xs[:, -1] - xs[:, -2]
    return d


def preprocess_spectra(S: np.ndarray, do_savgol: bool = True) -> np.ndarray:
    X = snv(np.asarray(S, dtype=float))
    if do_savgol:
        X = savgol_deriv1(X)
    return X


def find_raman_files(folder: Path = RAMAN_DIR) -> List[Path]:
    folder = Path(folder)
    if not folder.exists():
        return []
    return list(folder.glob("*.csv")) + list(folder.glob("*.xlsx")) + list(folder.glob("*.npy"))


def _is_float(x) -> bool:
    try:
        float(x)
        return True
    except Exception:
        return False


def load_raman_matrix(path: Path) -> Tuple[np.ndarray, np.ndarray, Optional[pd.DataFrame]]:
    path = Path(path)
    if path.suffix.lower() == ".npy":
        S = np.load(path)
        wn = np.arange(S.shape[1])
        return S, wn, None
    df = pd.read_csv(path) if path.suffix.lower() == ".csv" else pd.read_excel(path)
    meta = None
    if {"batch", "time"}.issubset({c.lower() for c in df.columns}):
        rename = {}
        for c in df.columns:
            cl = c.lower()
            if cl == "batch":
                rename[c] = "batch"
            elif cl == "time":
                rename[c] = "time"
        df = df.rename(columns=rename)
        meta = df[["batch", "time"]].copy()
        spec = df.drop(columns=["batch", "time"])
    else:
        spec = df.select_dtypes(include=[np.number])
    wn = np.array([float(c) if _is_float(c) else i for i, c in enumerate(spec.columns)])
    return spec.to_numpy(dtype=float), wn, meta


def compute_vip(pls: PLSRegression, X: np.ndarray, y: np.ndarray) -> np.ndarray:
    T = pls.x_scores_
    W = pls.x_weights_
    Q = pls.y_loadings_
    p = X.shape[1]
    H = W.shape[1]
    y = np.asarray(y, dtype=float).reshape(-1, 1)
    ssy = np.zeros(H)
    for h in range(H):
        y_hat_h = T[:, [h]] @ Q[[h], :]
        ssy[h] = float(np.sum(y_hat_h ** 2))
    ssy_sum = float(np.sum(ssy)) + 1e-12
    vip = np.zeros(p)
    for v in range(p):
        acc = 0.0
        for h in range(H):
            wn = float(np.linalg.norm(W[:, h]) + 1e-12)
            acc += ssy[h] * (W[v, h] / wn) ** 2
        vip[v] = np.sqrt(p * acc / ssy_sum)
    return vip


def fit_pls_soft_sensor(
    S: np.ndarray,
    y: np.ndarray,
    wavenumbers: np.ndarray,
    target_cn: str,
    n_components: int = 5,
) -> Optional[RamanSoftSensor]:
    mask = np.isfinite(np.asarray(y, dtype=float).reshape(-1)) & np.isfinite(S).all(axis=1)
    y = np.asarray(y, dtype=float).reshape(-1)[mask]
    S = S[mask]
    if len(y) < 40:
        return None
    X = preprocess_spectra(S)
    n_comp = max(1, min(n_components, X.shape[1] // 10, len(y) // 10))
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=42)
    pls = PLSRegression(n_components=n_comp)
    pls.fit(Xtr, ytr)
    pred = pls.predict(Xte).ravel()
    r2 = float(r2_score(yte, pred))
    try:
        importance = compute_vip(pls, Xtr, ytr)
    except Exception:
        importance = np.abs(pls.x_weights_).sum(axis=1)
    imp = (
        pd.DataFrame({"wavenumber": wavenumbers, "importance": importance, "vip": importance})
        .sort_values("importance", ascending=False)
        .head(30)
        .reset_index(drop=True)
    )
    return RamanSoftSensor(
        target_cn=target_cn,
        target_en=VAR_ENGLISH.get(target_cn, RAMAN_SOFT_TARGETS.get(target_cn, target_cn)),
        n_components=n_comp,
        r2_test=r2,
        wavenumbers=wavenumbers,
        loading_importance=imp,
    )


def synthesize_demo_raman(
    process_df: pd.DataFrame,
    n_bands: int = 120,
    seed: int = 0,
) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """Synthetic spectra mixed from process variables. Pipeline demo only."""
    rng = np.random.default_rng(seed)
    df = process_df.sample(n=min(8000, len(process_df)), random_state=seed).reset_index(drop=True)
    wn = np.linspace(400, 1800, n_bands)
    peaks = {
        "总糖": (560, 40),
        "氨基氮": (980, 35),
        "化学效价": (1450, 50),
        "粘度": (1200, 45),
    }
    S = rng.normal(0, 0.15, size=(len(df), n_bands))
    for col, (center, width) in peaks.items():
        if col not in df.columns:
            continue
        v = df[col].to_numpy(dtype=float)
        v = (v - np.nanmean(v)) / (np.nanstd(v) + 1e-8)
        v = np.nan_to_num(v, nan=0.0)
        shape = np.exp(-0.5 * ((wn - center) / width) ** 2)
        S += np.outer(v, shape)
    meta = pd.DataFrame({"batch": df[VAR_BATCH].values, "time": df[VAR_TIME].values})
    return S, wn, meta


def run_raman_pipeline(
    process_df: pd.DataFrame,
    raman_folder: Path = RAMAN_DIR,
    allow_synthetic_demo: bool = True,
) -> RamanBundle:
    files = find_raman_files(raman_folder)
    if files:
        S, wn, meta = load_raman_matrix(files[0])
        status = "real"
        proc = process_df[
            [VAR_BATCH, VAR_TIME, VAR_TARGET]
            + [c for c in RAMAN_SOFT_TARGETS if c in process_df.columns and c != VAR_TARGET]
        ].copy()
        proc = proc.rename(columns={VAR_BATCH: "batch", VAR_TIME: "time"})
        proc["batch"] = proc["batch"].astype(str)
        if meta is None:
            n = min(len(S), len(proc))
            S_use = S[:n]
            aligned = proc.iloc[:n].reset_index(drop=True)
        else:
            meta = meta.copy()
            meta["batch"] = meta["batch"].astype(str)
            meta["_i"] = np.arange(len(meta))
            aligned = meta.merge(proc.drop_duplicates(["batch", "time"]), on=["batch", "time"], how="inner")
            S_use = S[aligned["_i"].to_numpy()]
            aligned = aligned.drop(columns=["_i"], errors="ignore").reset_index(drop=True)
        out_meta = aligned[["batch", "time"]].copy()
        ymap = {}
        for c in list(RAMAN_SOFT_TARGETS.keys()) + [VAR_TARGET]:
            if c in aligned.columns:
                ymap[c] = np.asarray(aligned[c], dtype=float).reshape(-1)
    elif allow_synthetic_demo:
        S, wn, meta = synthesize_demo_raman(process_df)
        status = "synthetic_demo"
        rng_df = process_df.sample(n=min(8000, len(process_df)), random_state=0).reset_index(drop=True)
        S_use = S
        out_meta = meta.copy()
        ymap = {}
        for c in list(RAMAN_SOFT_TARGETS.keys()) + [VAR_TARGET]:
            if c in rng_df.columns:
                ymap[c] = np.asarray(rng_df[c], dtype=float).reshape(-1)
    else:
        return RamanBundle(sensors={}, soft_predictions=pd.DataFrame(), status="unavailable")

    X = preprocess_spectra(S_use)
    soft_df = out_meta.copy()
    sensors: Dict[str, RamanSoftSensor] = {}
    for t_cn, y in ymap.items():
        y = np.asarray(y, dtype=float).reshape(-1)
        if len(y) != len(S_use):
            n = min(len(y), len(S_use))
            y = y[:n]
            S_fit = S_use[:n]
            X_fit = X[:n]
        else:
            S_fit, X_fit = S_use, X
        sensor = fit_pls_soft_sensor(S_fit, y, wn, t_cn)
        if sensor is None:
            continue
        sensors[t_cn] = sensor
        pls = PLSRegression(n_components=sensor.n_components)
        y_fill = y.copy()
        y_fill[~np.isfinite(y_fill)] = np.nanmedian(y_fill[np.isfinite(y_fill)])
        pls.fit(X_fit, y_fill)
        pred = pls.predict(X_fit).ravel()
        col = f"soft_{sensor.target_en}"
        if len(pred) < len(soft_df):
            soft_df = soft_df.iloc[: len(pred)].copy()
        soft_df[col] = pred

    return RamanBundle(sensors=sensors, soft_predictions=soft_df, status=status)
