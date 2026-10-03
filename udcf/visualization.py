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

"""Clean English visualization — continuous 3-hue colormaps, clipped ranges."""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Patch

from .config import (
    CMAP_DIVERGING,
    CMAP_SEQUENTIAL,
    COLORS,
    FIGURES_DIR,
    HEATMAP_ROBUST_PCT,
    HEATMAP_STD_CLIP,
    VAR_ENGLISH,
)
from .influence_model import InfluenceResult

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "axes.unicode_minus": False,
        "figure.dpi": 140,
        "savefig.dpi": 200,
        "savefig.bbox": "tight",
        "axes.titlesize": 12,
        "axes.labelsize": 10,
        "axes.facecolor": COLORS["bg"],
        "figure.facecolor": "white",
        "axes.edgecolor": "#566573",
        "axes.grid": False,
    }
)


def _en(name: str) -> str:
    return VAR_ENGLISH.get(name, name)


def safe_filename(name: str) -> str:
    name = name.replace("Δ", "d").replace("△", "d").replace("/", "-")
    for ch in r'\/:*?"<>|()':
        name = name.replace(ch, "_")
    name = "_".join(name.split())
    while "__" in name:
        name = name.replace("__", "_")
    return name.strip("_")[:90]


def cmap_diverging():
    return LinearSegmentedColormap.from_list("div3", CMAP_DIVERGING, N=256)


def cmap_sequential():
    return LinearSegmentedColormap.from_list("seq3", CMAP_SEQUENTIAL, N=256)


def robust_sym_limit(data: np.ndarray, clip: float = HEATMAP_STD_CLIP, pct: float = HEATMAP_ROBUST_PCT) -> float:
    finite = np.asarray(data, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return 0.5
    p = float(np.percentile(np.abs(finite), pct))
    mx = float(np.max(np.abs(finite)))
    # Prefer covering most mass, but always leave headroom for the largest |β|
    v = max(p, mx * 0.92)
    return float(np.clip(v, 0.30, clip))


def _annotate_heatmap(ax, data, fmt="{:.2f}", thresh=None):
    """Sparse numeric labels only for strong cells."""
    if thresh is None:
        thresh = robust_sym_limit(data) * 0.55
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            v = data[i, j]
            if not np.isfinite(v) or abs(v) < thresh:
                continue
            ax.text(j, i, fmt.format(v), ha="center", va="center", fontsize=7, color="#1C2833")


def plot_global_influence_network(summary_df: pd.DataFrame, save_path: Optional[Path] = None):
    col = "partial_beta_std" if "partial_beta_std" in summary_df.columns else "partial_beta"
    piv = summary_df.groupby(["driver", "response"])[col].mean().unstack()
    data = piv.values.astype(float)
    vmax = robust_sym_limit(data)
    fig, ax = plt.subplots(figsize=(9.5, 7.2))
    im = ax.imshow(data, cmap=cmap_diverging(), aspect="auto", vmin=-vmax, vmax=vmax)
    ax.set_xticks(range(len(piv.columns)))
    ax.set_xticklabels(piv.columns, rotation=35, ha="right")
    ax.set_yticks(range(len(piv.index)))
    ax.set_yticklabels(piv.index)
    ax.set_xlabel("Response")
    ax.set_ylabel("Driver")
    ax.set_title(f"Global Influence (Time-Averaged Standardized β)\nscale clipped to ±{vmax:.2f}")
    _annotate_heatmap(ax, data)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Standardized partial β")
    fig.tight_layout()
    path = save_path or FIGURES_DIR / "01_global_influence_matrix.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
    return path


def plot_time_correlation_evolution(df, variables, target, time_col, save_path=None):
    hours = sorted(df[time_col].unique())
    mat = np.full((len(variables), len(hours)), np.nan)
    for vi, var in enumerate(variables):
        for hi, h in enumerate(hours):
            sub = df[df[time_col] == h][[var, target]].dropna()
            if len(sub) >= 25:
                mat[vi, hi] = sub[var].corr(sub[target], method="spearman")
    step = max(1, len(hours) // 50)
    idx = list(range(0, len(hours), step))
    mat_s = mat[:, idx]
    hours_s = [hours[i] for i in idx]

    fig, ax = plt.subplots(figsize=(11.5, 4.8))
    im = ax.imshow(mat_s, cmap=cmap_diverging(), aspect="auto", vmin=-1, vmax=1)
    ax.set_yticks(range(len(variables)))
    ax.set_yticklabels([_en(v) for v in variables])
    xt = np.linspace(0, len(hours_s) - 1, min(8, len(hours_s))).astype(int)
    ax.set_xticks(xt)
    ax.set_xticklabels([str(hours_s[i]) for i in xt])
    ax.set_xlabel("Fermentation Time (h)")
    ax.set_ylabel("Process Variable")
    ax.set_title(f"Spearman Correlation vs {_en(target)} over Time")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Spearman ρ  [-1, 1]")
    fig.tight_layout()
    path = save_path or FIGURES_DIR / "02_time_correlation_potency.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
    return path


def plot_propagation_heatmap(result: InfluenceResult, hour: int, save_path: Optional[Path] = None):
    if hour not in result.hours:
        hour = min(result.hours, key=lambda h: abs(h - hour))
    hi = result.hours.index(hour)
    # Use standardized β for readable color scale
    M = result.beta_std[hi]
    vmax = robust_sym_limit(M)
    fig, ax = plt.subplots(figsize=(9.5, 7.2))
    im = ax.imshow(M, cmap=cmap_diverging(), aspect="auto", vmin=-vmax, vmax=vmax)
    ax.set_xticks(range(len(result.responses)))
    ax.set_xticklabels([_en(r) for r in result.responses], rotation=35, ha="right")
    ax.set_yticks(range(len(result.drivers)))
    ax.set_yticklabels([_en(d) for d in result.drivers])
    ax.set_xlabel("Response")
    ax.set_ylabel("Driver")
    ax.set_title(f"Standardized Cross-Parameter Influence at t = {hour} h\n(±{vmax:.2f} clip)")
    _annotate_heatmap(ax, M)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("β_std = β · σ_driver / σ_response")
    fig.tight_layout()
    path = save_path or FIGURES_DIR / f"03_propagation_t{hour}h.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
    return path


def plot_time_influence_heatmap(result: InfluenceResult, response: str, save_path: Optional[Path] = None):
    if response not in result.responses:
        return None
    ri = result.responses.index(response)
    M = result.beta_std[:, :, ri].copy()
    # Mild temporal smoothing for a continuous gradient field
    try:
        from scipy.ndimage import gaussian_filter1d

        for j in range(M.shape[1]):
            col = M[:, j]
            mask = np.isfinite(col)
            if mask.sum() < 5:
                continue
            filled = pd.Series(col).interpolate(limit_direction="both").to_numpy()
            M[:, j] = gaussian_filter1d(filled, sigma=1.2)
    except Exception:
        pass
    step = max(1, len(result.hours) // 40)
    idx = list(range(0, len(result.hours), step))
    M_s = M[idx, :]
    hours_s = [result.hours[i] for i in idx]
    vmax = robust_sym_limit(M_s)

    fig, ax = plt.subplots(figsize=(10, 6.2))
    im = ax.imshow(M_s, cmap=cmap_diverging(), aspect="auto", vmin=-vmax, vmax=vmax, interpolation="bilinear")
    ax.set_xticks(range(len(result.drivers)))
    ax.set_xticklabels([_en(d) for d in result.drivers], rotation=35, ha="right")
    yt = np.linspace(0, len(hours_s) - 1, min(10, len(hours_s))).astype(int)
    ax.set_yticks(yt)
    ax.set_yticklabels([str(hours_s[i]) for i in yt])
    ax.set_xlabel("Driver")
    ax.set_ylabel("Fermentation Time (h)")
    ax.set_title(f"Time-Stratified Standardized Influence on {_en(response)}")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("β_std")
    fig.tight_layout()
    path = save_path or FIGURES_DIR / f"04_influence_on_{safe_filename(_en(response))}.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
    return path


def plot_alarm_bands(validation_df, driver, response, save_path=None):
    sub = validation_df[(validation_df["driver"] == driver) & (validation_df["response"] == response)].copy()
    if sub.empty:
        sub = validation_df[validation_df["response"] == response].copy()
        if sub.empty:
            return None
        driver = str(sub["driver"].iloc[0])
    g = (
        sub.groupby("hour")
        .agg(observed=("observed", "median"), alarm_low=("alarm_low", "median"), alarm_high=("alarm_high", "median"))
        .reset_index()
        .sort_values("hour")
    )
    coverage = float(sub["in_band"].mean()) * 100
    fig, ax = plt.subplots(figsize=(10.5, 4.6))
    ax.fill_between(g["hour"], g["alarm_low"], g["alarm_high"], color=COLORS["band"], alpha=COLORS["band_fill_alpha"], label="Alarm band")
    ax.plot(g["hour"], g["alarm_low"], color=COLORS["alarm_low"], lw=1.6, label="Low alarm")
    ax.plot(g["hour"], g["alarm_high"], color=COLORS["alarm_high"], lw=1.6, label="High alarm")
    ax.plot(g["hour"], g["observed"], color=COLORS["primary"], lw=2.0, label="Observed median")
    ax.set_xlabel("Fermentation Time (h)")
    ax.set_ylabel(response)
    ax.set_title(f"Alarm Validation: {driver} → {response}  (coverage {coverage:.1f}%)")
    ax.legend(loc="best", framealpha=0.95)
    ax.set_facecolor(COLORS["bg"])
    fig.tight_layout()
    path = save_path or FIGURES_DIR / f"05_alarm_{safe_filename(driver)}_to_{safe_filename(response)}.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
    return path


def plot_pairwise_coevolution(XX, YY, Z, xlabel, ylabel, zlabel, hour, save_path=None):
    if XX.size == 0:
        return None
    finite = Z[np.isfinite(Z)]
    if len(finite) == 0:
        return None
    vmin, vmax = np.percentile(finite, [5, 95])
    if vmin >= vmax:
        vmin, vmax = float(finite.min()), float(finite.max())
    fig, ax = plt.subplots(figsize=(7.8, 6.5))
    extent = [float(XX.min()), float(XX.max()), float(YY.min()), float(YY.max())]
    im = ax.imshow(
        Z,
        origin="lower",
        extent=extent,
        aspect="auto",
        cmap=cmap_sequential(),
        vmin=vmin,
        vmax=vmax,
        interpolation="bicubic",
    )
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(f"Co-evolution at t = {hour} h\n{zlabel} (smooth continuous gradient)")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(zlabel)
    fig.tight_layout()
    path = save_path or FIGURES_DIR / f"08_coevolution_t{hour}h.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
    return path


def plot_dynamic_control_scenario(control_df, save_path=None):
    df = control_df.copy().sort_values("expected_change")
    colors = [
        COLORS["alarm_high"] if s == "HIGH ALARM" else COLORS["alarm_low"] if s == "LOW ALARM" else COLORS["accent"]
        for s in df["status"]
    ]
    fig, ax = plt.subplots(figsize=(9.5, 5.5))
    ax.barh(df["response"], df["expected_change"], color=colors, edgecolor="white", height=0.7)
    ax.axvline(0, color=COLORS["neutral"], lw=1.0)
    d0 = df.iloc[0]
    ax.set_xlabel("Expected Change (native units)")
    ax.set_title(f"Snapshot at t = {int(d0['hour'])} h: Δ{d0['driver']} = {d0['driver_delta']:+.3f}")
    ax.legend(
        handles=[
            Patch(facecolor=COLORS["accent"], label="Within band"),
            Patch(facecolor=COLORS["alarm_high"], label="High alarm"),
            Patch(facecolor=COLORS["alarm_low"], label="Low alarm"),
        ],
        loc="lower right",
    )
    fig.tight_layout()
    path = save_path or FIGURES_DIR / "06_dynamic_control_scenario.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
    return path


def plot_full_horizon_trajectories(traj: pd.DataFrame, save_path: Optional[Path] = None, max_panels: int = 8):
    """
    Full fermentation timeline: baseline vs controlled path for each response
    after a sustained driver adjustment.
    """
    if traj.empty:
        return None
    driver = str(traj["driver"].iloc[0])
    delta = float(traj["driver_delta"].iloc[0])
    responses = list(traj.groupby("response")["expected_change"].apply(lambda s: s.abs().mean()).sort_values(ascending=False).index)
    responses = responses[:max_panels]
    n = len(responses)
    ncols = 2
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(12.5, 3.2 * nrows), sharex=True)
    axes = np.atleast_1d(axes).ravel()

    for ax, resp in zip(axes, responses):
        g = traj[traj["response"] == resp].sort_values("hour")
        ax.fill_between(
            g["hour"],
            g["alarm_low"],
            g["alarm_high"],
            color=COLORS["band"],
            alpha=COLORS["band_fill_alpha"],
            label="Alarm band",
        )
        ax.plot(g["hour"], g["baseline_median"], color=COLORS["neutral"], lw=1.6, ls="--", label="Baseline median")
        ax.plot(g["hour"], g["expected_value"], color=COLORS["primary"], lw=2.1, label="After control")
        ax.set_ylabel(resp, fontsize=9)
        ax.grid(True, alpha=0.2)
        ax.set_facecolor(COLORS["bg"])

    for ax in axes[n:]:
        ax.set_visible(False)
    axes[min(n - 1, len(axes) - 1)].set_xlabel("Fermentation Time (h)")
    axes[0].legend(loc="best", fontsize=8, framealpha=0.92)
    fig.suptitle(
        f"Full-Horizon Dynamic Regulation (0–{int(traj['hour'].max())} h)\n"
        f"Sustained adjustment Δ{driver} = {delta:+.3f} → all other variables over time",
        fontsize=12,
        y=1.01,
    )
    fig.tight_layout()
    path = save_path or FIGURES_DIR / "08_full_horizon_trajectories.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
    return path


def plot_horizon_delta_heatmap(traj: pd.DataFrame, save_path: Optional[Path] = None):
    """
    Time × response heatmap of expected Δ under sustained driver control.
    Continuous gradient over the whole batch timeline (not a single hour slice).
    """
    if traj.empty:
        return None
    mat = traj.pivot_table(index="response", columns="hour", values="expected_change", aggfunc="mean")
    mat = mat.reindex(sorted(mat.columns), axis=1)
    mat_i = mat.interpolate(axis=1, limit_direction="both")
    data = mat_i.to_numpy(dtype=float)
    # Row-wise robust scaling so variables with different units are all visible
    scaled = np.full_like(data, np.nan)
    for i in range(data.shape[0]):
        row = data[i]
        finite = row[np.isfinite(row)]
        if finite.size == 0:
            continue
        s = float(np.percentile(np.abs(finite), 90))
        s = s if s > 1e-12 else 1.0
        scaled[i] = row / s
    data = scaled
    finite = data[np.isfinite(data)]
    if finite.size == 0:
        return None
    vmax = float(np.clip(np.percentile(np.abs(finite), 95), 0.5, 2.5))

    driver = str(traj["driver"].iloc[0])
    delta = float(traj["driver_delta"].iloc[0])
    hours = list(mat_i.columns)

    fig, ax = plt.subplots(figsize=(12.5, 5.8))
    im = ax.imshow(
        data,
        aspect="auto",
        cmap=cmap_diverging(),
        vmin=-vmax,
        vmax=vmax,
        interpolation="bilinear",
        origin="upper",
    )
    ax.set_yticks(range(len(mat_i.index)))
    ax.set_yticklabels(list(mat_i.index))
    xt = np.linspace(0, len(hours) - 1, min(10, len(hours))).astype(int)
    ax.set_xticks(xt)
    ax.set_xticklabels([str(hours[i]) for i in xt])
    ax.set_xlabel("Fermentation Time (h)")
    ax.set_ylabel("Response Variable")
    ax.set_title(
        f"Time-Series Propagation Map (full batch, 0–{int(max(hours))} h)\n"
        f"Sustained Δ{driver} = {delta:+.3f}; each row scaled by its own 90% |Δ|"
    )
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Row-scaled expected Δ (unit-free)")
    fig.tight_layout()
    path = save_path or FIGURES_DIR / "08_horizon_delta_heatmap.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
    return path


def plot_horizon_delta_lines(traj: pd.DataFrame, save_path: Optional[Path] = None, top_n: int = 6):
    """Line chart: expected Δ of top responses vs fermentation time."""
    if traj.empty:
        return None
    rank = traj.groupby("response")["expected_change"].apply(lambda s: s.abs().mean()).sort_values(ascending=False)
    keep = list(rank.head(top_n).index)
    driver = str(traj["driver"].iloc[0])
    delta = float(traj["driver_delta"].iloc[0])

    palette = ["#1B4F72", "#D35400", "#1E8449", "#8E44AD", "#C0392B", "#17A589", "#F39C12", "#2C3E50"]
    fig, ax = plt.subplots(figsize=(11.5, 5.2))
    for i, resp in enumerate(keep):
        g = traj[traj["response"] == resp].sort_values("hour")
        ax.plot(g["hour"], g["expected_change"], lw=2.0, color=palette[i % len(palette)], label=resp)
    ax.axhline(0, color=COLORS["neutral"], lw=1.0)
    ax.set_xlabel("Fermentation Time (h)")
    ax.set_ylabel("Expected Δ (native units)")
    ax.set_title(
        f"Continuous Response Trajectories under Control\n"
        f"Δ{driver} = {delta:+.3f} held across 0–{int(traj['hour'].max())} h"
    )
    ax.legend(loc="best", fontsize=8, framealpha=0.92)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    path = save_path or FIGURES_DIR / "08_horizon_delta_lines.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
    return path


def plot_validation_summary(coverage, n_batches, n_records, save_path=None):
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.4))
    ax = axes[0]
    vals = [coverage * 100, (1 - coverage) * 100]
    bars = ax.bar(["In Band", "Out of Band"], vals, color=[COLORS["accent"], COLORS["alarm_high"]], width=0.55)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 1.5, f"{v:.1f}%", ha="center", fontweight="bold")
    ax.set_ylim(0, 115)
    ax.set_ylabel("Percentage (%)")
    ax.set_title("Alarm Band Coverage")

    ax = axes[1]
    bars = ax.bar(["Batches", "Observations"], [n_batches, n_records], color=[COLORS["neutral"], COLORS["secondary"]], width=0.55)
    for b, v in zip(bars, [n_batches, n_records]):
        ax.text(b.get_x() + b.get_width() / 2, v * 1.02, f"{v:,}", ha="center", fontweight="bold", fontsize=9)
    ax.set_title("Held-out Validation Size")
    fig.suptitle("Hierarchical Framework — Erythromycin Validation", fontsize=13, y=1.02)
    fig.tight_layout()
    path = save_path or FIGURES_DIR / "07_validation_summary.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
    return path


def plot_block_importance(block_results: dict, save_path: Optional[Path] = None):
    """Bar chart of each small block's test R² + top coefficients."""
    ids = list(block_results.keys())
    if not ids:
        return None
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    ax = axes[0]
    r2s = [block_results[i].r2_test for i in ids]
    labels = [block_results[i].block_label for i in ids]
    colors = [COLORS["accent"] if v >= 0 else COLORS["alarm_high"] for v in r2s]
    ax.barh(labels, r2s, color=colors, height=0.65)
    ax.axvline(0, color=COLORS["neutral"], lw=0.8)
    ax.set_xlabel("Held-out R²")
    ax.set_title("Step 2 — Small Block Models")
    lo = min(0.0, min(r2s) - 0.02)
    hi = max(r2s) + 0.05
    ax.set_xlim(lo, hi)

    ax = axes[1]
    rows = []
    for bid, br in block_results.items():
        # Prefer block-core coefficients for interpretation
        imp = br.feature_importance
        if "role" in imp.columns:
            core = imp[imp["role"] == "block"]
            use = core if len(core) else imp
        else:
            use = imp
        for _, r in use.head(3).iterrows():
            rows.append({"block": br.block_label, "feature": r["feature"], "coef": r["standardized_coef"]})
    top = pd.DataFrame(rows)
    if not top.empty:
        y = [f"{a} | {b}" for a, b in zip(top["block"], top["feature"])]
        colors = [COLORS["alarm_high"] if v < 0 else COLORS["accent"] for v in top["coef"]]
        ax.barh(y, top["coef"], color=colors, height=0.7)
    ax.axvline(0, color=COLORS["neutral"], lw=0.8)
    ax.set_xlabel("Standardized coefficient")
    ax.set_title("Top Interpretable Coefficients")
    fig.tight_layout()
    path = save_path or FIGURES_DIR / "09_block_models.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
    return path


def plot_meta_weights(meta_result, save_path: Optional[Path] = None):
    tab = meta_result.weight_table.copy()
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))

    ax = axes[0]
    colors = [COLORS["alarm_high"] if v < 0 else COLORS["primary"] for v in tab["standardized_weight"]]
    # Show signed abs_share for readability (weights themselves can be large)
    signed_share = tab["abs_share"] * np.sign(tab["standardized_weight"])
    ax.barh(tab["source"], signed_share, color=colors, height=0.7)
    ax.axvline(0, color=COLORS["neutral"], lw=0.8)
    ax.set_xlabel("Signed contribution share")
    ax.set_title(f"Meta fusion shares  (test R2 = {meta_result.r2_test:.3f})")

    ax = axes[1]
    ax.barh(tab["source"], tab["abs_share"], color=COLORS["secondary"], height=0.7)
    ax.set_xlabel("Absolute share")
    ax.set_xlim(0, max(0.05, float(tab["abs_share"].max()) * 1.15))
    ax.set_title("How much each block / soft sensor matters")
    fig.tight_layout()
    path = save_path or FIGURES_DIR / "10_meta_fusion_weights.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
    return path


def plot_raman_loadings(raman_bundle, save_path: Optional[Path] = None):
    if not raman_bundle.sensors:
        return None
    n = len(raman_bundle.sensors)
    fig, axes = plt.subplots(n, 1, figsize=(9.5, 2.6 * n), sharex=False)
    if n == 1:
        axes = [axes]
    for ax, (k, sensor) in zip(axes, raman_bundle.sensors.items()):
        imp = sensor.loading_importance.head(20)
        ax.bar(range(len(imp)), imp["importance"], color=COLORS["secondary"])
        ax.set_xticks(range(len(imp)))
        ax.set_xticklabels([f"{w:.0f}" for w in imp["wavenumber"]], rotation=45, ha="right", fontsize=7)
        ax.set_ylabel("VIP")
        ax.set_title(f"Raman PLS VIP → {sensor.target_en}  (R²={sensor.r2_test:.3f}) [{raman_bundle.status}]")
    axes[-1].set_xlabel("Wavenumber (cm⁻¹) — top bands")
    fig.tight_layout()
    path = save_path or FIGURES_DIR / "11_raman_soft_sensors.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
    return path


def plot_advantage_prediction(metrics: pd.DataFrame, save_path: Optional[Path] = None):
    """Proposed hierarchical model vs flat / single-block baselines."""
    df = metrics.dropna(subset=["r2_test"]).copy().sort_values("r2_test")
    colors = [
        COLORS.get("advantage", "#0E6655") if f == "Proposed" else COLORS["neutral"]
        for f in df["family"]
    ]
    fig, axes = plt.subplots(1, 2, figsize=(12.2, 4.8))

    ax = axes[0]
    ax.barh(df["model"], df["r2_test"], color=colors, height=0.68, edgecolor="white")
    for y, v in enumerate(df["r2_test"]):
        ax.text(v + 0.004, y, f"{v:.3f}", va="center", fontsize=8)
    ax.set_xlabel("Held-out R²")
    ax.set_title("Prediction Advantage")
    ax.set_xlim(0, max(0.05, float(df["r2_test"].max()) * 1.18))

    ax = axes[1]
    # Capability radar-like bars: interpretability / dynamics / safety
    prop = df[df["family"] == "Proposed"]
    base = df[df["family"] == "Baseline"]
    cats = ["Test R²", "Interpretability\nlevels", "Time-varying\ninfluence", "Alarm-aware\ncontrol"]
    if not prop.empty:
        p = prop.iloc[0]
        ours = [
            float(p["r2_test"]),
            float(p["interpretable_levels"]) / 3.0,
            1.0 if p["time_varying"] else 0.0,
            1.0 if p["alarm_aware"] else 0.0,
        ]
    else:
        ours = [0, 0, 0, 0]
    if not base.empty:
        b = base.sort_values("r2_test", ascending=False).iloc[0]
        theirs = [
            float(b["r2_test"]) if np.isfinite(b["r2_test"]) else 0.0,
            float(b["interpretable_levels"]) / 3.0,
            1.0 if b["time_varying"] else 0.0,
            1.0 if b["alarm_aware"] else 0.0,
        ]
    else:
        theirs = [0, 0, 0, 0]
    x = np.arange(len(cats))
    w = 0.36
    ax.bar(x - w / 2, ours, w, color=COLORS.get("advantage", "#0E6655"), label="Proposed framework")
    ax.bar(x + w / 2, theirs, w, color=COLORS["neutral"], label="Best flat baseline")
    ax.set_xticks(x)
    ax.set_xticklabels(cats, fontsize=8)
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("Normalized score")
    ax.set_title("Capability Profile")
    ax.legend(loc="upper right", fontsize=8, framealpha=0.92)

    fig.suptitle("Why the Hierarchical Time-Varying Framework Wins", fontsize=12, y=1.02)
    fig.tight_layout()
    path = save_path or FIGURES_DIR / "12_advantage_prediction.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
    return path


def plot_advantage_time_varying(tv: pd.DataFrame, save_path: Optional[Path] = None):
    """Static β vs time-local β_std(t) — core dynamic advantage."""
    if tv is None or tv.empty:
        return None
    fig, axes = plt.subplots(1, 2, figsize=(12.2, 4.6))

    ax = axes[0]
    ax.plot(tv["hour"], tv["beta_std_time_local"], color=COLORS["primary"], lw=2.1, label="Time-local β_std(t)")
    ax.axhline(float(tv["beta_std_static"].iloc[0]), color=COLORS["baseline"], lw=1.8, ls="--", label="Static β_std")
    ax.fill_between(
        tv["hour"],
        tv["beta_std_static"],
        tv["beta_std_time_local"],
        color=COLORS["secondary"],
        alpha=0.15,
        label="Stage-dependent gap",
    )
    ax.set_xlabel("Fermentation Time (h)")
    ax.set_ylabel("Standardized influence β_std")
    ax.set_title("Time Heterogeneity of Influence")
    ax.legend(loc="best", fontsize=8, framealpha=0.92)
    ax.grid(True, alpha=0.22)

    ax = axes[1]
    ax.fill_between(tv["hour"], 0, tv["abs_deviation"], color=COLORS.get("advantage", "#0E6655"), alpha=0.35)
    ax.plot(tv["hour"], tv["abs_deviation"], color=COLORS.get("advantage", "#0E6655"), lw=1.8)
    ax.set_xlabel("Fermentation Time (h)")
    ax.set_ylabel("|β_std(t) − β_static|")
    ax.set_title("Cost of Ignoring Time Structure")
    ax.grid(True, alpha=0.22)

    fig.suptitle("Advantage: Time-Local Partial Regression Captures Stage Effects", fontsize=12, y=1.02)
    fig.tight_layout()
    path = save_path or FIGURES_DIR / "12_advantage_time_varying.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
    return path


def plot_advantage_safety(safety: pd.DataFrame, save_path: Optional[Path] = None):
    """Dynamic alarm bands vs naive global bands under control scenarios."""
    if safety is None or safety.empty:
        return None
    df = safety.sort_values("framework_in_band")
    fig, ax = plt.subplots(figsize=(10.5, 5.0))
    y = np.arange(len(df))
    h = 0.35
    ax.barh(y + h / 2, df["framework_in_band"] * 100, h, color=COLORS.get("advantage", "#0E6655"), label="Dynamic bands (ours)")
    ax.barh(y - h / 2, df["naive_global_in_band"] * 100, h, color=COLORS["neutral"], label="Naive global bands")
    ax.set_yticks(y)
    ax.set_yticklabels(df["response"])
    ax.set_xlabel("Controlled trajectory in-band (%)")
    ax.set_xlim(0, 105)
    ax.set_title("Advantage: Dynamic Safety Constraints Stay Informative")
    ax.legend(loc="lower right", framealpha=0.92)
    fig.tight_layout()
    path = save_path or FIGURES_DIR / "12_advantage_safety.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
    return path


def plot_advantage_hierarchy(meta_table: pd.DataFrame, block_results: dict, save_path: Optional[Path] = None):
    """Three-level interpretability story unique to the hierarchical design."""
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.5))

    # Level 1 — block R²
    ax = axes[0]
    labels = [br.block_label for br in block_results.values()]
    r2s = [br.r2_test for br in block_results.values()]
    order = np.argsort(r2s)
    ax.barh([labels[i] for i in order], [r2s[i] for i in order], color=COLORS["primary"], height=0.65)
    ax.set_xlabel("Block R²")
    ax.set_title("Level 1 · Process Blocks")

    # Level 2 — meta shares
    ax = axes[1]
    tab = meta_table.copy().sort_values("abs_share")
    colors = [COLORS["alarm_high"] if v < 0 else COLORS["accent"] for v in tab["standardized_weight"]]
    signed = tab["abs_share"] * np.sign(tab["standardized_weight"])
    ax.barh(tab["source"], signed, color=colors, height=0.65)
    ax.axvline(0, color=COLORS["neutral"], lw=0.8)
    ax.set_xlabel("Signed meta share")
    ax.set_title("Level 2 · System Fusion")

    # Level 3 — conceptual capabilities
    ax = axes[2]
    items = [
        ("Screening → Blocks", 1.0),
        ("Blocks → Meta", 1.0),
        ("β_raw(t) Dynamics", 1.0),
        ("Full-horizon Control", 1.0),
        ("Safety Constraints", 1.0),
        ("Flat Ridge Only", 0.25),
    ]
    names = [a for a, _ in items]
    vals = [b for _, b in items]
    cols = [COLORS.get("advantage", "#0E6655") if v >= 0.9 else COLORS["baseline"] for v in vals]
    ax.barh(names[::-1], vals[::-1], color=cols[::-1], height=0.65)
    ax.set_xlim(0, 1.15)
    ax.set_xlabel("Coverage of modeling needs")
    ax.set_title("Level 3 · End-to-End Scope")

    fig.suptitle("Advantage: Multi-Level Interpretability (Variable → Block → System → Control)", fontsize=11, y=1.03)
    fig.tight_layout()
    path = save_path or FIGURES_DIR / "12_advantage_hierarchy.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
    return path
