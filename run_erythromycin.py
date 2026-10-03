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

"""Run the erythromycin validation pipeline (steps 1-7).

Usage (from the repository root):

    python run_erythromycin.py
    python run_erythromycin.py --force-reload
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from udcf.advantage import build_advantage_bundle
from udcf.blocks import block_predictions, fit_all_blocks
from udcf.config import FIGURES_DIR, RESULTS_DIR, STEPS_DIR, TABLES_DIR, VAR_BATCH, VAR_TARGET, VAR_TIME
from udcf.data_loader import load_or_cache
from udcf.influence_model import (
    build_dynamic_control_table,
    build_full_horizon_trajectory,
    compute_influence_model,
    english,
    horizon_delta_matrix,
    select_top_drivers,
    summarize_influence,
    validate_alarm_coverage,
)
from udcf.meta_model import fit_meta_model
from udcf.raman import run_raman_pipeline
from udcf.visualization import (
    plot_advantage_hierarchy,
    plot_advantage_prediction,
    plot_advantage_safety,
    plot_advantage_time_varying,
    plot_alarm_bands,
    plot_block_importance,
    plot_dynamic_control_scenario,
    plot_full_horizon_trajectories,
    plot_global_influence_network,
    plot_horizon_delta_heatmap,
    plot_horizon_delta_lines,
    plot_meta_weights,
    plot_propagation_heatmap,
    plot_raman_loadings,
    plot_time_correlation_evolution,
    plot_time_influence_heatmap,
    plot_validation_summary,
    safe_filename,
)


def _step_dir(n: int, name: str) -> Path:
    path = STEPS_DIR / f"step{n}_{name}"
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_step_readme(folder: Path, title: str, bullets: list[str]) -> None:
    lines = [f"# {title}", "", *[f"- {b}" for b in bullets], ""]
    (folder / "README.md").write_text("\n".join(lines), encoding="utf-8")


def write_validation_report(payload: dict) -> Path:
    lines = [
        "# Erythromycin Validation Report",
        "",
        "## Framework",
        "",
        "**Hierarchical Time-Varying Statistical Control Framework**",
        "",
        "Spearman screening -> disjoint block Ridge -> Meta Ridge fusion;",
        "parallel time-local partial regression -> full-horizon propagation -> dynamic safety bands.",
        "",
        "---",
        "",
        "## Metrics",
        "",
        "| Metric | Value |",
        "|--------|-------|",
        f"| Alarm coverage | **{payload['coverage']*100:.1f}%** |",
        f"| Meta-model test R2 | **{payload['meta_r2']:.3f}** |",
        f"| Train / test batches | {payload['n_train']} / {payload['n_test']} |",
        f"| Block models | {payload['n_blocks']} |",
        "",
        "---",
        "",
        "## Block model test R2",
        "",
        "| Block | R2 test | MAE test |",
        "|-------|---------|----------|",
    ]
    for row in payload["block_rows"]:
        lines.append(f"| {row['label']} | {row['r2']:.3f} | {row['mae']:.2f} |")

    lines += [
        "",
        "---",
        "",
        "## Meta fusion weights",
        "",
        "| Source | Weight | Abs share |",
        "|--------|--------|-----------|",
    ]
    for _, r in payload["meta_table"].iterrows():
        lines.append(f"| {r['source']} | {r['standardized_weight']:.3f} | {r['abs_share']:.1%} |")

    lines += ["", "---", "", "## Model advantage (vs baselines)", ""]
    for b in payload.get("advantage_bullets", []):
        lines.append(f"- {b}")

    lines += [
        "",
        "Figures: `12_advantage_prediction.png`, `12_advantage_time_varying.png`,",
        "`12_advantage_safety.png`, `12_advantage_hierarchy.png`.",
        "",
        "---",
        "",
        "## Dynamic control (full horizon)",
        "",
        payload.get("scenario_text", ""),
        "",
        "Primary tables: `full_horizon_trajectory.csv`, `horizon_delta_matrix.csv`.",
        "Primary figures: `08_full_horizon_trajectories.png`, `08_horizon_delta_heatmap.png`, `08_horizon_delta_lines.png`.",
        "",
    ]
    path = RESULTS_DIR / "validation_report.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description="Erythromycin hierarchical control validation")
    parser.add_argument("--force-reload", action="store_true", help="Rebuild the hourly cache from Excel files")
    parser.add_argument("--max-hour", type=int, default=190, help="Max fermentation hour")
    args = parser.parse_args()

    for folder in (FIGURES_DIR, TABLES_DIR, RESULTS_DIR, STEPS_DIR):
        folder.mkdir(parents=True, exist_ok=True)
    for png in FIGURES_DIR.glob("*.png"):
        png.unlink()

    print("=" * 64)
    print("Hierarchical Time-Varying Framework — Erythromycin")
    print("=" * 64)

    df = load_or_cache(force_reload=args.force_reload)
    df = df[df[VAR_TIME] <= args.max_hour].copy()
    batches = df[VAR_BATCH].unique()
    train_b, test_b = train_test_split(batches, test_size=0.2, random_state=42)
    train_df = df[df[VAR_BATCH].isin(train_b)].copy()
    test_df = df[df[VAR_BATCH].isin(test_b)].copy()
    print(f"Train: {len(train_b)} | Test: {len(test_b)} | Rows: {len(df):,}")

    s1 = _step_dir(1, "correlation")
    drivers = select_top_drivers(train_df)
    corr_rows = []
    for c in drivers:
        rho = train_df[c].corr(train_df[VAR_TARGET], method="spearman")
        corr_rows.append({"variable": english(c), "variable_cn": c, "spearman_vs_potency": rho})
    corr_df = pd.DataFrame(corr_rows).sort_values("spearman_vs_potency", key=lambda s: s.abs(), ascending=False)
    corr_df.to_csv(s1 / "top_drivers.csv", index=False, encoding="utf-8-sig")
    corr_df.to_csv(TABLES_DIR / "step1_top_drivers.csv", index=False, encoding="utf-8-sig")
    plot_time_correlation_evolution(df, drivers, VAR_TARGET, VAR_TIME, FIGURES_DIR / "02_time_correlation_potency.png")
    write_step_readme(
        s1,
        "Step 1 — Spearman Screening",
        [
            "Rank process variables by |Spearman rho| vs hourly potency net change.",
            "Screening discovers candidates; it does not claim causality.",
            "Output: top_drivers.csv + time-correlation heatmap (figure 02).",
        ],
    )
    print(f"[Step 1] Drivers: {[english(d) for d in drivers]}")

    s2 = _step_dir(2, "block_models")
    blocks = fit_all_blocks(train_df, test_df)
    block_rows = []
    for bid, br in blocks.items():
        br.feature_importance.to_csv(s2 / f"coefs_{bid}.csv", index=False, encoding="utf-8-sig")
        block_rows.append(
            {
                "block_id": bid,
                "label": br.block_label,
                "r2_train": br.r2_train,
                "r2_test": br.r2_test,
                "mae_test": br.mae_test,
                "n_features": len(br.features),
            }
        )
        print(f"  Block {br.block_label}: R2_test={br.r2_test:.3f}")
    pd.DataFrame(block_rows).to_csv(s2 / "block_summary.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(block_rows).to_csv(TABLES_DIR / "step2_block_summary.csv", index=False, encoding="utf-8-sig")
    plot_block_importance(blocks, FIGURES_DIR / "09_block_models.png")
    write_step_readme(
        s2,
        "Step 2 — Disjoint Block Ridge",
        [
            "Oxygenation / Carbon Feed / Nitrogen / Rheology / Environment are disjoint.",
            "Each block is Ridge on train-only standardized features plus a time term.",
            "Coefficients are signed, standardized, and directly interpretable.",
        ],
    )

    s3 = _step_dir(3, "influence_alarms")
    print("[Step 3] Time-local partial regression...")
    result = compute_influence_model(train_df, drivers=drivers, hour_range=(0, args.max_hour))
    coverage, val_df = validate_alarm_coverage(result, df, test_batches=list(test_b))
    print(f"  Alarm coverage: {coverage*100:.1f}%")
    summary = summarize_influence(result)
    summary.to_csv(s3 / "influence_coefficients.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(TABLES_DIR / "full_influence_coefficients.csv", index=False, encoding="utf-8-sig")
    top = summary.dropna(subset=["propagation_1sigma"]).assign(abs_p=lambda d: d["propagation_1sigma"].abs())
    top = top.sort_values("abs_p", ascending=False).drop(columns=["abs_p"])
    top.head(200).to_csv(TABLES_DIR / "influence_summary.csv", index=False, encoding="utf-8-sig")
    val_df.sample(n=min(150_000, len(val_df)), random_state=42).to_csv(
        TABLES_DIR / "alarm_validation_records.csv", index=False, encoding="utf-8-sig"
    )
    plot_global_influence_network(summary, FIGURES_DIR / "01_global_influence_matrix.png")
    for h in [30, 45, 60, 90, 135]:
        if result.hours[0] <= h <= result.hours[-1]:
            plot_propagation_heatmap(result, h)
    plot_time_influence_heatmap(result, VAR_TARGET)
    for _, row in top[["driver", "response"]].drop_duplicates().head(4).iterrows():
        plot_alarm_bands(val_df, row["driver"], row["response"])
    write_step_readme(
        s3,
        "Step 3 — Time-Local Influence and Alarms",
        [
            "beta_std = beta_raw * sigma_driver / sigma_response (heatmap quantity).",
            "Propagation / control use beta_raw only.",
            "Alarm bands: residual Q01-Q99 then half-width x 0.5.",
        ],
    )

    s4 = _step_dir(4, "dynamic_control")
    pot_i = result.responses.index(VAR_TARGET)
    score = np.nanmean(np.abs(result.beta_std[:, :, pot_i]), axis=0)
    score[~np.isfinite(score)] = -1
    scenario_driver = result.drivers[int(np.argmax(score))]
    delta = result.driver_sigma.get(scenario_driver, 1.0)

    traj = build_full_horizon_trajectory(
        result, df, scenario_driver, delta_driver=delta, hour_max=args.max_hour
    )
    traj.to_csv(s4 / "full_horizon_trajectory.csv", index=False, encoding="utf-8-sig")
    traj.to_csv(TABLES_DIR / "full_horizon_trajectory.csv", index=False, encoding="utf-8-sig")
    wide = horizon_delta_matrix(traj)
    wide.to_csv(s4 / "horizon_delta_matrix.csv", encoding="utf-8-sig")
    wide.to_csv(TABLES_DIR / "horizon_delta_matrix.csv", encoding="utf-8-sig")

    mid_hour = 90 if 90 in result.hours else result.hours[len(result.hours) // 2]
    scenario = build_dynamic_control_table(result, df, scenario_driver, mid_hour, delta_driver=delta)
    scenario.to_csv(s4 / "scenario_snapshot_t90.csv", index=False, encoding="utf-8-sig")
    scenario.to_csv(TABLES_DIR / "dynamic_control_scenario.csv", index=False, encoding="utf-8-sig")
    plot_dynamic_control_scenario(scenario, FIGURES_DIR / "06_dynamic_control_snapshot_t90.png")
    plot_full_horizon_trajectories(traj, FIGURES_DIR / "08_full_horizon_trajectories.png")
    plot_horizon_delta_heatmap(traj, FIGURES_DIR / "08_horizon_delta_heatmap.png")
    plot_horizon_delta_lines(traj, FIGURES_DIR / "08_horizon_delta_lines.png")
    write_step_readme(
        s4,
        "Step 4 — Full-Horizon Propagation",
        [
            "X_j_ctrl(t) = median_j(t) + beta_raw(t) * delta X_i for t in [0, T].",
            "Sustained operator move across the whole batch (not a single-hour slice).",
            "Tables: full_horizon_trajectory.csv, horizon_delta_matrix.csv.",
        ],
    )

    s5 = _step_dir(5, "meta_fusion")
    print("[Step 5] Meta-model fusion...")
    train_pred = block_predictions(train_df, blocks)
    test_pred = block_predictions(test_df, blocks)

    raman = run_raman_pipeline(df, allow_synthetic_demo=False)
    if raman.status != "unavailable" and not raman.soft_predictions.empty:
        s6 = _step_dir(6, "raman")
        soft = raman.soft_predictions.copy()
        soft_cols = [c for c in soft.columns if c.startswith("soft_")]
        soft.to_csv(s6 / "soft_predictions_sample.csv", index=False, encoding="utf-8-sig")
        train_m = train_pred.merge(
            soft.rename(columns={"batch": VAR_BATCH, "time": VAR_TIME}),
            on=[VAR_BATCH, VAR_TIME],
            how="left",
        )
        test_m = test_pred.merge(
            soft.rename(columns={"batch": VAR_BATCH, "time": VAR_TIME}),
            on=[VAR_BATCH, VAR_TIME],
            how="left",
        )
        for c in soft_cols:
            train_m[c] = train_m[c].fillna(train_m[c].median())
            test_m[c] = test_m[c].fillna(test_m[c].median())
            train_pred[c] = train_m[c]
            test_pred[c] = test_m[c]
        for sensor in raman.sensors.values():
            sensor.loading_importance.to_csv(
                s6 / f"loadings_{safe_filename(sensor.target_en)}.csv",
                index=False,
                encoding="utf-8-sig",
            )
        plot_raman_loadings(raman)
        with open(s6 / "raman_status.json", "w", encoding="utf-8") as f:
            json.dump(
                {
                    "status": raman.status,
                    "n_sensors": len(raman.sensors),
                    "sensors": {
                        k: {"en": v.target_en, "r2_test": v.r2_test, "n_comp": v.n_components}
                        for k, v in raman.sensors.items()
                    },
                },
                f,
                indent=2,
                ensure_ascii=False,
            )

    soft_cols_present = [c for c in train_pred.columns if c.startswith("soft_")]
    if soft_cols_present:
        meta = fit_meta_model(
            train_pred,
            test_pred,
            pred_cols=[c for c in train_pred.columns if c.startswith("pred_")],
            extra_train=train_pred[soft_cols_present],
            extra_test=test_pred[soft_cols_present],
        )
    else:
        meta = fit_meta_model(train_pred, test_pred)
    meta.weight_table.to_csv(s5 / "meta_weights.csv", index=False, encoding="utf-8-sig")
    meta.weight_table.to_csv(TABLES_DIR / "step5_meta_weights.csv", index=False, encoding="utf-8-sig")
    plot_meta_weights(meta)
    print(f"  Meta R2_test={meta.r2_test:.3f}")
    write_step_readme(
        s5,
        "Step 5 — Meta Ridge Fusion",
        [
            "Large model = Ridge on standardized block predictions.",
            "Weights are readable system-level fusion shares.",
            f"Test R2 = {meta.r2_test:.3f}, MAE = {meta.mae_test:.2f}.",
        ],
    )
    plot_validation_summary(coverage, len(test_b), len(val_df))

    s7 = _step_dir(7, "advantage")
    print("[Step 7] Advantage benchmarks...")
    adv = build_advantage_bundle(
        train_df=train_df,
        test_df=test_df,
        blocks=blocks,
        meta_r2=meta.r2_test,
        meta_mae=meta.mae_test,
        coverage=coverage,
        influence=result,
        traj=traj,
        drivers=drivers,
    )
    adv.metrics.to_csv(s7 / "advantage_metrics.csv", index=False, encoding="utf-8-sig")
    adv.metrics.to_csv(TABLES_DIR / "step7_advantage_metrics.csv", index=False, encoding="utf-8-sig")
    if not adv.time_varying.empty:
        adv.time_varying.to_csv(s7 / "time_varying_vs_static.csv", index=False, encoding="utf-8-sig")
    if not adv.safety.empty:
        adv.safety.to_csv(s7 / "safety_comparison.csv", index=False, encoding="utf-8-sig")
    plot_advantage_prediction(adv.metrics)
    plot_advantage_time_varying(adv.time_varying)
    plot_advantage_safety(adv.safety)
    plot_advantage_hierarchy(meta.weight_table, blocks)
    write_step_readme(
        s7,
        "Step 7 — Model Advantage Evidence",
        adv.narrative
        + [
            "Compares hierarchical Meta Ridge vs flat Ridge / single-block baselines.",
            "Shows time-local beta_std(t) vs static coefficient and dynamic vs naive safety bands.",
        ],
    )

    scenario_text = ""
    if not traj.empty:
        d0 = traj.iloc[0]
        n_alarm = int((traj["status"] != "OK").sum())
        scenario_text = (
            f"Sustained `delta {d0['driver']} = {d0['driver_delta']:+.3f}` across "
            f"**0-{int(traj['hour'].max())} h**. "
            f"Rows in trajectory table: {len(traj):,}; alarm flags: {n_alarm:,}."
        )
    elif not scenario.empty:
        d0 = scenario.iloc[0]
        scenario_text = f"Snapshot `delta {d0['driver']} = {d0['driver_delta']:+.3f}` at t={int(d0['hour'])} h."

    report = write_validation_report(
        {
            "coverage": coverage,
            "meta_r2": meta.r2_test,
            "n_train": len(train_b),
            "n_test": len(test_b),
            "n_blocks": len(blocks),
            "block_rows": [
                {"label": r["label"], "r2": r["r2_test"], "mae": r["mae_test"]} for r in block_rows
            ],
            "meta_table": meta.weight_table,
            "scenario_text": scenario_text,
            "advantage_bullets": adv.narrative,
        }
    )

    print("=" * 64)
    print("DONE")
    print(f"  Steps   : {STEPS_DIR}")
    print(f"  Figures : {FIGURES_DIR}")
    print(f"  Report  : {report}")
    print(f"  Coverage: {coverage*100:.1f}% | Meta R2: {meta.r2_test:.3f}")
    print("=" * 64)


if __name__ == "__main__":
    main()
