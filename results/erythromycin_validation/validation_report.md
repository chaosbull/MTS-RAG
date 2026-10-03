# Erythromycin Validation Report

## Framework

**Hierarchical Time-Varying Statistical Control Framework**

Spearman screening -> disjoint block Ridge -> Meta Ridge fusion;
parallel time-local partial regression -> full-horizon propagation -> dynamic safety bands.

---

## Metrics

| Metric | Value |
|--------|-------|
| Alarm coverage | **84.3%** |
| Meta-model test R2 | **0.269** |
| Train / test batches | 1572 / 394 |
| Block models | 5 |

---

## Block model test R2

| Block | R2 test | MAE test |
|-------|---------|----------|
| Oxygenation | 0.195 | 209.39 |
| Carbon Feed | 0.249 | 203.01 |
| Nitrogen | 0.200 | 209.90 |
| Rheology / Biomass | 0.233 | 208.35 |
| Environment | 0.217 | 210.48 |

---

## Meta fusion weights

| Source | Weight | Abs share |
|--------|--------|-----------|
| Environment | -146.983 | 30.7% |
| Carbon Feed | 123.313 | 25.8% |
| Rheology / Biomass | 95.476 | 20.0% |
| Oxygenation | 93.781 | 19.6% |
| Nitrogen | 18.604 | 3.9% |

---

## Model advantage (vs baselines)

- Meta fusion test R2 = 0.269 vs flat Ridge 0.228 (delta = +0.042).
- Meta fusion exceeds the best single block (0.249) by +0.020 R2.
- Time-local beta_std for Soybean Oil (d/h)->Potency differs from a static coefficient by mean |delta beta| = 0.065, showing fermentation-stage heterogeneity.
- Dynamic alarm coverage on held-out batches = 84.3%.
- Across responses, dynamic bands keep 89.2% of controlled trajectories in-band vs 78.6% for naive global bands.
- Unlike flat models, the hierarchy exposes Variable -> Block -> System -> Intervention layers.

Figures: `12_advantage_prediction.png`, `12_advantage_time_varying.png`,
`12_advantage_safety.png`, `12_advantage_hierarchy.png`.

---

## Dynamic control (full horizon)

Sustained `delta Soybean Oil (d/h) = +59.987` across **0-190 h**. Rows in trajectory table: 768; alarm flags: 83. See `full_horizon_trajectory.csv` and figures `08_*` / `12_advantage_*`.

Primary tables: `full_horizon_trajectory.csv`, `horizon_delta_matrix.csv`.
Primary figures: `08_full_horizon_trajectories.png`, `08_horizon_delta_heatmap.png`, `08_horizon_delta_lines.png`.
