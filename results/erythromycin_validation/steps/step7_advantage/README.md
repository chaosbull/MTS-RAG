# Step 7 — Model Advantage Evidence

- Meta fusion test R2 = 0.269 vs flat Ridge 0.228 (delta = +0.042).
- Meta fusion exceeds the best single block (0.249) by +0.020 R2.
- Time-local beta_std for Soybean Oil (d/h)->Potency differs from a static coefficient by mean |delta beta| = 0.065, showing fermentation-stage heterogeneity.
- Dynamic alarm coverage on held-out batches = 84.3%.
- Across responses, dynamic bands keep 89.2% of controlled trajectories in-band vs 78.6% for naive global bands.
- Unlike flat models, the hierarchy exposes Variable -> Block -> System -> Intervention layers.
- Compares hierarchical Meta Ridge vs flat Ridge / single-block baselines.
- Shows time-local β_std(t) vs static coefficient and dynamic vs naive safety bands.
