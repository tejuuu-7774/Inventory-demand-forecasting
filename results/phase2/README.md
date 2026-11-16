# Phase 2 result schemas

The authoritative corrected experiments use the `_v2` suffix. Their immutable run
metadata is under `metadata/` and includes configuration, data-frame and source
artifact hashes. Pre-remediation experiment files were removed during final
repository cleanup; their metrics and invalidation rationale remain documented in
`report/PHASE2_REMEDIATION_REPORT.md`.

`data_metadata.json` describes the common prepared dataset and split boundaries; it
is not a model-run manifest. Always pair a prediction file with its experiment-level
JSON in `metadata/`.

Main prediction files contain:

- deterministic experiment and model names;
- item-store series ID, item, store and category;
- forecast origin, forecast date and horizon step;
- actual demand and q10/q50/q90 forecasts.

Metrics files contain MAE, RMSE, WAPE, three pinball losses, mean pinball loss, 80% interval coverage and mean interval width. Calibration files repeat these measures by horizon, intermittency, volume, category and store.

Cold-start populations are saved separately because their metrics are not directly comparable to the complete 100-series test population. Inventory costs are explicit teaching assumptions rather than real Walmart costs.
