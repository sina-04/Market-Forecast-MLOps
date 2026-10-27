# Selected hourly model and frozen source artifacts

This directory contains the portion of the [Market Forecast hourly experiment](https://github.com/sina-04/Market-Forecast/tree/main/results/foolad-hourly-a100) required to build and verify the MLOps service.

The serving model is the validation-selected full-feature LSTM: 60 observed bars, 49 features, and 32 units. It was trained on an NVIDIA A100 and predicts next-bar log return, reconstructed as adjusted last traded price. This service uses CPU inference and the original fitted scalers.

## Included files

- `data/snapshot/`: frozen bars, original exports, daily adjustment basis, coverage, source exporter, and provenance/checksum records.
- `training-results/hourly-forecast/`: selected `model.keras`, `scalers.joblib`, ordered features, configuration, metadata, the saved 535 test windows and predictions, original metrics, and verification evidence.
- `provenance/training-source/`: the exact original training source, configuration, and dependency records.
- `bundle_checksums.json`: checksum manifest for the selected files present in this repository.
- `upstream_bundle_checksums.json`: the original, unchanged complete research manifest. It includes paths deliberately omitted from this focused repository and is retained for attribution and comparison, not used as a local completeness manifest.

All 97 entries in the original organized research manifest were verified before selecting files. Every selected original file retains its original bytes and SHA-256. The serving build verifies the local subset manifest before producing a release bundle.

## Deliberately omitted

Candidate-search checkpoints, OHLCV/Ridge model weights, the duplicate HDF5 export, charts, regenerated feature/cleaned tables, training archives, and notebook outputs remain in the upstream research project. The MLOps API serves the selected Keras model; the original comparison metrics are retained as evaluation evidence.

## Research limitations

The source covers 4,273 observed bars from October 2, 2021 through September 30, 2026. Fifteen source dates remain unresolved; indicators restart after them and affected windows are excluded. Adjustment factors use later corporate-action information. The resulting dataset supports retrospective replay, not a point-in-time trading backtest.

The selected LSTM has test RMSE 33.049 adjusted rials versus persistence at 32.793. The OHLCV comparison has lower RMSE but was not promoted using test results. See the [hourly methodology](../../docs/hourly_experiment.md) and [root README](../../README.md).
