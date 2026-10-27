# Production service validation

The service was implemented and validated on October 27, 2026 (Asia/Tehran), using Linux amd64 containers in Docker Desktop on Windows.

## Verified behavior

The original implementation passed 44 tests, including notebook validation, optional training integration, real-model inference, API validation, failed audit writes, feature causality, and restore checks. This focused repository omits the notebook and its one validation test; the remaining serving and research regression checks are retained.

Direct inference, HTTP inference, and the MLflow model wrapper reproduced all 535 saved selected-model forecasts under the original `rtol=1e-6`, `atol=0.001` adjusted-rial tolerance. The maximum absolute difference in the recorded run was 0.00289464 adjusted rial, within the combined relative-plus-absolute tolerance.

| Acceptance measurement | Recorded result |
| --- | --- |
| Paced load | 300 requests at 5 requests/second for 60 seconds |
| Maximum configured concurrency | 5 |
| Unexpected errors | 0 |
| p95 latency | 94.44 ms |
| Peak API container memory | 605,765,632 bytes / 577.7 MiB |
| Prometheus alert rules | 5 validated, with deliberate trigger scenarios |
| Provisioned Grafana panels | 9 verified through the API |
| Restart durability | 1,405 audit records retained in the latest recorded rehearsal |
| Backup/restore | 1,406 records restored; prediction and registry identity reproduced |
| MLflow outage | Prediction succeeded with MLflow stopped |
| Registration | Repeated registration reused the verified model version |
| Rollback | Previous verified release tested, then current release restored |

These measurements describe the original implementation run, not a benchmark guarantee. Its release fingerprint is retained in the machine-readable evidence. Extracting the selected source subset and preparing this repository changes the package fingerprint; the image build performs its own artifact and prediction verification.

## Evidence

- [HTTP parity and load measurements](validation/acceptance.json).
- [Restart, outage, backup/restore and monitoring results](validation/operations.json).
- [Rollback and recovery to the current release](validation/rollback.json).

Machine-specific backup locations and recovery project names were removed from the published evidence; measured results and release identities are preserved. No API keys, passwords, audit databases, or generated deployment state are included.

Use the [runbook](production_runbook.md) and the root README to reproduce the checks on a new installation. GitHub Actions is configured to perform regression, image, deployment, alert, load, and recovery checks after publication.

## Scientific interpretation

The service operationalizes a historical experiment. Its selected full-feature LSTM did not beat persistence on held-out RMSE, and its backward-adjusted source data depends on later corporate-action information. Live data ingestion, automatic retraining, trading, and predictive-profitability claims are outside this release.
