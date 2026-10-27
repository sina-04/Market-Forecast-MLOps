# Hourly LSTM historical replay service

This service deploys the existing validation-selected Foolad LSTM: 60 observed hourly bars, 49 features, and 32 units. It predicts the next observed bar's retrospectively adjusted last traded price. It serves the 535 verified held-out windows from the frozen October 5 export; it does not collect current market data.

The adjustment basis uses later corporate-action information. Responses, API documentation, and monitoring identify this as **historical replay**, not point-in-time live forecasting. The selected LSTM's test RMSE is 33.049 adjusted rials versus persistence at 32.793. Deployment does not establish trading profitability. Published research artifacts remain unchanged.

## Requirements and initial Windows deployment

Use Docker Desktop with Linux containers and Docker Compose supporting `!override` (2.24.4 or later; tested with 5.6.1). Docker runs Python 3.12, TensorFlow CPU 2.19.1, and Keras 3.13.2. Host Python is needed only for standard-library management scripts; the laptop's old research virtual environments are not used.

From the repository root in PowerShell:

```powershell
./deploy/setup.ps1
docker build --target production -t marketforecast-service:local .
python deploy/pin_release.py
docker compose up -d --no-build --wait mlflow
docker compose run --rm register
docker compose up -d --no-build --wait api prometheus grafana
docker compose ps
python deploy/export_release.py
```

`setup.py` creates `.env` only when absent. It generates a random API key and Grafana administrator password without printing them. Both `.env` and `deploy/local/` are ignored by Git and excluded from Docker's build context. Keep their filesystem permissions restricted to your account. Existing credentials are preserved.

The build verifies the complete published checksum manifest, builds a minimal inference bundle, and reproduces all 535 forecasts before producing the serving image. All Python dependencies are locked and base/monitoring images are pinned by digest. The serving process uses an unprivileged account, a read-only container filesystem, temporary scratch storage, 2 CPUs, and a 2 GiB memory limit. No GPU is required.

The export command copies the baked bundle into `artifacts/production/releases/<fingerprint>/`, verifies its files, and produces a ZIP with a SHA-256 sidecar. It contains no credentials. The same bundle is stored as an MLflow model artifact.

`pin_release.py` tags the image with its full release fingerprint, writes that tag into `.env`, and records its Docker image ID in `deploy/local/deployment.json`. Retain previous images; do not overwrite release tags. The image's verified bundle and code are authoritative. The registration file must match that fingerprint or API startup fails.

Registration checks real-model parity, logs a portable MLflow Python model and the frozen dataset, reloads the wrapper and checks all 535 predictions, then registers and promotes the version. Repeating the command reuses a verified version with the same release fingerprint. `candidate`, `champion`, and `previous` aliases describe promotion history; changing an alias does not change a running API.

## Local interfaces and credentials

All ports bind to localhost:

| Interface | Address |
| --- | --- |
| API documentation | http://127.0.0.1:8000/docs |
| MLflow | http://127.0.0.1:5000 |
| Prometheus | http://127.0.0.1:9090 |
| Grafana | http://127.0.0.1:3000/d/forecast-replay |

Grafana uses username `admin` and `GRAFANA_ADMIN_PASSWORD` from the local `.env`. Anonymous access is disabled. MLflow and Prometheus are local administrative interfaces; never publish their ports directly to the internet.

PowerShell example, loading the key without displaying it:

```powershell
$forecastKey = ((Get-Content -LiteralPath .env | Where-Object { $_ -like 'MARKETFORECAST_API_KEY=*' }) -split '=', 2)[1]
$forecastHeaders = @{ 'X-API-Key' = $forecastKey }
$forecastTimes = Invoke-RestMethod http://127.0.0.1:8000/v1/replay/timestamps -Headers $forecastHeaders
$forecastBody = @{ as_of = $forecastTimes.timestamps[0].as_of } | ConvertTo-Json
Invoke-RestMethod http://127.0.0.1:8000/v1/predict -Method Post -Headers $forecastHeaders -ContentType 'application/json' -Body $forecastBody
```

For curl, set `MARKETFORECAST_API_KEY` from your local secret manager or `.env` in the current shell, then use:

```bash
curl -H "X-API-Key: $MARKETFORECAST_API_KEY" http://127.0.0.1:8000/v1/replay/timestamps
curl -X POST http://127.0.0.1:8000/v1/predict \
  -H "X-API-Key: $MARKETFORECAST_API_KEY" -H 'Content-Type: application/json' \
  -d '{"as_of":"2025-05-03T12:00:00+03:30"}'
```

`as_of` is the final **input bar's start label**, with an explicit offset. The snapshot contains completed bars. The service normalizes offsets to `Asia/Tehran` and requires an exact eligible timestamp. The target is the next observed traded bar; overnight/weekend gaps need not equal one clock hour.

| Endpoint | Access and result |
| --- | --- |
| `GET /health/live` | Public local process liveness |
| `GET /health/ready` | Public local model and writable audit-store readiness |
| `GET /v1/model` | API key; release/version identity, model contract, snapshot coverage and limitations |
| `GET /v1/replay/timestamps` | API key; 535 supported input/target timestamp pairs |
| `POST /v1/predict` | API key; one `{ "as_of": "<offset timestamp>" }` request |
| `GET /metrics` | Local/internal Prometheus collection |

Predictions return a prediction/request ID, release fingerprint, registry version, input and target timestamps, adjusted-price prediction, persistence forecast, unit, replay mode, and limitations. Actual target prices are not returned or used as model inputs. Requests with extra fields, naive/malformed timestamps, or wrong types receive `422`; unsupported timestamps receive `404`; invalid credentials receive `401`. Inference and durable audit failures cannot produce successful predictions.

## Validation and monitoring

```powershell
docker build --target test -t marketforecast-test:local .
docker run --rm -e MARKETFORECAST_TRAIN_TEST=1 marketforecast-test:local pytest -q
docker compose exec -T api marketforecast-service verify
docker compose exec -T prometheus promtool check rules /etc/prometheus/alerts.yml
docker compose exec -T -w /etc/prometheus prometheus promtool test rules alerts.test.yml
python tools/production_acceptance.py
docker compose exec -T api marketforecast-service replay-evaluate
```

`python tools/production_operations_check.py` checks restart persistence, inference while MLflow is stopped, idempotent registration, backup and isolated restoration, Prometheus, and provisioned Grafana panels. It writes `artifacts/production/operations.json`. `python tools/production_rollback_check.py` rehearses rollback and return to the current image; it requires a retained previous registered release and writes `artifacts/production/rollback.json`. Run operational rehearsals separately from the load test because they intentionally restart services.

The acceptance script checks all 535 HTTP predictions, input failures, and authentication, then warms up and sends five requests per second for 60 seconds with at most five concurrent workers. It records results in `artifacts/production/acceptance.json`. Acceptance requires zero unexpected errors, p95 below 500 ms, and peak container memory below 2 GiB. Direct and HTTP parity use the original `atol=0.001` adjusted rial plus `rtol=1e-6`; the absolute difference can exceed 0.001 while satisfying that combined tolerance.

Predictions and failures are durably recorded in SQLite. Every 60 seconds the API evaluates the latest 100 distinct target timestamps, deduplicated by release. Until 100 distinct targets exist, quality and PSI metrics remain absent and the sample count shows why. Actual labels are joined by a separate evaluator after inference.

Dashboards show requests, latency, process resources, historical MAE/RMSE/MAPE versus persistence, and feature PSI against training-only references. PSI uses ten quantile bins with duplicate boundaries merged. Prometheus alerts appear in the dashboard's alerts table: unavailability over one minute, prediction errors above 1% and p95 above 500 ms for five minutes with at least 100 requests, historical RMSE above 1.2 times persistence, and feature PSI above 0.2 with 100 observations. These are review signals, not proof of live concept drift. No email, chat, or external notification is configured.

Prometheus retains 15 days of diagnostics. Audit records and MLflow artifacts persist until an operator archives them. Logs rotate through Docker's local logging driver. Sleeping/shutting down this laptop stops availability; Docker Desktop must be running.

## Backup, restore, and rollback

Stop MLflow while copying its artifact store; SQLite backup uses its online backup API. Use a fresh backup name each time:

```powershell
docker compose stop mlflow
docker compose run --rm --no-deps backup marketforecast-service backup --output /backups/acceptance
docker compose start mlflow
python deploy/export_backup.py --name acceptance
```

Backups are staged in the Docker `backup_data` volume. The export command copies them into `deploy/local/backups/` with your host account's ownership and verifies their checksums, avoiding Linux bind-mount ownership problems. They contain the audit database, MLflow database/artifacts, registry metadata, and a checksum manifest. Keep `.env` separately in secure storage, and retain the pinned image. Prometheus/Grafana histories are disposable; their configuration and dashboards are provisioned from Git.

Test restoration into a separate Compose project with fresh volumes and ports:

```powershell
docker compose -p marketforecast-recovery -f compose.yaml -f deploy/recovery.compose.yaml run --rm --no-deps restore
docker compose -p marketforecast-recovery -f compose.yaml -f deploy/recovery.compose.yaml up -d --no-build --wait mlflow api
python tools/production_acceptance.py --url http://127.0.0.1:8001 --duration 1 --output artifacts/production/recovery.json
docker compose -p marketforecast-recovery -f compose.yaml -f deploy/recovery.compose.yaml down
```

Restore validates all backup checksums and refuses nonempty destination volumes. If restoring an older backup, first select its matching image using `pin_release.py --image <retained-release-tag>`; use the appropriate credentials from secure storage. Never restore over running services. Recovery volumes remain after `down` for inspection; choose a fresh project name for another rehearsal.

For a model/code release rollback:

```powershell
python deploy/pin_release.py --image marketforecast-service:<previous-release-fingerprint>
docker compose run --rm register
docker compose up -d --no-build --force-recreate --wait api
```

Use the exact image ID recorded in deployment history to confirm the retained tag before changing the running service. Registry aliases alone do not reload it. After rollback, confirm readiness, identity, and a known prediction. Reapply the newer fingerprint using the same steps to return to the current release.

## Manual retraining and Linux deployment

Review source coverage, data quality, drift and baseline metrics before retraining. Use a new research output root and the existing Colab training workflow. Never retrain inside the API, fit new scalers on incoming requests, select a model using old held-out test results, or silently replace the frozen snapshot. A changed experiment needs its own documented chronological evaluation and verified serving contract before extending the bundle importer. This v1 importer deliberately accepts only the verified 535-window hourly publication.

For a new serving-code release, run the test suite, build with `docker build --target production -t marketforecast-service:local .`, pin it, register/promote it, recreate the API, and rerun acceptance. Build only the staging `:local` tag; do not rebuild over a fingerprint tag with `docker compose build` or `up --build`. Keep the prior image, registration, and backup. The final report is `docs/production_report.md`.

For a Linux server, copy the repository and exact frozen artifacts, install Docker Engine plus a compatible Compose plugin, run `python3 deploy/setup.py`, then follow the same build, pin, registration, and deployment steps. Preserve the image digest/dependency locks. For a different CPU architecture, build and reverify numerical parity before deployment; this release was tested on Linux amd64.

Keep services bound to localhost. Public access requires a separately configured TLS reverse proxy, access controls, and firewall; expose only the authenticated API through that gateway. Do not publish MLflow, metrics, or Grafana without their own administrative access controls. Budget, remote hosting, high availability, live data ingestion and automatic retraining remain outside this laptop-first release.
