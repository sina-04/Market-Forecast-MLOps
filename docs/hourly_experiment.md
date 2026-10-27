# Hourly Foolad experiment

The October 5, 2026 hourly import is frozen under `results/foolad-hourly-a100/data/snapshot`.
All 4,273 exported bars were reconstructed from 1,025 cached raw-trade CSVs and matched. Fifteen
daily coverage entries remain unresolved. There is no interpolation or synthetic replacement.

The original cached trade responses belong to the upstream research acquisition; their SHA-256
checksums and aggregation verification are retained in this snapshot. The snapshot retains the original hourly
export, original daily adjustment references, coverage log, exporter and normalized hourly bars.

The target is the next observed bar's adjusted last traded price. A bar timestamp labels its start,
and the input becomes available only when that bar completes. The next observed target can be the
following session's first bar; it is not necessarily exactly one clock hour later.

Daily corporate-action coefficients use next-session previous official close / current official close.
The cumulative factor applies once, uniformly, to all hourly prices in the corresponding session.
Hourly last prices are never interpreted as official daily closes. Backward adjustment remains
retrospective and depends on later corporate-action information.

Indicators restart after unresolved source dates. Every candidate window and its target must be
consecutive observed source bars without crossing an unresolved date. Known ordinary overnight,
holiday and weekend gaps are preserved. A total of 236 bars start at or after 13:00; they remain in
the dataset because their source session classification has not been independently established.

Relative OHLC/lag/indicator values, returns, log/relative volume, hour/calendar and elapsed-hour
features avoid using absolute historical price scales as inputs. Labels are scaled next-bar log
returns; price is reconstructed from the prior adjusted last price. Zero return produces persistence.

After indicator warm-up there are 4,009 usable rows and 49 input features. Common eligible targets:

| Split | Targets | First timestamp | Last timestamp |
|---|---:|---|---|
| Train | 2,490 | 2021-11-03 12:00:00 | 2024-09-01 11:00:00 |
| Validation | 533 | 2024-09-01 12:00:00 | 2025-05-03 12:00:00 |
| Test | 535 | 2025-05-04 09:00:00 | 2026-09-30 12:00:00 |

Training-only scalers and validation-only length/width/Ridge selection are retained. Six LSTMs
use lengths 12/24/60 observed bars and 32/64 units, batch 64, dropout 0.2, Adam 0.001, at most
100 epochs and early-stopping patience 12. The OHLCV comparison holds the selected architecture
and target timestamps constant. Test results are examined only after selection.

The local-source input bundle has explicit file checksums and a base Git commit marked as including
working-tree modifications. Training must have a TensorFlow-visible A100 GPU; there is no CPU fallback
for this configuration. The environment records the allocated NVIDIA model and memory.

Acquisition and training notebooks are maintained in the [upstream research repository](https://github.com/sina-04/Market-Forecast).
Training completed on NVIDIA A100-SXM4-40GB, with TensorFlow 2.20.0 and Python 3.13.15.
Both selected model formats were reloaded and reproduced all 535 exported test predictions in Colab.
The downloaded archive matched Colab's SHA-256, and all 77 original files passed checksum verification.
The selected serving artifacts are organized under [`results/foolad-hourly-a100`](../results/foolad-hourly-a100/README.md),
with the run in `training-results/hourly-forecast`. Candidate checkpoints and comparison model weights
remain upstream. Original selected-file metadata, source hashes and bytes are preserved; the original
complete manifest and a local subset manifest document the extraction.
An independent local CPU process also reloaded both selected formats, Ridge and the OHLCV model,
reproduced all three splits and all four saved metric sets, and wrote `local_verification.json`.
The OHLCV CPU/A100 prediction difference was at most 0.009377 adjusted rial; its explicit
cross-environment tolerance is 0.01 rial. Original test predictions and metrics remain unchanged.

| Model | Test MAE (adjusted rials) | Test RMSE (adjusted rials) | Test MAPE (%) |
|---|---:|---:|---:|
| Full-feature LSTM | 22.646 | 33.049 | 0.935 |
| Persistence | 20.322 | 32.793 | 0.832 |
| Ridge | 52.373 | 99.132 | 2.149 |
| OHLCV-only LSTM | 22.452 | 32.143 | 0.923 |

Validation selected the full-feature 60-bar, 32-unit LSTM. Its test RMSE was 0.78% worse than
persistence. The OHLCV comparison had 1.98% lower test RMSE than persistence but higher MAE/MAPE.
This secondary result is reported without promoting it through test-based model selection.

The daily result remains a historical experiment. Its official-close errors and the hourly last-price
errors use different horizons and targets, so direct raw-error comparisons cannot establish improvement.
Hourly conclusions should be based on the same-timestamp persistence/Ridge/OHLCV comparisons.
