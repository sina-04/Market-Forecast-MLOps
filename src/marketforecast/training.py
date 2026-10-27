"""Small validation-only search, checkpointing, and sealed final evaluation."""
from pathlib import Path
import importlib.metadata
import platform
import random
import subprocess
import shutil

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from .evaluation import metrics, plots
from .features import ohlcv_features
from .io import read_json, write_json
from .sequences import prepare


def build_model(tf, length, feature_count, units, config):
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(length, feature_count)),
        tf.keras.layers.LSTM(units),
        tf.keras.layers.Dropout(config["dropout"]),
        tf.keras.layers.Dense(1),
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=config["learning_rate"]), loss="mse")
    return model


def fit_candidate(tf, directory, splits, length, units, config):
    directory.mkdir(parents=True, exist_ok=True)
    checkpoint = directory / "best.keras"
    history_path = directory / "history.json"
    if checkpoint.exists() and history_path.exists():
        return tf.keras.models.load_model(checkpoint, compile=False), read_json(history_path)
    tf.keras.backend.clear_session()
    tf.keras.utils.set_random_seed(config["seed"])
    model = build_model(tf, length, splits["train"]["X"].shape[-1], units, config)
    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=config["patience"], restore_best_weights=True),
        tf.keras.callbacks.ModelCheckpoint(str(checkpoint), monitor="val_loss", save_best_only=True),
        tf.keras.callbacks.CSVLogger(str(directory / "epochs.csv")),
        tf.keras.callbacks.TerminateOnNaN(),
    ]
    # Explicit bounded datasets avoid large default private thread pools on CPU runtimes.
    def dataset(split):
        data = tf.data.Dataset.from_tensor_slices((split["X"], split["y"])).batch(config["batch_size"])
        options = tf.data.Options()
        options.threading.private_threadpool_size = 1
        return data.with_options(options).prefetch(1)
    history = model.fit(dataset(splits["train"]), validation_data=dataset(splits["validation"]),
                        epochs=config["epochs"], callbacks=callbacks, shuffle=False, verbose=2).history
    if not history["val_loss"] or not np.isfinite(history["val_loss"]).all():
        raise RuntimeError("Nonfinite training/validation losses; candidate is not usable")
    write_json(history_path, history)
    return tf.keras.models.load_model(checkpoint, compile=False), history


def restore_prices(values, scaler, persistence=None, target_mode="price"):
    restored = scaler.inverse_transform(np.asarray(values).reshape(-1, 1)).ravel()
    if target_mode == "log_return":
        if persistence is None:
            raise ValueError("Log-return predictions require prior observed prices")
        restored = np.asarray(persistence) * np.exp(restored)
    if not np.isfinite(restored).all():
        raise ValueError("Nonfinite reconstructed prices")
    return restored


def predict_original(model, x, scaler, persistence=None, target_mode="price"):
    # Direct eager inference avoids rebuilding a tf.data pool for every small prediction.
    predictions = np.concatenate([model(x[i:i + 256], training=False).numpy() for i in range(0, len(x), 256)])
    return restore_prices(predictions, scaler, persistence, target_mode)


def train(output, frame, names, plan, config, manifest, audits):
    import tensorflow as tf
    output = Path(output)
    random.seed(config["seed"])
    np.random.seed(config["seed"])
    tf.keras.utils.set_random_seed(config["seed"])
    gpu_info = subprocess.check_output(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"], text=True).strip() if shutil.which("nvidia-smi") else "No NVIDIA GPU"
    if config.get("required_gpu") and (config["required_gpu"] not in gpu_info or not tf.config.list_physical_devices("GPU")):
        raise RuntimeError(f"Required {config['required_gpu']} TensorFlow GPU unavailable: {gpu_info}")
    mode = config.get("target_mode", "price")
    try:
        tf.config.experimental.enable_op_determinism()
    except (AttributeError, RuntimeError):
        pass
    write_json(output / "environment.json", {
        "python": platform.python_version(), "platform": platform.platform(),
        "packages": {distribution.metadata["Name"]: distribution.version
                     for distribution in importlib.metadata.distributions() if distribution.metadata["Name"]},
        "devices": [d.name for d in tf.config.list_physical_devices()], "gpu": gpu_info, "seed": config["seed"],
    })
    candidates = []
    for length in config["sequence_lengths"]:
        splits, xs, ys = prepare(frame, names, length, plan)
        for units in config["lstm_units"]:
            label = f"lstm_{length}_{units}"
            print(f"Training/validating {label}", flush=True)
            model, _ = fit_candidate(tf, output / "candidates" / label, splits, length, units, config)
            prediction = predict_original(model, splits["validation"]["X"], ys, splits["validation"]["persistence"], mode)
            result = {"name": label, "length": length, "units": units,
                      **metrics(splits["validation"]["actual"], prediction)}
            candidates.append(result)
            write_json(output / "validation_search.json", candidates)
    selected = min(candidates, key=lambda r: r["RMSE"])
    splits, xs, ys = prepare(frame, names, selected["length"], plan)
    model = tf.keras.models.load_model(output / "candidates" / selected["name"] / "best.keras", compile=False)
    model.save(output / "model.keras")
    model.save(output / "model.h5")
    joblib.dump({"features": xs, "target": ys}, output / "scalers.joblib")
    write_json(output / "model_metadata.json", {
        "selected": selected, "feature_order": names,
        "target": manifest.get("target", "next observed session adj_official_close"), "unit": "retrospectively adjusted rial",
        "frequency": manifest.get("frequency", "daily"), "target_mode": mode,
        "adjustment_basis_date": manifest["end_date"], "snapshot_checksums": manifest["checksums"],
        "cutoffs": {name: {"first_target": str(split["dates"][0]), "last_target": str(split["dates"][-1]), "samples": len(split["y"])}
                    for name, split in splits.items()},
    })
    for name, split in splits.items():
        np.savez_compressed(output / f"sequences_{name}.npz", **split)
    # Ridge alpha selection uses validation only and the same selected-length inputs.
    ridge_scores = []
    ridge_x = {key: split["X"].reshape(len(split["X"]), -1) for key, split in splits.items()}
    for alpha in config["ridge_alphas"]:
        estimator = Ridge(alpha=alpha).fit(ridge_x["train"], splits["train"]["y"].ravel())
        predicted = restore_prices(estimator.predict(ridge_x["validation"]), ys, splits["validation"]["persistence"], mode)
        ridge_scores.append({"alpha": alpha, **metrics(splits["validation"]["actual"], predicted)})
    alpha = min(ridge_scores, key=lambda r: r["RMSE"])["alpha"]
    ridge = Ridge(alpha=alpha).fit(ridge_x["train"], splits["train"]["y"].ravel())
    joblib.dump(ridge, output / "ridge.joblib")
    write_json(output / "ridge_validation.json", ridge_scores)
    # Ablation holds architecture and target dates constant; fit independent scalers.
    base_splits, base_xs, base_ys = prepare(frame, ohlcv_features(frame), selected["length"], plan)
    base_model, base_history = fit_candidate(tf, output / "candidates" / "ohlcv", base_splits, selected["length"], selected["units"], config)
    base_model.save(output / "ohlcv_model.keras")
    joblib.dump({"features": base_xs, "target": base_ys}, output / "ohlcv_scalers.joblib")
    # Only now access held-out targets and predictions.
    test = splits["test"]
    test_predictions = predict_original(model, test["X"], ys, test["persistence"], mode)
    reloaded = tf.keras.models.load_model(output / "model.keras", compile=False)
    restored = joblib.load(output / "scalers.joblib")
    reproduced = predict_original(reloaded, test["X"], restored["target"], test["persistence"], mode)
    np.testing.assert_allclose(test_predictions, reproduced, rtol=1e-6, atol=1e-3)
    legacy = tf.keras.models.load_model(output / "model.h5", compile=False)
    np.testing.assert_allclose(test_predictions, predict_original(legacy, test["X"], ys, test["persistence"], mode), rtol=1e-6, atol=1e-3)
    predictions = pd.DataFrame({
        "date": test["dates"], "actual": test["actual"], "lstm": test_predictions,
        "persistence": test["persistence"],
        "ridge": restore_prices(ridge.predict(ridge_x["test"]), ys, test["persistence"], mode),
        "ohlcv_lstm": predict_original(base_model, base_splits["test"]["X"], base_ys, test["persistence"], mode),
    })
    results = {name: metrics(predictions["actual"], predictions[name]) for name in ["lstm", "persistence", "ridge", "ohlcv_lstm"]}
    predictions.to_csv(output / "test_predictions.csv", index=False)
    write_json(output / "test_metrics.json", results)
    histories = {"selected_lstm": read_json(output / "candidates" / selected["name"] / "history.json"), "ohlcv_lstm": base_history}
    plots(output, predictions, histories)
    write_report(output, manifest, audits, config, selected, results)
    write_json(output / "completion.json", {"status": "complete", "saved_model_reload_verified": True,
                                            "test_targets": len(test["y"]), "selected": selected})
    return results


def write_report(output, manifest, audits, config, selected, results):
    comparison = "\n".join(f"| {name} | {value['MAE']:.3f} | {value['RMSE']:.3f} | {value['MAPE']:.3f} |" for name, value in results.items())
    improvement = 100 * (1 - results["lstm"]["RMSE"] / results["persistence"]["RMSE"]) if results["persistence"]["RMSE"] else None
    conclusion = (f"LSTM test RMSE improvement against persistence: {improvement:.2f}% (negative means worse)."
                  if improvement is not None else "Persistence has zero RMSE; relative improvement is undefined.")
    metadata = read_json(output / "model_metadata.json")
    if manifest.get("frequency") == "hourly":
        return write_hourly_report(output, manifest, audits, config, selected, results, metadata, comparison, conclusion)
    report = f"""# MarketForecast research report

## Objective and data
Predict فولاد's next observed traded session's adjusted official closing price after the current session.
Source: {manifest['source']}. Period: {manifest['start_date']} to {manifest['end_date']}.
Retrieved: {manifest['retrieved_at_utc']}. Instrument IDs: {', '.join(manifest['instrument_ids'])}.
Prices are rials; volume is shares. The raw source, normalized raw CSV, and SHA-256 checksums are frozen.
Official closing price is pClosing (FinPy Final); pDrCotVal (FinPy Close) is last trade.

## Cleaning and corporate actions
{audits['cleaning']['raw_rows']} raw rows; {audits['cleaning']['clean_rows']} valid rows;
{audits['cleaning']['quarantined_rows']} quarantined rows. Reasons: {audits['cleaning']['reason_counts']}.
Adjustment basis: {audits['cleaning'].get('adjustment_basis', 'Complete valid cleaned history')}.
Quarantined candles are excluded; all input windows and their targets must be consecutive source sessions.
User CSV imports preserve original export bytes; HTTP response bodies and original retrieval/package
version metadata are unavailable when the exporter did not record them (see snapshot manifest).
No synthetic holiday/suspension candles or price interpolation. Large raw moves are flagged, not automatically removed.
Backward adjustment multiplies price fields by the reverse cumulative product of next-session
previous official close / current official close. Final factor = 1. Volume is unchanged.
This uses future corporate-action information: the study is retrospective on a frozen adjustment basis,
not a point-in-time backtest, forecast of unadjusted transaction prices, or evidence of achievable profit.

## Features and chronological preparation
{audits['features']['usable_rows']} usable feature rows after {audits['features']['warmup_or_nonfinite_rows']} warm-up/nonfinite exclusions.
Features include OHLCV, lags, returns, SMA/EMA 5/10/20, volatility/extrema, Wilder RSI-14 and ATR-14,
MACD 12/26/9, Bollinger 20, volume ratios, interaction and calendar variables.
All rolling calculations use current/past observations only, on the fixed adjusted series.
Common eligible target dates are shared by all window lengths. Split fractions are
{config['train_fraction']:.0%} / {config['validation_fraction']:.0%} / {1-config['train_fraction']-config['validation_fraction']:.0%}.
Cutoffs: {metadata['cutoffs']}.
Input and target StandardScalers are fitted independently on training inputs/targets only.

## Models and selection
One LSTM layer; dropout {config['dropout']}; dense linear output; Adam {config['learning_rate']}; MSE loss;
batch size {config['batch_size']}; at most {config['epochs']} epochs; early stopping patience {config['patience']}.
Validation-only search: lengths {config['sequence_lengths']}, units {config['lstm_units']}.
Selected: length {selected['length']}, units {selected['units']}, validation RMSE {selected['RMSE']:.3f} adjusted rials.
Seed: {config['seed']}. Ridge alpha is selected on validation. OHLCV ablation uses the selected architecture.
The test set is evaluated only after all selections; no train-plus-validation refit.

## Held-out results
MAE and RMSE are adjusted rials; MAPE is percent. MSE is also recorded in test_metrics.json.

| Model | MAE | RMSE | MAPE (%) |
|---|---:|---:|---:|
{comparison}

{conclusion}
The result is descriptive; performance is not guaranteed on later market regimes.
Five years of one stock provide limited independent samples, and overlapping windows are correlated.
The study does not include transaction costs, tradability/queues, or a trading strategy.

## Reproducibility and artifacts
The environment, configuration, snapshot checksums, feature order, split dates, scalers,
candidate logs, selected .keras/.h5 models, sequences, predictions, metrics, and figures are saved together.
Both selected saved model formats were reloaded and verified against exported predictions.
See loss_selected_lstm.png, loss_ohlcv_lstm.png, test_forecasts.png, and residuals.png.
"""
    (Path(output) / "report.md").write_text(report, encoding="utf-8")


def write_hourly_report(output, manifest, audits, config, selected, results, metadata, comparison, conclusion):
    report = f"""# Foolad hourly forecasting experiment

Predict the next observed traded bar's adjusted last traded price after the current bar completes.
Period: {manifest['start_date']} through {manifest['end_date']}. Source: {manifest['source']}.
One-hour, left-labelled aggregation; a timestamp identifies the start of its completed bar.
The next observed bar may follow overnight/weekend gaps. This is a different horizon and target from
the previous daily official-close experiment; their raw errors are not a like-for-like improvement measure.

## Data and adjustment
{audits['cleaning']['clean_rows']} valid hourly bars. {audits['cleaning']['unresolved_dates']} unresolved dates.
Raw returned-trade aggregation verification and source hashes are recorded in the snapshot.
No missing-bar interpolation. All returned hours are retained, including
{audits['cleaning']['late_hour_bars']} bars at/after 13:00; source session-hour classification is unverified.
Daily official-close action coefficients are applied uniformly to every bar in each day, never
inferred from hourly last prices. This is a fixed retrospective adjustment, not a point-in-time backtest.
Timezone basis: {manifest['timezone_basis']}. Prices are adjusted rials, volumes are returned shares.

## Features and splits
{audits['features']['usable_rows']} usable feature bars. Indicators restart after unresolved dates.
All candidate windows/targets exclude crossings of known missing days. Periods count observed bars.
Inputs include relative OHLC, returns/lags, normalized rolling indicators, log/relative volume,
hour-of-day and calendar features, and elapsed hours. Hourly labels are full timestamps.
Training-only scalers; chronological 70/15/15 common targets across candidate lengths.
Cutoffs: {metadata['cutoffs']}.

## Training and selection
One LSTM, dropout {config['dropout']}, Adam {config['learning_rate']}, MSE;
batch {config['batch_size']}, up to {config['epochs']} epochs, patience {config['patience']}.
Target mode: {metadata['target_mode']}. Log-return predictions reconstruct price from the prior
observed adjusted last price, giving zero predicted return the persistence forecast.
Validation-only search: {config['sequence_lengths']} observed bars × {config['lstm_units']} units.
Selected {selected['name']}; validation RMSE {selected['RMSE']:.3f} adjusted rials.
Ridge alpha selected on validation; OHLCV ablation uses the selected architecture and common timestamps.
Test targets are evaluated only after selection. Seed {config['seed']}. A100 enforced and recorded in environment.json.

## Held-out hourly results
MAE/RMSE in adjusted rials; MAPE in percent; MSE recorded in test_metrics.json.

| Model | MAE | RMSE | MAPE (%) |
|---|---:|---:|---:|
{comparison}

{conclusion}
More hourly bars do not guarantee better accuracy; overlapping samples are correlated.
There is no trading strategy, transaction-cost model, or profit claim.

## Reproducibility
Snapshot, configuration, sources, environment/GPU identity, timestamp splits, scalers, candidate logs,
selected .keras/.h5 models, predictions, metrics and plots are saved. Both model formats reload and
reproduce the exported predictions. The bundle retains raw-trade hashes; the original raw-trade cache
remains in the local exports folder.
"""
    (Path(output) / "report.md").write_text(report, encoding="utf-8")

