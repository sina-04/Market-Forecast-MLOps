"""Metrics in original adjusted rial units and reproducible figures."""
import numpy as np


def metrics(actual, predicted):
    actual, predicted = np.asarray(actual).reshape(-1), np.asarray(predicted).reshape(-1)
    if actual.shape != predicted.shape or not np.isfinite([actual, predicted]).all():
        raise ValueError("Predictions and actual values must be finite and aligned")
    error = predicted - actual
    mse = float(np.mean(error ** 2))
    nonzero = actual != 0
    return {"MAE": float(np.mean(np.abs(error))), "MSE": mse, "RMSE": float(np.sqrt(mse)),
            "MAPE": float(np.mean(np.abs(error[nonzero] / actual[nonzero])) * 100) if nonzero.any() else None,
            "samples": int(len(actual))}


def save_figure(fig, output, name):
    """Export print-quality PNG and scalable PDF/SVG from the original figure."""
    from pathlib import Path
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    for extension in ("png", "pdf", "svg"):
        fig.savefig(output / f"{name}.{extension}", dpi=600)


def history_plot(output, frame):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd
    fig, ax = plt.subplots(figsize=(12, 4))
    from .features import price_column
    hourly = "adj_hourly_last_price" in frame
    label = "Adjusted hourly last price (rial)" if hourly else "Adjusted official closing price (rial)"
    ax.plot(pd.to_datetime(frame["date"]), frame[price_column(frame)])
    ax.set(xlabel="Observed bar timestamp" if hourly else "Observed session date", ylabel=label,
           title=f"Foolad — {label}")
    fig.autofmt_xdate()
    fig.tight_layout()
    save_figure(fig, output, "adjusted_close_history")
    plt.close(fig)


def plots(output, predictions, histories):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from pathlib import Path
    output = Path(output)
    for name, history in histories.items():
        fig, ax = plt.subplots(figsize=(9, 4))
        ax.plot(history["loss"], label="Training MSE (scaled)")
        ax.plot(history["val_loss"], label="Validation MSE (scaled)")
        ax.set(xlabel="Epoch", ylabel="MSE", title=name)
        ax.legend()
        fig.tight_layout()
        save_figure(fig, output, f"loss_{name}")
        plt.close(fig)
    fig, ax = plt.subplots(figsize=(12, 5))
    for column in ["actual", "lstm", "persistence", "ridge", "ohlcv_lstm"]:
        ax.plot(predictions["date"], predictions[column], label=column, alpha=0.8)
    hourly = predictions["date"].astype(str).str.len().max() > 10
    ax.set(xlabel="Test bar timestamp" if hourly else "Test session",
           ylabel="Adjusted hourly last price (rial)" if hourly else "Adjusted official closing price (rial)")
    ax.tick_params(axis="x", rotation=45)
    ax.xaxis.set_major_locator(plt.MaxNLocator(8))
    ax.legend()
    fig.tight_layout()
    save_figure(fig, output, "test_forecasts")
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    errors = predictions["lstm"] - predictions["actual"]
    axes[0].plot(errors.to_numpy())
    axes[0].axhline(0, color="black", linewidth=1)
    axes[0].set(xlabel="Test observation", ylabel="LSTM residual (rial)")
    axes[1].hist(errors, bins=30)
    axes[1].set(xlabel="Residual (rial)", ylabel="Count")
    fig.tight_layout()
    save_figure(fig, output, "residuals")
    plt.close(fig)
