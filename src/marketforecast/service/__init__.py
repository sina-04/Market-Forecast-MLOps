"""Versioned historical-replay serving, separate from research orchestration."""

MODE = "historical_replay"
LIMITATION = (
    "Historical replay on a frozen, retrospectively adjusted dataset. "
    "Not point-in-time live forecasting or evidence of trading profitability. "
    "The selected LSTM did not outperform persistence on held-out RMSE."
)
