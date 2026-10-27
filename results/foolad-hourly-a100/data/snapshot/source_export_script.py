from pathlib import Path
import time
import jdatetime
import pandas as pd
import pytse_client as tse

SYMBOL = "فولاد"
START = pd.Timestamp(jdatetime.date(1400, 7, 10).togregorian())
END = pd.Timestamp(jdatetime.date(1405, 7, 9).togregorian())

ROOT = Path(__file__).resolve().parent / "exports" / "foolad_hourly"
TRADES = ROOT / "returned_trades"
TRADES.mkdir(parents=True, exist_ok=True)

def main():
    # Download and cache daily history to identify candidate trading dates.
    daily = tse.download(
        symbols=SYMBOL,
        adjust=False,
        include_jdate=True,
        write_to_csv=True,
    )[SYMBOL]
    daily.to_csv(ROOT / "daily_unadjusted.csv", index=False, encoding="utf-8-sig")

    dates = pd.to_datetime(daily["date"])
    days = sorted(set(dates[dates.between(START, END)].dt.date))
    if not days:
        raise RuntimeError("Daily history returned no dates in the requested range.")

    bars = []
    coverage = []

    for number, day in enumerate(days, start=1):
        path = TRADES / f"{day}.csv"
        print(f"[{number}/{len(days)}] {day}", flush=True)

        try:
            if path.exists():
                ticks = pd.read_csv(
                    path, index_col="datetime", parse_dates=["datetime"]
                )
            else:
                result = tse.get_trade_details(
                    SYMBOL,
                    start_date=day,
                    end_date=day,
                    timeframe=None,
                    aggregate=False,
                )
                frames = list(result.values())
                if not frames or all(frame.empty for frame in frames):
                    raise RuntimeError("No intraday trades returned")

                ticks = pd.concat(frames)
                ticks.index.name = "datetime"

                # Preserve exactly the trades returned by pytse-client.
                temporary = path.with_suffix(".tmp")
                ticks.to_csv(temporary, encoding="utf-8-sig")
                temporary.replace(path)

            # Stable ordering preserves returned order for equal timestamps.
            ordered = ticks.sort_index(kind="stable")
            grouped = ordered.resample("1h", closed="left", label="left")

            hourly = grouped.agg(
                open=("price", "first"),
                high=("price", "max"),
                low=("price", "min"),
                hourly_last_price=("price", "last"),
                volume=("volume", "sum"),
                returned_trade_count=("price", "size"),
            )

            # Export only hours containing returned trades.
            hourly = hourly.loc[hourly["returned_trade_count"] > 0]
            bars.append(hourly)
            coverage.append({
                "date": str(day),
                "status": "downloaded",
                "returned_trades": len(ticks),
                "hourly_bars": len(hourly),
                "error": "",
            })

        except Exception as error:
            print(f"  Unresolved: {error}", flush=True)
            coverage.append({
                "date": str(day),
                "status": "unresolved",
                "returned_trades": None,
                "hourly_bars": None,
                "error": str(error),
            })

        pd.DataFrame(coverage).to_csv(
            ROOT / "coverage.csv", index=False, encoding="utf-8-sig"
        )
        time.sleep(1)

    if not bars:
        raise RuntimeError("No hourly data downloaded. Inspect coverage.csv.")

    combined = pd.concat(bars).sort_index().reset_index()
    combined.to_csv(ROOT / "foolad_hourly.csv",
                    index=False, encoding="utf-8-sig")

    with pd.ExcelWriter(ROOT / "foolad_hourly.xlsx") as writer:
        combined.to_excel(writer, sheet_name="Hourly", index=False)
        pd.DataFrame(coverage).to_excel(
            writer, sheet_name="Coverage", index=False
        )

    unresolved = sum(row["status"] == "unresolved" for row in coverage)
    print(f"\nSaved {len(combined)} hourly bars to {ROOT}")
    print(f"Unresolved dates: {unresolved}")
    print("Rerun this script to retry unresolved dates.")

if __name__ == "__main__":
    main()