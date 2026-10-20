"""Turn the raw UCI minute-level readings into a clean daily series.

Input:  household_power_consumption.zip (or .txt) from the UCI Machine Learning Repository
        https://archive.ics.uci.edu/dataset/235/individual+household+electric+power+consumption
Output: data/daily_energy.csv with columns date, kwh, imputed

Usage:
    python prepare_data.py path/to/household_power_consumption.zip

The cleaned file is already in the repo, so you only need this to rebuild it.
"""

import sys

import numpy as np
import pandas as pd

MIN_COVERAGE = 0.8   # a day needs 80% of its minutes to be measured directly
OUT = "data/daily_energy.csv"


def load_minutes(path):
    df = pd.read_csv(path, sep=";", usecols=["Date", "Time", "Global_active_power"],
                     na_values=["?"], low_memory=False)
    ts = pd.to_datetime(df["Date"] + " " + df["Time"], format="%d/%m/%Y %H:%M:%S")
    return pd.Series(df["Global_active_power"].to_numpy(), index=ts, name="kw")


def to_daily(minutes):
    """Daily energy in kWh, with outage days filled in and flagged."""
    daily = minutes.resample("D").agg(["sum", "count"])
    coverage = daily["count"] / 1440
    # kW averaged over a minute -> kWh = kW / 60. Scale up for the few missing minutes.
    kwh = (daily["sum"] / 60) / coverage.replace(0, np.nan)
    kwh[coverage < MIN_COVERAGE] = np.nan

    # Drop the partial first and last days of the recording.
    kwh = kwh.iloc[1:-1]

    imputed = kwh.isna()
    # Fill outage days with the average of the same weekday 1 and 2 weeks before and after.
    neighbours = pd.concat([kwh.shift(k) for k in (7, -7, 14, -14)], axis=1)
    kwh = kwh.fillna(neighbours.mean(axis=1))
    kwh = kwh.interpolate(limit_direction="both")  # any day still missing
    return pd.DataFrame({"kwh": kwh.round(3), "imputed": imputed.astype(int)})


def main(path):
    daily = to_daily(load_minutes(path))
    daily.index.name = "date"
    daily.to_csv(OUT, date_format="%Y-%m-%d")
    print(f"Wrote {OUT}: {len(daily)} days, {daily['imputed'].sum()} imputed, "
          f"{daily.index.min().date()} to {daily.index.max().date()}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    main(sys.argv[1])
