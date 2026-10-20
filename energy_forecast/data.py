"""Load the cleaned daily series."""

import pandas as pd

DATA_PATH = "data/daily_energy.csv"


def load(path=DATA_PATH):
    """Return (kwh, imputed) as daily Series with no gaps."""
    df = pd.read_csv(path, parse_dates=["date"], index_col="date").asfreq("D")
    if df["kwh"].isna().any():
        raise ValueError("The daily series has gaps; rebuild it with prepare_data.py")
    return df["kwh"], df["imputed"].astype(bool)
