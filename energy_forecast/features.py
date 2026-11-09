"""Features for forecasting daily energy use 1 to 14 days ahead.

The model is "direct": for a forecast made at the end of day t (the origin) for
day t+h, it only sees data up to day t plus what is known in advance about day
t+h (weekday, season, public holidays). One model covers every horizon h,
with h as a feature.
"""

from datetime import date, timedelta

import numpy as np
import pandas as pd

HORIZON = 14


def easter(year):
    """Easter Sunday (Gregorian calendar), anonymous Gregorian algorithm."""
    a, b, c = year % 19, year // 100, year % 100
    d, e = b // 4, b % 4
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i, k = c // 4, c % 4
    l_ = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * l_) // 451
    month = (h + l_ - 7 * m + 114) // 31
    day = ((h + l_ - 7 * m + 114) % 31) + 1
    return date(year, month, day)


def french_holidays(year):
    """Public holidays in France (the data comes from a home near Paris)."""
    e = easter(year)
    fixed = [(1, 1), (5, 1), (5, 8), (7, 14), (8, 15), (11, 1), (11, 11), (12, 25)]
    moving = [e + timedelta(days=1), e + timedelta(days=39), e + timedelta(days=50)]  # Easter Mon, Ascension, Whit Mon
    return {date(year, m, d) for m, d in fixed} | set(moving)


def portuguese_holidays(year):
    """National public holidays in Portugal (used in live mode with national demand data)."""
    e = easter(year)
    fixed = [(1, 1), (4, 25), (5, 1), (6, 10), (8, 15), (10, 5), (11, 1), (12, 1), (12, 8), (12, 25)]
    moving = [e - timedelta(days=2), e, e + timedelta(days=60)]   # Good Friday, Easter, Corpus Christi
    return {date(year, m, d) for m, d in fixed} | set(moving)


HOLIDAYS = {"fr": french_holidays, "pt": portuguese_holidays}


def holiday_flags(days, country="fr"):
    years = {d.year for d in days}
    hol = set().union(*(HOLIDAYS[country](y) for y in years))
    return np.array([d.date() in hol for d in days], dtype=int)


def calendar_features(target_days, country="fr"):
    """What is known in advance about the day being forecast."""
    doy = target_days.day_of_year.to_numpy()
    return pd.DataFrame({
        "dow": target_days.day_of_week,
        "weekend": (target_days.day_of_week >= 5).astype(int),
        "holiday": holiday_flags(target_days, country),
        "doy_sin": np.sin(2 * np.pi * doy / 365.25),
        "doy_cos": np.cos(2 * np.pi * doy / 365.25),
    }, index=target_days)


def make_rows(y, origins, horizon=HORIZON, weather=None):
    """Build one feature row per (origin, horizon) pair.

    y: daily energy series with a DatetimeIndex and no gaps. Its holiday calendar is
       y.attrs["country"] ("fr" by default, "pt" in live mode).
    origins: positions in y of the last observed day for each forecast.
    weather: optional energy_forecast.weather.Weather; adds the temperature known for each target day.
    Returns a DataFrame of features, with columns origin, target_day and h for bookkeeping.
    """
    values = y.to_numpy()
    idx = y.index
    rows = []
    for t in origins:
        last7 = values[t - 6:t + 1]
        recent = {
            "lag_0": values[t], "lag_1": values[t - 1], "lag_2": values[t - 2],
            "mean_7": last7.mean(), "mean_28": values[t - 27:t + 1].mean(),
            "std_7": last7.std(),
        }
        for h in range(1, horizon + 1):
            k = -(-h // 7)                       # most recent same weekday that is already observed
            same_dow = values[t + h - 7 * k]
            # same weekday 52 weeks earlier; in the first year, fall back to last week's value
            year_ago = values[t + h - 364] if t + h - 364 >= 0 else same_dow
            rows.append({"origin": idx[t], "target_day": idx[t] + pd.Timedelta(days=h), "h": h,
                         **recent, "same_dow": same_dow, "year_ago": year_ago})
    out = pd.DataFrame(rows)
    cal = calendar_features(pd.DatetimeIndex(out["target_day"]), y.attrs.get("country", "fr"))
    out = pd.concat([out, cal.reset_index(drop=True)], axis=1)
    if weather is not None:
        from .weather import weather_features
        for name, values in weather_features(weather.temps(out["origin"], out["target_day"])).items():
            out[name] = values
    return out


FEATURES = ["h", "lag_0", "lag_1", "lag_2", "mean_7", "mean_28", "std_7", "same_dow", "year_ago",
            "dow", "weekend", "holiday", "doy_sin", "doy_cos"]
WEATHER_FEATURES = ["temp_c", "hdd", "cdd"]   # only in live mode, when temperature data exists
FIRST_ORIGIN = 27  # need 28 days of history for mean_28
