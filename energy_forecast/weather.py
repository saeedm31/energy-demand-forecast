"""Temperature for live mode: Portugal, from Open-Meteo (free, no key, CC BY 4.0).

One national number per day: the daily mean temperature ((max + min) / 2) of four cities,
weighted roughly by population. Two files, both in data/live/:

- pt_weather.csv            date, temp_c          what the temperature was (archive + recent days)
- pt_weather_forecasts.csv  issued, date, temp_c  every 16-day forecast, kept as it was issued

Keeping the forecasts as issued matters: a forecast made on a day may only use the weather forecast
known that morning, not the temperature that was measured later. Where no stored forecast exists
(the history before collection started), the measured temperature stands in for the forecast.
"""

import json
import os
import urllib.request
from datetime import date, timedelta

import numpy as np
import pandas as pd

LIVE_DIR = "data/live"
OBS_PATH = os.path.join(LIVE_DIR, "pt_weather.csv")
FORECAST_PATH = os.path.join(LIVE_DIR, "pt_weather_forecasts.csv")

# city, latitude, longitude, weight (about each metro area's share of the population of the four)
CITIES = [("Lisbon", 38.72, -9.14, 0.5), ("Porto", 41.15, -8.61, 0.3),
          ("Coimbra", 40.21, -8.43, 0.1), ("Faro", 37.02, -7.93, 0.1)]
ATTRIBUTION = "Weather: Open-Meteo.com (CC BY 4.0), mean of Lisbon, Porto, Coimbra and Faro"
_COORDS = ("latitude=" + ",".join(str(c[1]) for c in CITIES) +
           "&longitude=" + ",".join(str(c[2]) for c in CITIES))
_DAILY = "daily=temperature_2m_max,temperature_2m_min&timezone=Europe%2FLisbon"
ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive?" + _COORDS + "&start_date={start}&end_date={end}&" + _DAILY
FORECAST_URL = "https://api.open-meteo.com/v1/forecast?" + _COORDS + "&past_days={past}&forecast_days=16&" + _DAILY
BACKFILL_START = date(2019, 1, 1)
ARCHIVE_DELAY_DAYS = 6      # the archive is a few days behind; recent days come from the forecast service
PAST_DAYS = 14
HEATING_BASE, COOLING_BASE = 16.0, 22.0
SANE_C = (-15.0, 45.0)


def parse(payload):
    """Open-Meteo daily response (one location or a list of them) -> weighted mean temperature per day."""
    locations = payload if isinstance(payload, list) else [payload]
    if len(locations) != len(CITIES):
        raise ValueError(f"expected {len(CITIES)} locations, got {len(locations)}")
    parts = []
    for loc, (_, _, _, w) in zip(locations, CITIES):
        d = loc["daily"]
        hi = pd.to_numeric(pd.Series(d["temperature_2m_max"], dtype="object"), errors="coerce").to_numpy()
        lo = pd.to_numeric(pd.Series(d["temperature_2m_min"], dtype="object"), errors="coerce").to_numpy()
        parts.append(pd.Series((hi + lo) / 2 * w, index=pd.DatetimeIndex(pd.to_datetime(d["time"]), name="date")))
    temp = pd.concat(parts, axis=1).sum(axis=1, min_count=len(CITIES)) / sum(c[3] for c in CITIES)
    temp = temp.dropna().round(2)
    return temp[(temp >= SANE_C[0]) & (temp <= SANE_C[1])].rename("temp_c")


def _http_get(url):
    req = urllib.request.Request(url, headers={"User-Agent": "energy-demand-forecast (github.com/saeedm31/energy-demand-forecast)"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.loads(r.read().decode("utf-8"))


def read_observed(path=None):
    path = path or OBS_PATH
    if not os.path.exists(path):
        return pd.Series(dtype=float, name="temp_c", index=pd.DatetimeIndex([], name="date"))
    return pd.read_csv(path, parse_dates=["date"], index_col="date")["temp_c"]


def read_forecasts(path=None):
    path = path or FORECAST_PATH
    if not os.path.exists(path):
        return pd.DataFrame({"issued": pd.Series(dtype="datetime64[ns]"), "date": pd.Series(dtype="datetime64[ns]"),
                             "temp_c": pd.Series(dtype=float)})
    return pd.read_csv(path, parse_dates=["issued", "date"])


def collect(today, get=None, obs_path=None, fc_path=None, log=print):
    """Update measured temperatures and store today's 16-day forecast. The first run backfills from 2019."""
    get = get or _http_get
    obs_path, fc_path = obs_path or OBS_PATH, fc_path or FORECAST_PATH
    obs = read_observed(obs_path)
    if obs.empty:
        end = today - timedelta(days=ARCHIVE_DELAY_DAYS)
        obs = parse(get(ARCHIVE_URL.format(start=BACKFILL_START.isoformat(), end=end.isoformat())))
        log(f"Weather backfill: {len(obs)} days, {BACKFILL_START} to {end}")
    recent = parse(get(FORECAST_URL.format(past=PAST_DAYS)))
    today_ts = pd.Timestamp(today)
    past, future = recent[recent.index < today_ts], recent[recent.index >= today_ts]
    obs = pd.concat([obs[~obs.index.isin(past.index)], past]).sort_index()
    fcs = read_forecasts(fc_path)
    fcs = fcs[fcs["issued"] != today_ts]
    new = pd.DataFrame({"issued": today_ts, "date": future.index, "temp_c": future.to_numpy()})
    fcs = pd.concat([fcs, new], ignore_index=True).sort_values(["issued", "date"])
    os.makedirs(os.path.dirname(obs_path) or ".", exist_ok=True)
    obs.to_frame().to_csv(obs_path, index_label="date", date_format="%Y-%m-%d")
    fcs.to_csv(fc_path, index=False, date_format="%Y-%m-%d")
    log(f"Weather: {len(obs)} measured days up to {obs.index.max().date()}, forecast of {len(new)} days stored")
    return obs, fcs


class Weather:
    """Temperature as it was known when each forecast was made."""

    def __init__(self, observed, forecasts=None):
        self.observed = observed
        self.forecasts = forecasts if forecasts is not None else read_forecasts("/nonexistent")

    @classmethod
    def load(cls, obs_path=None, fc_path=None):
        obs = read_observed(obs_path)
        return cls(obs, read_forecasts(fc_path)) if len(obs) else None

    def temps(self, origins, targets):
        """For each (origin day, target day): the forecast issued the morning after the origin day if stored,
        else the measured temperature, else the latest forecast for that day (days not yet measured)."""
        rows = pd.DataFrame({"issued": pd.DatetimeIndex(origins) + pd.Timedelta(days=1),
                             "date": pd.DatetimeIndex(targets)})
        out = rows.merge(self.forecasts, on=["issued", "date"], how="left")["temp_c"].to_numpy(dtype=float)
        measured = self.observed.reindex(rows["date"]).to_numpy(dtype=float)
        out = np.where(np.isnan(out), measured, out)
        if np.isnan(out).any() and len(self.forecasts):
            latest = self.forecasts.sort_values("issued").drop_duplicates("date", keep="last").set_index("date")["temp_c"]
            out = np.where(np.isnan(out), latest.reindex(rows["date"]).to_numpy(dtype=float), out)
        return out

    def outlook(self, start, days=16):
        """Measured temperature up to yesterday and the latest forecast after, for charts."""
        latest = (self.forecasts.sort_values("issued").drop_duplicates("date", keep="last")
                  .set_index("date")["temp_c"] if len(self.forecasts) else pd.Series(dtype=float))
        future = latest[latest.index >= start].iloc[:days]
        return self.observed, future


def weather_features(temp):
    temp = np.asarray(temp, dtype=float)
    return {"temp_c": temp, "hdd": np.clip(HEATING_BASE - temp, 0, None), "cdd": np.clip(temp - COOLING_BASE, 0, None)}
