"""Live mode: real daily electricity demand for Portugal, collected every day.

Optional. The household dataset stays the default; live mode only runs when you turn it on
(the LIVE_MODE repository variable for the daily workflow, or `python live.py ...` by hand).

The cycle:
1. collect  - every day, add yesterday's national demand (GWh) from Energy-Charts (CC BY 4.0).
2. evaluate - the model is frozen at its training cutoff. Every day collected after that cutoff is
              new, unseen data, so it is used to test the model live (forecast vs what happened).
3. retrain  - not every day. Once at least `window_days` new days have been collected (minimum 7,
              default 28), a person can start a new training. It is promoted only if it passes a gate.

Data source: Energy-Charts public power API by Fraunhofer ISE, series "load" (MW), no key needed.
"""

import json
import os
import time
import urllib.error
import urllib.request
from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo

import joblib
import numpy as np
import pandas as pd

from . import backtest, models
from .features import HORIZON, make_rows

LIVE_DIR = "data/live"
DATA_PATH = os.path.join(LIVE_DIR, "pt_daily_demand.csv")
STATE_PATH = os.path.join(LIVE_DIR, "state.json")
MODEL_PATH = "models/live_model.joblib"

COUNTRY = "pt"
TZ = ZoneInfo("Europe/Lisbon")
API_URL = "https://api.energy-charts.info/v2/public_power?country={country}&start={start}&end={end}"
USER_AGENT = "energy-demand-forecast (github.com/saeedm31/energy-demand-forecast)"
ATTRIBUTION = "Data: energy-charts.info (Fraunhofer ISE), CC BY 4.0"
BACKFILL_START = date(2019, 1, 1)
CHUNK_DAYS = 120          # days per request during backfill
PAUSE_SECONDS = 31        # the API allows 2 requests per minute
REFRESH_DAYS = 7          # re-download the last days on every run, in case the source revised them

MIN_WINDOW_DAYS = 7
DEFAULT_WINDOW_DAYS = 28
MIN_TRAIN_DAYS = 400      # year-ago feature and interval calibration need more than a year
SANE_GWH = (40.0, 300.0)  # Portugal uses about 100-200 GWh a day
MAX_GAP_DAYS = 7          # short gaps are filled (and never trained on or scored); longer ones stop
GATE_WEEKS = 8
GATE_COVERAGE = (60.0, 95.0)


def is_on(path=None):
    """Live mode is on once live data has been collected (the daily workflow runs only if LIVE_MODE=true)."""
    return os.path.exists(path or DATA_PATH)


# ---------- 1. collect ----------

def parse_energy_charts(payload, series_id="load"):
    """Turn an Energy-Charts public_power response into complete daily totals in GWh.

    Returns a DataFrame indexed by local date (Europe/Lisbon) with columns gwh and points.
    A day is kept only if every interval is present and none is missing (null).
    Accepts the v2 format (series + data records) and the older v1 format (unix_seconds + production_types).
    """
    if "data" in payload and "series" in payload:                       # v2
        minutes = payload.get("interval_minutes")
        ts = pd.to_datetime([rec["timestamp"] for rec in payload["data"]], utc=True)
        values = [rec.get("values", {}).get(series_id) for rec in payload["data"]]
    elif "unix_seconds" in payload:                                    # v1
        minutes = None
        ts = pd.to_datetime(payload["unix_seconds"], unit="s", utc=True)
        names = {p["name"].lower(): p["data"] for p in payload.get("production_types", [])}
        values = names.get(series_id) or names.get("load")
        if values is None:
            raise ValueError(f"series {series_id!r} not in response")
    else:
        raise ValueError("unknown Energy-Charts response format")
    s = pd.Series(pd.to_numeric(pd.Series(values, dtype="object"), errors="coerce").to_numpy(), index=ts)
    s = s[~s.index.duplicated(keep="last")].sort_index()
    if s.empty:
        return pd.DataFrame(columns=["gwh", "points"])
    if not minutes:
        minutes = int(round(pd.Series(s.index).diff().dropna().dt.total_seconds().median() / 60))
    local = s.index.tz_convert(TZ)
    days = []
    for day, part in s.groupby(local.date):
        start = pd.Timestamp(day).tz_localize(TZ)
        end = (pd.Timestamp(day) + pd.Timedelta(days=1)).tz_localize(TZ)
        expected = int((end - start).total_seconds() // (minutes * 60))   # 23 or 25 hours on clock-change days
        if len(part) != expected or part.isna().any():
            continue                                                   # incomplete day: try again tomorrow
        gwh = part.sum() * minutes / 60 / 1000
        days.append({"date": pd.Timestamp(day), "gwh": round(float(gwh), 3), "points": len(part)})
    return pd.DataFrame(days, columns=["date", "gwh", "points"]).set_index("date")


def fetch(start, end, get=None, retries=3):
    """Download one date range. `get` can be replaced in tests (no network in tests)."""
    url = API_URL.format(country=COUNTRY, start=start.isoformat(), end=end.isoformat())
    get = get or _http_get
    for attempt in range(retries):
        try:
            return get(url)
        except urllib.error.HTTPError as e:
            if e.code == 429 and attempt < retries - 1:                # rate limited: wait as asked
                time.sleep(int(e.headers.get("Retry-After", 60)))
                continue
            raise


def _http_get(url):
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT, "Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.loads(r.read().decode("utf-8"))


def read_data(path=None):
    path = path or DATA_PATH
    if not os.path.exists(path):
        return pd.DataFrame(columns=["gwh"], index=pd.DatetimeIndex([], name="date"))
    return pd.read_csv(path, parse_dates=["date"], index_col="date")[["gwh"]]


def check_days(new):
    """Data checks before anything is saved. Returns (good rows, list of problems)."""
    problems = []
    bad = (new["gwh"] < SANE_GWH[0]) | (new["gwh"] > SANE_GWH[1])
    for day in new.index[bad]:
        problems.append(f"{day.date()}: {new.loc[day, 'gwh']} GWh is outside {SANE_GWH}")
    return new[~bad], problems


def collect(path=None, today=None, get=None, pause=None, log=print):
    """Add every complete day up to yesterday (Lisbon time). The first run backfills from 2019."""
    path = path or DATA_PATH
    pause = PAUSE_SECONDS if pause is None else pause
    today = today or datetime.now(TZ).date()
    yesterday = today - timedelta(days=1)
    have = read_data(path)
    start = (have.index.max().date() - timedelta(days=REFRESH_DAYS - 1)) if len(have) else BACKFILL_START
    if start > yesterday:
        log("Nothing to collect.")
        return have, []
    got, problems = [], []
    chunk_start = start
    while chunk_start <= yesterday:
        chunk_end = min(chunk_start + timedelta(days=CHUNK_DAYS - 1), yesterday)
        # ask for one extra day so the last local day is complete whatever the API's end convention is
        daily = parse_energy_charts(fetch(chunk_start, chunk_end + timedelta(days=1), get=get))
        daily = daily[(daily.index >= pd.Timestamp(chunk_start)) & (daily.index <= pd.Timestamp(chunk_end))]
        good, bad = check_days(daily)
        got.append(good)
        problems += bad
        log(f"{chunk_start} to {chunk_end}: {len(good)} complete days")
        chunk_start = chunk_end + timedelta(days=1)
        if chunk_start <= yesterday and pause:
            time.sleep(pause)
    new = pd.concat(got)[["gwh"]] if got else have.iloc[:0]
    merged = pd.concat([have[~have.index.isin(new.index)], new]).sort_index()
    merged = merged[~merged.index.duplicated(keep="last")]
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    merged.to_csv(path, index_label="date", date_format="%Y-%m-%d")
    log(f"Saved {len(merged)} days to {path} (added or refreshed {len(new)}).")
    for p in problems:
        log("Skipped: " + p)
    return merged, problems


def load(path=None):
    """Return (gwh, imputed) daily Series with no gaps, ready for the model.

    Missing days (up to MAX_GAP_DAYS in a row) are filled with the same weekday one week
    earlier and flagged as imputed, so they are never trained on or scored.
    """
    path = path or DATA_PATH
    df = read_data(path)
    if df.empty:
        raise FileNotFoundError(f"No live data yet: run python live.py collect ({path})")
    y = df["gwh"].asfreq("D")
    imputed = y.isna()
    runs = imputed.astype(int).groupby((~imputed).cumsum()).sum()
    if runs.max() > MAX_GAP_DAYS:
        raise ValueError(f"Live data has a gap longer than {MAX_GAP_DAYS} days; run collect again")
    for _ in range(MAX_GAP_DAYS):
        if not y.isna().any():
            break
        y = y.fillna(y.shift(7)).fillna(y.shift(-7))
    y = y.ffill().bfill()
    y.attrs["country"] = COUNTRY
    y.name = "gwh"
    return y, imputed.rename("imputed")


# ---------- state: what the model was trained on ----------

def read_state(path=None):
    path = path or STATE_PATH
    if not os.path.exists(path):
        return {"trained_until": None, "window_days": DEFAULT_WINDOW_DAYS, "history": []}
    with open(path) as f:
        return json.load(f)


def write_state(state, path=None):
    path = path or STATE_PATH
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(state, f, indent=2)
        f.write("\n")


def validate_window(window_days):
    window_days = int(window_days)
    if window_days < MIN_WINDOW_DAYS:
        raise ValueError(f"window_days must be at least {MIN_WINDOW_DAYS}")
    return window_days


def new_model():
    return models.BoostingModel()


def fit_until(y, imputed, cutoff, model=None):
    """Train the boosting model on data up to and including `cutoff` (a Timestamp)."""
    model = model or new_model()
    last = y.index.get_loc(cutoff)
    rows = backtest.training_rows(y, last)
    days = pd.DatetimeIndex(rows["target_day"])
    keep = ~imputed.reindex(days).to_numpy()
    model.fit(rows[keep], y.reindex(days).to_numpy()[keep])
    model.trained_until = cutoff
    return model


def current_model(y, imputed, state, path=None):
    """The model frozen at the state's cutoff (cached on disk; training is deterministic)."""
    if not state.get("trained_until"):
        return None
    cutoff = pd.Timestamp(state["trained_until"])
    path = path or MODEL_PATH
    if os.path.exists(path):
        model = joblib.load(path)
        if getattr(model, "trained_until", None) == cutoff:
            return model
    model = fit_until(y, imputed, cutoff)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    joblib.dump(model, path)
    return model


# ---------- 2. evaluate on days the model has not seen ----------

def live_evaluation(model, y, imputed, cutoff, horizon=HORIZON):
    """Score the frozen model on every forecast whose target day came after its training cutoff.

    Each forecast uses only data up to its own origin day, exactly as it would have been made live.
    """
    cutoff = pd.Timestamp(cutoff)
    first, last = y.index.get_loc(cutoff), len(y) - 2
    out = {"days_since_training": int((y.index[-1] - cutoff).days), "forecasts_scored": 0}
    if last < first:
        return out
    rows = make_rows(y, list(range(first, last + 1)), horizon)
    rows = rows[rows["target_day"] <= y.index[-1]].reset_index(drop=True)
    days = pd.DatetimeIndex(rows["target_day"])
    keep = ~imputed.reindex(days).to_numpy()
    rows, days = rows[keep].reset_index(drop=True), days[keep]
    if rows.empty:
        return out
    actual = y.reindex(days).to_numpy()
    point = model.predict(rows)
    lo, hi = model.predict_interval(rows)
    naive = rows["same_dow"].to_numpy()
    err = np.abs(point - actual)
    day_ahead = (rows["h"] == 1).to_numpy()
    out.update({
        "forecasts_scored": int(len(rows)),
        "target_days": int(days.nunique()),
        "mae_gwh": round(float(err.mean()), 3),
        "mape_pct": round(float(100 * (err / actual).mean()), 2),
        "day_ahead_mae_gwh": round(float(err[day_ahead].mean()), 3) if day_ahead.any() else None,
        "skill_vs_naive_pct": round(float(100 * (1 - err.mean() / np.abs(naive - actual).mean())), 1),
        "interval_coverage_pct": round(float(100 * np.mean((actual >= lo) & (actual <= hi))), 1),
    })
    return out


def status(y, imputed, state, today=None):
    """Plain facts for the dashboard and the retrain decision."""
    window = state.get("window_days", DEFAULT_WINDOW_DAYS)
    trained = state.get("trained_until")
    since = int((y.index[-1] - pd.Timestamp(trained)).days) if trained else None
    ready = since is None or since >= window
    today = today or datetime.now(TZ).date()
    return {
        "mode": "live",
        "source": ATTRIBUTION,
        "data_first_day": str(y.index[0].date()),
        "data_last_day": str(y.index[-1].date()),
        "days": int(len(y)),
        "data_age_days": int((pd.Timestamp(today) - y.index[-1]).days),
        "imputed_last_28_days": int(imputed.iloc[-28:].sum()),
        "trained_until": trained,
        "window_days": window,
        "new_days_since_training": since,
        "retrain_ready": bool(ready and len(y) >= MIN_TRAIN_DAYS),
        "next_retrain_after": (str((pd.Timestamp(trained) + pd.Timedelta(days=window)).date())
                               if trained else None),
    }


# ---------- 3. retrain, by a person, behind a gate ----------

def gate(y, imputed, weeks=GATE_WEEKS, make_models=None):
    """Re-run a short rolling backtest on the newest data. The new model is promoted only if it
    still beats the last-week baseline and its 80% range is honest."""
    make_models = make_models or (lambda: [models.SeasonalNaive(), new_model()])
    scores = backtest.score(backtest.backtest(y, imputed, make_models(), test_weeks=weeks)).set_index("model")
    gb = scores.loc["gradient_boosting"]
    coverage = float(gb.get("interval_coverage_pct", float("nan")))
    checks = {
        "beats_baseline": bool(gb["skill_vs_naive_pct"] > 0),
        "interval_honest": bool(GATE_COVERAGE[0] <= coverage <= GATE_COVERAGE[1]),
        "recent_data_complete": bool(imputed.iloc[-28:].sum() <= 3),
    }
    result = {"weeks": weeks, "mae_gwh": round(float(gb["mae_kwh"]), 3),
              "skill_vs_naive_pct": round(float(gb["skill_vs_naive_pct"]), 1),
              "interval_coverage_pct": round(coverage, 1), "checks": checks}
    return all(checks.values()), result


def retrain(y, imputed, state, window_days=None, force=False, now=None, gate_fn=None, model_path=None):
    """Start a new training if enough new days were collected. Returns (promoted, message, state)."""
    state = dict(state)
    if window_days is not None:
        state["window_days"] = validate_window(window_days)
    info = status(y, imputed, state)
    if len(y) < MIN_TRAIN_DAYS:
        return False, f"Need at least {MIN_TRAIN_DAYS} days of data, have {len(y)}.", state
    if not info["retrain_ready"] and not force:
        return False, (f"Only {info['new_days_since_training']} new days since training; "
                       f"the window is {state['window_days']}. Next retrain after {info['next_retrain_after']}."), state
    previous = None
    if state.get("trained_until"):
        old = current_model(y, imputed, state, path=model_path)
        previous = live_evaluation(old, y, imputed, state["trained_until"])
    ok, gate_result = (gate_fn or gate)(y, imputed)
    entry = {"run_at": (now or datetime.now(TZ)).isoformat(timespec="seconds"),
             "trained_until": str(y.index[-1].date()), "window_days": state["window_days"],
             "previous_model_live": previous, "gate": gate_result, "promoted": ok}
    state["history"] = (state.get("history", []) + [entry])[-24:]
    if not ok:
        failed = [k for k, v in gate_result["checks"].items() if not v]
        return False, "Gate failed: " + ", ".join(failed) + ". The current model stays.", state
    model = fit_until(y, imputed, y.index[-1])
    path = model_path or MODEL_PATH
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    joblib.dump(model, path)
    state["trained_until"] = str(y.index[-1].date())
    return True, f"New model trained on data up to {state['trained_until']} and promoted.", state


def forecast(model, y, horizon=HORIZON):
    rows = make_rows(y, [len(y) - 1], horizon)
    point = model.predict(rows)
    lo, hi = model.predict_interval(rows)
    lo, hi = np.minimum(lo, point), np.maximum(hi, point)    # the range always contains the forecast
    return pd.DataFrame({"date": rows["target_day"].dt.strftime("%Y-%m-%d"),
                         "weekday": rows["target_day"].dt.day_name(),
                         "holiday": rows["holiday"].astype(bool), "forecast_gwh": point.round(2),
                         "lower_gwh": lo.round(2), "upper_gwh": hi.round(2)})
