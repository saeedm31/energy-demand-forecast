"""Rolling-origin backtest: replay the past as if forecasting live.

Every week over the test period, train on everything before that week, forecast
the next 14 days, then compare with what actually happened. Days that were
filled in during cleaning (meter outages) are never scored.
"""

import numpy as np
import pandas as pd

from .features import FIRST_ORIGIN, HORIZON, make_rows


def training_rows(y, last_origin, horizon=HORIZON):
    """All (origin, h) rows whose target day is on or before position `last_origin`."""
    origins = range(FIRST_ORIGIN, last_origin)
    rows = make_rows(y, origins, horizon)
    end = y.index[last_origin]
    return rows[rows["target_day"] <= end].reset_index(drop=True)


def backtest(y, imputed, models, test_weeks=52, horizon=HORIZON, step=7):
    """Return one row per (model, origin, horizon) with forecast, actual and interval if any."""
    n = len(y)
    first_test_origin = n - horizon - step * (test_weeks - 1) - 1
    origins = list(range(first_test_origin, n - horizon, step))
    out = []
    for t in origins:
        train = training_rows(y, t, horizon)
        y_train = y.reindex(pd.DatetimeIndex(train["target_day"])).to_numpy()
        keep = ~imputed.reindex(pd.DatetimeIndex(train["target_day"])).to_numpy()   # don't learn from filled-in days
        test = make_rows(y, [t], horizon)
        actual = y.reindex(pd.DatetimeIndex(test["target_day"])).to_numpy()
        is_imputed = imputed.reindex(pd.DatetimeIndex(test["target_day"])).to_numpy()
        for model in models:
            model.fit(train[keep], y_train[keep])
            rec = test[["origin", "target_day", "h"]].copy()
            rec["model"] = model.name
            rec["forecast"] = model.predict(test)
            rec["actual"] = actual
            rec["imputed"] = is_imputed
            if hasattr(model, "predict_interval"):
                rec["lower"], rec["upper"] = model.predict_interval(test)
            out.append(rec)
    return pd.concat(out, ignore_index=True)


def score(results):
    """Error metrics per model, on measured (non-imputed) days only."""
    r = results[~results["imputed"].astype(bool)]
    naive_mae = (r[r["model"] == "seasonal_naive"]["forecast"] - r[r["model"] == "seasonal_naive"]["actual"]).abs().mean()
    rows = []
    for name, g in r.groupby("model", sort=False):
        err = g["forecast"] - g["actual"]
        row = {
            "model": name,
            "mae_kwh": err.abs().mean(),
            "rmse_kwh": np.sqrt((err ** 2).mean()),
            "mape_pct": 100 * (err.abs() / g["actual"]).mean(),
            "skill_vs_naive_pct": 100 * (1 - err.abs().mean() / naive_mae),
        }
        if g["lower"].notna().any() if "lower" in g else False:
            inside = (g["actual"] >= g["lower"]) & (g["actual"] <= g["upper"])
            row["interval_coverage_pct"] = 100 * inside.mean()
            row["interval_width_kwh"] = (g["upper"] - g["lower"]).mean()
        rows.append(row)
    return pd.DataFrame(rows)


def mae_by_horizon(results):
    r = results[~results["imputed"].astype(bool)]
    return (r.assign(abs_err=(r["forecast"] - r["actual"]).abs())
             .pivot_table(index="h", columns="model", values="abs_err", aggfunc="mean"))
