"""Produce the next 14-day forecast from the latest data."""

import os

import joblib
import pandas as pd

from . import data, models
from .backtest import training_rows
from .features import HORIZON, make_rows

MODEL_PATH = "models/model.joblib"


def load_or_train(y, imputed, path=None):
    """Use the model saved by train.py if it matches the data, otherwise train one now."""
    path = path or MODEL_PATH
    if os.path.exists(path):
        model = joblib.load(path)
        if getattr(model, "trained_until", None) == y.index[-1]:
            return model
    model = models.BoostingModel()
    rows = training_rows(y, len(y) - 1)
    keep = ~imputed.reindex(pd.DatetimeIndex(rows["target_day"])).to_numpy()
    model.fit(rows[keep], y.reindex(pd.DatetimeIndex(rows["target_day"])).to_numpy()[keep])
    model.trained_until = y.index[-1]
    return model


def forecast(model, y, horizon=HORIZON):
    """Forecast the days after the last observed day, with the 80% interval."""
    rows = make_rows(y, [len(y) - 1], horizon)
    point = model.predict(rows)
    lo, hi = model.predict_interval(rows)
    return pd.DataFrame({"date": rows["target_day"].dt.strftime("%Y-%m-%d"), "weekday": rows["target_day"].dt.day_name(),
                         "holiday": rows["holiday"].astype(bool), "forecast_kwh": point.round(2),
                         "lower_kwh": lo.round(2), "upper_kwh": hi.round(2)})


def load_all():
    y, imputed = data.load()
    return y, imputed, load_or_train(y, imputed)
