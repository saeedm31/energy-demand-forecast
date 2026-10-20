"""Forecasting models, all with the same fit / predict interface on feature rows."""

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .features import FEATURES

RANDOM_STATE = 42
NUMERIC = [f for f in FEATURES if f != "dow"]


class SeasonalNaive:
    """Baseline: next week looks like the same weekday last week."""
    name = "seasonal_naive"
    params = {}

    def fit(self, X, y):
        return self

    def predict(self, X):
        return X["same_dow"].to_numpy()


class RidgeModel:
    """Linear model on scaled features, weekday one-hot encoded."""
    name = "ridge"

    def __init__(self, alpha=3.0):
        self.params = {"alpha": alpha}
        pre = ColumnTransformer([
            ("num", make_pipeline(SimpleImputer(), StandardScaler()), NUMERIC),
            ("dow", OneHotEncoder(handle_unknown="ignore"), ["dow"]),
        ])
        self.pipe = make_pipeline(pre, Ridge(alpha=alpha))

    def fit(self, X, y):
        self.pipe.fit(X[FEATURES], y)
        return self

    def predict(self, X):
        return self.pipe.predict(X[FEATURES])


class BoostingModel:
    """Gradient boosting for the forecast, plus a calibrated 80% prediction interval.

    The interval starts from two quantile models (10th and 90th percentile). Quantile
    models are often too confident, so the interval is then widened with conformalized
    quantile regression (Romano et al., 2019): measure how far outside the interval the
    most recent `calibration_days` of targets fell, using models that did not see them,
    and widen by that amount.
    """
    name = "gradient_boosting"

    def __init__(self, max_iter=300, learning_rate=0.05, max_leaf_nodes=15, coverage=0.8,
                 calibration_days=91):
        self.coverage, self.calibration_days = coverage, calibration_days
        self.params = {"max_iter": max_iter, "learning_rate": learning_rate, "max_leaf_nodes": max_leaf_nodes,
                       "coverage": coverage, "calibration_days": calibration_days}
        alpha = (1 - coverage) / 2
        common = dict(max_iter=max_iter, learning_rate=learning_rate, max_leaf_nodes=max_leaf_nodes,
                      categorical_features=[FEATURES.index("dow")], random_state=RANDOM_STATE)
        self.point = HistGradientBoostingRegressor(loss="absolute_error", **common)
        self.lower = HistGradientBoostingRegressor(loss="quantile", quantile=alpha, **common)
        self.upper = HistGradientBoostingRegressor(loss="quantile", quantile=1 - alpha, **common)
        self.margin = 0.0

    def fit(self, X, y):
        y = np.asarray(y)
        # 1. Calibrate: fit the quantile models without the most recent days, score them on those days.
        cutoff = X["target_day"].max() - np.timedelta64(self.calibration_days, "D")
        cal = (X["target_day"] > cutoff).to_numpy()
        if cal.sum() >= 50 and (~cal).sum() >= 200:
            self.lower.fit(X.loc[~cal, FEATURES], y[~cal])
            self.upper.fit(X.loc[~cal, FEATURES], y[~cal])
            lo, hi = self._raw_interval(X[cal])
            scores = np.maximum(lo - y[cal], y[cal] - hi)
            n = len(scores)
            level = min(1.0, np.ceil((n + 1) * self.coverage) / n)
            self.margin = float(np.quantile(scores, level))
        # 2. Refit everything on all the data for the final forecasts.
        for m in (self.point, self.lower, self.upper):
            m.fit(X[FEATURES], y)
        return self

    def predict(self, X):
        return self.point.predict(X[FEATURES])

    def _raw_interval(self, X):
        lo, hi = self.lower.predict(X[FEATURES]), self.upper.predict(X[FEATURES])
        return np.minimum(lo, hi), np.maximum(lo, hi)

    def predict_interval(self, X):
        lo, hi = self._raw_interval(X)
        return np.maximum(lo - self.margin, 0.0), hi + self.margin


def all_models():
    return [SeasonalNaive(), RidgeModel(), BoostingModel()]
