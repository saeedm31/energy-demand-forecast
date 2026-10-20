"""Tests. Run with: python -m unittest"""

import json
import os
import sys
import tempfile
import types
import unittest
from datetime import date
from unittest import mock

import numpy as np
import pandas as pd

from energy_forecast import backtest, features, models
from energy_forecast.tracking import Tracker


def synthetic(days=500, seed=0):
    """Weekly + yearly pattern with noise, like household energy use."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2008-01-01", periods=days, freq="D")
    weekly = np.where(idx.dayofweek >= 5, 6.0, 0.0)
    yearly = 8 * np.cos(2 * np.pi * (idx.dayofyear - 15) / 365.25)
    y = pd.Series(25 + weekly + yearly + rng.normal(0, 2, days), index=idx)
    return y, pd.Series(False, index=idx)


class HolidayTests(unittest.TestCase):
    def test_easter_dates(self):
        self.assertEqual(features.easter(2010), date(2010, 4, 4))
        self.assertEqual(features.easter(2024), date(2024, 3, 31))

    def test_french_holidays_include_moving_ones(self):
        h = features.french_holidays(2010)
        self.assertIn(date(2010, 4, 5), h)    # Easter Monday
        self.assertIn(date(2010, 5, 13), h)   # Ascension
        self.assertIn(date(2010, 7, 14), h)   # Bastille Day
        self.assertEqual(len(h), 11)


class FeatureTests(unittest.TestCase):
    def test_no_future_leakage(self):
        """Changing anything after the origin must not change that origin's features."""
        y, _ = synthetic()
        t = 300
        before = features.make_rows(y, [t])
        y2 = y.copy()
        y2.iloc[t + 1:] = -999
        after = features.make_rows(y2, [t])
        pd.testing.assert_frame_equal(before, after)

    def test_same_weekday_lag(self):
        y, _ = synthetic()
        rows = features.make_rows(y, [300])
        for h in (1, 7, 8, 14):
            row = rows[rows["h"] == h].iloc[0]
            k = -(-h // 7)
            self.assertEqual(row["same_dow"], y.iloc[300 + h - 7 * k])
            self.assertEqual(pd.Timestamp(row["target_day"]).dayofweek,
                             y.index[300 + h - 7 * k].dayofweek)

    def test_one_row_per_horizon(self):
        y, _ = synthetic()
        rows = features.make_rows(y, [100, 200])
        self.assertEqual(len(rows), 2 * features.HORIZON)
        self.assertTrue(set(features.FEATURES) <= set(rows.columns))


class BacktestTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        y, imputed = synthetic()
        imputed.iloc[-10] = True   # a filled-in day inside the test period
        cls.y, cls.imputed = y, imputed
        fast = [models.SeasonalNaive(), models.RidgeModel(), models.BoostingModel(max_iter=60)]
        cls.results = backtest.backtest(y, imputed, fast, test_weeks=4)

    def test_shape(self):
        self.assertEqual(len(self.results), 3 * 4 * features.HORIZON)

    def test_training_never_sees_the_test_period(self):
        origin = len(self.y) - 30
        train = backtest.training_rows(self.y, origin)
        self.assertLessEqual(train["target_day"].max(), self.y.index[origin])

    def test_models_beat_the_naive_baseline(self):
        s = backtest.score(self.results).set_index("model")
        self.assertLess(s.loc["gradient_boosting", "mae_kwh"], s.loc["seasonal_naive", "mae_kwh"])
        self.assertLess(s.loc["ridge", "mae_kwh"], s.loc["seasonal_naive", "mae_kwh"])
        self.assertIn("interval_coverage_pct", s.columns)

    def test_imputed_days_are_not_scored(self):
        scored = backtest.score(self.results.assign(forecast=self.results["forecast"].where(
            ~self.results["imputed"].astype(bool), 1e6)))
        self.assertTrue((scored["mae_kwh"] < 100).all())


class IntervalTests(unittest.TestCase):
    def test_conformal_interval_contains_forecast_and_is_calibrated(self):
        y, imputed = synthetic(days=700, seed=1)
        model = models.BoostingModel(max_iter=100)
        train = backtest.training_rows(y, 600)
        model.fit(train, y.reindex(pd.DatetimeIndex(train["target_day"])).to_numpy())
        self.assertGreaterEqual(model.margin, -5)
        test = features.make_rows(y, list(range(600, 680)))
        actual = y.reindex(pd.DatetimeIndex(test["target_day"])).to_numpy()
        lo, hi = model.predict_interval(test)
        point = model.predict(test)
        self.assertTrue(np.all(lo <= hi))
        coverage = np.mean((actual >= lo) & (actual <= hi))
        self.assertGreater(coverage, 0.65)   # target 80%; loose bound for a small sample
        self.assertGreater(np.mean((point >= lo) & (point <= hi)), 0.9)


class TrackerTests(unittest.TestCase):
    def test_local_log(self):
        with tempfile.TemporaryDirectory() as d:
            t = Tracker(local_dir=d, use_mlflow=False)
            t.log_run("ridge", {"alpha": 3}, {"mae_kwh": 4.2, "nan_metric": float("nan")}, {"x": "1"})
            rec = json.loads(open(os.path.join(d, "runs.jsonl")).read())
            self.assertEqual(rec["metrics"], {"mae_kwh": 4.2})
            self.assertEqual(rec["params"], {"alpha": 3})

    def test_mlflow_calls(self):
        fake = types.SimpleNamespace(calls=[])
        for name in ("set_experiment", "log_params", "log_metrics", "set_tags", "log_artifact"):
            setattr(fake, name, (lambda n: lambda *a, **k: fake.calls.append(n))(name))

        class Run:
            def __enter__(self): fake.calls.append("start_run")
            def __exit__(self, *a): fake.calls.append("end_run")
        fake.start_run = lambda **k: Run()
        with mock.patch.dict(sys.modules, {"mlflow": fake}):
            t = Tracker()
            self.assertTrue(t.use_mlflow)
            t.log_run("gb", {"a": 1}, {"mae": 1.0}, {"t": "x"}, artifacts=["f.json"])
        self.assertEqual(fake.calls, ["set_experiment", "start_run", "log_params", "log_metrics",
                                      "set_tags", "log_artifact", "end_run"])


class ApiTests(unittest.TestCase):
    def test_endpoints(self):
        import api
        y, imputed = synthetic(days=420)
        real = models.BoostingModel
        small = lambda: real(max_iter=60)  # noqa: E731  (smaller model keeps the test fast)
        with mock.patch("energy_forecast.data.load", return_value=(y, imputed)), \
             mock.patch("energy_forecast.service.MODEL_PATH", "/nonexistent.joblib"), \
             mock.patch("energy_forecast.models.BoostingModel", small):
            api.state.cache_clear()
            c = api.app.test_client()
            fc = c.get("/api/forecast").get_json()
            self.assertEqual(len(fc["days"]), features.HORIZON)
            first = fc["days"][0]
            self.assertLessEqual(first["lower_kwh"], first["upper_kwh"])
            self.assertEqual(c.get("/health").get_json()["days"], 420)
            page = c.get("/")
            self.assertEqual(page.status_code, 200)
            self.assertIn(b"<svg", page.data)
        api.state.cache_clear()


if __name__ == "__main__":
    unittest.main()
