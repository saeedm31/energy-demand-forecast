"""Tests for temperature in live mode. No network: Open-Meteo is replaced by a fake."""

import os
import tempfile
import unittest
from datetime import date
from unittest import mock

import numpy as np
import pandas as pd

from energy_forecast import backtest, features, live, models, weather

from .test_live import live_series, small_model


def fake_open_meteo(start, end, tmax=20.0, tmin=10.0):
    days = [d.strftime("%Y-%m-%d") for d in pd.date_range(start, end)]
    one = {"daily": {"time": days, "temperature_2m_max": [tmax] * len(days), "temperature_2m_min": [tmin] * len(days)}}
    return [one for _ in weather.CITIES]


class ParseTests(unittest.TestCase):
    def test_weighted_daily_mean(self):
        payload = fake_open_meteo("2024-01-01", "2024-01-03")
        payload[0] = {"daily": {"time": payload[0]["daily"]["time"], "temperature_2m_max": [30] * 3,
                                "temperature_2m_min": [20] * 3}}   # Lisbon (weight 0.5) is warmer
        temp = weather.parse(payload)
        self.assertEqual(len(temp), 3)
        self.assertAlmostEqual(temp.iloc[0], 0.5 * 25 + 0.5 * 15)

    def test_missing_city_value_drops_the_day(self):
        payload = fake_open_meteo("2024-01-01", "2024-01-02")
        payload[1] = {"daily": {"time": payload[1]["daily"]["time"], "temperature_2m_max": [None, 20],
                                "temperature_2m_min": [None, 10]}}
        self.assertEqual(list(weather.parse(payload).index.day), [2])

    def test_wrong_number_of_locations(self):
        with self.assertRaises(ValueError):
            weather.parse(fake_open_meteo("2024-01-01", "2024-01-02")[:2])


class CollectTests(unittest.TestCase):
    def test_backfill_then_daily_forecast_log(self):
        calls = []

        def get(url):
            calls.append(url)
            if "archive" in url:
                return fake_open_meteo("2024-01-01", "2024-02-04")
            return fake_open_meteo("2024-01-27", "2024-02-25", tmax=22, tmin=12)   # 14 past days + 16 ahead

        with tempfile.TemporaryDirectory() as d, mock.patch.object(weather, "BACKFILL_START", date(2024, 1, 1)):
            paths = dict(obs_path=os.path.join(d, "w.csv"), fc_path=os.path.join(d, "f.csv"))
            obs, fcs = weather.collect(date(2024, 2, 10), get=get, log=lambda m: None, **paths)
            self.assertEqual(len(calls), 2)
            self.assertEqual(obs.index.min(), pd.Timestamp("2024-01-01"))
            self.assertEqual(obs.index.max(), pd.Timestamp("2024-02-09"))           # measured: up to yesterday
            self.assertEqual(obs.loc["2024-02-09"], 17.0)                           # recent days replace the archive
            self.assertEqual(fcs["issued"].unique().tolist(), [pd.Timestamp("2024-02-10")])
            self.assertEqual(len(fcs), 16)
            weather.collect(date(2024, 2, 10), get=get, log=lambda m: None, **paths)   # same day again: no duplicates
            self.assertEqual(len(weather.read_forecasts(paths["fc_path"])), 16)
            self.assertEqual(len(calls), 3)                                         # no second backfill


class WeatherLookupTests(unittest.TestCase):
    def test_uses_the_forecast_known_at_the_time(self):
        obs = pd.Series([10.0, 11.0, 12.0], index=pd.date_range("2024-03-01", periods=3), name="temp_c")
        fcs = pd.DataFrame({"issued": pd.to_datetime(["2024-03-02", "2024-03-02"]),
                            "date": pd.to_datetime(["2024-03-03", "2024-03-04"]), "temp_c": [15.0, 16.0]})
        w = weather.Weather(obs, fcs)
        # origin 03-01: forecast issued 03-02 exists for 03-03; 03-02 has no forecast -> measured value
        t = w.temps(pd.to_datetime(["2024-03-01", "2024-03-01", "2024-03-02"]),
                    pd.to_datetime(["2024-03-03", "2024-03-02", "2024-03-04"]))
        self.assertEqual(list(t), [15.0, 11.0, 16.0])

    def test_features_and_models_use_temperature(self):
        y, imputed = live_series(500)
        temp = pd.Series(18 + 8 * np.cos(2 * np.pi * (y.index.dayofyear - 200) / 365.25), index=y.index)
        w = weather.Weather(temp)
        rows = features.make_rows(y, [300], weather=w)
        self.assertTrue(set(features.WEATHER_FEATURES) <= set(rows.columns))
        self.assertEqual(rows["hdd"].min(), 0)
        train = backtest.training_rows(y, 450, weather=w)
        target = y.reindex(pd.DatetimeIndex(train["target_day"])).to_numpy()
        for m in (models.RidgeModel(), small_model()):
            m.fit(train, target)
            self.assertIn("temp_c", m.cols)
            self.assertEqual(len(m.predict(rows)), features.HORIZON)
        plain = features.make_rows(y, [300])
        self.assertNotIn("temp_c", plain.columns)
        self.assertEqual(models.columns(plain), features.FEATURES)

    def test_retrain_and_forecast_with_temperature(self):
        y, imputed = live_series(500)
        w = weather.Weather(pd.Series(15.0, index=y.index))
        state = {"trained_until": None, "window_days": 28, "history": []}
        with tempfile.TemporaryDirectory() as d, mock.patch("energy_forecast.live.new_model", small_model):
            ok, _, new = live.retrain(y, imputed, state, gate_fn=lambda a, b: (True, {"checks": {}}),
                                      model_path=os.path.join(d, "m.joblib"), weather=w)
            self.assertTrue(ok)
            self.assertTrue(new["history"][-1]["uses_temperature"])
            model = live.current_model(y, imputed, new, path=os.path.join(d, "m.joblib"), weather=w)
            self.assertTrue(model.uses_weather)
            fc = live.forecast(model, y, weather=w)
            self.assertIn("temp_c", fc.columns)


if __name__ == "__main__":
    unittest.main()
