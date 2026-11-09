"""Tests for the optional live mode. No network: the Energy-Charts API is replaced by a fake."""

import os
import tempfile
import unittest
from datetime import date
from unittest import mock

import pandas as pd

from energy_forecast import features, live, models

from .test_forecast import synthetic


def small_model():
    """A smaller boosting model keeps the tests fast."""
    return models.BoostingModel(max_iter=40)


def fake_payload(start, end, minutes=15, mw=6000.0, null_on=None, v1=False):
    """Energy-Charts-shaped response: one value every `minutes` from local midnight of start to end."""
    idx = pd.date_range(pd.Timestamp(start).tz_localize(live.TZ), pd.Timestamp(end).tz_localize(live.TZ),
                        freq=f"{minutes}min", inclusive="left")
    values = [None if (null_on and t.date() == null_on) else mw for t in idx]
    if v1:
        return {"unix_seconds": [int(t.timestamp()) for t in idx],
                "production_types": [{"name": "Load", "data": values}]}
    return {"schema_version": "2", "timezone": "Europe/Lisbon", "interval_minutes": minutes, "unit": "MW",
            "series": [{"id": "load", "name": "Load"}],
            "data": [{"timestamp": t.isoformat(), "values": {"load": v}} for t, v in zip(idx, values)],
            "license": "CC BY 4.0"}


def fake_get(url):
    q = dict(part.split("=") for part in url.split("?")[1].split("&"))
    return fake_payload(q["start"], q["end"])


def live_series(days=500, seed=0):
    """Synthetic national demand in GWh with a Portuguese calendar."""
    y, imputed = synthetic(days, seed)
    y = y * 5
    y.attrs["country"] = "pt"
    return y, imputed


class PortugueseHolidayTests(unittest.TestCase):
    def test_holidays_2024(self):
        h = features.portuguese_holidays(2024)
        self.assertIn(date(2024, 4, 25), h)    # Freedom Day
        self.assertIn(date(2024, 3, 29), h)    # Good Friday
        self.assertIn(date(2024, 5, 30), h)    # Corpus Christi
        self.assertIn(date(2024, 12, 8), h)
        self.assertEqual(len(h), 13)

    def test_series_country_picks_the_calendar(self):
        y, _ = synthetic(400)
        y_pt = y.copy()
        y_pt.attrs["country"] = "pt"
        # 2008-04-25 is a holiday in Portugal, not in France
        origin = y.index.get_loc(pd.Timestamp("2008-04-24"))
        fr = features.make_rows(y, [origin]).iloc[0]
        pt = features.make_rows(y_pt, [origin]).iloc[0]
        self.assertEqual((fr["holiday"], pt["holiday"]), (0, 1))


class ParseTests(unittest.TestCase):
    def test_daily_totals_and_incomplete_days(self):
        # 2024-03-31 is the spring clock change in Lisbon: 23 hours, 92 quarter-hours
        payload = fake_payload("2024-03-30", "2024-04-02", null_on=date(2024, 4, 1))
        daily = live.parse_energy_charts(payload)
        self.assertEqual(list(daily.index.date), [date(2024, 3, 30), date(2024, 3, 31)])
        self.assertEqual(daily.loc["2024-03-31", "points"], 92)
        self.assertAlmostEqual(daily.loc["2024-03-30", "gwh"], 6000 * 24 / 1000)
        self.assertAlmostEqual(daily.loc["2024-03-31", "gwh"], 6000 * 23 / 1000)

    def test_v1_format(self):
        daily = live.parse_energy_charts(fake_payload("2024-06-01", "2024-06-03", v1=True))
        self.assertEqual(len(daily), 2)
        self.assertAlmostEqual(daily["gwh"].iloc[0], 144.0)

    def test_missing_end_of_day_is_dropped(self):
        payload = fake_payload("2024-06-01", "2024-06-02")
        payload["data"] = payload["data"][:-4]
        self.assertEqual(len(live.parse_energy_charts(payload)), 0)

    def test_unknown_format(self):
        with self.assertRaises(ValueError):
            live.parse_energy_charts({"hello": 1})


class CollectTests(unittest.TestCase):
    def test_backfill_then_daily_update(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "pt.csv")
            with mock.patch.object(live, "BACKFILL_START", date(2024, 1, 1)), \
                 mock.patch.object(live, "CHUNK_DAYS", 20):
                data, problems = live.collect(path, today=date(2024, 2, 15), get=fake_get, pause=0, log=lambda m: None)
                self.assertEqual(data.index.min(), pd.Timestamp("2024-01-01"))
                self.assertEqual(data.index.max(), pd.Timestamp("2024-02-14"))   # never today
                self.assertEqual(len(data), 45)
                self.assertEqual(problems, [])
                calls = []
                data, _ = live.collect(path, today=date(2024, 2, 18),
                                       get=lambda u: calls.append(u) or fake_get(u), pause=0, log=lambda m: None)
                self.assertEqual(len(calls), 1)
                self.assertIn("start=2024-02-08", calls[0])                         # last 7 days refreshed
                self.assertEqual(data.index.max(), pd.Timestamp("2024-02-17"))
                self.assertFalse(data.index.duplicated().any())

    def test_values_outside_the_sane_range_are_not_saved(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "pt.csv")
            with mock.patch.object(live, "BACKFILL_START", date(2024, 1, 1)):
                data, problems = live.collect(path, today=date(2024, 1, 5), pause=0, log=lambda m: None,
                                              get=lambda u: fake_payload("2024-01-01", "2024-01-05", mw=50.0))
            self.assertEqual(len(data), 0)
            self.assertEqual(len(problems), 4)

    def test_rate_limit_is_retried(self):
        import urllib.error
        calls = []

        def flaky(url):
            calls.append(url)
            if len(calls) == 1:
                raise urllib.error.HTTPError(url, 429, "Too Many Requests", {"Retry-After": "0"}, None)
            return {"ok": True}
        with mock.patch.object(live.time, "sleep"):
            self.assertEqual(live.fetch(date(2024, 1, 1), date(2024, 1, 2), get=flaky), {"ok": True})
        self.assertEqual(len(calls), 2)


class LoadTests(unittest.TestCase):
    def write(self, d, days):
        path = os.path.join(d, "pt.csv")
        pd.DataFrame({"gwh": 150.0}, index=pd.DatetimeIndex(days, name="date")).to_csv(path)
        return path

    def test_short_gaps_are_filled_and_flagged(self):
        with tempfile.TemporaryDirectory() as d:
            days = pd.date_range("2024-01-01", "2024-02-29").delete([40, 41])
            y, imputed = live.load(self.write(d, days))
            self.assertEqual(len(y), 60)
            self.assertEqual(int(imputed.sum()), 2)
            self.assertFalse(y.isna().any())
            self.assertEqual(y.attrs["country"], "pt")

    def test_long_gap_stops(self):
        with tempfile.TemporaryDirectory() as d:
            days = pd.date_range("2024-01-01", "2024-02-29").delete(list(range(20, 30)))
            with self.assertRaises(ValueError):
                live.load(self.write(d, days))


class LiveCycleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.y, cls.imputed = live_series(500)
        cls.small = staticmethod(small_model)

    def test_live_test_only_scores_days_after_the_cutoff(self):
        cutoff = self.y.index[470]
        with mock.patch("energy_forecast.live.new_model", self.small):
            model = live.fit_until(self.y, self.imputed, cutoff)
        self.assertEqual(model.trained_until, cutoff)
        scores = live.live_evaluation(model, self.y, self.imputed, cutoff)
        last = len(self.y) - 1
        expected = sum(min(features.HORIZON, last - o) for o in range(470, last))
        self.assertEqual(scores["forecasts_scored"], expected)
        self.assertEqual(scores["target_days"], last - 470)
        self.assertEqual(scores["days_since_training"], 29)
        self.assertGreater(scores["skill_vs_naive_pct"], 0)

    def test_window_rules(self):
        state = {"trained_until": str(self.y.index[-10].date()), "window_days": 28, "history": []}
        info = live.status(self.y, self.imputed, state, today=date(2009, 5, 16))
        self.assertEqual(info["new_days_since_training"], 9)
        self.assertFalse(info["retrain_ready"])
        promoted, msg, _ = live.retrain(self.y, self.imputed, state)
        self.assertFalse(promoted)
        self.assertIn("Only 9 new days", msg)
        with self.assertRaises(ValueError):
            live.retrain(self.y, self.imputed, state, window_days=3)
        self.assertTrue(live.status(self.y, self.imputed, dict(state, window_days=7))["retrain_ready"])

    def test_retrain_promotes_only_through_the_gate(self):
        state = {"trained_until": str(self.y.index[-30].date()), "window_days": 28, "history": []}
        passed = lambda y, i: (True, {"checks": {"beats_baseline": True}})          # noqa: E731
        failed = lambda y, i: (False, {"checks": {"beats_baseline": False}})        # noqa: E731
        with tempfile.TemporaryDirectory() as d, \
             mock.patch("energy_forecast.live.new_model", self.small):
            path = os.path.join(d, "m.joblib")
            ok, msg, new = live.retrain(self.y, self.imputed, state, gate_fn=failed, model_path=path)
            self.assertFalse(ok)
            self.assertIn("beats_baseline", msg)
            self.assertEqual(new["trained_until"], state["trained_until"])          # old model stays
            self.assertEqual(len(new["history"]), 1)
            self.assertIn("mae_gwh", new["history"][0]["previous_model_live"])
            ok, msg, new = live.retrain(self.y, self.imputed, new, gate_fn=passed, model_path=path)
            self.assertTrue(ok)
            self.assertEqual(new["trained_until"], str(self.y.index[-1].date()))
            self.assertEqual(len(new["history"]), 2)

    def test_gate_checks(self):
        fast = lambda: [models.SeasonalNaive(), small_model()]   # noqa: E731
        ok, result = live.gate(self.y, self.imputed, weeks=2, make_models=fast)
        self.assertEqual(set(result["checks"]), {"beats_baseline", "interval_honest", "recent_data_complete"})
        self.assertTrue(result["checks"]["beats_baseline"])


class LiveApiTests(unittest.TestCase):
    def test_off_by_default_and_on_with_data(self):
        import api
        with tempfile.TemporaryDirectory() as d:
            paths = {"DATA_PATH": os.path.join(d, "pt.csv"), "STATE_PATH": os.path.join(d, "state.json"),
                     "MODEL_PATH": os.path.join(d, "m.joblib")}
            with mock.patch.multiple(live, **paths), \
                 mock.patch("energy_forecast.live.new_model", small_model):
                api.live_view.cache_clear()
                c = api.app.test_client()
                self.assertEqual(c.get("/api/live").status_code, 404)
                self.assertIn("Live mode is off", c.get("/live").get_json()["error"])

                y, _ = live_series(450)
                y.rename("gwh").to_frame().to_csv(paths["DATA_PATH"], index_label="date")
                body = c.get("/api/live").get_json()
                self.assertTrue(body["status"]["retrain_ready"])
                self.assertEqual(body["days"], [])                                   # no model yet

                live.write_state({"trained_until": str(y.index[-20].date()), "window_days": 28, "history": []},
                                 paths["STATE_PATH"])
                body = c.get("/api/live").get_json()
                self.assertEqual(len(body["days"]), features.HORIZON)
                self.assertEqual(body["status"]["new_days_since_training"], 19)
                self.assertGreater(body["live_test"]["forecasts_scored"], 0)
                page = c.get("/live")
                self.assertEqual(page.status_code, 200)
                self.assertIn(b"<svg", page.data)
                self.assertIn(b"energy-charts.info", page.data)
                api.live_view.cache_clear()


if __name__ == "__main__":
    unittest.main()
