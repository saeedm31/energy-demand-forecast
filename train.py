"""Backtest every model, log the runs, and train the final model.

Usage:
    python train.py              # full 52-week backtest (a few minutes)
    python train.py --weeks 8    # quick check

Writes reports/metrics.json and reports/backtest.csv, logs one run per model to
MLflow (or runs/runs.jsonl without MLflow), and saves models/model.joblib.
"""

import argparse
import json
import os

import joblib
import pandas as pd

from energy_forecast import backtest, data, models
from energy_forecast.backtest import training_rows
from energy_forecast.tracking import Tracker


def main(weeks):
    y, imputed = data.load()
    print(f"Data: {len(y)} days, {y.index.min().date()} to {y.index.max().date()}")

    results = backtest.backtest(y, imputed, models.all_models(), test_weeks=weeks)
    scores = backtest.score(results)
    by_h = backtest.mae_by_horizon(results)
    print(scores.round(2).to_string(index=False))

    os.makedirs("reports", exist_ok=True)
    results.to_csv("reports/backtest.csv", index=False)
    report = {
        "data": {"first_day": str(y.index.min().date()), "last_day": str(y.index.max().date()),
                 "days": len(y), "imputed_days": int(imputed.sum())},
        "backtest": {"weeks": weeks, "horizon_days": backtest.HORIZON,
                     "test_period": [str(results["target_day"].min().date()),
                                     str(results["target_day"].max().date())]},
        "scores": json.loads(scores.round(3).to_json(orient="records")),  # NaN -> null
        "mae_by_horizon": {str(h): row.round(3).to_dict() for h, row in by_h.iterrows()},
    }
    with open("reports/metrics.json", "w") as f:
        json.dump(report, f, indent=2)

    tracker = Tracker()
    tags = {"test_weeks": str(weeks), "data_last_day": report["data"]["last_day"]}
    for model in models.all_models():
        row = scores[scores["model"] == model.name].iloc[0].drop("model").to_dict()
        tracker.log_run(model.name, model.params, row, tags,
                        artifacts=["reports/metrics.json", "reports/backtest.csv"])
    print(f"Logged {len(scores)} runs to {tracker.backend}")

    final = models.BoostingModel()
    rows = training_rows(y, len(y) - 1)
    days = pd.DatetimeIndex(rows["target_day"])
    keep = ~imputed.reindex(days).to_numpy()
    final.fit(rows[keep], y.reindex(days).to_numpy()[keep])
    os.makedirs("models", exist_ok=True)
    final.trained_until = y.index[-1]
    joblib.dump(final, "models/model.joblib")
    print("Saved models/model.joblib")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--weeks", type=int, default=52, help="number of weekly forecasts to backtest")
    main(parser.parse_args().weeks)
