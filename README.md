# ⚡ Energy Demand Forecast

**Forecast a household's electricity use for the next 14 days, with honest uncertainty, on 4 years of real smart-meter data.**

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/ML-scikit--learn-F7931E?logo=scikitlearn)](https://scikit-learn.org/)
[![MLflow](https://img.shields.io/badge/Tracking-MLflow-0194E2?logo=mlflow)](https://mlflow.org/)
[![Flask](https://img.shields.io/badge/API-Flask-000000?logo=flask)](https://flask.palletsprojects.com/)
[![Tests](https://img.shields.io/badge/tests-passing-brightgreen)](#-tests)
[![License: MIT](https://img.shields.io/badge/License-MIT-green)](LICENSE)

---

## 💡 The problem

Energy suppliers, grid operators and smart-home systems all need to know **how much electricity will be used, and how sure they can be**. A forecast without uncertainty is hard to act on: you can't size a battery or plan a purchase on a single number.

This project forecasts daily consumption 1 to 14 days ahead and gives an 80% range that is checked to actually contain the true value about 80% of the time.

## 📈 Results

Measured with a **rolling-origin backtest**: every week of the final year, the models were retrained on all data up to that day, forecast the next 14 days, and were compared with what actually happened. That's 52 forecasts and 728 forecast days, never seen during training.

| Model | MAE (kWh/day) | RMSE | MAPE | Skill vs naive | 80% range coverage | Range width |
|---|---|---|---|---|---|---|
| Seasonal naive (baseline) | 6.26 | 8.14 | 27.2% | — | — | — |
| Ridge regression | 4.84 | 6.29 | 22.5% | 22.7% | — | — |
| **Gradient boosting + conformal range** | **4.38** | **5.82** | **18.8%** | **30.0%** | **77.6%** | 13.8 kWh |

Test period: 2009-11-20 to 2010-11-25. Average use is about 26 kWh/day.

- **Skill** is the error reduction against the naive baseline, "same weekday last week".
- **Coverage** is how often the true value fell inside the 80% range: the target is 80%.

**Error by days ahead** (MAE in kWh/day):

| Days ahead | 1 | 7 | 14 |
|---|---|---|---|
| Gradient boosting | 3.19 | 4.76 | 5.00 |
| Seasonal naive | 4.56 | 5.91 | 6.76 |

## 🧠 How it works

```
UCI smart-meter data (2M minute readings)
        │  prepare_data.py: daily kWh, outage days filled and flagged
        ▼
Features at forecast time ─ last 7 days, 4-week average, same weekday last week,
        │                   same week last year, weekday, season, French public holidays
        ▼
Models ─ seasonal naive (baseline) · Ridge · gradient boosting + quantile models
        │
        ▼
Conformal calibration ─ widen the quantile range until it really covers 80%
        │
        ▼
Rolling backtest ──► MLflow (or local log) ──► Flask API + dashboard
```

**Design choices worth noting**
- **One direct model for all horizons.** The forecast horizon (1–14 days) is a feature, so there's no error build-up from feeding forecasts back in as inputs.
- **No leakage, tested.** Features only use data up to the forecast day. A unit test changes every future value and checks the features don't move.
- **Outages handled honestly.** 21 days where the meter was mostly offline are filled in so the series has no gaps, but they're never used for training or scoring.
- **Calibrated uncertainty.** Raw quantile models tend to be too confident. Conformalized quantile regression ([Romano et al., 2019](https://arxiv.org/abs/1905.03222)) fixes this using the most recent 13 weeks of data.

## 🚀 Run it

```bash
git clone https://github.com/saeedm31/energy-demand-forecast.git
cd energy-demand-forecast
pip install -r requirements.txt

python train.py            # 52-week backtest, logs runs, saves the model (a few minutes)
python api.py              # dashboard at http://127.0.0.1:8000
```

The cleaned daily data is included (`data/daily_energy.csv`). To rebuild it from the raw UCI file:
`python prepare_data.py household_power_consumption.zip`

### Experiment tracking

Every backtest logs each model's parameters, metrics and reports as a run.
- **With MLflow:** `pip install mlflow`, run `python train.py`, then `mlflow ui` to compare runs.
- **Without MLflow:** runs are appended to `runs/runs.jsonl`.

### API

| Endpoint | Returns |
|---|---|
| `GET /` | Dashboard: last 8 weeks, the 14-day forecast and its 80% range |
| `GET /api/forecast` | Next 14 days as JSON: date, weekday, holiday flag, forecast, lower, upper |
| `GET /api/metrics` | Backtest results |
| `GET /health` | Model and data status |

**Docker:** `docker build -t energy-forecast . && docker run -p 8000:8000 energy-forecast`. The model is trained at build time.

## 📁 Project structure

```
├── prepare_data.py            # raw minute data → clean daily series
├── train.py                   # backtest, tracking, final model
├── api.py                     # Flask API and dashboard
├── energy_forecast/
│   ├── features.py            # leakage-free features, French holidays
│   ├── models.py              # baseline, Ridge, gradient boosting + conformal intervals
│   ├── backtest.py            # rolling-origin evaluation and metrics
│   ├── tracking.py            # MLflow or local run log
│   └── service.py             # forecast from the latest data
├── data/daily_energy.csv      # 1,440 days, 2006-12-17 to 2010-11-25
├── reports/metrics.json       # latest backtest results
└── tests/                     # unit tests
```

## ✅ Tests

```bash
python -m unittest
```

The tests cover:
- holiday dates
- **no future leakage** in the features
- the backtest never training on its test period
- models beating the baseline
- outage days excluded from scoring
- interval calibration
- the MLflow and local tracking paths
- the API

## ⚠️ Limitations and next steps

- One household, so the patterns are personal (holidays away, guests). Aggregated demand across many homes is smoother and easier to forecast.
- No weather input. Temperature is the main driver of heating demand, and adding a weather forecast would likely help most in winter.
- Next: weather features, hourly forecasts, a scheduled retraining job.

## 📚 Data

[Individual household electric power consumption](https://archive.ics.uci.edu/dataset/235/individual+household+electric+power+consumption), Georges Hébrail and Alice Bérard, EDF R&D. UCI Machine Learning Repository, CC BY 4.0. One house in Sceaux, near Paris, measured every minute from December 2006 to November 2010.

---

Built by [Saeed Hojjatpanah](https://github.com/saeedm31).
