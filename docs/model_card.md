# Model card: daily electricity forecast

| | |
|---|---|
| **Task** | Forecast a household's daily electricity use 1 to 14 days ahead, with an 80% range |
| **Model** | scikit-learn `HistGradientBoostingRegressor` (absolute error) for the forecast, two quantile models (10th and 90th percentile) for the range, widened by conformalized quantile regression |
| **Baselines** | seasonal naive ("same weekday last week") and Ridge regression |
| **Data** | One household's smart meter, 2006 to 2010 (see [data card](data_card.md)) |
| **Owner** | Saeed Hojjatpanah |

## Features (known at forecast time only)

Horizon (1-14 days) · use on the last 3 days · 7- and 28-day averages · 7-day spread · same weekday last week ·
same week last year · weekday and weekend flag · season (day of year as sine and cosine) · French public holidays. A unit test changes every future value and checks that
no feature moves.

## Calibrated range

Quantile models tend to be too confident. The calibration step holds out the most recent 13 weeks, measures how
far outside the range the true values fall, and adjusts the range by that margin, so it covers about 80% of days.
Then all models are refit on all the data.

## Intended use
Planning for one home or a small portfolio: sizing a battery, timing flexible loads, a supplier's short-term view.

## Not intended for
Grid-level forecasts or billing. Not tested on other homes.

## Evaluation
Rolling-origin backtest over the final year: 52 weekly forecasts, 728 forecast days, the model retrained each time
on data up to that day only. Metrics: MAE, RMSE, MAPE, skill against the seasonal naive baseline, error by days
ahead, coverage and width of the 80% range. Outage days are not scored. Results are in the
[README](../README.md#-results) and `reports/metrics.json`.

## Known limits
- One household: holidays away and guests make sudden drops the model only sees once they start.
- No weather input, which would help most in winter.

## Monitoring (production plan)
Weekly: MAE and coverage of the last 4 weeks against the backtest. Retrain weekly; alert if coverage falls below
70% or skill against the naive baseline turns negative.
