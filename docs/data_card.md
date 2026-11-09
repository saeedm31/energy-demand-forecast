# Data card: household smart-meter data

| | |
|---|---|
| **Source** | [Individual household electric power consumption](https://archive.ics.uci.edu/dataset/235/individual+household+electric+power+consumption), Georges Hébrail and Alice Bérard, EDF R&D, UCI Machine Learning Repository |
| **Licence** | CC BY 4.0 |
| **What** | One house in Sceaux, near Paris, measured every minute from December 2006 to November 2010 (about 2 million readings) |
| **In the repo** | The cleaned daily series, `data/daily_energy.csv`: 1,440 days, 2006-12-17 to 2010-11-25 |
| **Rebuild** | `python prepare_data.py household_power_consumption.zip` |
| **Personal data** | An anonymous household; no names or addresses |

## Preparation

- Minute readings of active power are summed into daily kWh.
- 21 days where the meter was mostly offline are filled so the series has no gaps, flagged, and never used for
  training or scoring.
- Average use is about 26 kWh a day, with clear winter peaks and drops when the household is away.

## Split

Rolling origin: every week of the final year (2009-11-20 to 2010-11-25), training uses all data up to that day and
the next 14 days are scored.

## Limits
One household, so the patterns are personal. Aggregated demand across many homes is smoother and easier to forecast.

## Live mode data (optional)

| | |
|---|---|
| Source | [Energy-Charts](https://www.energy-charts.info) public power API, Fraunhofer ISE |
| Series | `load`, Portugal, 15-minute values in MW |
| Licence | CC BY 4.0, credit "energy-charts.info" (shown on the dashboard) |
| Stored | `data/live/pt_daily_demand.csv`: one row per Lisbon day, GWh |
| Checks | every interval present (23 or 25 hours on clock-change days), no nulls, 40-300 GWh, no duplicates |
| Gaps | up to 7 missing days are filled with the same weekday a week earlier and flagged; never trained on or scored |
| Personal data | none: national totals |

## Live mode temperature (optional)

| | |
|---|---|
| Source | [Open-Meteo](https://open-meteo.com): historical archive (ERA5) and forecast API; no key |
| Licence | CC BY 4.0, non-commercial use; credit "Open-Meteo.com" (shown on the dashboard) |
| What | daily mean temperature, (max + min) / 2, of Lisbon 50%, Porto 30%, Coimbra 10%, Faro 10% (rough population weights) |
| Stored | `data/live/pt_weather.csv` (measured, from 2019) and `data/live/pt_weather_forecasts.csv` (every 16-day forecast, as issued) |
| Checks | all four cities present, -15 to 45 °C |
| Honesty | forecasts use the weather forecast stored on that morning; before collection started, measured temperature stands in |
