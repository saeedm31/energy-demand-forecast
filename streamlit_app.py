"""Streamlit demo: the household forecast, and live mode with Portugal's daily demand and temperature.

    streamlit run streamlit_app.py

Uses the same code as the Flask API (api.py); nothing is duplicated except the page layout.
"""

import json
import os

import pandas as pd
import streamlit as st

from api import live_version, svg_chart
from energy_forecast import live, service

st.set_page_config(page_title="Energy demand forecast", page_icon="⚡", layout="wide")

CHART_STYLE = """<style>
 .grid { stroke: rgba(128,128,128,.3); } .axis { fill: #8a8f98; font-size: 11px; font-family: sans-serif; }
 .hist { fill: none; stroke: #64748b; stroke-width: 2; }
 .fc { fill: none; stroke: #0d9488; stroke-width: 2.5; stroke-dasharray: 6 4; }
 .band { fill: #14b8a6; opacity: .2; } .now { stroke: #8a8f98; stroke-dasharray: 2 3; }
</style>"""

HOW_TO_READ_CHART = ("**How to read the chart:** the grey line is what was really used in the last 8 weeks. "
                     "The green dashed line is the forecast for the next 14 days. The green band is the 80% range: "
                     "on 8 days out of 10, the real value should land inside it.")


def chart(history, fc, step):
    """The same SVG chart as the Flask dashboard, shown as an image (Streamlit strips inline SVG from HTML)."""
    fc = fc.rename(columns=lambda c: c.replace("_gwh", "_kwh"))      # svg_chart expects *_kwh column names
    svg = svg_chart(history, fc, step=step).replace("<svg ", '<svg xmlns="http://www.w3.org/2000/svg" ', 1)
    svg = svg.replace(">", ">" + CHART_STYLE, 1)                       # styles inside the SVG itself
    st.image(svg, width="stretch")


def forecast_table(fc, unit, who):
    u = unit.lower()
    df = pd.DataFrame({
        "Day": pd.to_datetime(fc["date"]).dt.strftime("%a %d %b %Y") + fc["holiday"].map({True: " (holiday)", False: ""}),
        f"Forecast ({unit})": fc[f"forecast_{u}"],
        f"Likely low ({unit})": fc[f"lower_{u}"],
        f"Likely high ({unit})": fc[f"upper_{u}"],
    })
    if "temp_c" in fc:
        df["Temperature forecast (°C)"] = fc["temp_c"]
    st.dataframe(df, hide_index=True, width="stretch")
    st.caption(f"**How to read the table:** each row is one future day (not a customer). *Forecast* is the "
               f"electricity {who} is expected to use that day. *Likely low* and *likely high* are the 80% range.")


@st.cache_resource(show_spinner="Training the forecast model (a few seconds)...")
def household():
    y, imputed, model = service.load_all()
    return y, service.forecast(model, y)


@st.cache_resource(show_spinner="Training the live model on the collected days (about a minute)...")
def live_view(version):
    y, imputed = live.load()
    state = live.read_state()
    weather = live.load_weather()
    info = live.status(y, imputed, state)
    model = live.current_model(y, imputed, state, weather=weather)
    if model is None:
        return y, state, info, weather, None, None
    return (y, state, info, weather, live.live_evaluation(model, y, imputed, state["trained_until"], weather=weather),
            live.forecast(model, y, weather=weather))


st.title("⚡ Energy demand forecast")
st.write("This app predicts **how much electricity will be used each day for the next 14 days**, and how sure it is. "
         "Two examples: **one home near Paris** (a public dataset from 2006 to 2010) and **all of Portugal** "
         "(real data, updated every morning). "
         "[Code and docs on GitHub](https://github.com/saeedm31/energy-demand-forecast)")

home_tab, live_tab, how_tab = st.tabs(["🏠 One home near Paris (2006-2010)", "🔴 Portugal, live every day",
                                       "🧠 How it works"])

with home_tab:
    y, fc = household()
    first, last = y.index[0], y.index[-1]
    st.info(f"""**About this data**
- **Where:** one house in Sceaux, near Paris, France.
- **What:** the electricity this home used each day, in kWh, from its smart meter.
- **When:** {first:%d %b %Y} to {last:%d %b %Y} ({len(y):,} days). A public research dataset (UCI, EDF R&D).
- **Trained on:** this home's own history. Inputs: the last days and weeks, the weekday, the season and French
  public holidays. No weather data for this home.
- **Forecast:** because the data ends in {last:%B %Y}, the forecast is for the 14 days after {last:%d %b %Y},
  as if today were that day.""")
    if os.path.exists("reports/metrics.json"):
        with open("reports/metrics.json") as f:
            m = json.load(f)
        gb = next(s for s in m["scores"] if s["model"] == "gradient_boosting")
        st.subheader("How good is it?")
        c = st.columns(4)
        c[0].metric("Average error per day", f'{gb["mae_kwh"]:.1f} kWh',
                    help="On average, the forecast was this far from what the home really used. The home uses about 26 kWh a day.")
        c[1].metric("Better than a simple guess", f'{gb["skill_vs_naive_pct"]:.0f}%',
                    help='The simple guess is "same as the same weekday last week". The model makes this much less error.')
        c[2].metric("Days inside the 80% range", f'{gb["interval_coverage_pct"]:.0f}%',
                    help="The goal is 80%: the range should be right 8 days out of 10.")
        c[3].metric("Test forecasts", m["backtest"]["weeks"],
                    help="One forecast every week of the last year, each made using only the past.")
        st.caption(f'Tested on {m["backtest"]["test_period"][0]} to {m["backtest"]["test_period"][1]}: every week the '
                   "model was trained only on earlier days, then compared with what really happened.")
    st.subheader("The next 14 days")
    chart(y.iloc[-56:], fc, step=10)
    st.caption(HOW_TO_READ_CHART + " Unit: kWh per day for this one home.")
    forecast_table(fc, "kWh", "this home")

with live_tab:
    if not live.is_on():
        st.info("""**Live mode is off.** When it is on, this tab shows real data for **all of Portugal**:
- the electricity used in the whole country each day (from Energy-Charts, Fraunhofer ISE),
- the average temperature (Lisbon, Porto, Coimbra and Faro, from Open-Meteo),
- a 14-day forecast from today, and how well the model does on each new day.

New data arrives every morning. The model is retrained only when a person starts it, after at least 28 new days.""")
    else:
        y, state, info, weather, scores, lfc = live_view(live_version())
        trained = info["trained_until"]
        temp_line = ("the average of Lisbon, Porto, Coimbra and Faro (Open-Meteo): measured for past days, the "
                     "weather forecast for the next days. Cold days mean more heating, so more electricity."
                     if weather is not None else "not collected yet.")
        trained_line = (f"Portugal's data up to {trained}. Every day after that is new to the model and is used "
                        "to test it." if trained else "no model yet.")
        st.info(f"""**About this data**
- **Where:** all of Portugal: the electricity used in the whole country each day, in GWh (1 GWh = 1 million kWh).
- **From:** Energy-Charts (Fraunhofer ISE), downloaded automatically every morning.
  {info['data_first_day']} to {info['data_last_day']} ({info['days']:,} days).
- **Temperature:** {temp_line}
- **Trained on:** {trained_line}
- **Forecast:** the next 14 days after {info['data_last_day']}.""")
        c = st.columns(4)
        c[0].metric("Last day collected", info["data_last_day"])
        c[1].metric("Model trained on data up to", trained or "not yet")
        c[2].metric("New days since training",
                    info["new_days_since_training"] if info["new_days_since_training"] is not None else "–",
                    help="These days were not used for training, so they test the model honestly.")
        c[3].metric("Retrain available", "yes" if info["retrain_ready"] else f'after {info["next_retrain_after"]}',
                    help=f'A person can start a new training after {info["window_days"]} new days.')
        if lfc is None:
            st.warning("No model yet. Start the first training with the **Live retrain (manual)** workflow.")
            st.line_chart(y.iloc[-120:], y_label="GWh per day")
        else:
            if scores.get("forecasts_scored"):
                st.subheader("How good is it on days it has not seen?")
                c = st.columns(4)
                c[0].metric("Average error per day", f'{scores["mae_gwh"]:.1f} GWh',
                            f'{scores["mape_pct"]:.1f}% of the day', delta_color="off",
                            help="On average, how far the forecast was from the real value.")
                c[1].metric("Better than a simple guess", f'{scores["skill_vs_naive_pct"]:.0f}%',
                            help='The simple guess is "same as the same weekday last week".')
                c[2].metric("Days inside the 80% range", f'{scores["interval_coverage_pct"]:.0f}%',
                            help="The goal is 80%.")
                c[3].metric("New days tested", scores["target_days"])
            else:
                st.caption("The model was just trained; test scores appear as new days arrive.")
            st.subheader("The next 14 days")
            chart(y.iloc[-56:], lfc, step=25)
            st.caption(HOW_TO_READ_CHART + " Unit: GWh per day for all of Portugal.")
            forecast_table(lfc, "GWh", "all of Portugal")
        if weather is not None:
            st.subheader("Temperature")
            measured, future = weather.outlook(y.index[-1] + pd.Timedelta(days=1))
            t = pd.concat([measured[measured.index >= y.index[-1] - pd.Timedelta(days=55)].rename("Measured (°C)"),
                           future.rename("Forecast (°C)")], axis=1)
            st.line_chart(t, y_label="°C, daily mean")
            st.caption("Average daily temperature in Portugal (Lisbon, Porto, Coimbra, Faro, weighted by population). "
                       "The forecast part is the weather forecast the model uses for the next days.")
        if state.get("history"):
            with st.expander("Training runs (each one passed or failed a quality check)"):
                runs = [{"run": h["run_at"][:16].replace("T", " "), "data up to": h["trained_until"],
                         "window (days)": h["window_days"], "used": h["promoted"],
                         "check: error (GWh)": h["gate"].get("mae_gwh"),
                         "check: better than simple guess (%)": h["gate"].get("skill_vs_naive_pct"),
                         "check: days in 80% range (%)": h["gate"].get("interval_coverage_pct")}
                        for h in reversed(state["history"])]
                st.dataframe(pd.DataFrame(runs), hide_index=True, width="stretch")
    st.caption(live.ATTRIBUTION + " · " + live.wx.ATTRIBUTION)

with how_tab:
    st.markdown("""
#### Data sources

| | One home near Paris | Portugal (live) |
|---|---|---|
| What | electricity used by one house, kWh per day | electricity used by the whole country, GWh per day |
| Where from | UCI dataset, EDF R&D (public, CC BY 4.0) | Energy-Charts, Fraunhofer ISE (CC BY 4.0) |
| Dates | Dec 2006 to Nov 2010 | Jan 2019 to yesterday, a new day every morning |
| Weather | none | daily temperature: Lisbon, Porto, Coimbra, Faro (Open-Meteo, CC BY 4.0) |
| Holidays | French | Portuguese |

#### How the forecast is made
- **One model for all 14 days.** Gradient boosting learns from recent days, the weekday, the season, holidays and,
  for Portugal, the temperature.
- **No peeking at the future.** Each forecast only uses data up to its own day. A test checks this.
- **An honest range.** The 80% range is checked on recent data and widened until it is right about 8 days out of 10.
- **Tested like real life.** The model is trained on the past only, then compared with what really happened.
  It must beat the simple guess "same as the same weekday last week".
- **Live mode.** New data every morning. The model stays fixed and is tested on every new day. A person starts a
  new training after at least 28 new days, and the new model is used only if it passes a quality check.
- **Safe AI agent workflow.** Code changes can come from the GitHub Copilot agent; rules, tests and automatic review
  run on every pull request, and a person merges.

[Architecture](https://github.com/saeedm31/energy-demand-forecast/blob/main/docs/architecture.md) ·
[Model card](https://github.com/saeedm31/energy-demand-forecast/blob/main/docs/model_card.md) ·
[Data card](https://github.com/saeedm31/energy-demand-forecast/blob/main/docs/data_card.md)
""")
    if os.path.exists("docs/backtest.svg"):
        st.subheader("Test year for the home near Paris: forecast vs what really happened")
        st.image("docs/backtest.svg", width="stretch")

st.caption("Built by [Saeed Hojjatpanah](https://github.com/saeedm31).")
