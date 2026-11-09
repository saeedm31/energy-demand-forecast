"""Streamlit demo: the household forecast, and live mode with Portugal's daily demand.

    streamlit run streamlit_app.py

Uses the same code as the Flask API (api.py); nothing is duplicated except the page layout.
"""

import json
import os

import pandas as pd
import streamlit as st

from api import svg_chart
from energy_forecast import live, service

st.set_page_config(page_title="Energy demand forecast", page_icon="⚡", layout="wide")

CHART_STYLE = """<style>
 .grid { stroke: rgba(128,128,128,.3); } .axis { fill: #8a8f98; font-size: 11px; font-family: sans-serif; }
 .hist { fill: none; stroke: #64748b; stroke-width: 2; }
 .fc { fill: none; stroke: #0d9488; stroke-width: 2.5; stroke-dasharray: 6 4; }
 .band { fill: #14b8a6; opacity: .2; } .now { stroke: #8a8f98; stroke-dasharray: 2 3; }
</style>"""


def chart(history, fc, step):
    """The same SVG chart as the Flask dashboard, shown as an image (Streamlit strips inline SVG from HTML)."""
    fc = fc.rename(columns=lambda c: c.replace("_gwh", "_kwh"))      # svg_chart expects *_kwh column names
    svg = svg_chart(history, fc, step=step).replace("<svg ", '<svg xmlns="http://www.w3.org/2000/svg" ', 1)
    svg = svg.replace(">", ">" + CHART_STYLE, 1)                       # styles inside the SVG itself
    st.image(svg, width="stretch")


def forecast_table(fc, unit):
    df = pd.DataFrame({
        "Date": fc["date"],
        "Day": fc["weekday"] + fc["holiday"].map({True: " (holiday)", False: ""}),
        f"Forecast ({unit})": fc[f"forecast_{unit.lower()}"],
        "80% range": [f"{lo:.1f} – {hi:.1f}" for lo, hi in zip(fc[f"lower_{unit.lower()}"], fc[f"upper_{unit.lower()}"])],
    })
    st.dataframe(df, hide_index=True, width="stretch")


@st.cache_resource(show_spinner="Training the forecast model (a few seconds)...")
def household():
    y, imputed, model = service.load_all()
    return y, service.forecast(model, y)


@st.cache_resource(show_spinner="Training the live model on the collected days (about a minute)...")
def live_view(version):
    y, imputed = live.load()
    state = live.read_state()
    info = live.status(y, imputed, state)
    model = live.current_model(y, imputed, state)
    if model is None:
        return y, state, info, None, None
    return y, state, info, live.live_evaluation(model, y, imputed, state["trained_until"]), live.forecast(model, y)


def live_version():
    return tuple(os.path.getmtime(p) if os.path.exists(p) else 0 for p in (live.DATA_PATH, live.STATE_PATH))


st.title("⚡ Energy demand forecast")
st.caption("Daily electricity use 1 to 14 days ahead, with an 80% range that is checked to really cover 80% of days. "
           "[Code and docs on GitHub](https://github.com/saeedm31/energy-demand-forecast)")

home_tab, live_tab, how_tab = st.tabs(["🏠 Household (4 years of real data)", "🔴 Live: Portugal, every day", "🧠 How it works"])

with home_tab:
    y, fc = household()
    st.write(f"One home near Paris, smart-meter data up to **{y.index[-1].date()}**. "
             "The next 14 days, in kWh per day:")
    if os.path.exists("reports/metrics.json"):
        with open("reports/metrics.json") as f:
            m = json.load(f)
        gb = next(s for s in m["scores"] if s["model"] == "gradient_boosting")
        c = st.columns(4)
        c[0].metric("Average error per day", f'{gb["mae_kwh"]:.1f} kWh')
        c[1].metric("Better than last-week baseline", f'{gb["skill_vs_naive_pct"]:.0f}%')
        c[2].metric("Days inside the 80% range", f'{gb["interval_coverage_pct"]:.0f}%')
        c[3].metric("Weekly forecasts backtested", m["backtest"]["weeks"])
        st.caption(f'Measured on {m["backtest"]["test_period"][0]} to {m["backtest"]["test_period"][1]}: every week, '
                   "retrain on the past only, forecast 14 days, compare with what happened.")
    chart(y.iloc[-56:], fc, step=10)
    forecast_table(fc, "kWh")

with live_tab:
    if not live.is_on():
        st.info("**Live mode is off.** When it is on, a daily job adds yesterday's real electricity demand for "
                "Portugal, the model is tested on every new day it has not seen, and a person starts a retrain "
                "after at least 28 new days (minimum 7). Turn it on with the `LIVE_MODE` repository variable; "
                "see the README.")
    else:
        y, state, info, scores, lfc = live_view(live_version())
        st.write(f"Real national demand in GWh per day, collected every day. Data from **{info['data_first_day']}** "
                 f"to **{info['data_last_day']}** ({info['days']:,} days).")
        c = st.columns(4)
        c[0].metric("Last day collected", info["data_last_day"])
        c[1].metric("Model trained on data up to", info["trained_until"] or "not yet")
        c[2].metric("New days since training",
                    info["new_days_since_training"] if info["new_days_since_training"] is not None else "–")
        c[3].metric("Retrain available", "yes" if info["retrain_ready"] else f'after {info["next_retrain_after"]}')
        if lfc is None:
            st.warning("No model yet. Start the first training with the **Live retrain (manual)** workflow.")
            st.line_chart(y.iloc[-120:], y_label="GWh per day")
        else:
            if scores.get("forecasts_scored"):
                st.subheader("Test on days the model has not seen")
                c = st.columns(4)
                c[0].metric("Average error", f'{scores["mae_gwh"]:.1f} GWh', f'{scores["mape_pct"]:.1f}%',
                            delta_color="off")
                c[1].metric("Better than last-week baseline", f'{scores["skill_vs_naive_pct"]:.0f}%')
                c[2].metric("Days inside the 80% range", f'{scores["interval_coverage_pct"]:.0f}%')
                c[3].metric("Unseen days scored", scores["target_days"])
            else:
                st.caption("The model was just trained; scores appear as new days arrive.")
            st.subheader("Next 14 days")
            chart(y.iloc[-56:], lfc, step=25)
            forecast_table(lfc, "GWh")
        if state.get("history"):
            with st.expander("Training runs (each one passed or failed the gate)"):
                runs = [{"run": h["run_at"][:16].replace("T", " "), "data up to": h["trained_until"],
                         "window (days)": h["window_days"], "promoted": h["promoted"],
                         "gate MAE (GWh)": h["gate"].get("mae_gwh"),
                         "gate skill vs baseline (%)": h["gate"].get("skill_vs_naive_pct"),
                         "gate coverage (%)": h["gate"].get("interval_coverage_pct")}
                        for h in reversed(state["history"])]
                st.dataframe(pd.DataFrame(runs), hide_index=True, width="stretch")
    st.caption(live.ATTRIBUTION)

with how_tab:
    st.markdown("""
- **One direct model for all 14 days.** Gradient boosting with the horizon as a feature, so forecasts are never fed
  back in and errors do not build up.
- **No look-ahead, tested.** Features use only data up to the forecast day. A unit test changes every future value
  and checks that no feature moves.
- **Honest range.** Two quantile models give a first range; conformal calibration on the latest 13 weeks widens it
  until it really covers about 80% of days.
- **Measured like it would run.** A rolling backtest retrains every week on the past only. Baseline: same weekday
  last week.
- **Live mode.** Collect daily, keep the model fixed, test it on every new day, retrain only when a person starts it
  and only behind a gate.
- **Safe AI agent workflow.** Code changes can come from the GitHub Copilot agent; rules, tests and automatic review
  run on every pull request, and a person merges.

[Architecture](https://github.com/saeedm31/energy-demand-forecast/blob/main/docs/architecture.md) ·
[Model card](https://github.com/saeedm31/energy-demand-forecast/blob/main/docs/model_card.md) ·
[Data card](https://github.com/saeedm31/energy-demand-forecast/blob/main/docs/data_card.md)
""")
    if os.path.exists("docs/backtest.svg"):
        st.subheader("Backtest: forecast vs actual over the test year")
        st.image("docs/backtest.svg", width="stretch")

st.caption("Household data: UCI Individual household electric power consumption (EDF R&D), CC BY 4.0. "
           "Built by [Saeed Hojjatpanah](https://github.com/saeedm31).")
