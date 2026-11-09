"""Forecast API and a one-page dashboard.

    python api.py                  # http://127.0.0.1:8000
    gunicorn -b 0.0.0.0:8000 api:app

Endpoints:
    GET /                 dashboard: last 8 weeks + 14-day forecast with the 80% interval
    GET /api/forecast     next 14 days as JSON
    GET /api/metrics      backtest results (from train.py)
    GET /health           model and data status
    GET /live             live mode (optional): Portugal's national demand, collected daily
    GET /api/live         live mode as JSON: status, forecast, scores on days the model has not seen
"""

import json
import os
from functools import lru_cache
from html import escape

from flask import Flask, jsonify, request

from energy_forecast import live, service

app = Flask(__name__)
app.json.sort_keys = False


@lru_cache(maxsize=1)
def state():
    y, imputed, model = service.load_all()
    return y, model, service.forecast(model, y)


def metrics():
    if not os.path.exists("reports/metrics.json"):
        return None
    with open("reports/metrics.json") as f:
        return json.load(f)


@app.get("/api/forecast")
def api_forecast():
    raw_days = request.args.get("days", "14")
    try:
        days = int(raw_days)
    except ValueError:
        return jsonify(error="days must be a whole number from 1 to 14"), 400
    if not 1 <= days <= 14:
        return jsonify(error="days must be a whole number from 1 to 14"), 400

    y, _, fc = state()
    return jsonify(issued_after=str(y.index[-1].date()), unit="kWh per day", interval="80%",
                   days=fc.iloc[:days].to_dict(orient="records"))


@app.get("/api/metrics")
def api_metrics():
    m = metrics()
    return (jsonify(m), 200) if m else (jsonify(error="Run python train.py first"), 404)


@app.get("/health")
def health():
    y, model, _ = state()
    return jsonify(ok=True, model=model.name, data_last_day=str(y.index[-1].date()), days=len(y),
                   live_mode=live.is_on())


@lru_cache(maxsize=2)
def live_view(data_version):
    """Everything live mode shows, recomputed when the live data or state file changes."""
    y, imputed = live.load()
    st = live.read_state()
    weather = live.load_weather()
    info = live.status(y, imputed, st)
    info["uses_temperature"] = weather is not None
    model = live.current_model(y, imputed, st, weather=weather)
    if model is None:
        return y, info, None, None
    return (y, info, live.live_evaluation(model, y, imputed, st["trained_until"], weather=weather),
            live.forecast(model, y, weather=weather))


def live_version():
    paths = (live.DATA_PATH, live.STATE_PATH, live.wx.OBS_PATH, live.wx.FORECAST_PATH)
    return tuple(os.path.getmtime(p) if os.path.exists(p) else 0 for p in paths)


LIVE_OFF = {"error": "Live mode is off. Turn it on with the LIVE_MODE repository variable or run: python live.py collect"}


@app.get("/api/live")
def api_live():
    if not live.is_on():
        return jsonify(LIVE_OFF), 404
    y, info, scores, fc = live_view(live_version())
    return jsonify(status=info, live_test=scores, unit="GWh per day", interval="80%",
                   days=fc.to_dict(orient="records") if fc is not None else [])


def svg_chart(history, fc, width=900, height=320, pad=44, step=10):
    """Line chart of recent history and the forecast band, as inline SVG (no JavaScript needed)."""
    xs = list(history.index.strftime("%Y-%m-%d")) + list(fc["date"])
    values = list(history.values) + list(fc["upper_kwh"]) + list(fc["lower_kwh"])
    lo_v, hi_v = 0, max(values) * 1.1
    n = len(xs)

    def px(i):
        return pad + i * (width - 2 * pad) / (n - 1)

    def py(v):
        return height - pad - (v - lo_v) * (height - 2 * pad) / (hi_v - lo_v)

    h = len(history)
    hist_pts = " ".join(f"{px(i):.1f},{py(v):.1f}" for i, v in enumerate(history.values))
    fc_pts = " ".join(f"{px(h - 1 + i):.1f},{py(v):.1f}" for i, v in
                      enumerate([history.values[-1]] + list(fc["forecast_kwh"])))
    band = ([f"{px(h + i):.1f},{py(v):.1f}" for i, v in enumerate(fc["upper_kwh"])] +
            [f"{px(h + i):.1f},{py(v):.1f}" for i, v in reversed(list(enumerate(fc["lower_kwh"])))])
    grid = "".join(
        f'<line x1="{pad}" x2="{width - pad}" y1="{py(v):.1f}" y2="{py(v):.1f}" class="grid"/>'
        f'<text x="{pad - 6}" y="{py(v) + 4:.1f}" text-anchor="end" class="axis">{v:.0f}</text>'
        for v in range(0, int(hi_v) + 1, step))
    ticks = "".join(f'<text x="{px(i):.1f}" y="{height - pad + 18}" text-anchor="middle" class="axis">'
                    f'{escape(xs[i][5:])}</text>' for i in range(0, n, 7))
    return f'''<svg viewBox="0 0 {width} {height}" role="img" aria-label="Daily energy use and forecast">
  {grid}{ticks}
  <polygon points="{' '.join(band)}" class="band"/>
  <polyline points="{hist_pts}" class="hist"/>
  <polyline points="{fc_pts}" class="fc"/>
  <line x1="{px(h - 1):.1f}" x2="{px(h - 1):.1f}" y1="{pad / 2}" y2="{height - pad}" class="now"/>
  <text x="{px(h - 1) + 6:.1f}" y="{pad / 2 + 10}" class="axis">forecast →</text>
</svg>'''


PAGE = """<!DOCTYPE html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><title>Energy demand forecast</title>
<style>
 :root {{ --bg:#f7f7f4; --card:#fff; --text:#1d2321; --muted:#66706b; --line:#e2e2dc; --accent:#0f766e; --hist:#334155; }}
 @media (prefers-color-scheme: dark) {{ :root {{ --bg:#0f1312; --card:#171c1b; --text:#e6ebe9; --muted:#97a29d; --line:#29302e; --accent:#2dd4bf; --hist:#cbd5e1; }} }}
 body {{ font-family: system-ui, sans-serif; background: var(--bg); color: var(--text); margin: 0; padding: 2rem 1rem; }}
 .wrap {{ max-width: 960px; margin: 0 auto; }} h1 {{ margin: 0 0 .2rem; }} p.sub {{ color: var(--muted); margin: 0 0 1.5rem; }}
 .card {{ background: var(--card); border: 1px solid var(--line); border-radius: 12px; padding: 1rem; margin-bottom: 1rem; overflow-x: auto; }}
 svg {{ width: 100%; height: auto; min-width: 560px; }}
 .grid {{ stroke: var(--line); }} .axis {{ fill: var(--muted); font-size: 11px; }}
 .hist {{ fill: none; stroke: var(--hist); stroke-width: 2; }} .fc {{ fill: none; stroke: var(--accent); stroke-width: 2.5; stroke-dasharray: 6 4; }}
 .band {{ fill: var(--accent); opacity: .18; }} .now {{ stroke: var(--muted); stroke-dasharray: 2 3; }}
 table {{ border-collapse: collapse; width: 100%; font-size: .9rem; }} td, th {{ padding: .35rem .5rem; border-bottom: 1px solid var(--line); text-align: right; }}
 th:first-child, td:first-child, th:nth-child(2), td:nth-child(2) {{ text-align: left; }}
 .stats {{ display: flex; flex-wrap: wrap; gap: 1rem; }} .stat b {{ display: block; font-size: 1.4rem; }} .stat span {{ color: var(--muted); font-size: .8rem; }}
 footer {{ color: var(--muted); font-size: .8rem; margin-top: 1rem; }}
</style></head><body><div class="wrap">
<h1>⚡ Household energy demand forecast</h1>
<p class="sub">Next 14 days of electricity use (kWh per day), with an 80% prediction interval. Data up to {last_day}.</p>
{stats}
<div class="card">{chart}</div>
<div class="card"><table><tr><th>Date</th><th>Day</th><th>Forecast</th><th>80% range</th></tr>{rows}</table></div>
<footer>Model: gradient boosting with conformal intervals · Data: UCI household power consumption (EDF R&amp;D) ·
<a href="/api/forecast">JSON</a> · <a href="/api/metrics">backtest metrics</a> · <a href="/live">live mode</a></footer>
</div></body></html>"""


@app.get("/")
def index():
    y, _, fc = state()
    m = metrics()
    stats = ""
    if m:
        gb = next(s for s in m["scores"] if s["model"] == "gradient_boosting")
        stats = ('<div class="card stats">'
                 f'<div class="stat"><b>{gb["mae_kwh"]:.1f} kWh</b><span>average error per day</span></div>'
                 f'<div class="stat"><b>{gb["skill_vs_naive_pct"]:.0f}%</b><span>better than last-week baseline</span></div>'
                 f'<div class="stat"><b>{gb.get("interval_coverage_pct", 0):.0f}%</b><span>of days inside the 80% range</span></div>'
                 f'<div class="stat"><b>{m["backtest"]["weeks"]}</b><span>weekly forecasts backtested</span></div></div>')
    rows = "".join(
        f'<tr><td>{escape(r.date)}</td><td>{escape(r.weekday)}{" 🎉" if r.holiday else ""}</td>'
        f'<td><b>{r.forecast_kwh:.1f}</b></td><td>{r.lower_kwh:.1f} – {r.upper_kwh:.1f}</td></tr>'
        for r in fc.itertuples())
    return PAGE.format(last_day=y.index[-1].date(), stats=stats, chart=svg_chart(y.iloc[-56:], fc), rows=rows)


LIVE_PAGE = PAGE[:PAGE.index("<h1>")] + """<h1>⚡ Live mode: Portugal's electricity demand</h1>
<p class="sub">Real national demand (GWh per day), collected every day. The model is retrained only when a person
starts it, after at least {window} new days; until then every new day tests the model on data it has not seen.</p>
<div class="card stats">{stats}</div>
{body}
<footer>{source} · <a href="/api/live">JSON</a> · <a href="/">household forecast</a></footer>
</div></body></html>"""


def stat(value, label):
    return f'<div class="stat"><b>{escape(str(value))}</b><span>{escape(label)}</span></div>'


@app.get("/live")
def live_page():
    if not live.is_on():
        return jsonify(LIVE_OFF), 404
    y, info, scores, fc = live_view(live_version())
    stats = (stat(info["data_last_day"], "last day collected") +
             stat(info["trained_until"] or "not yet", "model trained on data up to") +
             stat(info["new_days_since_training"] if info["new_days_since_training"] is not None else "–",
                  "new days since training") +
             stat("yes" if info["retrain_ready"] else f'after {info["next_retrain_after"]}', "retrain available"))
    if fc is None:
        body = '<div class="card">No model yet. Start the first training with <code>python live.py retrain</code>.</div>'
    else:
        test = ""
        if scores.get("forecasts_scored"):
            test = ('<div class="card stats">' +
                    stat(f'{scores["mae_gwh"]:.1f} GWh', "average error on unseen days") +
                    stat(f'{scores["mape_pct"]:.1f}%', "average error in percent") +
                    stat(f'{scores["skill_vs_naive_pct"]:.0f}%', "better than last-week baseline") +
                    stat(f'{scores["interval_coverage_pct"]:.0f}%', "of days inside the 80% range") +
                    stat(scores["target_days"], "unseen days scored") + "</div>")
        chart_fc = fc.rename(columns={"forecast_gwh": "forecast_kwh", "lower_gwh": "lower_kwh", "upper_gwh": "upper_kwh"})
        rows = "".join(
            f'<tr><td>{escape(r.date)}</td><td>{escape(r.weekday)}{" 🎉" if r.holiday else ""}</td>'
            f'<td><b>{r.forecast_gwh:.1f}</b></td><td>{r.lower_gwh:.1f} – {r.upper_gwh:.1f}</td></tr>'
            for r in fc.itertuples())
        body = (test + f'<div class="card">{svg_chart(y.iloc[-56:], chart_fc, step=25)}</div>'
                '<div class="card"><table><tr><th>Date</th><th>Day</th><th>Forecast (GWh)</th><th>80% range</th></tr>'
                f'{rows}</table></div>')
    return LIVE_PAGE.format(window=info["window_days"], stats=stats, body=body, source=escape(info["source"]))


if __name__ == "__main__":
    app.run(host=os.environ.get("HOST", "127.0.0.1"), port=int(os.environ.get("PORT", 8000)))
