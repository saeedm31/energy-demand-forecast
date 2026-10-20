"""Draw the backtest chart used in the README (docs/backtest.svg).

Usage:
    python train.py          # writes reports/backtest.csv
    python plot_backtest.py

Shows every day of the test year once: the forecast made 1 to 7 days earlier,
its 80% range, and what actually happened. Plain SVG, no plotting library needed.
"""

import os

import pandas as pd

W, H = 900, 360
LEFT, RIGHT, TOP, BOTTOM = 48, 16, 56, 40

STYLE = """
  .bg { fill: #fcfcfb; } .grid { stroke: #e4e3df; stroke-width: 1; }
  .axis { fill: #52514e; font: 12px system-ui, -apple-system, sans-serif; }
  .title { fill: #0b0b0b; font: 600 15px system-ui, -apple-system, sans-serif; }
  .actual { fill: none; stroke: #52514e; stroke-width: 1.5; }
  .fc { fill: none; stroke: #2a78d6; stroke-width: 2; }
  .band { fill: #2a78d6; opacity: .18; }
  .key-actual { stroke: #52514e; stroke-width: 2; } .key-fc { stroke: #2a78d6; stroke-width: 2.5; }
  @media (prefers-color-scheme: dark) {
    .bg { fill: #1a1a19; } .grid { stroke: #34332f; } .axis { fill: #c3c2b7; } .title { fill: #f0efec; }
    .actual, .key-actual { stroke: #c3c2b7; } .fc, .key-fc { stroke: #3987e5; } .band { fill: #3987e5; opacity: .28; }
  }
"""


def main(path="reports/backtest.csv", out="docs/backtest.svg"):
    r = pd.read_csv(path, parse_dates=["target_day"])
    r = r[(r["model"] == "gradient_boosting") & (r["h"] <= 7)].sort_values("target_day")
    days, actual = r["target_day"].tolist(), r["actual"].to_numpy()
    fc, lo, hi = r["forecast"].to_numpy(), r["lower"].to_numpy(), r["upper"].to_numpy()
    measured = ~r["imputed"].astype(bool).to_numpy()   # outage days are not scored
    inside = 100 * ((actual >= lo) & (actual <= hi))[measured].mean()

    top = max(actual.max(), hi.max()) * 1.05
    n = len(days)

    def x(i):
        return LEFT + i * (W - LEFT - RIGHT) / (n - 1)

    def y(v):
        return H - BOTTOM - v * (H - TOP - BOTTOM) / top

    def line(vals):
        return " ".join(f"{x(i):.0f},{y(v):.0f}" for i, v in enumerate(vals))

    band = line(hi) + " " + " ".join(f"{x(i):.0f},{y(v):.0f}" for i, v in reversed(list(enumerate(lo))))
    grid = "".join(f'<line class="grid" x1="{LEFT}" x2="{W - RIGHT}" y1="{y(v):.1f}" y2="{y(v):.1f}"/>'
                   f'<text class="axis" x="{LEFT - 8}" y="{y(v) + 4:.1f}" text-anchor="end">{v}</text>'
                   for v in range(0, int(top) + 1, 10))
    months = "".join(f'<text class="axis" x="{x(i):.1f}" y="{H - BOTTOM + 20}" text-anchor="middle">'
                     f'{d.strftime("%b")}</text>'
                     for i, d in enumerate(days) if d.day == 1)
    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" role="img"
  aria-label="Backtest: daily electricity use and the forecast made up to a week earlier, with its 80% range">
<style>{STYLE}</style>
<rect class="bg" width="{W}" height="{H}" rx="8"/>
<text class="title" x="{LEFT}" y="24">Forecast vs actual, every day of the test year (kWh per day)</text>
<line class="key-actual" x1="{LEFT}" x2="{LEFT + 22}" y1="42" y2="42"/>
<text class="axis" x="{LEFT + 28}" y="46">Actual</text>
<line class="key-fc" x1="{LEFT + 92}" x2="{LEFT + 114}" y1="42" y2="42"/>
<text class="axis" x="{LEFT + 120}" y="46">Forecast, 1–7 days ahead</text>
<rect class="band" x="{LEFT + 290}" y="36" width="22" height="12" rx="2"/>
<text class="axis" x="{LEFT + 318}" y="46">80% range (true value inside on {inside:.0f}% of days)</text>
{grid}{months}
<polygon class="band" points="{band}"/>
<polyline class="actual" points="{line(actual)}"/>
<polyline class="fc" points="{line(fc)}"/>
</svg>
"""
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        f.write(svg)
    print(f"Wrote {out}: {n} days, {inside:.1f}% inside the 80% range")


if __name__ == "__main__":
    main()
