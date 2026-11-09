"""Live mode command line (optional; the household dataset stays the default).

    python live.py collect                     # add new days of Portugal's demand and temperature (first run backfills from 2019)
    python live.py status                      # data, training cutoff, live test scores, is a retrain due?
    python live.py retrain                     # new training if >= window_days new days (default 28)
    python live.py retrain --window-days 14    # change the window (minimum 7)
    python live.py retrain --force             # train now, still behind the gate

The daily GitHub workflow runs `collect` only when the repository variable LIVE_MODE is true.
Retraining is started by a person (the "Live retrain" workflow, or this command).
"""

import argparse
import json
import sys

from energy_forecast import live


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("collect", help="download new complete days")
    sub.add_parser("status", help="show live status and scores")
    r = sub.add_parser("retrain", help="train a new model if enough new days were collected")
    r.add_argument("--window-days", type=int, default=None,
                   help=f"new days needed before a retrain (minimum {live.MIN_WINDOW_DAYS})")
    r.add_argument("--force", action="store_true", help="ignore the window (the gate still applies)")
    args = parser.parse_args(argv)

    if args.cmd == "collect":
        live.collect()
        try:
            live.collect_weather()
        except Exception as e:          # temperature is optional: demand data is kept either way
            print(f"Weather not updated: {e}")
        return 0

    y, imputed = live.load()
    state = live.read_state()
    weather = live.load_weather()

    if args.cmd == "status":
        info = live.status(y, imputed, state)
        info["uses_temperature"] = weather is not None
        model = live.current_model(y, imputed, state, weather=weather)
        if model is not None:
            info["live_test"] = live.live_evaluation(model, y, imputed, state["trained_until"], weather=weather)
        print(json.dumps(info, indent=2))
        return 0

    runs_before = len(state.get("history", []))
    try:
        promoted, message, state = live.retrain(y, imputed, state, window_days=args.window_days, force=args.force,
                                                weather=weather)
    except ValueError as e:
        print(f"Error: {e}")
        return 2
    live.write_state(state)
    print(message)
    if len(state.get("history", [])) > runs_before:      # a training ran: show the gate result
        print(json.dumps(state["history"][-1], indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
