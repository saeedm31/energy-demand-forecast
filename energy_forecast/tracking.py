"""Experiment tracking: MLflow when it is installed, a local JSON-lines log otherwise.

Both record the same thing per run: parameters, metrics, tags and artifact files,
so every backtest is reproducible and comparable.
"""

import json
import os
import time


class Tracker:
    def __init__(self, experiment="energy-demand-forecast", local_dir="runs", use_mlflow=None):
        self.experiment, self.local_dir = experiment, local_dir
        if use_mlflow is None:
            try:
                import mlflow  # noqa: F401
                use_mlflow = True
            except ImportError:
                use_mlflow = False
        self.use_mlflow = use_mlflow
        if use_mlflow:
            import mlflow
            self.mlflow = mlflow
            mlflow.set_experiment(experiment)

    @property
    def backend(self):
        return "mlflow" if self.use_mlflow else f"local ({self.local_dir}/runs.jsonl)"

    def log_run(self, name, params, metrics, tags=None, artifacts=()):
        metrics = {k: float(v) for k, v in metrics.items() if v == v}  # drop NaN
        if self.use_mlflow:
            with self.mlflow.start_run(run_name=name):
                self.mlflow.log_params(params)
                self.mlflow.log_metrics(metrics)
                self.mlflow.set_tags(tags or {})
                for path in artifacts:
                    self.mlflow.log_artifact(path)
            return
        os.makedirs(self.local_dir, exist_ok=True)
        record = {"experiment": self.experiment, "run": name, "time": time.strftime("%Y-%m-%dT%H:%M:%S"),
                  "params": params, "metrics": metrics, "tags": tags or {}, "artifacts": list(artifacts)}
        with open(os.path.join(self.local_dir, "runs.jsonl"), "a") as f:
            f.write(json.dumps(record) + "\n")
