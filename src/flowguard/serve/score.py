"""Score a CSV of flows with a trained detector: the offline half of the detection service.

Input: a CSV written by CICFlowMeter, for example from a pcap captured in your own lab. Use the
corrected CICFlowMeter the improved datasets were made with
(https://github.com/GintsEngelen/CICFlowMeter), so the columns match the training data. Map any
column that is named differently under `rename:` in configs/features.yaml.

Every flow goes through the saved feature pipeline (models/pipeline.joblib) and the saved
detector (models/p1/hybrid.joblib by default), exactly like the P1 test flows. The output CSV
puts each flow's identifying columns (time, addresses, ports) next to the decision and, for the
hybrid, the reason for it: known_attack, unknown_anomaly or benign.

    python -m flowguard.serve.score capture.csv                    # writes capture_scored.csv
    python -m flowguard.serve.score capture.csv --model xgboost --out alerts.csv

Only capture traffic on machines and networks you own (README, Ethics and lab safety). Live
traffic differs from the lab datasets, so treat these results as a demo, not as a measurement.
"""

from __future__ import annotations

import argparse
import json
import logging
from collections import Counter
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from flowguard.config import Paths, load_config
from flowguard.data.load import normalize_chunk
from flowguard.data.schema import BENIGN
from flowguard.features import pipeline as fp

log = logging.getLogger(__name__)

# Copied to the output when the CSV has them. None of them is a model input.
ID_COLUMNS = ("Flow ID", "Timestamp", "Src IP", "Src Port", "Dst IP", "Dst Port", "Protocol", "Label")
HYBRID_COLUMNS = ("prediction", "reason", "likely_family", "attack_confidence", "anomaly_score")


def load_detector(paths: Paths, model: str = "hybrid") -> tuple[Any, Any]:
    """(fitted feature pipeline, fitted detector) saved by `make features train`."""
    pipe_path = paths.models / fp.PIPELINE_FILE
    model_path = paths.models / "p1" / f"{model}.joblib"
    for path in (pipe_path, model_path):
        if not path.exists():
            raise FileNotFoundError(f"{path} not found. Train first: `make features train` "
                                    "(README, Running the pipeline).")
    return fp.load(pipe_path), joblib.load(model_path)


def score_frame(flows: pd.DataFrame, pipe: Any, detector: Any, features_cfg: dict[str, Any]) -> pd.DataFrame:
    """One row per flow: its identifying columns, the decision, and what the decision rests on.

    Rows without any feature value (a header line repeated inside the CSV, an empty line) are
    left out, as cleaning does for the datasets.
    """
    # Identifying columns are copied as the CSV wrote them (ports stay 22, not 22.0); only the
    # features are converted to numbers.
    as_written = flows.rename(columns=lambda c: str(c).strip()).rename(columns=features_cfg.get("rename") or {})
    flows = normalize_chunk(flows, features_cfg)
    needed = list(pipe.feature_names_in_)
    missing = [c for c in needed if c not in flows.columns]
    if missing:
        shown = ", ".join(missing[:8]) + (", ..." if len(missing) > 8 else "")
        raise ValueError(f"The CSV lacks {len(missing)} feature column(s) the model was trained on: {shown}. "
                         "If your flow tool names them differently, map them under `rename:` in "
                         "configs/features.yaml.")
    X_raw = flows[needed].replace([np.inf, -np.inf], np.nan)  # as in cleaning: infinite rates are missing
    has_values = X_raw.notna().any(axis=1).to_numpy()
    X_raw = X_raw[has_values]

    out = as_written.loc[has_values, [c for c in ID_COLUMNS if c in as_written.columns]].reset_index(drop=True)
    is_hybrid = hasattr(detector, "decide")               # the hybrid gives a reason with each decision
    if not len(out):                                      # same columns as a scored chunk, no rows
        for column in HYBRID_COLUMNS if is_hybrid else ("prediction", "attack_score"):
            out[column] = pd.Series(dtype=object)
    elif is_hybrid:
        decision = detector.decide(fp.transform(pipe, X_raw))
        for column in HYBRID_COLUMNS:
            out[column] = np.asarray(decision[column])
    else:                                                 # a single classifier from models/p1/
        X = fp.transform(pipe, X_raw)
        proba = detector.predict_proba(X)
        out["prediction"] = detector.classes_[proba.argmax(axis=1)]
        out["attack_score"] = detector.attack_score(X, proba)
    out["alert"] = out["prediction"] != BENIGN
    return out


def score_csv(csv: Path, out: Path, paths: Paths | None = None, model: str = "hybrid",
              chunksize: int | None = None) -> dict[str, Any]:
    """Score `csv` in chunks (captures can be large) into `out`. Returns a summary."""
    paths = paths or Paths.default()
    features_cfg = load_config("features", paths.root)
    if chunksize is None:
        chunksize = int(load_config("experiments", paths.root).get("data", {}).get("chunksize", 500_000))
    pipe, detector = load_detector(paths, model)

    predictions: Counter[str] = Counter()
    rows_read = 0
    out.parent.mkdir(parents=True, exist_ok=True)
    reader = pd.read_csv(csv, chunksize=chunksize, low_memory=False, skipinitialspace=True,
                         dtype={c: "string" for c in features_cfg.get("string_columns", [])})
    with open(out, "w", newline="", encoding="utf-8") as fh:
        for part, chunk in enumerate(reader):
            rows_read += len(chunk)
            scored = score_frame(chunk, pipe, detector, features_cfg)
            scored.to_csv(fh, index=False, header=part == 0)
            predictions.update(scored["prediction"].astype(str))

    scored_rows = sum(predictions.values())
    return {"input": str(csv), "output": str(out), "model": model, "rows_read": rows_read,
            "flows_scored": scored_rows, "rows_without_features": rows_read - scored_rows,
            "alerts": scored_rows - predictions.get(BENIGN, 0), "predictions": dict(predictions.most_common())}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("csv", type=Path, help="CICFlowMeter CSV to score")
    parser.add_argument("--out", type=Path, help="output CSV (default: <name>_scored.csv next to the input)")
    parser.add_argument("--model", default="hybrid", help="a model saved in models/p1/ (default: hybrid)")
    args = parser.parse_args(argv)
    if not args.csv.is_file():
        parser.error(f"{args.csv} not found")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    out = args.out or args.csv.with_name(f"{args.csv.stem}_scored.csv")
    summary = score_csv(args.csv, out, model=args.model)
    log.info("%d of %d flows raised an alert -> %s", summary["alerts"], summary["flows_scored"], out)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
