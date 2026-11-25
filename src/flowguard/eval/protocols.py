"""Evaluation protocols P1-P3 on held-out test data (P4 lives in flowguard.adversarial).

P1  In-dataset:     models from `make train`, scored on the CIC-IDS2017 P1 test split.
P2  Cross-dataset:  the same models, scored on CSE-CIC-IDS2018 (a different network), on the
                    families both datasets share. Families only 2018 has are reported apart:
                    a model cannot name them, but it can still raise an alert.
                    2018 is sampled while loading (5% of benign, 50% of attack flows), which
                    would inflate precision, F1 and PR-AUC tenfold in the attacks' favour. Every
                    P2 flow is therefore weighted by 1 / its inclusion probability, so the scores
                    reflect the real class balance (data.load.prevalence_weights). McNemar's test
                    and the confusion matrices count the sampled flows themselves.
P3  Leave-one-family-out: for each family, refit the feature pipeline and retrain the hybrid's
                    classifier without it, rebuild the hybrid, and check what happens to that
                    family's flows at test time. Nothing fitted in P3, the scaling and feature
                    selection included, has seen the held-out family.

Every P1/P2 score comes with a 95% bootstrap confidence interval, and every pair of models is
compared with McNemar's test on the same flows (Holm-adjusted p-values), see eval/stats.py.

Every result is written to reports/tables/ (JSON + confusion CSV) and logged to MLflow when
installed. `python -m flowguard.eval.report` turns them into the report tables and figures.

    python -m flowguard.eval.protocols                    # P1, P2 (if 2018 is processed), P3
    python -m flowguard.eval.protocols --protocol P1 --models mlp hybrid
    python -m flowguard.eval.protocols --protocol P3 --families PortScan Botnet
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
from typing import Any

import joblib
import numpy as np
import pandas as pd

from flowguard import tracking
from flowguard.config import Paths, load_config
from flowguard.data.io import SPLIT, load_p1, load_processed
from flowguard.data.load import prevalence_weights
from flowguard.data.schema import BENIGN, FAMILY, LABEL_RAW
from flowguard.data.split import TEST, TRAIN, VAL, leave_family_out
from flowguard.eval.metrics import (
    attack_pr_auc,
    classification_metrics,
    confusion_table,
    detection_rate,
    flatten,
    throughput,
    unknown_detection,
)
from flowguard.eval.stats import bootstrap_ci, pairwise_mcnemar
from flowguard.features import pipeline as fp

log = logging.getLogger(__name__)

THROUGHPUT_ROWS = 50_000


@dataclass
class Context:
    paths: Paths
    cfg: dict[str, Any]
    pipe: Any
    mlflow: Any = None

    @property
    def seed(self) -> int:
        return int(self.cfg.get("seed", 42))


def load_context(paths: Paths | None = None, use_mlflow: bool = True) -> Context:
    paths = paths or Paths.default()
    cfg = load_config("experiments", paths.root)
    pipe_path = paths.models / fp.PIPELINE_FILE
    if not pipe_path.exists():
        raise FileNotFoundError(f"{pipe_path} not found. Run `make features` and `make train` first.")
    mlflow = tracking.get_mlflow(use_mlflow, paths, cfg.get("mlflow_experiment", "flowguard"))
    return Context(paths, cfg, fp.load(pipe_path), mlflow)


def load_model(paths: Paths, name: str) -> Any:
    path = paths.models / "p1" / f"{name}.joblib"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run `make train` first.")
    return joblib.load(path)


def _save(ctx: Context, stem: str, record: dict[str, Any], confusion: pd.DataFrame | None = None) -> None:
    ctx.paths.tables.mkdir(parents=True, exist_ok=True)
    (ctx.paths.tables / f"{stem}.json").write_text(json.dumps(record, indent=2, default=str))
    if confusion is not None:
        confusion.to_csv(ctx.paths.tables / f"{stem}_confusion.csv")


def _stats_cfg(ctx: Context) -> dict[str, Any]:
    s = ctx.cfg.get("stats", {})
    return {"n_boot": int(s.get("n_boot", 1000)), "alpha": float(s.get("alpha", 0.05))}


def _with_ci(ctx: Context, metrics: dict[str, Any], y: np.ndarray, pred: np.ndarray, labels: list[str],
             sample_weight: np.ndarray | None = None) -> None:
    s = _stats_cfg(ctx)
    metrics["ci"] = bootstrap_ci(y, pred, labels=labels, n_boot=s["n_boot"], alpha=s["alpha"], seed=ctx.seed,
                                 sample_weight=sample_weight)


def _compare(ctx: Context, stem: str, y: np.ndarray, predictions: dict[str, np.ndarray]) -> pd.DataFrame | None:
    """McNemar's test for every pair of models on the same flows, family-level and alert-level."""
    if len(predictions) < 2:
        return None
    alpha = _stats_cfg(ctx)["alpha"]
    frames = []
    for level, binary in (("family", False), ("alert", True)):
        table = pd.DataFrame(pairwise_mcnemar(y, predictions, binary=binary, alpha=alpha))
        table.insert(0, "level", level)
        frames.append(table)
    table = pd.concat(frames, ignore_index=True)
    ctx.paths.tables.mkdir(parents=True, exist_ok=True)
    table.to_csv(ctx.paths.tables / f"{stem}.csv", index=False)
    return table


def _log(ctx: Context, run_name: str, tags: dict[str, str], metrics: dict[str, Any]) -> None:
    with tracking.run(ctx.mlflow, run_name, tags):
        if ctx.mlflow:
            ctx.mlflow.log_metrics(flatten(metrics, "test."))


# --------------------------------------------------------------------------- P1

def run_p1(ctx: Context, models: list[str] | None = None) -> dict[str, dict]:
    df = load_p1(ctx.paths)
    test = df[df[SPLIT] == TEST]
    X, y = fp.transform(ctx.pipe, test), test[FAMILY].to_numpy()
    log.info("P1: %d test flows", len(X))

    labels = sorted(set(y))
    results, predictions = {}, {}
    for name in models or ctx.cfg["protocols"]["P1_in_dataset"]:
        model = load_model(ctx.paths, name)
        pred = np.asarray(model.predict(X), dtype=object)
        predictions[name] = pred
        metrics = classification_metrics(y, pred)
        metrics["pr_auc"] = attack_pr_auc(y, model.attack_score(X))
        _with_ci(ctx, metrics, y, pred, labels)
        metrics["flows_per_second"] = throughput(model.predict, X.iloc[:THROUGHPUT_ROWS], repeats=1)
        record = {"protocol": "P1", "model": name, "test_rows": len(X), "metrics": metrics}
        _save(ctx, f"p1_test_{name}", record, confusion_table(y, pred))
        _log(ctx, f"P1-test-{name}", {"protocol": "P1", "stage": "test", "model": name}, metrics)
        results[name] = record
        ci = metrics["ci"]["macro_f1"]
        log.info("P1 %-18s macro-F1 %.4f [%.4f, %.4f]  benign FPR %7.1f/10k  PR-AUC %.4f",
                 name, metrics["macro_f1"], ci["low"], ci["high"], metrics["benign_fpr_per_10k"], metrics["pr_auc"])
    _compare(ctx, "p1_mcnemar", y, predictions)
    return results


# --------------------------------------------------------------------------- P2

def run_p2(ctx: Context, models: list[str] | None = None) -> dict[str, dict]:
    train_families = set(load_processed("cic-ids2017", ctx.paths, columns=[FAMILY])[FAMILY])
    test = load_processed("cse-cic-ids2018", ctx.paths)
    y = test[FAMILY].to_numpy()
    weights = prevalence_weights(test[LABEL_RAW], "cse-cic-ids2018", ctx.paths)
    shared = sorted(train_families & set(y))
    in_shared = np.isin(y, shared)
    only_2018 = sorted(set(y) - train_families)
    X = fp.transform(ctx.pipe, test)
    log.info("P2: %d flows, shared families %s, 2018-only %s, prevalence weights %s",
             len(X), shared, only_2018, sorted(set(np.round(weights, 3))))

    ys, ws = y[in_shared], weights[in_shared]
    results, predictions = {}, {}
    for name in models or ctx.cfg["protocols"]["P2_cross_dataset"]:
        model = load_model(ctx.paths, name)
        pred = np.asarray(model.predict(X), dtype=object)
        ps = pred[in_shared]
        predictions[name] = ps
        score = model.attack_score(X)
        metrics = classification_metrics(ys, ps, labels=shared, sample_weight=ws)
        metrics["macro_f1_unweighted"] = classification_metrics(ys, ps, labels=shared)["macro_f1"]
        metrics["pr_auc"] = attack_pr_auc(ys, score[in_shared], sample_weight=ws)
        _with_ci(ctx, metrics, ys, ps, shared, sample_weight=ws)
        if only_2018:
            metrics["unseen_family_detection_rate"] = detection_rate(y[~in_shared], pred[~in_shared],
                                                                     weights[~in_shared])
        record = {"protocol": "P2", "model": name, "test_rows": int(in_shared.sum()),
                  "shared_families": shared, "families_only_in_2018": only_2018,
                  "prevalence_weights": sorted({float(v) for v in weights}), "metrics": metrics}
        labels = sorted(set(y[in_shared]) | set(pred[in_shared]))
        _save(ctx, f"p2_test_{name}", record, confusion_table(y[in_shared], pred[in_shared], labels))
        _log(ctx, f"P2-test-{name}", {"protocol": "P2", "stage": "test", "model": name}, metrics)
        results[name] = record
        ci = metrics["ci"]["macro_f1"]
        log.info("P2 %-18s macro-F1 %.4f [%.4f, %.4f]  benign FPR %7.1f/10k  PR-AUC %.4f",
                 name, metrics["macro_f1"], ci["low"], ci["high"], metrics["benign_fpr_per_10k"], metrics["pr_auc"])
    _compare(ctx, "p2_mcnemar", y[in_shared], predictions)
    return results


# --------------------------------------------------------------------------- P3

def run_p3(ctx: Context, families: list[str] | None = None) -> dict[str, dict]:
    from flowguard.models import build_classifier
    from flowguard.models.hybrid import build_hybrid

    df = load_p1(ctx.paths)
    split = df[SPLIT]
    y_all = df[FAMILY].to_numpy()
    features_cfg = load_config("features", ctx.paths.root)
    present = set(y_all)
    p3_cfg = ctx.cfg.get("p3", {})
    requested = list(families or p3_cfg.get("families") or sorted(present - {BENIGN}))
    if BENIGN in requested:
        raise ValueError("P3 holds out attack families; Benign cannot be held out")
    families = [f for f in requested if f in present]
    skipped = sorted(set(requested) - present)
    if skipped:
        log.warning("P3: families not in the data, skipped: %s", skipped)

    from flowguard.models.tune import tuned_models_cfg

    name = ctx.cfg.get("hybrid", {}).get("classifier", "mlp")
    models_cfg = tuned_models_cfg(ctx.cfg, ctx.paths)
    if p3_cfg.get("max_train_rows"):
        models_cfg.setdefault(name, {})["max_train_rows"] = int(p3_cfg["max_train_rows"])

    results: dict[str, dict] = {}
    for family in families:
        masks = leave_family_out(df, split, family)
        # The saved P1 pipeline was fitted on training data that includes `family`: its quantiles
        # and its feature selection have seen the "unseen" attack. Refit it without that family.
        pipe = fp.fit_pipeline(df[masks.train], features_cfg, ctx.seed)
        X_tr, y_tr = fp.transform(pipe, df[masks.train]), y_all[masks.train]
        X_val, y_val = fp.transform(pipe, df[masks.val]), y_all[masks.val]
        X_te, y_te = fp.transform(pipe, df[masks.test]), y_all[masks.test]
        log.info("P3 hold out %-12s train %d, val %d, test %d (held-out flows %d), %d features",
                 family, len(y_tr), len(y_val), len(y_te), int((y_te == family).sum()), X_tr.shape[1])

        classifier = build_classifier(name, models_cfg, ctx.seed).fit(X_tr, y_tr, X_val, y_val)
        hybrid = build_hybrid(ctx.cfg, classifier, ctx.seed).fit(X_tr, y_tr, X_val, y_val, fit_classifier=False)

        record: dict[str, Any] = {"protocol": "P3", "held_out": family, "classifier": name,
                                  "pipeline_refit_without_held_out": True, "n_features": int(X_tr.shape[1])}
        for label, model in ((name, classifier), ("hybrid", hybrid)):
            pred = np.asarray(model.predict(X_te), dtype=object)
            outcome = unknown_detection(y_te, pred, family)
            called = pd.Series(pred[y_te == family]).value_counts(normalize=True)
            outcome["held_out_predicted_as"] = {k: round(float(v), 4) for k, v in called.items()}
            record[label] = outcome
            _log(ctx, f"P3-{family}-{label}",
                 {"protocol": "P3", "held_out": family, "model": label}, outcome)
        record["hybrid_calibration"] = hybrid.calibration_
        _save(ctx, f"p3_{family}", record)
        results[family] = record
        log.info("P3 %-12s %s alone: alert %.3f | hybrid: alert %.3f, as Unknown %.3f",
                 family, name, record[name]["flagged_any_alert"],
                 record["hybrid"]["flagged_any_alert"], record["hybrid"]["flagged_unknown"])
    return results


# --------------------------------------------------------------------------- CLI

def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run evaluation protocols P1-P3 on test data")
    parser.add_argument("--protocol", nargs="+", choices=["P1", "P2", "P3", "P4"], default=["P1", "P2", "P3"])
    parser.add_argument("--models", nargs="+", help="P1/P2 models (default: configs/experiments.yaml)")
    parser.add_argument("--families", nargs="+", help="P3 families to hold out (default: config)")
    parser.add_argument("--no-mlflow", action="store_true")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    ctx = load_context(use_mlflow=not args.no_mlflow)
    if "P1" in args.protocol:
        run_p1(ctx, args.models)
    if "P2" in args.protocol:
        if ctx.paths.processed_file("cse-cic-ids2018").exists():
            run_p2(ctx, args.models)
        else:
            log.warning("P2 skipped: CSE-CIC-IDS2018 is not processed yet (make data)")
    if "P3" in args.protocol:
        run_p3(ctx, args.families)
    if "P4" in args.protocol:
        log.warning("P4 is run by the adversarial lab: `make attack`")


if __name__ == "__main__":
    main()


__all__ = ["TEST", "TRAIN", "VAL", "Context", "load_context", "run_p1", "run_p2", "run_p3"]
