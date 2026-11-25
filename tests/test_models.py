import json

import numpy as np
import pandas as pd
import pytest
import yaml

from flowguard.config import load_config
from flowguard.data.split import temporal_split
from flowguard.features import pipeline as fp
from flowguard.models import CLASSIFIERS, TRIVIAL, build_classifier
from flowguard.models.base import stratified_cap
from tests.conftest import REPO

FAMILIES = {"Benign": 600, "DoS": 200, "PortScan": 120, "Botnet": 40}


def _separable(seed: int = 0) -> tuple[pd.DataFrame, np.ndarray]:
    """Each family is a well-separated blob in 6 features."""
    rng = np.random.default_rng(seed)
    X, y = [], []
    for k, (family, n) in enumerate(FAMILIES.items()):
        X.append(rng.normal(loc=k * 5.0, scale=1.0, size=(n, 6)))
        y += [family] * n
    return pd.DataFrame(np.vstack(X), columns=[f"f{i}" for i in range(6)]).astype("float32"), np.array(y, dtype=object)


def _models_cfg() -> dict:
    cfg = load_config("experiments", REPO)["models"]
    cfg["softmax_regression"] = {**cfg["softmax_regression"], "lr": 0.05, "batch_size": 64, "epochs": 60}
    cfg["mlp"] = {**cfg["mlp"], "hidden": [32, 16], "lr": 0.01, "batch_size": 64, "epochs": 40}
    cfg["random_forest"] = {**cfg["random_forest"], "n_estimators": 20}
    cfg["xgboost"] = {**cfg["xgboost"], "n_estimators": 50, "early_stopping_rounds": 5}
    return cfg


@pytest.mark.parametrize("name", [n for n in CLASSIFIERS if n not in TRIVIAL])
def test_classifier_learns_and_speaks_family_names(name):
    if name == "xgboost":
        pytest.importorskip("xgboost")
    X, y = _separable(0)
    X_val, y_val = _separable(1)
    model = build_classifier(name, _models_cfg())
    model.fit(X, y, X_val, y_val)
    pred = model.predict(X_val)
    assert set(pred) <= set(FAMILIES)
    assert (pred == y_val).mean() > 0.95
    proba = model.predict_proba(X_val)
    assert proba.shape == (len(X_val), len(FAMILIES)) and np.allclose(proba.sum(axis=1), 1)
    score = model.attack_score(X_val)
    assert score[y_val == "Benign"].mean() < score[y_val != "Benign"].mean()
    assert model.summary()["n_train_rows"] == len(X)


def test_early_stopping_watches_the_class_weighted_validation_loss():
    """Training minimizes a class-balanced loss, so early stopping must watch the same one,
    with the training class weights applied to the validation flows."""
    X, y = _separable(0)
    X_val, y_val = _separable(1)
    model = build_classifier("mlp", _models_cfg()).fit(X, y, X_val, y_val)
    counts = {c: (y == c).sum() for c in model.classes_}
    per_class = {c: min(len(y) / (len(counts) * n), 100) for c, n in counts.items()}
    w_val = np.array([per_class[c] for c in y_val])
    codes = model.encoder_.transform(y_val)
    est = model.estimator
    # the restored weights are those of the best epoch, so the weighted loss now is its loss
    assert est.loss(X_val, codes, w_val) == pytest.approx(est.best_loss_, rel=1e-9)
    assert est.loss(X_val, codes) != pytest.approx(est.best_loss_, rel=1e-6)   # not the unweighted one
    assert 1 <= est.best_epoch_ <= est.n_epochs_


def test_trivial_baselines_through_the_wrapper():
    X, y = _separable(0)
    majority = build_classifier("majority", {}).fit(X, y)
    assert set(majority.predict(X)) == {"Benign"}
    assert (majority.attack_score(X) == 0).all()
    guess = build_classifier("random_prior", {}).fit(X, y)
    assert (guess.predict(X) == y).mean() < 0.7
    assert guess.summary()["params"]["strategy"] == "random_prior"


def test_unknown_model_name():
    with pytest.raises(KeyError):
        build_classifier("svm", {})


def test_stratified_cap_keeps_rare_classes():
    y = np.array(["Benign"] * 100_000 + ["Botnet"] * 30, dtype=object)
    idx = stratified_cap(y, max_rows=10_000, min_per_class=1000)
    assert (y[idx] == "Botnet").sum() == 30
    assert 9_000 < len(idx) < 11_100
    assert len(stratified_cap(y, max_rows=None)) == len(y)


def test_train_run_end_to_end(project):
    from flowguard.models.train import run

    X, y = _separable(0)
    df = X.copy()
    df.insert(0, "row_id", np.arange(len(df)))
    df.insert(1, "dataset", "cic-ids2017")
    df.insert(2, "day", "monday")
    df.insert(3, "timestamp", pd.Timestamp("2017-07-03") + pd.to_timedelta(np.arange(len(df)), unit="s"))
    df.insert(4, "label_raw", y)
    df.insert(5, "family", y)
    df.to_parquet(project.processed_file("cic-ids2017"), index=False)
    pd.DataFrame({"row_id": df["row_id"], "split": temporal_split(df)}).to_parquet(
        project.split_file("cic-ids2017"), index=False)
    pipe = fp.fit_pipeline(df, load_config("features", project.root))
    fp.save(pipe, project.models / fp.PIPELINE_FILE)

    # The configured learning rates and batch sizes are sized for millions of flows. On 960
    # synthetic rows the same settings take too few optimizer steps, so use the small-data
    # settings of _models_cfg (the softmax model otherwise lands right at the 0.9 bar).
    exp = project.root / "configs" / "experiments.yaml"
    cfg = yaml.safe_load(exp.read_text())
    small = _models_cfg()
    for name in ("softmax_regression", "mlp"):
        cfg["models"][name] = small[name]
    exp.write_text(yaml.safe_dump(cfg))

    results = run(["softmax_regression", "mlp"], use_mlflow=False, paths=project)
    assert results["softmax_regression"]["validation"]["macro_f1"] > 0.9
    assert results["mlp"]["validation"]["macro_f1"] > 0.9
    assert (project.models / "p1" / "mlp.joblib").exists()
    saved = json.loads((project.tables / "p1_val_softmax_regression.json").read_text())
    assert saved["model"] == "softmax_regression" and "per_class" in saved["validation"]
