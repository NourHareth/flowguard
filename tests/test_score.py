"""Scoring your own CICFlowMeter CSV with the trained detector (flowguard.serve.score)."""

import numpy as np
import pandas as pd
import pytest

from flowguard.features import pipeline as fp
from flowguard.serve.score import load_detector, score_csv
from tests.test_protocols import ctx  # noqa: F401  (pytest fixture)


def _capture(context, n=300):
    """Flows in CICFlowMeter's shape: the model's feature columns plus identifying columns."""
    d17 = pd.read_parquet(context.paths.processed_file("cic-ids2017"))
    rows = d17.sample(n, random_state=0).reset_index(drop=True)
    feats = [c for c in rows.columns if c.startswith("feat_")]
    csv = rows[feats].copy()
    csv.insert(0, "Timestamp", "2026-11-25 10:00:00")
    csv.insert(1, "Src IP", "10.0.0.5")
    csv.insert(2, "Dst IP", "10.0.0.9")
    csv.insert(3, "Dst Port", 22)
    csv["Label"] = "NeedManualLabel"   # what CICFlowMeter writes for unlabelled traffic
    return rows, csv


def test_every_flow_gets_the_detectors_decision_and_reason(ctx, tmp_path):  # noqa: F811
    rows, csv = _capture(ctx)
    csv.to_csv(tmp_path / "capture.csv", index=False)
    summary = score_csv(tmp_path / "capture.csv", tmp_path / "out.csv", ctx.paths)
    out = pd.read_csv(tmp_path / "out.csv")

    assert len(out) == len(rows) == summary["flows_scored"] == summary["rows_read"]
    assert {"Timestamp", "Src IP", "Dst IP", "Dst Port", "Label", "prediction", "reason", "likely_family",
            "attack_confidence", "anomaly_score", "alert"} <= set(out.columns)
    pipe, hybrid = load_detector(ctx.paths)
    expected = hybrid.predict(fp.transform(pipe, rows))   # the same flows, scored in memory
    assert (out["prediction"].to_numpy() == expected).all()
    assert summary["alerts"] == int((expected != "Benign").sum()) == int(out["alert"].sum())
    assert set(out["reason"]) <= {"known_attack", "unknown_anomaly", "benign"}
    assert (tmp_path / "out.csv").read_text().splitlines()[1].split(",")[3] == "22"   # port as written


def test_chunked_scoring_matches_one_pass(ctx, tmp_path):  # noqa: F811
    _, csv = _capture(ctx)
    csv.to_csv(tmp_path / "capture.csv", index=False)
    score_csv(tmp_path / "capture.csv", tmp_path / "whole.csv", ctx.paths)
    score_csv(tmp_path / "capture.csv", tmp_path / "chunked.csv", ctx.paths, chunksize=37)
    pd.testing.assert_frame_equal(pd.read_csv(tmp_path / "whole.csv"), pd.read_csv(tmp_path / "chunked.csv"))


def test_infinite_values_and_repeated_header_lines(ctx, tmp_path):  # noqa: F811
    _, csv = _capture(ctx, n=50)
    csv.loc[3, "feat_0"] = np.inf                        # CICFlowMeter writes Infinity for some rates
    path = tmp_path / "capture.csv"
    text = csv.to_csv(index=False).replace(",inf,", ",Infinity,")      # CICFlowMeter's spelling
    assert "Infinity" in text
    header, body = text.split("\n", 1)
    path.write_text(header + "\n" + body + header + "\n")  # header repeated at the end, as in merged files
    summary = score_csv(path, tmp_path / "out.csv", ctx.paths)
    assert summary["rows_read"] == 51 and summary["flows_scored"] == 50 and summary["rows_without_features"] == 1


def test_a_plain_classifier_can_score_too(ctx, tmp_path):  # noqa: F811
    _, csv = _capture(ctx, n=40)
    csv.to_csv(tmp_path / "capture.csv", index=False)
    summary = score_csv(tmp_path / "capture.csv", tmp_path / "out.csv", ctx.paths, model="softmax_regression")
    out = pd.read_csv(tmp_path / "out.csv")
    assert {"prediction", "attack_score", "alert"} <= set(out.columns) and "reason" not in out.columns
    assert summary["model"] == "softmax_regression" and summary["flows_scored"] == 40


def test_missing_columns_and_missing_models_explain_the_fix(ctx, tmp_path):  # noqa: F811
    _, csv = _capture(ctx, n=10)
    csv.drop(columns=["feat_0", "feat_3"]).to_csv(tmp_path / "capture.csv", index=False)
    with pytest.raises(ValueError, match=r"lacks 2 feature column.*rename"):
        score_csv(tmp_path / "capture.csv", tmp_path / "out.csv", ctx.paths)
    csv.to_csv(tmp_path / "capture.csv", index=False)
    with pytest.raises(FileNotFoundError, match="Train first"):
        score_csv(tmp_path / "capture.csv", tmp_path / "out.csv", ctx.paths, model="nothing")
