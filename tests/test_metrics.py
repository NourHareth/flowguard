import numpy as np
import pytest

from flowguard.eval.metrics import (
    UNKNOWN,
    attack_pr_auc,
    benign_fpr_per_10k,
    classification_metrics,
    confusion_table,
    detection_rate,
    evasion_rate,
    flatten,
    throughput,
    unknown_detection,
)

Y_TRUE = ["Benign"] * 8 + ["DoS"] * 4 + ["PortScan"] * 2
Y_PRED = ["Benign"] * 7 + ["DoS"] + ["DoS"] * 3 + ["Benign"] + ["DoS", "PortScan"]


def test_alert_rates():
    assert benign_fpr_per_10k(Y_TRUE, Y_PRED) == pytest.approx(10_000 / 8)
    assert detection_rate(Y_TRUE, Y_PRED) == pytest.approx(5 / 6)   # the misfamilied scan still alerted


def test_classification_metrics_by_hand():
    m = classification_metrics(Y_TRUE, Y_PRED)
    dos = m["per_class"]["DoS"]
    assert dos["precision"] == pytest.approx(3 / 5) and dos["recall"] == pytest.approx(3 / 4)
    scan = m["per_class"]["PortScan"]
    assert scan["precision"] == 1.0 and scan["recall"] == 0.5 and scan["support"] == 2
    benign_f1 = 2 * (7 / 8) * (7 / 8) / (7 / 8 + 7 / 8)
    dos_f1 = 2 * 0.6 * 0.75 / 1.35
    scan_f1 = 2 * 1 * 0.5 / 1.5
    assert m["macro_f1"] == pytest.approx((benign_f1 + dos_f1 + scan_f1) / 3)
    assert m["n"] == 14


def test_macro_f1_restricted_to_shared_families():
    m = classification_metrics(Y_TRUE, Y_PRED, labels=["Benign", "DoS"])
    assert set(m["per_class"]) == {"Benign", "DoS"}


def test_confusion_table_includes_unknown():
    table = confusion_table(["Benign", "DoS"], ["Benign", UNKNOWN])
    assert table.loc["DoS", UNKNOWN] == 1 and table.to_numpy().sum() == 2


def test_pr_auc():
    assert attack_pr_auc(["Benign", "DoS", "DoS"], [0.1, 0.9, 0.8]) == 1.0
    assert np.isnan(attack_pr_auc(["Benign", "Benign"], [0.1, 0.2]))


def test_unknown_detection():
    y = ["Benign", "Benign", "Botnet", "Botnet", "Botnet", "Botnet"]
    p = ["Benign", "Benign", UNKNOWN, UNKNOWN, "DoS", "Benign"]
    out = unknown_detection(y, p, "Botnet")
    assert out == {"flagged_unknown": 0.5, "flagged_any_alert": 0.75,
                   "missed_as_benign": 0.25, "benign_fpr_per_10k": 0.0}
    with pytest.raises(ValueError):
        unknown_detection(y, p, "DDoS")


def test_evasion_rate_counts_only_caught_attacks():
    y = ["DoS", "DoS", "DoS", "Benign"]
    clean = ["DoS", "DoS", "Benign", "Benign"]   # the third was already missed
    adv = ["Benign", "DoS", "Benign", "Benign"]
    assert evasion_rate(y, clean, adv) == {"evasion_rate": 0.5, "n_caught": 2}


def test_integer_weights_equal_repeated_flows():
    """Weight 3 on a flow must score exactly like that flow appearing three times."""
    rng = np.random.default_rng(0)
    fams = np.array(["Benign", "DoS", "PortScan"], dtype=object)
    y = rng.choice(fams, 400, p=[0.7, 0.2, 0.1])
    pred = np.where(rng.random(400) < 0.8, y, rng.choice(fams, 400))
    score = rng.random(400) + (y != "Benign")
    w = rng.integers(1, 4, 400)
    rep = np.repeat(np.arange(400), w)
    weighted = classification_metrics(y, pred, sample_weight=w)
    repeated = classification_metrics(y[rep], pred[rep])
    for key in ("macro_f1", "weighted_f1", "benign_fpr_per_10k", "detection_rate", "accuracy"):
        assert weighted[key] == pytest.approx(repeated[key]), key
    assert weighted["per_class"]["DoS"]["support"] == int((y == "DoS").sum())   # flows, not weight
    assert weighted["prevalence_weighted"] and not repeated["prevalence_weighted"]
    assert attack_pr_auc(y, score, sample_weight=w) == pytest.approx(attack_pr_auc(y[rep], score[rep]))


def test_prevalence_weights_undo_the_loading_sample(project):
    import json

    from flowguard.data.load import inclusion_probability, prevalence_weights, sampling_fractions

    labels = ["BENIGN", " benign ", "DoS Hulk", "DoS Hulk - Attempted", None]
    # "Attempted" flows become Benign later but were sampled at the attack rate.
    assert inclusion_probability(labels, 0.05, 0.5).tolist() == [0.05, 0.05, 0.5, 0.5, 0.5]
    assert sampling_fractions("cse-cic-ids2018", project) == (0.05, 0.5)       # from the config
    project.tables.mkdir(parents=True)
    (project.tables / "load_cse-cic-ids2018.json").write_text(
        json.dumps({"benign_fraction": 0.1, "attack_fraction": 1.0}))
    assert sampling_fractions("cse-cic-ids2018", project) == (0.1, 1.0)        # what loading used
    assert prevalence_weights(["BENIGN", "DDoS"], "cse-cic-ids2018", project).tolist() == [10.0, 1.0]


def test_flatten_and_throughput():
    flat = flatten({"macro_f1": 0.9, "per_class": {"DoS": {"recall": 1.0}}, "bad": float("nan")}, "val.")
    assert flat == {"val.macro_f1": 0.9, "val.per_class.DoS.recall": 1.0}
    assert throughput(lambda x: sum(x), list(range(1000)), repeats=2) > 0
