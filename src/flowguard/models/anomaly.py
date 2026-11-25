"""Anomaly detector for unseen attacks: our Isolation Forest, trained on benign flows only.

It has no threshold of its own. The hybrid detector (models/hybrid.py) sets the alert threshold
theta on benign validation flows, from what is left of the false-alert budget after the
classifier's threshold tau.
"""

from __future__ import annotations

from typing import Any

from flowguard.models.scratch import IsolationForest


def make_isolation_forest(cfg: dict[str, Any], seed: int = 42) -> IsolationForest:
    return IsolationForest(
        n_estimators=int(cfg.get("n_estimators", 300)),
        max_samples=int(cfg.get("max_samples", 256)),
        random_state=seed,
    )
