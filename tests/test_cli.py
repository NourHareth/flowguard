"""The `python -m` commands, run in separate processes the way the Makefile runs them.

`python -m X` runs X as __main__. Anything X saves with joblib must still load when a different
command is __main__, which the in-process tests elsewhere cannot see.
"""

import os
import subprocess
import sys

from tests.conftest import REPO
from tests.test_protocols import ctx  # noqa: F401  (pytest fixture)


def _run(module, root, *args):
    path = [str(REPO / "src"), *filter(None, [os.environ.get("PYTHONPATH")])]
    env = {**os.environ, "FLOWGUARD_ROOT": str(root), "PYTHONPATH": os.pathsep.join(path)}
    return subprocess.run([sys.executable, "-m", module, *args], env=env, capture_output=True, text=True,
                          timeout=600)


def test_saved_pipeline_and_hybrid_load_from_other_commands(ctx):  # noqa: F811
    # make features -> make train (hybrid) -> make eval, each in its own process
    for module, args in (("flowguard.features.pipeline", ()),
                         ("flowguard.models.hybrid", ()),
                         ("flowguard.eval.protocols", ("--protocol", "P1", "--models", "hybrid", "--no-mlflow"))):
        result = _run(module, ctx.paths.root, *args)
        assert result.returncode == 0, f"{module} failed:\n{result.stderr[-3000:]}"
    assert (ctx.paths.tables / "p1_test_hybrid.json").exists()


def test_score_command(ctx, tmp_path):  # noqa: F811
    import json

    import pandas as pd

    flows = pd.read_parquet(ctx.paths.processed_file("cic-ids2017")).head(25)
    capture = tmp_path / "capture.csv"
    flows[[c for c in flows.columns if c.startswith("feat_")]].assign(**{"Src IP": "10.0.0.5"}).to_csv(
        capture, index=False)
    result = _run("flowguard.serve.score", ctx.paths.root, str(capture))
    assert result.returncode == 0, result.stderr[-3000:]
    assert json.loads(result.stdout)["flows_scored"] == 25
    assert len(pd.read_csv(tmp_path / "capture_scored.csv")) == 25      # default name, next to the input
    missing = _run("flowguard.serve.score", ctx.paths.root, str(tmp_path / "nope.csv"))
    assert missing.returncode != 0 and "not found" in missing.stderr
