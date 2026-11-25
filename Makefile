# Every number in the report is rebuilt from these targets.
# On Windows without `make`, run the python commands shown on each line directly.
PY ?= python

# Targets for parts that are not built yet stop here with a message instead of doing nothing.
# Replace the $(PLANNED) line with the command in the target's comment once the part exists.
PLANNED = $(PY) -c "import sys; sys.exit('make $@: not built yet, see Project status in the README')"

.PHONY: install data features tune train eval analyze ablate attack explain report score serve demo test lint clean

install:
	$(PY) -m pip install -e ".[all]"

data:          ## raw zips -> cleaned, labeled Parquet + P1 split in data/processed
	$(PY) -m flowguard.data.load
	$(PY) -m flowguard.data.clean
	$(PY) -m flowguard.data.split

features:      ## fit the preprocessing pipeline on the P1 training split
	$(PY) -m flowguard.features.pipeline

tune:          ## random hyperparameter search on the validation split (before train)
	$(PY) -m flowguard.models.tune

train:         ## train the P1 classifiers, then build and calibrate the hybrid detector
	$(PY) -m flowguard.models.train
	$(PY) -m flowguard.models.hybrid

eval:          ## run protocols P1-P3 and log to MLflow
	$(PY) -m flowguard.eval.protocols

analyze:       ## calibration, loss and learning curves, drift, adaptation (after eval)
	$(PY) -m flowguard.eval.calibration
	$(PY) -m flowguard.eval.curves
	$(PY) -m flowguard.eval.drift
	$(PY) -m flowguard.eval.adaptation

ablate:        ## ablation study: retrains our models once per variant (slow)
	$(PY) -m flowguard.eval.ablation

attack:        ## protocol P4 (planned): $(PY) -m flowguard.adversarial.attacks
	$(PLANNED)

explain:       ## SHAP figures (planned): $(PY) -m flowguard.explain.shap_explain
	$(PLANNED)

report:        ## result tables and figures into reports/
	$(PY) -m flowguard.eval.report

score:         ## your own CICFlowMeter CSV through the trained hybrid: make score CSV=capture.csv
	$(PY) -m flowguard.serve.score $(CSV)

serve:         ## detection API (planned): $(PY) -m uvicorn flowguard.serve.api:app --reload
	$(PLANNED)

demo:          ## dashboard (planned): $(PY) -m streamlit run app/dashboard.py
	$(PLANNED)

test:          ## what CI runs (.github/workflows/tests.yml)
	$(PY) -m pytest

lint:
	$(PY) -m ruff check src tests app

clean:
	$(PY) -c "import shutil; [shutil.rmtree(p, ignore_errors=True) for p in ('mlruns', '.pytest_cache', '.ruff_cache')]"
