# LEPTOLIC round 3 — NHEFS development workflow

Read [the Thai evidence review and protocol](REVIEW_TH.md) before interpreting results.
The original notebook is preserved. This code reuses its cohort/target functions;
it does not establish that unverified CDC codebook interpretations are correct.

From the repository root, with Python 3.12:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r round3/requirements.txt
python -m unittest discover -s tests -v
python -m round3.workflow --work-dir round3_work
```

Or open `LEPTOLIC_Round3_Review.ipynb` in a Jupyter kernel with these dependencies.
Run the notebook from the checkout. For Colab, first place the full `dev-by-gunner`
checkout in the runtime and set its working directory to that checkout.
Opening the notebook alone is insufficient because it imports repository modules.

CDC access through `https://wwwn.cdc.gov` is required. Keep TLS verification enabled.
The onboarding instance has `/workspace/lebtolic-env/bin/python` with dependencies.
Its actual data run currently fails with proxy CONNECT 403; tests passing does not
mean a real model was trained. No new data-dependent metrics or band cutoffs are committed.

On success, inspect `round3_work/outputs/` for original target audits and
`round3_work/round3_outputs/` for `metrics.json`, `data_quality.json`,
`split_manifest.csv`, `test_predictions.csv`, `model.json` and `demo.json`.
The ZIP is `round3_work/LEPTOLIC_round3_outputs.zip`.
Choose a new work directory for each completed run; existing nonempty result directories are rejected.
An incomplete download may be retried in the same directory when no round3 results exist.
Do not treat preexisting cached raw files as proof of provenance: the original loader
reuses files by name. Use official fresh downloads for validation and retain source hashes.

To experiment after a successful real-data run, create `input.json`:

```json
{"age_years": 50, "sex_female": 0, "baseline_sbp": 120,
 "baseline_dbp": 75, "sleep_hours": 7}
```

```bash
python -m round3.predict --model round3_work/round3_outputs/model.json --input input.json
```

These sample inputs may be rejected if outside the actual training range. Missing
sleep is allowed only in a five-feature model and is imputed from Train. Four-feature
models ignore sleep. Prior diagnosis is outside the cohort (the CLI also has
`--previously-diagnosed`, which rejects such use).

Level 1–3 means lower/middle/upper score ranges based on Tune quantiles. It is not a
clinical category and does not imply validated Thai risk. No threshold or risk percent
is invented before training. Test is the previously exposed historical split, not
new external validation. The workflow stops if three distinct cutoffs or a positive
calibration slope cannot be supported.

Tests use explicitly invented fixtures only to verify software behavior, including
export/reload consistency and that Test/Calibration changes cannot affect candidate
selection. Their scores are never reported as NHEFS results.
