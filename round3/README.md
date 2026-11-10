# LEPTOLIC round 3 — selected approach 2

**Exploratory recall comparison:** [several alert levels and a fixed-budget model study](results/recall_20261110/README_TH.md).
No replacement policy is adopted automatically. The existing 14.5% policy remains the current choice.

**Latest policy:** [FPR budget 14.5%](results/threshold_20261110_fpr145/README_TH.md).
Development-selected threshold 0.091868; historical Test FP 45, recall 32%.
Use `results/threshold_20261110_fpr145/model.json` for this policy.

**Threshold policy update:** [FPR budget 20%](results/threshold_20261110/README_TH.md), selected from
development OOF. Historical Test FP 250 -> 58, recall 92% -> 36%; probability metrics unchanged.
Use `results/threshold_20261110/model.json` for this policy; the original fitted model/results are retained.

**Real-data run complete:** [Thai report, plots, model and evidence](results/nhefs_20261105/README_TH.md).
Legacy AUC 0.6761 → new 0.7054; Brier 0.05653 → 0.05579, but the primary paired
Brier CI includes zero. Do not claim overall superiority or Thai validation.
Notebook Run all and CLI results match exactly. The historical HTTP 403 attempt
remains recorded; the supplied ZIP resolved data access for this experiment.

Read [the repeated nested-CV protocol](CV_PROTOCOL_TH.md) before new comparisons.
[The original Thai review](REVIEW_TH.md) describes saved notebook outputs, not
newly reproduced metrics. [Current status](RESULTS_STATUS_TH.md) records what has
been tested and what is still blocked.

## Setup

From the checkout root, with Python 3.12:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r round3/requirements.txt
python -m unittest discover -s tests -v
```

The cloud instance already has `/workspace/lebtolic-env/bin/python` with dependencies.
Jupyter additionally needs ipykernel/nbformat and a notebook frontend. Open
`LEPTOLIC_Round3_Review.ipynb` from the checkout root in that environment.
For Colab, first put the full `dev-by-gunner` checkout in the runtime and use it as
the working directory; opening the notebook alone does not provide its modules.

## Real-data input

If the CDC host is accessible:

```bash
python -m round3.cv_study --work-dir round3_work/nested_run_01
```

If you downloaded `NHEFS_data_and_layouts.zip` in Colab:

```bash
python -m round3.cv_study --data-zip /path/to/NHEFS_data_and_layouts.zip --work-dir round3_work/nested_run_01
```

The ZIP must contain `N82int.txt`, `N86int.txt`, `N87int.txt`, `N92int.txt` and
the four corresponding `intvYY.inputs.labels.txt` files. Files can be in the ZIP
root or an enclosing folder. Required filenames must be unique.
The importer checks required files, rejects HTML/empty files, parses layouts and
all wave records, and records hashes before installing inputs in the run's `raw/`.
It never extracts arbitrary ZIP paths. File-format validation is not independent
authentication of CDC origin. `input_archive_manifest.json` records this distinction.
No CDC requests are needed when all inputs are supplied in the ZIP.

Each attempt needs a new work directory. `protocol.json` and `study_status.json`
are kept even on failure; intermediate files do not establish completed training.

## Output and interpretation

Check `study_status.json` before interpreting any artifacts. A complete run writes:

- `metrics.json` and `report_TH.md`: comparisons and uncertainty.
- `data_audit.json`, `additional_exclusions.csv`, `split_manifest.csv`: provenance,
  joins, target changes and preserved split membership.
- `test_predictions.csv` and `development_oof.csv`: paired predictions.
- `model.json`: portable final parameters, not executable pickle.

Original target audits remain under the work directory's `outputs/`.
The primary criterion is a strictly positive lower 95% paired-bootstrap bound for
legacy-minus-new Test Brier. A positive result is not guaranteed. Test was already
exposed; this is not external validation or proof of use in Thai people.
Both models use identical guarded cohort/Test IDs. Saved scores from a different
cohort cannot serve as the direct comparator if additional exclusions change it.

The conservative death-year guard still needs official codebook confirmation.
Preprocessing, calibration and selection stay inside development training folds.
Thresholds/bands use development out-of-fold predictions. Unsupported three-level
splits return a null level and warning. Nonempty bands alone do not validate categories.

## Try a trained model

After a successful run, create `input.json`, for example:

```json
{"age_years": 50, "sex_female": 0, "baseline_sbp": 120,
 "baseline_dbp": 75, "sleep_hours": 7}
```

```bash
python -m round3.predict --model round3_work/nested_run_01/model.json --input input.json
```

Sample values may be rejected outside actual training ranges. Missing sleep can
be imputed in a five-feature model. Four-feature models ignore sleep. Prior diagnosis
is outside the cohort; `--previously-diagnosed` rejects that use.

The single-Tune experiment is retained as `python -m round3.workflow`; it is not
the selected protocol for new round 3 comparisons. All tests use explicitly invented
fixtures for software behavior only. Their metrics are never reported as NHEFS results.

Try the committed fitted model with a hypothetical input:

```bash
python -m round3.predict --model round3/results/nhefs_20261105/model.json --input round3/results/nhefs_20261105/demo_input.json
```

To regenerate the published plots from the committed paired scores:

```bash
python -m pip install -r round3/requirements-plots.txt
python -m round3.render_results --results round3/results/nhefs_20261105
```
