"""Small invented fixtures test software only; never evidence of model performance."""
import unittest
import contextlib
import io
import json
from pathlib import Path
import tempfile
from unittest.mock import patch
import numpy as np

from round3.predict import PortableModel, predict_file
from round3.workflow import (load_original, levels, level_cuts, band_report, wilson,
                             select_threshold, evaluate, fit_candidates)
from round3.workflow import run


class Round3Tests(unittest.TestCase):
    def test_no_predicted_positive_is_undefined_precision(self):
        result = evaluate(np.array([0, 1]), np.array([.1, .2]), .5, load_original())
        self.assertIsNone(result['precision'])
        self.assertEqual(result['recall'], 0)
        self.assertEqual(result['confusion_matrix'], dict(tn=1, fp=0, fn=1, tp=0))

    def test_tied_levels_and_empty_band(self):
        cuts = [.1, .2]
        np.testing.assert_array_equal(levels([.05, .1, .2], cuts), [1, 2, 3])
        report = band_report(np.array([0, 1]), np.array([.05, .3]), cuts)
        self.assertEqual(sum(r['n'] for r in report), 2)
        self.assertIsNone(report[1]['observed_rate'])
        with self.assertRaises(ValueError):
            level_cuts([.1, .1, .1])

    def test_wilson_zero_events_is_not_zero_uncertainty(self):
        interval = wilson(0, 25)
        self.assertGreater(interval[1], 0)
        self.assertIsNone(wilson(0, 0))

    def test_threshold_uses_demo_f2_objective(self):
        y = np.array([0, 0, 1, 1])
        result = select_threshold(y, np.array([.01, .02, .08, .1]))
        self.assertEqual(result['threshold'], .08)
        self.assertEqual(result['tune_f2'], 1.)

    def test_candidates_do_not_use_test_or_calibration(self):
        # Altering held-out values/labels must not change candidate scores or choice.
        rng = np.random.default_rng(9)
        x = rng.normal(size=(240, 5)); y = np.tile([0, 1], 120)
        split = dict(train=np.arange(160), tune=np.arange(160, 200),
                     calibration=np.arange(200, 220), test=np.arange(220, 240))
        records, selected, _ = fit_candidates(x, y, split)
        x[200:] = 100; y[200:] = 0
        changed, chosen, _ = fit_candidates(x, y, split)
        self.assertEqual(records, changed)
        self.assertEqual(selected, chosen)

    def test_portable_logistic(self):
        state = dict(selected={'family': 'logistic'}, imputer_median=[2.],
                     scaler_mean=[2.], scaler_scale=[2.], coefficients=[[1.]], intercept=[0.])
        model = PortableModel(state)
        expected = 1/(1+np.exp(-1))
        np.testing.assert_allclose(model.predict_proba([[np.nan], [4.]])[:, 1], [.5, expected])

    def test_portable_tree(self):
        state = dict(selected={'family': 'tree'}, imputer_median=[0.],
                     scaler_mean=[0.], scaler_scale=[1.], tree=dict(
                         children_left=[1, -1, -1], children_right=[2, -1, -1],
                         feature=[0, -2, -2], threshold=[.5, -2, -2],
                         value=[[[.5, .5]], [[.8, .2]], [[.3, .7]]]))
        np.testing.assert_allclose(PortableModel(state).predict_proba([[0.], [1.]])[:, 1], [.2, .7])

    def test_pipeline_export_reload_on_invented_fixture(self):
        # This validates wiring/export, NOT NHEFS accuracy or clinical validity.
        original = load_original()
        rng = np.random.default_rng(42)
        cohort = []
        for i in range(1000):
            age = float(rng.uniform(25, 80))
            y = int(rng.random() < 1/(1+np.exp(-((age-55)/12))))
            cohort.append(dict(participant_id=str(i), age_years=age,
                               sex_female=float(rng.integers(0, 2)),
                               baseline_sbp=float(rng.uniform(90, 129)),
                               baseline_dbp=float(rng.uniform(55, 79)),
                               sleep_hours=float(rng.uniform(4, 9)), target=y))
        original['prepare_cohort'] = lambda root: (cohort, {})
        with tempfile.TemporaryDirectory() as tmp, \
             patch('round3.workflow.load_original', return_value=original), \
             patch('round3.workflow.data_quality', return_value={'fixture_only': True}), \
             contextlib.redirect_stdout(io.StringIO()):
            result = run(Path(tmp))
            out = Path(tmp)/'round3_outputs'
            state = json.loads((out/'model.json').read_text())
            expected = json.loads((out/'demo.json').read_text())
            values = {k: float(v) for k, v in zip(state['features'], state['imputer_median'])}
            values['sex_female'] = int(round(values['sex_female']))
            actual = predict_file(out/'model.json', values)
            self.assertAlmostEqual(actual['experimental_probability_8y'],
                                   expected['experimental_probability_8y'], places=12)
            self.assertEqual(actual['level'], expected['level'])
            if result['test_levels'] is not None:
                self.assertEqual(sum(b['n'] for b in result['test_levels']), result['test_selected']['n'])
            else:
                self.assertIn('empty', result['level_status'])
                self.assertIsNone(actual['level'])
                self.assertIsNone(state['level_cuts'])
            self.assertTrue((Path(tmp)/'LEPTOLIC_round3_outputs.zip').is_file())


if __name__ == '__main__':
    unittest.main()
