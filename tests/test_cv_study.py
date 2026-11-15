"""Invented fixtures test logic; never report their metrics as real NHEFS results."""
import unittest
import tempfile
from pathlib import Path

import numpy as np
from round3.cv_study import (FeatureSlice, PROTOCOL, candidate_grid, experiment,
                             export_model, paired_bootstrap, target_guard, write_report)
from round3.predict import PortableModel
from round3.workflow import level_cuts, load_original


class StudyTests(unittest.TestCase):
    def test_empty_development_level_is_rejected(self):
        with self.assertRaises(ValueError):
            level_cuts([.01, .01, .01, .01, .2, .2])

    def test_entirely_missing_feature_is_rejected(self):
        x = np.zeros((20, 5)); x[:, 4] = np.nan
        with self.assertRaises(ValueError):
            FeatureSlice(5).fit(x)
        FeatureSlice(4).fit(x)

    def test_conservative_death_guard(self):
        row = dict(target=0, horizon_date='1991-06-01')
        self.assertIsNotNone(target_guard(row, {'87': {'DEATHYR': '87'}}, load_original()))
        self.assertIsNotNone(target_guard(row, {'92': {'NEWDODYR': '91'}}, load_original()))
        self.assertIsNone(target_guard(row, {'92': {'NEWDODYR': '92'}}, load_original()))
        self.assertIsNone(target_guard(dict(target=1), {'87': {'DEATHYR': '87'}}, load_original()))

    def test_paired_bootstrap_identical_models_cannot_prove_improvement(self):
        y = np.array([0, 1]*10); p = np.linspace(.01, .3, 20)
        result = paired_bootstrap(y, p, p, 100)
        self.assertEqual(result['brier_old_minus_new']['ci95'], [0., 0.])
        self.assertFalse(result['primary_supported'])

    def test_all_exported_candidate_types_match_sklearn(self):
        rng = np.random.default_rng(2)
        x = rng.normal(size=(240, 5)); y = (x[:, 0]+rng.normal(size=240) > 0).astype(int)
        search, configs = candidate_grid(42, 2, 2)
        for model, config in zip(search.param_grid['pipeline'], configs):
            model.fit(x, y)
            state = export_model(model, config, x, ['a', 'b', 'c', 'd', 'e'], np.array([.1, .2]), .1)
            actual = PortableModel(state).predict_proba(x[:, :config['features']])
            np.testing.assert_allclose(actual, model.predict_proba(x), atol=1e-12)

    def test_nested_study_test_labels_do_not_select_model_threshold_or_bands(self):
        original = load_original(); rng = np.random.default_rng(42); cohort = []
        for i in range(650):
            age = float(rng.uniform(25, 80))
            y = int(rng.random() < 1/(1+np.exp(-(age-55)/10)))
            cohort.append(dict(participant_id=str(i), age_years=age, sex_female=float(rng.integers(2)),
                               baseline_sbp=float(rng.uniform(90, 129)), baseline_dbp=float(rng.uniform(55, 79)),
                               sleep_hours=float(rng.uniform(4, 9)), target=y))
        split = original['split_ids']([r['target'] for r in cohort])
        protocol = {**PROTOCOL, 'outer_folds': 2, 'repeats': 2, 'inner_folds': 2,
                    'calibration_folds': 2, 'bootstrap_repetitions': 30}
        first, predictions, oof, state = experiment(cohort, split, original, protocol)
        changed = [dict(row) for row in cohort]
        for i in split['test']:
            changed[i]['target'] = 1-changed[i]['target']
        second, _, _, state2 = experiment(changed, split, original, protocol)
        for key in ['final_selected', 'final_inner_candidates', 'threshold', 'legacy_threshold', 'level_cuts']:
            self.assertEqual(first[key], second[key])
        self.assertEqual(state, state2)
        self.assertEqual(len(oof), first['development_n']*2)
        self.assertEqual(len(predictions), first['test_n'])
        xtest = np.array([[cohort[i][f] for f in state['features']] for i in split['test']])
        np.testing.assert_allclose(PortableModel(state).predict_proba(xtest)[:, 1],
                                   [r['new_score'] for r in predictions], atol=1e-12)
        self.assertEqual(sum(b['n'] for b in first['test_calibration_new']['test_bins']), first['test_n'])
        with tempfile.TemporaryDirectory() as tmp:
            write_report(Path(tmp), first)
            report = (Path(tmp)/'report_TH.md').read_text(encoding='utf-8')
            self.assertIn('internal validation', report)


if __name__ == '__main__':
    unittest.main()
