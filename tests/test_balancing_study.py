"""Synthetic software fixtures only, never NHEFS performance evidence."""
import unittest
import numpy as np
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold

from round3.balancing_study import (TrainingOnlyLogistic, decision, encode_extra, export_fit,
                                    metrics, nested_evaluate, policies)
from round3.predict import PortableModel


class BalancingTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(913)
        self.x = rng.normal(size=(120, 4))
        self.y = np.r_[np.zeros(96, int), np.ones(24, int)]
        self.x[self.y == 1, 0] += 1
        self.x[0, 0] = np.nan

    def test_preprocessing_before_resampling_and_validation_is_inert(self):
        x, y = self.x[:110], self.y[:110]
        model = TrainingOnlyLogistic('oversample', random_state=12).fit(x, y)
        np.testing.assert_allclose(model.imputer_.statistics_, np.nanmedian(x, axis=0))
        np.testing.assert_allclose(model.scaler_.mean_, np.nanmean(np.where(np.isnan(x),
            np.nanmedian(x, axis=0), x), axis=0))
        self.assertEqual(model.fit_class_counts_, [96, 96])
        sampled = model.sample_indices_
        np.testing.assert_array_equal(sampled[:len(y)], np.arange(len(y)))
        self.assertTrue(np.all(y[sampled[len(y):]] == 1))
        before = model.scaler_.mean_.copy()
        self.assertEqual(model.predict_proba(np.full((3, 4), 1e6)).shape, (3, 2))
        np.testing.assert_array_equal(before, model.scaler_.mean_)

    def test_class_weight_uses_training_y_without_resampling(self):
        m = TrainingOnlyLogistic('balanced').fit(self.x[:110], self.y[:110])
        self.assertEqual(m.logistic_.class_weight, 'balanced')
        self.assertEqual(m.original_class_counts_, [96, 14])
        self.assertEqual(m.fit_class_counts_, [96, 14])
        self.assertEqual(len(m.sample_indices_), 110)

    def test_sampling_reproducible_and_different_seeds_change_samples(self):
        a = TrainingOnlyLogistic('oversample', random_state=1).fit(self.x, self.y)
        b = TrainingOnlyLogistic('oversample', random_state=1).fit(self.x, self.y)
        c = TrainingOnlyLogistic('oversample', random_state=2).fit(self.x, self.y)
        np.testing.assert_array_equal(a.sample_indices_, b.sample_indices_)
        self.assertFalse(np.array_equal(a.sample_indices_, c.sample_indices_))

    def test_nested_threshold_reconstructs_from_outer_training_only(self):
        protocol = dict(repeats=2, outer_folds=3, inner_folds=3, C=.1)
        folds = list(RepeatedStratifiedKFold(n_splits=3, n_repeats=2, random_state=42).split(self.x, self.y))
        result = nested_evaluate(self.x, self.y, 'balanced', folds, protocol)
        train, valid = folds[0]
        inner = StratifiedKFold(3, shuffle=True, random_state=100)
        p = np.full(len(train), np.nan)
        for j, (it, iv) in enumerate(inner.split(self.x[train], self.y[train])):
            self.assertFalse(set(train[it]) & set(valid))
            model = TrainingOnlyLogistic('balanced', .1, 1000+j).fit(self.x[train[it]], self.y[train[it]])
            p[iv] = model.predict_proba(self.x[train[iv]])[:, 1]
        expected = policies(self.y[train], p)
        for row in result['folds'][:2]:
            self.assertEqual(row['threshold'], expected[row['policy']]['threshold'])
        self.assertEqual(result['scores'].shape, (2, 120))
        for row in result['repeats']:
            self.assertEqual(row['tp']+row['fn'], 24)
            self.assertEqual(row['fp']+row['tn'], 96)
        for row in result['folds']:
            if row['policy'] == 'fpr145':
                self.assertLessEqual(row['inner_fpr'], .145)

    def test_development_decision_rejects_inconsistent_improvement(self):
        base = [dict(repeat=i, policy=p, brier=.05, f2=.3, recall=.4, fpr=.14)
                for i in range(3) for p in ['f2', 'fpr145']]
        candidate = [{**r, 'brier': .049, 'f2': .31, 'recall': .41, 'fpr': .13} for r in base]
        e = {'none': {'repeats': base}, 'balanced': {'repeats': candidate}}
        self.assertEqual(decision(e, ['balanced'])['chosen'], 'balanced')
        candidate[0]['brier'] = .051
        self.assertEqual(decision(e, ['balanced'])['chosen'], 'none')

    def test_codebook_special_values_do_not_become_numeric_weight_or_no_smoking(self):
        values = encode_extra(dict(PMR5b1='995', PMR5b2='997', L1='8', L2a='7'))
        self.assertTrue(np.isnan(values[0])); self.assertEqual(values[1:], [1., 0., 0., 1.])
        self.assertEqual(encode_extra(dict(PMR5b1='140', PMR5b2='142', L1='1', L2a='1')),
                         [141., 0., 1., 0., 0.])
        # L1=NO and L2a=skip is a verified never-100-cigarettes response.
        self.assertEqual(encode_extra(dict(PMR5b1='140', PMR5b2='997', L1='2', L2a='7')),
                         [140., 0., 0., 0., 0.])
        # Contradictory codes remain unknown.
        self.assertEqual(encode_extra(dict(L1='2', L2a='1'))[-1], 1.)

    def test_metrics_f2_and_undefined_precision(self):
        y = np.array([1, 1, 0, 0]); p = np.array([.7, .2, .8, .1])
        r = metrics(y, p, .5, include_calibration=False)
        self.assertEqual([r[k] for k in ['tp', 'fp', 'fn', 'tn']], [1, 1, 1, 1])
        self.assertEqual(r['f2'], .5)
        self.assertIsNone(metrics(y, p, .9, include_calibration=False)['precision'])

    def test_export_roundtrip_with_missing_values(self):
        for mode in ['none', 'balanced', 'oversample']:
            model = TrainingOnlyLogistic(mode).fit(self.x, self.y)
            p = model.predict_proba(self.x)
            state = export_fit(model, ['a', 'b', 'c', 'd'], policies(self.y, p[:, 1]))
            np.testing.assert_allclose(PortableModel(state).predict_proba(self.x), p, atol=1e-12)


if __name__ == '__main__':
    unittest.main()
