"""Paired repeated nested threshold evaluation, with development-only fallback.

Real data only in run(); unit tests use explicitly synthetic software fixtures.
No model/threshold decision consumes historical Test outcomes.
"""
import argparse
import csv
import hashlib
import json
import platform
from collections import Counter
from pathlib import Path

import numpy as np
import scipy
import sklearn
from scipy.optimize import brentq, minimize
from scipy.special import expit, logit
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold
from sklearn.preprocessing import StandardScaler

from .cv_study import guarded_cohort
from .predict import PortableModel
from .threshold_policy import read_csv, select_budget_threshold
from .workflow import load_original, save_json, select_threshold, wilson

FEATURES = ['age_years', 'sex_female', 'baseline_sbp', 'baseline_dbp']
MAIN = ['none', 'balanced', 'oversample']
FALLBACK = ['weight', 'smoking', 'weight_smoking']
PROTOCOL = dict(seed=42, outer_folds=5, repeats=3, inner_folds=3, C=.1,
                threshold_policies=['f2', 'fpr145'], max_fpr=.145,
                bootstrap_repetitions=2000,
                selection='development only; all-repeat joint criterion',
                hyperparameters='C fixed before experiment; nesting selects thresholds',
                test_scope='previously exposed, exploratory internal evaluation')


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_csv(path, rows):
    if not rows:
        return
    with Path(path).open('w', newline='', encoding='utf-8') as stream:
        w = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator='\n')
        w.writeheader(); w.writerows(rows)


class TrainingOnlyLogistic(ClassifierMixin, BaseEstimator):
    """Fit preprocessing on original training rows, then balance training only."""
    def __init__(self, mode='none', C=.1, random_state=42):
        self.mode = mode
        self.C = C
        self.random_state = random_state

    def fit(self, x, y):
        x, y = np.asarray(x, float), np.asarray(y, int)
        if self.mode not in MAIN or set(np.unique(y)) != {0, 1}:
            raise ValueError('Need a supported mode and both binary classes')
        if np.any(np.all(np.isnan(x), axis=0)):
            raise ValueError('Entirely missing training feature')
        self.imputer_ = SimpleImputer(strategy='median').fit(x)
        filled = self.imputer_.transform(x)
        self.scaler_ = StandardScaler().fit(filled)
        z = self.scaler_.transform(filled)
        indices = np.arange(len(y))
        if self.mode == 'oversample':
            positive = np.flatnonzero(y == 1)
            extra = int(np.sum(y == 0) - len(positive))
            if extra < 0:
                raise ValueError('This protocol oversamples a minority positive class only')
            indices = np.r_[indices, np.random.default_rng(self.random_state).choice(
                positive, size=extra, replace=True)]
        self.sample_indices_ = indices
        self.fit_class_counts_ = np.bincount(y[indices], minlength=2).tolist()
        self.original_class_counts_ = np.bincount(y, minlength=2).tolist()
        self.logistic_ = LogisticRegression(C=self.C, max_iter=2000,
            random_state=self.random_state,
            class_weight='balanced' if self.mode == 'balanced' else None)
        self.logistic_.fit(z[indices], y[indices])
        self.classes_ = self.logistic_.classes_
        self.n_features_in_ = x.shape[1]
        return self

    def predict_proba(self, x):
        return self.logistic_.predict_proba(self.scaler_.transform(self.imputer_.transform(x)))


def calibration(y, p):
    """Evaluation only: these fitted diagnostics never alter prediction scores."""
    z = logit(np.clip(p, 1e-10, 1-1e-10))
    intercept_only = brentq(lambda a: np.mean(expit(z+a))-np.mean(y), -50, 50)
    design = np.column_stack([np.ones(len(z)), z])
    def loss(beta):
        eta = design @ beta
        return np.mean(np.logaddexp(0, eta)-y*eta)
    def jac(beta):
        return design.T @ (expit(design @ beta)-y)/len(y)
    fit = minimize(loss, [0., 1.], jac=jac, method='BFGS', options={'gtol': 1e-8})
    # Sparse validation events can cause unstable calibration estimates.
    ok = bool(fit.success and np.isfinite(fit.x).all())
    return dict(mean_probability=float(np.mean(p)), observed_rate=float(np.mean(y)),
                calibration_in_large=float(intercept_only),
                calibration_intercept=float(fit.x[0]) if ok else None,
                calibration_slope=float(fit.x[1]) if ok else None,
                calibration_fit_converged=ok)


def metrics(y, p, threshold, include_calibration=True):
    y, p = np.asarray(y), np.asarray(p)
    pred = p >= threshold
    tp = int(np.sum(pred & (y == 1))); fp = int(np.sum(pred & (y == 0)))
    fn = int(np.sum(~pred & (y == 1))); tn = int(np.sum(~pred & (y == 0)))
    result = dict(n=len(y), events=int(y.sum()), tp=tp, fp=fp, fn=fn, tn=tn,
        recall=tp/(tp+fn), precision=tp/(tp+fp) if tp+fp else None,
        f2=5*tp/(5*tp+4*fn+fp), fpr=fp/(fp+tn),
        roc_auc=float(roc_auc_score(y, p)), ap=float(average_precision_score(y, p)),
        brier=float(brier_score_loss(y, p)))
    if include_calibration:
        result.update(calibration(y, p))
    return result


def policies(y, p):
    f2 = select_threshold(y, p)
    f2['objective'] = 'max_F2_on_development_OOF_only'
    return dict(f2=f2, fpr145=select_budget_threshold(y, p, .145))


def export_fit(model, feature_names, choices):
    """Portable parameters for research replay; no executable pickle or policy replacement."""
    return dict(features=feature_names,
        selected=dict(family='logistic', calibration='none', C=model.C, balancing=model.mode),
        imputer_median=model.imputer_.statistics_.tolist(),
        scaler_mean=model.scaler_.mean_.tolist(), scaler_scale=model.scaler_.scale_.tolist(),
        coefficients=model.logistic_.coef_.tolist(), intercept=model.logistic_.intercept_.tolist(),
        study_calibration=dict(method='none'), threshold_policies=choices,
        original_class_counts=model.original_class_counts_, fitted_class_counts=model.fit_class_counts_,
        usage='Research replay with PortableModel and explicitly encoded feature vectors; not the legacy predict_file interface',
        status='exploratory; does not replace existing model/policy')


def summary(rows, group_keys, metric_keys):
    groups = {}
    for row in rows:
        key = tuple(row[k] for k in group_keys)
        groups.setdefault(key, []).append(row)
    result = []
    for key, values in groups.items():
        for metric in metric_keys:
            a = [r[metric] for r in values if r.get(metric) is not None]
            result.append({**dict(zip(group_keys, key)), 'metric': metric, 'valid_n': len(a),
                'mean': float(np.mean(a)) if a else None,
                'sd': float(np.std(a, ddof=1)) if len(a) > 1 else None,
                'min': float(np.min(a)) if a else None, 'max': float(np.max(a)) if a else None})
    return result


def nested_evaluate(x, y, variant, folds, protocol=PROTOCOL):
    mode = variant if variant in MAIN else 'none'
    scores = np.full((protocol['repeats'], len(y)), np.nan)
    predictions = {k: np.zeros_like(scores, dtype=bool) for k in ['f2', 'fpr145']}
    fold_rows, fitted_counts, bin_rows = [], [], []
    for index, (train, valid) in enumerate(folds):
        seed = 100+index
        inner = StratifiedKFold(protocol['inner_folds'], shuffle=True, random_state=seed)
        inner_scores = np.full(len(train), np.nan)
        for j, (it, iv) in enumerate(inner.split(x[train], y[train])):
            model = TrainingOnlyLogistic(mode, protocol['C'], seed*10+j).fit(x[train[it]], y[train[it]])
            inner_scores[iv] = model.predict_proba(x[train[iv]])[:, 1]
            fitted_counts.append(dict(variant=variant, outer=index, stage=f'inner_{j}',
                seed=seed*10+j, train_n=len(it), valid_n=len(iv),
                original_negative=model.original_class_counts_[0],
                original_positive=model.original_class_counts_[1],
                fitted_negative=model.fit_class_counts_[0], fitted_positive=model.fit_class_counts_[1]))
        assert np.isfinite(inner_scores).all()
        selected = policies(y[train], inner_scores)
        model = TrainingOnlyLogistic(mode, protocol['C'], seed).fit(x[train], y[train])
        p = model.predict_proba(x[valid])[:, 1]
        repeat, fold = divmod(index, protocol['outer_folds'])
        scores[repeat, valid] = p
        fitted_counts.append(dict(variant=variant, outer=index, stage='outer', seed=seed,
            train_n=len(train), valid_n=len(valid), original_negative=model.original_class_counts_[0],
            original_positive=model.original_class_counts_[1], fitted_negative=model.fit_class_counts_[0],
            fitted_positive=model.fit_class_counts_[1]))
        for policy, choice in selected.items():
            threshold = choice['threshold']
            predictions[policy][repeat, valid] = p >= threshold
            fold_rows.append(dict(variant=variant, repeat=repeat, fold=fold, policy=policy,
                threshold=threshold, inner_fpr=float(np.mean(inner_scores[y[train] == 0] >= threshold)),
                **metrics(y[valid], p, threshold)))
        cuts = np.unique(np.quantile(inner_scores, [.2, .4, .6, .8]))
        bin_rows += reliability_bins(y[valid], p, cuts, variant, f'outer_{index}')
    assert np.isfinite(scores).all()
    repeat_rows = []
    for repeat in range(protocol['repeats']):
        for policy in predictions:
            # Threshold varies by outer fold: use stored decisions for operating metrics.
            p = scores[repeat]; pred = predictions[policy][repeat]
            stats = metrics(y, pred.astype(float), .5, include_calibration=False)
            stats.update(roc_auc=float(roc_auc_score(y, p)), ap=float(average_precision_score(y, p)),
                         brier=float(brier_score_loss(y, p)), **calibration(y, p))
            repeat_rows.append(dict(variant=variant, repeat=repeat, policy=policy, **stats))
    return dict(scores=scores, predictions=predictions, folds=fold_rows, repeats=repeat_rows,
                fit_counts=fitted_counts, bins=bin_rows)


def reliability_bins(y, p, cuts, variant, scope):
    assigned = np.searchsorted(cuts, p, side='right'); rows = []
    for b in range(len(cuts)+1):
        mask = assigned == b; n = int(mask.sum()); events = int(y[mask].sum())
        interval = wilson(events, n)
        rows.append(dict(variant=variant, scope=scope, bin=b, n=n, events=events,
            lower_cut=float(cuts[b-1]) if b else 0., upper_cut=float(cuts[b]) if b < len(cuts) else 1.,
            mean_probability=float(p[mask].mean()) if n else None,
            observed_rate=events/n if n else None, wilson_low=interval[0] if interval else None,
            wilson_high=interval[1] if interval else None))
    return rows


def decision(evaluations, candidates):
    base = {(r['repeat'], r['policy']): r for r in evaluations['none']['repeats']}
    checks = {}
    for candidate in candidates:
        rows = {(r['repeat'], r['policy']): r for r in evaluations[candidate]['repeats']}
        conditions = dict(
            brier_lower_every_repeat=all(rows[k]['brier'] < base[k]['brier'] for k in base),
            f2_higher_every_repeat=all(rows[k]['f2'] > base[k]['f2'] for k in base if k[1] == 'f2'),
            budget_recall_higher_every_repeat=all(rows[k]['recall'] > base[k]['recall'] for k in base if k[1] == 'fpr145'),
            mean_budget_fpr_not_higher=np.mean([rows[k]['fpr'] for k in base if k[1] == 'fpr145'])
                <= np.mean([base[k]['fpr'] for k in base if k[1] == 'fpr145']))
        conditions = {k: bool(v) for k, v in conditions.items()}
        checks[candidate] = dict(**conditions, eligible=all(conditions.values()))
    eligible = [name for name in candidates if checks[name]['eligible']]
    best = min(eligible, key=lambda name: np.mean([r['brier'] for r in evaluations[name]['repeats']])) if eligible else 'none'
    return dict(checks=checks, chosen=best,
                note='Development screening rule, not significance or external-validation evidence')


def extra_features(raw, original):
    spec = original['parse_layout']((raw/'intv82.inputs.labels.txt').read_text(encoding='utf-8-sig'))
    anchors = {'PMR5b1': (4149, 4152), 'PMR5b2': (4152, 4155), 'L1': (1990, 1991), 'L2a': (1991, 1992)}
    if any(spec.get(k) != v for k, v in anchors.items()):
        raise ValueError('Baseline additional-feature layout changed')
    rows = original['read_wave'](raw/'N82int.txt', spec, ['SEQNUM1', *anchors], 'SEQNUM1')
    return {ident: encode_extra(row) for ident, row in rows.items()}


def encode_extra(row):
    def number(key):
        try:
            return int(row[key])
        except (ValueError, TypeError, KeyError):
            return None
    readings = [number(k) for k in ['PMR5b1', 'PMR5b2']]
    readings = [v for v in readings if v is not None and 69 <= v <= 300]
    weight = float(np.mean(readings)) if readings else np.nan
    ever, current = number('L1'), number('L2a')
    if ever == 2 and current in (7, None):
        status = 'never_100'
    elif ever == 1 and current in (1, 2):
        status = 'current' if current == 1 else 'former'
    else:
        status = 'unknown'  # Unknown/contradiction/proxy skip is never silently NO.
    return [weight, float(not readings), float(status == 'current'),
            float(status == 'former'), float(status == 'unknown')]


def input_data(source, repository):
    original = load_original()
    reference = json.loads((repository/'round3/results/nhefs_20261105/input_archive_manifest.json').read_text())
    source_rows = []
    for f in reference['files']:
        path = source/'raw'/f['name']
        if not path.exists():
            source.joinpath('raw').mkdir(parents=True, exist_ok=True)
            original['download'](source/'raw', f['name'])
        sha = digest(path)
        if sha != f['sha256']:
            raise ValueError('Input differs from historical archive: '+f['name'])
        source_rows.append(dict(name=f['name'], sha256=sha, bytes=path.stat().st_size,
                                url=f['expected_source_url'], matches_historical_archive=True))
    cohort, audit = original['prepare_cohort'](source)
    split = original['split_ids']([r['target'] for r in cohort])
    kept, excluded, joins = guarded_cohort(source, cohort, original)
    if excluded or len(kept) != len(cohort):
        raise ValueError('Cohort changed; do not silently compare different people')
    x = np.array([[r[f] for f in FEATURES] for r in cohort], float)
    y = np.array([r['target'] for r in cohort], int)
    dev = np.sort(np.concatenate([split[k] for k in ['train', 'tune', 'calibration']]))
    test = split['test']
    counts = [len(y), int(y.sum()), len(dev), int(y[dev].sum()), len(test), int(y[test].sum())]
    if counts != [2720, 166, 2311, 141, 409, 25]:
        raise ValueError('Historical cohort/split counts changed')
    if not np.isfinite(x).all() or set(dev) & set(test):
        raise ValueError('Invalid required predictors or overlapping split')
    split_hashes = {k: hashlib.sha256('\n'.join(cohort[int(i)]['participant_id'] for i in indices).encode()).hexdigest()
                    for k, indices in {**split, 'development': dev}.items()}
    return original, cohort, audit, x, y, dev, test, dict(
        source_files=source_rows, split_id_sha256=split_hashes, counts=counts, joins=joins,
        source_claim='CDC download; byte-identical to previously supplied archive',
        additional_exclusions=excluded)


def missing_audit(source, cohort, dev, extras, original):
    merged = read_csv(source/'outputs/NHEFS_Merged_Four_Waves_Raw.csv')
    reasons = {r['participant_id']: r['reason'] for r in read_csv(source/'outputs/NHEFS_Target_Audit.csv')}
    development_ids = {cohort[int(i)]['participant_id'] for i in dev}
    included_ids = {r['participant_id'] for r in cohort}
    groups = {}
    for row in merged:
        ident = row['participant_id']
        baseline_reasons = {'invalid_baseline_date', 'baseline_prior_diagnosis_or_unknown',
                            'baseline_age_sex_or_bp_missing', 'baseline_measured_bp_above_project_cutoff'}
        group = ('included_development' if ident in development_ids else 'included_test'
                 if ident in included_ids else 'excluded_at_baseline'
                 if reasons[ident] in baseline_reasons else 'eligible_baseline_excluded_followup')
        baseline = {k[4:]: v for k, v in row.items() if k.startswith('b82_')}
        values = original['baseline_features'](baseline)
        values.update(dict(zip(['weight_lb', 'weight_missing', 'current_smoker', 'former_smoker', 'smoking_unknown'], extras[ident])))
        groups.setdefault(group, []).append(values)
    rows = []
    for group, values in groups.items():
        for feature in values[0]:
            a = np.array([v[feature] for v in values]); observed = a[np.isfinite(a)]
            rows.append(dict(group=group, feature=feature, n=len(a), missing=int(np.isnan(a).sum()),
                mean=float(observed.mean()) if len(observed) else None,
                median=float(np.median(observed)) if len(observed) else None))
    return rows, dict(Counter(reasons.values()))


def missed_events(evaluation, x, y, feature_names):
    rows = []
    for policy, predictions in evaluation['predictions'].items():
        misses = (~predictions).sum(axis=0)
        for count in range(predictions.shape[0]+1):
            group = (y == 1) & (misses == count)
            for j, feature in enumerate(feature_names):
                values = x[group, j]; finite = values[np.isfinite(values)]
                rows.append(dict(policy=policy, missed_repeats=count, n_events=int(group.sum()),
                    feature=feature, missing=int(np.isnan(values).sum()),
                    mean=float(finite.mean()) if len(finite) else None,
                    median=float(np.median(finite)) if len(finite) else None))
    return rows


def bootstrap_test(y, scores, final_policies, repetitions):
    """Same bootstrap people for every model/policy; no tuning in the bootstrap."""
    rng = np.random.default_rng(42); samples = {}; differences = {}
    for _ in range(repetitions):
        ids = rng.integers(0, len(y), len(y)); target = y[ids]
        if len(np.unique(target)) < 2:
            continue
        draw = {}
        for variant, p in scores.items():
            for policy, choice in final_policies[variant].items():
                values = metrics(target, p[ids], choice['threshold'], include_calibration=False)
                draw[(variant, policy)] = values
                for metric, value in values.items():
                    if value is not None:
                        samples.setdefault((variant, policy, metric), []).append(value)
        for (variant, policy), values in draw.items():
            if variant == 'none':
                continue
            base = draw[('none', policy)]
            for metric in ['recall', 'precision', 'f2', 'fpr', 'roc_auc', 'brier']:
                if values[metric] is not None and base[metric] is not None:
                    differences.setdefault((variant, policy, metric), []).append(values[metric]-base[metric])
    def rows(samples):
        return [dict(variant=k[0], policy=k[1], metric=k[2], valid_draws=len(a),
                     lower=float(np.quantile(a, .025)), upper=float(np.quantile(a, .975)))
                for k, a in samples.items()]
    return rows(samples), rows(differences)


def run(source, output, repository=None):
    source, output = Path(source), Path(output)
    repository = Path(repository) if repository else Path(__file__).resolve().parents[1]
    if output.exists() and any(output.iterdir()):
        raise ValueError('Use a fresh output directory')
    output.mkdir(parents=True, exist_ok=True)
    save_json(output/'protocol.json', PROTOCOL)
    save_json(output/'study_status.json', dict(status='started', real_data_metrics_available=False))
    try:
        original, cohort, audit, x, y, dev, test, provenance = input_data(source, repository)
        save_json(output/'provenance.json', provenance)
        xd, yd = x[dev], y[dev]
        folds = list(RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=42).split(xd, yd))
        write_csv(output/'outer_membership.csv', [dict(development_row=int(v), repeat=i//5, fold=i%5)
            for i, (_, valid) in enumerate(folds) for v in valid])
        evaluations = {}; matrices = {name: x for name in MAIN}; feature_sets = {name: FEATURES for name in MAIN}
        for variant in MAIN:
            print('Evaluating', variant, flush=True)
            evaluations[variant] = nested_evaluate(xd, yd, variant, folds)
        main_decision = decision(evaluations, MAIN[1:])
        save_json(output/'balancing_decision_before_test.json', main_decision)
        use_fallback = main_decision['chosen'] == 'none'
        print('Development balancing decision:', main_decision['chosen'], 'fallback:', use_fallback, flush=True)
        if use_fallback:
            extras = extra_features(source/'raw', original)
            ex = np.array([extras[r['participant_id']] for r in cohort])
            extra_names = ['weight_lb', 'weight_missing', 'current_smoker', 'former_smoker', 'smoking_unknown']
            missing, reasons = missing_audit(source, cohort, dev, extras, original)
            write_csv(output/'missingness_audit.csv', missing)
            save_json(output/'exclusion_counts.json', reasons)
            sleep = np.array([r['sleep_hours'] for r in cohort])[:, None]
            write_csv(output/'missed_event_audit.csv', missed_events(evaluations['none'],
                np.column_stack([xd, sleep[dev], ex[dev]]), yd, FEATURES+['sleep_hours']+extra_names))
            for variant, columns in [('weight', [0, 1]), ('smoking', [2, 3, 4]), ('weight_smoking', [0, 1, 2, 3, 4])]:
                matrices[variant] = np.column_stack([x, ex[:, columns]])
                feature_sets[variant] = FEATURES+[extra_names[i] for i in columns]
                print('Evaluating fallback', variant, flush=True)
                evaluations[variant] = nested_evaluate(matrices[variant][dev], yd, variant, folds)
            fallback_decision = decision(evaluations, FALLBACK)
        else:
            fallback_decision = None
        chosen = fallback_decision['chosen'] if use_fallback else main_decision['chosen']
        final_policies = {name: policies(yd, e['scores'].mean(axis=0)) for name, e in evaluations.items()}
        final_models = {name: TrainingOnlyLogistic(name if name in MAIN else 'none', .1, 42).fit(matrices[name][dev], yd)
                        for name in evaluations}
        exported = {name: export_fit(m, feature_sets[name], final_policies[name]) for name, m in final_models.items()}
        save_json(output/'candidate_models.json', exported)
        save_json(output/'decision_before_test.json', dict(balancing=main_decision, fallback=fallback_decision,
            chosen=chosen, policy_action='retain existing shipped model/policy' if chosen == 'none' else 'candidate requires further validation',
            final_thresholds=final_policies, feature_sets=feature_sets, test_used_for_selection=False))
        fold_rows = [r for e in evaluations.values() for r in e['folds']]
        repeat_rows = [r for e in evaluations.values() for r in e['repeats']]
        keys = ['tp', 'fp', 'fn', 'tn', 'recall', 'precision', 'f2', 'fpr', 'roc_auc', 'ap', 'brier',
                'mean_probability', 'calibration_in_large', 'calibration_intercept', 'calibration_slope']
        for name, rows in [('outer_folds', fold_rows), ('repeat_metrics', repeat_rows),
            ('outer_summary', summary(fold_rows, ['variant', 'policy'], ['threshold', *keys])),
            ('repeat_summary', summary(repeat_rows, ['variant', 'policy'], keys)),
            ('fit_counts', [r for e in evaluations.values() for r in e['fit_counts']])]:
            write_csv(output/(name+'.csv'), rows)
        oof_rows = []
        for name, e in evaluations.items():
            for repeat in range(3):
                for i in range(len(yd)):
                    oof_rows.append(dict(variant=name, repeat=repeat, development_row=i, target=int(yd[i]),
                        score=float(e['scores'][repeat, i]), f2_prediction=int(e['predictions']['f2'][repeat, i]),
                        fpr145_prediction=int(e['predictions']['fpr145'][repeat, i])))
        write_csv(output/'development_oof.csv', oof_rows)
        bins = [r for e in evaluations.values() for r in e['bins']]
        # Only after writing the development-only decision do we generate Test predictions.
        yt = y[test]
        test_scores = {name: m.predict_proba(matrices[name][test])[:, 1] for name, m in final_models.items()}
        export_errors = {name: float(np.max(abs(PortableModel(exported[name]).predict_proba(matrices[name][test])[:, 1]-p)))
                         for name, p in test_scores.items()}
        if max(export_errors.values()) > 1e-12:
            raise ValueError('Portable research model replay differs from fitted estimator')
        test_rows = [dict(variant=name, policy=policy, threshold=choice['threshold'],
                         **metrics(yt, test_scores[name], choice['threshold']))
                     for name in evaluations for policy, choice in final_policies[name].items()]
        write_csv(output/'test_metrics.csv', test_rows)
        write_csv(output/'test_scores.csv', [dict(test_row=i, target=int(yt[i]),
            **{name: float(p[i]) for name, p in test_scores.items()}) for i in range(len(test))])
        for name, e in evaluations.items():
            cuts = np.unique(np.quantile(e['scores'].mean(axis=0), [.2, .4, .6, .8]))
            bins += reliability_bins(yt, test_scores[name], cuts, name, 'historical_test')
        write_csv(output/'calibration_bins.csv', bins)
        print('Bootstrapping fixed Test predictions', flush=True)
        cis, differences = bootstrap_test(yt, test_scores, final_policies, PROTOCOL['bootstrap_repetitions'])
        write_csv(output/'test_bootstrap_ci.csv', cis)
        write_csv(output/'test_paired_difference_ci.csv', differences)
        saved_rows = read_csv(repository/'round3/results/nhefs_20261105/test_pair_scores.csv')
        saved = np.array([float(r['new_score']) for r in saved_rows])
        old_state = json.loads((repository/'round3/results/nhefs_20261105/model.json').read_text())
        portable = PortableModel(old_state).predict_proba(x[test])[:, 1]
        if not np.array_equal(yt, np.array([int(r['target']) for r in saved_rows])):
            raise ValueError('Test order/labels differ from historical scores')
        delta = float(np.max(abs(portable-saved)))
        refit_delta = float(np.max(abs(test_scores['none']-saved)))
        if delta > 1e-12 or refit_delta > 1e-8:
            raise ValueError('Historical score reproduction failed')
        verification = dict(status='completed', source_commit='4f45d68b7ffa6e665392412d3d0f0d2569f947b3',
            python=platform.python_version(), numpy=np.__version__, sklearn=sklearn.__version__, scipy=scipy.__version__,
            protocol_sha256=digest(repository/'round3/BALANCING_PROTOCOL_TH.md'),
            code_sha256=digest(__file__), baseline_portable_max_error=delta, baseline_refit_max_error=refit_delta,
            candidate_export_max_errors=export_errors,
            decision_sha256=digest(output/'decision_before_test.json'),
            fallback_executed=use_fallback, chosen_on_development=chosen,
            outcome='reported first diagnosis within 8 years, not measured disease',
            current_model_replaced=False, test_selected_anything=False,
            uncertainty='fold/repeat SD descriptive; Test bootstrap conditional on fixed fitted models',
            publication='Anonymous row indices only; original participant IDs/raw survey files stay local')
        save_json(output/'verification.json', verification)
        save_json(output/'study_status.json', dict(status='completed', real_data_metrics_available=True,
            chosen_on_development=chosen, fallback_executed=use_fallback))
        print('Completed:', chosen, flush=True)
        return verification
    except Exception as exc:
        save_json(output/'study_status.json', dict(status='failed', real_data_metrics_available=False,
            error_type=type(exc).__name__, error=str(exc)))
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True, help='Local raw/ directory parent; missing files downloaded from CDC')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    run(args.source, args.output)
