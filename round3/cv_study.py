"""Predeclared nested repeated-CV study. Never selects using the historical Test."""
import argparse
import hashlib
import json
from pathlib import Path
import platform

import numpy as np
import sklearn
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.calibration import CalibratedClassifierCV
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import GridSearchCV, RepeatedStratifiedKFold, StratifiedKFold, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from .workflow import (DISCLAIMER, NOTEBOOK, band_report, data_quality, evaluate,
                       level_cuts, load_original, save_json, select_threshold, wilson)

PROTOCOL = dict(seed=42, outer_folds=5, repeats=3, inner_folds=3,
                calibration_folds=3, primary='paired_Test_Brier_old_minus_new',
                bootstrap_repetitions=2000,
                improvement_rule='95% paired-bootstrap Brier difference CI strictly above zero',
                secondary=['ROC_AUC', 'average_precision', 'log_loss', 'recall', 'precision'],
                warning='Historical Test was previously exposed; no external/Thai validation')


class FeatureSlice(TransformerMixin, BaseEstimator):
    def __init__(self, count=4):
        self.count = count

    def fit(self, x, y=None):
        if np.any(np.all(np.isnan(np.asarray(x)[:, :self.count]), axis=0)):
            raise ValueError('A candidate feature is entirely missing in this training fold')
        return self

    def transform(self, x):
        return np.asarray(x)[:, :self.count]


def candidate_grid(seed, inner=3, calibration_folds=3):
    """Calibration is fitted inside inner training folds, never on their validation."""
    estimators, params = [], []
    for count in (4, 5):
        choices = [('logistic', c) for c in (.1, 1., 10.)] + [('tree', d) for d in (2, 3)]
        for family, value in choices:
            base = (LogisticRegression(C=value, max_iter=2000, random_state=seed) if family == 'logistic'
                    else DecisionTreeClassifier(max_depth=value, min_samples_leaf=50, random_state=seed))
            for calibration in ('none', 'sigmoid'):
                pipeline = make_pipeline(FeatureSlice(count), SimpleImputer(strategy='median'),
                                         StandardScaler(), base)
                model = pipeline if calibration == 'none' else CalibratedClassifierCV(
                    pipeline, method='sigmoid', ensemble=False,
                    cv=StratifiedKFold(calibration_folds, shuffle=True, random_state=seed))
                estimators.append(model)
                params.append(dict(features=count, family=family, value=value, calibration=calibration))
    # One wrapper parameter lets GridSearchCV compare complete calibrated pipelines.
    search = GridSearchCV(make_pipeline(estimators[0]),
                          {'pipeline': estimators}, scoring='neg_brier_score',
                          cv=StratifiedKFold(inner, shuffle=True, random_state=seed),
                          n_jobs=1, error_score=np.nan, refit=True)
    return search, params


def search_model(x, y, seed, inner, calibration_folds):
    search, params = candidate_grid(seed, inner, calibration_folds)
    search.fit(x, y)
    records = []
    for i, p in enumerate(params):
        score = search.cv_results_['mean_test_score'][i]
        records.append({**p, 'mean_inner_brier': float(-score) if np.isfinite(score) else None,
                        'status': 'valid' if np.isfinite(score) else 'failed_training_fold'})
    index = search.best_index_
    if not np.isfinite(search.best_score_):
        raise ValueError('No valid candidate')
    return search.best_estimator_.steps[0][1], records[index], records


def fit_legacy(x, y, original, split=None):
    """Original NumPy algorithm; same five ridges and separate Platt calibration."""
    if split is None:
        ids = np.arange(len(y))
        train, remainder = train_test_split(ids, train_size=55/85, stratify=y, random_state=42)
        tune, calibration = train_test_split(remainder, test_size=.5, stratify=y[remainder], random_state=42)
    else:
        train, tune, calibration = (split[k] for k in ('train', 'tune', 'calibration'))
    prep = original['fit_preprocessor'](x[train]); z = original['transform'](x, prep)
    choices = []
    for ridge in (.01, .1, 1., 10., 100.):
        coef = original['fit_logistic'](z[train], y[train], ridge)
        p = original['sigmoid'](original['logits'](z[tune], coef))
        choices.append((brier_score_loss(y[tune], p), ridge, coef))
    _, ridge, coef = min(choices, key=lambda item: item[0])
    score = original['logits'](z[calibration], coef)
    cal = original['fit_logistic'](score[:, None], y[calibration], ridge=.1)
    if cal[1] <= 0:
        raise ValueError('Legacy calibration slope is nonpositive')

    def predict(values):
        z = original['transform'](values, prep)
        return original['sigmoid'](cal[0]+cal[1]*original['logits'](z, coef))
    return predict, {'ridge': ridge, 'calibration': cal}


def target_guard(row, followups, original):
    """Conservative additional exclusion, pending official codebook confirmation.

    Never label someone event-free through the horizon if a recorded death year
    precedes/overlaps that horizon. No invented death date within its year.
    """
    deaths = []
    for wave, follow in followups.items():
        year = original['integer'](follow.get('NEWDODYR' if wave == '92' else 'DEATHYR'))
        if year is not None:
            if 20 <= year <= 93:
                year += 1900
            if 1971 <= year <= 1993:
                deaths.append(year)
    if row['target'] == 0 and deaths and min(deaths) <= int(row['horizon_date'][:4]):
        return 'negative_with_death_before_or_overlapping_horizon'
    return None


def guarded_cohort(root, cohort, original):
    waves = {}
    for wave, config in original['WAVES'].items():
        raw = root/'raw'
        layout = original['parse_layout']((raw/config['layout']).read_text(encoding='utf-8-sig'))
        waves[wave] = original['read_wave'](raw/config['data'], layout, config['fields'], config['id'])
    kept, excluded = [], []
    for row in cohort:
        ident = row['participant_id']
        follow = {w: rows[ident] for w, rows in waves.items() if w != '82' and ident in rows}
        reason = target_guard(row, follow, original)
        if reason:
            excluded.append(dict(participant_id=ident, reason=reason))
        else:
            kept.append(row)
    joins = {w: dict(rows=len(rows), matched_baseline=len(set(rows) & set(waves['82'])),
                    outside_baseline=len(set(rows)-set(waves['82']))) for w, rows in waves.items()}
    return kept, excluded, joins


def paired_bootstrap(y, old, new, repetitions=2000):
    rng = np.random.default_rng(42)
    samples = {k: [] for k in ('brier_old_minus_new', 'auc_new_minus_old', 'ap_new_minus_old')}
    for _ in range(repetitions):
        ids = rng.integers(0, len(y), len(y)); target = y[ids]
        samples['brier_old_minus_new'].append(float(np.mean((old[ids]-target)**2-(new[ids]-target)**2)))
        if len(np.unique(target)) == 2:
            samples['auc_new_minus_old'].append(roc_auc_score(target, new[ids])-roc_auc_score(target, old[ids]))
            samples['ap_new_minus_old'].append(average_precision_score(target, new[ids])-average_precision_score(target, old[ids]))
    result = {k: dict(ci95=np.quantile(values, [.025, .975]).tolist(), valid_draws=len(values))
              for k, values in samples.items()}
    result['point_differences'] = dict(
        brier_old_minus_new=float(np.mean((old-y)**2-(new-y)**2)),
        auc_new_minus_old=float(roc_auc_score(y, new)-roc_auc_score(y, old)),
        ap_new_minus_old=float(average_precision_score(y, new)-average_precision_score(y, old)))
    result['primary_supported'] = result['brier_old_minus_new']['ci95'][0] > 0
    result['scope'] = 'conditional on fixed fitted models and historical Test; no training uncertainty or external validation'
    return result


def reliability_report(y, scores, development_scores):
    cuts = np.unique(np.quantile(development_scores, [.2, .4, .6, .8]))
    assigned = np.searchsorted(cuts, scores, side='right')
    bins = []
    for index in range(len(cuts)+1):
        mask = assigned == index; n = int(mask.sum()); events = int(y[mask].sum())
        bins.append(dict(n=n, events=events, mean_score=float(scores[mask].mean()) if n else None,
                         observed_rate=events/n if n else None, observed_rate_ci95=wilson(events, n)))
    return dict(development_quantile_cutoffs=cuts.tolist(), test_bins=bins)


def write_report(root, result):
    comparison = result['paired_test_comparison']
    conclusion = ('มีหลักฐานภายใน Test เดิมว่า Brier ลดลงตามเกณฑ์ที่กำหนด'
                  if comparison['primary_supported'] else
                  'ยังไม่มีหลักฐานเพียงพอว่า Brier ดีขึ้นจริงตามเกณฑ์ที่กำหนด')
    lines = ['# ผลการทดลอง NHEFS: repeated nested cross-validation', '', conclusion, '',
             'นี่เป็น internal validation บน Test ที่เคยเปิดเผยแล้ว ไม่ใช่ external validation หรือการยืนยันกับคนไทย', '',
             f"Development {result['development_n']} คน / {result['development_events']} เหตุการณ์; "
             f"Test {result['test_n']} คน / {result['test_events']} เหตุการณ์", '',
             '| Metric | โมเดลเดิม threshold 0.5 | โมเดลใหม่ threshold OOF |',
             '|---|---:|---:|']
    for key in ['roc_auc', 'average_precision', 'brier_score', 'log_loss', 'recall', 'precision', 'specificity']:
        lines.append(f"| {key} | {result['test_legacy_at_0_5'][key]} | {result['test_new'][key]} |")
    lines += ['', 'การเปรียบเทียบ recall ข้างบนรวมผลจาก threshold ที่ต่างกัน '
              'ต้องอ่าน test_legacy_with_same_threshold_policy และ test_new_at_0_5 ใน metrics.json ประกอบ', '',
              'ผลต่าง paired bootstrap (ค่าบวกหมายถึงดีขึ้นตามทิศทางที่ระบุ):', '',
              '```json', json.dumps(comparison, ensure_ascii=False, indent=2), '```', '',
              'ระดับคะแนน:', '', '```json',
              json.dumps(result['test_levels'], ensure_ascii=False, indent=2), '```', '',
              'ค่าเฉลี่ย/SD ของแต่ละ fold ใช้แสดงความแกว่งเท่านั้น folds และ repeats ไม่เป็นอิสระต่อกัน', '']
    for metric in ['roc_auc', 'average_precision', 'brier_score']:
        for model in ['legacy', 'new']:
            values = [fold[model][metric] for fold in result['folds']]
            lines.append(f"- Outer {model} {metric}: mean={np.mean(values):.6f}, SD={np.std(values, ddof=1):.6f}")
    lines += ['', 'ข้อจำกัด: การเลือก 20 candidates, ข้อมูลย้อนหลัง, reported diagnosis, '
              'จำนวนเหตุการณ์น้อย และ conservative death guard ยังต้องยืนยัน codebook', '', DISCLAIMER]
    (root/'report_TH.md').write_text('\n'.join(lines), encoding='utf-8')


def export_model(model, selected, x, features, cuts, threshold):
    if selected['calibration'] == 'sigmoid':
        calibrated = model.calibrated_classifiers_[0]
        pipeline = calibrated.estimator
        sigmoid = calibrated.calibrators[0]
        calibration = dict(method='sigmoid', a=float(sigmoid.a_), b=float(sigmoid.b_),
                           input='margin' if selected['family'] == 'logistic' else 'probability')
    else:
        pipeline = model
        calibration = dict(method='none')
    estimator = pipeline.steps[-1][1]; n = selected['features']
    state = dict(selected=selected, features=features[:n],
                 training_range={f: [float(np.nanmin(x[:, j])), float(np.nanmax(x[:, j]))]
                                 for j, f in enumerate(features[:n])},
                 imputer_median=pipeline.steps[1][1].statistics_.tolist(),
                 scaler_mean=pipeline.steps[2][1].mean_.tolist(),
                 scaler_scale=pipeline.steps[2][1].scale_.tolist(),
                 study_calibration=calibration, level_cuts=cuts.tolist() if cuts is not None else None,
                 threshold=threshold, limitations=DISCLAIMER)
    if selected['family'] == 'logistic':
        state.update(coefficients=estimator.coef_.tolist(), intercept=estimator.intercept_.tolist())
    else:
        state['tree'] = {k: getattr(estimator.tree_, k).tolist() for k in
                         ['children_left', 'children_right', 'feature', 'threshold', 'value']}
    return state


def experiment(cohort, split, original, protocol):
    features = original['FEATURES']
    x = np.array([[r[f] for f in features] for r in cohort], float)
    y = np.array([r['target'] for r in cohort], int)
    development = np.concatenate([split[k] for k in ('train', 'tune', 'calibration')])
    development.sort(); test = split['test']
    xd, yd = x[development], y[development]
    if any(len(np.unique(y[ids])) != 2 for ids in split.values()):
        raise ValueError('Each preserved historical split must retain both outcome classes')
    if not np.isfinite(x[:, :4]).all() or np.isnan(xd[:, 4]).all():
        raise ValueError('Need four complete baseline features and observed development sleep to compare legacy')
    oof = np.full((protocol['repeats'], len(development)), np.nan)
    legacy_oof = np.full_like(oof, np.nan); folds = []
    outer = RepeatedStratifiedKFold(n_splits=protocol['outer_folds'], n_repeats=protocol['repeats'], random_state=42)
    for index, (train, valid) in enumerate(outer.split(xd, yd)):
        model, selected, _ = search_model(xd[train], yd[train], 100+index,
                                           protocol['inner_folds'], protocol['calibration_folds'])
        new = model.predict_proba(xd[valid])[:, 1]
        baseline, _ = fit_legacy(xd[train], yd[train], original)
        old = baseline(xd[valid]); repeat = index//protocol['outer_folds']
        oof[repeat, valid] = new; legacy_oof[repeat, valid] = old
        folds.append(dict(repeat=repeat, fold=index % protocol['outer_folds'], selected=selected,
                          n=len(valid), events=int(yd[valid].sum()),
                          new=evaluate(yd[valid], new, .5, original),
                          legacy=evaluate(yd[valid], old, .5, original)))
    if not np.isfinite(oof).all() or not np.isfinite(legacy_oof).all():
        raise ValueError('Incomplete out-of-fold coverage')
    # Each person contributes once to threshold/band fitting, by averaging repeats.
    development_score = oof.mean(axis=0); old_development_score = legacy_oof.mean(axis=0)
    threshold = select_threshold(yd, development_score)
    old_threshold = select_threshold(yd, old_development_score)
    try:
        cuts = level_cuts(development_score); band_status = 'three_nonempty_development_bands'
    except ValueError as exc:
        cuts = None; band_status = str(exc)
    # Final tuning still only sees development; Test first used after all choices.
    model, selected, candidates = search_model(xd, yd, 42, protocol['inner_folds'], protocol['calibration_folds'])
    baseline, baseline_fit = fit_legacy(x, y, original, split)
    new = model.predict_proba(x[test])[:, 1]; old = baseline(x[test])
    constant = np.full(len(test), y[split['train']].mean())
    result = dict(protocol=protocol, development_n=len(yd), development_events=int(yd.sum()),
                  test_n=len(test), test_events=int(y[test].sum()), folds=folds,
                  final_selected=selected, final_inner_candidates=candidates,
                  threshold=threshold, legacy_threshold=old_threshold, level_status=band_status,
                  level_cuts=cuts.tolist() if cuts is not None else None,
                  test_new=evaluate(y[test], new, threshold['threshold'], original),
                  test_new_at_0_5=evaluate(y[test], new, .5, original),
                  test_legacy_at_0_5=evaluate(y[test], old, .5, original),
                  test_legacy_with_same_threshold_policy=evaluate(y[test], old, old_threshold['threshold'], original),
                  test_constant=evaluate(y[test], constant, .5, original), legacy_fit=baseline_fit,
                  paired_test_comparison=paired_bootstrap(y[test], old, new, protocol['bootstrap_repetitions']),
                  test_calibration_new=reliability_report(y[test], new, development_score),
                  test_calibration_legacy=reliability_report(y[test], old, old_development_score),
                  development_levels=band_report(yd, development_score, cuts) if cuts is not None else None,
                  test_levels=band_report(y[test], new, cuts) if cuts is not None else None,
                  limitations=DISCLAIMER)
    rows = [dict(participant_id=cohort[i]['participant_id'], target=int(y[i]),
                 legacy_score=float(a), new_score=float(b)) for i, a, b in zip(test, old, new)]
    oof_rows = [dict(participant_id=cohort[i]['participant_id'], target=int(y[i]), repeat=r,
                     legacy_score=float(legacy_oof[r, j]), new_score=float(oof[r, j]))
                for r in range(protocol['repeats']) for j, i in enumerate(development)]
    state = export_model(model, selected, xd, features, cuts, threshold['threshold'])
    return result, rows, oof_rows, state


def run_study(root, data_zip=None):
    root.mkdir(parents=True, exist_ok=True)
    status = root/'study_status.json'
    if status.exists():
        raise ValueError('Use a new work directory for each attempt; preserve the previous run status')
    save_json(root/'protocol.json', PROTOCOL)
    save_json(status, dict(status='started', real_data_metrics_available=False))
    try:
        input_manifest = None
        if data_zip is not None:
            from .data_archive import import_data_zip
            input_manifest = import_data_zip(data_zip, root)
        original = load_original()
        initial, audit = original['prepare_cohort'](root)
        if input_manifest is not None:
            audit['data_origin'] = 'user_supplied_NHEFS_format_archive'
            audit['provenance_note'] = input_manifest['provenance_note']
            for source in audit['sources']:
                source['url_role'] = 'expected_CDC_URL_not_fetched_in_this_run'
            save_json(root/'outputs'/'NHEFS_Data_Audit.json', audit)
        initial_split = original['split_ids']([r['target'] for r in initial])
        initial_membership = {initial[int(i)]['participant_id']: name
                              for name, ids in initial_split.items() for i in ids}
        cohort, excluded, joins = guarded_cohort(root, initial, original)
        split = {name: np.array([i for i, row in enumerate(cohort)
                                if initial_membership[row['participant_id']] == name], int)
                 for name in initial_split}
        original['write_csv'](root/'split_manifest.csv',
                              [dict(participant_id=row['participant_id'], target=row['target'],
                                    split=initial_membership[row['participant_id']]) for row in cohort],
                              ['participant_id', 'target', 'split'])
        original['write_csv'](root/'additional_exclusions.csv', excluded, ['participant_id', 'reason'])
        quality = dict(summary=data_quality(root, cohort, original), joins=joins,
                       additional_exclusions=excluded, original_data_audit=audit,
                       input_archive=input_manifest,
                       target_note='conservative death guard; codebook confirmation pending')
        save_json(root/'data_audit.json', quality)
        result, predictions, oof, state = experiment(cohort, split, original, PROTOCOL)
        result['versions'] = dict(python=platform.python_version(), numpy=np.__version__, sklearn=sklearn.__version__)
        result['source_notebook_sha256'] = hashlib.sha256(NOTEBOOK.read_bytes()).hexdigest()
        result['study_code_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        result['comparison_scope'] = 'Both models on identical guarded cohort and preserved historical split IDs'
        state['input_provenance'] = dict(data_origin=audit['data_origin'], sources=audit['sources'])
        save_json(root/'metrics.json', result); save_json(root/'model.json', state)
        write_report(root, result)
        columns = ['participant_id', 'target', 'legacy_score', 'new_score']
        original['write_csv'](root/'test_predictions.csv', predictions, columns)
        original['write_csv'](root/'development_oof.csv', oof, ['participant_id', 'target', 'repeat', 'legacy_score', 'new_score'])
        save_json(status, dict(status='completed', real_data_metrics_available=True,
                               primary_improvement_supported=result['paired_test_comparison']['primary_supported']))
        return result
    except Exception as exc:
        save_json(status, dict(status='failed', real_data_metrics_available=False,
                               error_type=type(exc).__name__, error=str(exc)))
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--work-dir', type=Path, required=True)
    parser.add_argument('--data-zip', type=Path, help='Colab ZIP containing four NHEFS data and four layout files; no CDC download required')
    args = parser.parse_args()
    run_study(args.work_dir.resolve(), args.data_zip)
