"""NHEFS round 3: preserve the original target; compare development candidates.

Run from the repository root: python -m round3.workflow --work-dir round3_work
This executes the reviewed core cell of the original notebook, not its saved outputs.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import platform
import zipfile

import numpy as np
import sklearn
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

REPO = Path(__file__).resolve().parents[1]
NOTEBOOK = REPO / 'LEPTOLIC_NHEFS_Future_Prediction.ipynb'
DISCLAIMER = ('คะแนนทดลองของกลุ่ม NHEFS สหรัฐฯ ในอดีต สำหรับการรายงานวินิจฉัยใน 8 ปี '
              'ยังไม่ยืนยันกับคนไทย ไม่ใช่การวินิจฉัยหรือคำแนะนำการรักษา '
              'ระดับ 1 ไม่ได้แปลว่าไม่มีโอกาสเกิดเหตุการณ์')


def load_original():
    notebook = json.loads(NOTEBOOK.read_text(encoding='utf-8'))
    core = ''.join(notebook['cells'][1]['source'])
    scope = {}
    exec(compile(core, str(NOTEBOOK) + ':cell1', 'exec'), scope)
    return scope


def save_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


def wilson(events, n):
    """Approximate binomial 95% CI; descriptive, not survey weighted."""
    if n == 0:
        return None
    z = 1.959963984540054
    p = events / n
    denom = 1 + z*z/n
    center = (p + z*z/(2*n)) / denom
    half = z * math.sqrt(p*(1-p)/n + z*z/(4*n*n)) / denom
    return [max(0., center-half), min(1., center+half)]


def level_cuts(development_scores):
    cuts = np.quantile(development_scores, [1/3, 2/3])
    if not np.isfinite(cuts).all() or cuts[0] >= cuts[1]:
        raise ValueError('Scores do not support three distinct levels; do not invent cutoffs')
    return cuts


def levels(scores, cuts):
    # Equal-to-cutoff goes to the higher level; ties are never split by person ID.
    return np.searchsorted(cuts, scores, side='right') + 1


def band_report(y, scores, cuts):
    assigned = levels(scores, cuts)
    result = []
    for band in (1, 2, 3):
        mask = assigned == band
        n, events = int(mask.sum()), int(y[mask].sum())
        result.append({'level': band, 'n': n, 'events': events,
                       'observed_rate': events/n if n else None,
                       'observed_rate_95ci_wilson': wilson(events, n),
                       'mean_score': float(scores[mask].mean()) if n else None})
    return result


def select_threshold(y, scores):
    """Predeclared demo objective: maximize F2 on Tune, favor recall.

    F2 gives recall more weight; it is not a clinically optimized objective.
    Ties favor higher precision then a higher threshold. Test is never an input.
    """
    best = None
    for threshold in np.unique(scores):
        pred = scores >= threshold
        tp = int(np.sum(pred & (y == 1)))
        fp = int(np.sum(pred & (y == 0)))
        fn = int(np.sum(~pred & (y == 1)))
        f2 = 5*tp / (5*tp + 4*fn + fp) if 5*tp + 4*fn + fp else 0.
        precision = tp/(tp+fp) if tp+fp else 0.
        key = (f2, precision, float(threshold))
        if best is None or key > best:
            best = key
    if best is None:
        raise ValueError('Empty Tune scores')
    return {'threshold': best[2], 'tune_f2': best[0], 'objective': 'max_F2_on_tune_only'}


def evaluate(y, scores, threshold, original):
    report = original['metrics'](y, scores)
    pred = scores >= threshold
    tp = int(np.sum(pred & (y == 1))); fp = int(np.sum(pred & (y == 0)))
    tn = int(np.sum(~pred & (y == 0))); fn = int(np.sum(~pred & (y == 1)))
    precision = tp/(tp+fp) if tp+fp else None
    recall = tp/(tp+fn) if tp+fn else None
    report.update(threshold=float(threshold), confusion_matrix=dict(tn=tn, fp=fp, fn=fn, tp=tp),
                  precision=precision, recall=recall, specificity=tn/(tn+fp) if tn+fp else None,
                  accuracy=(tp+tn)/len(y),
                  balanced_accuracy=(recall + tn/(tn+fp))/2 if recall is not None and tn+fp else None,
                  f1=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else None,
                  recall_95ci_wilson=wilson(tp, tp+fn),
                  precision_95ci_wilson=wilson(tp, tp+fp),
                  average_precision=float(average_precision_score(y, scores)))
    return report


def bootstrap(y, scores, original, repetitions=1000):
    rng = np.random.default_rng(42)
    aucs, briers = [], []
    for _ in range(repetitions):
        ids = rng.integers(0, len(y), len(y))
        a = original['auc'](y[ids], scores[ids])
        if a is not None:
            aucs.append(a)
        briers.append(float(np.mean((scores[ids]-y[ids])**2)))
    return {'repetitions': repetitions, 'valid_auc_repetitions': len(aucs),
            'roc_auc_95ci': np.quantile(aucs, [.025, .975]).tolist() if aucs else None,
            'brier_95ci': np.quantile(briers, [.025, .975]).tolist()}


def calibration_table(y, scores):
    # Five fixed score intervals, including empty bins; no fitted test cutoffs.
    result = []
    for lo, hi in zip(np.arange(0, 1, .2), np.arange(.2, 1.2, .2)):
        mask = (scores >= lo) & ((scores < hi) if hi < .999 else (scores <= 1))
        n = int(mask.sum()); events = int(y[mask].sum())
        result.append(dict(lower=float(lo), upper=min(1., float(hi)), n=n, events=events,
                           mean_score=float(scores[mask].mean()) if n else None,
                           observed_rate=events/n if n else None,
                           observed_rate_95ci=wilson(events, n)))
    return result


def data_quality(root, cohort, original):
    # Retain exclusion reasons and compare baseline distributions of included/excluded.
    config = original['WAVES']['82']
    raw = root/'raw'
    spec = original['parse_layout']((raw/config['layout']).read_text(encoding='utf-8-sig'))
    baseline = original['read_wave'](raw/config['data'], spec, config['fields'], config['id'])
    included_ids = {r['participant_id'] for r in cohort}
    result = {}
    for name, included in [('included', True), ('excluded', False)]:
        rows = [original['baseline_features'](row) for ident, row in baseline.items()
                if (ident in included_ids) == included]
        group = {'n': len(rows), 'features': {}}
        for f in original['FEATURES']:
            values = np.array([r[f] for r in rows], float)
            finite = values[np.isfinite(values)]
            group['features'][f] = {'missing_n': int((~np.isfinite(values)).sum()),
                                    'median': float(np.median(finite)) if len(finite) else None,
                                    'min': float(finite.min()) if len(finite) else None,
                                    'max': float(finite.max()) if len(finite) else None}
        result[name] = group
    return result


def fit_candidates(x, y, split):
    records, fitted = [], []
    for feature_n in (4, 5):
        for family, param in [('logistic', c) for c in (.01, .1, 1., 10., 100.)] + [('tree', 2), ('tree', 3)]:
            estimator = (LogisticRegression(C=param, max_iter=2000, random_state=42)
                         if family == 'logistic' else
                         DecisionTreeClassifier(max_depth=param, min_samples_leaf=50, random_state=42))
            model = make_pipeline(SimpleImputer(strategy='median'), StandardScaler(), estimator)
            model.fit(x[split['train'], :feature_n], y[split['train']])
            p = model.predict_proba(x[split['tune'], :feature_n])[:, 1]
            record = dict(features=feature_n, family=family, parameter=param,
                          parameter_name='C' if family == 'logistic' else 'max_depth',
                          tune_brier=float(np.mean((p-y[split['tune']])**2)))
            records.append(record); fitted.append(model)
    winner = min(range(len(records)), key=lambda i: records[i]['tune_brier'])
    return records, records[winner], fitted[winner]


def predict_demo(model, calibration, original, ranges, cuts, values, previously_diagnosed=False):
    if previously_diagnosed:
        raise ValueError('Outside cohort: prior diagnosis')
    features = list(ranges)
    row, warnings = [], []
    for f in features:
        value = values.get(f)
        if value is None or (isinstance(value, float) and math.isnan(value)):
            if f != 'sleep_hours':
                raise ValueError('Missing required input: '+f)
            row.append(np.nan); warnings.append('Missing sleep imputed from Train'); continue
        value = float(value)
        if not math.isfinite(value) or not ranges[f][0] <= value <= ranges[f][1]:
            raise ValueError('Outside training range: '+f)
        if f == 'sex_female' and value not in (0, 1):
            raise ValueError('sex_female must be 0 or 1')
        row.append(value)
    if float(values['baseline_sbp']) >= 130 or float(values['baseline_dbp']) >= 80:
        raise ValueError('Outside baseline BP eligibility')
    raw = float(model.predict_proba([row])[0, 1])
    score = float(apply_calibration(np.array([raw]), calibration, original)[0])
    return dict(experimental_probability_8y=score, level=int(levels([score], cuts)[0]),
                meaning='ระดับสัมพัทธ์ตามจุดแบ่งคะแนน Tune ของ NHEFS',
                warnings=warnings, limitations=DISCLAIMER)


def apply_calibration(scores, coefficients, original):
    p = np.clip(scores, 1e-12, 1-1e-12)
    return original['sigmoid'](coefficients[0] + coefficients[1]*np.log(p/(1-p)))


def run(root):
    root.mkdir(parents=True, exist_ok=True)
    out = root/'round3_outputs'; out.mkdir(exist_ok=True)
    if any(out.iterdir()):
        raise ValueError('Output directory is not empty; choose a new --work-dir to preserve previous runs')
    original = load_original()
    cohort, audit = original['prepare_cohort'](root)
    save_json(out/'data_quality.json', data_quality(root, cohort, original))
    features = original['FEATURES']
    x = np.array([[r[f] for f in features] for r in cohort], float)
    y = np.array([r['target'] for r in cohort], int)
    split = original['split_ids'](y)
    if any(len(np.unique(y[ids])) != 2 for ids in split.values()):
        raise ValueError('Each split must contain both classes')
    records, selected, model = fit_candidates(x, y, split)
    n = selected['features']
    raw = model.predict_proba(x[:, :n])[:, 1]
    cal = split['calibration']
    p = np.clip(raw[cal], 1e-12, 1-1e-12)
    coefficients = original['fit_logistic'](np.log(p/(1-p))[:, None], y[cal], ridge=.1)
    if coefficients[1] <= 0:
        raise ValueError('Nonpositive calibration slope; review before publishing demo')
    calibrated = apply_calibration(raw, coefficients, original)
    tune, test = split['tune'], split['test']
    cuts = level_cuts(calibrated[tune])
    threshold = select_threshold(y[tune], calibrated[tune])
    t = threshold['threshold']
    test_report = evaluate(y[test], calibrated[test], t, original)
    results = {'status': 'executed_on_official_CDC_files', 'seed': 42,
               'source_notebook_sha256': hashlib.sha256(NOTEBOOK.read_bytes()).hexdigest(),
               'versions': dict(python=platform.python_version(), numpy=np.__version__, sklearn=sklearn.__version__),
               'cohort_n': len(cohort), 'events': int(y.sum()),
               'splits': {k: dict(n=len(ids), events=int(y[ids].sum())) for k, ids in split.items()},
               'development_candidates': records, 'selected_on_tune_brier': selected,
               'calibration': {'policy': 'Platt retained a priori; no test-based selection',
                               'coefficients': coefficients},
               'threshold_choice': threshold, 'level_cutoffs_from_tune': cuts.tolist(),
               'tune_levels': band_report(y[tune], calibrated[tune], cuts),
               'test_levels': band_report(y[test], calibrated[test], cuts),
               'test_selected': test_report,
               'test_at_0_5': evaluate(y[test], calibrated[test], .5, original),
               'test_uncalibrated_diagnostic': evaluate(y[test], raw[test], .5, original),
               'test_constant_baseline': evaluate(y[test], np.full(len(test), y[split['train']].mean()), .5, original),
               'test_uncertainty': bootstrap(y[test], calibrated[test], original),
               'test_calibration_fixed_bins': calibration_table(y[test], calibrated[test]),
               'test_status': 'Previously reported test reused; not a new untouched external validation',
               'limitations': DISCLAIMER}
    save_json(out/'metrics.json', results)
    split_name = {int(i): k for k, ids in split.items() for i in ids}
    original['write_csv'](out/'split_manifest.csv',
                          [dict(participant_id=r['participant_id'], split=split_name[i], target=r['target'])
                           for i, r in enumerate(cohort)], ['participant_id', 'split', 'target'])
    original['write_csv'](out/'test_predictions.csv',
                          [dict(participant_id=cohort[i]['participant_id'], target=int(y[i]),
                                score=float(calibrated[i]), level=int(levels([calibrated[i]], cuts)[0])) for i in test],
                          ['participant_id', 'target', 'score', 'level'])
    ranges = {f: [float(np.nanmin(x[split['train'], j])), float(np.nanmax(x[split['train'], j]))]
              for j, f in enumerate(features[:n])}
    demo = {f: float(np.nanmedian(x[split['train'], j])) for j, f in enumerate(features[:n])}
    demo['sex_female'] = int(round(demo['sex_female']))
    save_json(out/'demo.json', predict_demo(model, coefficients, original, ranges, cuts, demo))
    # Portable JSON parameters, never pickle/load untrusted model files.
    estimator = model.steps[-1][1]
    state = {'selected': selected, 'features': features[:n], 'training_range': ranges,
             'imputer_median': model.steps[0][1].statistics_.tolist(),
             'scaler_mean': model.steps[1][1].mean_.tolist(), 'scaler_scale': model.steps[1][1].scale_.tolist(),
             'calibration': coefficients, 'level_cuts': cuts.tolist(), 'threshold': t,
             'limitations': DISCLAIMER}
    if selected['family'] == 'logistic':
        state.update(coefficients=estimator.coef_.tolist(), intercept=estimator.intercept_.tolist())
    else:
        tree = estimator.tree_
        state.update(tree={k: getattr(tree, k).tolist() for k in
                           ['children_left', 'children_right', 'feature', 'threshold', 'value']})
    save_json(out/'model.json', state)
    with zipfile.ZipFile(root/'LEPTOLIC_round3_outputs.zip', 'w', zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(out.iterdir()):
            if path.is_file(): archive.write(path, path.name)
    print(json.dumps({'selected': selected, 'test': test_report, 'output': str(out)}, indent=2))
    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--work-dir', type=Path, default=REPO/'round3_work')
    args = parser.parse_args()
    run(args.work_dir.resolve())
