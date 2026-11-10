"""Choose an empirical OOF false-positive budget before evaluating historical Test."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from .workflow import evaluate, load_original, save_json, wilson


def select_budget_threshold(y, scores, max_fpr=.2):
    y, scores = np.asarray(y), np.asarray(scores, float)
    if y.ndim != 1 or scores.shape != y.shape or not len(y):
        raise ValueError('Expected equally sized nonempty vectors')
    if not np.isfinite(scores).all() or np.any((scores < 0) | (scores > 1)):
        raise ValueError('Scores must be finite probabilities')
    if set(np.unique(y)) != {0, 1} or not 0 <= max_fpr <= 1:
        raise ValueError('Both binary classes and a budget in [0,1] are required')
    negatives = int(np.sum(y == 0)); positives = int(np.sum(y == 1))
    allowed = math.floor(max_fpr*negatives)
    best = None
    candidates = np.r_[np.unique(scores), np.nextafter(scores.max(), np.inf)]
    for threshold in candidates:
        predicted = scores >= threshold
        tp = int(np.sum(predicted & (y == 1))); fp = int(np.sum(predicted & (y == 0)))
        if fp > allowed:
            continue
        # Highest recall; ties prefer fewer false positives, then higher threshold.
        key = (tp, -fp, float(threshold))
        if best is None or key > best:
            best = key
    tp, minus_fp, threshold = best
    return dict(threshold=threshold, max_fpr=max_fpr, allowed_false_positives=allowed,
                development_tp=tp, development_fp=-minus_fp, development_positives=positives,
                development_negatives=negatives, development_recall=tp/positives,
                development_fpr=-minus_fp/negatives,
                objective='maximize recall subject to empirical development OOF FPR budget')


def read_csv(path):
    with path.open(encoding='utf-8-sig') as stream:
        return list(csv.DictReader(stream))


def aggregate_oof(rows, repeats=3):
    grouped = {}; seen = set()
    for row in rows:
        ident, repeat = row['participant_id'], int(row['repeat'])
        if (ident, repeat) in seen or repeat not in range(repeats):
            raise ValueError('Duplicate or unexpected OOF repeat')
        seen.add((ident, repeat))
        target = int(row['target'])
        group = grouped.setdefault(ident, dict(target=target, scores=[]))
        if group['target'] != target:
            raise ValueError('Inconsistent target across repeats')
        group['scores'].append(float(row['new_score']))
    if not grouped or any(len(g['scores']) != repeats for g in grouped.values()):
        raise ValueError('Incomplete OOF coverage')
    ids = list(grouped)
    return ids, np.array([grouped[i]['target'] for i in ids]), np.array([np.mean(grouped[i]['scores']) for i in ids])


def operating_metrics(y, scores, threshold):
    result = evaluate(y, scores, threshold, load_original())
    cm = result['confusion_matrix']; nnegative = cm['fp']+cm['tn']
    result['false_positive_rate'] = cm['fp']/nnegative
    result['false_positive_rate_ci95'] = wilson(cm['fp'], nnegative)
    result['alerts'] = cm['tp']+cm['fp']
    return result


def run(source, output, max_fpr=.2):
    source, output = Path(source), Path(output)
    if output.exists() and any(output.iterdir()):
        raise ValueError('Use an empty output directory')
    output.mkdir(parents=True, exist_ok=True)
    protocol = dict(max_fpr=max_fpr, selection_data='mean repeated development OOF scores per person',
                    tie_rule='maximum TP, minimum FP, highest threshold',
                    test_scope='previously exposed historical Test; exploratory policy update',
                    guarantee='empirical development constraint, not a population or Test guarantee')
    save_json(output/'protocol.json', protocol)
    ids, yd, pd = aggregate_oof(read_csv(source/'development_oof.csv'))
    policy = select_budget_threshold(yd, pd, max_fpr)
    save_json(output/'selection.json', policy)  # Freeze before opening Test predictions.
    test = read_csv(source/'test_predictions.csv')
    if not set(ids).isdisjoint(row['participant_id'] for row in test):
        raise ValueError('Development/Test ID overlap')
    y = np.array([int(row['target']) for row in test]); p = np.array([float(row['new_score']) for row in test])
    original_model = json.loads((source/'model.json').read_text())
    old_threshold = original_model['threshold']
    new_model = dict(original_model, threshold=policy['threshold'], threshold_policy=policy)
    save_json(output/'model.json', new_model)
    result = dict(protocol=protocol, selection=policy,
                  development=operating_metrics(yd, pd, policy['threshold']),
                  test_f2=operating_metrics(y, p, old_threshold),
                  test_budget=operating_metrics(y, p, policy['threshold']),
                  source_hashes={name:hashlib.sha256((source/name).read_bytes()).hexdigest()
                                 for name in ['model.json','development_oof.csv','test_predictions.csv']})
    save_json(output/'metrics.json', result)
    for name, target, score in [('development_scores.csv', yd, pd), ('test_scores.csv', y, p)]:
        with (output/name).open('w', encoding='utf-8', newline='') as stream:
            writer = csv.writer(stream, lineterminator='\n'); writer.writerow(['row','target','score'])
            writer.writerows((i,int(t),float(s)) for i,(t,s) in enumerate(zip(target, score)))
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--max-fpr',type=float,default=.2)
    args = parser.parse_args()
    result = run(args.source,args.output,args.max_fpr)
    print(json.dumps(result,ensure_ascii=False,indent=2))
