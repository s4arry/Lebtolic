"""Render committed aggregate/pair-score results; never refit or select a model.

Install requirements-plots.txt, then python -m round3.render_results --results DIR.
"""
import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_curve


def render(root):
    results = json.loads((root/'metrics.json').read_text(encoding='utf-8'))
    with (root/'test_pair_scores.csv').open(encoding='utf-8') as stream:
        rows = list(csv.DictReader(stream))
    y = np.array([int(r['target']) for r in rows])
    old = np.array([float(r['legacy_score']) for r in rows])
    new = np.array([float(r['new_score']) for r in rows])
    blue, orange = '#275DAD', '#C45A26'
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
                         'axes.spines.top': False, 'axes.spines.right': False,
                         'svg.fonttype': 'none'})
    fig, axes = plt.subplots(2, 2, figsize=(12, 8.5), layout='constrained')
    ax = axes[0, 0]
    for label, score, key, color in [('Legacy', old, 'test_legacy_at_0_5', orange),
                                     ('New', new, 'test_new', blue)]:
        fpr, tpr, _ = roc_curve(y, score)
        ax.plot(fpr, tpr, color=color, label=f"{label}: AUC {results[key]['roc_auc']:.3f}")
    ax.plot([0, 1], [0, 1], color='#999999', linestyle='--', linewidth=1)
    ax.set(xlabel='False positive rate', ylabel='True positive rate', title='A. Ranking on the historical Test')
    ax.legend(loc='lower right', frameon=False)

    ax = axes[0, 1]
    comparison = results['paired_test_comparison']
    point = comparison['point_differences']['brier_old_minus_new']
    lo, hi = comparison['brier_old_minus_new']['ci95']
    verdict = 'met' if comparison['primary_supported'] else 'not met'
    ax.errorbar(point, 0, xerr=[[point-lo], [hi-point]], fmt='o', color=blue, capsize=7)
    ax.axvline(0, color='#666666', linestyle='--')
    ax.set(yticks=[], ylim=(-.6, .6), xlabel='Brier(legacy) - Brier(new)',
           title=f'B. Primary improvement criterion {verdict}')
    ax.text(.03, .08, f'95% paired CI: [{lo:.6f}, {hi:.6f}]\nPositive means lower Brier for new',
            transform=ax.transAxes, fontsize=9)
    ax.ticklabel_format(axis='x', style='sci', scilimits=(-3, 3))

    ax = axes[1, 0]
    bands = results['test_levels']
    if bands is not None:
        rates = np.array([b['observed_rate'] for b in bands])*100
        lows = np.array([b['observed_rate_95ci_wilson'][0] for b in bands])*100
        highs = np.array([b['observed_rate_95ci_wilson'][1] for b in bands])*100
        ax.errorbar([1, 2, 3], rates, yerr=[rates-lows, highs-rates], fmt='o', color=blue, capsize=6,
                    label='Observed event rate; Wilson 95% CI')
        ax.plot([1, 2, 3], [b['mean_score']*100 for b in bands], 's--', color=orange, label='Mean score')
        ax.set_xticks([1, 2, 3], [f"Level {b['level']}\n{b['events']}/{b['n']} events" for b in bands])
        ax.set_ylim(bottom=0)
        ax.legend(frameon=False, fontsize=8)
    ax.set(ylabel='Percent', title='C. Relative bands; not clinical risk categories')

    ax = axes[1, 1]
    old_op, new_op = results['test_legacy_with_same_threshold_policy'], results['test_new']
    for offset, item, label, color in [(-.17, old_op, 'Legacy OOF-F2', orange), (.17, new_op, 'New OOF-F2', blue)]:
        values = [100*item[k] if item[k] is not None else np.nan for k in ['recall', 'precision']]
        ax.bar(np.array([0, 1])+offset, values, width=.32,
               label=f"{label}; FP={item['confusion_matrix']['fp']}", color=color)
    ax.set(xticks=[0, 1], xticklabels=['Recall', 'Precision'], ylabel='Percent', ylim=(0, 110),
           title='D. Higher recall comes with more false alerts')
    ax.legend(frameon=False, fontsize=9)
    fig.suptitle(f"NHEFS internal Test: n={len(y)}, events={int(y.sum())}\n"
                 'Fixed protocol; previously exposed Test; no Thai/external validation', fontsize=13)
    fig.savefig(root/'results_summary.png', dpi=180)
    fig.savefig(root/'results_summary.svg')
    svg = root/'results_summary.svg'
    svg.write_text('\n'.join(line.rstrip() for line in svg.read_text().splitlines())+'\n', encoding='utf-8')
    plt.close(fig)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', type=Path, required=True)
    render(parser.parse_args().results)
