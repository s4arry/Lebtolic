"""Verify saved numeric outputs and render the completed balancing study."""
import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

from .balancing_study import MAIN, FALLBACK, digest, policies, write_csv
from .threshold_policy import read_csv
from .workflow import save_json


LABELS = {'none': '4 features', 'balanced': 'Class weights', 'oversample': 'Oversampling',
          'weight': '+ weight', 'smoking': '+ smoking', 'weight_smoking': '+ weight + smoking'}


def verify(root):
    if json.loads((root/'study_status.json').read_text())['status'] != 'completed':
        raise ValueError('Only render a completed real-data run')
    saved = json.loads((root/'verification.json').read_text())
    if digest(Path(__file__).with_name('balancing_study.py')) != saved['code_sha256']:
        raise ValueError('Study code changed after the run')
    if digest(root/'decision_before_test.json') != saved['decision_sha256']:
        raise ValueError('Frozen decision changed')
    oof = read_csv(root/'development_oof.csv')
    variants = list(dict.fromkeys(r['variant'] for r in oof))
    stored = {(r['variant'], int(r['repeat']), r['policy']): r for r in read_csv(root/'repeat_metrics.csv')}
    decision = json.loads((root/'decision_before_test.json').read_text())
    fold_rows = {(r['variant'], int(r['repeat']), int(r['fold']), r['policy']): r for r in read_csv(root/'outer_folds.csv')}
    membership = {(int(r['repeat']), int(r['development_row'])): int(r['fold']) for r in read_csv(root/'outer_membership.csv')}
    def check(target, score, predicted, row):
        tp = int(np.sum((target == 1) & predicted)); fp = int(np.sum((target == 0) & predicted))
        fn = int(np.sum((target == 1) & ~predicted)); tn = int(np.sum((target == 0) & ~predicted))
        expected = dict(tp=tp, fp=fp, fn=fn, tn=tn, recall=tp/(tp+fn), fpr=fp/(fp+tn),
            precision=tp/(tp+fp) if tp+fp else None, f2=5*tp/(5*tp+4*fn+fp),
            brier=float(np.mean((target-score)**2)), roc_auc=float(roc_auc_score(target, score)))
        for key, value in expected.items():
            if value is None:
                assert row[key] == ''
            else:
                np.testing.assert_allclose(float(row[key]), value, rtol=0, atol=1e-12)
    for name in variants:
        repeated = []
        for repeat in range(3):
            rows = sorted([r for r in oof if r['variant'] == name and int(r['repeat']) == repeat],
                          key=lambda r: int(r['development_row']))
            assert [int(r['development_row']) for r in rows] == list(range(2311))
            y = np.array([int(r['target']) for r in rows]); p = np.array([float(r['score']) for r in rows])
            repeated.append(p)
            for policy in ['f2', 'fpr145']:
                pred = np.array([int(r[policy+'_prediction']) for r in rows], bool)
                check(y, p, pred, stored[(name, repeat, policy)])
                for fold in range(5):
                    mask = np.array([membership[(repeat, i)] == fold for i in range(len(y))])
                    row = fold_rows[(name, repeat, fold, policy)]
                    np.testing.assert_array_equal(pred[mask], p[mask] >= float(row['threshold']))
                    check(y[mask], p[mask], pred[mask], row)
        recomputed = policies(y, np.mean(repeated, axis=0))
        assert recomputed == decision['final_thresholds'][name]
    rows = read_csv(root/'test_scores.csv'); y = np.array([int(r['target']) for r in rows])
    for row in read_csv(root/'test_metrics.csv'):
        p = np.array([float(r[row['variant']]) for r in rows]); t = float(row['threshold'])
        check(y, p, p >= t, row)
        assert t == decision['final_thresholds'][row['variant']][row['policy']]['threshold']
    for r in read_csv(root/'fit_counts.csv'):
        assert int(r['original_negative'])+int(r['original_positive']) == int(r['train_n'])
        if r['variant'] == 'oversample':
            assert r['fitted_positive'] == r['fitted_negative']
        else:
            assert r['fitted_positive'] == r['original_positive']
            assert r['fitted_negative'] == r['original_negative']
    checks = dict(status='passed', checked_variants=variants, development_rows_per_repeat=2311,
        repeats=3, outer_fold_metric_rows=len(fold_rows), test_metric_rows=12,
        checks=['All fold/repeat/test confusion matrices and ROC-AUC/Brier/F2 recalculated from saved predictions',
                'Every development person has exactly one prediction per repeat per method',
                'Outer decisions equal scores compared with stored training-selected thresholds',
                'Final thresholds reproduce exactly from development OOF; no Test input',
                'All resampling count records match training-only policy',
                'Study code and frozen decision hashes match'])
    save_json(root/'independent_verification.json', checks)


def render(root):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    rows = read_csv(root/'repeat_metrics.csv')
    variants = [k for k in MAIN+FALLBACK if any(r['variant'] == k for r in rows)]
    def mean_sd(name, policy, metric):
        a = [float(r[metric]) for r in rows if r['variant'] == name and r['policy'] == policy]
        return np.mean(a), np.std(a, ddof=1)
    x = np.arange(len(variants)); labels = [LABELS[k] for k in variants]
    colors = ['#23395B', '#BB4430', '#D68C45', '#379392', '#7768AE', '#26834A'][:len(variants)]
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False,
                         'svg.hashsalt': 'lebtolic-balancing-20261111'})
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), layout='constrained')
    for ax, metric, title in [(axes[0, 0], 'roc_auc', 'Development ROC-AUC'),
                              (axes[1, 0], 'mean_probability', 'Mean predicted score vs observed event rate')]:
        stats = np.array([mean_sd(k, 'f2', metric) for k in variants])
        ax.bar(x, stats[:, 0], yerr=stats[:, 1], capsize=4, color=colors)
        ax.set_xticks(x, labels, rotation=20, ha='right'); ax.set_title(title)
        if metric == 'roc_auc': ax.set_ylim(.5, .8)
        else:
            ax.axhline(141/2311, color='#111111', linestyle='--', label='Observed 6.10%')
            ax.legend(); ax.set_ylim(0, .52)
    ax = axes[0, 1]
    recall = np.array([mean_sd(k, 'fpr145', 'recall') for k in variants])
    fpr = np.array([mean_sd(k, 'fpr145', 'fpr') for k in variants])
    ax.bar(x-.18, recall[:, 0], .36, yerr=recall[:, 1], capsize=3, color='#26834A', label='Recall')
    ax.bar(x+.18, fpr[:, 0], .36, yerr=fpr[:, 1], capsize=3, color='#BB4430', label='Observed FPR')
    ax.axhline(.145, color='black', linestyle='--', linewidth=1, label='Inner selection budget 14.5%')
    ax.set_xticks(x, labels, rotation=20, ha='right'); ax.set_ylim(0, .55)
    ax.set_title('Development: nested FPR-budget policy'); ax.legend(fontsize=8)
    ax = axes[1, 1]
    bins = read_csv(root/'calibration_bins.csv')
    for name in ['none', 'balanced', 'oversample', 'weight_smoking']:
        rs = [r for r in bins if r['variant'] == name and r['scope'] == 'historical_test' and int(r['n'])]
        a = np.array([[float(r[k]) for k in ['mean_probability', 'observed_rate', 'wilson_low', 'wilson_high']] for r in rs])
        ax.errorbar(a[:, 0], a[:, 1], yerr=np.maximum(0., [a[:, 1]-a[:, 2], a[:, 3]-a[:, 1]]), marker='o', capsize=3,
                    color=colors[variants.index(name)], label=LABELS[name])
    ax.plot([0, 1], [0, 1], '--', color='#777777', linewidth=1)
    ax.set(xlabel='Mean predicted score', ylabel='Observed event rate', xlim=(0, .8), ylim=(0, .8),
           title='Historical Test calibration (25 events; exploratory)')
    ax.legend(fontsize=8)
    fig.suptitle('LEPTOLIC: balancing and baseline-feature fallback\nDevelopment bars: 3-repeat mean ± SD; calibration bars: Wilson 95% intervals', fontsize=14)
    fig.savefig(root/'comparison.png', dpi=160)
    fig.savefig(root/'comparison.svg', metadata={'Date': None})
    svg = root/'comparison.svg'
    svg.write_text('\n'.join(line.rstrip() for line in svg.read_text().splitlines())+'\n')
    plt.close(fig)


def report(root):
    repeats = read_csv(root/'repeat_metrics.csv'); test = read_csv(root/'test_metrics.csv')
    variants = list(dict.fromkeys(r['variant'] for r in repeats))
    decision = json.loads((root/'decision_before_test.json').read_text())
    def stat(name, policy, metric, digits=4, pct=False):
        a = np.array([float(r[metric]) for r in repeats if r['variant'] == name and r['policy'] == policy])
        if pct: a *= 100
        return f'{a.mean():.{digits}f} ± {a.std(ddof=1):.{digits}f}'
    lines = ['# LEPTOLIC: balancing และแผนสำรอง — ผลรันจริง 11 พฤศจิกายน 2026', '',
        '**Balancing ไม่ผ่านเกณฑ์ development; แผนสำรองทำงานแล้วและเลือก +weight+smoking เป็น candidate ทดลองจาก development เท่านั้น**', '',
        'โมเดลหลักและนโยบายงบ FPR 14.5% เดิมยังคงอยู่ตามเดิม การเลือก candidate นี้ไม่ใช่หลักฐานว่าดีกว่าทุกด้าน '
        'และไม่ได้เลือกใหม่จาก Test ผล Test ด้าน Brier แย่ลงเล็กน้อยและช่วงผลต่างยังคร่อมศูนย์', '',
        'ดาวน์โหลดต้นฉบับ CDC ได้ในเครื่องใหม่ SHA-256 ตรงกับ 8 ไฟล์ใน ZIP รอบก่อนทั้งหมด '
        'จึงไม่ต้องใช้ ZIP แนบซ้ำสำหรับรอบนี้ ตรวจ cohort 2,720/166 events, development 2,311/141, '
        'Test 409/25; ทำนายคะแนน baseline ซ้ำตรงรอบก่อนทุกค่าถึงความละเอียดที่บันทึกไว้', '',
        '## วิธีประเมิน', '',
        '[Protocol ที่เขียนก่อนการเปรียบเทียบ](../../BALANCING_PROTOCOL_TH.md) · '
        '[Codebook/skip codes](../../CODEBOOK_BALANCING_TH.md) · [โค้ด](../../balancing_study.py)', '',
        'ใช้ Logistic Regression C=.1 ตามโมเดลเดิม ไม่มีการค้นหา C เพิ่ม เพื่อแยกผลวิธี balancing '
        'Outer 5-fold × 3 repeats ใช้สมาชิกเดียวกันทุก variant; inner 3-fold เลือก threshold ใหม่ทุก outer fold '
        '(F2 หรือ recall ภายใต้ empirical FPR≤14.5%) ทุก preprocessing และ resampling fit เฉพาะ training '
        'ดังนั้นเป็น repeated nested **threshold** validation ไม่ใช่ nested hyperparameter search', '',
        'ตาราง development ด้านล่างเป็น mean ± SD ของ **3 pooled repeats**: หนึ่งคนปรากฏหนึ่งครั้งต่อ repeat '
        'ไม่รวมซ้ำเป็น 6,933 คนอิสระ [15-fold mean/SD/min/max](outer_summary.csv) และ '
        '[ผลทุก fold](outer_folds.csv) แสดงความแปรปรวนเพิ่ม รวม calibration slope/intercept '
        'SD เหล่านี้ไม่ใช่ confidence interval ของประชากร; folds/repeats ใช้คนร่วมกัน', '',
        '## 1. Probability/ranking บน development', '',
        '| วิธี | ROC-AUC | Brier (ต่ำดีกว่า) | คะแนนเฉลี่ย | Calibration intercept-in-large | Calibration slope |',
        '|---|---:|---:|---:|---:|---:|']
    for name in variants:
        lines.append('| '+LABELS[name]+' | '+' | '.join(stat(name, 'f2', m) for m in
                     ['roc_auc', 'brier', 'mean_probability', 'calibration_in_large', 'calibration_slope'])+' |')
    lines += ['', 'อัตราเหตุการณ์ development จริง 6.10%; balanced/oversample ให้คะแนนเฉลี่ยประมาณ 46% '
        'จึงเสีย calibration อย่างมาก คะแนนดิบหลัง balancing ไม่ใช่ความเสี่ยงที่ปรับกลับตาม prevalence แล้ว '
        'ผลนี้ไม่ใช่ข้อพิสูจน์ว่า balancing ทุกแบบร่วมกับ calibration จะไม่มีประโยชน์ '
        'รอบนี้ไม่ได้ทดลอง prior correction หรือ fitted recalibration เพิ่ม', '',
        '## 2. Operating metrics บน outer validation', '',
        '| วิธี | Policy | TP | FP | FN | TN | Recall % | Precision % | F2 | FPR % |',
        '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for name in variants:
        for policy in ['f2', 'fpr145']:
            values = [stat(name, policy, m, 1) for m in ['tp', 'fp', 'fn', 'tn']]
            values += [stat(name, policy, m, 2, True) for m in ['recall', 'precision']]
            values += [stat(name, policy, 'f2', 4), stat(name, policy, 'fpr', 2, True)]
            lines.append('| '+LABELS[name]+' | '+policy+' | '+' | '.join(values)+' |')
    lines += ['', '14.5% เป็นข้อจำกัดตอนเลือก threshold จาก inner OOF ไม่รับประกัน outer/Test FPR '
        'ส่วน F2-policy ไม่มีเพดาน FP และไม่ได้ออกแบบให้ recall ต้องสูงที่สุดอย่างเดียว', '',
        '## 3. แผนสำรองที่ดำเนินการจริง', '',
        '- ตรวจ codebook ก่อนเพิ่มน้ำหนักวัด baseline (pounds, ไม่ใช่ BMI) และประวัติ smoking; '
        'ใช้ +weight, +smoking, +ทั้งคู่ บน cohort/folds เดิม ไม่ตัดคนที่น้ำหนักหายออก',
        '- Weight และ weight+smoking ผ่านเกณฑ์ development ที่กำหนด; smoking เพียงอย่างเดียวไม่ผ่าน '
        'เลือก weight+smoking เพราะ mean development Brier ต่ำสุดในกลุ่มที่ผ่าน '
        'ผลต่าง Brier กับ weight-only เล็กมาก จึงไม่อ้างว่า smoking เพิ่มประโยชน์ที่ยืนยันแล้ว',
        '- รุ่นรวมมี 6 ตัวแปรตั้งต้น และ encoded columns 9 ช่อง รวม weight-missing และ smoking category flags '
        'unknown smoking เป็นศูนย์ใน cohort นี้ จึงไม่มีข้อมูลพอเรียนผลของหมวด unknown สำหรับคนนอก cohort',
        '- การคัดเลือกจากผล development หลาย variant ยังมี selection optimism; ตัวเลขนี้ประเมินแต่ละ '
        'fixed workflow ไม่ใช่ unbiased evaluation ของขั้นตอนเลือกผู้ชนะจากทั้งหมด '
        'ต้องประเมิน candidate ที่ freeze แล้วบนข้อมูลใหม่ก่อนอ้างความเหนือกว่า', '',
        '[กติกาและผลตัดสินที่ freeze ก่อน Test](decision_before_test.json) · '
        '[พารามิเตอร์ที่ fit จริงทุก variant](candidate_models.json)', '',
        '## 4. Test เดิม — exploratory เท่านั้น', '',
        '| วิธี | Policy | Threshold % | TP | FP | FN | TN | Recall % | Precision % | F2 | ROC-AUC | Brier |',
        '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for row in test:
        vals = [LABELS[row['variant']], row['policy'], f"{float(row['threshold'])*100:.3f}"]
        vals += [row[k] for k in ['tp', 'fp', 'fn', 'tn']]
        vals += [f'{float(row[k])*100:.2f}' for k in ['recall', 'precision']]
        vals += [f'{float(row[k]):.5f}' for k in ['f2', 'roc_auc', 'brier']]
        lines.append('| '+' | '.join(vals)+' |')
    ci = read_csv(root/'test_paired_difference_ci.csv')
    lines += ['', 'ผลต่าง candidate weight+smoking ลบ baseline 4 features ของรอบนี้ ณ FPR-policy '
        '(paired bootstrap 2,000 draws; fixed fitted models):', '', '| Metric | 95% percentile interval |', '|---|---:|']
    for metric in ['recall', 'precision', 'f2', 'roc_auc', 'brier']:
        row = next(r for r in ci if r['variant'] == 'weight_smoking' and r['policy'] == 'fpr145' and r['metric'] == metric)
        lines.append(f"| Δ{metric} | [{float(row['lower']):.5f}, {float(row['upper']):.5f}] |")
    lines += ['', 'ΔBrier >0 แปลว่า candidate แย่กว่า; Δrecall/F2/AUC >0 แปลว่าสูงกว่า '
        'ช่วง recall แตะศูนย์ และ F2/Brier/AUC คร่อมศูนย์ จึงยังยืนยันความเหนือกว่าไม่ได้ '
        '[CI ของทุก metric รวม TP/FP/FN/TN](test_bootstrap_ci.csv) · '
        '[paired differences ทุก variant](test_paired_difference_ci.csv)', '',
        '**เหตุที่ baseline F2 ในตารางไม่ใช่ TP23/FP250 เดิม:** คะแนนโมเดลเดิมตรงกัน '
        'แต่ threshold ถูกเลือกใหม่บน OOF ของ fixed-C four-feature workflow รอบนี้ (5.954%) '
        'ต่างจาก threshold เดิม 4.17% ที่มาจาก workflow เลือกหลาย candidate '
        'ส่วน threshold FPR รอบนี้ 9.286% ให้ TP8/FP42; นโยบายที่คงใช้อยู่เดิม 9.187% ให้ TP8/FP45 '
        'ไม่มีการเขียนทับนโยบายเดิม และไม่ควรอ้างการลด FP จาก threshold ใหม่ว่า AUC ดีขึ้น', '',
        'แม้ smoking-only ดูดีกว่า candidate ในบาง Test metric ก็ไม่เปลี่ยนผู้ชนะตาม Test '
        'โมเดลหลักเดิมยังอยู่; candidate ที่เลือกจาก development ถูกส่งออกเป็นงานทดลองแยกตาม protocol', '',
        '## 5. Missed events และ missingness', '',
        '- Baseline FPR-policy พลาด 92 จาก 141 development events ครบทั้ง 3 repeats; '
        'กลุ่มนี้มี SBP เฉลี่ย 113.34 เทียบ 122.23 ใน 35 events ที่จับได้ทุกรอบ '
        'เป็นคำอธิบาย score separation ใน cohort ไม่ใช่ความสัมพันธ์เชิงสาเหตุ',
        '- F2-policy พลาด 18 events ทุกรอบ และจับได้ 80 events ทุกรอบ; '
        'ผลที่เหลือแกว่งตาม fold/threshold แสดงว่าไม่ควรสรุปจาก confusion matrix เดียว',
        '- Development น้ำหนักหาย 16/2,311 (0.69%); sleep หาย 84/2,311 (3.63%); '
        'core 4 features ไม่มี missing เพราะ cohort rule คัดออกแล้ว '
        'การ impute ไม่สามารถแก้ selection bias จากผู้ถูกคัดออกได้',
        '- ผ่าน baseline แต่ถูกคัดออกด้วย follow-up/ประวัติขัดแย้ง 654 คน มีอายุเฉลี่ย 59.53 ปี '
        'เทียบ 47.24 ใน development; 462 คนไม่มี completed surviving final interview '
        'ไม่ได้กำหนดคนเหล่านี้เป็น negative หรืออ้างอัตราเหตุการณ์ของกลุ่มที่ไม่รู้ผล', '',
        '[Missed-event groups](missed_event_audit.csv) · [Missingness](missingness_audit.csv) · '
        '[เหตุผลคัดออก/รวมเข้า](exclusion_counts.json)', '',
        '## กราฟ', '', '![Development comparison and historical Test calibration](comparison.png)', '',
        '[SVG](comparison.svg) · [Calibration bins และ Wilson CI](calibration_bins.csv)', '',
        '## รันซ้ำและตรวจหลักฐาน', '',
        '```bash', 'python -m pip install -r round3/requirements-balancing.txt',
        'python -m unittest discover -s tests -v',
        'OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m round3.balancing_study \\',
        '  --source /path/to/nhefs_source --output round3_work/balancing_new_run',
        '# source/raw/ มี 8 ไฟล์จาก ZIP เดิม หรือให้ดาวน์โหลดจาก CDC',
        '# การ render กราฟต้องมี matplotlib เพิ่ม',
        'python -m round3.report_balancing --results round3_work/balancing_new_run', '```', '',
        'ไฟล์ source ต้องตรง hashes ก่อนจึงรันต่อ รุ่น dependencies จริงแยกใน requirements-balancing '
        'เพราะต่างจาก environment รอบก่อน แต่ refit baseline Test score ตรงเดิมทุกค่า '
        'ค่าที่บันทึกและข้อจำกัด numerical environment อยู่ใน [verification.json](verification.json)', '',
        '[Software tests](software_tests.txt) · [Independent checks](independent_verification.json) · '
        '[Raw provenance/split hashes](provenance.json) · [Codebook hashes](codebook_provenance.json) · '
        '[Anonymous OOF predictions](development_oof.csv) · [Test predictions](test_scores.csv)', '',
        'candidate_models.json เก็บ coefficients, imputer/scaler, class counts และ threshold สองนโยบาย '
        'นำ replay ผ่าน PortableModel ด้วย feature order ที่ระบุและ encoding ใน encode_extra ได้ '
        '**ไม่ใช้ CLI round3.predict เดิมกับ candidate 9 columns** เพราะ legacy input form ยังรองรับ '
        'schema เดิม; รอบนี้ส่งออกเพื่อวิจัยและไม่ได้เปลี่ยน inference interface ของโมเดลหลัก', '',
        '## ข้อจำกัด', '',
        'ข้อมูล NHEFS สหรัฐฯ ในอดีต, reported/self-or-proxy diagnosis, selected cohort ที่ SBP<130/DBP<80, '
        'เหตุการณ์น้อย, year-only timing, incomplete follow-up, ไม่มี IPCW/competing-risk correction '
        'และไม่ถ่วง survey weights Test ถูกเปิดดูหลายรอบ; bootstrap ไม่รวมความแปรปรวนจากการฝึก/เลือกโมเดล '
        'ไม่มี external หรือ Thai validation และไม่มีหลักฐานว่าใช้วินิจฉัยหรือประเมินคนไทยได้ '
        '“ไม่มีรายงานวินิจฉัยใน 8 ปี” ไม่เท่ากับ “ไม่เคยเป็นความดันสูงตลอดชีวิต”', '']
    (root/'README_TH.md').write_text('\n'.join(lines), encoding='utf-8')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', type=Path, required=True)
    args = parser.parse_args()
    verify(args.results); render(args.results); report(args.results)
