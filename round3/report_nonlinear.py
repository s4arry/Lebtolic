"""Independently check and report the nonlinear Logistic experiment."""
import argparse,json
from pathlib import Path
from collections import Counter
import numpy as np
from sklearn.metrics import roc_auc_score
from .balancing_study import digest,policies
from .threshold_policy import read_csv
from .workflow import save_json
from .nonlinear_study import VARIANTS,CONFIGS,choose

LABELS={'none':'เดิม C=.1','linear':'Linear เลือก C','quadratic':'Quadratic เลือก C','selected':'เลือกทั้งรูปแบบ/C'}

def run(root):
    assert json.loads((root/'study_status.json').read_text())['status']=='completed'
    verification=json.loads((root/'verification.json').read_text());decision=json.loads((root/'decision_before_test.json').read_text())
    assert digest(Path(__file__).with_name('nonlinear_study.py'))==verification['code_sha256']
    assert digest(root/'decision_before_test.json')==verification['decision_sha256']
    repeatrows=read_csv(root/'repeat_metrics.csv');foldrows=read_csv(root/'outer_folds.csv')
    oof=read_csv(root/'development_oof.csv');test=read_csv(root/'test_metrics.csv')
    member={(int(r['repeat']),int(r['development_row'])):int(r['fold']) for r in read_csv(root/'outer_membership.csv')}
    def check(y,p,pred,row):
        tp=int(np.sum((y==1)&pred));fp=int(np.sum((y==0)&pred));fn=int(np.sum((y==1)&~pred));tn=int(np.sum((y==0)&~pred))
        expected=dict(tp=tp,fp=fp,fn=fn,tn=tn,recall=tp/(tp+fn),precision=tp/(tp+fp) if tp+fp else None,
            f2=5*tp/(5*tp+4*fn+fp),fpr=fp/(fp+tn),brier=float(np.mean((y-p)**2)),roc_auc=float(roc_auc_score(y,p)))
        for k,v in expected.items():
            if v is not None:np.testing.assert_allclose(float(row[k]),v,rtol=0,atol=1e-12)
    cached={}
    for v in VARIANTS:
        for r in range(3):
            rs=sorted([s for s in oof if s['variant']==v and int(s['repeat'])==r],key=lambda s:int(s['development_row']))
            assert [int(s['development_row']) for s in rs]==list(range(2311))
            cached[(v,r)]=(np.array([int(s['target']) for s in rs]),np.array([float(s['score']) for s in rs]),rs)
    for row in repeatrows+foldrows:
        v,r=row['variant'],int(row['repeat']);y,p,rs=cached[(v,r)]
        pred=np.array([int(s[row['policy']+'_prediction']) for s in rs],bool)
        mask=np.array([member[(r,j)]==int(row['fold']) for j in range(len(y))]) if 'fold' in row else np.ones(len(y),bool)
        if 'fold' in row:np.testing.assert_array_equal(pred[mask],p[mask]>=float(row['threshold']))
        check(y[mask],p[mask],pred[mask],row)
    inner=read_csv(root/'final_inner_oof.csv');y=np.array([int(r['target']) for r in inner])
    for v in VARIANTS:
        config=decision['final']['groups'][v]['chosen']
        assert policies(y,np.array([float(r[config]) for r in inner]))==decision['thresholds'][v]
    records={k:c['fold_briers'] for k,c in decision['final']['candidates'].items()}
    assert choose(records,CONFIGS)==decision['final']['groups']['selected']
    scores=read_csv(root/'test_scores.csv');yt=np.array([int(r['target']) for r in scores])
    for row in test:
        p=np.array([float(r[row['variant']]) for r in scores]);check(yt,p,p>=float(row['threshold']),row)
    old=[r for r in read_csv(root.parent/'calibration_20261111/development_oof.csv') if r['variant']=='none']
    new=[r for r in oof if r['variant']=='none'];assert len(old)==len(new)
    for a,b in zip(old,new):
        for k in ['repeat','development_row','target','f2_prediction','fpr145_prediction']:assert a[k]==b[k]
        np.testing.assert_allclose(float(a['score']),float(b['score']),atol=1e-12,rtol=0)
    save_json(root/'independent_verification.json',dict(status='passed',checks=[
        'All outer/repeat/Test confusion counts, AUC, Brier and F2 reconstructed from predictions',
        'OOF coverage and outer thresholds checked; final thresholds reconstructed from inner OOF',
        'Final shape/C selection reconstructed from inner fold Brier without Test',
        'Comparator reproduces prior OOF scores within 1e-12 and decisions exactly',
        'Code and frozen-decision hashes verified']))
    def vals(v,k,policy='fpr145',source=repeatrows):return np.array([float(r[k]) for r in source if r['variant']==v and r['policy']==policy])
    def stat(v,k,policy='fpr145',pct=False):
        a=vals(v,k,policy)*(100 if pct else 1)
        return f'{a.mean():.5f} ± {a.std(ddof=1):.5f}'
    choices=Counter(c['groups']['selected']['chosen'] for c in json.loads((root/'outer_choices.json').read_text()))
    lines=['# Nonlinear Logistic ต่อรุ่นน้ำหนัก + สูบบุหรี่ — ผลจริง 11 พฤศจิกายน 2026','',
        '**การเพิ่มพจน์กำลังสองยังไม่ให้หลักฐานเพียงพอว่าดีกว่ารุ่นเดิมบน development จึงไม่แทนรุ่นเดิม**','',
        'ทดลอง Logistic Regression 6 candidates: linear/quadratic × C=.01/.1/1 '
        'ใช้ข้อมูล อายุ เพศ SBP DBP น้ำหนัก และประวัติสูบบุหรี่เดิม '
        'เพิ่มเฉพาะ squared standardized terms ของอายุ/SBP/DBP/น้ำหนัก รวม encoded terms จาก 9 เป็น 13 '
        'ไม่มี interaction, balancing หรือ calibration เพิ่มในรอบนี้','',
        '[Protocol ก่อนรัน](../../NONLINEAR_PROTOCOL_TH.md) · [โค้ด](../../nonlinear_study.py) · [Decision ก่อน Test](decision_before_test.json)','',
        '## วิธีเลือกและขอบเขตผล','',
        'Cohort 2720/166 events, development 2311/141, Test 409/25 และ split IDs/hashes เดิม '
        'Outer 5-fold × 3 repeats; inner 3-fold เลือก shape/C ด้วย Brier และ one-SE heuristic '
        'เลือก linear ก่อน quadratic และ C ต่ำก่อน C สูงเมื่ออยู่ในช่วง best+SE '
        'ทุก median/scaler/expanded scaler fit จาก training เท่านั้น '
        'เลือก thresholds F2/FPR14.5% จาก inner OOF ภายใน outer training '
        'Final method/thresholds freeze จาก development ก่อน Test','',
        f"Final selected candidate: **{decision['final']['groups']['selected']['chosen']}**; outer choices: **{dict(choices)}**. "
        'นี่คือผลของขั้นตอนคัดเลือกที่ทดลอง ไม่ได้แปลว่าผ่านเกณฑ์ปรับปรุงหรือถูกนำมาแทนรุ่นเดิม '
        'one-SE เป็นกติกาให้ความสำคัญกับความเรียบง่าย ไม่ใช่ statistical CI','',
        '## Development probability metrics','',
        'Mean ± SD ของ 3 pooled repeats; แต่ละคนปรากฏหนึ่งครั้งต่อ repeat ไม่ถือ repeats เป็นคนเพิ่ม','',
        '| Workflow | ROC-AUC ↑ | AP ↑ | Brier ↓ | Calibration slope (ideal 1) |','|---|---:|---:|---:|---:|']
    for v in VARIANTS:lines.append('| '+LABELS[v]+' | '+' | '.join(stat(v,k) for k in ['roc_auc','ap','brier','calibration_slope'])+' |')
    lines+=['','## Development operating metrics','','| Workflow | Policy | Recall % | Precision % | F2 | FPR % |','|---|---|---:|---:|---:|---:|']
    for v in VARIANTS:
        for policy in ['f2','fpr145']:
            lines.append('| '+LABELS[v]+' | '+policy+' | '+' | '.join([stat(v,'recall',policy,True),stat(v,'precision',policy,True),stat(v,'f2',policy),stat(v,'fpr',policy,True)])+' |')
    lines+=['','[TP/FP/FN/TN ทุก repeat](repeat_metrics.csv) · [ทุก outer fold](outer_folds.csv) · [Mean/SD/range](outer_summary.csv)','',
        'Recall ของ workflows ที่เลือก C เพิ่มเล็กน้อยพร้อม FPR ประมาณ 16% ซึ่งเกินงบ 14.5% ใน outer validation '
        'งบ 14.5% ใช้บังคับเฉพาะตอนเลือก threshold จาก inner OOF จึงไม่รับประกันค่าที่พบในชุดใหม่ '
        'ผลนี้ชี้ว่าต้องดูจำนวนคำเตือนผิดควบคู่กับ recall','',
        '## ช่องว่างชุดฝึกกับ outer validation','','ค่าเฉลี่ย 15 folds; ต่างจาก pooled-repeat AUC ด้านบนเพราะวิธีรวมคะแนนต่างกัน','',
        '| Workflow | Train AUC | Validation AUC | Train Brier | Validation Brier |','|---|---:|---:|---:|---:|']
    for v in VARIANTS:
        lines.append('| '+LABELS[v]+' | '+' | '.join(f'{vals(v,k,source=foldrows).mean():.6f}' for k in ['train_auc','roc_auc','train_brier','brier'])+' |')
    lines+=['','ค่าชุดฝึกดีกว่า validation ทุก workflow แต่ช่องว่างอย่างเดียวไม่พิสูจน์ overfitting '
        'การเพิ่ม squares ในการทดลองนี้ไม่ได้เพิ่มความสามารถแยกกลุ่มใน outer validation '
        'regularization ที่แรงขึ้นอาจลดทั้งความแกว่งและสัญญาณที่มีประโยชน์ได้','',
        '## Final inner selection','','| Candidate | Inner mean Brier | SD |','|---|---:|---:|---:|']
    for k,c in decision['final']['candidates'].items():lines.append(f"| {k} | {c['mean_brier']:.8f} | {c['sd']:.8f} |")
    lines+=['', 'Quadratic C=.1 มี inner point estimate ต่ำที่สุด แต่ linear C=.01 อยู่ในช่วง one-SE '
        'จึงถูกเลือกตามกติกาที่ตั้งไว้ก่อนรัน การเลือกนี้ไม่ได้ทำให้ outer results ดีกว่า comparator '
        'และไม่มีการเปลี่ยนกติกาหลังเห็นผล','',
        '## Test เดิม — exploratory เท่านั้น','','| Workflow | Policy | Threshold % | TP | FP | FN | TN | Recall % | Precision % | F2 | AUC | Brier |',
        '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for r in test:
        cols=[LABELS[r['variant']],r['policy'],f"{float(r['threshold'])*100:.4f}"]+[r[k] for k in ['tp','fp','fn','tn']]
        cols += [f"{float(r[k])*100:.2f}" for k in ['recall','precision']]+[f"{float(r[k]):.6f}" for k in ['f2','roc_auc','brier']]
        lines.append('| '+' | '.join(cols)+' |')
    lines+=['','แม้ quadratic จะมี Test AUC/Brier ดีกว่า comparator ในรอบนี้ ก็ไม่เลือกย้อนหลังจาก Test '
        'ผล development ไม่สนับสนุนความเหนือกว่าอย่างสม่ำเสมอ Test มีเพียง 25 events '
        'และเคยเปิดดูหลายครั้ง จึงไม่ใช่ผลยืนยันบนชุดใหม่','',
        'ตัวเทียบ none คือรุ่น weight+smoking C=.1 จาก calibration follow-up '
        'ใช้ final threshold จาก inner OOF แบบเดียวกัน: Test TP10/FP49 ณ FPR-policy '
        'ไม่ใช่ policy รุ่นก่อนที่ใช้ mean repeated outer OOF และได้ TP13/FP52 '
        'ทั้งคะแนนและการตัดสิน baseline ในรอบนี้ตรวจตรงกับ calibration follow-up แล้ว','',
        'Paired bootstrap 2000 draws, candidate − comparator (FPR-policy):','',
        '| Workflow | Metric | 95% percentile interval |','|---|---|---:|']
    for r in read_csv(root/'test_paired_difference_ci.csv'):
        if r['variant'] in ['quadratic','selected'] and r['policy']=='fpr145' and r['metric'] in ['roc_auc','brier','recall','precision','f2']:
            lines.append(f"| {LABELS[r['variant']]} | {r['metric']} | [{float(r['lower']):.6g}, {float(r['upper']):.6g}] |")
    lines+=['','ช่วงเหล่านี้ conditional on fixed fitted models ไม่รวม training/selection uncertainty '
        '[CI ทุก metrics](test_bootstrap_ci.csv) · [Calibration bins](calibration_bins.csv)','',
        '## กราฟและการทำซ้ำ','','![Development nonlinear comparison](comparison.png)','',
        '```bash','python -m pip install -r round3/requirements-balancing.txt',
        'OPENBLAS_NUM_THREADS=1 python -m round3.nonlinear_study --source /path/to/nhefs_source --output round3_work/nonlinear_new',
        '# Additional plotting dependency: matplotlib',
        'python -m round3.report_nonlinear --results round3_work/nonlinear_new','```','',
        '[Software tests](software_tests.txt) · [Independent verification](independent_verification.json) · '
        '[Model parameters](candidate_models.json) · [Final inner OOF](final_inner_oof.csv) · [Provenance](provenance.json)','',
        'Parameters replay ด้วย PortableModel และ 9 encoded input columns เดิม; quadratic powers และ expanded scaler '
        'ถูกเก็บใน JSON การตีความ coefficient ของ squared term ต้องร่วมกับ linear term '
        'จึงไม่ควรใช้เครื่องหมาย coefficient ตัวเดียวอธิบายว่าเพิ่มหรือลดความเสี่ยงเสมอ','',
        '## ข้อสรุปและข้อจำกัด','','ยังไม่แทนรุ่น weight+smoking C=.1 ด้วย nonlinear/selected workflow '
        'การทดลองนี้ได้ตรวจสมมติฐานเพิ่มความโค้งแล้ว แต่ไม่มีหลักฐานพอว่าดีขึ้น '
        'การลองรูปแบบเพิ่มเรื่อย ๆ บนข้อมูลเดิมอาจทำให้เลือกผลบังเอิญ '
        'งานถัดไปควรกำหนดคำถามเฉพาะ เช่น balancing ที่ threshold .5 ตามโจทย์วิชา '
        'หรือหาข้อมูลตั้งต้นเพิ่มเติมที่ยืนยันจาก codebook พร้อมหาชุดข้อมูลใหม่สำหรับตรวจสอบ','',
        'ผลใช้กับ reported first diagnosis ภายใน 8 ปีใน selected historical NHEFS US cohort '
        'ยังมี selection bias, missing follow-up, history of model selection และไม่มี external/Thai validation '
        'ไม่ใช่การวินิจฉัยจากการวัดจริงหรือหลักฐานว่าไม่เคยเป็นโรคตลอดชีวิต','']
    (root/'README_TH.md').write_text('\n'.join(lines),encoding='utf-8')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,3,figsize=(14,4.5),layout='constrained')
    for ax,key,title in zip(axes,['roc_auc','brier','fpr'],['Development ROC-AUC','Development Brier (lower is better)','Observed FPR under inner 14.5% budget']):
        a=np.array([vals(v,key) for v in VARIANTS]);ax.errorbar(range(4),a.mean(axis=1),yerr=a.std(axis=1,ddof=1),fmt='o',capsize=5)
        ax.set_xticks(range(4),['Original','Tuned linear','Tuned quadratic','Selected'],rotation=20,ha='right')
        ax.set_title(title);ax.grid(axis='y',alpha=.2)
        if key=='fpr':ax.axhline(.145,color='gray',ls='--')
    fig.suptitle('Weight + smoking: 3-repeat mean ± SD (not confidence intervals)')
    fig.savefig(root/'comparison.png',dpi=170);plt.close(fig)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--results',type=Path,required=True);a=p.parse_args();run(a.results)
