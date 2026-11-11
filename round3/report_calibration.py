"""Verify predictions and render the calibration follow-up report."""
import argparse,json,shutil
from collections import Counter
from pathlib import Path
import numpy as np
from sklearn.metrics import roc_auc_score
from .balancing_study import digest,policies,write_csv
from .threshold_policy import read_csv
from .workflow import save_json

METHODS=['none','intercept','sigmoid','selected']

def run(root):
    if json.loads((root/'study_status.json').read_text())['status']!='completed':raise ValueError('Run incomplete')
    decision=json.loads((root/'decision_before_test.json').read_text());verification=json.loads((root/'verification.json').read_text())
    assert digest(Path(__file__).with_name('calibration_study.py'))==verification['code_sha256']
    assert digest(root/'decision_before_test.json')==verification['decision_sha256']
    oof=read_csv(root/'development_oof.csv');rows=read_csv(root/'repeat_metrics.csv')
    def check(y,p,pred,row):
        tp=int(np.sum((y==1)&pred));fp=int(np.sum((y==0)&pred));fn=int(np.sum((y==1)&~pred));tn=int(np.sum((y==0)&~pred))
        expected=dict(tp=tp,fp=fp,fn=fn,tn=tn,recall=tp/(tp+fn),precision=tp/(tp+fp) if tp+fp else None,
            f2=5*tp/(5*tp+4*fn+fp),fpr=fp/(fp+tn),brier=float(np.mean((y-p)**2)),roc_auc=float(roc_auc_score(y,p)))
        for k,v in expected.items():
            if v is not None:np.testing.assert_allclose(float(row[k]),v,atol=1e-12,rtol=0)
    for row in rows:
        rs=sorted([r for r in oof if r['variant']==row['variant'] and r['repeat']==row['repeat']],key=lambda r:int(r['development_row']))
        assert [int(r['development_row']) for r in rs]==list(range(2311))
        y=np.array([int(r['target']) for r in rs]);p=np.array([float(r['score']) for r in rs])
        pred=np.array([int(r[row['policy']+'_prediction']) for r in rs],bool);check(y,p,pred,row)
    inner=read_csv(root/'final_inner_oof.csv');y=np.array([int(r['target']) for r in inner])
    for m in METHODS:
        key=decision['final']['chosen'] if m=='selected' else m
        assert policies(y,np.array([float(r[key]) for r in inner]))==decision['thresholds'][m]
    test=read_csv(root/'test_metrics.csv');scores=read_csv(root/'test_scores.csv');yt=np.array([int(r['target']) for r in scores])
    for row in test:
        p=np.array([float(r[row['variant']]) for r in scores]);check(yt,p,p>=float(row['threshold']),row)
    prior=root.parent/'balancing_20261111/development_oof.csv'
    old=[r for r in read_csv(prior) if r['variant']=='weight_smoking']
    new=[r for r in oof if r['variant']=='none']
    assert len(old)==len(new)
    for a,b in zip(old,new):
        for key in ['repeat','development_row','target','f2_prediction','fpr145_prediction']:assert a[key]==b[key]
        np.testing.assert_allclose(float(a['score']),float(b['score']),atol=1e-12,rtol=0)
    save_json(root/'independent_verification.json',dict(status='passed',checks=[
        'All repeat/test confusion counts, AUC, Brier, F2 independently reconstructed',
        '2311 distinct development rows per repeat and candidate',
        'Final thresholds exactly reconstructed from saved inner OOF; Test unused',
        'Uncalibrated outer OOF scores match prior weight_smoking within 1e-12; decisions match exactly',
        'Code hash and pre-Test decision hash verified']))
    def values(m,key,policy='fpr145'):
        return np.array([float(r[key]) for r in rows if r['variant']==m and r['policy']==policy])
    def stat(m,key,policy='fpr145',pct=False):
        a=values(m,key,policy)*(100 if pct else 1)
        return f'{a.mean():.5f} ± {a.std(ddof=1):.5f}'
    choicecounts=Counter(c['chosen'] for c in json.loads((root/'outer_choices.json').read_text()))
    lines=['# Calibration รุ่นน้ำหนัก + ประวัติสูบบุหรี่: ผลจริง 11 พฤศจิกายน 2026','',
        '**ยังไม่มีหลักฐานจาก development ให้เพิ่ม calibration ในรุ่นทดลองนี้**','',
        f"Final method: **{decision['final']['chosen']}**; outer selections: **{dict(choicecounts)}**. "
        'กติกาเลือกวิธีพิจารณา inner Brier และเลือกวิธีง่ายกว่าหากอยู่ในช่วง one-SE; '
        'ช่วงนี้เป็น heuristic สำหรับลดความซับซ้อน ไม่ใช่ confidence interval เชิงอนุมาน','',
        '[Protocol ก่อนรัน](../../CALIBRATION_PROTOCOL_TH.md) · [โค้ด](../../calibration_study.py) · '
        '[Decision ก่อน Test](decision_before_test.json)','',
        '## ปัจจัยที่ตรวจและสิ่งที่พัฒนา','',
        '- รุ่นตั้งต้นมี calibration slope บน development ใกล้ 1 และค่าเฉลี่ยคะแนนใกล้อัตราเหตุการณ์จริงอยู่แล้ว '
        'จึงมีพื้นที่ให้ calibration เพิ่มประโยชน์ไม่มาก',
        '- ทดลอง intercept correction เพื่อปรับระดับคะแนน และ sigmoid เพื่อปรับระดับกับความชัน '
        'ใช้พารามิเตอร์น้อย เพราะมี development events 141 คน; ไม่เลือก isotonic ที่ยืดหยุ่นมากในรอบนี้',
        '- เพิ่มการเลือกวิธี calibration ใน inner 3-fold โดยมี calibration cross-fitting 3-fold '
        'ซ้อนใน inner training อีกชั้น ทุก preprocessing/calibrator เห็นเฉพาะ training ของตน',
        '- ประเมินด้วย outer 5-fold × 3 repeats เดิม และเลือก F2/FPR14.5% threshold ภายใน outer training '
        'รายงาน fixed candidates และ selected workflow แยกกัน',
        '- คง C=.1, cohort 2720/166 events, development 2311/141 และ Test 409/25 '
        'ไม่เพิ่มฟีเจอร์หรือเปลี่ยนนิยาม outcome ในรอบนี้','',
        '## Development: probability metrics','',
        'Mean ± SD ของ 3 pooled repeats; repeats/folds ไม่อิสระ ไม่ถือว่าเป็นคนเพิ่ม','',
        '| Method | ROC-AUC | Average precision | Brier ↓ | Calibration intercept-in-large (ideal 0) | Calibration slope (ideal 1) |',
        '|---|---:|---:|---:|---:|---:|']
    for m in METHODS:lines.append('| '+m+' | '+' | '.join(stat(m,k) for k in ['roc_auc','ap','brier','calibration_in_large','calibration_slope'])+' |')
    lines+=['','## Development: threshold policies','','| Method | Policy | Recall % | Precision % | F2 | FPR % |','|---|---|---:|---:|---:|---:|']
    for m in METHODS:
        for policy in ['f2','fpr145']:
            lines.append('| '+m+' | '+policy+' | '+' | '.join([stat(m,'recall',policy,True),stat(m,'precision',policy,True),stat(m,'f2',policy),stat(m,'fpr',policy,True)])+' |')
    lines+=['','Sigmoid ทำให้ slope ใกล้ 1 มากขึ้น แต่ Brier ของ outer validation ไม่ลด และ budget recall '
        'ไม่ได้ดีขึ้น ส่วน intercept มีผลน้อยมาก การแก้ calibration slope เพียงค่าเดียวจึงไม่เพียงพอให้เลือกโมเดล','',
        '[TP/FP/FN/TN และ metrics ทุก repeat](repeat_metrics.csv) · [ทุก outer fold](outer_folds.csv) · '
        '[Outer mean/SD/range](outer_summary.csv)','',
        '## Final inner selection ก่อนเปิด Test','','| Method | Inner mean Brier | SD across 3 folds |','|---|---:|---:|']
    for m,c in decision['final']['candidates'].items():lines.append(f"| {m} | {c['mean_brier']:.8f} | {c['sd']:.8f} |")
    lines+=['',f"Best point estimate: {decision['final']['best_mean_method']}; one-SE limit = {decision['final']['one_se_limit']:.8f}. "
        'ผลต่างเล็กเมื่อเทียบความแกว่งระหว่าง folds จึงเลือก none ตามกติกาที่กำหนดก่อนรัน '
        'ไม่เปลี่ยนไปเลือก sigmoid เพราะผล Test ดูดีกว่า','',
        '## Historical Test: exploratory','','| Method | Policy | Threshold % | TP | FP | FN | TN | Recall % | Precision % | F2 | AUC | Brier |',
        '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for r in test:
        cols=[r['variant'],r['policy'],f"{float(r['threshold'])*100:.4f}"]+[r[k] for k in ['tp','fp','fn','tn']]
        cols += [f"{float(r[k])*100:.2f}" for k in ['recall','precision']]+[f"{float(r[k]):.6f}" for k in ['f2','roc_auc','brier']]
        lines.append('| '+' | '.join(cols)+' |')
    lines+=['','**เหตุที่ none รอบนี้จับ 10 คนแทน 13 คนของรุ่นทดลองรอบก่อน:** คะแนนของโมเดลเหมือนเดิมทุกค่า '
        'แต่ final threshold รอบนี้เลือกจาก inner 3-fold OOF ตาม protocol ใหม่; รอบก่อนเลือกจากค่าเฉลี่ย '
        'repeated outer OOF การเปลี่ยนจำนวนแจ้งเตือนนี้มาจาก threshold ไม่ใช่โมเดลเสื่อมลง '
        'ไฟล์นโยบายรอบก่อนยังอยู่ และไม่มีการเขียนทับ default','',
        'AUC ของ final models เท่ากัน เพราะ intercept/sigmoid ในรอบนี้เป็นการแปลงคะแนนที่รักษาลำดับ '
        'ส่วน pooled outer AUC อาจต่างเล็กน้อยเพราะแต่ละ fold ใช้ mapping ต่างกัน','',
        'Paired Test differences (candidate − uncalibrated, FPR-policy); Brier ต่ำกว่าดีกว่า:', '',
        '| Method | Metric | 95% paired bootstrap interval |','|---|---|---:|']
    for r in read_csv(root/'test_paired_difference_ci.csv'):
        if r['variant']!='selected' and r['policy']=='fpr145' and r['metric'] in ['brier','recall','precision','f2']:
            lines.append(f"| {r['variant']} | {r['metric']} | [{float(r['lower']):.6g}, {float(r['upper']):.6g}] |")
    lines+=['','Bootstrap 2000 draws เป็น uncertainty ของ fixed fitted models บน Test เดิม '
        'ไม่รวมความแปรปรวนการฝึกและการเลือกโมเดล [CI ทุก metric](test_bootstrap_ci.csv)','',
        '## กราฟและหลักฐาน','','![Calibration comparison](comparison.png)','',
        '[Calibration bins + Wilson intervals](calibration_bins.csv) · [Parameters ทุกโมเดล](candidate_models.json) · '
        '[Final inner OOF](final_inner_oof.csv) · [Verification](independent_verification.json) · [Software tests](software_tests.txt)','',
        '## รันซ้ำ','','```bash','python -m pip install -r round3/requirements-balancing.txt',
        'OPENBLAS_NUM_THREADS=1 python -m round3.calibration_study --source /path/to/nhefs_source --output round3_work/calibration_new',
        '# Rendering additionally requires matplotlib',
        'python -m round3.report_calibration --results round3_work/calibration_new','```','',
        'none ในรายงานนี้หมายถึงรุ่นน้ำหนัก+สูบบุหรี่ที่ไม่ calibrate ไม่ใช่โมเดลหลัก 4 features '
        'พารามิเตอร์ replay ด้วย PortableModel และ 9 encoded columns ตามลำดับใน JSON '
        'legacy predict_file input form ยังไม่รองรับ schema นี้','',
        '## ข้อจำกัดและขั้นต่อไป','','การ calibration รอบนี้ยังไม่ให้เหตุผลเพียงพอสำหรับการเปลี่ยนโมเดล '
        'แนวทางถัดไปที่มีเหตุผลคือทดลอง nonlinear effects ของอายุ/ความดัน/น้ำหนักแบบจำกัดความซับซ้อน '
        'เพื่อเพิ่มข้อมูลในการจัดลำดับคะแนน พร้อมทดสอบบนข้อมูลใหม่เมื่อหาได้ '
        'นี่เป็นข้อเสนอ ยังไม่ได้รันในงาน calibration นี้','',
        'Test เคยเปิดดูหลายครั้ง และการเลือก feature set มาจากการทดลองก่อนหน้า '
        'nested validation รอบนี้ไม่ลบประวัติ selection optimism เหล่านั้น '
        'ข้อมูลเป็น reported first diagnosis ภายใน 8 ปีใน selected historical US cohort '
        'ไม่ใช่การยืนยันโรคจากการวัด ไม่มี external/Thai validation และไม่ใช่หลักฐานว่าไม่เคยเป็นโรคตลอดชีวิต','']
    (root/'README_TH.md').write_text('\n'.join(lines),encoding='utf-8')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,3,figsize=(14,4.5),layout='constrained')
    for ax,key,title in zip(axes,['brier','calibration_slope','recall'],['Development Brier (lower is better)','Calibration slope (ideal 1)','Recall at nested FPR budget']):
        a=np.array([values(m,key) for m in METHODS[:3]])
        ax.errorbar(range(3),a.mean(axis=1),yerr=a.std(axis=1,ddof=1),fmt='o',capsize=5)
        ax.set_xticks(range(3),METHODS[:3]);ax.set_title(title);ax.grid(axis='y',alpha=.2)
        if key=='calibration_slope':ax.axhline(1,color='gray',ls='--')
    fig.suptitle('Weight + smoking model: 3-repeat mean ± SD (not confidence intervals)')
    fig.savefig(root/'comparison.png',dpi=170);plt.close(fig)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--results',type=Path,required=True);a=p.parse_args();run(a.results)
