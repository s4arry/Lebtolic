# รุ่นอ้างอิง — weight + smoking จาก befec00

สถานะ: **research reference ที่ผู้ใช้เลือก** ไม่ใช่ clinical model หรือหลักฐานว่าเหนือกว่า main
[ผลเทียบ main](../round3/results/main_comparison_20261111/README_TH.md)

## Artifact ที่ยึดร่วมกัน

- [model.json](model.json): คัดลอก key `none` จาก calibration_20261111/candidate_models.json ตรงทุก parameter
- [manifest.json](manifest.json): source commit, SHA-256 และ primary policy
- [metrics.json](metrics.json): Test metrics ของรุ่น/threshold เดียวกัน
- [ตัวอย่าง input](example_encoded.json), [ผลจริงจากตัวอย่าง](example_output.json): input สมมติ ไม่ใช่ผู้ป่วย

Logistic Regression C=.1, ไม่ balance และไม่เพิ่ม calibration
fit บน development 2311 คน; imputer/scaler เรียนจาก development ตอน fit final model
primary policy `fpr145`, threshold **0.09803399686084495**; F2 threshold เก็บเป็นทางเลือกใน JSON
แต่ไม่ใช่ค่า default ของรายงานปัจจุบัน

## ผลที่อ้างอิง

| Metric | Development CV mean | Test เดิม |
|---|---:|---:|
| ROC-AUC | 0.7294 | 0.7034 |
| Recall | 43.97% | 40% |
| Precision | 17.11% | 16.95% |
| F2 | 0.3346 | 0.3145 |
| FPR | 13.84% | 12.76% |
| Brier | 0.055488 | 0.057010 |

Test n409/events25, TP10/FP49/FN15/TN335; Accuracy84.35%
[ผลทุก fold และความไม่แน่นอน](../round3/results/calibration_20261111/README_TH.md)
Test ไม่ใช่ชุดใหม่ที่ไม่เคยเห็น ไม่มี external หรือ Thai validation

## Encoded input สำหรับ replay งานวิจัย

| ลำดับ | ชื่อ | ความหมาย |
|---|---|---|
| 0 | age_years | อายุ baseline |
| 1 | sex_female | 0 ชาย / 1 หญิง ตาม encoding ข้อมูลต้นทาง |
| 2 | baseline_sbp | mmHg; cohort SBP<130 |
| 3 | baseline_dbp | mmHg; cohort DBP<80 |
| 4 | weight_lb | น้ำหนัก baseline หน่วย **pounds** จาก PMR5b1/b2; ไม่มีค่าใช้ NaN |
| 5 | weight_missing | 1 เมื่อน้ำหนักไม่มีข้อมูลที่ใช้ได้ มิฉะนั้น 0 |
| 6 | current_smoker | 1 เมื่อ L1=1 และ L2a=1 |
| 7 | former_smoker | 1 เมื่อ L1=1 และ L2a=2 |
| 8 | smoking_unknown | หมวดไม่ทราบ/ขัดแย้ง; ไม่มีตัวอย่างหมวดนี้ใน cohort ที่ฝึก |

หมวด reference ของ smoking คือรายงานไม่เคยสูบถึง 100 มวน (L1=2 และ L2a=7/ว่าง)
ไม่ควรตีความเป็นไม่เคยสูบแม้แต่มวนเดียว ตัวแปร smoking flags ต้องตรงกฎใน
[codebook/encoding](../round3/CODEBOOK_BALANCING_TH.md) และ `balancing_study.encode_extra`
น้ำหนัก 995/996/997/999 ไม่ใช่ pounds; ใช้เฉพาะ reading 69–300 และเฉลี่ย readings ที่ใช้ได้
unknown smoking ไม่มีข้อมูลให้เรียนผลใน cohort นี้ จึงไม่อ้าง generalization ของหมวดนั้น

## โหลดผลโดยไม่ retrain

รันจาก repository root หลังติดตั้ง requirements:

```python
import json
from pathlib import Path
from round3.predict import PortableModel

state = json.loads(Path('model/model.json').read_text())
example = json.loads(Path('model/example_encoded.json').read_text())
score = float(PortableModel(state).predict_proba([example['row']])[0, 1])
threshold = state['threshold_policies']['fpr145']['threshold']
print({'experimental_score': score, 'experimental_alert': score >= threshold})
```

นี่เป็น numerical replay API ซึ่งรับ encoded vector โดยตรง ไม่ใช่แบบฟอร์มคัดกรองที่มี
input eligibility validation ครบสำหรับผู้ใช้ทั่วไป และไม่ควรนำผลไปใช้ตัดสินใจรักษา
legacy `round3.predict` CLI และ notebook input form รุ่น 5-feature ยังไม่รองรับ schema นี้
ฟอร์มสำหรับ demo ของรุ่นนี้ (รับค่าแบบคนทั่วไป + guardrails) อยู่ที่ [demo/README.md](../demo/README.md)

## ขอบเขต

ผลลัพธ์คือ reported diagnosis ภายใน 8 ปีใน selected historical NHEFS cohort
ไม่ใช่การตรวจโรคจากการวัด ไม่มีหลักฐานยืนยันกับคนไทย คัดผู้ติดตามไม่พอ/ประวัติไม่แน่ใจออก
จึงยังมี selection bias และไม่ใช่คำยืนยันว่าไม่มีโรคตลอดชีวิต
การ calibration และ nonlinear ที่ลองเพิ่มยังไม่ผ่านเกณฑ์พอให้เปลี่ยนรุ่นอ้างอิง
[ประวัติการทดลองทั้งหมด](../EXPERIMENTS.md)
