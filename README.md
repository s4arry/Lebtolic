# LEPTOLIC — INT182

ต้นแบบวิจัยเพื่อทำนาย **การรายงานว่าได้รับการวินิจฉัยความดันสูงครั้งแรกภายใน 8 ปี**
จาก NHEFS สหรัฐฯ ในอดีต ยังไม่มี external หรือ Thai validation

## รุ่นอ้างอิงปัจจุบันบน main

หลัง merge `dev-by-gunner` ผ่าน PR #2 (15 พฤศจิกายน 2026) รุ่นอ้างอิงสำหรับงานวิจัยคือ
**Logistic Regression C=0.1 + weight + smoking**, ไม่ใช้ class balancing และไม่เพิ่ม calibration
เลือกจาก Development ตาม protocol ไม่ได้เลือกจากผล Test

- 6 ตัวแปรตั้งต้น → 9 encoded columns
- Primary policy: `fpr145`, threshold **0.09803399686084495** (มาจาก development inner OOF)
- Fit final preprocessing/model บน development 2,311 คน

**[รายละเอียดและ encoding](model/README.md)** · **[พารามิเตอร์](model/model.json)** ·
**[manifest](model/manifest.json)** · **[รายงานเปรียบเทียบ main เดิม](round3/results/main_comparison_20261111/README_TH.md)**

| Metric | Development CV (mean 3 repeats) | Test เดิม |
|---|---:|---:|
| ROC-AUC | 0.7294 | 0.7034 |
| Recall ณ FPR-policy | 43.97% | 40.00% |
| Precision ณ FPR-policy | 17.11% | 16.95% |
| F2 ณ FPR-policy | 0.3346 | 0.3145 |
| FPR ที่พบจริง | 13.84% | 12.76% |
| Brier ↓ | 0.05549 | 0.05701 |

Cohort **2,720 คน / 166 events**; development **2,311/141**; Test **409/25**
Test TP **10**, FP **49**, FN **15**, TN **335**; accuracy **84.35%**.
Accuracy สูงไม่แปลว่าจับเหตุการณ์ได้ดี เพราะเหตุการณ์มีเพียงประมาณ 6.1%

## ข้อจำกัดและการตีความ

การเปรียบเทียบกับ notebook เดิมบน main พบ ROC-AUC เพิ่มจาก 0.6761 เป็น 0.7034
แต่ Brier แย่ลงจาก 0.05653 เป็น 0.05701 และ paired uncertainty intervals คร่อมศูนย์
จึง **ยังยืนยันไม่ได้ว่ารุ่นใหม่เหนือกว่าโดยรวม** การเพิ่ม Recall ขึ้นกับ threshold/policy
และแลกกับ false positives ไม่ใช่ผลจากโมเดลเพียงอย่างเดียว

Test ชุดเดิมถูกเปิดดูหลายครั้งแล้ว เป็น exploratory internal evaluation ไม่ใช่ unseen external validation
ผลลัพธ์เป็น reported diagnosis ใน cohort NHEFS สหรัฐฯ ในอดีต ไม่ใช่การวัดยืนยันโรค
ยังไม่มี Thai validation และไม่ใช่เครื่องมือคัดกรอง วินิจฉัย หรือใช้ตัดสินใจรักษารายบุคคล

## สื่อสำหรับนำเสนอ

[เนื้อหา 12 หัวข้อ พร้อมภาพและ CSV](presentation/README.md) · [สรุปโมเดลและข้อแลกเปลี่ยน](presentation/MODEL_SUMMARY_TH.md)

## เริ่มใช้งาน / ทำซ้ำ

```bash
python -m pip install -r requirements.txt
```

- **Demo ฟอร์มกรอกค่า:** [LEPTOLIC_Demo.ipynb](LEPTOLIC_Demo.ipynb) · [วิธีใช้และ guardrails](demo/README.md)
- โหลดพารามิเตอร์และดู encoding: [model/README.md](model/README.md)
- รายงาน calibration: [calibration_20261111](round3/results/calibration_20261111/README_TH.md)
- รายละเอียดวิธีทดลองและผลความแปรปรวน: [round3](round3/README.md)
- ประวัติการทดลอง/รุ่นที่ไม่ถูกเลือก: [EXPERIMENTS.md](EXPERIMENTS.md)

## โครงสร้างที่ใช้เป็นหลัก

| ตำแหน่ง | หน้าที่ |
|---|---|
| `model/` | พารามิเตอร์, manifest, metrics และตัวอย่างของรุ่นอ้างอิง |
| `demo/` + `LEPTOLIC_Demo.ipynb` | ฟอร์ม demo ของรุ่นอ้างอิง: แปลงคำตอบเป็น 9 columns + guardrails |
| `round3/compare_main.py` | รัน notebook algorithm ของ main เดิมที่ pin commit และเทียบรุ่นอ้างอิง |
| `round3/calibration_study.py` | ทดลอง calibration ที่นำไปสู่รุ่นอ้างอิง |
| `round3/results/` | หลักฐานรอบต่าง ๆ แยกโฟลเดอร์ ไม่ใช้ผลคนละรุ่นปะปนกัน |
| `tests/` | Software tests; ไม่ใช่ผลประสิทธิภาพทางการแพทย์ |
| `LEPTOLIC_NHEFS_Future_Prediction.ipynb` | Notebook baseline เดิม เก็บไว้เป็นประวัติและใช้ตรวจสอบ |
| `LEPTOLIC_Round3_Review.ipynb` | Notebook การทดลองรอบแรก ไม่ใช่ demo ของรุ่นอ้างอิง |

[คู่มือพื้นฐานวิชา](AI_DATA_SCIENCE_FUNDAMENTALS.md) ·
[คำอธิบาย notebook baseline เดิม](CODE_WALKTHROUGH.md)
