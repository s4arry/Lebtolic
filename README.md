# LEPTOLIC — INT182

ต้นแบบวิจัยเพื่อทำนาย **การรายงานว่าได้รับการวินิจฉัยความดันสูงครั้งแรกภายใน 8 ปี**
จาก NHEFS สหรัฐฯ ในอดีต ยังไม่มี external หรือ Thai validation

## รุ่นอ้างอิงของ branch นี้

ยึดรุ่นที่ผู้ใช้เลือกจาก **commit `befec00`**: Logistic Regression C=.1,
ไม่มี balancing และไม่เพิ่ม calibration ใช้อายุ เพศ SBP DBP น้ำหนัก และประวัติสูบบุหรี่
(6 ตัวแปรตั้งต้น → 9 encoded columns) ใช้ primary policy **FPR 14.5%**
ที่เลือก threshold จาก development inner OOF = **0.09803399686084495**

**[ข้อมูลโมเดลและวิธีใช้](model/README.md)** · **[พารามิเตอร์รุ่นอ้างอิง](model/model.json)** ·
**[ผลเทียบ main จริง](round3/results/main_comparison_20261111/README_TH.md)**

| Metric | Development CV (mean 3 repeats) | Test เดิม |
|---|---:|---:|
| ROC-AUC | 0.7294 | 0.7034 |
| Recall ณ FPR-policy | 43.97% | 40.00% |
| Precision ณ FPR-policy | 17.11% | 16.95% |
| F2 ณ FPR-policy | 0.3346 | 0.3145 |
| FPR ที่พบจริง | 13.84% | 12.76% |
| Brier ↓ | 0.05549 | 0.05701 |

Cohort **2720 คน / 166 events**; development **2311/141**; Test **409/25**
Test TP **10**, FP **49**, FN **15**, TN **335**; accuracy **84.35%**
Accuracy สูงไม่แปลว่าจับเหตุการณ์ได้ดี เพราะเหตุการณ์มีเพียงประมาณ 6.1%

## ข้อสรุปจากการเทียบ main

Notebook จริงบน main (`6ce4577`) เป็นรุ่น 5 features + Platt: Test AUC **0.6761**,
Brier **0.05653** เทียบรุ่นอ้างอิง AUC **0.7034**, Brier **0.05701**
AUC สูงขึ้น แต่ Brier แย่ลงเล็กน้อย และ paired uncertainty ยังไม่ยืนยัน superiority
ที่ threshold .5 ทั้งสองไม่ flag ใคร; การเพิ่ม recall เมื่อเปลี่ยน threshold
จึงไม่ควรตีความว่าเป็นผลจากความสามารถโมเดลทั้งหมด

รุ่นนี้เป็น **รุ่นอ้างอิงที่ผู้ใช้เลือกสำหรับงานวิจัย** ไม่ใช่โมเดลที่พิสูจน์แล้วว่าดีกว่าทุกด้าน
ไม่ได้ merge หรือแทน main และยังไม่พร้อมใช้คัดกรองจริง/ประเมินความเสี่ยงส่วนบุคคล
Test เคยเปิดดูหลายครั้งแล้ว ผลทั้งหมดเป็น exploratory internal evaluation

## สื่อสำหรับนำเสนอ

[เนื้อหา 12 หัวข้อ พร้อมภาพและ CSV](presentation/README.md) · [สรุปโมเดลและข้อแลกเปลี่ยน](presentation/MODEL_SUMMARY_TH.md)

## เริ่มใช้งาน / ทำซ้ำ

```bash
python -m pip install -r requirements.txt
```

- **Demo ฟอร์มกรอกค่า:** [LEPTOLIC_Demo.ipynb](LEPTOLIC_Demo.ipynb) · [วิธีใช้และ guardrails](demo/README.md)
- โหลดพารามิเตอร์และดู encoding: [model/README.md](model/README.md)
- ทำซ้ำการเลือก calibration ที่ได้รุ่นนี้: [รายงาน befec00](round3/results/calibration_20261111/README_TH.md)
- รายละเอียดวิธีทดลองและผลความแปรปรวน: [round3](round3/README.md)
- ประวัติการทดลอง/รุ่นที่ไม่ถูกเลือก: [EXPERIMENTS.md](EXPERIMENTS.md)

## โครงสร้างที่ใช้เป็นหลัก

| ตำแหน่ง | หน้าที่ |
|---|---|
| `model/` | พารามิเตอร์, manifest, metrics และตัวอย่างของรุ่นอ้างอิงเพียงรุ่นเดียว |
| `demo/` + `LEPTOLIC_Demo.ipynb` | ฟอร์ม demo ของรุ่นอ้างอิง: แปลงคำตอบเป็น 9 columns + guardrails |
| `round3/compare_main.py` | รัน notebook algorithm ของ main ที่ pin commit และเทียบรุ่นอ้างอิง |
| `round3/calibration_study.py` | ทดลอง calibration ที่นำไปสู่รุ่น befec00 |
| `round3/results/` | หลักฐานรอบต่าง ๆ แยกโฟลเดอร์ ไม่ใช้ผลคนละรุ่นปะปนกัน |
| `tests/` | Software tests; ไม่ใช่ผลประสิทธิภาพทางการแพทย์ |
| `LEPTOLIC_NHEFS_Future_Prediction.ipynb` | Notebook baseline เดิมจาก main; เก็บไว้ใช้ตรวจ/อ่านข้อมูล ห้ามสับสนกับรุ่นอ้างอิง |
| `LEPTOLIC_Round3_Review.ipynb` | Notebook ของการทดลองรอบแรก ไม่ใช่ demo ของรุ่นอ้างอิง |

[คู่มือพื้นฐานวิชา](AI_DATA_SCIENCE_FUNDAMENTALS.md) ·
[คำอธิบาย notebook baseline เดิม](CODE_WALKTHROUGH.md)
