# เปรียบเทียบ main จริง กับรุ่นอ้างอิง befec00

**AUC สูงขึ้น แต่ยังไม่ยืนยันว่าดีกว่าโดยรวม: Brier แย่ลงเล็กน้อยและ paired intervals คร่อมศูนย์**

- main ที่ตรวจ: `6ce4577dc3fa40ea1d3fe1e4e726119089c378ba`
- รุ่นที่ผู้ใช้เลือก: `befec00faa9d341b983c984adc087ce0cf4585a8` → weight+smoking, Logistic C=.1, ไม่มี calibration
- อ่านและ fit notebook จริงของ main ซ้ำ ได้ lambda=10 และ AUC/Brier ตรง saved output ถึง tolerance 1e-12
- main notebook มี 5 features (อายุ เพศ SBP DBP ชั่วโมงนอน) + Platt; ฝึก base บน Train 1495 คน ส่วน befec00 มี 6 ตัวแปรตั้งต้น/9 encoded columns และ fit บน development 2311 คน จึงเปรียบเทียบทั้ง workflow ไม่ใช่แยกผลของฟีเจอร์หรืออัลกอริทึมอย่างเดียว

## สิ่งที่พบใน main

ไฟล์ `LEPTOLIC_Model_Update_Summary.md` บน main กล่าวถึง 6 features/refit/AUC .682 แต่ notebook ที่ commit จริงยังมี 5 features/AUC .6761458333 และไม่มีการแก้ตามคำอธิบายนั้น รายงานนี้ยึด executable notebook และผล fit ซ้ำ ไม่ถือ summary เป็นหลักฐานว่าโมเดล 6-feature ถูกส่งขึ้นแล้ว

## Test เดิมชุดเดียวกัน: 409 คน / 25 events

| Model | Policy | Threshold % | AUC | Brier ↓ | TP | FP | FN | TN | Recall % | Precision % | FPR % |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| main | default_0_5 | 50.0000 | 0.676146 | 0.056529 | 0 | 0 | 25 | 384 | 0.00 | ไม่กำหนด | 0.00 |
| main | f2 | 2.8431 | 0.676146 | 0.056529 | 25 | 379 | 0 | 5 | 100.00 | 6.19 | 98.70 |
| main | fpr145 | 8.8765 | 0.676146 | 0.056529 | 2 | 14 | 23 | 370 | 8.00 | 12.50 | 3.65 |
| befec00 | default_0_5 | 50.0000 | 0.703438 | 0.057010 | 0 | 0 | 25 | 384 | 0.00 | ไม่กำหนด | 0.00 |
| befec00 | f2 | 6.0787 | 0.703438 | 0.057010 | 15 | 119 | 10 | 265 | 60.00 | 11.19 | 30.99 |
| befec00 | fpr145 | 9.8034 | 0.703438 | 0.057010 | 10 | 49 | 15 | 335 | 40.00 | 16.95 | 12.76 |

**default_0_5 ของ main คือกติกาที่อยู่ใน notebook จริง** ส่วน F2/FPR145 ของ main เป็นนโยบายทดลองเพิ่มเติมในงานเปรียบเทียบนี้ เลือกจาก development OOF เท่านั้น ไม่ใช่นโยบายที่เคยมีบน main

เพื่อเทียบกติกา threshold ให้ใกล้กัน main สร้าง 3-fold development OOF โดยในแต่ละ training แบ่ง Train/Tune/Calibration ตามสัดส่วน 55:15:15 แล้วฝึก algorithm เดิม befec00 ใช้ inner OOF/threshold ที่ freeze ไว้ใน commit เดิม ทั้งสองเลือก threshold ก่อนคำนวณ Test รอบนี้ การคัดเลือก feature/model ในอดีตยังเป็นข้อจำกัด

**ไม่ควรสรุปว่า recall เพิ่มจาก 0% เป็น 40% เพราะโมเดลเก่งขึ้นเพียงอย่างเดียว:** ที่ threshold .5 ทั้งสองไม่ flag ใคร เมื่อใช้ FPR-policy แต่ละโมเดลจึงเริ่ม flag และแม้ใช้กติกางบเดียวกัน FPR ที่ Test ได้จริงก็ต่างกัน (main 3.65%, befec00 12.76%) คำเตือนผิดที่เพิ่มเป็นส่วนหนึ่งของ recall ที่เพิ่ม

Main OOF threshold ถ่ายโอนไป final model ได้ไม่พอดี: F2 เตือนเกือบทุกคน และ FPR-policy เตือนน้อยกว่างบมาก สะท้อนความไม่เสถียรของคะแนนจาก workflow ที่แยกข้อมูล calibrate และมี events น้อย ไม่ใช่เหตุผลให้เลือก threshold ใหม่จาก Test

## Paired uncertainty (befec00 − main)

| Metric | 95% bootstrap interval |
|---|---:|
| recall | [0.100000, 0.545455] |
| precision | [-0.133344, 0.205894] |
| f2 | [0.043871, 0.408153] |
| fpr | [0.059278, 0.126289] |
| roc_auc | [-0.053275, 0.112247] |
| brier | [-0.001983, 0.002821] |

AUC point difference +0.027292; Brier difference +0.000481 (ค่าบวกของ Brier คือแย่กว่า) ทั้งสอง interval คร่อมศูนย์ จึงไม่ยืนยัน superiority bootstrap 2000 draws เป็นความไม่แน่นอน conditional on fixed models ไม่รวม training/selection uncertainty

## การตัดสินใจจัด branch

ยึด befec00 เป็น **รุ่นอ้างอิงสำหรับงานวิจัยตามที่ผู้ใช้เลือก** ไม่ใช่ผู้ชนะที่เลือกใหม่จาก Test ไม่มีการ merge หรือแก้ main และไม่อ้างว่าพร้อมใช้งานจริง จัด README และ model/ ให้ชี้ไปยัง artifact/threshold/metrics รุ่นเดียวกัน การทดลองอื่นยังตรวจย้อนหลังได้ใน [สารบัญ](../../../EXPERIMENTS.md) โดยไม่ลบหลักฐานผลที่ไม่ดีขึ้น

## ทำซ้ำ

```bash
git fetch origin main
OPENBLAS_NUM_THREADS=1 python -m round3.compare_main --source /path/to/nhefs_source --output round3_work/main_compare_new
```

สคริปต์ใช้ commit main ที่ pin ในโค้ด หาก git object ไม่มีให้ fetch commit นั้นก่อน [Verification](verification.json) · [ตรวจ metrics จาก predictions](independent_verification.json) · [Train/Tune/Calibration/Test diagnostics ของ main](main_split_metrics.csv) · [Paired Test scores](test_scores.csv)

Test ถูกเปิดดูหลายครั้งแล้ว เป็น exploratory internal evaluation เท่านั้น ไม่มี external/Thai validation; label คือ reported diagnosis ใน 8 ปี ไม่ใช่การวัดยืนยันโรค
