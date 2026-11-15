# ปรับ threshold เพื่อลดคำเตือนผิด — 10 พฤศจิกายน 2026

ผู้ใช้เลือกเกณฑ์ **false positive rate (FPR) ไม่เกิน 20% ของผู้ไม่เกิดเหตุการณ์**
เลือก threshold จาก development OOF เท่านั้น แล้วจึงรายงาน Test เดิม
ผล: **FP ลดจาก 250 เหลือ 58 แต่ Recall ลดจาก 92% เหลือ 36%**
จึงเป็นการเปลี่ยน trade-off การแจ้งเตือน ไม่ใช่การเพิ่มความสามารถของโมเดล

## วิธีเลือกก่อนดู Test ของรอบนี้

ใช้ผลจริงจาก [การทดลอง NHEFS เดิม](../nhefs_20261105/README_TH.md)
แต่ละ development person มี OOF scores 3 ค่า นำมาเฉลี่ยก่อนเลือก threshold
รวม 2,311 คน / 141 เหตุการณ์ / 2,170 คนที่ไม่เกิดเหตุการณ์
กำหนดจำนวน FP สูงสุด floor(0.2 × 2,170) = 434
เลือกจุดตัดที่ TP สูงสุดภายใต้งบนี้ หาก TP เท่ากันเลือก FP น้อยกว่า แล้วเลือก threshold สูงกว่า
คะแนนที่เสมอกันไม่ถูกแยกตาม ID

บันทึก [protocol](protocol.json) และ [จุดตัดที่เลือก](selection.json) ก่อนเปิด Test predictions ใน runner
ได้ threshold **0.0848886261 (8.4889%)** เทียบ F2 เดิม 0.0416553970
Development มี FP 425/2,170 = **19.59%**, TP 52/141 = Recall **36.88%**
การผ่านงบเป็น empirical constraint บน development ไม่รับประกัน Test หรือประชากรใหม่

## ผล Test จริงบนโมเดลและคะแนนชุดเดิม

| ค่า | F2 เดิม | งบ FPR 20% |
|---|---:|---:|
| Threshold | 0.041655 | 0.084889 |
| TP | 23 | 9 |
| FN | 2 | 16 |
| FP | 250 | 58 |
| TN | 134 | 326 |
| Recall | 92.00% | 36.00% |
| Precision | 8.42% | 13.43% |
| FPR = FP / 384 | 65.10% | 15.10% |
| Specificity | 34.90% | 84.90% |
| จำนวนคำเตือนทั้งหมด | 273 | 67 |
| Balanced accuracy | 0.63448 | 0.60448 |
| F1 | 0.15436 | 0.19565 |
| ROC-AUC | 0.705417 | 0.705417 |
| Brier | 0.055795 | 0.055795 |

FP ลด 192 คน (76.8%) แต่จับเหตุการณ์ได้น้อยลง 14 จากทั้งหมด 25 คน
ยังมี 58 จาก 67 คำเตือนที่ไม่เกิดเหตุการณ์ตาม label แม้ FPR ต่ำกว่า 20%
เพราะ FPR หารด้วยผู้ไม่เกิดเหตุการณ์ 384 คน ไม่ใช่จำนวนคำเตือน 67 คน
Recall 95% Wilson CI **20.25–55.48%**; FPR CI **11.87–19.03%**
CI นี้ใช้บรรยายความไม่แน่นอนของ Test เดิมที่มีเหตุการณ์น้อย ไม่รับประกันผลในอนาคต

## สิ่งที่เปลี่ยนในการใช้งาน

ใช้ [model.json](model.json) ในโฟลเดอร์นี้สำหรับนโยบายใหม่
ตรวจแล้วว่าทุก parameter เท่าเดิม ยกเว้น threshold และ metadata ของนโยบาย
Probability, calibration, features และจุดแบ่งระดับ 1–3 จึงเหมือนเดิม
CLI เพิ่ม `experimental_alert` เมื่อโมเดลมี threshold policy โดยแจ้งเตือนเมื่อ score >= threshold
ระดับ 1–3 กับ binary alert เป็นคนละเกณฑ์ จึงอาจมีระดับ 3 ที่ยังไม่ถึง threshold แจ้งเตือน

ตัวอย่างสมมติอายุ 50 ปี เพศชาย SBP 120 DBP 75 ยังได้ score **0.0687572** และระดับ **2**
แต่ `experimental_alert=false` ภายใต้ threshold ใหม่

```bash
python -m round3.predict --model round3/results/threshold_20261110/model.json \
  --input round3/results/threshold_20261110/demo_input.json
```

## หลักฐานและทำซ้ำ

- [metrics.json](metrics.json): metrics และ intervals เต็ม
- [development_scores.csv](development_scores.csv): หนึ่งแถวต่อคนหลังเฉลี่ย OOF ใช้ตรวจการเลือกจุดตัด
- [test_scores.csv](test_scores.csv): probability เดิม ใช้ตรวจ confusion matrix
- [verification.json](verification.json): ตรวจ parameter preservation และ metrics แยกจาก runner
- [software_tests.txt](software_tests.txt): 25 tests ผ่าน (19 เดิม + 6 ของ threshold policy)
- CSV ใช้ลำดับแถวแทน participant IDs; ไฟล์ต้นทางที่มี IDs อยู่ใน work directory เท่านั้น

```bash
python -m round3.threshold_policy --source round3_work/real_nhefs_20261105_run01 \
  --output round3_work/threshold_new_attempt --max-fpr 0.2
```

Runner ต้องใช้ผลการทดลองเดิมที่มี development_oof.csv, test_predictions.csv และ model.json
ใช้ output directory ใหม่เพื่อรักษาผลรอบก่อน
Tests ตรวจ tied scores, zero budget, invalid inputs, optimality ใน candidate thresholds,
OOF coverage และเปลี่ยน Test labels แล้วจุดตัดที่เลือกไม่เปลี่ยน

นี่เป็น policy update แบบ exploratory หลังเคยเห็น Test แล้ว ไม่ใช่การประเมินชุดทดสอบใหม่
ผล Brier improvement criterion จากรอบก่อนยังไม่ผ่านเหมือนเดิม
ยังไม่ยืนยันสำหรับคนไทยหรือการใช้งานคลินิก หากต้องการจับเหตุการณ์มากกว่านี้ต้องยอมรับ
คำเตือนผิดมากขึ้น หรือพัฒนาคุณภาพคะแนนด้วยข้อมูลและการประเมินใหม่
