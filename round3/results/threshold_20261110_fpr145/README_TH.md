# งบคำเตือนผิด FPR 14.5% — 10 พฤศจิกายน 2026

ปรับตามคำขอผู้ใช้จากเพดาน FPR 20% เป็น **14.5% ของผู้ไม่เกิดเหตุการณ์**
14.5% เป็นงบ FPR ไม่ใช่ probability threshold
ใช้ development OOF เลือก threshold ที่จับเหตุการณ์มากที่สุดภายใต้งบ แล้วรายงาน Test เดิม

## ผลเลือกจาก development

- 2,311 คน / events 141 / non-events 2,170; เฉลี่ย OOF 3 repeats ต่อคน
- อนุญาต FP สูงสุด floor(0.145 × 2,170) = **314**
- ได้ threshold **0.0918681015 = 9.1868%**
- FP จริง 311/2,170 = **14.33%**; TP 42/141 = Recall **29.79%**
- กติกา tie: TP มากสุด, FP น้อยสุด, threshold สูงสุด คะแนนเสมอไม่แยกตาม ID
- บันทึก protocol/selection ก่อนเปิด Test predictions ใน runner ของรอบนี้

## ผล Test ชุดเดิม 409 คน / 25 เหตุการณ์

| ค่า | F2 เดิม | งบ 20% | งบ 14.5% ล่าสุด |
|---|---:|---:|---:|
| Probability threshold | 4.1655% | 8.4889% | 9.1868% |
| TP | 23 | 9 | 8 |
| FN | 2 | 16 | 17 |
| FP | 250 | 58 | 45 |
| TN | 134 | 326 | 339 |
| FPR = FP/384 | 65.10% | 15.10% | 11.72% |
| Recall | 92.00% | 36.00% | 32.00% |
| Precision | 8.42% | 13.43% | 15.09% |
| Specificity | 34.90% | 84.90% | 88.28% |
| จำนวนคำเตือน | 273 | 67 | 53 |
| Balanced accuracy | 0.63448 | 0.60448 | 0.60141 |

เทียบงบ 20% คำเตือนผิดลด 13 คน แต่จับเหตุการณ์น้อยลง 1 คน
ยังพลาด 17 จาก 25 เหตุการณ์ และ 45 จาก 53 คำเตือนไม่เกิดเหตุการณ์ตาม label
FPR หารด้วย non-events 384 คน จึงต่างจากสัดส่วนคำเตือนที่ผิด

FPR Test 95% Wilson CI **8.87–15.32%**; Recall CI **17.21–51.59%**
แม้ point estimate FPR Test ต่ำกว่า 14.5% ขอบบน CI ยังสูงกว่าเป้าหมาย
ข้อจำกัดเป็น empirical budget บน development ไม่รับประกัน Test/ประชากรใหม่

## โมเดลและการใช้งาน

Probability และพารามิเตอร์เท่าเดิมทั้งหมด ยกเว้น threshold และ threshold policy metadata
AUC **0.705417**, Brier **0.055795**, AP **0.131922** เท่าเดิม
ระดับ 1–3 ใช้เกณฑ์เดิมและมีความหมายต่างจาก binary experimental_alert
ไม่ได้เปลี่ยน features, calibration หรือฝึกใหม่เพื่อให้ผ่านผล Test

```bash
python -m round3.predict --model round3/results/threshold_20261110_fpr145/model.json \
  --input round3/results/threshold_20261110_fpr145/demo_input.json
```

ตัวอย่างสมมติใน demo_input ยังได้ score 0.0687572, ระดับ 2 และ experimental_alert=false

## หลักฐานและทำซ้ำ

- [metrics](metrics.json), [selection](selection.json), [protocol](protocol.json), [model](model.json)
- [development scores](development_scores.csv), [Test scores](test_scores.csv): ลำดับแถวแทน original IDs
- [verification](verification.json): ตรวจทุก candidate threshold ว่า TP สูงสุดที่ทำได้ภายใต้งบคือ 42,
  parameters เหมือนเดิม และคำนวณ confusion matrix แยกได้ตรงกัน
- [software tests](software_tests.txt): ทดสอบ threshold policy 6 ข้อผ่านในรอบนี้
  (ชุดเต็ม 25 ข้อผ่านในรอบงบ 20% ก่อนหน้า; รอบนี้โค้ดไม่เปลี่ยน)

```bash
python -m round3.threshold_policy --source round3_work/real_nhefs_20261105_run01 \
  --output round3_work/threshold_fpr145_new_attempt --max-fpr 0.145
```

เป็นการทดลองนโยบายตามผู้ใช้บน Test ที่เคยเปิดเผยแล้ว ไม่ใช่ test ใหม่หรือหลักฐานว่าโมเดลดีขึ้น
การลองงบหลายค่าตามผล Test เสี่ยงต่อการเลือกตามผลย้อนหลัง ต้องตรวจภายนอกก่อนใช้จริง
ข้อสรุป primary Brier จากการทดลองเดิมยังไม่ผ่านเหมือนเดิม และยังไม่ยืนยันกับคนไทย
