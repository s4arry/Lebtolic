# 06 · BALANCING 50/50 — จับเหตุการณ์เพิ่ม แลก false positives

**ข้อความบนสไลด์:** การทดลองก่อนหน้าใช้ Logistic 4 ฟีเจอร์ C=0.1, balance เฉพาะ training fold; validation/test สัดส่วนจริง

| วิธีฝึก | Train | Test threshold | TP / FP / FN / TN | Recall | Precision | AUC |
|---|---|---:|---|---:|---:|---:|
| ไม่ balance | สัดส่วนจริง | 0.5 | 0 / 0 / 25 / 384 | 0% | ไม่กำหนด | 0.7054 |
| Class weights | น้ำหนักรวมสองคลาสเท่ากัน | 0.5 | 16 / 136 / 9 / 248 | 64% | 10.53% | 0.7092 |
| Random oversampling | เพิ่ม class 1 จน 50/50 | 0.5 | 17 / 138 / 8 / 246 | 68% | 10.97% | 0.7151 |

**คำพูดประกอบ:** ผล 0.5 คำนวณจาก predicted scores จริงที่บันทึกไว้ AUC เปลี่ยนเล็กน้อย ไม่ควรกล่าวว่า “เท่าเดิม” แบบเป๊ะ คะแนนหลัง balancing ยังไม่ได้แก้ calibration: development Brier ประมาณ 0.232 เทียบไม่ balance 0.0564 ผลนี้เป็นการทดลอง 4-feature ก่อนหน้า ไม่ใช่รุ่น 6-feature ที่เลือก และ random oversampling เป็นการสุ่มเพิ่ม ไม่ใช่ undersampling 91:91 และ fit โมเดลสุดท้ายบน development 2,311 คน ไม่ใช่ Train 1,495 คนตามแผนที่เคยเสนอ ซึ่งยังต้องทดลองแยก

**ภาพ:** [balancing_at_05.png](balancing_at_05.png) · [balancing_at_05.csv](balancing_at_05.csv)

**แหล่งข้อมูล:** `round3/results/balancing_20261111/test_scores.csv`, `test_metrics.csv`, `repeat_summary.csv`; Test เดิมเคยเปิดดูแล้ว
