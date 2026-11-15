# สรุปการอัปเดตโมเดล LEPTOLIC — 15 พฤศจิกายน 2026

## สถานะปัจจุบัน

Branch `dev-by-gunner` ถูก merge เข้า `main` ผ่าน [PR #2](https://github.com/s4arry/Lebtolic/pull/2)
เรียบร้อยแล้ว โดยเพิ่ม workflow การทดลอง Round 3, model artifacts, demo, tests และเอกสารประกอบ

**รุ่นอ้างอิงสำหรับงานวิจัย:** Logistic Regression (L2), C=0.1, เพิ่ม baseline weight และ smoking,
ไม่ใช้ class balancing และไม่เพิ่ม calibration เพิ่มเติม รุ่นนี้เลือกจากผล Development ตาม protocol
ไม่ใช่การเลือกผู้ชนะย้อนหลังจาก Test

- Source candidate commit: `befec00faa9d341b983c984adc087ce0cf4585a8`
- Model artifact: [model/model.json](model/model.json)
- Manifest / provenance: [model/manifest.json](model/manifest.json)
- Feature schema: 6 ตัวแปรตั้งต้น → 9 encoded columns
- Primary policy: `fpr145`
- Threshold: `0.09803399686084495`, เลือกจาก development inner OOF
- Fit final model/preprocessing บน Development 2,311 คน

## ผลการประเมิน

| Metric | Development CV (mean 3 repeats) | Test เดิม |
|---|---:|---:|
| ROC-AUC | 0.7294 | 0.7034 |
| Recall ณ FPR-policy | 43.97% | 40.00% |
| Precision ณ FPR-policy | 17.11% | 16.95% |
| F2 ณ FPR-policy | 0.3346 | 0.3145 |
| FPR ที่พบจริง | 13.84% | 12.76% |
| Brier (ต่ำดีกว่า) | 0.055488 | 0.057010 |

Test มี 409 คน / 25 events: TP=10, FP=49, FN=15, TN=335

### เทียบกับ main เดิม

| Metric | main เดิม (5 features + Platt) | รุ่นอ้างอิง (weight + smoking) |
|---|---:|---:|
| Test ROC-AUC | 0.676146 | 0.703438 |
| Test Brier | 0.056529 | 0.057010 |

ROC-AUC มี point estimate สูงขึ้น แต่ Brier แย่ลงเล็กน้อย และ paired 95% intervals
ของ AUC/Brier คร่อมศูนย์ จึงยังยืนยัน superiority โดยรวมไม่ได้
ที่ threshold 0.5 ทั้งสองรุ่นไม่ flag ใคร การเปลี่ยน threshold/policy เปลี่ยน trade-off
ระหว่าง Recall กับ false positives จึงห้ามอธิบายการเพิ่ม Recall ว่าเกิดจาก model quality เพียงอย่างเดียว

## เหตุผลที่ไม่เลือกทางเลือกอื่น

- Class balancing และ oversampling ทำให้ probability calibration เสียหายมากใน workflow ที่ทดลอง
  (Development Brier ประมาณ 0.232 เทียบประมาณ 0.055–0.056 ใน unbalanced candidates)
- Calibration follow-up เลือกวิธี `none` จาก inner Brier แบบ one-SE ทั้ง 15 outer folds และ final development
- Nonlinear follow-up ไม่ผ่าน development improvement screening; tuned quadratic ไม่ได้ดีกว่า linear
- Smoking-only ให้ Test metrics บางตัวสูงกว่า candidate ในบาง policy แต่ไม่เปลี่ยนผู้ชนะย้อนหลังจาก Test

## ไฟล์สำคัญ

- [Model card / encoding / วิธี replay](model/README.md)
- [ผลเปรียบเทียบ main เดิม](round3/results/main_comparison_20261111/README_TH.md)
- [Balancing และ fallback](round3/results/balancing_20261111/README_TH.md)
- [Calibration follow-up](round3/results/calibration_20261111/README_TH.md)
- [Nonlinear follow-up](round3/results/nonlinear_20261111/README_TH.md)
- [สารบัญการทดลอง](EXPERIMENTS.md)

## ข้อจำกัด

ผลลัพธ์คือ reported diagnosis ภายใน 8 ปีใน selected historical NHEFS cohort ของสหรัฐฯ
ไม่ใช่การวัดยืนยันโรค ไม่มี external validation หรือ Thai validation, มีเพียง 25 events ใน Test
และ Test ถูกเปิดดูหลายครั้งแล้ว จึงเป็น exploratory internal evaluation เท่านั้น
ยังไม่เหมาะกับการคัดกรอง วินิจฉัย หรือใช้ตัดสินใจรักษารายบุคคล

เอกสารนี้อัปเดตให้ตรงกับ artifacts ที่ merge เข้า main แล้ว; ดูรายละเอียดตัวเลข,
protocol, fold variation และ uncertainty ในรายงาน Round 3 ที่ลิงก์ไว้ด้านบน
