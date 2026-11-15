# ไดอารี่การทดลอง LEPTOLIC — INT182

บันทึกจากไฟล์จริงและ Git history ตรวจวันที่ 15 พฤศจิกายน 2026 (เวลาไทย)
ผลวันที่ 5–14 พ.ย. ด้านล่างคือ **ผลที่บันทึกไว้เดิม** ไม่ใช่การฝึกใหม่ในวันที่จัดระเบียบ
แต่ละวันอาจมีหลายการทดลอง จึงแยกตามวิธีและ commit ไม่ตีความว่า commit date คือเวลาเริ่มฝึก

## รุ่นไหนเป็นปัจจุบัน

ตอนเริ่มตรวจ remote `dev-by-gunner` อยู่ที่ `ca61f16` ซึ่งใหม่กว่า `4f45d68` ในข้อความส่งต่อ
`model/model.json`, demo และสไลด์ใช้รุ่น **befec00: อายุ เพศ SBP DBP น้ำหนัก ประวัติสูบบุหรี่**
(6 ตัวแปร → 9 encoded columns), Logistic C=.1, ไม่ balancing/ไม่ calibrate, FPR-policy 14.5%
threshold **0.09803399686084495** ไม่ใช่ 0.145
ผล Test ที่บันทึก: AUC .703438, Brier .057010, TP10 FP49 FN15 TN335

รุ่น 4 ตัวแปร F2 ที่ข้อความส่งต่อระบุยังอยู่ครบที่
[results/nhefs_20261105/model.json](results/nhefs_20261105/model.json)
threshold **0.04165539699069321** รุ่นนี้กับรุ่น demo เป็นคนละ artifact/นโยบาย ห้ามผสมผลกัน
การจัดระเบียบนี้รักษารุ่น demo ปัจจุบันไว้ขณะรอผู้ใช้ยืนยันว่าจะย้อนกลับหรือไม่
ไม่ได้ตัดสินผู้ชนะจาก Test เพิ่มเติม

## 5 พฤศจิกายน — baseline, repeated nested CV และผลข้อมูลจริง

- วิธี: ใช้ NHEFS 4 waves, cohort 2720/events166; development 2311/141; Test 409/25
  nested CV outer 5 folds × 3 repeats, inner 3 folds; เลือก configuration ด้วย Brier
- เลือก Logistic C=.1 อายุ/เพศ/SBP/DBP ไม่เพิ่ม calibration; F2 threshold จากคะแนน
  development repeated OOF ที่เฉลี่ยต่อคน ได้ TP23 FN2 FP250 TN134, recall92%, precision8.42%
- Test AUC .705417 และ Brier .055795 เทียบ legacy 5 features+Platt .676146/.056529
  primary Brier เดิม−ใหม่ CI [-.000062,+.001682] คร่อมศูนย์ จึงไม่ยืนยัน superiority
- Notebook Run all ตรง CLI เป็นหลักฐานเดิมใน verification.json ไม่ใช่การรันใหม่วันนี้
- เก็บ notebook baseline เพราะ `workflow.load_original()` execute core cell จริง
  ส่วน Review notebook เป็น workflow เก่า ไม่ใช่ demo ปัจจุบัน
- หลักฐาน: [รายงาน](results/nhefs_20261105/README_TH.md), metrics/protocol/model/verification,
  test_pair_scores.csv, executed_notebook.ipynb ในโฟลเดอร์เดียวกัน
- Commit: [c3945b3](https://github.com/s4arry/Lebtolic/commit/c3945b33b3adf65af16f1bbe4b9229c69ba51a7c),
  [c9b9d10](https://github.com/s4arry/Lebtolic/commit/c9b9d10f71d4168ade3571dd3fd3daf028bd1eae)
- Attempt แรกที่ CDC ตอบ 403 ยังอยู่ `results/nested_cv_attempt_01/` ห้ามนับเป็นการฝึกสำเร็จ

## 10 พฤศจิกายน — ลด FPR ด้วย threshold โดยไม่เปลี่ยนโมเดล

| วิธี | Threshold | Test TP/FN/FP/TN | การตัดสินใจในรอบนั้น |
|---|---:|---|---|
| F2 เดิม | .041655397 | 23/2/250/134 | จับมาก แลกคำเตือนผิดมาก |
| งบ FPR20% | .0848886261 | 9/16/58/326 | ทดลองนโยบายตามคำขอ ไม่ได้เพิ่ม AUC/Brier |
| งบ FPR14.5% | .0918681015 | 8/17/45/339 | ปรับนโยบายในรอบนั้น; อย่าสับสนกับรุ่น befec00 |

เลือกจาก development OOF ของโมเดลเดิมเท่านั้น; coefficient และ probability คงเดิม
หลักฐาน: [20%](results/threshold_20261110/README_TH.md),
[14.5%](results/threshold_20261110_fpr145/README_TH.md), selection/protocol/verification ในแต่ละโฟลเดอร์
Commit [8cfa49f](https://github.com/s4arry/Lebtolic/commit/8cfa49fee25bf0568696ecc0469ff556a3abe560),
[24f7896](https://github.com/s4arry/Lebtolic/commit/24f7896fbd984adf7598aa267778bda438c39d48)

## 10 พฤศจิกายน — recall และหลายระดับการเตือน

ทดลอง linear/quadratic 4/5 features รวม 9 configurations; nested selection ด้วย partial AUC
ที่ FPR≤14.5%; ได้ linear 4 features C=1, Test TP8 FP42 (FN17 TN342)
เทียบ C=.1 TP8 FP45 ยังจับเหตุการณ์ไม่เพิ่ม และ outer AUC/Brier ไม่ดีขึ้น
จึงไม่แทนโมเดลหลัก ไม่เลือกตาม FP ที่ลดเพียงสามคนบน Test
มี threshold profiles 14.5/20/30/40% และ F2 เป็นการสำรวจ
[รายงาน](results/recall_20261110/README_TH.md), selection/protocol/verification/metrics
Commit [4f45d68](https://github.com/s4arry/Lebtolic/commit/4f45d68b7ffa6e665392412d3d0f0d2569f947b3)

## 11 พฤศจิกายน — balancing และแผนสำรอง: รันแล้วบน remote

ข้อความส่งต่อว่า “ยังไม่ได้รัน” ล้าสมัย: มีผลและโค้ดจริงใน
[balancing_20261111](results/balancing_20261111/README_TH.md)
เปรียบเทียบ none, class_weight=balanced, oversampling positives ใกล้50:50 เฉพาะ training
validation/test คง prevalence เดิม; threshold F2/FPR14.5% เลือกจาก inner OOF ทุก outer fold
ใช้ cohort/splits เดิมและตรวจ source hashes; ไม่ทำ resampling ก่อนแบ่ง fold
Balancing ไม่ผ่าน development screening จึงไม่เลือก; รัน missed-event/missingness audits
และตรวจ codebook ก่อนเพิ่ม measured weight/smoking เป็นแผนสำรอง
รุ่นรวม development mean AUC .72943/Brier .055488 เทียบ 4-feature .64632/.056408
แต่ Test Brier แย่ลง .055795→.057010 และ CI คร่อมศูนย์ จึงยังไม่พิสูจน์ว่าดีกว่าทุกด้าน
รุ่น fallback เป็น research candidate ณ commit นี้; ต่อมาถูกจัดเป็น reference ตามบันทึก 14 พ.ย.
เก็บ protocol, decisions ก่อน Test, fit_counts, OOF/outer membership, audits, provenance,
codebook, candidates และผล uncertainty เพื่อทำซ้ำและตรวจ leakage ได้
Commit [995ff23](https://github.com/s4arry/Lebtolic/commit/995ff2356eb81b0ce7f6a3fd4aaa5d8379f8878d)

## 11 พฤศจิกายน — calibration

บนรุ่น weight+smoking เปรียบเทียบ none/intercept/sigmoid; calibrator fit เฉพาะ training
nested selection ด้วย inner Brier+one-SE เลือก none ทุก 15 outer fits และ final
mean Brier none .055488, intercept .055491, sigmoid .055532 จึงไม่เพิ่ม calibration
Final threshold เปลี่ยนเป็น inner OOF: FPR14.5% .09803399686084495, Test TP10 FP49
ต่างจาก policy รอบ balancing ที่ใช้ repeated outer OOF; ห้ามนำ threshold ต่างรอบมาปะปน
[รายงาน](results/calibration_20261111/README_TH.md), decision_before_test/outer_choices,
final_inner_oof/candidate_models/verification
Commit [befec00](https://github.com/s4arry/Lebtolic/commit/befec00faa9d341b983c984adc087ce0cf4585a8)

## 11 พฤศจิกายน — nonlinear

เพิ่ม squares เฉพาะ standardized age/SBP/DBP/weight (9→13 encoded terms)
linear/quadratic × C=.01/.1/1; one-SE เลือก linear .01 final
quadratic mean outer-repeat AUC .72045/Brier .055803 เทียบ baseline .72943/.055488
ไม่ผ่าน improvement screening จึงไม่แทนรุ่นเดิม แม้ Test บาง metric ดูดีขึ้น
[รายงาน](results/nonlinear_20261111/README_TH.md), protocol/decisions/outer choices/verification
Commit [b974ac9](https://github.com/s4arry/Lebtolic/commit/b974ac98fe83e46b577e1549f07b0787f9327254)

## 14 พฤศจิกายน — จัด reference, เทียบ main และทำ demo/สไลด์

Git history และ README ระบุเลือก befec00 เป็น research reference แล้วคัดลอก parameters
ไป model/model.json พร้อม manifest/hash/metrics ไม่เลือก sigmoid หรือ nonlinear จาก Test
เทียบกับ executable notebook main ที่ pin 6ce4577: AUC .676146→.703438 แต่ Brier
.056529→.057010; CI ไม่ยืนยัน superiority; ที่ threshold .5 ทั้งสองไม่เตือนใคร
ดังนั้น recall ที่เพิ่มส่วนหนึ่งมาจากการเปลี่ยน policy ไม่ใช่คุณภาพโมเดลเพียงอย่างเดียว
หลักฐาน [main comparison](results/main_comparison_20261111/README_TH.md),
[model](../model/README.md), [presentation](../presentation/README.md), [demo](../demo/README.md)
Commit [f4ceef7](https://github.com/s4arry/Lebtolic/commit/f4ceef7),
[d4f5887](https://github.com/s4arry/Lebtolic/commit/d4f5887),
[d2d8578](https://github.com/s4arry/Lebtolic/commit/d2d8578), merge [ca61f16](https://github.com/s4arry/Lebtolic/commit/ca61f16)

## 15 พฤศจิกายน — ตรวจ repository และรวมบันทึกส่วนเกิน

ตรวจ checkout เดิมสะอาดก่อน fetch branch จริง ไม่มี AGENTS.md ใน workspace
อ่าน README/protocol/รายงาน/notebook outputs และสื่อประกอบ พบข้อมูลส่งต่อเก่ากว่า remote
ไม่ฝึกข้อมูลดิบใหม่ และไม่ถือ metrics ที่บันทึกไว้ว่าเป็นผลรันใหม่
เก็บโมเดลทั้งสองรุ่น, scripts ฝึก/ทำนาย/ทดลอง balancing, dependencies แต่ละ numerical environment,
notebooks, score CSV, protocol/selection/verification และเอกสารที่อ้างถึงกัน
ไฟล์เหล่านี้เป็นเส้นทางใช้งานหรือหลักฐานที่จำเป็น จึงไม่ลบเพื่อให้เหลือโมเดลเดียวจนทำงานใหม่สูญหาย

### บันทึกที่รวมเข้าไดอารี่ก่อนลบ

ลบเฉพาะ progress logs และ repeat-check summaries 5 ไฟล์ด้านล่าง
ตรวจ tracked text รวม notebook/code/docs แล้วไม่มีการอ่านหรืออ้างชื่อไฟล์เหล่านี้ก่อนเพิ่มไดอารี่
ตัว runner ยังรันได้โดยไม่ต้องมี log เดิม เนื้อหาทั้งหมดและ SHA-256 ย้ายมาด้านล่าง
ส่วน structured metrics/verification และผลไม่ดีขึ้นยังเก็บไว้ใน results ตามเดิม
ใช้ `git show <commit>:<path>` เรียกต้นฉบับกลับได้ ไม่ rewrite history

#### `round3/results/balancing_20261111/run_log.txt`

เหตุผลลบ: รวมบันทึกเดิมไว้ที่เดียว ไม่ใช่ input ของโค้ด
ต้นฉบับ: [995ff23](https://github.com/s4arry/Lebtolic/blob/995ff2356eb81b0ce7f6a3fd4aaa5d8379f8878d/round3/results/balancing_20261111/run_log.txt)

SHA-256: `b1219284d4c6b4e3607de380dd958e26f5e81a0da0991915b3ae2c196304c296`

```text
Evaluating none
Evaluating balanced
Evaluating oversample
Development balancing decision: none fallback: True
Evaluating fallback weight
Evaluating fallback smoking
Evaluating fallback weight_smoking
Bootstrapping fixed Test predictions
Completed: weight_smoking

```

#### `round3/results/calibration_20261111/run_log.txt`

เหตุผลลบ: รวมบันทึกเดิมไว้ที่เดียว ไม่ใช่ input ของโค้ด
ต้นฉบับ: [befec00](https://github.com/s4arry/Lebtolic/blob/befec00faa9d341b983c984adc087ce0cf4585a8/round3/results/calibration_20261111/run_log.txt)

SHA-256: `748144239db4c801548c07e44a079b1c7f04b227bff891d8c0cefbaac403e025`

```text
Outer 1 /15 selected none
Outer 2 /15 selected none
Outer 3 /15 selected none
Outer 4 /15 selected none
Outer 5 /15 selected none
Outer 6 /15 selected none
Outer 7 /15 selected none
Outer 8 /15 selected none
Outer 9 /15 selected none
Outer 10 /15 selected none
Outer 11 /15 selected none
Outer 12 /15 selected none
Outer 13 /15 selected none
Outer 14 /15 selected none
Outer 15 /15 selected none
Complete none {'final_calibrated': False, 'brier_lower_every_repeat': False, 'mean_budget_recall_not_lower': True, 'mean_budget_fpr_not_higher': True}

```

#### `round3/results/nonlinear_20261111/run_log.txt`

เหตุผลลบ: รวมบันทึกเดิมไว้ที่เดียว ไม่ใช่ input ของโค้ด
ต้นฉบับ: [b974ac9](https://github.com/s4arry/Lebtolic/blob/b974ac98fe83e46b577e1549f07b0787f9327254/round3/results/nonlinear_20261111/run_log.txt)

SHA-256: `a970770bfb7e76f12fb16dcd25c86f909d426d731da3107e43dda8bdc1e4ffa3`

```text
Outer 1 /15 linear:0.01
Outer 2 /15 linear:0.01
Outer 3 /15 linear:0.01
Outer 4 /15 linear:0.01
Outer 5 /15 linear:0.01
Outer 6 /15 linear:0.01
Outer 7 /15 linear:0.01
Outer 8 /15 linear:0.01
Outer 9 /15 linear:0.01
Outer 10 /15 linear:0.01
Outer 11 /15 linear:0.01
Outer 12 /15 linear:0.01
Outer 13 /15 linear:0.1
Outer 14 /15 linear:0.1
Outer 15 /15 linear:0.01
Complete linear:0.01 {'brier_lower_every_repeat': False, 'auc_higher_every_repeat': False, 'mean_f2_not_lower': True, 'mean_budget_recall_not_lower': True, 'mean_budget_fpr_not_higher': False}

```

#### `round3/results/balancing_20261111/repeat_run_check.json`

เหตุผลลบ: รวมบันทึกเดิมไว้ที่เดียว ไม่ใช่ input ของโค้ด
ต้นฉบับ: [995ff23](https://github.com/s4arry/Lebtolic/blob/995ff2356eb81b0ce7f6a3fd4aaa5d8379f8878d/round3/results/balancing_20261111/repeat_run_check.json)

SHA-256: `8c4c96233bb90276052f68ba00f3342e98f978b9c8950a8e44a58fa7616c3580`

```text
{
  "passed": true,
  "exact_file_matches": {
    "decision_before_test.json": true,
    "development_oof.csv": true,
    "test_scores.csv": true,
    "test_metrics.csv": true,
    "repeat_metrics.csv": true,
    "test_paired_difference_ci.csv": true
  },
  "note": "Initial completed run vs rerun after adding portable parameter export; scientific predictions/decisions unchanged"
}
```

#### `round3/results/calibration_20261111/repeat_run_check.json`

เหตุผลลบ: รวมบันทึกเดิมไว้ที่เดียว ไม่ใช่ input ของโค้ด
ต้นฉบับ: [befec00](https://github.com/s4arry/Lebtolic/blob/befec00faa9d341b983c984adc087ce0cf4585a8/round3/results/calibration_20261111/repeat_run_check.json)

SHA-256: `753d68487b103feef0ee9d0d375ca6c84593800058ec026fabcc5376ace2b890`

```text
{
  "decision_before_test.json": true,
  "development_oof.csv": true,
  "test_scores.csv": true,
  "repeat_metrics.csv": true
}
```

## ข้อจำกัดที่ต้องเก็บในรายงานเสมอ

- Test ถูกเปิดดูหลายรอบ ผลต่อเนื่องเป็น exploratory internal evaluation ไม่ใช่ untouched Test
- F2 ปี 5 พ.ย. เลือกจาก averaged repeated OOF ยังไม่ได้ nested threshold ครบทุก outer fold
  ผล nested thresholds วันที่ 11 พ.ย. เป็นอีก workflow ไม่ได้แก้ข้อจำกัดรุ่นเก่าย้อนหลัง
- เป้าหมายคือรายงานการวินิจฉัยครั้งแรกภายใน 8 ปี ไม่ใช่ measured BP ณ ปีที่8
  และ “ไม่เกิดเหตุการณ์ตามนิยาม” ไม่เท่ากับ “ไม่มีความดันสูงตลอดชีวิต”
- Historical selected US cohort, follow-up/selection bias และ events น้อย ยังไม่มี external/Thai validation
- ห้ามปรับจาก Test ซ้ำเพื่อบังคับให้ดีขึ้น; คงผลลบและความไม่แน่นอนตามจริง

## ผลตรวจรันใหม่ 15 พฤศจิกายน 2026

- Python3.12.14 ติดตั้ง numerical dependencies ตาม requirements.txt ที่ root
  NumPy2.3.5 / SciPy1.17.0 / scikit-learn1.8.0
- พบ test_demo_form ต้องใช้ pytest แต่ dependency เดิมไม่ได้ระบุ จึงเพิ่ม requirements-dev.txt
  คำสั่ง unittest อย่างเดียวไม่เก็บ pytest function tests ครบ; ใช้ `python -m pytest -q`
- Software tests **70 passed**; `pip check` ไม่พบ dependency ที่ขัดกัน
- คำนวณ AUC/Brier ใหม่จาก saved Test scores ของ12 variants และตรวจ confusion matrices
  ใน test_metrics.csv เทียบคะแนนและ threshold ที่บันทึกไว้ ได้ตรงกัน tolerance1e-12
  ไม่ได้ fit models ใหม่หรือดาวน์โหลดข้อมูลดิบในรอบนี้
- สร้าง PRESENTATION_TOPIC4_TH.md และ TOPIC4_MODEL_COMPARISON.csv แยก model comparison
  ออกจาก threshold policies และไม่เลือกโมเดลใหม่ตาม Test AUC สูงสุด
- ยังไม่เปลี่ยน parameters, default policy ของ demo, notebooks หรือสไลด์เดิม
