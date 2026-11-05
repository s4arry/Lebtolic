# รายละเอียดการแก้ไข LEPTOLIC รอบ 3

อัปเดตวันที่ 5 พฤศจิกายน 2026 บน branch `dev-by-gunner`

## ผลงานที่ส่งขึ้น GitHub และสถานะ

พัฒนาโค้ดตามแนวทางที่ 2: **repeated nested cross-validation ภายในข้อมูลพัฒนา**
รักษาคำถาม NHEFS เดิม คือการรายงานว่าได้รับการวินิจฉัยความดันสูงครั้งแรกภายใน 8 ปี
ไม่เปลี่ยนเป็นการคัดกรองความดันปัจจุบัน และไม่อ้างว่าประเมินคนไทยได้แล้ว

**ส่วนโค้ดและการทดสอบซอฟต์แวร์ส่งขึ้น branch แล้ว แต่การทดลองข้อมูลจริงยังไม่เสร็จ**
ไม่มีคะแนนใหม่ที่ฝึกจาก NHEFS จริง และยังไม่มีหลักฐานว่าคะแนนโมเดลดีขึ้น
การรันข้อมูลจริงหยุดก่อนฝึกเพราะ CDC ตอบ proxy CONNECT 403
ขณะตรวจยังไม่มีไฟล์ `NHEFS_data_and_layouts.zip` ใน workspace/ไฟล์แนบ
ไฟล์ ZIP ที่มีอยู่เป็นสกิล ไม่ใช่ข้อมูล NHEFS

## แก้อะไร เพราะอะไร และอยู่ที่ไหน

| ส่วน | ก่อนแก้ | สิ่งที่เปลี่ยนและเหตุผล | ไฟล์ |
|---|---|---|---|
| วิธีเลือกโมเดล | ใช้ Tune split ครั้งเดียว | เพิ่ม outer stratified 5-fold × 3 repeats และ inner stratified 3-fold เพื่อประเมินความแกว่งของ workflow และลดการเลือกจาก split เดียว | [cv_study.py](cv_study.py) |
| ฟีเจอร์/โมเดลที่ทดลอง | Logistic 5 features เป็นหลัก | เปรียบเทียบ 4/5 features × Logistic C=.1/1/10 หรือ Tree depth=2/3 × ไม่ calibration/sigmoid รวม 20 configurations | [cv_study.py](cv_study.py) |
| ป้องกัน leakage | มี Train/Tune/Calibration แยกอยู่แล้ว | วาง imputer/scaler ใน pipeline ทุก fold; การเลือก configuration และ sigmoid calibration อยู่ภายใน development training folds ไม่ใช้ Test ตัดสินใจ | [cv_study.py](cv_study.py) |
| Sleep หายทั้งหมด | Imputer อาจตัดคอลัมน์ แต่รายงานยังเป็น 5 features | เพิ่ม FeatureSlice ตรวจ all-missing และให้ candidate/fold ล้มเหลวอย่างเปิดเผย ไม่รายงานว่าฝึกครบ 5 features ทั้งที่ถูกตัด | [cv_study.py](cv_study.py), [workflow.py](workflow.py) |
| การเทียบโมเดลเดิม | อาจนำ saved score มาเทียบกับ cohort ใหม่โดยตรง | ฝึก legacy NumPy algorithm ใหม่บน guarded cohort และ Test IDs เดียวกับโมเดลใหม่ พร้อมรักษา historical split membership หลังคัดคนเพิ่ม | [cv_study.py](cv_study.py) |
| Threshold | ใช้ 0.5 และพบ Recall 0% ใน saved output | เพิ่ม F2 threshold จาก development OOF ของทั้ง legacy/new พร้อมรายงานที่ 0.5 ด้วย ไม่ตีความการลด threshold ว่าเป็น algorithm improvement | [cv_study.py](cv_study.py), [workflow.py](workflow.py) |
| นิยามคำว่า “ดีขึ้น” | เสี่ยงสรุปจาก point estimate ค่าเดียว | กำหนด Brier เดิมลบใหม่เป็น primary comparison ใช้ paired bootstrap 2,000 รอบ; ถ้าขอบล่าง 95% CI ไม่เหนือ 0 ให้รายงานว่ายังพิสูจน์ไม่ได้ตามเกณฑ์นี้ | [CV_PROTOCOL_TH.md](CV_PROTOCOL_TH.md) |
| ระดับ 1–3 | จุดแบ่งต่างกันได้แต่ระดับหนึ่งอาจไม่มีคน | ตรวจ development bands ต้องไม่ว่าง หากไม่รองรับให้ level=null และคำอธิบาย ไม่สร้างสามระดับปลอม พร้อมรายงาน n/events/rate/Wilson CI ต่อระดับ | [workflow.py](workflow.py), [cv_study.py](cv_study.py) |
| Precision ไม่มีคำทำนายบวก | โค้ดเดิมใช้ค่า 0 แทนการหารศูนย์ | รายงาน null เมื่อ TP+FP=0 พร้อม confusion matrix แยกจาก Recall=0 ซึ่งคำนวณได้ | [workflow.py](workflow.py) |
| Target กับการเสียชีวิต | เส้นทาง label ลบอาจรับประวัติที่ขัดกัน | เพิ่ม conservative exclusion สำหรับ label 0 ที่ recorded death-year ก่อนหรือทับปี horizon; ยังต้องยืนยันรหัสกับ codebook ไม่ถือว่าแก้ target ครบแล้ว | [cv_study.py](cv_study.py) |
| การอ่าน core notebook | อ้างตำแหน่ง cell คงที่ | ค้นหา core cell จากฟังก์ชันที่ต้องใช้และตรวจว่าพบเพียงหนึ่ง cell; ยังเป็นการใช้ core ของ notebook เดิม | [workflow.py](workflow.py) |
| รับข้อมูลเมื่อ runtime เข้า CDC ไม่ได้ | ต้องดาวน์โหลดจาก runtime | รองรับ --data-zip สำหรับ ZIP จาก Colab ตรวจไฟล์ครบ 8 ไฟล์, ชื่อซ้ำ, ไฟล์ว่าง/HTML, layouts และ records ก่อนนำเข้า พร้อม hashes/provenance | [data_archive.py](data_archive.py), [cv_study.py](cv_study.py) |
| การส่งออกโมเดล | ใช้ได้เฉพาะรูปแบบเดิม | เพิ่ม portable JSON สำหรับ Logistic/Tree และ sigmoid parameters; ตรวจ prediction ให้ตรง scikit-learn ไม่ใช้ executable pickle | [predict.py](predict.py), [cv_study.py](cv_study.py) |
| หลักฐานและการตรวจย้อนหลัง | ผลกับที่มาข้อมูลอาจปะปน | เพิ่ม protocol/status ของ attempt, input manifest, split manifest, additional exclusions, source hashes และ paired predictions; ไม่เขียนว่ามีผลจริงเมื่อรันล้มเหลว | [cv_study.py](cv_study.py), [RESULTS_STATUS_TH.md](RESULTS_STATUS_TH.md) |
| Notebook สำหรับใช้งาน | การใช้งานรอบใหม่ยังเป็น CLI เป็นหลัก | Notebook เรียก nested study รับ ZIP ที่วางไว้ใน checkout และแสดง report/metrics/model หลังรันสำเร็จ | [LEPTOLIC_Round3_Review.ipynb](../LEPTOLIC_Round3_Review.ipynb) |

## ทดสอบอะไรแล้ว

รันบนโค้ด branch ปัจจุบันด้วย Python environment `/workspace/lebtolic-env`:

```bash
PYTHONDONTWRITEBYTECODE=1 /workspace/lebtolic-env/bin/python -m unittest discover -s tests -v
```

ผล: **19 tests ผ่าน** เป็นข้อมูลแต่งที่ระบุชัดว่าใช้ตรวจซอฟต์แวร์เท่านั้น
ไม่ใช้คะแนนจาก fixtures เป็นผลโมเดล NHEFS

| ชุดทดสอบ | จำนวน | สิ่งที่ครอบคลุม |
|---|---:|---|
| [test_round3.py](../tests/test_round3.py) | 8 | นิยาม precision, threshold F2, การแบ่งระดับ, Wilson CI, ไม่ใช้ held-out data เลือก candidates และ export/reload ของ workflow เดิม |
| [test_cv_study.py](../tests/test_cv_study.py) | 6 | ระดับว่าง, feature หายทั้งหมด, death guard, paired comparison ที่โมเดลเหมือนกัน, export prediction ครบ 20 configurations และเปลี่ยน Test labels แล้วการเลือกโมเดล/threshold/ระดับไม่เปลี่ยน |
| [test_data_archive.py](../tests/test_data_archive.py) | 5 | ZIP ที่อ่านได้พร้อม manifest, ไฟล์ขาด, HTML, ชื่อซ้ำ และ record width ผิด |

Notebook ผ่านการตรวจ schema/Python cells ระหว่างพัฒนา แต่ยังไม่ได้ยืนยัน Run all ครบด้วยข้อมูลจริง
tests ผ่านไม่ได้ยืนยันความถูกต้องของการตีความ codebook หรือประสิทธิภาพโมเดลจริง

## คะแนนเดิมที่ตรวจพบ — ไม่ใช่ผลหลังแก้

ข้อมูลนี้มาจาก saved outputs ของ notebook เดิม เก็บสำเนาพร้อม hash ใน
[saved_output_evidence.json](saved_output_evidence.json)

| รายการ | Saved output เดิม | ผลโมเดลใหม่จากข้อมูลจริง |
|---|---:|---|
| Cohort คน / เหตุการณ์ | 2,720 / 166 | ยังไม่มี |
| Test คน / เหตุการณ์ | 409 / 25 | ยังตรวจใหม่ไม่ได้ |
| ROC-AUC | 0.6761458333 | ยังไม่มี |
| Brier หลัง calibration | 0.0565291855 | ยังไม่มี |
| Brier ก่อน calibration | 0.0560729313 | ยังไม่มี |
| Constant-baseline Brier | 0.0573885312 | ยังไม่มี |
| Recall ที่ threshold 0.5 | 0/25 = 0% | ยังไม่มี |
| TN / FP / FN / TP ที่ 0.5 | 384 / 0 / 25 / 0 | ยังไม่มี |

ผลเดิมทำให้เห็นว่า threshold 0.5 ไม่จับเหตุการณ์ และ calibration ไม่ได้ทำให้ Brier ดีขึ้นใน Test ที่บันทึกไว้
จึงกำหนดการเลือก calibration จาก development และรายงาน threshold trade-offs อย่างเปิดเผย

## เหตุใดยังรันข้อมูลจริงไม่เสร็จ

คำสั่งข้อมูลจริงที่ทดลอง:

```bash
/workspace/lebtolic-env/bin/python -m round3.cv_study --work-dir round3/results/nested_cv_attempt_01
```

ล้มเหลวก่อนฝึกที่ดาวน์โหลด `N82int.txt` ด้วย `URLError: Tunnel connection failed: 403 Forbidden`
ดู [สถานะ attempt](results/nested_cv_attempt_01/study_status.json) และ
[protocol ที่บันทึกก่อนเริ่ม](results/nested_cv_attempt_01/protocol.json)
การตรวจ runtime ล่าสุดยังพบ custom allowlist ว่าง จึงไม่ถือว่า CDC network draft ถูกนำมาใช้แล้ว

## วิธีทำต่อเมื่อได้ข้อมูล

ZIP ต้องมี `N82int.txt`, `N86int.txt`, `N87int.txt`, `N92int.txt`
และ `intv82.inputs.labels.txt`, `intv86.inputs.labels.txt`, `intv87.inputs.labels.txt`, `intv92.inputs.labels.txt`

```bash
python -m round3.cv_study --data-zip /path/to/NHEFS_data_and_layouts.zip --work-dir round3_work/nested_from_zip_01
```

เมื่อรันสำเร็จจึงมี `metrics.json`, `report_TH.md`, `model.json`, `data_audit.json`,
`split_manifest.csv`, `additional_exclusions.csv`, `test_predictions.csv` และ `development_oof.csv`
ตรวจ `study_status.json` ก่อนอ่านผล และใช้ work directory ใหม่ในแต่ละ attempt

การตรวจรูปแบบ ZIP ไม่ยืนยัน CDC origin โดยอิสระ จึงเก็บที่มาที่ผู้ส่งแจ้งและ hashes แยกไว้
ต้องตรวจคู่มือและจำนวนคนที่ถูก guard คัดออกก่อนรับรอง target
Test เดิมเคยเปิดเผยแล้ว แม้ได้ CI สนับสนุนการลด Brier ก็ยังเป็น internal evidence
ไม่ใช่ external validation หรือหลักฐานว่านำไปประเมินความเสี่ยงคนไทยได้
