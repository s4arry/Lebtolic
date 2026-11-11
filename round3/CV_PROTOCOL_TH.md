# แนวทาง 2: protocol ก่อนรันข้อมูลจริง

**ส่วนต่อยอด 11 พ.ย.:** [Balancing protocol](BALANCING_PROTOCOL_TH.md) และ
[ผลจริง](results/balancing_20261111/README_TH.md) เพิ่มการเลือก **threshold** ภายในทุก
outer training fold สำหรับ F2/FPR14.5% โดย fix Logistic C=.1 และคง cohort/splits เดิม
เอกสารด้านล่างเป็น protocol ของรอบก่อน ซึ่งเลือก F2 หลังรวม outer OOF;
ไม่ควรใช้ผลรอบก่อนอ้างว่า threshold เคยผ่าน nested validation แล้ว

สถานะล่าสุด: รันกับ ZIP NHEFS สำเร็จแล้ว ดู [ผลจริงและข้อจำกัด](results/nhefs_20261105/README_TH.md)
เกณฑ์การเลือกโมเดลและตัดสินผลด้านล่างกำหนดก่อนเห็นผล และไม่เปลี่ยนหลัง primary CI คร่อมศูนย์
รับ ZIP ข้อมูลจาก Colab ได้แล้วโดยใช้ `--data-zip`; ดู [วิธีรัน](README.md)
และ [หลักฐานสถานะล่าสุด](RESULTS_STATUS_TH.md)
เอกสารนี้แทน protocol แบบ single Tune ในการทดลองใหม่ ไม่ได้แก้ผลบันทึกเดิมย้อนหลัง

## คำถามและเกณฑ์ตัดสิน

คงคำถาม reported first diagnosis within 8 years และไม่อ้างใช้กับคนไทยได้
Primary endpoint ของการเปรียบเทียบคือ Test Brier ของโมเดลเดิมลบโมเดลใหม่
ใช้ paired bootstrap 2,000 รอบ สุ่มคนชุดเดียวกันให้ทั้งสองโมเดล
มีหลักฐานภายใน Test ว่า Brier ลดลงเมื่อขอบล่างของช่วง percentile 95% มากกว่า 0
ถ้าช่วงคร่อมศูนย์ ให้สรุปว่ายังพิสูจน์การปรับปรุงตามเกณฑ์นี้ไม่ได้ แม้ point estimate ดีขึ้น
ไม่รับประกันผลบวก และไม่เปลี่ยน metric หลักหลังเห็นผล

Secondary: ROC-AUC, AP, log loss, calibration, recall, precision, specificity และ confusion matrix
รายงานทั้ง threshold 0.5 และกติกาเลือก threshold F2 จาก development OOF สำหรับทั้งโมเดลเดิม/ใหม่
ไม่อ้างว่าการเพิ่ม recall จาก threshold ที่ต่างกันคือการปรับปรุงความสามารถโมเดล
bootstrap เป็น uncertainty ของ fixed fitted models บน Test ไม่ครอบคลุม training/selection uncertainty
Test เดิมเคยเปิดเผยแล้ว ผลจึงเป็น internal evidence ไม่ใช่การพิสูจน์ทั่วไปหรือ external validation

## ข้อมูลและ target

- ใช้ official CDC URLs เดิม เก็บ raw hashes/URLs/bytes ใน data audit
- ID และ split membership สร้างจาก cohort เดิมก่อน additional exclusions แล้วรักษาสมาชิกเดิม
  ไม่เปลี่ยน seed เพื่อย้ายคนที่ผลไม่ดีออกจาก Test
- เพิ่ม conservative guard: label 0 ที่มี recorded death year ก่อนหรือทับปี horizon ถูกคัดออก
  ไม่สมมติวันตายภายในปี กฎนี้ยังต้องยืนยันความหมาย death/status กับ codebook ทางการ
- โมเดลเดิมและใหม่เปรียบเทียบบน guarded cohort เดียวกันเสมอ หากคนเปลี่ยนห้ามเทียบกับ
  AUC 0.676 เดิมโดยอ้างว่าเป็น algorithm improvement โดยตรง
- เก็บเพิ่มเติม exclusion list, join counts และ data-quality summary
- ถ้า baseline features ไม่ครบ หรือไม่มี sleep ใน development จนสร้าง legacy comparator ไม่ได้ ให้หยุด

## Nested repeated validation

1. รวม historical Train/Tune/Calibration เป็น development; historical Test ยังคงแยกไว้
2. Outer repeated stratified 5-fold × 3 repeats ใช้ประเมิน workflow ไม่ใช้เลือกผู้ชนะจาก outer score
3. ภายในแต่ละ outer training ใช้ inner stratified 3-fold เลือก mean Brier ต่ำสุด
4. Candidates 20 แบบ: 4/5 features × (Logistic C=.1/1/10 หรือ Tree depth=2/3, min leaf=50)
   × (ไม่ calibration หรือ sigmoid calibration)
5. Imputer/scaler fit ภายใน training pipeline ทุก fold; sigmoid calibration ใช้ 3-fold
   ภายใน inner training อีกชั้น ด้วย ensemble=False ไม่มีการใช้ inner validation มาฝึก calibrator
6. Feature ที่หายทั้ง training fold ทำให้ candidate/fold นั้นล้มเหลวอย่างเปิดเผย
   ไม่แอบตัด sleep แล้วรายงานว่าเป็นโมเดล 5 features
7. ประเมิน legacy algorithm บน outer validation เดียวกัน โดยแบ่ง outer training เป็น
   Train/Tune/Calibration ในสัดส่วน 55:15:15; ใช้ NumPy solver และ ridge grid เดิม
8. เมื่อครบ outer folds แต่ละคนได้ OOF predictions 3 ค่า เฉลี่ยเป็นคะแนนต่อคน
   เพื่อเลือก F2 threshold และจุดแบ่งระดับ ไม่ถือสาม repeats ว่าเป็นสามคนอิสระ
9. Mean/SD ของ outer folds ใช้แสดงความแกว่ง ไม่ใช้ naive CI ที่ถือ folds เป็นอิสระ
10. เลือก final configuration ด้วย inner CV บน development ทั้งหมด ฝึกแล้วจึงประเมิน Test

ข้อควรอ่าน: legacy workflow กันข้อมูลบางส่วนไว้ tuning/calibration ส่วนใหม่ใช้การ cross-fit
จึงเปรียบเทียบทั้ง workflow ไม่ใช่ผลของอัลกอริทึมเพียงอย่างเดียว
threshold/จุดแบ่งจากค่าเฉลี่ย repeated OOF อาจถ่ายโอนไป final fitted model ได้ไม่พอดี
ต้องรายงานผล Test และการกระจายคะแนนตามจริง ไม่ปรับจุดแบ่งซ้ำจาก Test

## ระดับและการทดลองใช้งาน

ใช้ terciles ของ development OOF scores และห้ามมี development band ว่าง
หากข้อมูลไม่รองรับ ให้คืน level=null พร้อมคำอธิบาย ไม่สร้างสามระดับปลอม
รายงาน n/events/observed rate/Wilson CI ต่อระดับ Test โดยไม่บังคับให้อัตราเรียงกัน
ถ้าระดับใดมีเหตุการณ์น้อย ยังไม่ถือว่าเป็นกลุ่มที่ผ่านการตรวจสอบแล้ว
ระดับเป็นคะแนนสัมพัทธ์ใน historical cohort ไม่ใช่ clinical cutoffs

ส่งออก portable JSON ที่เก็บ sigmoid parameters และ estimator parameters
`round3.predict` รองรับทั้ง legacy round3 JSON และ JSON ของ nested study
ตรวจ export/reload เทียบกับ scikit-learn ทุก 20 candidate types ด้วย software fixtures
fixtures ไม่ใช่ NHEFS และไม่ใช้รายงาน model performance

## คำสั่งและไฟล์ผลจริง

จาก checkout root และ Python environment ที่ติดตั้ง `round3/requirements.txt`:

```bash
python -m unittest discover -s tests -v
python -m round3.cv_study --work-dir round3_work/nested_run_01
python -m round3.cv_study --data-zip /path/to/NHEFS_data_and_layouts.zip --work-dir round3_work/nested_from_zip_01
python -m round3.predict --model round3_work/nested_run_01/model.json --input input.json
```

หากมี ZIP จาก Colab ใช้เส้นทางนี้แทนการดาวน์โหลด CDC:

```bash
python -m round3.cv_study --data-zip /path/NHEFS_data_and_layouts.zip --work-dir round3_work/nested_zip_01
```

ZIP ต้องมีไฟล์ข้อมูลและ layouts ครบ 8 ไฟล์ ยอมรับไฟล์อยู่ในโฟลเดอร์ย่อยได้
ห้ามมีชื่อไฟล์เป้าหมายซ้ำ ระบบตรวจรูปแบบ ASCII records, ID, layout fields และ HTML responses
ก่อนนำเข้าข้อมูล และเก็บ SHA-256/ขนาดไฟล์/จำนวนแถวใน input_archive_manifest.json
การตรวจรูปแบบไม่ใช่การยืนยัน CDC origin โดยอิสระ จึงเก็บ provenance note ไว้ด้วย
ได้รับ ZIP และมีผลจริงแล้วใน results/nhefs_20261105; หลักฐาน HTTP 403 เดิมเก็บไว้เป็นประวัติ

เก็บ protocol ก่อนดาวน์โหลดและเขียน `study_status.json` ทุกครั้ง
ถ้าล้มเหลวให้เลือก work directory ใหม่เพื่อรักษาหลักฐาน attempt เดิม
เมื่อสำเร็จจึงมี `metrics.json`, `report_TH.md`, `model.json`, `data_audit.json`,
`test_predictions.csv`, `development_oof.csv` โดย raw และ original target outputs อยู่ใน work directory
ไม่เก็บข้อมูลแต่งเป็นผลจริง ไม่รายงานคะแนนใหม่จาก saved outputs เดิม

การตรวจล่าสุด: ใช้ ZIP แทน CDC connection รันข้อมูลจริงครบ และ guard คัดเพิ่ม 0 คน
อ่านผลและ uncertainty แล้วพบว่า primary CI ยังคร่อมศูนย์ ดูรายงานผลจริง
สิ่งที่ยังเหลือคือยืนยัน codebook/skip logic ทั้งหมดและ external validation ไม่อ้างว่างานผ่านเพราะ CV อย่างเดียว
หากใช้ ZIP ที่มีข้อมูลครบ ไม่ต้องรอ network setting เพื่อฝึก แต่ยังต้องตรวจ codebook และ provenance
