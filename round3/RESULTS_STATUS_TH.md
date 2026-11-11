# สถานะผลแนวทาง 2

## 11 พฤศจิกายน 2026: balancing และแผนสำรองเสร็จแล้ว

[อ่านรายงานล่าสุด](results/balancing_20261111/README_TH.md)
เครื่องใหม่ดาวน์โหลด CDC ได้ครบ 8 ไฟล์และ hashes ตรง ZIP เดิม; cohort/split เดิมครบ
ทดลอง none, balanced, oversampling บน 5×3 outer folds และเลือก threshold ใน inner 3-fold
ทั้ง F2/FPR 14.5%; balancing ไม่ผ่าน development criterion จึงรันแผนสำรองตามที่อนุมัติไว้
ตรวจ codebook แล้วเพิ่ม measured weight/smoking; เลือกรุ่นรวมจาก development ก่อน Test
Development ROC-AUC เฉลี่ย 0.6463 → 0.7294 และ budget recall 29.6% → 44.0%
แต่ Test เดิมไม่ยืนยันว่าดีกว่าทุกด้าน (Brier 0.05579 → 0.05701; paired CI คร่อมศูนย์)
ยังคงโมเดลหลักเดิม; รุ่นเพิ่มฟีเจอร์เป็น research candidate ที่ต้องประเมินบนข้อมูลใหม่
Software tests 35 ข้อผ่าน, refit baseline scores ตรงเดิม, export ทุก variant ตรง sklearn
ตรวจค่าทุก fold/repeat/Test ซ้ำจาก predictions และรันซ้ำได้ผล/decision ตรงกัน

## ประวัติผลหลังรับ ZIP — 5 พฤศจิกายน 2026

**รันข้อมูลจริงสำเร็จแล้ว แต่ยังไม่ผ่านเกณฑ์หลักที่จะพิสูจน์ว่า Brier ดีขึ้น**

อ่าน [รายงานผลจริงฉบับเต็ม](results/nhefs_20261105/README_TH.md) พร้อมกราฟ metrics โมเดล และหลักฐานการตรวจสอบ

| รายการ | Legacy ที่ฝึกซ้ำ | โมเดลใหม่ |
|---|---:|---:|
| Test คน / เหตุการณ์ | 409 / 25 | 409 / 25 |
| ROC-AUC | 0.676146 | 0.705417 |
| Average precision | 0.122941 | 0.131922 |
| Brier | 0.056529 | 0.055795 |
| Log loss | 0.222744 | 0.216680 |

Primary paired Brier difference = +0.000734; 95% CI [-0.000062, +0.001682] คร่อมศูนย์
จึงไม่สรุปว่าเหนือกว่าโดยรวม แม้ point estimates ดีขึ้นใน Test นี้
Outer CV AUC/AP ไม่ดีขึ้นโดยสม่ำเสมอ; Test เดิมเคยเปิดเผยและยังไม่มี external/Thai validation

## สิ่งที่ทำสำเร็จแล้ว

- รับ NHEFS_compressed.zip ครบ 8 ไฟล์ และ parse 4 waves: 12,220/3,608/9,998/9,281 แถว
- Cohort 2,720 คน/166 เหตุการณ์, development 2,311/141, Test 409/25
- Death guard คัดเพิ่ม 0 คน; Test IDs จึงตรงกับเดิม
- ทำซ้ำ Legacy AUC/Brier เดิมได้ถึง tolerance 1e-14
- Outer 5-fold × 3 repeats และ inner 3-fold รันครบตาม protocol เดิม
- เลือก Logistic 4 features, C=0.1, ไม่ calibration ด้วยข้อมูลพัฒนา
- ตรวจ JSON prediction ครบ 409 Test rows ตรง stored scores ถึง 1e-12
- Notebook Run all สำเร็จ ผล metrics ทุกค่าตรง CLI ทุกประการ
- Software tests 19 ข้อผ่าน แยกจากผลทดสอบโมเดลจริง

## Trade-off ที่ต้องเปิดเผย

ที่ threshold 0.5 ทั้ง legacy/new Recall ยังเป็น 0%
เมื่อใช้กติกา F2 จาก OOF: Legacy Recall 60%, FP 125, Precision 10.71%;
ใหม่ Recall 92%, FP 250, Precision 8.42% จึงยังไม่พร้อมอ้างว่าเป็นเครื่องมือคัดกรองใช้งานจริง
ระดับ 1/2/3 มีอัตราเหตุการณ์ Test 1.94%/5.22%/12.50% แต่เป็นระดับสัมพัทธ์ ไม่ใช่ clinical categories

## ประวัติ blocker และข้อจำกัดที่เหลือ

ก่อนรับ ZIP ดาวน์โหลด CDC ไม่ได้ด้วย proxy CONNECT 403
เก็บหลักฐานเดิมไว้ใน results/nested_cv_attempt_01 และไม่ลบหรือแก้สถานะล้มเหลวเก่า
รอบล่าสุดใช้ ZIP ที่ผู้ใช้ส่งจึงไม่ต้องเข้าถึง CDC จาก runtime

ตรวจ layouts และ labels ได้ แต่ยังไม่ได้ยืนยัน value codes/skip logic ทั้งหมดกับคู่มือทางการ
ที่มา CDC เป็นข้อมูลที่ผู้ส่งแจ้ง ยังไม่มี independently published checksum ที่ตรวจเทียบ
ข้อมูลเก่า reported diagnosis, selection/censoring และจำนวนเหตุการณ์น้อยยังเป็นข้อจำกัด
raw data และ original participant IDs เก็บเฉพาะ work directory ไม่ commit
