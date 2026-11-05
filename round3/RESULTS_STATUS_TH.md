# สถานะการทดลองแนวทาง 2 — 5 พฤศจิกายน 2026

**ยังไม่มีผลโมเดลใหม่จากข้อมูล NHEFS จริง และยังพิสูจน์ไม่ได้ว่าผลดีขึ้น**

## หลักฐานที่ตรวจแล้ว

- Software tests ผ่าน 19 ข้อ ด้วยข้อมูลแต่งที่ใช้ตรวจซอฟต์แวร์เท่านั้น
- pip check ผ่าน ไม่มี broken requirements
- ตรวจ nested CV ว่าเปลี่ยน Test labels แล้วไม่เปลี่ยนการเลือกโมเดล threshold หรือระดับ
- ตรวจ JSON prediction เทียบ scikit-learn ครบ 20 candidate configurations
- ตรวจ ZIP importer: ไฟล์ครบ/ซ้ำ/หาย/HTML/ความกว้างข้อมูล และ manifest ของไฟล์
- ไม่มี raw NHEFS หรือ ZIP ข้อมูลใน workspace ปัจจุบัน พบเพียง ZIP ของสกิล
- ตรวจ CDC ล่าสุดยังได้ proxy CONNECT 403; runtime custom allowlist ยังไม่มี wwwn.cdc.gov
- การทดลองข้อมูลจริงครั้งก่อนหยุดที่ N82int.txt ก่อนฝึก เก็บหลักฐานไว้ใน
  `results/nested_cv_attempt_01/study_status.json` และ `protocol.json`

## คำสั่งที่ทดสอบซอฟต์แวร์

```bash
PYTHONDONTWRITEBYTECODE=1 /workspace/lebtolic-env/bin/python -m unittest discover -s tests -v
/workspace/lebtolic-env/bin/python -m pip check
```

19 tests ที่ผ่านไม่ใช่หลักฐานว่าโมเดลแม่นขึ้นหรือ notebook รันกับข้อมูลจริงครบแล้ว
Notebook ใหม่ผ่านการตรวจ schema เท่านั้น ส่วนผลจากการรันจริงยังติดข้อมูล

## ผลอ้างอิงเดิม ไม่ใช่คะแนนจากการฝึกรอบใหม่

| Metric | saved output เดิม | ผลใหม่ |
|---|---:|---|
| ROC-AUC | 0.6761458333 | ยังไม่มี |
| Brier หลัง calibration | 0.0565291855 | ยังไม่มี |
| Brier ก่อน calibration | 0.0560729313 | ยังไม่มี |
| Recall ที่ threshold 0.5 | 0/25 = 0% | ยังไม่มี |
| Test คน / เหตุการณ์ | 409 / 25 | ยังตรวจใหม่ไม่ได้ |

เกณฑ์หลักที่กำหนดไว้ก่อนรันคือ Brier เดิมลบใหม่พร้อม paired bootstrap 95% CI
ถ้าขอบล่างไม่มากกว่า 0 จะสรุปว่ายังไม่มีหลักฐานพอว่าดีขึ้น
โมเดลเดิมและใหม่ต้องใช้ guarded cohort และสมาชิก Test ชุดเดียวกัน
หากคัดคนเพิ่ม ห้ามนำคะแนนจาก cohort เดิมมาอ้างเป็นผลปรับปรุงของอัลกอริทึมโดยตรง

## สิ่งที่ต้องใช้เพื่อทำต่อ

แนบ ZIP ที่ดาวน์โหลดจาก Colab ซึ่งมี N82int.txt, N86int.txt, N87int.txt, N92int.txt
และ intv82/intv86/intv87/intv92.inputs.labels.txt รวม 8 ไฟล์
จากนั้นใช้ `--data-zip` ตาม CV_PROTOCOL_TH.md ได้โดยไม่ต้องดาวน์โหลด CDC ผ่าน runtime นี้
ระบบจะตรวจรูปแบบและเก็บ hashes; รูปแบบที่ถูกต้องไม่ใช่การยืนยันที่มาโดยอิสระ

เมื่อรันได้ ต้องตรวจ codebook, additional exclusions และผลจริงก่อนรับรองข้อสรุป
กฎ death-year guard ยังเป็น conservative policy ที่ต้องยืนยันความหมายรหัส
Test เดิมเคยเปิดเผยแล้ว ผลยังเป็น internal validation และไม่ยืนยันกับคนไทย
