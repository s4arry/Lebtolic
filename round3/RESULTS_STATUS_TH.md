# สถานะผลแนวทาง 2

**ยังไม่มีผลทดสอบโมเดลใหม่จากข้อมูล NHEFS จริง จึงยังพิสูจน์ไม่ได้ว่าผลลัพธ์ดีขึ้น**

## สิ่งที่ทำและตรวจแล้ว

- พัฒนา repeated nested CV: outer 5-fold × 3 repeats, inner 3-fold
- เปรียบเทียบ 4/5 features, Logistic/Tree และมี/ไม่มี sigmoid calibration รวม 20 configurations
- กำหนด primary Brier comparison และ paired bootstrap CI ก่อนเห็นผลจริง
- ตรวจการใช้ development เท่านั้นในการเลือกโมเดล threshold และจุดแบ่งระดับ
- ส่งออก JSON และทดสอบผล prediction เทียบ scikit-learn ทุก candidate type
- เพิ่มรับ ZIP จาก Colab โดยตรวจ data/layout ครบ 8 ไฟล์และบันทึก hashes/provenance
- Tests ผ่าน 19 ข้อ และ pip check ผ่าน ทั้งหมดเป็นการตรวจซอฟต์แวร์ด้วยข้อมูลแต่ง
  ไม่ใช่คะแนนประสิทธิภาพโมเดล NHEFS

## สถานะข้อมูลจริง

คำสั่งที่ทดลองจริง:

```bash
/workspace/lebtolic-env/bin/python -m round3.cv_study --work-dir round3/results/nested_cv_attempt_01
```

ผล: exit 1 ก่อนฝึก ที่การดาวน์โหลด `N82int.txt` ด้วย
`URLError: Tunnel connection failed: 403 Forbidden`
ดูหลักฐาน `results/nested_cv_attempt_01/study_status.json` และ protocol ของ attempt นั้น

Network draft มี CDC แล้ว แต่ policy ที่ตรวจใน runtime ยังไม่มี host นี้ใน custom allowlist
ขณะตรวจยังไม่พบไฟล์ `NHEFS_data_and_layouts.zip` ที่แนบในแชท พบเพียง ZIP สกิล
โค้ดดาวน์โหลด Colab ที่ส่งในแชทช่วยสร้าง ZIP ได้ แต่ไม่ได้แนบเนื้อหาข้อมูลมาในตัวมันเอง

เมื่อได้รับ ZIP ให้ใช้คำสั่ง:

```bash
python -m round3.cv_study --data-zip /path/to/NHEFS_data_and_layouts.zip --work-dir round3_work/nested_from_zip_01
```

ไม่ต้องดาวน์โหลด CDC ซ้ำ แต่ต้องตรวจคู่มือรหัส/สถานะเพิ่มเติมก่อนรับรอง conservative death guard
รูปแบบไฟล์ที่ตรวจผ่านไม่ใช่การยืนยันแหล่งกำเนิดอย่างอิสระ จึงเก็บ provenance แยกอย่างชัดเจน

## ผลอ้างอิงเดิม แยกจากผลใหม่

| รายการ | saved output เดิม | ผลใหม่จากข้อมูลจริง |
|---|---:|---|
| ROC-AUC | 0.6761458333 | ยังไม่มี |
| Brier หลัง calibration | 0.0565291855 | ยังไม่มี |
| Recall ที่ threshold 0.5 | 0/25 = 0% | ยังไม่มี |
| Test คน / เหตุการณ์ | 409 / 25 | ยังตรวจซ้ำไม่ได้ |

โมเดลเดิมและใหม่ต้องประเมินบนคนชุดเดียวกัน หาก cohort เปลี่ยนจะไม่ใช้คะแนนจาก cohort เก่ามาอ้าง improvement
ถ้า CI ของผลต่าง Brier คร่อมศูนย์จะรายงานว่ายังไม่มีหลักฐานพอ ไม่ปรับ protocol เพื่อให้ได้คำตอบบวก
แม้ผลดีขึ้นก็เป็น internal evidence บน Test ที่เคยเปิดเผยแล้ว ไม่ยืนยันความเสี่ยงสำหรับคนไทย
