# 09 · LIMITATIONS → DEMO — รู้ขอบเขตก่อนทดลอง

**ข้อความบนสไลด์**

- ข้อมูล NHEFS เป็นประชากรสหรัฐฯ ในอดีต และเป็น reported diagnosis ไม่ใช่ผลวัดยืนยันโรค
- Baseline cohort จำกัดผู้รายงานว่าไม่เคยได้รับการวินิจฉัยและมี SBP <130 / DBP <80
- คัดผู้ที่ผลไม่แน่ใจหรือติดตามไม่ครบ จึงมีโอกาสเกิด selection bias
- Test เดิมมีเหตุการณ์ 25 คนและเคยเปิดดูหลายครั้ง; ไม่มี external หรือ Thai validation
- คะแนน/threshold เป็นงานวิจัย ไม่ใช่คำวินิจฉัยหรือคำแนะนำรักษา

**คำพูดประกอบก่อน demo:** ตัวอย่างเป็นข้อมูลสมมติ ผลเป็น experimental score ของโมเดลอ้างอิง ระบบ replay ต้องส่ง 9 encoded columns ตาม `model/README.md`; ฟอร์มใน notebook เดิมรองรับ schema เก่า และยังไม่ใช่ฟอร์มประเมินผู้ใช้ทั่วไป โปรดตรวจหน่วยน้ำหนักเป็น pounds และ encoding สูบบุหรี่

**Demo ที่รันได้:** ดู `model/example_encoded.json`, `model/example_output.json` และคำสั่งใน `model/README.md` ไม่ใช้ข้อมูลผู้ฟังจริง

**แหล่งข้อมูล:** `model/README.md`, `round3/results/main_comparison_20261111/README_TH.md`
