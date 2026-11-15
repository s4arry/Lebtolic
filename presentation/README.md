# LEPTOLIC: เนื้อหาสำหรับสไลด์นำเสนอ

โฟลเดอร์ละหนึ่งหัวข้อ มี `content.md` สำหรับข้อความบนสไลด์ คำพูด และแหล่งข้อมูล
ไฟล์ PNG และ CSV ที่เกี่ยวข้องอยู่ในโฟลเดอร์ของหน้านั้น ชื่อสมาชิกเป็นช่องว่างให้ทีมกรอก

| หน้า | โฟลเดอร์ |
|---:|---|
| 1 | [01_cover](01_cover/content.md) |
| 2 | [02_team](02_team/content.md) |
| 3 | [03_pipeline](03_pipeline/content.md) |
| 4 | [04_features](04_features/content.md) |
| 5 | [05_dataset](05_dataset/content.md) |
| 6 | [06_sample_per_class](06_sample_per_class/content.md) |
| 7 | [07_model](07_model/content.md) |
| 8 | [08_model_comparison](08_model_comparison/content.md) |
| 9 | [09_balancing](09_balancing/content.md) |
| 10 | [10_test_performance](10_test_performance/content.md) |
| 11 | [11_next_steps](11_next_steps/content.md) |
| 12 | [12_limitations_demo](12_limitations_demo/content.md) |

## เครื่องมือสร้างกราฟ

กราฟสร้างด้วย Python, pandas, scikit-learn และ Matplotlib จากไฟล์ผลจริงใน repository
**ไม่ได้สร้างด้วย Power BI** เพราะ environment นี้ไม่มี Power BI เชื่อมต่อ
CSV ที่แนบแต่ละหน้าเป็น chart source สำหรับนำเข้า Power BI ต่อได้
ภาพเป็น PNG ความละเอียด 200 dpi; ตารางตัวอย่างใช้รหัสสมมติ ไม่เผย participant ID ต้นฉบับ

## ความตรงของเนื้อหา

รุ่นหลักอ้างอิง `befec00` ตาม `model/model.json`: ไม่มี Platt เพิ่มและไม่ได้ balance
หน้าที่อภิปราย balancing เป็นผล 4-feature experiment ก่อนหน้า; ไม่มีการอ้างว่าทดลอง undersampling 91:91 แล้ว
Test เคยเปิดดูหลายรอบ จึงเรียก exploratory internal evaluation ไม่ใช่การยืนยันด้วยข้อมูลใหม่
ชื่อสมาชิกยังไม่ได้รับข้อมูล จึงเว้นช่องให้ทีมกรอก

[สรุปโมเดลและข้อแลกเปลี่ยน](MODEL_SUMMARY_TH.md) · [คู่มือนำ CSV เข้า Power BI](POWER_BI_GUIDE.md)
