# โค้ดและผลการทดลอง LEPTOLIC

**รุ่นอ้างอิงของ branch:** [weight+smoking จาก befec00](../model/README.md)
Logistic C=.1, ไม่ calibrate, primary threshold 0.09803399686084495
ผล comparison กับ main ยังไม่ยืนยันว่าเหนือกว่าทุกด้าน

## คำสั่งที่เกี่ยวกับรุ่นอ้างอิง

ใช้ Python 3.12 และ numerical environment ที่บันทึกไว้:

```bash
python -m pip install -r requirements.txt
python -m round3.calibration_study --source /path/to/nhefs_source --output round3_work/calibration_new
python -m round3.compare_main --source /path/to/nhefs_source --output round3_work/main_comparison_new
```

รันจาก repository root; `source/raw/` มี 8 ไฟล์ข้อมูล/labels เดิม หรือสคริปต์ดาวน์โหลดจาก CDC
ต้องตรวจ hashes ตรง historical archive จึงรันต่อ ใช้ output directory ใหม่เสมอ
`compare_main` ต้องมี git object ของ main commit ที่ pin ในสคริปต์ (`git fetch origin main`)

- [ผล calibration และข้อจำกัด](results/calibration_20261111/README_TH.md)
- [ผลเปรียบเทียบ main ที่รันจริง](results/main_comparison_20261111/README_TH.md)
- [วิธีโหลด model.json และ encoded inputs](../model/README.md)
- [สารบัญการทดลองอื่น](../EXPERIMENTS.md)

## Software checks และกราฟ

```bash
python -m unittest discover -s tests -v
# matplotlib เป็น dependency เพิ่มเฉพาะการวาดกราฟ
python -m round3.report_calibration --results round3/results/calibration_20261111
```

Tests เป็น software fixtures ไม่ใช่หลักฐานว่าโมเดลใช้กับคนไทยหรือผู้ป่วยได้
`requirements.txt` ภายในโฟลเดอร์ round3 เป็น environment ประวัติ; รุ่นอ้างอิงใช้
`requirements.txt` ที่ repository root ซึ่งอ้างถึง `requirements-balancing.txt`

`round3.predict` / notebook demo เดิมรองรับ schema รุ่นก่อน ห้ามส่ง current model 9 columns
ผ่าน legacy input form ให้ใช้ PortableModel ตามคู่มือรุ่นอ้างอิง
