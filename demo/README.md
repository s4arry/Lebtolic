# Demo form — รุ่นอ้างอิง befec00

ฟอร์มกรอกค่าแบบคนทั่วไปสำหรับ [model/model.json](../model/model.json)
ก่อนหน้านี้รุ่น weight + smoking มีแค่ replay API ที่ต้องป้อนเลข 9 ช่องเอง และฟอร์มใน notebook เดิมรองรับแค่ schema 5 features (มีชั่วโมงนอน)

```
คำตอบของผู้ใช้ → guardrails → 9 encoded columns → PortableModel → คะแนน + แจ้งเตือนเมื่อ ≥ threshold 9.80%
```

ไม่มีการ retrain หรือแก้โมเดล ให้คะแนนตรงกับ `model/example_output.json` (test `test_reproduces_committed_example_output`)

## วิธีเปิด

- **Colab:** เปิด [`LEPTOLIC_Demo.ipynb`](../LEPTOLIC_Demo.ipynb) → Run all (เซลล์แรก clone repo ให้เอง)
- **เครื่องตัวเอง:** `pip install -r requirements.txt` แล้วเปิด notebook จาก repo root

## Input → encoded column

| ช่องในฟอร์ม | แปลงเป็น |
|---|---|
| อายุ, ความดันบน/ล่าง | `age_years`, `baseline_sbp`, `baseline_dbp` ตรงตัว |
| เพศ | `sex_female` = 1 ถ้าหญิง |
| น้ำหนัก kg หรือ lb | `weight_lb` (kg × 2.2046); ไม่ทราบ → NaN + `weight_missing`=1 แล้ว model ใช้ median 147 lb |
| การสูบบุหรี่ | never → 0,0 · former → `former_smoker`=1 · current → `current_smoker`=1; `smoking_unknown` = 0 เสมอ |

## Guardrails (ระบบปฏิเสธ ไม่ทำนาย)

| เงื่อนไข | เหตุผล |
|---|---|
| เคยได้รับการวินิจฉัยแล้ว | cohort มีเฉพาะคนที่ยังไม่เคยวินิจฉัย |
| DBP ≥ SBP | น่าจะกรอกสลับ |
| SBP ≥ 130 หรือ DBP ≥ 80 | เกณฑ์ cohort |
| อายุนอก 32–85, SBP นอก 80–129, DBP นอก 44–79 | ช่วงของ development 2,311 คน จาก `round3/results/nhefs_20261105/model.json` (`training_range`) |
| น้ำหนักนอก 69–300 lb (≈31–136 kg) | ช่วงค่าที่ codebook ถือว่าเป็นน้ำหนักจริง (PMR5b1/b2) — เป็นช่วงตามนิยาม ไม่ใช่ min/max ที่สังเกตได้ |
| สูบบุหรี่ "ไม่ทราบ" | ไม่มีคนกลุ่มนี้ใน cohort ฝึก (coefficient = 0) |

## ตัวอย่างสำหรับ demo (ค่าสมมติ ไม่ใช่คนจริง)

| ระดับ | Input | คะแนน |
|---|---|---:|
| ต่ำ | ชาย 40 ปี · 100/65 · 60 kg · ไม่เคยสูบ | 0.80% |
| กลาง | หญิง 50 ปี · 118/74 · 70 kg · ไม่เคยสูบ | 6.93% |
| สูง | หญิง 55 ปี · 128/78 · 100 kg · ยังสูบ | 42.01% → แจ้งเตือน |

ส่วน "ทำไมได้คะแนนนี้" แสดง coefficient × ค่า standardized ของแต่ละตัวแปร (หน่วย log-odds เทียบคนเฉลี่ยในข้อมูล) ผลรวมบวก intercept = logit ของคะแนน (มี test ตรวจ)

## แก้ notebook

แก้ที่ `demo/build_notebook.py` แล้วรัน `python -m demo.build_notebook` อย่าแก้ .ipynb ด้วยมือ

## ขอบเขต

คะแนนคือโอกาส **รายงานการวินิจฉัยครั้งแรกภายใน 8 ปี** ใน NHEFS สหรัฐฯ 1982–1992 ไม่ใช่การวินิจฉัย ยังไม่ทดสอบกับคนไทย Test เดิมเคยเปิดดูหลายรอบ
