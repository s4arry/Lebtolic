# LEPTOLIC — Code Walkthrough

> เอกสารประวัติ: อธิบาย notebook baseline เดิม 5 features ไม่ใช่รุ่นอ้างอิงปัจจุบัน ดู [model/README.md](model/README.md)

คู่มืออ่านโค้ดใน `LEPTOLIC_NHEFS_Future_Prediction.ipynb` แบบไล่จากต้นจนจบ  
เป้าหมายคือให้เข้าใจว่าแต่ละส่วนรับข้อมูลอะไร ทำอะไร ส่งผลลัพธ์อะไร และเกี่ยวข้องกับการสร้างโมเดลตรงไหน

> **ขอบเขตสำคัญ:** LEPTOLIC เป็น prototype สำหรับทำนายว่า ผู้เข้าร่วม NHEFS ในอดีตจะมี **รายงานการได้รับการวินิจฉัยความดันโลหิตสูงครั้งแรกภายใน 8 ปี** หรือไม่ ไม่ใช่เครื่องมือวินิจฉัยโรค ไม่ใช่การทำนายการเกิดโรคจริงในระดับบุคคล และยังไม่ได้รับการยืนยันกับประชากรไทยยุคปัจจุบัน

---

## 0. ภาพรวมการทำงาน

โค้ดแบ่งเป็น 5 ช่วงใหญ่:

1. **เตรียมเครื่องมือและกติกาข้อมูล** — import library, กำหนดปีติดตาม, feature และชื่อไฟล์
2. **ดาวน์โหลดและอ่านข้อมูล** — โหลดข้อมูล NHEFS 4 ช่วงเวลา แล้วอ่านไฟล์ fixed-width ตาม layout ของ CDC
3. **สร้าง cohort และ target** — จับคู่ผู้เข้าร่วมคนเดิม คัดกลุ่มตามเกณฑ์ และสร้างคำตอบ 0/1 จากประวัติการติดตาม
4. **ฝึกและประเมินโมเดล** — แบ่งข้อมูล, preprocess, train Logistic Regression, เลือก regularization, calibrate probability และประเมินบน Test
5. **ทดลองทำนายและส่งออก** — รับ input, ตรวจสอบข้อมูล, คำนวณคะแนน และบันทึกไฟล์ผลลัพธ์

ภาพจำที่ควรมี:

```text
CDC raw files
    |
    v
Download + parse fixed-width records
    |
    v
Join the same participant across 4 waves
    |
    v
Eligibility checks + build 8-year target
    |
    v
Feature matrix X + target y
    |
    v
Train / Tune / Calibration / Test
    |
    v
Preprocess -> Logistic Regression -> Platt calibration
    |
    v
Evaluate on untouched Test set
    |
    v
predict_future() + export ZIP
```

---

## 1. ส่วนเริ่มต้น: imports และค่ากำหนดหลัก

โค้ดหลักทั้งหมดอยู่ใน code cell แรกของ Notebook

### 1.1 Imports

| Library / module | ใช้ทำอะไร |
|---|---|
| `pathlib.Path` | จัดการ path และ folder เช่น raw, outputs |
| `urllib.request` | ดาวน์โหลดไฟล์จาก CDC |
| `datetime.date` | ตรวจสอบวันที่สัมภาษณ์และคำนวณวันครบ 8 ปี |
| `collections.Counter` | นับจำนวน target และเหตุผลที่คัดคนออก |
| `csv` | อ่าน/เขียน CSV |
| `hashlib` | คำนวณ SHA-256 เพื่อตรวจสอบไฟล์ต้นทาง |
| `json` | บันทึก model bundle, audit และ metrics |
| `math` | ตรวจสอบตัวเลข เช่น NaN และ infinity |
| `re` | อ่านรูปแบบ layout และตรวจสอบ participant ID |
| `zipfile` | รวมไฟล์ผลลัพธ์เป็น ZIP |
| `numpy` | คำนวณ matrix, Logistic Regression และ metrics |

จุดที่ควรรู้: โปรเจกต์นี้เขียน Logistic Regression เองด้วย NumPy ไม่ได้เรียก scikit-learn เพื่อฝึกโมเดล

### 1.2 Constants

- `BASE` — URL ต้นทางข้อมูล NHEFS ของ CDC
- `HORIZON_YEARS = 8` — ช่วงเวลาที่ต้องการทำนาย
- `SEED = 42` — seed สำหรับ random number generator เพื่อให้การสุ่มแบ่งชุดข้อมูลทำซ้ำได้
- `FEATURES` — รายชื่อ input 5 ตัวที่โมเดลจะใช้

```python
FEATURES = [
    'age_years',
    'sex_female',
    'baseline_sbp',
    'baseline_dbp',
    'sleep_hours'
]
```

**Feature** คือข้อมูลที่ป้อนให้โมเดลเพื่อให้มันเรียนรู้ความสัมพันธ์กับผลลัพธ์ เช่น อายุหรือค่าความดันตั้งต้น ส่วน **Target** คือคำตอบที่เราต้องการให้โมเดลเรียนรู้ ในโปรเจกต์นี้ target คือ 0 หรือ 1

### 1.3 WAVES

`WAVES` เป็น dictionary ที่บอกว่าแต่ละ wave ใช้ไฟล์อะไร, layout อะไร, คอลัมน์ใดเป็น participant ID และต้องอ่าน field ไหนบ้าง

| Key | ช่วงเวลา | ID field |
|---|---|---|
| `82` | Baseline 1982–1984 | `SEQNUM1` |
| `86` | Follow-up 1986 | `SEQNO` |
| `87` | Follow-up 1987 | `SEQNO` |
| `92` | Follow-up 1992 | `HANESEQ` |

ชื่อ ID ไม่เหมือนกันทุก wave จึงต้องกำหนดไว้แยกกันก่อนนำข้อมูลมา join

---

## 2. Helper functions สำหรับทำความสะอาดข้อมูล

### 2.1 `integer(value)`

พยายามแปลงค่าเป็น integer ถ้าแปลงไม่ได้จะคืน `None`

ตัวอย่าง:
- `"42"` → `42`
- `""` → `None`
- `"unknown"` → `None`

ใช้กับข้อมูลที่ควรเป็นจำนวนเต็ม เช่น ปีหรือรหัสคำตอบ

### 2.2 `valid_number(value, lo, hi)`

แปลงค่าเป็น float และตรวจว่า:
- เป็นตัวเลขจริง
- ไม่ใช่ NaN หรือ infinity
- อยู่ในช่วง `lo` ถึง `hi`

ถ้าไม่ผ่านจะคืน `np.nan`

ตัวอย่าง: `valid_number("120", 60, 300)` ได้ 120 แต่ `valid_number("999", 60, 300)` ได้ NaN

ทำไมไม่ใช้ 97, 98 หรือ 99 เป็นตัวเลขตามปกติ? แบบสอบถามเก่ามักใช้เลขพิเศษแทน refused, don't know หรือคำตอบที่ไม่มีข้อมูล ตัวเลขเหล่านี้ไม่ใช่ค่าทางกายภาพจริง

### 2.3 `interview_date(...)`

รับ row และชื่อ field ของปี/เดือน/วัน แล้วสร้าง `datetime.date`

- ตรวจว่าปีอยู่ในช่วงที่ยอมรับ
- แปลงปี 2 หลักเป็นปี 1900+
- ตรวจวันที่จริง เช่น 31 กุมภาพันธ์ต้องไม่ผ่าน
- ถ้าวันที่ผิดหรือข้อมูลไม่ครบจะคืน `None`

ฟังก์ชันนี้สำคัญต่อการสร้าง target เพราะเราต้องรู้ว่า interview เกิดก่อนหรือหลังวันครบ 8 ปี

### 2.4 `anniversary(start, years)`

คำนวณวันครบรอบ เช่น วันเริ่มต้น + 8 ปี กรณีวันที่ 29 กุมภาพันธ์ที่ไม่มีในปีปลายทางจะปรับเป็นวันที่ 28 กุมภาพันธ์

### 2.5 `bp_component(...)`

สร้างค่าความดัน systolic หรือ diastolic:
1. ลองใช้ค่าที่สรุปไว้ใน field ค่าเฉลี่ยก่อน
2. ถ้าไม่มีค่าที่ใช้ได้ ให้ลองใช้การวัดครั้งที่ 2 และ 3
3. ใช้ค่าเฉลี่ยของครั้งที่ 2 และ 3 เฉพาะเมื่อทั้งสองค่าถูกต้อง

### 2.6 `baseline_features(row)`

แปลงข้อมูลดิบของ baseline ให้เป็น feature ที่โมเดลใช้:

| Feature | แหล่งข้อมูล | ความหมาย |
|---|---|---|
| `age_years` | `SUBAGE` | อายุเป็นปี |
| `sex_female` | `SUBSEX` | 0 = ชาย, 1 = หญิง ตาม mapping ในโค้ด |
| `baseline_sbp` | `PMR4cAVS` หรือค่าเฉลี่ยครั้ง 2/3 | ความดันตัวบน หน่วย mmHg |
| `baseline_dbp` | `PMR4cAVD` หรือค่าเฉลี่ยครั้ง 2/3 | ความดันตัวล่าง หน่วย mmHg |
| `sleep_hours` | `OH7` | ชั่วโมงนอน |

ฟังก์ชันนี้เป็นจุดแปลงจากชื่อ field ของ CDC ให้เป็นชื่อ feature ที่อ่านง่ายและใช้ในขั้นตอนฝึกโมเดล

ข้อสังเกต: `sleep_hours` อาจเป็น NaN ได้ เพราะเป็นข้อมูล optional ส่วน feature 4 ตัวแรกต้องมีค่าที่ใช้ได้เพื่อเข้า cohort

---

## 3. ส่วนสำคัญที่สุดของการเตรียมข้อมูล: `build_label()`

ฟังก์ชันนี้รับข้อมูล baseline และ follow-up ของผู้เข้าร่วมหนึ่งคน แล้วคืนค่า 3 อย่าง:

```python
y, reason, details = build_label(baseline, followups)
```

- `y` — target: 1, 0 หรือ `None`
- `reason` — เหตุผลว่าทำไมจึงได้ label หรือถูกคัดออก
- `details` — วันที่ baseline, วันครบ 8 ปี, ปีที่รายงานการวินิจฉัย ฯลฯ

### 3.1 ใครมีสิทธิ์เข้า cohort?

โค้ดตรวจว่า:
1. วันที่ baseline ถูกต้อง
2. ตอบ `D2 == 2` ซึ่งหมายถึงไม่เคยได้รับการวินิจฉัยตามกติกาของแบบสำรวจ
3. มีอายุ เพศ และค่าความดัน baseline ที่ใช้ได้
4. SBP ต้องต่ำกว่า 130 และ DBP ต้องต่ำกว่า 80 ตามเกณฑ์ของ prototype

นี่คือ **เกณฑ์คัดกลุ่มตัวอย่างของโปรเจกต์** ไม่ใช่ข้อสรุปทางการแพทย์ว่าบุคคลนั้นไม่มีโรคทุกกรณี

### 3.2 สร้างช่วงเวลา 8 ปี

ใช้วันที่สัมภาษณ์ baseline เป็นจุดเริ่มต้น แล้วคำนวณวันครบ 8 ปี เช่น:

```text
Baseline interview: 1983-06-15
8-year horizon:     1991-06-15
```

จากนั้นตรวจข้อมูล follow-up ว่ามีรายงานการวินิจฉัยครั้งแรกหรือไม่ และเมื่อใด

### 3.3 Target 1 หมายถึงอะไร?

`target = 1` เมื่อมีปีที่รายงานการวินิจฉัย และช่วงวันที่ทั้งปีอยู่ก่อนหรือเท่ากับวันครบ 8 ปี

เนื่องจากข้อมูลมีเพียง “ปี” ไม่ใช่วัน/เดือนที่วินิจฉัย โค้ดจึงมองปีนั้นเป็นช่วงตั้งแต่ 1 มกราคมถึง 31 ธันวาคม ไม่สร้างวันวินิจฉัยปลอมขึ้นมา

### 3.4 Target 0 หมายถึงอะไร?

มีได้สองกรณีหลัก:
- มีปีวินิจฉัยที่อยู่หลังช่วง 8 ปีอย่างชัดเจน
- ไม่มีรายงานการวินิจฉัย และมี follow-up สุดท้ายที่เชื่อถือได้ ครบอย่างน้อย 8 ปี พร้อมคำตอบ explicit no

### 3.5 ทำไมบางคนได้ `None`?

`None` หมายถึง **ยังระบุคำตอบไม่ได้อย่างน่าเชื่อถือ** ไม่ใช่ 0

ตัวอย่างเหตุผล:
- baseline date ไม่ถูกต้อง
- เคยได้รับการวินิจฉัยหรือไม่ทราบสถานะตั้งแต่ baseline
- follow-up ไม่ครบ 8 ปี
- คำตอบ unknown, skipped, refused
- รายงานว่ามีโรคแต่ไม่มีปีที่เชื่อถือได้
- ปีวินิจฉัยคร่อมวันครบ 8 ปี
- ข้อมูลขัดแย้งกัน เช่น รายงานว่าเคยเป็น แต่ภายหลังตอบ no
- มีข้อมูลการเสียชีวิตหรือวันสัมภาษณ์ที่ขัดกับลำดับเหตุการณ์

**ห้ามแปลง `None` เป็น 0 อัตโนมัติ** เพราะจะทำให้ข้อมูลที่ไม่รู้ถูกตีความว่าไม่เกิดเหตุการณ์จริง

---

## 4. อ่านไฟล์ข้อมูล NHEFS

### 4.1 `parse_layout(text)`

CDC ให้ไฟล์ layout ที่บอกตำแหน่งและความกว้างของแต่ละ field ในไฟล์ raw แบบ fixed-width

ตัวอย่างแนวคิด:

```text
ตำแหน่ง 1-5   = participant ID
ตำแหน่ง 6-7   = field ถัดไป
...
```

ฟังก์ชันใช้ regular expression อ่านตำแหน่งเริ่มต้นและความกว้าง แล้วเก็บเป็น dictionary

### 4.2 `read_wave(path, layout, fields, idfield)`

อ่านแต่ละบรรทัดของ fixed-width file โดยใช้ตำแหน่งจาก layout:
- ตรวจว่า field ที่ต้องใช้มีอยู่ใน layout
- ตัด substring ตามตำแหน่ง
- ตรวจความยาว record
- ตรวจว่า ID มีรูปแบบ 5 หลักและไม่ซ้ำ
- เก็บแต่ละคนใน dictionary โดยใช้ ID เป็น key

การใช้ ID เป็น key ทำให้ค้นหาคนเดิมข้าม wave ได้

### 4.3 `download(raw, name)`

ดาวน์โหลดไฟล์จาก CDC:
- ถ้าไฟล์มีอยู่แล้ว ไม่ดาวน์โหลดซ้ำ
- ดาวน์โหลดลงไฟล์ชั่วคราว `.part`
- ตรวจขนาดและป้องกันกรณี server ส่ง HTML error page มาแทนข้อมูล
- สำเร็จแล้วจึงเปลี่ยนชื่อเป็นไฟล์จริง
- หากล้มเหลวจะลบไฟล์ชั่วคราวและแจ้ง error

จุดประสงค์คือไม่ให้ไฟล์ดาวน์โหลดไม่สมบูรณ์ถูกนำไปอ่านเหมือนเป็นข้อมูลจริง

### 4.4 `write_csv(path, rows, columns)`

เขียนข้อมูลเป็น CSV โดยใช้ UTF-8 with BOM เพื่อให้เปิดในโปรแกรม spreadsheet ได้สะดวกขึ้น

---

## 5. `prepare_cohort(root)`: รวมทุกอย่างให้เป็นชุดข้อมูลพร้อมฝึก

นี่คือฟังก์ชันหลักของ data pipeline

ลำดับการทำงาน:
1. สร้าง folder `raw/` และ `outputs/`
2. ดาวน์โหลดไฟล์ data และ layout ของแต่ละ wave
3. อ่าน layout แล้วตรวจ anchor positions บาง field เพื่อจับความเปลี่ยนแปลงของ layout
4. อ่านข้อมูลแต่ละ wave เป็น dictionary ที่ค้นด้วย participant ID
5. วนผู้เข้าร่วม baseline ปี 1982–1984
6. หา record ของคนเดียวกันใน wave 1986, 1987 และ 1992
7. เรียก `build_label()` เพื่อคัดเข้า/ออกและสร้าง target
8. สร้าง merged raw data, target audit และ cohort CSV
9. สร้าง `NHEFS_Data_Audit.json` พร้อมจำนวนแถว แหล่งข้อมูล และเหตุผลการคัดออก
10. ตรวจขั้นต่ำว่ามีตัวอย่างเพียงพอและมีทั้ง target 0 และ 1 ก่อนอนุญาตให้ฝึกโมเดล

ไฟล์สำคัญที่ได้:
- `NHEFS_Merged_Four_Waves_Raw.csv` — ข้อมูลดิบที่รวมตาม ID
- `NHEFS_Target_Audit.csv` — ผลการพิจารณา label ของแต่ละคน รวมเหตุผลที่คัดออก
- `NHEFS_Future_8y_Cohort.csv` — เฉพาะคนที่มี label 0/1 ที่ใช้ฝึกได้
- `NHEFS_Data_Audit.json` — สรุปจำนวนข้อมูล แหล่งที่มา และข้อจำกัด

**ทำไมต้องมี Target Audit?** เพราะการสร้าง label จากข้อมูลติดตามหลายปีมีการตัดสินใจและกรณีที่ไม่ชัดเจน เราจึงควรตรวจย้อนหลังได้ว่าคนหนึ่งถูกคัดออกด้วยเหตุผลใด

---

## 6. คณิตศาสตร์ของโมเดล

### 6.1 `sigmoid(z)`

แปลงค่าตัวเลขใด ๆ ให้เป็นช่วง 0 ถึง 1:

```python
sigmoid(z) = 1 / (1 + exp(-z))
```

ใน Logistic Regression ค่า output นี้ตีความเป็น probability estimate ของ class 1 ภายใต้โมเดล

### 6.2 `fit_logistic(x, y, ridge)`

ฟังก์ชันฝึก Logistic Regression ด้วย NumPy:
1. เพิ่มคอลัมน์ 1 เพื่อแทน intercept หรือ bias
2. เริ่ม coefficient ทุกตัวที่ 0
3. คำนวณ prediction ด้วย sigmoid
4. คำนวณ gradient และ Hessian
5. แก้ระบบสมการเพื่อหา Newton step
6. ใช้ line search ลด step หาก loss ยังไม่ดีขึ้น
7. ทำซ้ำจน gradient เล็กพอ หรือแจ้งว่าไม่ converge

โมเดลมีสมการหลัก:

```text
z = b + w1*x1 + w2*x2 + ... + w5*x5
p = sigmoid(z)
```

- `w` คือ coefficient/weight ที่โมเดลเรียนรู้
- `b` คือ intercept
- `p` คือ probability estimate ของ target 1

### 6.3 L2 / Ridge regularization

ค่า `ridge` เป็นตัวควบคุม penalty ที่ลงโทษ coefficient ที่มีขนาดใหญ่เกินไป ช่วยลดความซับซ้อนและความเสี่ยง overfitting

ในโค้ด intercept ไม่ถูกลงโทษ (`penalty[0,0] = 0`) แต่ coefficient ของ feature ถูกลงโทษ

ค่า ridge มากไม่ได้แปลว่าดีกว่าเสมอ จึงทดลองหลายค่าแล้วเลือกจาก Tune set

### 6.4 `logits(x, coef)`

คำนวณค่า linear score ก่อนผ่าน sigmoid:

```text
b + w1*x1 + ... + w5*x5
```

ค่านี้ยังไม่ใช่ probability เพราะอาจเป็นค่าติดลบหรือมากกว่า 1 ได้

---

## 7. แบ่งข้อมูลและเตรียม feature

### 7.1 `split_ids(y)`

แบ่งตัวอย่างเป็น 4 ชุดแบบ stratified:

| Split | สัดส่วน | ใช้ทำอะไร |
|---|---:|---|
| Train | 55% | เรียนรู้ preprocessing และ coefficient |
| Tune | 15% | เลือกค่า ridge |
| Calibration | 15% | ปรับ probability หลังฝึก |
| Test | 15% | ประเมินผลสุดท้าย |

Stratified หมายถึงพยายามรักษาสัดส่วน target 0/1 ของแต่ละชุดให้ใกล้กับชุดทั้งหมด

แต่ละคนอยู่ได้เพียง split เดียว และ seed 42 ช่วยให้การสุ่มทำซ้ำได้

### 7.2 `fit_preprocessor(x)`

คำนวณค่าที่ต้องใช้แปลงข้อมูล:
- median ของแต่ละ feature — ใช้แทนค่าว่าง
- mean — ใช้สำหรับ standardization
- standard deviation — ใช้ปรับ scale

ฟังก์ชันนี้ถูก fit จาก Train เท่านั้น

### 7.3 `transform(x, prep)`

ใช้ค่าจาก Train:
1. เติม NaN ด้วย median
2. ลบ mean
3. หารด้วย standard deviation

สูตร:

```text
z = (x - mean) / standard_deviation
```

ทำให้ feature ที่มีหน่วยต่างกัน เช่น อายุ, mmHg และชั่วโมงนอน อยู่บน scale ที่เปรียบเทียบและ optimize ได้สะดวกขึ้น

ห้ามคำนวณ mean/standard deviation จาก Test เพราะจะเป็น data leakage

---

## 8. `train_model(cohort, root, audit)`

นี่คือ orchestration ของการฝึกโมเดล

### ลำดับการทำงาน

1. แยก feature เป็น matrix `x` และ target เป็น array `y`
2. แบ่ง split ด้วย `split_ids()`
3. fit preprocessing จาก Train แล้ว transform ทุก split ด้วยค่าชุดเดียวกัน
4. ทดลอง ridge = 0.01, 0.1, 1, 10, 100
5. ฝึกโมเดลแต่ละค่าโดยใช้ Train
6. ประเมิน Brier score บน Tune
7. เลือกค่า ridge ที่มี Tune Brier ต่ำที่สุด
8. นำ score จาก Calibration ไป fit Platt calibration
9. ตรวจว่า calibration slope เป็นบวก
10. ประเมินโมเดลที่เลือกแล้วบน Test เพียงครั้งเดียว
11. เปรียบเทียบกับ constant baseline ที่ทำนายทุกคนด้วย event prevalence จาก Train
12. คำนวณ bootstrap interval ของ ROC-AUC และ calibration bins
13. บันทึก model, metrics, predictions, feature dictionary และ report
14. zip ไฟล์ทั้งหมด

### 8.1 ทำไมต้องมี Tune?

Train ใช้เรียนรู้ coefficient ส่วน Tune ใช้เลือก hyperparameter เช่น ridge หากเลือกจาก Test จะทำให้ Test ไม่ใช่ชุดประเมินอิสระอีกต่อไป

### 8.2 Platt calibration คืออะไรในโค้ดนี้?

หลัง Logistic Regression ได้ score แล้ว โค้ด fit Logistic Regression อีกตัวบน Calibration set เพื่อแปลง score ให้ probability มีความสอดคล้องกับอัตราที่เกิดขึ้นจริงมากขึ้น

แนวคิด:

```text
original score -> calibration logistic model -> calibrated probability
```

Calibration ไม่ได้ทำให้โมเดลแม่นขึ้นทุกด้านโดยอัตโนมัติ และไม่ใช่การยืนยันว่า probability ใช้กับประชากรอื่นได้

### 8.3 Constant baseline

baseline นี้ไม่ใช้ feature ของแต่ละคน แต่ทำนายทุกคนด้วยสัดส่วน target 1 ที่พบใน Train

เช่น ถ้า Train มี target 1 อยู่ 20% ก็ให้ทุกคนมี probability 0.20

โมเดลที่ซับซ้อนควรถูกเปรียบเทียบกับ baseline ง่าย ๆ เสมอ ถ้าไม่ชนะ baseline ก็ต้องรายงานตามจริง ไม่ควรเลือกแสดงเฉพาะตัวเลขที่ดูดี

---

## 9. Metrics: ฟังก์ชันที่ใช้ประเมิน

### 9.1 `auc(y, p)`

คำนวณ ROC-AUC จาก ranking ของ probability:
- 1.0 = จัดลำดับ positive เหนือ negative ได้ดีมาก
- 0.5 = ใกล้เคียงการสุ่ม
- ต่ำกว่า 0.5 = จัดลำดับกลับทิศทาง

AUC วัดความสามารถในการแยก/จัดอันดับ ไม่ได้บอกว่า probability ถูก calibrate ดีหรือไม่

### 9.2 `metrics(y, p)`

ใช้ threshold 0.5 เพื่อเปลี่ยน probability เป็น class แล้วคำนวณ:

| Metric | ความหมาย |
|---|---|
| Accuracy | สัดส่วนที่ทาย class ถูกทั้งหมด |
| Precision | ในกลุ่มที่ทายว่าเกิดเหตุการณ์ มีกี่ส่วนที่เป็น positive จริง |
| Recall | ใน positive จริง จับได้กี่ส่วน |
| Specificity | ใน negative จริง ทายถูกว่า negative กี่ส่วน |
| Balanced accuracy | ค่าเฉลี่ย Recall และ Specificity |
| F1 | ค่าเฉลี่ยแบบ harmonic ระหว่าง Precision และ Recall |
| Brier score | ค่าเฉลี่ยกำลังสองของความต่างระหว่าง probability กับ label; ยิ่งต่ำยิ่งดี |
| Log loss | ลงโทษ probability ที่มั่นใจผิดอย่างรุนแรง; ยิ่งต่ำยิ่งดี |
| ROC-AUC | ความสามารถในการจัดอันดับ positive ให้สูงกว่า negative |

Confusion matrix:
- TN — negative ที่ทายถูก
- FP — negative ที่ทายผิดเป็น positive
- FN — positive ที่ทายผิดเป็น negative
- TP — positive ที่ทายถูก

**Threshold 0.5 เป็นค่า demo ในโค้ด ไม่ใช่ threshold ทางการแพทย์**

### 9.3 Bootstrap 95% interval

สุ่มตัวอย่างจาก Test set แบบมีการคืนตัวอย่าง 200 รอบ แล้วคำนวณ AUC แต่ละรอบ จากนั้นใช้ percentile 2.5% และ 97.5% เป็นช่วงความไม่แน่นอนโดยประมาณ

นี่ไม่ใช่การแบ่ง Test ใหม่เพื่อเลือกโมเดล แต่เป็นการประเมินความไม่แน่นอนของ metric บน Test ที่กันไว้

### 9.4 Calibration bins

แบ่ง probability เป็นช่วง 0.0–0.1, 0.1–0.2, ... แล้วเปรียบเทียบ:
- mean predicted — probability เฉลี่ยที่โมเดลทำนาย
- observed rate — สัดส่วน positive จริงในกลุ่มนั้น

ถ้าโมเดลให้ค่าเฉลี่ย 0.20 ในกลุ่มหนึ่ง แต่เกิดจริง 0.40 แสดงว่าค่าทำนายต่ำกว่าความถี่ที่สังเกตได้ในกลุ่มนั้น

---

## 10. `predict_future()`: ใช้โมเดลกับ input ใหม่

ฟังก์ชันนี้ไม่ได้ฝึกโมเดลใหม่ แต่ใช้ coefficient และ preprocessing ที่บันทึกไว้แล้ว

ลำดับ:
1. ปฏิเสธคนที่ระบุว่าเคยได้รับการวินิจฉัยมาก่อน
2. ตรวจว่า model bundle มาจากการฝึกด้วยข้อมูล NHEFS จริง
3. ตรวจ input ทุก feature ว่ามีค่าและเป็น finite number
4. อนุญาตให้ sleep หายได้ แล้วเติมด้วย median จาก Train
5. ตรวจค่าต้องอยู่ใน training range
6. ตรวจ sex เป็น 0 หรือ 1
7. ตรวจ SBP < 130 และ DBP < 80 ตาม cohort rule
8. transform input ด้วย preprocessing จาก Train
9. คำนวณ logistic score
10. ใช้ Platt calibration แล้วคืน probability พร้อม warning

ตัวอย่างการเรียก:

```python
INPUT = {
    'age_years': 50,
    'sex_female': 0,
    'baseline_sbp': 120,
    'baseline_dbp': 75,
    'sleep_hours': 7
}

answer = predict_future(
    model_bundle,
    INPUT,
    previously_diagnosed=False
)

print(answer['experimental_probability_8y'])
```

ค่า output เป็น **experimental probability สำหรับ endpoint ของชุดข้อมูลนี้** ไม่ควรนำไปใช้ตัดสินใจรักษาหรือประเมินความเสี่ยงสุขภาพจริง

---

## 11. Notebook cells ทำหน้าที่อะไร?

| Cell | ประเภท | หน้าที่ |
|---:|---|---|
| 0 | Markdown | อธิบายเป้าหมาย ขอบเขต และข้อจำกัด |
| 1 | Code | imports, constants, data pipeline, model, metrics และ prediction functions |
| 2 | Markdown | หัวข้อเตรียมข้อมูล |
| 3 | Code | ดาวน์โหลดข้อมูลจริงและสร้าง cohort |
| 4 | Markdown | หัวข้อฝึกและประเมิน |
| 5 | Code | เรียก `train_model()` และแสดง report |
| 6 | Markdown | หัวข้อทดลอง prediction |
| 7 | Code | สร้างตัวอย่าง input และเรียก `predict_future()` |
| 8 | Code | สร้าง interactive form ด้วย ipywidgets หากติดตั้งได้ |
| 9 | Markdown | อธิบาย ZIP output |

---

## 12. Output files

| File | มีอะไร |
|---|---|
| `NHEFS_Merged_Four_Waves_Raw.csv` | ข้อมูลดิบจากแต่ละ wave ที่จับคู่ด้วย ID |
| `NHEFS_Target_Audit.csv` | target, เหตุผลที่คัดออก และ metadata |
| `NHEFS_Future_8y_Cohort.csv` | cohort ที่มี target ใช้ฝึกได้ |
| `NHEFS_Data_Audit.json` | จำนวน record, target counts, source hashes และ limitations |
| `NHEFS_Model_Data.csv` | features, target และ split ของแต่ละคน |
| `NHEFS_Future_Model.json` | coefficients, preprocessing, calibration และ metadata ของโมเดล |
| `NHEFS_Model_Metrics.json` | ผลประเมินและข้อมูล split |
| `NHEFS_Test_Predictions.csv` | label จริงเทียบกับ probability บน Test |
| `NHEFS_Test_Calibration.csv` | predicted rate เทียบกับ observed rate ราย bin |
| `NHEFS_Feature_Dictionary.csv` | ความหมาย หน่วย และแหล่งของ feature |
| `NHEFS_Future_Report.md` | รายงานสรุปผลและข้อจำกัด |

---

## 13. ถ้าจะอ่านโค้ดเพื่อเรียนรู้ แนะนำลำดับนี้

อย่าเริ่มจากอ่านทุกบรรทัดของ notebook ตามลำดับ เพราะ cell แรกมีทั้ง data engineering, statistics และ optimization อยู่ด้วยกัน

ลำดับที่แนะนำ:

1. อ่าน `FEATURES` และ `baseline_features()` เพื่อเข้าใจว่า input ของโมเดลคืออะไร
2. อ่าน `build_label()` เพื่อเข้าใจว่า target 0/1 มาจากไหน
3. อ่าน `prepare_cohort()` เพื่อเข้าใจการแปลง raw data เป็น dataset
4. อ่าน `split_ids()`, `fit_preprocessor()`, `transform()` เพื่อเข้าใจการเตรียมข้อมูล
5. อ่าน `sigmoid()`, `fit_logistic()`, `logits()` เพื่อเข้าใจตัวโมเดล
6. อ่าน `train_model()` เพื่อเข้าใจการ train/tune/calibrate/test
7. อ่าน `metrics()` และ `auc()` เพื่อเข้าใจการวัดผล
8. อ่าน `predict_future()` เพื่อเข้าใจการใช้งานหลัง train เสร็จ

อ่านคู่มือนี้ควบคู่กับ [AI และ Data Science Fundamentals](AI_DATA_SCIENCE_FUNDAMENTALS.md) โดยเฉพาะคำว่า feature, target, train/test split, overfitting, probability และ calibration

---

## 14. ข้อจำกัดที่ต้องจำ

- Notebook ที่อยู่ใน repository ไม่มี trained model หรือผลคะแนนจริงแนบมา ต้องรันดาวน์โหลด NHEFS และฝึกใหม่
- ถ้าดาวน์โหลดข้อมูลไม่ได้ pipeline จะหยุด ไม่ใช้ synthetic data แทน
- ข้อมูลเป็น cohort สหรัฐฯ ในอดีตและ diagnosis เป็น self/proxy report
- การคัดเฉพาะคนที่มี label ชัดอาจทำให้เกิด selection bias และ informative censoring
- ไม่มี IPCW หรือ competing-risk adjustment
- ไม่มี survey-weighted national estimates
- feature ที่ใช้จริงมี 5 ตัว ไม่ใช่ feature schema เดิม 11 ตัว; BMI และ dietary features ไม่ได้ถูกนำเข้าโมเดล
- ผลลัพธ์ยังไม่ผ่าน external validation หรือ clinical validation สำหรับคนไทยยุคปัจจุบัน

แหล่งข้อมูลหลัก: [CDC/NCHS — NHEFS](https://wwwn.cdc.gov/nchs/nhanes/nhefs/default.aspx)
