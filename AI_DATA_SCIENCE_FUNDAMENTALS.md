# AI & Data Science Fundamentals — เรียนจากศูนย์ผ่าน LEPTOLIC

คู่มือนี้ออกแบบมาสำหรับคนที่ยังไม่เคยเรียน AI, Machine Learning หรือ Data Science มาก่อน โดยจะเริ่มจากคำศัพท์พื้นฐาน แล้วค่อยเชื่อมกลับไปยังโค้ดในโปรเจกต์ LEPTOLIC

ไม่จำเป็นต้องรู้แคลคูลัสหรือสถิติขั้นสูงก่อนอ่าน แต่ควรค่อย ๆ ทำความเข้าใจว่า **ข้อมูลมาจากไหน, เราต้องการให้โมเดลเรียนรู้อะไร, มันเรียนรู้อย่างไร และเรารู้ได้อย่างไรว่าผลลัพธ์น่าเชื่อถือแค่ไหน**

> **ข้อควรจำเกี่ยวกับ LEPTOLIC:** โมเดลนี้ทำนาย “รายงานการได้รับการวินิจฉัยความดันสูงครั้งแรกภายใน 8 ปี” จากข้อมูล NHEFS ในอดีต ไม่ใช่การวินิจฉัยโรคหรือการทำนายความเสี่ยงทางการแพทย์ที่ผ่านการยืนยัน

---

# Part 1 — ภาพใหญ่ของ AI และ Data Science

## 1. AI, Machine Learning และ Data Science ต่างกันอย่างไร?

### Artificial Intelligence (AI)

AI เป็นคำกว้าง ๆ ที่หมายถึงการสร้างระบบที่ทำงานบางอย่างซึ่งปกติต้องใช้ความสามารถแบบฉลาด เช่น การเข้าใจภาษา การวางแผน การรู้จำภาพ หรือการตัดสินใจ

### Machine Learning (ML)

Machine Learning เป็นส่วนหนึ่งของ AI ที่ให้คอมพิวเตอร์เรียนรู้รูปแบบจากข้อมูล แทนที่เราจะเขียนกฎทุกกรณีด้วยมือ

ตัวอย่าง:
- เขียนกฎเอง: ถ้าอายุเกิน 60 ให้คะแนนความเสี่ยง 10
- ใช้ ML: ให้ตัวอย่างข้อมูลจำนวนมากพร้อมผลลัพธ์ แล้วให้โมเดลเรียนรู้ coefficient ที่เชื่อม input กับผลลัพธ์

LEPTOLIC ใช้ Machine Learning แบบ supervised learning

### Data Science

Data Science เป็นกระบวนการใช้ข้อมูลเพื่อทำความเข้าใจปัญหา วิเคราะห์ และสรุปผล อาจใช้สถิติ การเขียนโปรแกรม การสร้างกราฟ และ Machine Learning

**ML ไม่ใช่ทั้ง Data Science** — การตรวจคุณภาพข้อมูล การเลือกกลุ่มตัวอย่าง และการตรวจว่าคำตอบที่ใช้ฝึกถูกต้องหรือไม่ ก็เป็นงาน Data Science ที่สำคัญมาก

---

## 2. Machine Learning เรียนรู้จากอะไร?

ลองนึกถึงการสอนเด็กแยกภาพแมวกับสุนัข:
1. เราให้ภาพจำนวนมาก
2. บอกแต่ละภาพว่าคือแมวหรือสุนัข
3. ระบบค้นหารูปแบบที่ช่วยแยกสองกลุ่ม
4. เราให้ภาพใหม่ที่ไม่เคยเห็น
5. ตรวจว่าระบบแยกได้ถูกต้องหรือไม่

LEPTOLIC ใช้แนวคิดเดียวกัน แต่แทนภาพด้วยข้อมูลสุขภาพตั้งต้น และแทนคำตอบแมว/สุนัขด้วย target 0/1

### Supervised Learning

การเรียนรู้แบบมีผู้สอน คือการฝึกด้วยข้อมูลที่มีคำตอบกำกับไว้แล้ว

| คำศัพท์ | ความหมาย | ใน LEPTOLIC |
|---|---|---|
| Sample / Observation | หนึ่งตัวอย่าง | ผู้เข้าร่วม NHEFS หนึ่งคน |
| Feature / Input | ข้อมูลที่ใช้ทำนาย | อายุ เพศ SBP DBP ชั่วโมงนอน |
| Target / Label | คำตอบที่ต้องการทำนาย | รายงานวินิจฉัยภายใน 8 ปี: 0/1 |
| Prediction | คำตอบที่โมเดลคำนวณ | probability ที่โมเดลประเมิน |
| Model | ฟังก์ชันที่เรียนรู้จากข้อมูล | Logistic Regression + calibration |

### Unsupervised Learning

การเรียนรู้แบบไม่มี label เช่น การจัดกลุ่มลูกค้าตามพฤติกรรม โดยไม่มีคำตอบล่วงหน้าว่าแต่ละคนอยู่กลุ่มใด

LEPTOLIC **ไม่ได้ใช้** unsupervised learning

### Reinforcement Learning

การเรียนรู้จากการลองทำ action และรับ reward/penalty เช่น AI เล่นเกม เรียนรู้ว่าการกระทำใดทำให้ได้คะแนนสูง

LEPTOLIC **ไม่ได้ใช้** reinforcement learning เช่นกัน

---

# Part 2 — เข้าใจข้อมูลก่อนเข้าใจโมเดล

## 3. Dataset, Row, Column คืออะไร?

Dataset คือชุดข้อมูลที่จัดเป็นตาราง

ตัวอย่างจำลอง:

| participant_id | age_years | baseline_sbp | sleep_hours | target |
|---|---:|---:|---:|---:|
| A001 | 45 | 115 | 7 | 0 |
| A002 | 52 | 124 | 6 | 1 |
| A003 | 38 | 110 | 8 | 0 |

- **Row** = หนึ่ง record หรือหนึ่งตัวอย่าง
- **Column** = ตัวแปรหนึ่งชนิด
- **Cell** = ค่าของตัวแปรหนึ่งตัวใน record หนึ่งแถว
- **Dataset** = ตารางทั้งหมด

ใน LEPTOLIC หนึ่ง row ใน cohort แทนผู้เข้าร่วมหนึ่งคน ส่วน columns มีทั้ง participant ID, features, target และข้อมูลประกอบเกี่ยวกับวันที่

## 4. Raw Data, Clean Data และ Model-ready Data

ข้อมูลไม่ได้พร้อมใช้กับโมเดลทันที

### Raw Data

ข้อมูลต้นฉบับจากแหล่งข้อมูล เช่น NHEFS fixed-width files ซึ่งอาจมี:
- รหัสคำตอบ
- ค่าว่าง
- special codes
- field ที่มีตำแหน่งตาม layout
- ข้อมูลหลายปีแยกเป็นหลายไฟล์

### Clean Data

ข้อมูลที่ผ่านการอ่าน แปลงชนิด ตรวจช่วงค่า และจัดการความผิดปกติ เช่น แปลงอายุเป็นตัวเลข และไม่ตีความรหัส refused เป็นจำนวนชั่วโมงนอน

### Model-ready Data

ตารางที่จัด feature และ target เรียบร้อยแล้ว พร้อมนำไปแบ่งชุดและฝึกโมเดล

ใน LEPTOLIC:
- `NHEFS_Merged_Four_Waves_Raw.csv` ใกล้เคียงข้อมูลดิบที่รวมตาม ID
- `NHEFS_Future_8y_Cohort.csv` คือ cohort ที่ผ่านกฎสร้าง label แล้ว
- `NHEFS_Model_Data.csv` เพิ่มข้อมูลว่าแต่ละคนอยู่ Train/Tune/Calibration/Test

## 5. Data Cleaning คืออะไร และทำไมสำคัญ?

Data cleaning คือการตรวจและจัดการข้อมูลที่ไม่สมบูรณ์หรือไม่สมเหตุสมผล

ตัวอย่างใน LEPTOLIC:
- อายุที่ไม่อยู่ในช่วงที่กำหนด → ใช้ไม่ได้
- ความดันที่เป็นรหัสพิเศษหรืออยู่นอกช่วงที่ยอมรับ → ใช้ไม่ได้
- วันสัมภาษณ์ที่ไม่มีจริง → ใช้ไม่ได้
- ID ซ้ำ → หยุดการอ่าน
- ไฟล์ที่ดาวน์โหลดมาเป็น HTML error page → ไม่ยอมรับเป็นข้อมูล

**Garbage in, garbage out:** ถ้าข้อมูลผิด โมเดลอาจเรียนรู้ความสัมพันธ์ผิด แม้โค้ดฝึกจะเขียนถูกก็ตาม

## 6. Missing Value, NaN และ Unknown ไม่ใช่สิ่งเดียวกันกับ 0

- `0` เป็นค่าตัวเลขจริง เช่น target = 0 หมายถึงไม่พบเหตุการณ์ตามนิยาม label
- `NaN` หมายถึงไม่มีค่าตัวเลขที่ใช้ได้
- `None` ใน Python อาจใช้แทนการไม่มีคำตอบหรือยังสรุปไม่ได้
- Unknown / refused / skipped เป็นสถานะจากแบบสอบถาม ไม่ใช่คำตอบว่า “ไม่”

ตัวอย่าง:
- ไม่มีรายงานการวินิจฉัย และติดตามครบ 8 ปีพร้อมตอบ no → อาจ label 0
- ขาดการติดตามก่อนครบ 8 ปี → ไม่รู้ว่าเกิดเหตุการณ์หลังจากนั้นหรือไม่ จึงไม่ควร label 0
- รายงานว่าเคยได้รับการวินิจฉัยแต่ไม่ทราบปี → ไม่ควรเดาว่าเป็นก่อนหรือหลัง 8 ปี

ในโค้ด `build_label()` จึงคืน `None` พร้อม reason สำหรับกรณีที่ตอบไม่ได้อย่างน่าเชื่อถือ

---

# Part 3 — สร้างโจทย์ให้ Machine Learning

## 7. Feature คืออะไร?

Feature คือข้อมูลที่โมเดลได้รับเป็น input เพื่อใช้ทำนาย

LEPTOLIC ใช้ 5 features:

| Feature | ความหมาย | หน่วย/รูปแบบ | แหล่งข้อมูล |
|---|---|---|---|
| `age_years` | อายุ | ปี | `SUBAGE` |
| `sex_female` | เพศตาม coding ในแบบสำรวจ | 0 = ชาย, 1 = หญิง | `SUBSEX` |
| `baseline_sbp` | ความดันตัวบน | mmHg | `PMR4cAVS` หรือค่าเฉลี่ยการวัดครั้ง 2/3 |
| `baseline_dbp` | ความดันตัวล่าง | mmHg | `PMR4cAVD` หรือค่าเฉลี่ยการวัดครั้ง 2/3 |
| `sleep_hours` | ชั่วโมงนอน | ชั่วโมง | `OH7` |

**Feature ไม่จำเป็นต้องเป็นตัวเลขตั้งแต่ต้น** แต่โมเดลนี้รับ feature เป็นตัวเลข จึงต้อง encode เพศเป็น 0/1

### ทำไมชื่อ baseline ถึงสำคัญ?

เพราะค่าพวกนี้มาจากช่วงเริ่มต้น ก่อนช่วงเวลาที่เราต้องการทำนาย เราไม่ควรเอาข้อมูลที่เกิดหลัง target event มาเป็น input เพราะจะเป็นการแอบเห็นคำตอบล่วงหน้า

## 8. Target / Label คืออะไร?

Target คือคำตอบที่เราต้องการให้โมเดลเรียนรู้

ในโปรเจกต์:
- `1` = มีรายงานการได้รับการวินิจฉัยครั้งแรกภายใน 8 ปี
- `0` = มีหลักฐานเพียงพอว่าไม่เกิดรายงานการวินิจฉัยภายใน 8 ปี ตามกติกาของโครงการ
- `None` = ไม่สามารถระบุ label ได้อย่างน่าเชื่อถือ จึงไม่เอาไปฝึก

Target นี้ไม่ได้หมายถึง “เริ่มเป็นโรคจริงเมื่อใด” แต่หมายถึง **การได้รับการวินิจฉัยที่ถูกรายงานในการติดตาม**

## 9. Cohort คืออะไร?

Cohort คือกลุ่มคนที่ถูกเลือกเข้ามาศึกษาตามเงื่อนไขที่กำหนด

LEPTOLIC ไม่ได้เอาผู้เข้าร่วม NHEFS ทุกคนมาฝึกทันที แต่คัดเฉพาะกลุ่มที่:
- baseline ตอบว่าไม่เคยได้รับการวินิจฉัยตาม field D2
- มีอายุ เพศ และค่าความดัน baseline ที่ใช้ได้
- SBP < 130 และ DBP < 80 ตามเกณฑ์ของ prototype
- มี target ที่ระบุได้อย่างน่าเชื่อถือ

การเลือก cohort เปลี่ยนความหมายของโมเดล: โมเดลนี้เรียนรู้จากคนที่ผ่านเงื่อนไขดังกล่าว ไม่ได้อธิบายประชากรทุกคน

## 10. Horizon คืออะไร?

Horizon คือช่วงเวลาข้างหน้าที่กำหนดให้โมเดลทำนาย

LEPTOLIC ใช้ 8 ปีนับจากวันที่สัมภาษณ์ baseline

ตัวอย่าง:
- Baseline: 1983-06-15
- Horizon: 1991-06-15

ถ้า target เป็น “ภายใน 8 ปี” เราต้องรู้ว่ารายงานการวินิจฉัยเกิดก่อนหรือหลังเส้นแบ่งนี้

ใน NHEFS รู้ปีที่รายงาน แต่ไม่รู้วันจริงเสมอไป ถ้าปีนั้นคร่อมเส้นแบ่ง 8 ปี โค้ดจะตัดกรณีออก แทนการเดาว่าเกิดก่อนหรือหลัง

---

# Part 4 — เข้าใจประเภทของโจทย์และโมเดล

## 11. Classification vs Regression

คำว่า Regression ในชื่อ Logistic Regression ทำให้สับสนได้

### Classification

ทำนายว่าอยู่ class ไหน เช่น:
- spam / not spam
- cat / dog
- target 1 / target 0

LEPTOLIC เป็น **Binary Classification** เพราะมี 2 classes

### Regression

ทำนายค่าตัวเลขต่อเนื่อง เช่น:
- ราคาบ้าน
- อุณหภูมิ
- ยอดขายเดือนหน้า

### แล้วทำไมเรียกว่า Logistic Regression?

เพราะโมเดลเริ่มจาก linear combination แล้วใช้ logistic/sigmoid function เพื่อสร้าง probability สำหรับ classification

ชื่อมีคำว่า Regression แต่การใช้งานในโปรเจกต์นี้คือ binary classification

## 12. Logistic Regression ทำงานอย่างไร?

โมเดลรับ feature หลายตัว แล้วคำนวณ weighted sum:

```text
z = b + w1*x1 + w2*x2 + ... + w5*x5
```

- `x` = feature input
- `w` = coefficient หรือ weight ที่โมเดลเรียนรู้
- `b` = intercept หรือค่าตั้งต้น
- `z` = score ก่อนแปลงเป็น probability

จากนั้นใช้ sigmoid:

```text
p = 1 / (1 + exp(-z))
```

Sigmoid บีบค่าใด ๆ ให้กลายเป็นตัวเลขระหว่าง 0 กับ 1

ตัวอย่างเชิงแนวคิด:
- z ติดลบมาก → p ใกล้ 0
- z = 0 → p = 0.5
- z เป็นบวกมาก → p ใกล้ 1

ค่า p คือ probability estimate ที่โมเดลสร้างขึ้น ไม่ใช่การรับประกันว่าเหตุการณ์จะเกิดจริง

## 13. Coefficient / Weight คืออะไร?

Weight บอกว่า feature แต่ละตัวมีผลต่อ score `z` อย่างไร เมื่อ feature อื่นคงที่

- coefficient เป็นบวก → เพิ่ม feature แล้ว score มีแนวโน้มเพิ่ม
- coefficient เป็นลบ → เพิ่ม feature แล้ว score มีแนวโน้มลด
- ขนาด coefficient ต้องตีความร่วมกับ scale ของ feature

**ข้อควรระวัง:** coefficient ไม่ได้พิสูจน์เหตุและผล (causation) และไม่ได้แปลว่า feature นั้นเป็นสาเหตุของโรค

LEPTOLIC standardize feature ก่อน train ดังนั้น coefficient ที่เรียนรู้สัมพันธ์กับ feature ที่ถูก standardize แล้ว ไม่ใช่หน่วยดิบโดยตรง

## 14. Intercept คืออะไร?

Intercept หรือ `b` เป็นค่าตั้งต้นของ linear score เมื่อ standardized features ทุกตัวเป็น 0

หลัง standardization ค่า 0 หมายถึงค่าประมาณค่าเฉลี่ยของ feature ใน Train (หลังเติม missing)

มันช่วยให้โมเดลปรับระดับ score โดยรวม ไม่ได้เป็น feature เพิ่มอีกตัวหนึ่ง

## 15. Model Training คืออะไร?

Training คือกระบวนการปรับ coefficient ให้ prediction สอดคล้องกับ target ในข้อมูล Train

โมเดลเริ่มจาก coefficient ตั้งต้น แล้ว:
1. คำนวณ score
2. แปลงเป็น probability
3. เปรียบเทียบกับ label จริงผ่าน loss function
4. ปรับ coefficient
5. ทำซ้ำจนได้คำตอบที่ optimizer ยอมรับว่า converge

ใน LEPTOLIC ฟังก์ชันหลักคือ `fit_logistic()`

## 16. Loss Function คืออะไร?

Loss คือค่าที่บอกว่าโมเดลทำนายผิดมากแค่ไหนในมุมของ objective ที่เลือก

สำหรับ Logistic Regression ใช้ log loss / negative log-likelihood เป็นส่วนหลักของ objective

แนวคิด:
- ถ้า label จริงเป็น 1 แต่โมเดลให้ probability 0.99 → loss ต่ำ
- ถ้า label จริงเป็น 1 แต่โมเดลให้ probability 0.01 → loss สูงมาก

การฝึกคือการหาค่า coefficient ที่ทำให้ objective ต่ำลง โดยมี regularization เพิ่มเข้ามาด้วย

## 17. Optimization คืออะไร?

Optimization คือวิธีหาค่าพารามิเตอร์ที่ทำให้ loss ต่ำที่สุดหรือใกล้ที่สุด

LEPTOLIC ใช้ Newton-style optimization:
- gradient บอกทิศทางและความชันของ objective
- Hessian บอกข้อมูลความโค้ง
- Newton step ใช้ทั้งสองอย่างเพื่อประมาณการปรับ coefficient
- line search ลดขนาด step หากการปรับครั้งนั้นทำให้ objective แย่ลง

ไม่จำเป็นต้องเข้าใจ matrix calculus ตั้งแต่วันแรก แต่ควรรู้ว่า optimizer ไม่ได้ “เดาคำตอบ” มันคำนวณการเปลี่ยน coefficient จาก loss ที่วัดได้

---

# Part 5 — เตรียมข้อมูลก่อนฝึก

## 18. Missing Value Imputation คืออะไร?

Imputation คือการแทนค่าที่หายไปด้วยค่าที่เลือกไว้ แทนที่จะลบแถวนั้นทิ้งทุกครั้ง

LEPTOLIC ใช้ median ของแต่ละ feature ที่คำนวณจาก Train

ตัวอย่าง:
- ชั่วโมงนอนใน Train: 5, 6, 7, 8, missing
- median = 6.5
- ค่าว่างจะถูกแทนด้วย 6.5

Median มักทนต่อ outlier ได้ดีกว่า mean

ใน LEPTOLIC อนุญาตให้ sleep หายได้ แต่ age, sex และ blood pressure ที่จำเป็นต่อ cohort ต้องมีค่าตั้งแต่ขั้นสร้าง cohort แล้ว

## 19. Standardization คืออะไร?

Feature แต่ละตัวมีหน่วยต่างกัน:
- อายุ: ปี
- ความดัน: mmHg
- การนอน: ชั่วโมง

Standardization แปลง feature ให้มี scale ใกล้กัน:

```text
z = (x - mean) / standard deviation
```

- mean = ค่าเฉลี่ย
- standard deviation = ส่วนเบี่ยงเบนมาตรฐาน
- z = ค่าที่แสดงว่าข้อมูลอยู่ห่างจาก mean กี่ standard deviations

ตัวอย่าง:
- ถ้า z = 0 แปลว่าเท่ากับ mean
- ถ้า z = 1 แปลว่าสูงกว่า mean 1 standard deviation
- ถ้า z = -1 แปลว่าต่ำกว่า mean 1 standard deviation

ช่วยให้ optimizer ทำงานกับ feature ที่ scale ต่างกันได้สะดวกขึ้น

## 20. ทำไมต้อง fit preprocessing จาก Train เท่านั้น?

สมมติเราคำนวณ mean จากข้อมูลทั้งหมด รวม Test ด้วย เท่ากับขั้นตอนเตรียมข้อมูลได้เห็นข้อมูลที่ควรเก็บไว้ประเมินแล้ว

สิ่งนี้เป็นรูปแบบหนึ่งของ **Data Leakage**

กฎ:
- fit median/mean/scale จาก Train
- ใช้ค่าที่ได้ transform Train, Tune, Calibration และ Test
- ห้าม fit ใหม่บน Test

LEPTOLIC ทำตามกฎนี้ผ่าน `fit_preprocessor(x[train])` และ `transform(..., prep)`

---

# Part 6 — แบ่งข้อมูลเพื่อไม่หลอกตัวเอง

## 21. Train, Validation/Tune, Calibration และ Test คืออะไร?

ไม่ควรฝึกและวัดผลบนข้อมูลชุดเดียวกัน เพราะโมเดลอาจจำตัวอย่างที่เคยเห็นได้

LEPTOLIC แบ่งดังนี้:

| Set | สัดส่วน | หน้าที่ |
|---|---:|---|
| Train | 55% | เรียนรู้ preprocessing และ coefficient |
| Tune | 15% | เลือก hyperparameter เช่น ridge |
| Calibration | 15% | เรียนรู้การปรับ probability |
| Test | 15% | ประเมินผลสุดท้าย |

### Train

เหมือนชุดโจทย์ที่ใช้สอนโมเดล

### Tune

ใช้เปรียบเทียบการตั้งค่าหลายแบบ เช่น ridge 0.01, 0.1, 1, 10, 100 แล้วเลือกค่าที่เหมาะตาม Brier score

### Calibration

ใช้ปรับความหมายของ probability หลังโมเดลหลักถูกฝึกแล้ว

### Test

เหมือนข้อสอบที่กันไว้จนสุดท้าย ไม่ควรใช้เลือกโมเดลหรือปรับค่าระหว่างพัฒนา

## 22. Stratified Split คืออะไร?

ถ้า target 1 มีน้อยกว่า target 0 มาก การสุ่มแบบธรรมดาอาจทำให้บาง split มี positive น้อยเกินไป

Stratified split จึงแบ่งกลุ่ม target 0 และ target 1 แยกกันก่อน แล้วค่อยกระจายเข้าแต่ละ split เพื่อให้สัดส่วนใกล้เคียงกัน

LEPTOLIC ใช้ `split_ids()` และ `SEED = 42`

Seed ไม่ได้ทำให้โมเดลแม่นขึ้น แต่ช่วยให้การสุ่มทำซ้ำได้

## 23. Data Leakage คืออะไร?

Data leakage คือการที่ข้อมูลหรือความรู้ซึ่งไม่ควรมีตอน train แอบไหลเข้ามาในขั้นตอนฝึกหรือเลือกโมเดล

ตัวอย่าง:
- ใช้ค่าความดันจาก follow-up หลังช่วง baseline เป็น feature เพื่อทำนายเหตุการณ์ในช่วงเดียวกัน
- ใช้ Test set เลือก hyperparameter
- คำนวณ mean และ standard deviation จากข้อมูลทั้งหมดก่อน split
- เอาข้อมูลหลังเหตุการณ์มาเป็น input

ผลคือคะแนนอาจดูดีเกินจริง แต่เมื่อนำไปใช้กับข้อมูลใหม่จะทำงานแย่ลง

## 24. Overfitting และ Underfitting

### Overfitting

โมเดลเรียนรู้รายละเอียดเฉพาะของ Train มากเกินไป จนทำคะแนนกับข้อมูลที่ไม่เคยเห็นได้ไม่ดี

เปรียบเหมือนจำเฉลยข้อสอบเก่า แต่แก้โจทย์ใหม่ไม่ได้

### Underfitting

โมเดลง่ายเกินไปหรือเรียนรู้รูปแบบสำคัญไม่พอ จึงทำได้ไม่ดีทั้ง Train และข้อมูลใหม่

เป้าหมายไม่ใช่ทำ Train score ให้สูงที่สุด แต่ต้อง generalize ไปยังข้อมูลที่ไม่เคยเห็น

---

# Part 7 — Regularization และ Hyperparameter

## 25. Regularization คืออะไร?

Regularization เพิ่ม penalty เข้าไปใน objective เพื่อไม่ให้ coefficient ใหญ่เกินจำเป็น

LEPTOLIC ใช้ L2 regularization หรือ Ridge penalty

แนวคิดโดยย่อ:

```text
objective = logistic loss + ridge * sum(weight^2) / 2
```

- ridge ต่ำ → penalty อ่อน
- ridge สูง → penalty แรงกว่า
- intercept ไม่ถูกลงโทษใน implementation นี้

ไม่ได้หมายความว่า ridge สูงที่สุดดีที่สุด เพราะ penalty มากเกินไปก็ทำให้โมเดลเรียนรู้ได้น้อยเกินไป

## 26. Hyperparameter คืออะไร?

Parameter คือค่าที่โมเดลเรียนรู้จากข้อมูล เช่น coefficient

Hyperparameter คือค่าที่ผู้พัฒนากำหนดก่อนหรือเลือกผ่าน validation เช่น:
- ridge strength
- จำนวนชั้นของ neural network
- learning rate ใน optimizer บางชนิด

LEPTOLIC ทดลอง ridge 5 ค่า แล้วเลือกค่าที่ให้ Tune Brier score ต่ำสุด

---

# Part 8 — Probability และ Calibration

## 27. Probability ที่โมเดลทำนายหมายความว่าอะไร?

ถ้าโมเดลคืนค่า 0.30 หมายถึงโมเดลประเมิน probability 30% สำหรับ target ที่กำหนด ภายใต้ข้อมูลและสมมติฐานของโมเดล

ไม่ได้หมายความว่า:
- มีโอกาส 30% ที่จะเป็นโรคทางคลินิกอย่างแน่นอน
- โมเดลถูกต้อง 30%
- ความแม่นยำของโมเดลคือ 30%

Probability เป็นค่าประเมิน ไม่ใช่คำรับรอง

## 28. Calibration คืออะไร?

โมเดลอาจจัดลำดับคนได้ดี แต่ probability ที่ให้สูงหรือต่ำเกินจริงก็ได้

Calibration คือการตรวจ/ปรับให้ probability สอดคล้องกับความถี่ที่สังเกตได้มากขึ้น

ตัวอย่าง:
- โมเดลให้กลุ่มคน 100 คนมี probability เฉลี่ย 0.20
- ถ้า label 1 เกิดจริงประมาณ 20 คน แสดงว่ากลุ่มนี้ calibrated ได้ดีโดยประมาณ
- ถ้าเกิดจริง 40 คน โมเดลอาจประเมินต่ำไป

LEPTOLIC ใช้ **Platt scaling**:
1. คำนวณ score จาก Logistic Regression หลัก
2. ใช้ score และ label ของ Calibration set ฝึก logistic mapping อีกตัว
3. นำ mapping นี้ไปแปลง score ของ Test หรือ input ใหม่เป็น calibrated probability

Calibration ไม่ได้แก้ selection bias และไม่ทำให้โมเดลใช้กับประชากรอื่นได้โดยอัตโนมัติ

---

# Part 9 — อ่านผลการประเมินโมเดล

## 29. Confusion Matrix

หลังใช้ threshold แปลง probability เป็น class จะมีผล 4 แบบ:

| | ทำนาย Positive | ทำนาย Negative |
|---|---|---|
| จริง Positive | TP | FN |
| จริง Negative | FP | TN |

- TP: ทายว่าเกิดเหตุการณ์ และ label จริงเป็น 1
- TN: ทายว่าไม่เกิด และ label จริงเป็น 0
- FP: เตือนว่าเกิด แต่ label จริงเป็น 0
- FN: ทายว่าไม่เกิด แต่ label จริงเป็น 1

## 30. Metrics ที่ควรรู้

| Metric | คำถามที่ตอบ | แนวโน้มที่ดี |
|---|---|---|
| Accuracy | ทายถูกทั้งหมดกี่ส่วน? | สูง |
| Precision | ที่ทาย positive มี positive จริงกี่ส่วน? | สูง |
| Recall | จาก positive จริง จับได้กี่ส่วน? | สูง |
| Specificity | จาก negative จริง ทาย negative ถูกกี่ส่วน? | สูง |
| F1 | สมดุลระหว่าง precision และ recall เป็นอย่างไร? | สูง |
| Balanced Accuracy | เฉลี่ยความสามารถจับ positive และ negative | สูง |
| ROC-AUC | จัดอันดับ positive สูงกว่า negative ได้ดีแค่ไหน? | สูง |
| Brier Score | probability ใกล้ label จริงแค่ไหน? | ต่ำ |
| Log Loss | probability ที่มั่นใจผิดถูกลงโทษมากแค่ไหน? | ต่ำ |

ไม่มี metric ตัวเดียวที่สรุปคุณภาพโมเดลได้ทั้งหมด

### Accuracy อาจหลอกเราได้อย่างไร?

สมมติ dataset มี:
- 95 คนเป็น 0
- 5 คนเป็น 1

โมเดลที่ทาย 0 ทุกคนจะได้ accuracy 95% แต่จับ positive ไม่ได้เลย

จึงควรดู Recall, Precision, Balanced Accuracy, AUC และ metrics อื่นร่วมกัน

## 31. ROC-AUC

ROC-AUC วัดความสามารถในการจัดอันดับ:
- AUC = 1.0: แยก positive/negative ได้สมบูรณ์ในชุดนั้น
- AUC = 0.5: ใกล้การสุ่ม
- AUC < 0.5: อาจจัดอันดับกลับทิศ

AUC ไม่ได้บอกว่าค่า probability เช่น 0.7 หมายถึงเกิดจริง 70% หรือไม่

## 32. Brier Score

คำนวณค่าเฉลี่ยของ:

```text
(predicted_probability - actual_label)^2
```

ยิ่งต่ำยิ่งดี โดยพิจารณาทั้งความใกล้ของ probability และความถูกต้องของ prediction

ใน LEPTOLIC ใช้ Brier score เลือก ridge บน Tune และรายงานผล Test ด้วย

## 33. Log Loss

Log loss ลงโทษ prediction ที่มั่นใจผิดอย่างรุนแรง

เช่น ถ้าเหตุการณ์เกิดจริง (label 1) แต่โมเดลให้ probability 0.001 จะเสีย loss มากกว่าการให้ 0.40

จึงช่วยตรวจว่าโมเดลให้ probability ที่มั่นใจเกินไปผิด ๆ หรือไม่

## 34. Threshold คืออะไร?

Probability ยังเป็นเลข 0–1 ถ้าต้องการแปลงเป็น class ต้องกำหนด threshold

ตัวอย่าง threshold 0.5:
- p >= 0.5 → predict 1
- p < 0.5 → predict 0

เปลี่ยน threshold แล้ว Precision/Recall และ confusion matrix เปลี่ยนได้

LEPTOLIC ใช้ 0.5 เพื่อสาธิตเท่านั้น ไม่ใช่ clinical threshold และไม่ควรตีความเป็นเกณฑ์ตัดสินใจสุขภาพ

## 35. Baseline Model คืออะไร?

Baseline คือโมเดลง่าย ๆ ที่ใช้เป็นจุดเปรียบเทียบ

LEPTOLIC ใช้ constant baseline:
- คำนวณสัดส่วน target 1 จาก Train
- ทำนาย probability เดียวกันให้ทุกคนใน Test

ถ้าโมเดลหลักทำได้ไม่ดีกว่า baseline ง่าย ๆ ก็ยังไม่มีหลักฐานว่าความซับซ้อนของโมเดลช่วยเพิ่มประโยชน์

---

# Part 10 — ความไม่แน่นอนและข้อจำกัดของข้อมูล

## 36. Bootstrap คืออะไร?

Bootstrap เป็นวิธีประมาณความไม่แน่นอนโดยสุ่มตัวอย่างจากชุดข้อมูลเดิมแบบมีการคืนตัวอย่าง (sampling with replacement) หลายรอบ แล้วคำนวณ metric ซ้ำ

LEPTOLIC:
- สุ่มจาก Test 200 รอบ
- คำนวณ ROC-AUC ในแต่ละรอบ
- ใช้ percentile 2.5% และ 97.5% เป็น bootstrap 95% interval

ช่วงนี้บอกความไม่แน่นอนของ metric ภายใต้ข้อมูล Test ที่มี ไม่ได้ยืนยันว่าโมเดลจะทำงานเหมือนกันในประชากรอื่น

## 37. Selection Bias คืออะไร?

Selection bias เกิดเมื่อกลุ่มตัวอย่างที่นำมาวิเคราะห์แตกต่างจากกลุ่มที่เราต้องการอ้างอิงอย่างเป็นระบบ

LEPTOLIC ตัดคนที่ไม่มี label ชัดเจนออก เช่น คนที่ติดตามไม่ครบหรือมีรายงานไม่ชัด ทำให้กลุ่มที่เหลืออาจไม่เหมือนผู้เข้าร่วมทั้งหมด

ดังนั้นผลที่ได้อาจสะท้อนเฉพาะกลุ่มที่มีข้อมูลติดตามเพียงพอ

## 38. Censoring และ Informative Censoring

Censoring เกิดเมื่อเราไม่สามารถสังเกตผลลัพธ์ของคนหนึ่งได้ครบตลอดช่วงเวลาที่สนใจ เช่น หายจากการติดตามหรือเสียชีวิตก่อนครบ 8 ปี

ถ้าการหายไปจากการติดตามเกี่ยวข้องกับความเสี่ยงหรือสุขภาพของบุคคล ไม่ใช่การหายแบบสุ่ม เราเรียกปัญหานี้ว่า informative censoring

LEPTOLIC ไม่แปลงคนที่ติดตามไม่ครบเป็น target 0 แต่การคัดออกก็ยังอาจทำให้เกิด selection bias ได้ และ prototype นี้ยังไม่ได้ใช้วิธี IPCW หรือ competing-risk adjustment

## 39. External Validation คืออะไร?

การทดสอบบนข้อมูลที่มาจากประชากรหรือช่วงเวลาที่แตกต่างจากข้อมูลฝึก

ตัวอย่าง:
- Train จาก cohort สหรัฐฯ ในอดีต
- External validation บน cohort สมัยใหม่ที่เก็บข้อมูลอย่างเหมาะสม

การที่ Test set ภายใน NHEFS ทำคะแนนดี ไม่ได้พิสูจน์ว่าใช้กับคนไทยยุคปัจจุบันได้

LEPTOLIC ยังไม่มี external validation ดังกล่าว

## 40. Correlation ไม่เท่ากับ Causation

Correlation คือความสัมพันธ์ที่ข้อมูลแสดงให้เห็น ส่วน causation คือความสัมพันธ์เชิงเหตุและผล

ถ้าโมเดลพบว่า feature บางตัวสัมพันธ์กับ target ไม่ได้แปลว่าการเปลี่ยน feature นั้นจะทำให้ target เปลี่ยนตาม

Machine Learning ทั่วไปมักเรียนรู้รูปแบบเพื่อทำนาย ไม่ได้พิสูจน์สาเหตุ

---

# Part 11 — เชื่อมความรู้กับโค้ดจริงใน LEPTOLIC

| แนวคิด | อยู่ตรงไหนใน Notebook | ทำหน้าที่ |
|---|---|---|
| Feature definition | `FEATURES` | ระบุ input ของโมเดล |
| Data cleaning | `integer()`, `valid_number()` | แปลง/ตรวจค่าดิบ |
| Feature engineering | `baseline_features()` | สร้าง input จาก field ของ CDC |
| Label construction | `build_label()` | สร้าง target 0/1 หรือคัดออก |
| Data parsing | `parse_layout()`, `read_wave()` | อ่าน fixed-width data |
| Data pipeline | `prepare_cohort()` | รวม wave และสร้าง cohort |
| Probability function | `sigmoid()` | แปลง score เป็น 0–1 |
| Model training | `fit_logistic()` | เรียนรู้ coefficient |
| Prediction score | `logits()` | คำนวณ weighted sum |
| Data splitting | `split_ids()` | แยก Train/Tune/Calibration/Test |
| Imputation + scaling | `fit_preprocessor()`, `transform()` | เติม missing และ standardize |
| Hyperparameter selection | `train_model()` | เลือก ridge จาก Tune |
| Calibration | `train_model()` และ calibration coefficients | ปรับ probability |
| Evaluation | `auc()`, `metrics()` | วัดคุณภาพโมเดล |
| Inference | `predict_future()` | ใช้โมเดลกับ input ใหม่ |

---

# Part 12 — ลำดับการเรียนรู้สำหรับผู้เริ่มต้น

ไม่จำเป็นต้องเรียนทุกอย่างพร้อมกัน แนะนำให้เรียนเป็นชั้น ๆ

## Level 1: เข้าใจข้อมูล

ควรรู้:
- dataset, row, column
- data type: integer, float, string, boolean
- missing value และ NaN
- CSV และ JSON
- data cleaning
- feature และ target

**เป้าหมาย:** อธิบายได้ว่าหนึ่งแถวแทนอะไร และแต่ละ column ใช้ทำอะไร

## Level 2: เข้าใจโจทย์ Machine Learning

ควรรู้:
- supervised learning
- classification vs regression
- binary classification
- sample, feature, label
- cohort และ prediction horizon

**เป้าหมาย:** อธิบายได้ว่าโมเดลรับอะไรเข้าไปและกำลังพยายามทำนายอะไร

## Level 3: เข้าใจคณิตศาสตร์ที่จำเป็น

ควรรู้:
- mean และ median
- standard deviation
- probability
- weighted sum
- sigmoid
- loss function
- coefficient และ intercept

**เป้าหมาย:** อ่านสมการ Logistic Regression แล้วอธิบายแต่ละตัวแปรได้

## Level 4: เข้าใจการฝึกโมเดล

ควรรู้:
- train/test split
- validation/tuning
- preprocessing
- data leakage
- overfitting/underfitting
- regularization
- hyperparameter

**เป้าหมาย:** อธิบายได้ว่าทำไมเราไม่ควร train และ test ด้วยข้อมูลชุดเดียวกัน

## Level 5: เข้าใจการประเมิน

ควรรู้:
- confusion matrix
- precision, recall, specificity
- F1, accuracy, balanced accuracy
- ROC-AUC
- Brier score และ log loss
- calibration และ threshold
- baseline model

**เป้าหมาย:** อ่าน metrics report แล้วบอกได้ว่า metric แต่ละตัวตอบคำถามอะไร และมีข้อจำกัดอะไร

## Level 6: เข้าใจคุณภาพงานวิจัยข้อมูล

ควรรู้:
- sampling bias / selection bias
- censoring
- external validation
- correlation vs causation
- data provenance และ reproducibility

**เป้าหมาย:** ไม่สรุปเกินกว่าที่ข้อมูลและการทดลองรองรับ

---

# Part 13 — คำถามทดสอบความเข้าใจ

ลองตอบคำถามเหล่านี้ด้วยคำพูดของตัวเองก่อนเปิดดูโค้ด:

1. หนึ่ง row ใน `NHEFS_Future_8y_Cohort.csv` หมายถึงอะไร?
2. Feature กับ target ต่างกันอย่างไร?
3. ทำไม LEPTOLIC ต้องใช้ข้อมูล follow-up เพื่อสร้าง target แต่ไม่เอาค่าความดันจาก follow-up มาเป็น baseline feature?
4. ทำไมคนที่ติดตามไม่ครบ 8 ปีจึงไม่ควรถูกกำหนด target = 0?
5. Logistic Regression เปลี่ยน feature หลายตัวให้เป็น probability ได้อย่างไร?
6. Coefficient กับ hyperparameter ต่างกันอย่างไร?
7. ทำไมต้องคำนวณ median, mean และ standard deviation จาก Train เท่านั้น?
8. Tune set ต่างจาก Test set อย่างไร?
9. ทำไม ROC-AUC สูงไม่ได้แปลว่า probability calibrated ดี?
10. ถ้า Accuracy สูง แต่ Recall ต่ำมาก ควรระวังอะไร?
11. ทำไมต้องเปรียบเทียบโมเดลกับ constant baseline?
12. ทำไมผลจาก cohort สหรัฐฯ ในอดีตจึงยังใช้ยืนยันความเสี่ยงของคนไทยยุคปัจจุบันไม่ได้?

ถ้าตอบได้โดยไม่ท่องจำ และสามารถชี้ไปยัง function ที่เกี่ยวข้องใน Notebook ได้ แปลว่าเริ่มเข้าใจ pipeline แล้ว

---

## สรุปสุดท้าย

สิ่งที่สำคัญที่สุดของ Machine Learning ไม่ใช่แค่การเลือก algorithm แต่คือการสร้างโจทย์และข้อมูลให้ถูกต้อง:

1. เข้าใจว่าข้อมูลแต่ละช่องหมายถึงอะไร
2. กำหนด feature และ target ให้ตรงกับคำถาม
3. สร้าง label อย่างระมัดระวัง ไม่เดาค่าที่ไม่รู้
4. แยกข้อมูลเพื่อป้องกัน leakage
5. ฝึกโมเดลและเลือก hyperparameter ด้วยชุดที่เหมาะสม
6. ประเมินด้วยหลาย metric และเปรียบเทียบ baseline
7. รายงานข้อจำกัดและไม่อ้างเกินข้อมูล

สำหรับ LEPTOLIC การสร้าง target จากประวัติ follow-up อย่างระมัดระวังมีความสำคัญพอ ๆ กับ Logistic Regression เพราะถ้าคำตอบที่ใช้สอนโมเดลผิด โมเดลก็เรียนรู้จากคำตอบที่ผิดตั้งแต่ต้น

แหล่งข้อมูล:
- [CDC/NCHS — NHEFS](https://wwwn.cdc.gov/nchs/nhanes/nhefs/default.aspx)
- [Google Machine Learning Crash Course](https://developers.google.com/machine-learning/crash-course)
- [scikit-learn — Model evaluation](https://scikit-learn.org/stable/modules/model_evaluation.html)
- [scikit-learn — Preprocessing data](https://scikit-learn.org/stable/modules/preprocessing.html)
