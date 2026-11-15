# สรุปโมเดลและข้อแลกเปลี่ยนสำหรับผู้นำเสนอ

## รุ่นที่ยึดเป็นหลัก

รุ่น `befec00`: Logistic Regression C=.1 มี L2 regularization; ฟีเจอร์พื้นฐาน 6 ตัวคือ อายุ เพศ SBP DBP น้ำหนักที่วัด และประวัติสูบบุหรี่ แปลงเป็น 9 encoded columns รุ่นนี้ไม่ได้ balance หรือเพิ่ม Platt calibration; เลือก threshold **0.09803399686084495** จาก development ภายใต้งบ empirical FPR 14.5%

## ผลหลักและคำแปล

| ตัวพิจารณา | Development CV mean | Test เดิม (409 คน, 25 events) |
|---|---:|---:|
| ROC-AUC | 0.7294 | 0.7034 |
| Recall | 43.97% | 40.00% |
| Precision | 17.11% | 16.95% |
| F2 | 0.3346 | 0.3145 |
| FPR | 13.84% | 12.76% |
| Brier | 0.055488 | 0.057010 |

Test TP10, FP49, FN15, TN335: ในคนที่รายงานเหตุการณ์ 25 คน โมเดลแจ้งเตือน 10 คนและพลาด 15 คน; ในคนที่ไม่มีเหตุการณ์ 384 คน แจ้งเตือนผิด 49 คน จึงมีคนได้รับแจ้งเตือนรวม 59 คน และ 10 คนในนั้นรายงานเหตุการณ์จริง

**ข้อดี:** Development AUC ของชุดฟีเจอร์นี้สูงกว่าตัวเทียบ 4-feature ในการทดลองเดียวกัน และคัดจับเหตุการณ์ได้มากกว่าการใช้ threshold .5 ซึ่งไม่เตือนใคร

**ข้อแลกเปลี่ยน:** ยังพลาด 60% ของเหตุการณ์ Test และมี false alerts 49 คน; Brier สูงกว่า main เล็กน้อย; threshold หรือ FPR ไม่รับประกันว่าจะคงที่ในข้อมูลชุดใหม่

## ความหมายของการเทียบ

- `main` notebook จริงใช้ 5 features + Platt, Test AUC 0.6761 / Brier 0.05653; รุ่นอ้างอิง AUC 0.7034 / Brier 0.05701 แต่รูปแบบการฝึกและจำนวนข้อมูลฝึกต่างกัน เปรียบเทียบเป็น **workflow** ไม่ใช่อ้างผลของตัวแปรอย่างเดียว
- Confidence intervals ของผลต่าง Test AUC และ Brier คร่อมศูนย์ ไม่ยืนยัน superiority
- การทดลอง balancing 50/50 ในสไลด์ 06 เป็น **4-feature experiment ก่อนหน้า** และใช้ random oversampling เพิ่ม class 1 ไม่ใช่รุ่นอ้างอิงและไม่ใช่การ undersample
- คะแนน Test เคยเปิดดูหลายครั้ง; การเลือกและการทดลองหลายรอบมี selection optimism ไม่มี external/Thai validation

## ขอบเขตของคำกล่าว

รายงานผลว่า “การรายงานว่าได้รับการวินิจฉัยครั้งแรกภายใน 8 ปี” ไม่กล่าวว่าเป็นโรคจากการวัดแน่นอน; class 0 หมายถึงไม่พบรายงานในช่วงติดตามที่รองรับ ไม่ใช่ไม่เคยเป็นตลอดชีวิต โมเดลเหมาะเป็นต้นแบบวิจัยและการเรียน ไม่ใช่เครื่องมือคัดกรอง/รักษาที่ผ่านการยืนยัน

แหล่ง: [`model/README.md`](https://github.com/s4arry/Lebtolic/blob/dev-by-gunner/model/README.md), [`calibration report`](https://github.com/s4arry/Lebtolic/blob/dev-by-gunner/round3/results/calibration_20261111/README_TH.md), [`main comparison`](https://github.com/s4arry/Lebtolic/blob/dev-by-gunner/round3/results/main_comparison_20261111/README_TH.md)
