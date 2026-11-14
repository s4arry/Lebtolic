# 04 · MODEL — Ridge Logistic Regression

**ชื่อรุ่นที่เลือก:** weight + smoking · Logistic Regression · C=0.1 · ไม่ปรับ calibration เพิ่ม

**ข้อความบนสไลด์**

- Input 6 ตัวแปรตั้งต้น → 9 encoded columns; เติมน้ำหนักที่หายด้วย median ของ training
- L2 regularization ลดความซับซ้อนของสัมประสิทธิ์; C น้อยลงหมายถึง regularization มากขึ้น
- ทดลอง Platt/sigmoid และ intercept calibration แล้ว แต่ development ยังไม่สนับสนุนการเพิ่ม
- ทดลอง flag ด้วย threshold **9.8034%** ที่เลือกจาก development ภายใต้ empirical FPR budget 14.5%

**คำพูดประกอบ:** Baseline 5-feature ใน main ใช้ Platt ส่วนรุ่นอ้างอิงที่เลือกจาก `befec00` ไม่ใช้ Platt เพิ่ม การเลือก calibration อยู่ภายใน training folds; sigmoid ทำให้ calibration slope ใกล้ 1 แต่ไม่ได้ช่วย Brier และ recall สม่ำเสมอบน outer validation คะแนนเป็น research score สำหรับ cohort นี้

**ภาพ:** [calibration_comparison.png](calibration_comparison.png) · [calibration_metrics.csv](calibration_metrics.csv) แสดง 3-repeat mean ± SD; SD ไม่ใช่ confidence interval

**แหล่งข้อมูล:** `model/model.json`, `round3/results/calibration_20261111/README_TH.md`
