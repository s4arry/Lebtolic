# 01 · FINAL PIPELINE — จากข้อมูล 4 ช่วง สู่คะแนนใน 8 ปี

**ข้อความบนสไลด์**

1. จับคู่ผู้เข้าร่วมคนเดิมใน 1982–84, 1986, 1987 และ 1992 ด้วย ID
2. ตรวจ baseline eligibility และ follow-up; กรณีผลลัพธ์ไม่ชัดเจนถูกคัดออก
3. ได้ cohort 2,720 คน → development 2,311 คน / test 409 คน
4. Development: preprocessing ภายใน training folds → repeated nested CV → เลือกโมเดล/threshold
5. Fit รุ่นอ้างอิงจาก development ทั้งหมด → ประเมินบน historical test ที่เคยเปิดดู

**คำพูดประกอบ:** ข้อมูล 1 แถวคือ 1 คน ฟีเจอร์ต้องมี ณ baseline เป้าหมายคือการรายงานการวินิจฉัย ไม่ใช่ผลวัดยืนยันในอนาคต ในการประเมินแต่ละ fold median/scaler และการเลือก threshold เรียนรู้จากข้อมูลฝึก ไม่ใช้ผล test เลือก

**คำเตือน:** รอบนี้เรียก 4 ช่วงข้อมูลหรือ waves; ไม่ยืนยันว่าทุกคนถูกสัมภาษณ์ครบทั้ง 4 ครั้ง

**แหล่งข้อมูล:** `round3/results/calibration_20261111/README_TH.md`, `round3/CV_PROTOCOL_TH.md` และ `round3/data_archive.py`
