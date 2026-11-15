# ต่อรุ่นน้ำหนัก + สูบบุหรี่: nonlinear Logistic — protocol ก่อนรัน

เริ่มจาก commit befec00 รุ่นตั้งต้นคือ weight_smoking Logistic C=.1 ไม่ calibration
ไม่ใช่ baseline 5-feature เดิม ใช้ข้อมูล hashes/cohort/splits เดิมทุกประการ
Development 2311/141 events และ historical Test 409/25; Test ถูกเปิดดูแล้วหลายรอบ

สมมติฐาน: อายุ, SBP, DBP, น้ำหนัก อาจสัมพันธ์กับ log-odds แบบโค้ง
ทดลองเพิ่มเฉพาะ squared terms ของตัวแปรต่อเนื่อง 4 ตัว หลัง standardize
ไม่มี interaction, ไม่ยกกำลังสอง sex/missing/smoking indicators และไม่เพิ่มตัวแปรจาก follow-up
ยังเป็น Logistic Regression + L2 regularization ตามแนวคิดในวิชา

Candidates 6 แบบ: linear/quadratic × C=.01/.1/1 (C น้อย = regularization มาก)
รายงาน 4 workflows: none (linear C=.1 รุ่นล่าสุด), linear ที่เลือก C,
quadratic ที่เลือก C และ selected ที่เลือกทั้ง shape/C
SimpleImputer median, scaler ก่อน square และ scaler หลังขยาย terms เรียนจาก training เท่านั้น
9 encoded columns เดิม → 13 terms เมื่อเพิ่ม squares; ไม่มี balancing/calibration เพิ่มในรอบนี้
เพื่อแยกผลของ nonlinear terms ออกจากการเปลี่ยนวิธีฝึกอย่างอื่น

Outer stratified 5-fold × 3 repeats seed42; inner stratified3-fold seed100+outer index
เลือก mean inner Brier ต่ำสุด แล้วใช้ one-SE heuristic (SD ของ inner folds / sqrt3)
เพื่อเลือกโมเดลเรียบง่ายกว่าในช่วง best+SE: linear ก่อน quadratic และ C ต่ำก่อน C สูง
ใช้กติกานี้ทั้งในแต่ละ family และใน selected workflow; ไม่ตีความ SE เป็น statistical CI
Thresholds F2/FPR14.5% เลือกจาก inner OOF ของ candidate ที่เลือก ภายใน outer training เท่านั้น
Final shape/C/thresholds ใช้ inner CV บน development ทั้งหมด seed42 และ freeze ก่อนประเมิน Test
คงวิธี final threshold เดียวกับ calibration follow-up เพื่อให้เทียบได้ตรงกัน

รายงาน Train/outer-validation AUC/Brier เพื่อตรวจช่องว่าง overfit, TP/FP/FN/TN,
recall/precision/F2/FPR, ROC-AUC/AP, Brier และ calibration mean/slope/intercept/bins
แสดง mean/SD/range ของ folds/repeats; ไม่ถือ repeats เป็นคนเพิ่มหรือ folds เป็นอิสระ
Test bootstrap 2000 draws แบบ paired, conditional on fixed fitted models

Development screening เพื่อสนับสนุน selected workflow: Brier ลดทุก repeat,
AUC สูงขึ้นทุก repeat, mean F2 ที่ F2-policy ไม่ลด, mean budget recall ไม่ลด,
mean budget FPR ไม่เพิ่ม เทียบ none ทั้งหมด; ถ้าไม่ครบให้รายงาน trade-offs และยังไม่แทนโมเดลหลัก
การตรวจนี้ไม่ใช่ statistical proof และไม่ใช้ Test เลือก variant หรือเปลี่ยน threshold
แม้ผ่านก็เป็น research candidate; ต้องมีข้อมูลใหม่ยืนยันก่อนใช้งานจริงหรืออ้างกับคนไทย
ประวัติ feature selection และการทดลองหลายรอบยังมี selection optimism
