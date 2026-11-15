# การทดลอง balancing และแผนสำรอง (11 พฤศจิกายน 2026)

กำหนดวิธีก่อนรันการเปรียบเทียบใหม่ ต่อจาก commit `4f45d68` โดยไม่เปลี่ยน target,
cohort, historical Train/Tune/Calibration/Test membership หรือโมเดลใช้งานปัจจุบัน
ข้อมูลจริงดาวน์โหลดจาก CDC และต้องมี SHA-256 ตรงกับทั้ง 8 ไฟล์ในหลักฐานรอบก่อน
หากไม่ตรงให้หยุด ไม่แทนข้อมูลด้วยข้อมูลจำลอง

## คำถามและการแบ่งข้อมูล

ทำนาย **การรายงานว่าได้รับการวินิจฉัยครั้งแรกภายใน 8 ปี** ใน selected historical
NHEFS cohort ไม่ใช่โรคที่ยืนยันจากการวัด และไม่ใช่คำยืนยันว่าไม่เคยเป็นโรคตลอดชีวิต
รักษา cohort 2,720/166 events; development 2,311/141; historical Test 409/25
แต่ตรวจจำนวนจากข้อมูลจริงอีกครั้ง และตรวจ reproduction ของคะแนนเดิม
Test เคยถูกเปิดดูหลายครั้งแล้ว เป็น exploratory internal evaluation เท่านั้น

## การทดลองหลัก

Logistic Regression 4 features (age, sex, SBP, DBP), L2, **C=0.1 คงที่** ตามโมเดล
ที่เลือกไว้ก่อนหน้า เปรียบเทียบ none / `class_weight='balanced'` / random oversampling
โดยเพิ่มเฉพาะ positive training rows แบบ replacement จนจำนวนเท่ากับ negatives
ไม่ใช้ SMOTE ไม่เปลี่ยนสมาชิก validation/test และไม่เพิ่มข้อมูลใหม่จริง

Imputer median และ StandardScaler เรียนจาก training rows ก่อน resampling เท่านั้น
balanced weights คำนวณจาก y ของ training fold เท่านั้น; RNG seed บันทึกทุก fold
oversampling ทำหลัง preprocessing และทำเฉพาะใน fit
คง C เดียวกันเพื่อเทียบวิธีฝึกกับโมเดลที่เลือกไว้ ไม่ค้นหา hyperparameter เพิ่มในรอบนี้
การเพิ่มจำนวนแถวทำให้ effective L2 penalty ต่างจาก weighting ได้ จึงรายงานเป็น
การเปรียบเทียบ implementation ทั้งวิธี ไม่อ้างว่าต่างกันเพราะ class ratio อย่างเดียว

## Repeated nested validation รวมการเลือก threshold

- Outer stratified 5 folds × 3 repeats, seed=42; fold membership ตรงกันทุกวิธี
- ภายใน outer training สร้าง inner stratified 3-fold OOF, seed=100+outer_index
- ใช้ inner OOF เลือก (a) max F2, ties: precision แล้ว threshold สูงกว่า และ
  (b) recall สูงสุดภายใต้ empirical FPR ≤ 0.145, ties: FP ต่ำกว่า แล้ว threshold สูงกว่า
- ฝึกบน outer training ทั้งหมด แล้วใช้ threshold ที่ freeze แล้วกับ outer validation
- นี่คือ nested **threshold selection**; ไม่มี hyperparameter search ในการทดลองหลัก
- การเลือก threshold ใช้ข้อมูลพัฒนาเท่านั้น; FPR 14.5% เป็นข้อจำกัดเชิงประจักษ์ในชุด
  ที่เลือก threshold ไม่รับประกัน FPR ใน outer validation, Test หรือประชากรใหม่
- Final thresholds เลือกจากค่าเฉลี่ย repeated outer OOF ต่อคน จากนั้น fit development
  ทั้งหมดและ freeze ผลการตัดสิน/thresholds ลงไฟล์ ก่อนคำนวณผล Test ใหม่

รายงาน TP/FP/FN/TN, recall, precision, F2, FPR, ROC-AUC, AP, Brier, calibration
(mean probability, observed rate, calibration-in-the-large intercept, intercept/slope
และ reliability bins) ราย fold และ pooled ต่อ repeat ทุกคนปรากฏหนึ่งครั้งต่อ repeat
รายงาน mean/SD/range ของ 15 folds และ 3 repeats ไม่ถือ folds/repeats เป็นอิสระ
ไม่สร้าง naive CI จากจำนวน folds; Test uncertainty เป็น paired person bootstrap
2,000 รอบของ fixed predictions ไม่ครอบคลุม training/selection uncertainty
คะแนน balanced/oversampled ไม่ผ่าน probability calibration อาจมี prevalence shift
จึงต้องอ่าน Brier/calibration ไม่ตีความคะแนนดิบเป็นความเสี่ยงที่เชื่อถือได้

## เกณฑ์เปลี่ยนโมเดลและการเริ่มแผนสำรอง

ใช้ development เท่านั้น: candidate ต้องมี Brier ต่ำกว่า, F2 ณ F2-policy สูงกว่า,
recall ณ FPR-policy สูงกว่า **ทุก repeat** และ mean outer FPR ของ budget-policy
ต้องไม่สูงกว่า baseline จึงเป็น candidate ที่ควรพิจารณาต่อ ไม่ใช่ statistical proof
เกณฑ์นี้ตั้งอย่างอนุรักษ์นิยมเพื่อคงโมเดลเดิมถ้าผลต่างไม่สม่ำเสมอ
หากไม่มี balancing method ผ่าน ให้รันแผนสำรองทันทีโดยไม่เลือกจาก Test

แผนสำรองที่กำหนดก่อนดูผล balancing:

1. วิเคราะห์ missed events จาก outer validation เท่านั้น แยกสอง threshold policies
   และแสดงจำนวนที่พลาด 0/1/2/3 repeats กับสรุป baseline features และข้อมูลหาย
2. ตรวจ exclusion reasons และ baseline feature missingness ในผู้เข้าร่วม/ถูกคัดออก
   ไม่ตั้ง target=0 ให้ผู้ที่ไม่มีข้อมูลผลลัพธ์ และไม่เติม missing outcomes
3. ตรวจ codebook 1982 และ SAS layout ก่อนเพิ่มตัวแปร; จำกัดการทดลองไว้ที่
   measured weight (PMR5b1/PMR5b2) และ smoking history (L1/L2a)
4. เปรียบเทียบ +weight, +smoking, +both กับ 4-feature baseline บน folds เดิม
   ทุก variant ใช้ C=.1, ไม่ balancing และ nested thresholds ตามกติกาเดียวกัน
   Median ของ weight เรียนจาก training fold; unknown smoking เป็นหมวดแยก
5. ไม่เรียก weight ว่า BMI เพราะไม่มี height ที่ตรวจยืนยันสำหรับ baseline นี้
   ไม่ใช้ตัวแปรจาก follow-up หลังเวลาเริ่มทำนาย
6. เลือก candidate ที่ผ่านเกณฑ์เดียวกันด้วย mean development Brier ต่ำสุด
   หากไม่มีผ่าน ให้คงโมเดล/นโยบายเดิม ไม่เลือกจากผล Test

การทดลอง fallback เป็น adaptive/exploratory ต่อจากงานหลายรอบ ต้องมีข้อมูลใหม่
เพื่อยืนยันผล แม้ development ดีขึ้นก็ไม่ถือว่ายืนยันการใช้กับคนไทย
