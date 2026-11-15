# Calibration follow-up — protocol before results

ต่อจาก commit 995ff23 ใช้ cohort, split และ outer 5 folds × 3 repeats เดิม
รุ่นตั้งต้นคือ weight_smoking: Logistic C=.1, ไม่มี balancing, 9 encoded columns
ตรวจ hashes ของข้อมูลเดิมครบก่อนเริ่ม ไม่เพิ่มฟีเจอร์หรือเปลี่ยน target ในรอบนี้

สมมติฐาน: intercept correction อาจแก้ระดับความเสี่ยงโดยรวม; sigmoid อาจแก้ทั้ง
ระดับและความชัน แต่ development เดิมมี calibration slope ใกล้ 1 อยู่แล้ว จึงอาจไม่มี
ประโยชน์เพิ่ม การ calibration ไม่รับประกัน AUC/recall ดีขึ้น โดยการแปลงแบบ monotonic
จะรักษาลำดับคะแนนของ final model ไว้

วิธี: none, intercept-only, sigmoid (Platt with smoothed targets).
ไม่ใช้ isotonic เนื่องจากเหตุการณ์เพียง 141 คนและจำนวนใน training ย่อยยิ่งน้อย
ใช้ sklearn _SigmoidCalibration ตามเวอร์ชันที่ pin ไว้ และส่งออก a/b สำหรับ replay
sigmoid ต้องมี slope เป็นบวก (a<0 ใน sklearn convention) มิฉะนั้นหยุดอย่างเปิดเผย

ในแต่ละ outer training: inner stratified 3-fold เปรียบเทียบ mean Brier
แต่ละ inner training สร้าง calibration OOF ด้วย stratified 3-fold อีกชั้น
ทุก imputer/scaler/base model เรียนจาก training ของชั้นนั้นเท่านั้น
fit calibrator จาก calibration OOF scores และ labels ใน training เท่านั้น
แล้ว refit base บน training ทั้งหมดก่อนทำนาย validation
เลือกวิธีด้วย one-standard-error heuristic: best mean Brier + SD ของ 3 inner folds / sqrt(3)
เลือก none ก่อน intercept ก่อน sigmoid หากอยู่ในช่วงนี้ ไม่ตีความ SE เป็น inferential CI
เลือก threshold F2/FPR14.5% จาก inner OOF ของวิธีที่เลือกก่อนทำนาย outer validation
รายงานทั้ง fixed candidates และ selected workflow; ไม่เลือกจาก Test

Final selection ใช้กติกาเดียวกันบน development ทั้งหมด และ final thresholds ใช้ inner OOF
ของวิธีนั้น (ต่างจากรอบก่อนที่ใช้ mean repeated outer OOF) บันทึก decision/model ก่อน Test
รายงาน probability metrics, calibration, TP/FP/FN/TN, recall/precision/F2/FPR ทุก fold/repeat
และ Test; bootstrap Test 2000 draws conditional on fitted models

เกณฑ์สนับสนุน calibration เพิ่ม: selected workflow ต้องเลือก calibrated method ใน final
และมี Brier ต่ำกว่า uncalibrated ทุก repeat โดย budget recall เฉลี่ยไม่ลดและ FPR เฉลี่ย
ไม่เพิ่ม เกณฑ์ screening นี้ไม่ใช่หลักฐานเชิงสถิติ; ไม่มีผ่านให้คง uncalibrated research candidate
โมเดลหลัก 4 features ไม่ถูกแทนอัตโนมัติ Test เดิมถูกเปิดดูหลายครั้ง เป็น exploratory เท่านั้น
การเลือก feature set ในรอบก่อนยังเป็น selection history ที่ nested รอบนี้ลบไม่ได้
