# หัวข้อ 4) Confusion matrix และ ROC เปรียบเทียบโมเดลที่ทดลอง

**นี่คือผลการทดลองที่ทำและบันทึกไว้แล้ว วันที่ 5–14 พฤศจิกายน 2026**
วันที่ 15 พ.ย. ตรวจคำนวณ metrics ซ้ำจาก saved Test scores ไม่ได้ฝึกข้อมูลดิบใหม่
ใช้ Test เดิม 409 คน: เกิดเหตุการณ์25 ไม่เกิดเหตุการณ์ตามนิยาม384
เป้าหมายคือรายงานการวินิจฉัยความดันสูงครั้งแรกใน8ปี ไม่ใช่การวัดความดันปีที่8

## 4.1 ตารางเปรียบเทียบ ROC-AUC ของโมเดลที่มี saved Test predictions

AUC ยิ่งสูงยิ่งแยกกลุ่มได้ดี; Brier ยิ่งต่ำยิ่งดี; ทั้งสองไม่ขึ้นกับ threshold
4 ตัวแปร = อายุ เพศ SBP DBP; 5 ตัวแปร legacy เพิ่มชั่วโมงนอน; 6 ตัวแปรเพิ่มน้ำหนัก/สูบบุหรี่
ตารางนี้มี12 fitted score variants; selected workflow ที่ได้คะแนนซ้ำกับรุ่นเดิมไม่ลงเป็นโมเดลใหม่ซ้ำ
20 configurations ใน nested CV รอบแรกและ grids ในรอบอื่นเป็น candidates ภายใน development
ไม่ได้มี Test predictions แยกทุก configuration จึงไม่สร้าง Test AUC ให้รุ่นที่ไม่มีหลักฐาน

| รหัส | ทดลองเมื่อ | โมเดล/วิธี | Test ROC-AUC ↑ | Test Brier ↓ | สถานะ/เหตุผล |
|---|---|---|---:|---:|---|
| M01 | 5 พ.ย. / เทียบซ้ำ 14 พ.ย. | Legacy 5 ตัวแปร + Platt | 0.6761 | 0.05653 | baseline |
| M02 | 5 พ.ย. | LR 4 ตัวแปร C=.1 | 0.7054 | 0.05579 | รุ่น F2 เดิม |
| M03 | 10 พ.ย. | LR 4 ตัวแปร C=1 | 0.7057 | 0.05577 | ไม่แทนรุ่นเดิม |
| M04 | 11 พ.ย. | LR 4 ตัวแปร + class weights | 0.7092 | 0.21491 | ไม่เลือก balancing |
| M05 | 11 พ.ย. | LR 4 ตัวแปร + oversampling | 0.7151 | 0.21469 | ไม่เลือก balancing |
| M06 | 11 พ.ย. | LR 4 ตัวแปร + น้ำหนัก | 0.6971 | 0.05710 | candidate |
| M07 | 11 พ.ย. | LR 4 ตัวแปร + สูบบุหรี่ | 0.7254 | 0.05562 | candidate; ไม่เลือกจาก Test AUC |
| M08 | 11 พ.ย. / จัดเป็น reference 14 พ.ย. | LR + น้ำหนักและสูบบุหรี่ C=.1 | 0.7034 | 0.05701 | รุ่นที่ demo/สไลด์ใช้อยู่ |
| M09 | 11 พ.ย. | M08 + intercept calibration | 0.7034 | 0.05701 | ไม่เพิ่ม calibration |
| M10 | 11 พ.ย. | M08 + sigmoid calibration | 0.7034 | 0.05684 | ไม่เพิ่ม calibration |
| M11 | 11 พ.ย. | LR 6 ตัวแปร เลือก C แบบ nested (final .01) | 0.6896 | 0.05647 | ไม่ผ่าน development screening |
| M12 | 11 พ.ย. | Quadratic LR เลือก C แบบ nested | 0.7123 | 0.05664 | ไม่ผ่าน development screening |

**รุ่นที่ใช้นำเสนอจาก demo ปัจจุบันคือ M08** ไม่ใช่รุ่นที่ Test AUC สูงสุด
M07 มี Test AUC สูงสุดในตาราง (.7254) แต่ห้ามเลือกย้อนหลังจาก Test
M08 ถูกจัดเป็น research reference ตามประวัติ branch; ไม่ได้พิสูจน์ว่าดีที่สุดทุก metric
Balancing ทำให้ Brier สูงขึ้นมากในการทดลองนี้ จึงไม่ได้เลือกใช้ แม้ Test AUC เพิ่มเล็กน้อย

## 4.2 Confusion matrix ของรุ่นอ้างอิงปัจจุบัน (M08)

Logistic C=.1, ไม่ balancing/ไม่เพิ่ม calibration, threshold=.09803399686084495
เลือก threshold จาก development inner OOF ภายใต้งบ FPR14.5% งบนี้ไม่ใช่ threshold=.145

| ผลจริงตามนิยามติดตาม ↓ / ผลทำนาย → | ไม่แจ้งเตือน | แจ้งเตือน | รวม |
|---|---:|---:|---:|
| ไม่เกิดเหตุการณ์ | TN=335 | FP=49 | 384 |
| เกิดเหตุการณ์ | FN=15 | TP=10 | 25 |
| รวม | 350 | 59 | 409 |

| ตัวชี้วัด | ค่า |
|---|---:|
| ROC-AUC | .7034 |
| Recall = 10/25 | 40.00% |
| Precision = 10/59 | 16.95% |
| Specificity = 335/384 | 87.24% |
| FPR = 49/384 | 12.76% |
| Accuracy = 345/409 | 84.35% |
| F2 | .3145 |
| Brier | .05701 |

FP=เตือนแต่ไม่พบเหตุการณ์ตามนิยามติดตาม; FN=ไม่เตือนทั้งที่พบเหตุการณ์
ไม่ใช่หลักฐานว่า FP จะไม่มีความดันสูงตลอดชีวิต
Accuracy สูงส่วนหนึ่งเพราะ non-events มาก จึงต้องดู recall/precision ควบคู่

## 4.3 เปลี่ยน threshold แล้ว confusion matrix เปลี่ยนอย่างไร

แถวที่เป็น M02 ใช้ score model เดียวกัน AUC=.7054/Brier=.05579 คงเดิม
แถว M08 ใช้ AUC=.7034/Brier=.05701 คงเดิม
F2 ให้น้ำหนัก recall มากขึ้น; FPR-policy ลดคำเตือนผิดโดยยอมพลาดเหตุการณ์เพิ่ม

| รุ่น/รอบเลือก threshold | Policy | Threshold | TP | FN | FP | TN | Recall | Precision |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| M01 legacy จริง | default | .500000 | 0 | 25 | 0 | 384 | 0% | ไม่กำหนด |
| M02 วันที่5พ.ย. | F2 เดิม | .041655 | 23 | 2 | 250 | 134 | 92% | 8.42% |
| M02 วันที่10พ.ย. | FPR20% | .084889 | 9 | 16 | 58 | 326 | 36% | 13.43% |
| M02 วันที่10พ.ย. | FPR14.5% | .091868 | 8 | 17 | 45 | 339 | 32% | 15.09% |
| M02 รอบ balancing 11พ.ย. | F2 inner OOF | .059544 | 16 | 9 | 143 | 241 | 64% | 10.06% |
| M02 รอบ balancing 11พ.ย. | FPR14.5% inner OOF | .092858 | 8 | 17 | 42 | 342 | 32% | 16.00% |
| M08 รอบ fallback ก่อน calibration study | FPR14.5% repeated outer OOF | .094595 | 13 | 12 | 52 | 332 | 52% | 20.00% |
| M08 ปัจจุบัน | F2 inner OOF | .060787 | 15 | 10 | 119 | 265 | 60% | 11.19% |
| M08 ปัจจุบัน | FPR14.5% inner OOF | .098034 | 10 | 15 | 49 | 335 | 40% | 16.95% |
| M08 ปัจจุบัน | default .5 | .500000 | 0 | 25 | 0 | 384 | 0% | ไม่กำหนด |

จุดตัดจาก averaged repeated outer OOF กับ inner OOF ต่าง workflow จึงได้ค่าต่างกัน
ไม่ใช่การเปลี่ยน threshold โดยดู Test; Test เดิมถูกเปิดเผยหลายครั้งอยู่แล้วจึงเป็น exploratory

## 4.4 สรุปสำหรับพูดหน้าชั้น

“เราได้ทดลอง Logistic Regression หลายชุดตัวแปร การปรับสมดุลข้อมูล การ calibration
และพจน์กำลังสอง ตารางนี้ใช้ Test เดิม409คน รุ่นที่ demo ปัจจุบันใช้อยู่มี ROC-AUC .7034
และที่ threshold .098034 จับเหตุการณ์ได้10จาก25คน พลาด15คน และเตือนผิด49คน
เราไม่ได้เลือกรุ่นจาก Test AUC สูงสุด เพราะเลือกวิธีจาก development ก่อนประเมิน Test
ผลยังไม่ยืนยันว่าดีกว่าทุกด้าน และยังไม่ผ่านการตรวจสอบกับประชากรไทย”

## แหล่งตัวเลขและขอบเขต

- ตาราง CSV พร้อม path/column ของคะแนน: [TOPIC4_MODEL_COMPARISON.csv](TOPIC4_MODEL_COMPARISON.csv)
- [main เทียบ reference](results/main_comparison_20261111/test_metrics.csv)
- [4-feature/F2 เดิม](results/nhefs_20261105/metrics.json)
- [FPR20%](results/threshold_20261110/metrics.json), [FPR14.5%](results/threshold_20261110_fpr145/metrics.json)
- [Recall C=1](results/recall_20261110/metrics.json)
- [Balancing/เพิ่ม features](results/balancing_20261111/test_metrics.csv)
- [Calibration](results/calibration_20261111/test_metrics.csv)
- [Nonlinear](results/nonlinear_20261111/test_metrics.csv)
- [ไดอารี่/เหตุผล/commit](EXPERIMENT_DIARY_TH.md)

Test เคยเปิดดูหลายรอบ ผลเป็น exploratory internal evaluation; ไม่ใช่การฝึก/ทดสอบใหม่วันนี้
M02 Brier ดีกว่า legacy เล็กน้อยแต่ primary paired CI คร่อมศูนย์
M08 มี Brier แย่กว่า legacy จึงไม่อ้างว่าดีที่สุดทุกมิติ
ไม่มี external/Thai validation; ไม่ใช่โมเดลวินิจฉัยหรือความเสี่ยงที่ยืนยันแล้วสำหรับคนไทย
