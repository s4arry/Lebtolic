# ประวัติการทดลอง — ไม่ใช่รุ่นอ้างอิงทั้งหมด

รุ่นอ้างอิงเพียงรุ่นเดียวของ branch นี้อยู่ที่ [model/](model/README.md)
เลือกตามคำขอผู้ใช้จาก `befec00`, key `none` ใน calibration follow-up
ไม่เลือก sigmoid หรือ nonlinear จากผล Test ที่ดูดีขึ้นบางค่า

| ลำดับ | การทดลอง/หลักฐาน | สถานะ |
|---|---|---|
| 1 | [ผล 5 พ.ย.](round3/results/nhefs_20261105/README_TH.md) | เปรียบเทียบ legacy กับ 4-feature selected workflow; ประวัติ |
| 2 | [FPR 20%](round3/results/threshold_20261110/README_TH.md) / [FPR 14.5%](round3/results/threshold_20261110_fpr145/README_TH.md) | นโยบายของโมเดล 4-feature เดิม ไม่ใช่รุ่นน้ำหนัก+สูบบุหรี่ |
| 3 | [Recall study](round3/results/recall_20261110/README_TH.md) | การทดลองหลาย alert budgets; ประวัติ |
| 4 | [Balancing + fallback](round3/results/balancing_20261111/README_TH.md) | เพิ่มน้ำหนัก+สูบบุหรี่; final threshold มาจาก repeated outer OOF |
| 5 | [Calibration follow-up](round3/results/calibration_20261111/README_TH.md) | **ต้นทางรุ่นอ้างอิง: none, C=.1, final threshold จาก inner OOF** |
| 6 | [Nonlinear follow-up](round3/results/nonlinear_20261111/README_TH.md) | ไม่ผ่าน development improvement screening; ไม่แทนรุ่นอ้างอิง |
| 7 | [เทียบ main จริง](round3/results/main_comparison_20261111/README_TH.md) | AUC สูงขึ้นแต่ Brier แย่ลง; ยังไม่ยืนยัน superiority |

## เอกสารประวัติ

- [CHANGES_TH](round3/CHANGES_TH.md), [RESULTS_STATUS_TH](round3/RESULTS_STATUS_TH.md): บันทึกตามลำดับเวลา
- [REVIEW_TH](round3/REVIEW_TH.md), [CV_PROTOCOL_TH](round3/CV_PROTOCOL_TH.md): ทบทวนและ protocol รุ่นแรก
- [Balancing protocol](round3/BALANCING_PROTOCOL_TH.md), [Codebook](round3/CODEBOOK_BALANCING_TH.md)
- [Calibration protocol](round3/CALIBRATION_PROTOCOL_TH.md), [Nonlinear protocol](round3/NONLINEAR_PROTOCOL_TH.md)
- [Attempt ที่เคยล้มเหลว](round3/results/nested_cv_attempt_01/study_status.json): เก็บตามจริง ไม่ใช่ผลฝึกสำเร็จ

ไม่ย้าย/ลบ source modules หรือ result artifacts ที่เชื่อมกันเพื่อให้ commands และ hashes
ในหลักฐานเดิมยังตรวจได้ ประวัติทุก commit ยังคงอยู่ ไม่ force-push หรือ rewrite history
