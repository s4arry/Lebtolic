# 03 · DATASET — 2 classes · 2,720 คน · event 6.1%

**ข้อความบนสไลด์**

- Class 0: **2,554 คน (93.9%)** — มีข้อมูลสนับสนุนว่าไม่มีรายงานวินิจฉัยภายใน 8 ปี
- Class 1: **166 คน (6.1%)** — รายงานการวินิจฉัยครั้งแรกภายใน 8 ปี
- Development **2,311 / 141 events**; historical test **409 / 25 events**
- คนที่ผลไม่แน่ใจหรือข้อมูลติดตามไม่เพียงพอถูกคัดออก ไม่ถือเป็น class 0

**คำพูดประกอบ:** ความไม่สมดุลทำให้ทายว่าไม่เกิดทุกคนก็ถูก 93.9% แต่จับเหตุการณ์ไม่ได้เลย ต้องดู recall, precision และ false positives ร่วมกัน ข้อมูลไม่ได้ยืนยันว่า class 0 ไม่เคยเป็นความดันสูงตลอดชีวิต

**ข้อมูลทำกราฟ:** [dataset_counts.csv](dataset_counts.csv) · ภาพ [class_distribution.png](class_distribution.png), [cohort_funnel.png](cohort_funnel.png)

**แหล่งข้อมูล:** `round3/results/nhefs_20261105/data_audit.json`, `round3/results/calibration_20261111/provenance.json`
