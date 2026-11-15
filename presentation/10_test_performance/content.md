# 07 · TEST PERFORMANCE — 409 คน / 25 เหตุการณ์

**รุ่นอ้างอิง `befec00`, threshold 9.8034%**

| Metric | ผล Test เดิม |
|---|---:|
| ROC-AUC | 0.7034 |
| Recall | 40.00% (10/25) |
| Precision | 16.95% (10/59) |
| F2 | 0.3145 |
| FPR | 12.76% (49/384) |
| Brier | 0.057010 |
| Accuracy | 84.35% (345/409) |
| TP / FP / FN / TN | 10 / 49 / 15 / 335 |

**คำพูดประกอบ:** รุ่นนี้จับได้ 10 จาก 25 เหตุการณ์ พลาด 15 คน และเตือนผิด 49 คน จากผู้ถูกเตือน 59 คน มีเหตุการณ์จริง 10 คน Accuracy ต้องอ่านร่วมกับ recall เพราะข้อมูลส่วนใหญ่เป็น class 0 Test ไม่เคยใช้ fit โมเดลนี้ แต่เคยถูกเปิดดูซ้ำในการพัฒนาหลายรอบ จึงเป็น exploratory internal evaluation ไม่ใช่ชุดยืนยันใหม่

**ภาพ:** [confusion_matrix.png](confusion_matrix.png), [roc_curve.png](roc_curve.png) · [confusion_matrix.csv](confusion_matrix.csv), [roc_points.csv](roc_points.csv)

**แหล่งข้อมูล:** `round3/results/calibration_20261111/test_metrics.csv`, `test_scores.csv` (variant none)
