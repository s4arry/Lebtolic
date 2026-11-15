# นำข้อมูลภาพไปสร้างต่อใน Power BI

กราฟ PNG ที่ให้มาสร้างด้วย Python/Matplotlib **ไม่ได้ export จาก Power BI** เพราะ session นี้ไม่มี Power BI Desktop/บริการที่ใช้งานได้ ไฟล์ CSV ด้านล่างเป็นข้อมูลจริงสรุป พร้อมนำเข้า Power BI Desktop บนเครื่องที่มีโปรแกรม

| ภาพ | CSV | วิธีวางฟิลด์ |
|---|---|---|
| Class distribution | `05_dataset/dataset_counts.csv` | Clustered bar: Y=`class`, X=sum(`people`) |
| Cohort funnel | `05_dataset/cohort_funnel.csv` | Bar chart: Y=`stage`, X=sum(`people`); เรียงตามลำดับในเอกสาร ไม่เรียงตามชื่อ |
| Calibration comparison | `07_model/calibration_metrics.csv` | Filter `metric`, category=`method`, value=`mean`; `sd` คือ SD 3 repeats |
| Feature AUC | `08_model_comparison/feature_auc.csv` | Dot/bar: category=`feature_set`, value=`auc_mean`; `auc_sd` คือ SD 3 repeats ไม่ใช่ CI |
| Balancing at .5 | `09_balancing/balancing_at_05.csv` | Clustered column: axis=`method`, values=`tp`,`fp` |
| Confusion matrix | `10_test_performance/confusion_matrix.csv` | Matrix: rows=`actual`, values=`pred_no_alert`,`pred_alert` |
| ROC curve | `10_test_performance/roc_points.csv` | Line/scatter: X=`fpr`, Y=`tpr`, sort X ascending; `threshold` แถวแรกว่างเพราะเป็น infinity |

ใน Power BI ใช้ Get data → Text/CSV แล้วเลือกไฟล์ที่ต้องการ ตั้งชนิดข้อมูลเป็น decimal/whole number ตามคอลัมน์ อัตรา recall/precision/FPR ใน CSV เก็บเป็น 0–1 หากแสดงเป็นเปอร์เซ็นต์ให้ตั้ง format เป็น Percentage โดยไม่คูณซ้ำ

หมายเหตุ: แหล่งข้อมูลแต่ละภาพต่างการทดลอง ต้องคงชื่อรุ่น threshold และ Test/development ตามไฟล์ `content.md` ของหน้านั้น ไม่รวมค่าข้ามโมเดลเข้ากราฟเดียวโดยไม่ระบุ
