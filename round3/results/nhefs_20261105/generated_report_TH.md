# ผลการทดลอง NHEFS: repeated nested cross-validation

ยังไม่มีหลักฐานเพียงพอว่า Brier ดีขึ้นจริงตามเกณฑ์ที่กำหนด

นี่เป็น internal validation บน Test ที่เคยเปิดเผยแล้ว ไม่ใช่ external validation หรือการยืนยันกับคนไทย

Development 2311 คน / 141 เหตุการณ์; Test 409 คน / 25 เหตุการณ์

| Metric | โมเดลเดิม threshold 0.5 | โมเดลใหม่ threshold OOF |
|---|---:|---:|
| roc_auc | 0.6761458333333333 | 0.7054166666666667 |
| average_precision | 0.12294096861198632 | 0.13192156156018403 |
| brier_score | 0.056529185516960026 | 0.055794904802818256 |
| log_loss | 0.2227443553335793 | 0.21668041925552595 |
| recall | 0.0 | 0.92 |
| precision | None | 0.08424908424908426 |
| specificity | 1.0 | 0.3489583333333333 |

การเปรียบเทียบ recall ข้างบนรวมผลจาก threshold ที่ต่างกัน ต้องอ่าน test_legacy_with_same_threshold_policy และ test_new_at_0_5 ใน metrics.json ประกอบ

ผลต่าง paired bootstrap (ค่าบวกหมายถึงดีขึ้นตามทิศทางที่ระบุ):

```json
{
  "brier_old_minus_new": {
    "ci95": [
      -6.215657078143886e-05,
      0.0016821509447153755
    ],
    "valid_draws": 2000
  },
  "auc_new_minus_old": {
    "ci95": [
      0.0022733569712468675,
      0.06167967812124841
    ],
    "valid_draws": 2000
  },
  "ap_new_minus_old": {
    "ci95": [
      -0.008383520933377034,
      0.0349369312027765
    ],
    "valid_draws": 2000
  },
  "point_differences": {
    "brier_old_minus_new": 0.000734280714141766,
    "auc_new_minus_old": 0.029270833333333357,
    "ap_new_minus_old": 0.00898059294819771
  },
  "primary_supported": false,
  "scope": "conditional on fixed fitted models and historical Test; no training uncertainty or external validation"
}
```

ระดับคะแนน:

```json
[
  {
    "level": 1,
    "n": 155,
    "events": 3,
    "observed_rate": 0.01935483870967742,
    "observed_rate_95ci_wilson": [
      0.0066038767216294765,
      0.0553538696612183
    ],
    "mean_score": 0.03019499157193172
  },
  {
    "level": 2,
    "n": 134,
    "events": 7,
    "observed_rate": 0.05223880597014925,
    "observed_rate_95ci_wilson": [
      0.025532002054082972,
      0.1039026322956776
    ],
    "mean_score": 0.05493090981427678
  },
  {
    "level": 3,
    "n": 120,
    "events": 15,
    "observed_rate": 0.125,
    "observed_rate_95ci_wilson": [
      0.07723497273395374,
      0.1960294020800739
    ],
    "mean_score": 0.09505171671122259
  }
]
```

ค่าเฉลี่ย/SD ของแต่ละ fold ใช้แสดงความแกว่งเท่านั้น folds และ repeats ไม่เป็นอิสระต่อกัน

- Outer legacy roc_auc: mean=0.645982, SD=0.045786
- Outer new roc_auc: mean=0.642611, SD=0.047815
- Outer legacy average_precision: mean=0.122421, SD=0.036445
- Outer new average_precision: mean=0.119015, SD=0.038350
- Outer legacy brier_score: mean=0.056695, SD=0.001012
- Outer new brier_score: mean=0.056492, SD=0.001028

ข้อจำกัด: การเลือก 20 candidates, ข้อมูลย้อนหลัง, reported diagnosis, จำนวนเหตุการณ์น้อย และ conservative death guard ยังต้องยืนยัน codebook

คะแนนทดลองของกลุ่ม NHEFS สหรัฐฯ ในอดีต สำหรับการรายงานวินิจฉัยใน 8 ปี ยังไม่ยืนยันกับคนไทย ไม่ใช่การวินิจฉัยหรือคำแนะนำการรักษา ระดับ 1 ไม่ได้แปลว่าไม่มีโอกาสเกิดเหตุการณ์