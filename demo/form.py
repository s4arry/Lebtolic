"""Turn human-readable answers into the 9 encoded columns of model/model.json.

The reference model (befec00) only exposes a numerical replay API: it expects an
already-encoded vector. This module adds the missing layer for the demo:

    human answers -> eligibility guardrails -> 9 encoded columns -> score -> alert

Nothing here retrains or changes the model. Encoding follows model/README.md and
round3/balancing_study.encode_extra exactly.
"""
from dataclasses import dataclass, field
import json
import math
from pathlib import Path

from round3.predict import PortableModel

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / 'model' / 'model.json'
POLICY = 'fpr145'
LB_PER_KG = 2.2046226218

# Development ranges (same 2,311-person development set) recorded in
# round3/results/nhefs_20261105/model.json -> training_range.
RANGES = {
    'age_years': (32.0, 85.0),
    'baseline_sbp': (80.0, 129.0),
    'baseline_dbp': (44.0, 79.0),
}
# Weight: valid measured readings per codebook (PMR5b1/b2 codes 069-300 pounds).
WEIGHT_LB_RANGE = (69.0, 300.0)

SMOKING = {
    # label: (current_smoker, former_smoker)
    'never': (0.0, 0.0),    # reference: never smoked >=100 cigarettes
    'former': (0.0, 1.0),
    'current': (1.0, 0.0),
}
SMOKING_TH = {
    'never': 'ไม่เคยสูบ (หรือสูบรวมไม่ถึง 100 มวน)',
    'former': 'เคยสูบ แต่เลิกแล้ว',
    'current': 'ปัจจุบันยังสูบอยู่',
}
FEATURE_TH = {
    'age_years': 'อายุ',
    'sex_female': 'เพศหญิง',
    'baseline_sbp': 'ความดันตัวบน (SBP)',
    'baseline_dbp': 'ความดันตัวล่าง (DBP)',
    'weight_lb': 'น้ำหนัก',
    'weight_missing': 'ไม่ทราบน้ำหนัก',
    'current_smoker': 'ยังสูบบุหรี่',
    'former_smoker': 'เคยสูบบุหรี่',
    'smoking_unknown': 'ไม่ทราบการสูบบุหรี่',
}


class InputRejected(ValueError):
    """Input is outside what the model was built for; message is Thai, user-facing."""


@dataclass
class Answers:
    age: float
    sex: str                     # 'male' | 'female'
    sbp: float                   # mmHg
    dbp: float                   # mmHg
    weight: float | None         # None = unknown
    weight_unit: str = 'kg'      # 'kg' | 'lb'
    smoking: str = 'never'       # 'never' | 'former' | 'current'
    previously_diagnosed: bool = False


@dataclass
class Result:
    score: float
    threshold: float
    alert: bool
    row: list
    warnings: list = field(default_factory=list)
    contributions: dict = field(default_factory=dict)


def load_model(path=MODEL_PATH):
    state = json.loads(Path(path).read_text())
    return PortableModel(state), state


def _number(value, name):
    try:
        value = float(value)
    except (TypeError, ValueError):
        raise InputRejected(f'{name}: ต้องเป็นตัวเลข') from None
    if not math.isfinite(value):
        raise InputRejected(f'{name}: ต้องเป็นตัวเลข')
    return value


def _check_range(value, key, name, unit):
    low, high = RANGES[key]
    if not low <= value <= high:
        raise InputRejected(
            f'{name} {value:g} {unit} อยู่นอกช่วงข้อมูลที่โมเดลเรียน ({low:g}–{high:g} {unit})')


def encode(answers: Answers):
    """Return (row, warnings). Raise InputRejected when out of scope."""
    if answers.previously_diagnosed:
        raise InputRejected('โมเดลนี้ใช้กับคนที่ยังไม่เคยได้รับการวินิจฉัยความดันสูงเท่านั้น')

    age = _number(answers.age, 'อายุ')
    sbp = _number(answers.sbp, 'ความดันตัวบน')
    dbp = _number(answers.dbp, 'ความดันตัวล่าง')

    # Typo check first (swapped fields), then cohort eligibility: the clearer reason wins.
    if dbp >= sbp:
        raise InputRejected('ความดันตัวล่างต้องน้อยกว่าตัวบน — ตรวจว่ากรอกสลับกันหรือไม่')
    if sbp >= 130 or dbp >= 80:
        raise InputRejected(
            f'ความดัน {sbp:g}/{dbp:g} สูงเกินเกณฑ์ของกลุ่มข้อมูล (ต้องต่ำกว่า 130/80) '
            '— โมเดลทำนายเฉพาะคนที่ความดันยังปกติ')
    _check_range(age, 'age_years', 'อายุ', 'ปี')
    _check_range(sbp, 'baseline_sbp', 'ความดันตัวบน', 'mmHg')
    _check_range(dbp, 'baseline_dbp', 'ความดันตัวล่าง', 'mmHg')

    if answers.sex not in ('male', 'female'):
        raise InputRejected('เพศ: เลือก ชาย หรือ หญิง')
    if answers.smoking not in SMOKING:
        raise InputRejected('การสูบบุหรี่: ต้องเลือกหนึ่งตัวเลือก '
                            '(ข้อมูลฝึกไม่มีกลุ่ม "ไม่ทราบ" ให้โมเดลเรียน)')

    warnings = []
    if answers.weight is None or answers.weight == '':
        weight_lb, weight_missing = math.nan, 1.0
        warnings.append('ไม่ได้กรอกน้ำหนัก: โมเดลแทนด้วยค่ากลางของข้อมูลฝึก (147 lb ≈ 67 kg) '
                        '— ผลจะหยาบกว่าปกติ')
    else:
        weight = _number(answers.weight, 'น้ำหนัก')
        if answers.weight_unit == 'kg':
            weight_lb = weight * LB_PER_KG
        elif answers.weight_unit == 'lb':
            weight_lb = weight
        else:
            raise InputRejected('หน่วยน้ำหนัก: เลือก kg หรือ lb')
        low, high = WEIGHT_LB_RANGE
        if not low <= weight_lb <= high:
            raise InputRejected(
                f'น้ำหนัก {weight:g} {answers.weight_unit} อยู่นอกช่วงที่ข้อมูลรองรับ '
                f'({low / LB_PER_KG:.0f}–{high / LB_PER_KG:.0f} kg หรือ {low:g}–{high:g} lb)')
        weight_missing = 0.0

    current, former = SMOKING[answers.smoking]
    row = [age, 1.0 if answers.sex == 'female' else 0.0, sbp, dbp,
           weight_lb, weight_missing, current, former, 0.0]
    return row, warnings


def contributions(state, row):
    """Per-feature contribution to the log-odds (coef * standardized value).

    Positive = pushes the score up relative to an average development person.
    Imputation is applied first, exactly as PortableModel does.
    """
    out = {}
    for j, name in enumerate(state['features']):
        value = state['imputer_median'][j] if math.isnan(row[j]) else row[j]
        z = (value - state['scaler_mean'][j]) / state['scaler_scale'][j]
        out[name] = state['coefficients'][0][j] * z
    return out


def predict(answers: Answers, model=None, state=None):
    if model is None or state is None:
        model, state = load_model()
    row, warnings = encode(answers)
    score = float(model.predict_proba([row])[0, 1])
    threshold = state['threshold_policies'][POLICY]['threshold']
    return Result(score=score, threshold=threshold, alert=score >= threshold,
                  row=row, warnings=warnings, contributions=contributions(state, row))


# Hypothetical demo profiles (not real people). Scores are computed, not assumed;
# tests check they land below / near / above the threshold.
PRESETS = {
    'ต่ำ': Answers(age=40, sex='male', sbp=100, dbp=65, weight=60, smoking='never'),
    'กลาง': Answers(age=50, sex='female', sbp=118, dbp=74, weight=70, smoking='never'),
    'สูง': Answers(age=55, sex='female', sbp=128, dbp=78, weight=100, smoking='current'),
}
