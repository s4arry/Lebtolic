import json
import math
from pathlib import Path

import pytest

from demo.form import (Answers, InputRejected, LB_PER_KG, PRESETS, encode, load_model,
                       predict)

ROOT = Path(__file__).resolve().parents[1]
MODEL, STATE = load_model()


def run(**kw):
    base = dict(age=50, sex='male', sbp=120, dbp=75, weight=160, weight_unit='lb', smoking='never')
    base.update(kw)
    return predict(Answers(**base), MODEL, STATE)


def test_reproduces_committed_example_output():
    expected = json.loads((ROOT/'model/example_output.json').read_text())
    example = json.loads((ROOT/'model/example_encoded.json').read_text())
    result = run()
    assert result.row == example['row']
    assert result.score == pytest.approx(expected['experimental_score'], abs=1e-12)
    assert result.alert == expected['experimental_alert']
    assert result.threshold == expected['threshold']


def test_encoding_matches_model_readme():
    row, _ = encode(Answers(age=40, sex='female', sbp=110, dbp=70, weight=70, smoking='current'))
    assert row[:4] == [40, 1.0, 110, 70]
    assert row[4] == pytest.approx(70*LB_PER_KG)
    assert row[5:] == [0.0, 1.0, 0.0, 0.0]
    assert encode(Answers(40, 'male', 110, 70, 70, smoking='former'))[0][6:] == [0.0, 1.0, 0.0]
    assert encode(Answers(40, 'male', 110, 70, 70, smoking='never'))[0][6:] == [0.0, 0.0, 0.0]


def test_kg_and_lb_give_same_score():
    assert run(weight=70, weight_unit='kg').score == pytest.approx(
        run(weight=70*LB_PER_KG, weight_unit='lb').score, abs=1e-12)


def test_missing_weight_sets_indicator_and_warns():
    result = run(weight=None)
    assert math.isnan(result.row[4]) and result.row[5] == 1.0
    assert result.warnings and 0 < result.score < 1


@pytest.mark.parametrize('kw', [
    dict(previously_diagnosed=True),
    dict(sbp=130), dict(dbp=80), dict(sbp=135, dbp=85),
    dict(sbp=70, dbp=115),                       # swapped fields
    dict(age=31), dict(age=86), dict(sbp=79, dbp=50), dict(dbp=43),
    dict(weight=68, weight_unit='lb'), dict(weight=301, weight_unit='lb'), dict(weight=200, weight_unit='kg'),
    dict(weight=70, weight_unit='stone'),
    dict(smoking='unknown'), dict(sex='other'), dict(age='abc'), dict(age=float('nan')),
])
def test_out_of_scope_inputs_are_rejected(kw):
    with pytest.raises(InputRejected):
        run(**kw)


def test_swapped_bp_gets_swap_message():
    with pytest.raises(InputRejected, match='สลับ'):
        run(sbp=70, dbp=115)


def test_range_edges_are_accepted():
    for kw in [dict(age=32), dict(age=85), dict(sbp=129, dbp=79), dict(sbp=80, dbp=44),
               dict(weight=69, weight_unit='lb'), dict(weight=300, weight_unit='lb')]:
        assert 0 < run(**kw).score < 1


def test_presets_land_low_middle_high():
    scores = {k: predict(a, MODEL, STATE) for k, a in PRESETS.items()}
    threshold = STATE['threshold_policies']['fpr145']['threshold']
    assert scores['ต่ำ'].score < 0.02
    assert 0.04 < scores['กลาง'].score < threshold and not scores['กลาง'].alert
    assert scores['สูง'].score > threshold and scores['สูง'].alert


def test_contributions_sum_to_logit():
    result = run(weight=None, smoking='current')
    logit = math.log(result.score/(1-result.score))
    assert sum(result.contributions.values()) + STATE['intercept'][0] == pytest.approx(logit, abs=1e-9)
