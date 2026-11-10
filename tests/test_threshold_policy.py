import unittest
import tempfile
import csv
import json
from pathlib import Path
import numpy as np
from round3.threshold_policy import select_budget_threshold, aggregate_oof, run


class ThresholdPolicyTests(unittest.TestCase):
    def test_budget_is_count_based_and_ties_are_not_split(self):
        y=np.array([0]*10+[1]*3)
        p=np.array([.1]*8+[.8]*2+[.7,.8,.9])
        chosen=select_budget_threshold(y,p,.2)
        self.assertEqual(chosen['development_fp'],2)
        self.assertEqual(chosen['development_tp'],3)
        self.assertEqual(chosen['threshold'],.7)

    def test_zero_budget_may_require_no_alerts(self):
        chosen=select_budget_threshold([0,1],[.5,.5],0)
        self.assertGreater(chosen['threshold'],.5)
        self.assertEqual(chosen['development_fp'],0)
        self.assertEqual(chosen['development_tp'],0)

    def test_selected_threshold_is_optimal_over_all_achievable_groups(self):
        rng=np.random.default_rng(4)
        y=np.array([0]*100+[1]*20);p=np.round(rng.uniform(size=120),2)
        chosen=select_budget_threshold(y,p,.2)
        for t in np.r_[np.unique(p),np.nextafter(p.max(),np.inf)]:
            fp=int(np.sum((p>=t)&(y==0)));tp=int(np.sum((p>=t)&(y==1)))
            if fp<=20:self.assertLessEqual(tp,chosen['development_tp'])

    def test_invalid_inputs_are_rejected(self):
        for y,p,budget in [([0,1],[.1,np.nan],.2),([0,0],[.1,.2],.2),([0,1],[.1,.2],-1)]:
            with self.assertRaises(ValueError):select_budget_threshold(y,p,budget)

    def test_oof_is_one_person_one_target_and_three_predictions(self):
        rows=[dict(participant_id='x',repeat=str(i),target='1',new_score=str(.1*i)) for i in range(3)]
        ids,y,p=aggregate_oof(rows)
        self.assertEqual(ids,['x']);self.assertAlmostEqual(p[0],.1)
        with self.assertRaises(ValueError):aggregate_oof(rows+rows[:1])
        with self.assertRaises(ValueError):aggregate_oof(rows[:2])

    def test_test_labels_cannot_affect_selection_and_model_parameters_are_preserved(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); source=root/'source';source.mkdir()
            with (source/'development_oof.csv').open('w',newline='') as stream:
                w=csv.writer(stream);w.writerow(['participant_id','repeat','target','new_score'])
                for i in range(12):
                    for repeat in range(3):w.writerow([f'd{i}',repeat,int(i>=10),i/20])
            old=dict(threshold=.2,coefficients=[[1,2,3,4]],features=['a','b','c','d'])
            (source/'model.json').write_text(json.dumps(old))
            def write_test(flip):
                with (source/'test_predictions.csv').open('w',newline='') as stream:
                    w=csv.writer(stream);w.writerow(['participant_id','target','new_score'])
                    for i in range(8):w.writerow([f't{i}',int(i%2==flip),i/10])
            write_test(False); first=run(source,root/'first')
            write_test(True); second=run(source,root/'second')
            self.assertEqual(first['selection'],second['selection'])
            state=json.loads((root/'second/model.json').read_text())
            self.assertEqual(state['coefficients'],old['coefficients'])
            self.assertEqual(state['features'],old['features'])


if __name__=='__main__':unittest.main()
