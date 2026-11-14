"""Prevent accidental mixing of selected parameters, policy and reported results."""
import csv
import hashlib
import json
import unittest
from pathlib import Path
import numpy as np
from round3.predict import PortableModel

ROOT=Path(__file__).resolve().parents[1]

class ReferenceModelTests(unittest.TestCase):
    def test_reference_is_exact_befec00_uncalibrated_artifact(self):
        path=ROOT/'model/model.json';state=json.loads(path.read_text())
        source=json.loads((ROOT/'round3/results/calibration_20261111/candidate_models.json').read_text())['none']
        manifest=json.loads((ROOT/'model/manifest.json').read_text())
        self.assertEqual(state,source)
        self.assertEqual(manifest['model_sha256'],hashlib.sha256(path.read_bytes()).hexdigest())
        self.assertEqual(manifest['source_commit'],'befec00faa9d341b983c984adc087ce0cf4585a8')
        self.assertEqual(state['study_calibration']['method'],'none')
        self.assertEqual(manifest['threshold'],state['threshold_policies']['fpr145']['threshold'])
        example=json.loads((ROOT/'model/example_encoded.json').read_text())
        expected=json.loads((ROOT/'model/example_output.json').read_text())
        self.assertAlmostEqual(float(PortableModel(state).predict_proba([example['row']])[0,1]),expected['experimental_score'],places=12)
    def test_primary_report_matches_saved_test_scores_at_pinned_threshold(self):
        state=json.loads((ROOT/'model/model.json').read_text());t=state['threshold_policies']['fpr145']['threshold']
        with (ROOT/'round3/results/calibration_20261111/test_scores.csv').open() as f:rows=list(csv.DictReader(f))
        y=np.array([int(r['target']) for r in rows]);p=np.array([float(r['none']) for r in rows]);pred=p>=t
        expected={'tp':int(np.sum(pred&(y==1))),'fp':int(np.sum(pred&(y==0))),
                  'fn':int(np.sum(~pred&(y==1))),'tn':int(np.sum(~pred&(y==0)))}
        reported=json.loads((ROOT/'model/metrics.json').read_text())['test']
        for k,v in expected.items():self.assertEqual(reported[k],v)
        self.assertEqual(expected,dict(tp=10,fp=49,fn=15,tn=335))
        self.assertAlmostEqual(reported['brier'],float(np.mean((p-y)**2)),places=12)

if __name__=='__main__':unittest.main()
