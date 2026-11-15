"""Invented fixtures test software behavior, never scientific performance."""
import unittest
import numpy as np
from scipy.special import expit,logit
from sklearn.model_selection import StratifiedKFold
from round3.calibration_study import fit_model,predict,select_method
from round3.balancing_study import TrainingOnlyLogistic

class CalibrationTests(unittest.TestCase):
    def setUp(self):
        rng=np.random.default_rng(91)
        self.x=rng.normal(size=(300,9));self.y=rng.binomial(1,expit(self.x[:,0]-1.5))
        self.x[::20,1]=np.nan
    def test_intercept_fits_training_oof_prevalence(self):
        state=fit_model(self.x,self.y,'intercept',42)
        scores=np.zeros(len(self.y))
        for j,(tr,va) in enumerate(StratifiedKFold(3,shuffle=True,random_state=42).split(self.x,self.y)):
            m=TrainingOnlyLogistic('none',.1,420+j).fit(self.x[tr],self.y[tr])
            scores[va]=m.predict_proba(self.x[va])[:,1]
        corrected=expit(logit(scores)-state['study_calibration']['b'])
        self.assertAlmostEqual(corrected.mean(),self.y.mean(),places=10)
    def test_calibration_preserves_ranking_and_preprocessing(self):
        raw=fit_model(self.x,self.y,'none',42);p=predict(raw,self.x)
        for method in ['intercept','sigmoid']:
            s=fit_model(self.x,self.y,method,42)
            np.testing.assert_array_equal(np.argsort(p),np.argsort(predict(s,self.x)))
            for key in ['imputer_median','scaler_mean','scaler_scale','coefficients']:
                np.testing.assert_allclose(raw[key],s[key])
            before=s['scaler_mean'].copy()
            predict(s,np.full((4,9),1000.))
            self.assertEqual(before,s['scaler_mean'])
    def test_selection_prefers_simple_if_within_one_se(self):
        r={'none':[.05]*3,'intercept':[.048,.049,.05],'sigmoid':[.0492]*3}
        self.assertEqual(select_method(r)['chosen'],'intercept')
        r={'none':[.05]*3,'intercept':[.049,.05,.051],'sigmoid':[.048,.05,.052]}
        self.assertEqual(select_method(r)['chosen'],'none')
    def test_selection_accepts_clear_gain(self):
        r={'none':[.06]*3,'intercept':[.05]*3,'sigmoid':[.04]*3}
        self.assertEqual(select_method(r)['chosen'],'sigmoid')

if __name__=='__main__':unittest.main()
