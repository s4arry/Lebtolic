"""Invented fixtures validate software; their scores are not NHEFS evidence."""
import unittest
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler,PolynomialFeatures
from sklearn.linear_model import LogisticRegression
from round3.cv_study import FeatureSlice
from round3.recall_study import export,model_search
from round3.predict import PortableModel


class RecallStudyTests(unittest.TestCase):
    def test_quadratic_and_linear_exports_match_sklearn(self):
        rng=np.random.default_rng(12);x=rng.normal(size=(240,5));y=(x[:,0]+rng.normal(size=240)>0).astype(int)
        x[0,4]=np.nan
        for n,degree in [(4,1),(5,1),(4,2)]:
            steps=[('features',FeatureSlice(n)),('imputer',SimpleImputer(strategy='median')),('scale',StandardScaler())]
            if degree==2:steps += [('poly',PolynomialFeatures(2,include_bias=False)),('poly_scale',StandardScaler())]
            steps += [('logistic',LogisticRegression(C=.1,max_iter=3000))]
            model=Pipeline(steps).fit(x,y)
            state=export(model,dict(features=n,degree=degree,C=.1),dict(threshold=.2),x,list('abcde'),None)
            np.testing.assert_allclose(PortableModel(state).predict_proba(x[:,:n]),model.predict_proba(x),atol=1e-12)

    def test_search_policy_is_fitted_using_its_oof_scores(self):
        rng=np.random.default_rng(7);x=rng.normal(size=(240,5));y=(x[:,0]+rng.normal(size=240)>0).astype(int)
        _,chosen,policy,p,records=model_search(x,y,42)
        self.assertEqual(len(records),9)
        self.assertEqual(policy['development_tp'],int(np.sum((p>=policy['threshold'])&(y==1))))
        self.assertLessEqual(np.sum((p>=policy['threshold'])&(y==0)),int(.145*np.sum(y==0)))


if __name__=='__main__':unittest.main()
