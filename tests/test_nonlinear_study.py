"""Synthetic fixtures verify transforms, training boundaries and selection."""
import unittest
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from round3.nonlinear_study import SquareTerms,fit_model,predict,choose

class NonlinearTests(unittest.TestCase):
    def setUp(self):
        rng=np.random.default_rng(141)
        self.x=rng.normal(size=(240,9));self.y=np.r_[np.zeros(200,int),np.ones(40,int)]
        self.x[::10,4]=np.nan
    def test_squares_only_named_continuous_columns(self):
        z=np.arange(18,dtype=float).reshape(2,9)
        t=SquareTerms().fit(z);out=t.transform(z)
        self.assertEqual(out.shape,(2,13));np.testing.assert_array_equal(out[:,:9],z)
        np.testing.assert_array_equal(out[:,9:],z[:,[0,2,3,4]]**2)
        powers=t.powers_;manual=np.prod(z[:,None,:]**powers[None,:,:],axis=2)
        np.testing.assert_array_equal(out,manual)
    def test_portable_replay_against_independently_fitted_expansion(self):
        x,y=self.x,self.y
        imputer=SimpleImputer(strategy='median');scale=StandardScaler()
        z=scale.fit_transform(imputer.fit_transform(x))
        expanded=np.column_stack([z,z[:,0]**2,z[:,2]**2,z[:,3]**2,z[:,4]**2])
        es=StandardScaler();expanded=es.fit_transform(expanded)
        model=LogisticRegression(C=.1,max_iter=2000,random_state=42).fit(expanded,y)
        state=fit_model(x,y,'quadratic:0.1',42)
        np.testing.assert_allclose(predict(state,x),model.predict_proba(expanded)[:,1],atol=1e-12)
        self.assertEqual(len(state['coefficients'][0]),13)
    def test_validation_does_not_change_training_preprocessing(self):
        x=self.x[:220];state=fit_model(x,self.y[:220],'quadratic:0.1',42)
        np.testing.assert_allclose(state['imputer_median'],np.nanmedian(x,axis=0))
        before=np.array(state['polynomial_mean']).copy()
        predict(state,np.full((3,9),1000.))
        np.testing.assert_array_equal(before,state['polynomial_mean'])
    def test_one_se_prefers_linear_and_stronger_regularization(self):
        configs=['linear:0.01','linear:0.1','quadratic:0.01']
        records={configs[0]:[.051]*3,configs[1]:[.049]*3,configs[2]:[.045,.05,.055]}
        self.assertEqual(choose(records,configs)['chosen'],'linear:0.1')
        records={configs[0]:[.05]*3,configs[1]:[.05]*3,configs[2]:[.05]*3}
        self.assertEqual(choose(records,configs)['chosen'],'linear:0.01')

if __name__=='__main__':unittest.main()
