"""Exploratory recall development at fixed FPR budget; historical Test is exposed."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, StratifiedKFold, RepeatedStratifiedKFold, cross_val_predict
from sklearn.metrics import make_scorer, roc_auc_score

from .cv_study import FeatureSlice
from .threshold_policy import select_budget_threshold, operating_metrics, read_csv, aggregate_oof
from .workflow import save_json, load_original, level_cuts, band_report, DISCLAIMER

PROTOCOL = dict(max_fpr=.145, outer_folds=5, repeats=3, inner_folds=3, seed=42,
                objective='mean inner standardized partial ROC-AUC at max_fpr=0.145',
                candidates='linear 4/5 features and quadratic 4 features; C=.01/.1/1; no calibration',
                selection='all feature scaling, transformations, model and threshold choices inside development',
                interpretation='exploratory follow-up on previously exposed Test; not independent confirmation')


def model_search(x, y, seed):
    models, configs = [], []
    for count, degree in [(4,1),(5,1),(4,2)]:
        for c in [.01,.1,1.]:
            steps=[('features',FeatureSlice(count)),('imputer',SimpleImputer(strategy='median')),
                   ('scale',StandardScaler())]
            if degree==2:
                steps += [('poly',PolynomialFeatures(degree=2,include_bias=False)),('poly_scale',StandardScaler())]
            steps += [('logistic',LogisticRegression(C=c,max_iter=3000,random_state=42))]
            models.append(Pipeline(steps)); configs.append(dict(features=count,degree=degree,C=c))
    search=GridSearchCV(Pipeline([('model',models[0])]),{'model':models},
        scoring=make_scorer(roc_auc_score,response_method='predict_proba',max_fpr=.145),
        cv=StratifiedKFold(3,shuffle=True,random_state=seed),n_jobs=1,error_score='raise')
    search.fit(x,y)
    records=[{**config,'mean_inner_partial_auc':float(search.cv_results_['mean_test_score'][i])}
             for i,config in enumerate(configs)]
    model=search.best_estimator_.named_steps['model']
    p=cross_val_predict(model,x,y,cv=StratifiedKFold(3,shuffle=True,random_state=seed),method='predict_proba')[:,1]
    policy=select_budget_threshold(y,p,.145)
    return model,records[search.best_index_],policy,p,records


def export(model,selected,policy,x,features,cuts):
    n=selected['features'];lr=model.named_steps['logistic']
    state=dict(selected={**selected,'family':'logistic','calibration':'none'},features=features[:n],
        training_range={f:[float(np.nanmin(x[:,i])),float(np.nanmax(x[:,i]))] for i,f in enumerate(features[:n])},
        imputer_median=model.named_steps['imputer'].statistics_.tolist(),
        scaler_mean=model.named_steps['scale'].mean_.tolist(),scaler_scale=model.named_steps['scale'].scale_.tolist(),
        coefficients=lr.coef_.tolist(),intercept=lr.intercept_.tolist(),study_calibration={'method':'none'},
        threshold=policy['threshold'],threshold_policy=policy,
        level_cuts=cuts.tolist() if cuts is not None else None,limitations=DISCLAIMER)
    if 'poly' in model.named_steps:
        state.update(polynomial_powers=model.named_steps['poly'].powers_.tolist(),
                     polynomial_mean=model.named_steps['poly_scale'].mean_.tolist(),
                     polynomial_scale=model.named_steps['poly_scale'].scale_.tolist())
    return state


def run(source,output):
    source,output=Path(source),Path(output)
    if output.exists() and any(output.iterdir()):raise ValueError('Use a new output directory')
    output.mkdir(parents=True,exist_ok=True);save_json(output/'protocol.json',PROTOCOL)
    features=load_original()['FEATURES']
    cohort={r['participant_id']:r for r in read_csv(source/'outputs/NHEFS_Future_8y_Cohort.csv')}
    membership=read_csv(source/'split_manifest.csv')
    dev=[r for r in membership if r['split']!='test'];test=[r for r in membership if r['split']=='test']
    def arrays(rows):
        return np.array([[float(cohort[r['participant_id']][f]) for f in features] for r in rows]),np.array([int(r['target']) for r in rows])
    xd,yd=arrays(dev)
    ids,old_y,old_p=aggregate_oof(read_csv(source/'development_oof.csv'))
    old_mapping={ident:(int(y),float(p)) for ident,y,p in zip(ids,old_y,old_p)}
    old_p=np.array([old_mapping[r['participant_id']][1] for r in dev])
    assert all(old_mapping[r['participant_id']][0]==int(r['target']) for r in dev)
    # Comparator trained independently in each outer training set; threshold from its inner OOF.
    folds=[];oof=np.full((3,len(dev)),np.nan);baseline_oof=np.full_like(oof,np.nan)
    outer=RepeatedStratifiedKFold(n_splits=5,n_repeats=3,random_state=42)
    for i,(train,valid) in enumerate(outer.split(xd,yd)):
        model,chosen,policy,_,_=model_search(xd[train],yd[train],100+i)
        p=model.predict_proba(xd[valid])[:,1];oof[i//5,valid]=p
        base=Pipeline([('features',FeatureSlice(4)),('imputer',SimpleImputer(strategy='median')),
                       ('scale',StandardScaler()),('logistic',LogisticRegression(C=.1,max_iter=3000))])
        bp=cross_val_predict(base,xd[train],yd[train],cv=StratifiedKFold(3,shuffle=True,random_state=100+i),method='predict_proba')[:,1]
        bpolicy=select_budget_threshold(yd[train],bp,.145);base.fit(xd[train],yd[train])
        pv=base.predict_proba(xd[valid])[:,1];baseline_oof[i//5,valid]=pv
        folds.append(dict(repeat=i//5,fold=i%5,selected=chosen,threshold=policy,
                          candidate=operating_metrics(yd[valid],p,policy['threshold']),
                          comparator=operating_metrics(yd[valid],pv,bpolicy['threshold'])))
        print(f'Outer fold {i+1}/15 complete',flush=True)
    assert np.isfinite(oof).all() and np.isfinite(baseline_oof).all()
    model,chosen,policy,final_oof,records=model_search(xd,yd,42)
    try:cuts=level_cuts(final_oof)
    except ValueError:cuts=None
    state=export(model,chosen,policy,xd,features,cuts)
    save_json(output/'model.json',state);save_json(output/'selection.json',dict(selected=chosen,policy=policy,candidates=records))
    with (output/'development_selection_scores.csv').open('w',newline='') as f:
        w=csv.writer(f,lineterminator='\n');w.writerow(['row','target','inner_oof_score'])
        w.writerows((i,int(y),float(p)) for i,(y,p) in enumerate(zip(yd,final_oof)))
    # All choices frozen before generating Test predictions below.
    xt,yt=arrays(test);p=model.predict_proba(xt)[:,1]
    original_model=json.loads((source/'model.json').read_text())
    from .predict import PortableModel
    old_test=PortableModel(original_model).predict_proba(xt[:,:4])[:,1]
    old_policy=select_budget_threshold(yd,old_p,.145)
    result=dict(protocol=PROTOCOL,selected=chosen,selection=policy,folds=folds,
        test_candidate=operating_metrics(yt,p,policy['threshold']),
        test_current_145=operating_metrics(yt,old_test,old_policy['threshold']),
        test_levels=band_report(yt,p,cuts) if cuts is not None else None,
        source_model_sha256=hashlib.sha256((source/'model.json').read_bytes()).hexdigest())
    for label,scores in [('candidate',oof),('comparator',baseline_oof)]:
        result[f'outer_{label}']={k:dict(mean=float(np.mean([f[label][k] for f in folds])),
            sd=float(np.std([f[label][k] for f in folds],ddof=1)))
            for k in ['recall','false_positive_rate','roc_auc','brier_score']}
    save_json(output/'metrics.json',result)
    with (output/'paired_test_scores.csv').open('w',newline='') as f:
        w=csv.writer(f,lineterminator='\n');w.writerow(['row','target','current_score','candidate_score'])
        w.writerows((i,int(y),float(a),float(b)) for i,(y,a,b) in enumerate(zip(yt,old_test,p)))
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();run(args.source,args.output)
