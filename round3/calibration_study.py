"""Nested training-only calibration follow-up; historical Test is exploratory."""
import argparse
import json
import platform
from pathlib import Path
import numpy as np
import sklearn
from scipy.optimize import brentq
from scipy.special import expit, logit
from sklearn.calibration import _SigmoidCalibration
from sklearn.model_selection import StratifiedKFold, RepeatedStratifiedKFold
from .balancing_study import (TrainingOnlyLogistic, FEATURES, input_data, extra_features,
    export_fit, metrics, policies, summary, reliability_bins, bootstrap_test, digest, write_csv)
from .predict import PortableModel
from .workflow import save_json

METHODS = ['none', 'intercept', 'sigmoid']
NAMES = FEATURES + ['weight_lb','weight_missing','current_smoker','former_smoker','smoking_unknown']

def fit_model(x, y, method, seed):
    """Calibration data are cross-fitted exclusively within supplied training data."""
    base = TrainingOnlyLogistic('none', .1, seed)
    a, b = -1., 0.
    if method != 'none':
        p = np.full(len(y), np.nan)
        for j, (tr, va) in enumerate(StratifiedKFold(3, shuffle=True, random_state=seed).split(x,y)):
            m = TrainingOnlyLogistic('none', .1, seed*10+j).fit(x[tr], y[tr])
            p[va] = m.predict_proba(x[va])[:,1]
        assert np.isfinite(p).all()
        z = logit(np.clip(p, 1e-10, 1-1e-10))
        if method == 'intercept':
            shift = brentq(lambda c: np.mean(expit(z+c))-np.mean(y), -50, 50)
            b = -shift
        elif method == 'sigmoid':
            c = _SigmoidCalibration().fit(z,y)
            a,b = float(c.a_),float(c.b_)
            if a >= 0: raise ValueError('Nonpositive calibration slope')
        else: raise ValueError(method)
    base.fit(x,y)
    state = export_fit(base, NAMES, {})
    state['selected']['calibration'] = method
    state['study_calibration'] = ({'method':'none'} if method == 'none' else
        dict(method='sigmoid', a=a, b=b, input='margin'))
    return state

def predict(state,x):
    return PortableModel(state).predict_proba(x)[:,1]

def select_method(records):
    best = min(METHODS, key=lambda m: np.mean(records[m]))
    limit = np.mean(records[best])+np.std(records[best],ddof=1)/np.sqrt(len(records[best]))
    chosen = next(m for m in METHODS if np.mean(records[m]) <= limit)
    return dict(chosen=chosen, best_mean_method=best, one_se_limit=float(limit),
        candidates={m:dict(mean_brier=float(np.mean(v)),sd=float(np.std(v,ddof=1)),fold_briers=v)
                    for m,v in records.items()})

def inner_select(x,y,seed):
    scores={m:np.full(len(y),np.nan) for m in METHODS}; records={m:[] for m in METHODS}
    for j,(tr,va) in enumerate(StratifiedKFold(3,shuffle=True,random_state=seed).split(x,y)):
        for method in METHODS:
            state=fit_model(x[tr],y[tr],method,seed*10+j)
            p=predict(state,x[va]);scores[method][va]=p
            records[method].append(float(np.mean((p-y[va])**2)))
    return select_method(records),scores

def run(source,output):
    source,output=Path(source),Path(output)
    if output.exists() and any(output.iterdir()): raise ValueError('Use a new output directory')
    output.mkdir(parents=True,exist_ok=True)
    repo=Path(__file__).resolve().parents[1]
    save_json(output/'protocol.json',dict(methods=METHODS,outer_folds=5,repeats=3,inner_folds=3,
        calibration_folds=3,seed=42,C=.1,selection='one-SE inner Brier, prefer none/intercept/sigmoid',
        protocol_sha256=digest(repo/'round3/CALIBRATION_PROTOCOL_TH.md')))
    save_json(output/'study_status.json',dict(status='started'))
    try:
        original,cohort,audit,x,y,dev,test,provenance=input_data(source,repo)
        ex=extra_features(source/'raw',original)
        x=np.column_stack([x,np.array([ex[r['participant_id']] for r in cohort])]);xd,yd=x[dev],y[dev]
        save_json(output/'provenance.json',provenance)
        variants=METHODS+['selected'];oof={m:np.full((3,len(dev)),np.nan) for m in variants}
        decisions={m:{p:np.zeros((3,len(dev)),bool) for p in ['f2','fpr145']} for m in variants}
        folds=[];choices=[];bins=[]
        for i,(tr,va) in enumerate(RepeatedStratifiedKFold(n_splits=5,n_repeats=3,random_state=42).split(xd,yd)):
            choice,inner=inner_select(xd[tr],yd[tr],100+i)
            choices.append(dict(outer=i,**choice)); states={m:fit_model(xd[tr],yd[tr],m,100+i) for m in METHODS}
            for variant in variants:
                method=choice['chosen'] if variant=='selected' else variant
                p=predict(states[method],xd[va]);oof[variant][i//5,va]=p
                for policy,c in policies(yd[tr],inner[method]).items():
                    decisions[variant][policy][i//5,va]=p>=c['threshold']
                    folds.append(dict(variant=variant,repeat=i//5,fold=i%5,method=method,policy=policy,
                        threshold=c['threshold'],**metrics(yd[va],p,c['threshold'])))
                bins+=reliability_bins(yd[va],p,np.unique(np.quantile(inner[method],[.2,.4,.6,.8])),variant,f'outer_{i}')
            print('Outer',i+1,'/15 selected',choice['chosen'],flush=True)
        repeats=[];oofrows=[]
        for variant in variants:
            assert np.isfinite(oof[variant]).all()
            for r in range(3):
                p=oof[variant][r]
                for policy in ['f2','fpr145']:
                    pred=decisions[variant][policy][r]
                    stats=metrics(yd,p,.5)
                    operating=metrics(yd,pred.astype(float),.5,False)
                    for k in ['tp','fp','fn','tn','recall','precision','f2','fpr']:stats[k]=operating[k]
                    repeats.append(dict(variant=variant,repeat=r,policy=policy,**stats))
                oofrows += [dict(variant=variant,repeat=r,development_row=j,target=int(yd[j]),score=float(p[j]),
                    f2_prediction=int(decisions[variant]['f2'][r,j]),fpr145_prediction=int(decisions[variant]['fpr145'][r,j])) for j in range(len(yd))]
        final,inner=inner_select(xd,yd,42)
        write_csv(output/'final_inner_oof.csv',[dict(development_row=j,target=int(yd[j]),**{m:float(inner[m][j]) for m in METHODS}) for j in range(len(yd))])
        states={m:fit_model(xd,yd,m,42) for m in METHODS}
        states['selected']=states[final['chosen']]
        thresholds={m:policies(yd,inner[final['chosen'] if m=='selected' else m]) for m in variants}
        def values(m,key):return np.array([r[key] for r in repeats if r['variant']==m and r['policy']=='fpr145'])
        support=dict(final_calibrated=final['chosen']!='none',
            brier_lower_every_repeat=bool(np.all(values('selected','brier')<values('none','brier'))),
            mean_budget_recall_not_lower=bool(values('selected','recall').mean()>=values('none','recall').mean()),
            mean_budget_fpr_not_higher=bool(values('selected','fpr').mean()<=values('none','fpr').mean()))
        save_json(output/'decision_before_test.json',dict(final=final,thresholds=thresholds,screening=support,
            calibration_supported=all(support.values()),test_used_for_selection=False))
        for m in states:states[m]={**states[m],'threshold_policies':thresholds[m]}
        save_json(output/'candidate_models.json',states)
        save_json(output/'outer_choices.json',choices)
        write_csv(output/'outer_folds.csv',folds);write_csv(output/'repeat_metrics.csv',repeats)
        write_csv(output/'development_oof.csv',oofrows)
        keys=['tp','fp','fn','tn','recall','precision','f2','fpr','roc_auc','ap','brier','mean_probability',
              'calibration_in_large','calibration_intercept','calibration_slope']
        write_csv(output/'outer_summary.csv',summary(folds,['variant','policy'],keys))
        write_csv(output/'repeat_summary.csv',summary(repeats,['variant','policy'],keys))
        # Frozen selection/model/threshold files above precede Test evaluation.
        scores={m:predict(s,x[test]) for m,s in states.items()};yt=y[test]
        rows=[dict(variant=m,policy=k,threshold=c['threshold'],**metrics(yt,scores[m],c['threshold']))
              for m in variants for k,c in thresholds[m].items()]
        write_csv(output/'test_metrics.csv',rows)
        write_csv(output/'test_scores.csv',[dict(test_row=j,target=int(yt[j]),**{m:float(p[j]) for m,p in scores.items()}) for j in range(len(test))])
        for m in variants:
            bins+=reliability_bins(yt,scores[m],np.unique(np.quantile(inner[final['chosen'] if m=='selected' else m],[.2,.4,.6,.8])),m,'historical_test')
        write_csv(output/'calibration_bins.csv',bins)
        cis,deltas=bootstrap_test(yt,scores,thresholds,2000)
        write_csv(output/'test_bootstrap_ci.csv',cis);write_csv(output/'test_paired_difference_ci.csv',deltas)
        prior=json.loads((repo/'round3/results/balancing_20261111/candidate_models.json').read_text())['weight_smoking']
        err=float(np.max(abs(predict(prior,x[test])-scores['none'])))
        assert err<1e-12
        save_json(output/'verification.json',dict(python=platform.python_version(),numpy=np.__version__,sklearn=sklearn.__version__,
            code_sha256=digest(__file__),source_commit='995ff2356eb81b0ce7f6a3fd4aaa5d8379f8878d',
            uncalibrated_prior_test_max_error=err,decision_sha256=digest(output/'decision_before_test.json'),
            final_method=final['chosen'],calibration_supported=all(support.values()),main_model_replaced=False))
        save_json(output/'study_status.json',dict(status='completed',real_data_metrics_available=True))
        print('Complete',final['chosen'],support,flush=True)
    except Exception as exc:
        save_json(output/'study_status.json',dict(status='failed',error=str(exc)));raise

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',required=True,type=Path);p.add_argument('--output',required=True,type=Path)
    a=p.parse_args();run(a.source,a.output)
