"""Small prespecified nonlinear Logistic study on the weight+smoking cohort."""
import argparse
import json
import platform
from pathlib import Path
import numpy as np
import sklearn
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, RepeatedStratifiedKFold
from .balancing_study import (FEATURES,input_data,extra_features,metrics,policies,summary,
    reliability_bins,bootstrap_test,digest,write_csv)
from .predict import PortableModel
from .workflow import save_json

NAMES=FEATURES+['weight_lb','weight_missing','current_smoker','former_smoker','smoking_unknown']
CONFIGS=[f'{family}:{c}' for family in ['linear','quadratic'] for c in [.01,.1,1.]]
VARIANTS=['none','linear','quadratic','selected']

class SquareTerms(TransformerMixin,BaseEstimator):
    def fit(self,x,y=None):
        if x.shape[1]!=9:raise ValueError('Expected nine encoded features')
        self.powers_=np.vstack([np.eye(9,dtype=int),2*np.eye(9,dtype=int)[[0,2,3,4]]])
        return self
    def transform(self,x):
        return np.column_stack([x,x[:,[0,2,3,4]]**2])

def fit_model(x,y,config,seed):
    family,c=config.split(':');c=float(c)
    if np.any(np.all(np.isnan(x),axis=0)):raise ValueError('Entirely missing training column')
    steps=[('imputer',SimpleImputer(strategy='median')),('scale',StandardScaler())]
    if family=='quadratic':steps += [('square',SquareTerms()),('expanded_scale',StandardScaler())]
    elif family!='linear':raise ValueError(family)
    steps += [('lr',LogisticRegression(C=c,max_iter=2000,random_state=seed))]
    model=Pipeline(steps).fit(x,y);lr=model.named_steps['lr']
    state=dict(features=NAMES,selected=dict(family='logistic',shape=family,C=c,calibration='none'),
        imputer_median=model.named_steps['imputer'].statistics_.tolist(),
        scaler_mean=model.named_steps['scale'].mean_.tolist(),scaler_scale=model.named_steps['scale'].scale_.tolist(),
        coefficients=lr.coef_.tolist(),intercept=lr.intercept_.tolist(),study_calibration={'method':'none'},
        status='research candidate; no automatic replacement',
        term_names=NAMES+([NAMES[i]+'_squared_standardized' for i in [0,2,3,4]] if family=='quadratic' else []))
    if family=='quadratic':state.update(polynomial_powers=model.named_steps['square'].powers_.tolist(),
        polynomial_mean=model.named_steps['expanded_scale'].mean_.tolist(),
        polynomial_scale=model.named_steps['expanded_scale'].scale_.tolist())
    error=float(np.max(abs(model.predict_proba(x)[:,1]-predict(state,x))))
    if error>1e-12:raise ValueError('Portable model replay failed')
    return state

def predict(state,x):return PortableModel(state).predict_proba(x)[:,1]

def choose(records,configs):
    best=min(configs,key=lambda k:np.mean(records[k]))
    limit=np.mean(records[best])+np.std(records[best],ddof=1)/np.sqrt(len(records[best]))
    chosen=next(k for k in configs if np.mean(records[k])<=limit)
    return dict(chosen=chosen,best_mean_candidate=best,one_se_limit=float(limit))

def inner_select(x,y,seed):
    scores={k:np.full(len(y),np.nan) for k in CONFIGS};records={k:[] for k in CONFIGS}
    for j,(tr,va) in enumerate(StratifiedKFold(3,shuffle=True,random_state=seed).split(x,y)):
        for config in CONFIGS:
            state=fit_model(x[tr],y[tr],config,seed*10+j);p=predict(state,x[va])
            scores[config][va]=p;records[config].append(float(np.mean((p-y[va])**2)))
    groups={'none':dict(chosen='linear:0.1'),'linear':choose(records,CONFIGS[:3]),
            'quadratic':choose(records,CONFIGS[3:]),'selected':choose(records,CONFIGS)}
    return dict(groups=groups,candidates={k:dict(mean_brier=float(np.mean(v)),sd=float(np.std(v,ddof=1)),fold_briers=v)
                 for k,v in records.items()}),scores

def run(source,output):
    source,output=Path(source),Path(output);repo=Path(__file__).resolve().parents[1]
    if output.exists() and any(output.iterdir()):raise ValueError('Use a new output directory')
    output.mkdir(parents=True,exist_ok=True)
    save_json(output/'protocol.json',dict(configs=CONFIGS,variants=VARIANTS,outer_folds=5,repeats=3,inner_folds=3,seed=42,
        protocol_sha256=digest(repo/'round3/NONLINEAR_PROTOCOL_TH.md')))
    save_json(output/'study_status.json',dict(status='started'))
    try:
        original,cohort,audit,x,y,dev,test,provenance=input_data(source,repo)
        extra=extra_features(source/'raw',original)
        x=np.column_stack([x,np.array([extra[r['participant_id']] for r in cohort])]);xd,yd=x[dev],y[dev]
        save_json(output/'provenance.json',provenance)
        oof={v:np.full((3,len(dev)),np.nan) for v in VARIANTS}
        decisions={v:{p:np.zeros((3,len(dev)),bool) for p in ['f2','fpr145']} for v in VARIANTS}
        folds=[];choices=[];bins=[];membership=[]
        for i,(tr,va) in enumerate(RepeatedStratifiedKFold(n_splits=5,n_repeats=3,random_state=42).split(xd,yd)):
            choice,inner=inner_select(xd[tr],yd[tr],100+i);choices.append(dict(outer=i,**choice))
            states={k:fit_model(xd[tr],yd[tr],k,100+i) for k in set(g['chosen'] for g in choice['groups'].values())}
            membership += [dict(repeat=i//5,fold=i%5,development_row=int(j)) for j in va]
            for variant in VARIANTS:
                config=choice['groups'][variant]['chosen'];state=states[config]
                p=predict(state,xd[va]);oof[variant][i//5,va]=p
                trainstats=metrics(yd[tr],predict(state,xd[tr]),.5,False)
                for policy,c in policies(yd[tr],inner[config]).items():
                    decisions[variant][policy][i//5,va]=p>=c['threshold']
                    folds.append(dict(variant=variant,repeat=i//5,fold=i%5,config=config,policy=policy,threshold=c['threshold'],
                        train_auc=trainstats['roc_auc'],train_brier=trainstats['brier'],**metrics(yd[va],p,c['threshold'])))
                bins+=reliability_bins(yd[va],p,np.unique(np.quantile(inner[config],[.2,.4,.6,.8])),variant,f'outer_{i}')
            print('Outer',i+1,'/15',choice['groups']['selected']['chosen'],flush=True)
        repeats=[];oofrows=[]
        for variant in VARIANTS:
            assert np.isfinite(oof[variant]).all()
            for r in range(3):
                p=oof[variant][r]
                for policy in ['f2','fpr145']:
                    stats=metrics(yd,p,.5);op=metrics(yd,decisions[variant][policy][r].astype(float),.5,False)
                    for k in ['tp','fp','fn','tn','recall','precision','f2','fpr']:stats[k]=op[k]
                    repeats.append(dict(variant=variant,repeat=r,policy=policy,**stats))
                oofrows += [dict(variant=variant,repeat=r,development_row=j,target=int(yd[j]),score=float(p[j]),
                    f2_prediction=int(decisions[variant]['f2'][r,j]),fpr145_prediction=int(decisions[variant]['fpr145'][r,j])) for j in range(len(yd))]
        final,inner=inner_select(xd,yd,42)
        states={v:fit_model(xd,yd,final['groups'][v]['chosen'],42) for v in VARIANTS}
        thresholds={v:policies(yd,inner[final['groups'][v]['chosen']]) for v in VARIANTS}
        def values(v,k,policy='fpr145'):return np.array([r[k] for r in repeats if r['variant']==v and r['policy']==policy])
        screening=dict(brier_lower_every_repeat=bool(np.all(values('selected','brier')<values('none','brier'))),
            auc_higher_every_repeat=bool(np.all(values('selected','roc_auc')>values('none','roc_auc'))),
            mean_f2_not_lower=bool(values('selected','f2','f2').mean()>=values('none','f2','f2').mean()),
            mean_budget_recall_not_lower=bool(values('selected','recall').mean()>=values('none','recall').mean()),
            mean_budget_fpr_not_higher=bool(values('selected','fpr').mean()<=values('none','fpr').mean()))
        save_json(output/'decision_before_test.json',dict(final=final,thresholds=thresholds,screening=screening,
            improvement_supported=all(screening.values()),test_used_for_selection=False))
        for v in VARIANTS:states[v]['threshold_policies']=thresholds[v]
        save_json(output/'candidate_models.json',states);save_json(output/'outer_choices.json',choices)
        write_csv(output/'final_inner_oof.csv',[dict(development_row=j,target=int(yd[j]),**{k:float(inner[k][j]) for k in CONFIGS}) for j in range(len(yd))])
        write_csv(output/'outer_membership.csv',membership);write_csv(output/'outer_folds.csv',folds)
        write_csv(output/'repeat_metrics.csv',repeats);write_csv(output/'development_oof.csv',oofrows)
        keys=['tp','fp','fn','tn','recall','precision','f2','fpr','roc_auc','ap','brier','calibration_in_large','calibration_slope']
        write_csv(output/'outer_summary.csv',summary(folds,['variant','policy'],keys+['train_auc','train_brier']))
        write_csv(output/'repeat_summary.csv',summary(repeats,['variant','policy'],keys))
        # Final decision and model files are frozen above; Test enters only here.
        scores={v:predict(s,x[test]) for v,s in states.items()};yt=y[test]
        testrows=[dict(variant=v,policy=k,threshold=c['threshold'],**metrics(yt,scores[v],c['threshold'])) for v in VARIANTS for k,c in thresholds[v].items()]
        write_csv(output/'test_metrics.csv',testrows)
        write_csv(output/'test_scores.csv',[dict(test_row=j,target=int(yt[j]),**{v:float(p[j]) for v,p in scores.items()}) for j in range(len(test))])
        for v in VARIANTS:bins+=reliability_bins(yt,scores[v],np.unique(np.quantile(inner[final['groups'][v]['chosen']],[.2,.4,.6,.8])),v,'historical_test')
        write_csv(output/'calibration_bins.csv',bins)
        cis,deltas=bootstrap_test(yt,scores,thresholds,2000)
        write_csv(output/'test_bootstrap_ci.csv',cis);write_csv(output/'test_paired_difference_ci.csv',deltas)
        prior=json.loads((repo/'round3/results/calibration_20261111/candidate_models.json').read_text())['none']
        err=float(np.max(abs(predict(prior,x[test])-scores['none'])));assert err<1e-12
        save_json(output/'verification.json',dict(python=platform.python_version(),numpy=np.__version__,sklearn=sklearn.__version__,
            code_sha256=digest(__file__),source_commit='befec00faa9d341b983c984adc087ce0cf4585a8',
            baseline_test_max_error=err,decision_sha256=digest(output/'decision_before_test.json'),
            final_config=final['groups']['selected']['chosen'],improvement_supported=all(screening.values()),main_model_replaced=False))
        save_json(output/'study_status.json',dict(status='completed',real_data_metrics_available=True))
        print('Complete',final['groups']['selected']['chosen'],screening,flush=True)
    except Exception as exc:
        save_json(output/'study_status.json',dict(status='failed',error=str(exc)));raise

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',required=True,type=Path);p.add_argument('--output',required=True,type=Path)
    a=p.parse_args();run(a.source,a.output)
