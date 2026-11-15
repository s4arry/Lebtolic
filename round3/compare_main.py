"""Reproduce the pinned main notebook and compare the user's befec00 candidate.

Does not select/retrain the candidate using Test. Main's default threshold and
additional development-only policies are explicitly distinguished.
"""
import argparse,json,subprocess,hashlib
from pathlib import Path
import numpy as np
from sklearn.model_selection import StratifiedKFold
from .balancing_study import input_data,extra_features,metrics,policies,write_csv,digest,bootstrap_test
from .cv_study import fit_legacy
from .predict import PortableModel
from .workflow import save_json,NOTEBOOK

MAIN_SHA='6ce4577dc3fa40ea1d3fe1e4e726119089c378ba'
CANDIDATE_SHA='befec00faa9d341b983c984adc087ce0cf4585a8'

def run(source,output):
    repo=Path(__file__).resolve().parents[1];source,output=Path(source),Path(output)
    if output.exists() and any(output.iterdir()):raise ValueError('Use a new output directory')
    output.mkdir(parents=True,exist_ok=True)
    save_json(output/'protocol.json',dict(main_commit=MAIN_SHA,candidate_commit=CANDIDATE_SHA,
        purpose='Compare fixed workflows, not select a model from exposed Test',
        extra_threshold_policy='3-fold development OOF seed42, F2 and FPR145; preserve main default .5',
        candidate='weight_smoking none from calibration_20261111',test_scope='historical, already exposed',
        acceptance='Do not claim overall superiority if ranking and probability metrics disagree or paired intervals include zero'))
    save_json(output/'study_status.json',dict(status='started'))
    try:
        content=subprocess.check_output(['git','show',MAIN_SHA+':LEPTOLIC_NHEFS_Future_Prediction.ipynb'],cwd=repo)
        if hashlib.sha256(content).hexdigest()!=digest(NOTEBOOK):raise ValueError('Local notebook differs from pinned main')
        original,cohort,audit,x,y,dev,test,provenance=input_data(source,repo)
        x5=np.array([[r[f] for f in original['FEATURES']] for r in cohort],float)
        split=original['split_ids'](y)
        # Fit exact main algorithm on its historical Train/Tune/Calibration split.
        main,mainfit=fit_legacy(x5,y,original,split)
        pd=np.full(len(dev),np.nan)
        for i,(tr,va) in enumerate(StratifiedKFold(3,shuffle=True,random_state=42).split(x5[dev],y[dev])):
            fitted,_=fit_legacy(x5[dev[tr]],y[dev[tr]],original)
            pd[va]=fitted(x5[dev[va]])
        mainpolicies=policies(y[dev],pd)
        path=repo/'round3/results/calibration_20261111/candidate_models.json'
        state=json.loads(path.read_text())['none']
        expected=json.loads(subprocess.check_output(['git','show',CANDIDATE_SHA+':round3/results/calibration_20261111/candidate_models.json'],cwd=repo))['none']
        if state!=expected:raise ValueError('Candidate changed after befec00')
        extra=extra_features(source/'raw',original);x9=np.column_stack([x,np.array([extra[r['participant_id']] for r in cohort])])
        chosenpolicies=state['threshold_policies']
        finalpolicies={'none':mainpolicies,'befec00':chosenpolicies}
        save_json(output/'decision_before_test.json',dict(main=mainpolicies,befec00=chosenpolicies,
            main_selected_ridge=float(mainfit['ridge']),candidate_fixed=True,test_used_for_choices=False))
        # Comparisons below do not alter choices above.
        scores={'none':main(x5[test]),'befec00':PortableModel(state).predict_proba(x9[test])[:,1]};yt=y[test]
        rows=[]
        for name in scores:
            for label,threshold in [('default_0_5',.5)]+[(k,v['threshold']) for k,v in finalpolicies[name].items()]:
                rows.append(dict(model=name,policy=label,threshold=threshold,**metrics(yt,scores[name],threshold)))
        write_csv(output/'test_metrics.csv',rows)
        write_csv(output/'test_scores.csv',[dict(test_row=j,target=int(yt[j]),main_score=float(scores['none'][j]),befec00_score=float(scores['befec00'][j])) for j in range(len(test))])
        write_csv(output/'main_development_oof.csv',[dict(development_row=j,target=int(y[dev[j]]),score=float(pd[j])) for j in range(len(dev))])
        diagnostics=[]
        for name,ids in split.items():diagnostics.append(dict(split=name,**metrics(y[ids],main(x5[ids]),.5)))
        write_csv(output/'main_split_metrics.csv',diagnostics)
        cis,deltas=bootstrap_test(yt,scores,finalpolicies,2000)
        write_csv(output/'test_bootstrap_ci.csv',cis);write_csv(output/'test_paired_difference_ci.csv',deltas)
        nb=json.loads(content);saved=None
        for c in nb['cells']:
            for out in c.get('outputs',[]):
                text=''.join(out.get('text',[]))
                try:v=json.loads(text)
                except (ValueError,TypeError):continue
                if isinstance(v,dict) and 'selected_ridge' in v:saved=v
        assert saved is not None
        assert float(mainfit['ridge'])==saved['selected_ridge']
        base=rows[0]
        np.testing.assert_allclose(base['roc_auc'],saved['test_model']['roc_auc'],atol=1e-12,rtol=0)
        np.testing.assert_allclose(base['brier'],saved['test_model']['brier_score'],atol=1e-12,rtol=0)
        save_json(output/'verification.json',dict(main_commit=MAIN_SHA,candidate_commit=CANDIDATE_SHA,
            main_notebook_sha256=hashlib.sha256(content).hexdigest(),code_sha256=digest(__file__),
            main_saved_auc_reproduced=True,main_saved_brier_reproduced=True,main_ridge=float(mainfit['ridge']),
            main_platt_coefficients=np.asarray(mainfit['calibration']).tolist(),
            counts=provenance['counts'],same_test_membership=True,test_selected_model=False,
            main_documentation_conflict='Main summary claims 6 features/refit/AUC .682; actual notebook and outputs have 5 features/AUC .6761458333'))
        save_json(output/'study_status.json',dict(status='completed'))
        print(json.dumps(rows,indent=2),flush=True)
    except Exception as exc:
        save_json(output/'study_status.json',dict(status='failed',error=str(exc)));raise

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();run(a.source,a.output)
