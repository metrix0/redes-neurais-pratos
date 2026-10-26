import argparse, json, time
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, Normalizer
from sklearn.neighbors import KNeighborsRegressor
from baseline_m2 import split_indices

ROOT = Path(__file__).resolve().parents[1]

def ridge(alpha, log):
    pipeline = make_pipeline(StandardScaler(), Ridge(alpha=alpha))
    if log:
        return TransformedTargetRegressor(regressor=pipeline, func=np.log1p, inverse_func=np.expm1)
    return pipeline

def scores(y, pred):
    return {'mae':float(mean_absolute_error(y,pred)), 'r2':float(r2_score(y,pred)),
            'pred_std':float(np.std(pred)), 'pred_p10_p50_p90':np.quantile(pred,[.1,.5,.9]).tolist()}

def main():
    start = time.perf_counter()
    features = np.load(ROOT / 'data/resnet_features.npz')
    original = np.load(ROOT / 'data/color_features.npz')
    X, y, groups, indices = (features[k] for k in ['X','y','groups','indices'])
    train_original, val_original, test_original, _ = split_indices(
        len(original['y']), original['groups'], 42)
    train = np.flatnonzero(np.isin(indices,train_original))
    val = np.flatnonzero(np.isin(indices,val_original))
    test = np.flatnonzero(np.isin(indices,test_original))
    split = {'train':indices[train].tolist(), 'validation':indices[val].tolist(),
             'test':indices[test].tolist(), 'seed':42}
    (ROOT / 'data/splits.json').write_text(json.dumps(split,indent=2))
    assert not set(groups[train]) & set(groups[val])
    cv = list(GroupKFold(n_splits=3).split(X[train],y[train],groups[train]))
    candidates = []
    for log in [False, True]:
        for alpha in [1.,10.,100.,1000.,10000.]:
            candidates.append((f'resnet_ridge_{"log" if log else "raw"}_{alpha:g}', lambda a=alpha,l=log:ridge(a,l)))
    for leaf in [2,8,20]:
        candidates.append((f'resnet_trees_leaf{leaf}',
            lambda l=leaf:ExtraTreesRegressor(n_estimators=200,min_samples_leaf=l,
                max_features=.7,random_state=42,n_jobs=4)))
    for k in [5,15,30]:
        candidates.append((f'resnet_knn_{k}', lambda n=k:make_pipeline(
            Normalizer(),KNeighborsRegressor(n_neighbors=n,weights='distance',n_jobs=4))))
    leaderboard=[]
    for name, build in candidates:
        fold_scores=[]
        for a,b in cv:
            model=build()
            model.fit(X[train[a]],y[train[a]])
            fold_scores.append(mean_absolute_error(y[train[b]],np.maximum(0,model.predict(X[train[b]]))))
        row={'name':name,'cv_mae':float(np.mean(fold_scores)),'cv_folds':list(map(float,fold_scores))}
        leaderboard.append(row)
        print(name,'train-group-CV MAE',round(row['cv_mae'],4),flush=True)
    winner=min(leaderboard,key=lambda r:r['cv_mae'])
    build=dict(candidates)[winner['name']]
    model=build().fit(X[train],y[train])
    pred=np.maximum(0,model.predict(X[val]))
    old=make_pipeline(StandardScaler(),Ridge(alpha=10.)).fit(original['X'][indices[train]],y[train])
    oldpred=old.predict(original['X'][indices[val]])
    mean_pred=np.full(len(val),y[train].mean())
    median_pred=np.full(len(val),np.median(y[train]))
    validation={'mean':scores(y[val],mean_pred),'median':scores(y[val],median_pred),
                'color_ridge':scores(y[val],oldpred),'resnet':scores(y[val],pred)}
    joblib.dump(model,ROOT/'models/regressor.joblib')
    joblib.dump(old,ROOT/'models/color_baseline.joblib')
    config={'name':winner['name'],'extractor':'resnet18','weights':'IMAGENET1K_V1',
            'trained_parameters':'regression head only; backbone frozen',
            'validation_mae':validation['resnet']['mae'],'currency':'USD',
            'train_count':len(train),'validation_count':len(val),'test_count':len(test)}
    (ROOT/'models/config.json').write_text(json.dumps(config,indent=2))
    rows=pd.read_csv(ROOT/'data/manifest.csv').iloc[indices[val]].copy()
    rows['original_index']=indices[val]
    rows['color_prediction']=oldpred
    rows['resnet_prediction']=pred
    rows['absolute_error']=abs(rows.price-pred)
    rows.to_csv(ROOT/'results/validation_predictions.csv',index=False)
    bands=[]
    for lo,hi in [(0,5),(5,10),(10,20),(20,float('inf'))]:
        mask=(y[val]>=lo)&(y[val]<hi)
        if mask.any():
            bands.append({'actual_price_band':f'{lo}–{hi}', 'n':int(mask.sum()),
                'actual_mean':float(y[val][mask].mean()),
                'old_pred_mean':float(oldpred[mask].mean()),
                'new_pred_mean':float(pred[mask].mean()),
                'old_mae':float(abs(y[val][mask]-oldpred[mask]).mean()),
                'new_mae':float(abs(y[val][mask]-pred[mask]).mean())})
    results={'selected':winner['name'],'selection':'3-fold GroupKFold on training restaurants only',
        'cv':leaderboard,'validation':validation,'bands':bands,
        'counts':{'train':len(train),'validation':len(val),'reserved_test':len(test)},
        'test_evaluated':False,'seconds':time.perf_counter()-start}
    (ROOT/'results/comparison.json').write_text(json.dumps(results,indent=2))
    print(json.dumps(results['validation'],indent=2),flush=True)
if __name__ == '__main__':
    main()
