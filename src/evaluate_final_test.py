import json, hashlib, pathlib, subprocess
import numpy as np
import joblib, sklearn
from sklearn.metrics import mean_absolute_error, r2_score
p=pathlib.Path('.')
out=p/'results/final_test.json'
assert not out.exists(), 'Final test already evaluated; use saved results'
with np.load(p/'data/resnet_features.npz') as d, np.load(p/'data/color_features.npz') as c:
    X,y,groups,indices=[d[k] for k in ('X','y','groups','indices')]
    split=json.loads((p/'data/splits.json').read_text())
    tr,va,te=[np.flatnonzero(np.isin(indices,split[k])) for k in ('train','validation','test')]
    assert (len(tr),len(va),len(te))==(1064,331,250)
    assert not (set(groups[tr])&set(groups[te]) or set(groups[va])&set(groups[te]))
    models={'color_ridge':joblib.load(p/'models/color_baseline.joblib'),'resnet':joblib.load(p/'models/regressor.joblib')}
    predictions={'mean':np.full(len(te),y[tr].mean()),'median':np.full(len(te),np.median(y[tr])),'color_ridge':models['color_ridge'].predict(c['X'][indices[te]]),'resnet':np.maximum(0,models['resnet'].predict(X[te]))}
    actual=y[te]
    # Paired bootstrap by restaurant, not by individual image.
    g=np.unique(groups[te])
    blocks=[np.flatnonzero(groups[te]==v) for v in g]
    errors={k:np.abs(actual-v) for k,v in predictions.items()}
    rng=np.random.default_rng(2026)
    boot=np.array([[e[ix].mean() for e in errors.values()] for ix in (np.concatenate([blocks[j] for j in rng.integers(0,len(g),len(g))]) for _ in range(10000))])
    result={'source_commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),'test_evaluated':True,'test_count':len(te),'test_restaurants':len(g),'split_seed':split['seed'],'model_selection':'unchanged; saved models, no refit','sklearn':sklearn.__version__,'bootstrap':{'unit':'restaurant','replicates':10000,'seed':2026},'model_sha256':{str(f):hashlib.sha256(f.read_bytes()).hexdigest() for f in (p/'models/color_baseline.joblib',p/'models/regressor.joblib')},'test':{},'paired_differences':{}}
    for i,(k,v) in enumerate(predictions.items()):
        result['test'][k]={'mae':float(mean_absolute_error(actual,v)),'r2':float(r2_score(actual,v)),'mae_bootstrap_sd':float(boot[:,i].std(ddof=1)),'mae_ci95':np.quantile(boot[:,i],[.025,.975]).tolist()}
    for i,k in enumerate(predictions):
        if k!='resnet':
            delta=boot[:,i]-boot[:,-1]
            result['paired_differences'][k]={'mae_reduction':float(errors[k].mean()-errors['resnet'].mean()),'bootstrap_sd':float(delta.std(ddof=1)),'ci95':np.quantile(delta,[.025,.975]).tolist()}
    out.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))

