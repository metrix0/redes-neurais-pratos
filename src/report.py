import json
from pathlib import Path
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import mean_absolute_error
ROOT=Path(__file__).resolve().parents[1]

def main():
    result=json.loads((ROOT/'results/comparison.json').read_text())
    frame=pd.read_csv(ROOT/'results/validation_predictions.csv')
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10})
    fig,axes=plt.subplots(1,2,figsize=(11,4.5))
    names=['Média do treino','Mediana do treino','Ridge com cores','ResNet-18']
    mae=[result['validation'][k]['mae'] for k in ['mean','median','color_ridge','resnet']]
    bars=axes[0].barh(names,mae,color=['#b5bcae','#b5bcae','#b5bcae','#3c674e'])
    axes[0].invert_yaxis();axes[0].set_xlim(0,max(mae)*1.2)
    for b,m in zip(bars,mae):axes[0].text(m+.08,b.get_y()+b.get_height()/2,f'US$ {m:.2f}',va='center')
    axes[0].set_xlabel('MAE na validação (US$)')
    axes[1].scatter(frame.price,frame.color_prediction,label='Ridge com cores',alpha=.4,s=12,color='#b99a76')
    axes[1].scatter(frame.price,frame.resnet_prediction,label='ResNet-18',alpha=.55,s=12,color='#3c674e')
    hi=max(frame.price.max(),frame.resnet_prediction.max())*1.05
    axes[1].plot([0,hi],[0,hi],'--',lw=1,color='#888')
    axes[1].set(xlabel='Preço anunciado (US$)',ylabel='Preço previsto (US$)',xlim=(0,hi),ylim=(0,hi))
    axes[1].legend(frameon=False,fontsize=8)
    for a in axes:
        a.spines[['top','right']].set_visible(False)
    fig.tight_layout();fig.savefig(ROOT/'results/comparison.png',dpi=160);plt.close(fig)
    data=np.load(ROOT/'data/resnet_features.npz')
    splits=json.loads((ROOT/'data/splits.json').read_text())
    train=np.flatnonzero(np.isin(data['indices'],splits['train']))
    val=np.flatnonzero(np.isin(data['indices'],splits['validation']))
    base=joblib.load(ROOT/'models/regressor.joblib')
    curve=[]
    for seed in [42,123,2026]:
        order=np.random.default_rng(seed).permutation(train)
        for fraction in [.25,.5,1.]:
            ids=order[:round(len(order)*fraction)]
            model=clone(base).fit(data['X'][ids],data['y'][ids])
            curve.append({'seed':seed,'fraction':fraction,'n':len(ids),
                'train_mae':mean_absolute_error(data['y'][ids],np.maximum(0,model.predict(data['X'][ids]))),
                'validation_mae':mean_absolute_error(data['y'][val],np.maximum(0,model.predict(data['X'][val])))})
    pd.DataFrame(curve).to_csv(ROOT/'results/learning_curve.csv',index=False)
    print('Report figures and three-seed learning curve generated',flush=True)
if __name__=='__main__':
    main()
