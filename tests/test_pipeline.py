import json
from pathlib import Path
import numpy as np
import pandas as pd
from fastapi.testclient import TestClient
from src.api import app
from src.predict import predict

ROOT=Path(__file__).resolve().parents[1]

def test_restaurants_are_separate_and_test_was_reserved():
    splits=json.loads((ROOT/'data/splits.json').read_text())
    rows=pd.read_csv(ROOT/'data/manifest.csv')
    groups={k:set(rows.iloc[ids].restaurant) for k,ids in splits.items() if isinstance(ids,list)}
    assert not groups['train'] & groups['validation']
    assert not groups['test'] & groups['validation']
    assert not groups['test'] & groups['train']
    assert json.loads((ROOT/'results/comparison.json').read_text())['test_evaluated'] is False

def test_api_rejects_invalid_images():
    with TestClient(app) as client:
        response=client.post('/api/predict',files={'file':('x.txt',b'hello','text/plain')})
        assert response.status_code==415
        response=client.post('/api/predict',files={'file':('x.png',b'not a png','image/png')})
        assert response.status_code==422

def test_real_uploads_match_measured_predictions():
    rows=pd.read_csv(ROOT/'results/validation_predictions.csv')
    selected=rows.sort_values('resnet_prediction').iloc[[0,-1]]
    prices=[]
    with TestClient(app) as client:
        for row in selected.itertuples():
            path=ROOT/'.cache/images'/(row.sha256+'.img')
            raw=path.read_bytes()
            response=client.post('/api/predict',files={'file':('photo.jpg',raw,'image/jpeg')})
            assert response.status_code==200
            result=response.json()
            assert result['currency']=='USD'
            assert abs(result['price']-row.resnet_prediction)<.03
            prices.append(result['price'])
    assert abs(prices[0]-prices[1])>5
