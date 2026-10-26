import json
import threading
from pathlib import Path
from fastapi import FastAPI, UploadFile, HTTPException
from fastapi.staticfiles import StaticFiles
from starlette.concurrency import run_in_threadpool
from .predict import predict

ROOT = Path(__file__).resolve().parents[1]
app = FastAPI(title='À mesa')
lock = threading.Lock()

@app.get('/api/model')
def metadata():
    return json.loads((ROOT / 'models/config.json').read_text())

def locked_predict(raw):
    with lock:
        return predict(raw)

@app.post('/api/predict')
async def estimate(file: UploadFile):
    if file.content_type not in ('image/jpeg','image/png','image/webp'):
        raise HTTPException(415,'Escolha JPG, PNG ou WEBP.')
    try:
        raw = await file.read(10*1024*1024+1)
        if len(raw)>10*1024*1024:
            raise HTTPException(413,'A foto precisa ter até 10 MB.')
        return await run_in_threadpool(locked_predict,raw)
    except HTTPException:
        raise
    except Exception:
        raise HTTPException(422,'Não foi possível ler a imagem.')
    finally:
        await file.close()

app.mount('/',StaticFiles(directory=ROOT/'web',html=True),name='web')
