import argparse, io, json
from functools import lru_cache
from pathlib import Path
import joblib
import numpy as np
import torch
from PIL import Image
from torchvision.models import resnet18, ResNet18_Weights

ROOT = Path(__file__).resolve().parents[1]

@lru_cache(maxsize=1)
def load():
    torch.set_num_threads(4)
    backbone = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    backbone.fc = torch.nn.Identity()
    backbone.eval()
    head = joblib.load(ROOT / 'models/regressor.joblib')
    config = json.loads((ROOT / 'models/config.json').read_text())
    return backbone, ResNet18_Weights.IMAGENET1K_V1.transforms(), head, config

def predict(raw):
    if len(raw) > 10*1024*1024:
        raise ValueError('A imagem deve ter até 10 MB.')
    backbone, transform, head, config = load()
    with Image.open(io.BytesIO(raw)) as image:
        if image.width * image.height > 40_000_000:
            raise ValueError('Use uma imagem com até 40 megapixels.')
        tensor = transform(image.convert('RGB')).unsqueeze(0)
    with torch.inference_mode():
        features = backbone(tensor).numpy()
    value = float(max(0, head.predict(features)[0]))
    if not np.isfinite(value):
        raise ValueError('Não foi possível estimar o preço.')
    return {'price': value, 'currency':'USD', 'validation_mae':config['validation_mae'],
            'model':config['name']}

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('image', type=Path)
    args = parser.parse_args()
    print(json.dumps(predict(args.image.read_bytes()),indent=2))
