import argparse, hashlib, json, time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision.models import resnet18, ResNet18_Weights

ROOT = Path(__file__).resolve().parents[1]
WEIGHTS = ResNet18_Weights.IMAGENET1K_V1

class Images(Dataset):
    def __init__(self, rows):
        self.rows = rows
        self.transform = WEIGHTS.transforms()
    def __len__(self):
        return len(self.rows)
    def __getitem__(self, i):
        row = self.rows.iloc[i]
        path = ROOT / '.cache/images' / (row.sha256 + '.img')
        with Image.open(path) as image:
            tensor = self.transform(image.convert('RGB'))
        return tensor

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--threads', type=int, default=4)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    frame = pd.read_csv(ROOT / 'data/manifest.csv')
    valid = [i for i, row in frame.iterrows()
             if (ROOT / '.cache/images' / (row.sha256 + '.img')).exists()]
    rows = frame.iloc[valid]
    model = resnet18(weights=WEIGHTS)
    model.fc = torch.nn.Identity()
    model.eval()
    loader = DataLoader(Images(rows), batch_size=32, num_workers=0)
    blocks = []
    start = time.perf_counter()
    with torch.inference_mode():
        for i, images in enumerate(loader):
            blocks.append(model(images).numpy())
            if (i+1) % 5 == 0:
                print(f'ResNet features: {min((i+1)*32,len(rows))}/{len(rows)}', flush=True)
    X = np.concatenate(blocks)
    np.savez_compressed(ROOT / 'data/resnet_features.npz', X=X,
        indices=np.asarray(valid), y=rows.price.to_numpy(), groups=rows.restaurant.to_numpy(dtype=str))
    meta = {'extractor':'torchvision ResNet-18 IMAGENET1K_V1', 'dimensions':512,
            'seconds':time.perf_counter()-start, 'images':len(rows),
            'torch':torch.__version__, 'weights_url':WEIGHTS.url,
            'transforms':str(WEIGHTS.transforms())}
    (ROOT / 'results/extraction.json').write_text(json.dumps(meta,indent=2))
    print(json.dumps(meta),flush=True)
if __name__ == '__main__':
    main()
