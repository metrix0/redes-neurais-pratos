import argparse, hashlib, io, json, time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import pandas as pd
import requests
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]

def download(row, folder):
    target = folder / (row.sha256 + '.img')
    if target.exists() and hashlib.sha256(target.read_bytes()).hexdigest() == row.sha256:
        return row.Index, 'cached'
    try:
        response = requests.get(row.url, timeout=(8, 20), headers={'User-Agent': 'Mozilla/5.0'})
        response.raise_for_status()
        raw = response.content
        if len(raw) > 10_000_000:
            return row.Index, 'too_large'
        if hashlib.sha256(raw).hexdigest() != row.sha256:
            return row.Index, 'changed'
        Image.open(io.BytesIO(raw)).verify()
        target.write_bytes(raw)
        return row.Index, 'downloaded'
    except Exception as error:
        return row.Index, type(error).__name__

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--workers', type=int, default=24)
    args = parser.parse_args()
    folder = ROOT / '.cache/images'
    folder.mkdir(parents=True, exist_ok=True)
    frame = pd.read_csv(ROOT / 'data/manifest.csv')
    status = {}
    start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(download, row, folder) for row in frame.itertuples()]
        for i, future in enumerate(as_completed(futures), 1):
            index, kind = future.result()
            status[str(index)] = kind
            if i % 100 == 0 or i == len(frame):
                ok = sum(v in ('cached', 'downloaded') for v in status.values())
                print(f'{i}/{len(frame)} URLs; {ok} original images verified', flush=True)
    result = {'seconds': time.perf_counter()-start, 'status': status}
    (ROOT / 'results/download_status.json').write_text(json.dumps(result, indent=2))
if __name__ == '__main__':
    main()
