"""Baseline de preços por imagem para o M2."""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
from PIL import Image, ImageOps
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

RAW_CSV = ('https://raw.githubusercontent.com/MahmoudHanyFathalla/'
           'AI-Powered-Restaurant-Finder/main/Scripts/restaurantmenuchanges.csv')
FRACTIONS = [0.25, 0.50, 0.75, 1.00]
SEEDS = [42, 123, 2026]


def find_col(frame, alternatives):
    lower = {str(c).lower(): c for c in frame.columns}
    for name in alternatives:
        if name.lower() in lower:
            return lower[name.lower()]
    return None


def price_number(value):
    if pd.isna(value):
        return np.nan
    s = str(value).strip().replace('$', '').replace('USD', '').replace('US$', '')
    s = re.sub(r'[^0-9.,-]', '', s)
    if not s:
        return np.nan
    if ',' in s and '.' not in s:
        s = s.replace(',', '.')
    else:
        s = s.replace(',', '')
    try:
        x = float(s)
        return x if 0 < x < 5000 else np.nan
    except ValueError:
        return np.nan


def load_csv(csv_path: str, workdir: Path):
    p = workdir / 'restaurantmenuchanges.csv'
    if csv_path:
        p = Path(csv_path)
    if p.exists() and p.stat().st_size < 100:
        p.unlink()
    if not p.exists():
        print('Baixando CSV publico: ', RAW_CSV, flush=True)
        with requests.get(RAW_CSV, timeout=120, stream=True) as response:
            response.raise_for_status()
            with p.open('wb') as dest:
                for chunk in response.iter_content(chunk_size=1 << 20):
                    dest.write(chunk)
    return pd.read_csv(p, low_memory=False, on_bad_lines='skip')


def features_from_image(raw):
    img = Image.open(io.BytesIO(raw)).convert('RGB')
    img = ImageOps.fit(img, (96, 96))
    pix = np.asarray(img, dtype=np.float32) / 255.
    hist = np.concatenate([np.histogram(pix[:,:,c], bins=16, range=(0, 1), density=False)[0]
                           for c in range(3)]).astype(np.float32)
    hist /= pix.shape[0] * pix.shape[1]
    brightness = pix.mean(axis=2)
    extra = [*pix.mean(axis=(0,1)), *pix.std(axis=(0,1)),
             brightness.mean(), brightness.std()]
    return np.concatenate([hist, np.asarray(extra, dtype=np.float32)])


def fetch_one(row):
    url, price, group = row
    try:
        response = requests.get(url, timeout=(5, 10), headers={'User-Agent': 'Mozilla/5.0'},
                                allow_redirects=True)
        if response.status_code != 200:
            return None
        if len(response.content) > 7_000_000 or len(response.content) < 300:
            return None
        feats = features_from_image(response.content)
        sha = hashlib.sha256(response.content).hexdigest()
        return (feats, float(price), str(group), sha, url)
    except Exception:
        return None


def samples(df, max_images, workers, workdir):
    img_col = find_col(df, ['menuItemImageUrl','imageUrl','image_url','image'])
    price_col = find_col(df, ['menuItemCurrentPrice','price','currentPrice'])
    if img_col is None or price_col is None:
        raise RuntimeError(f'Colunas obrigatorias ausentes; colunas encontradas: {list(df.columns)}')
    group_col = find_col(df, ['restaurantId','restaurant_id','restaurantName',
                              'storeId','storeName','restaurantName','restaurant'])
    rows = pd.DataFrame({'url':df[img_col].astype(str).str.strip(),
                         'price':df[price_col].map(price_number),
                         'group':df[group_col].astype(str) if group_col else 'sem_restaurante'})
    rows = rows[rows['url'].str.startswith(('http://','https://')) & rows.price.notna()]
    if rows.empty:
        raise RuntimeError('Nenhuma imagem com preco valido foi encontrada no CSV.')
    count_with_price_url = len(rows)
    conflicts = rows.groupby('url')['price'].nunique()
    bad_urls = set(conflicts[conflicts > 1].index)
    rows = rows[~rows.url.isin(bad_urls)].drop_duplicates('url')
    rows = rows.sample(frac=1, random_state=42).reset_index(drop=True)
    attempts = rows.head(max_images * 3)
    got=[]
    print(f'CSV: {len(df)} registros; {count_with_price_url} com URL+preco; '
          f'{len(bad_urls)} URLs conflitantes; {len(rows)} URLs unicas sem conflito.', flush=True)
    print(f'Baixando ate {max_images} imagens reais; tentando ate {len(attempts)} URLs...',flush=True)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        batch_size = min(250, max(100, max_images))
        indexed = list(enumerate(attempts[['url', 'price', 'group']].itertuples(index=False, name=None)))
        for start in range(0, len(indexed), batch_size):
            batch = indexed[start:start + batch_size]
            futures = {pool.submit(fetch_one, r): i for i, r in batch}
            for future in as_completed(futures):
                item = future.result()
                if item is not None:
                    got.append((futures[future], item))
            hashes = {}
            for _, item in got:
                hashes.setdefault(item[3], set()).add(round(item[1], 2))
            good_count = sum(len(prices) == 1 for prices in hashes.values())
            print(f'Tentativas: {min(start + len(batch), len(indexed))}; imagens obtidas: {len(got)}; distintas válidas: {good_count}', flush=True)
            if good_count >= max_images:
                break
    got=[item for _,item in sorted(got,key=lambda pair:pair[0])]
    by_hash={}
    for item in got:
        by_hash.setdefault(item[3],[]).append(item)
    conflicting_hashes={key for key,items in by_hash.items()
                        if len({round(x[1],2) for x in items})>1}
    out=[items[0] for key,items in by_hash.items()
         if key not in conflicting_hashes][:max_images]
    print(f'Fotos com precos contraditorios em URLs diferentes: {len(conflicting_hashes)}',flush=True)
    if len(out) < 60:
        raise RuntimeError(f'Apenas {len(out)} imagens validas entre {len(attempts)} URLs tentadas. '
                           'Troque a fonte por imagens com preco verificavel; nao invente metricas.')
    X=np.stack([x[0] for x in out])
    y=np.array([x[1] for x in out], dtype=np.float64)
    groups=np.array([x[2] for x in out],dtype=str)
    print(f'Imagens validas e distintas: {len(out)}; restaurantes/grupos: {len(set(groups))}',flush=True)
    np.savez_compressed(workdir/'caracteristicas_reais.npz', X=X,y=y,groups=groups)
    return X,y,groups,{'csv_rows':len(df),'url_price_pairs':count_with_price_url,
                       'conflicting_urls':len(bad_urls),'distinct_images':len(out),
                       'group_col':group_col,'image_col':img_col,'price_col':price_col}


def split_indices(n, groups, seed):
    inds=np.arange(n)
    use_groups = len(set(groups)) >= 10 and len(set(groups))/n >= 0.005
    if use_groups:
        spl=GroupShuffleSplit(n_splits=1,test_size=0.30,random_state=seed)
        train_i,hold_i=next(spl.split(inds,groups=groups))
        if len(set(groups[hold_i])) >= 2:
            spl2=GroupShuffleSplit(n_splits=1,test_size=0.50,random_state=seed+1)
            val_rel,test_rel=next(spl2.split(hold_i,groups=groups[hold_i]))
            val_i,test_i=hold_i[val_rel],hold_i[test_rel]
        else:
            use_groups=False
    if not use_groups:
        train_i,hold_i=train_test_split(inds,test_size=0.30,random_state=seed)
        val_i,test_i=train_test_split(hold_i,test_size=0.50,random_state=seed+1)
    if min(len(train_i),len(val_i),len(test_i)) < 4:
        raise RuntimeError('Muito poucos exemplos na divisao; aumente max-images ou reveja os grupos.')
    return train_i,val_i,test_i, ('por restaurante' if use_groups else 'por imagem')


def run_experiment(X,y,groups):
    all_train=[]; all_val=[]; counts=[]; median_errors=[]; t0=time.perf_counter(); split_type=None
    for seed in SEEDS:
        train_i,val_i, _test_i, kind = split_indices(len(y),groups,42)
        split_type=kind
        median_errors.append(float(mean_absolute_error(y[val_i], np.full(len(val_i), np.median(y[train_i])))))
        rng=np.random.default_rng(seed)
        train_order=rng.permutation(train_i)
        errors_train=[]; errors_val=[]; ns=[]
        for f in FRACTIONS:
            n=max(5,int(round(len(train_i)*f)))
            subset=train_order[:n]
            model=make_pipeline(StandardScaler(),Ridge(alpha=10.0))
            model.fit(X[subset],y[subset])
            errors_train.append(mean_absolute_error(y[subset],model.predict(X[subset])))
            errors_val.append(mean_absolute_error(y[val_i],model.predict(X[val_i])))
            ns.append(n)
        all_train.append(errors_train);all_val.append(errors_val);counts.append(ns)
        print(f'Semente {seed}: treino={len(train_i)}, val={len(val_i)}, '
              f'teste reservado={len(_test_i)}; MAE val={errors_val[-1]:.3f} USD',flush=True)
    elapsed=time.perf_counter()-t0
    all_train=np.asarray(all_train);all_val=np.asarray(all_val)
    results={'seeds':SEEDS,'train_fractions':FRACTIONS,
             'train_counts':np.mean(counts,axis=0).round(0).astype(int).tolist(),
             'train_mae_mean':all_train.mean(axis=0).tolist(),
             'train_mae_std':all_train.std(axis=0,ddof=1).tolist(),
             'val_mae_mean':all_val.mean(axis=0).tolist(),
             'val_mae_std':all_val.std(axis=0,ddof=1).tolist(),
             'val_mae_by_seed':all_val.tolist(),
             'train_duration_seconds':elapsed,'split_type':split_type,
             'test_evaluated':False, 'median_val_mae':float(np.mean(median_errors)), 'split_seed':42, 'split_counts':{'train':len(train_i),'validation':len(val_i),'test':len(_test_i)}}
    return results


def graph(results, path):
    frac=np.asarray(results['train_fractions'])*100
    fig,ax=plt.subplots(figsize=(8.8,4.5),dpi=170)
    tr=np.asarray(results['train_mae_mean']); va=np.asarray(results['val_mae_mean'])
    st=np.asarray(results['train_mae_std']);sv=np.asarray(results['val_mae_std'])
    ax.plot(frac,tr, marker='o', linewidth=2.2,label='Treinamento', color='#2775AE')
    ax.fill_between(frac,tr-st,tr+st,alpha=.17,color='#2775AE')
    ax.plot(frac,va,marker='s',linewidth=2.2,label='Validação',color='#E17C34')
    ax.fill_between(frac,va-sv,va+sv,alpha=.15,color='#E17C34')
    ax.set_xticks(frac);ax.set_xlim(20,105)
    ax.set_xlabel('Porcentagem do conjunto de treinamento (%)')
    ax.set_ylabel('Erro absoluto médio — MAE (US$)')
    ax.set_title('Curva de aprendizado do baseline — Ridge',fontweight='bold')
    ax.grid(alpha=.22);ax.legend(frameon=False)
    fig.tight_layout();fig.savefig(path,bbox_inches='tight');plt.close(fig)


def make_pdf(result, graph_file, pdf_file):
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_CENTER
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import cm
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as PdfImage
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    from matplotlib import font_manager
    pdfmetrics.registerFont(TTFont('T2',font_manager.findfont('DejaVu Sans')))
    pdfmetrics.registerFont(TTFont('T2Bold',font_manager.findfont(font_manager.FontProperties(family='DejaVu Sans',weight='bold'))))
    pdfmetrics.registerFontFamily('T2',normal='T2',bold='T2Bold',italic='T2',boldItalic='T2Bold')
    styles=getSampleStyleSheet()
    for name in styles.byName:
        styles[name].fontName='T2Bold' if name.startswith('Heading') or name=='Title' else 'T2' 
    styles.add(ParagraphStyle(name='M2Title',parent=styles['Heading1'],fontSize=16,
                              leading=20,spaceAfter=12,textColor=colors.HexColor('#172437')))
    styles.add(ParagraphStyle(name='M2Small',parent=styles['Normal'],fontSize=9,leading=14,
                              spaceAfter=9,textColor=colors.HexColor('#5A6775')))
    styles.add(ParagraphStyle(name='M2Body',parent=styles['Normal'],fontSize=10.5,
                              leading=16,spaceAfter=8))
    doc=SimpleDocTemplate(str(pdf_file), pagesize=(21*cm,29.7*cm),rightMargin=2*cm,
                          leftMargin=2*cm,topMargin=2*cm,bottomMargin=2*cm)
    metric=result['val_mae_mean'][-1]; std=result['val_mae_std'][-1]
    t=result['total_duration_seconds']; fit=result['train_duration_seconds']
    processing='leitura das características salvas' if result.get('processing_mode')=='cached_features' else 'obtenção das imagens'
    paragraph=(
        f'O baseline foi implementado com regressão Ridge (alpha = 10) nos dados Restaurant Menu Items (Kaggle), utilizando histogramas '
        f'RGB (16 intervalos por canal) e estatísticas de cor/brilho extraídas de '
        f'{result["distinct_images"]} fotografias reais associadas aos preços anunciados. '
        f'O MAE de validação com 100% dos dados de treino foi '
        f'<b>US$ {metric:.2f} ± {std:.2f}</b> (média ± desvio-padrão, três sementes: '
        f'42, 123 e 2026). A divisão {result["split_type"]} reservou '
        f'{result["split_counts"]["train"]} imagens para treino, '
        f'{result["split_counts"]["validation"]} para validação e '
        f'{result["split_counts"]["test"]} para teste (alvo 70/15/15%). '
        f'A curva apresenta os erros em treino e validação para diferentes frações do '
        f'conjunto de treinamento; o tempo de ajuste e avaliação foi de '
        f'<b>{fit:.2f} s</b>, e o processamento completo, incluindo {processing}, '
        f'levou <b>{t:.1f} s</b>, em CPU. A mediana dos preços do treino obteve '
        f'MAE de US$ {result["median_val_mae"]:.2f} na mesma validação. '
        f'As sementes variam os subconjuntos da curva; com 100% do treino, Ridge é '
        f'determinístico. O conjunto de teste não foi consultado.'
    )
    story=[Paragraph('T2 — Redes Neurais e Aprendizado Profundo',styles['M2Title']),
           Paragraph('João Vitor R. Almeida e Leonardo M. C. Prado | Marco M2 — 05/11/2026',styles['M2Small']),
           Paragraph('Baseline: estimativa do preço de pratos por imagem',styles['Heading2']),
           Spacer(1,9),PdfImage(str(graph_file),width=16.5*cm,height=8.45*cm),
           Spacer(1,12),Paragraph(paragraph,styles['M2Body'])]
    doc.build(story)
    return paragraph


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--csv',default='',help='CSV local (opcional; senao baixa do GitHub)')
    ap.add_argument('--out',default='resultados_m2')
    ap.add_argument('--cached-input',default='',help='Diretorio com caracteristicas da execucao real')
    ap.add_argument('--max-images',type=int,default=2000)
    ap.add_argument('--workers',type=int,default=16)
    args=ap.parse_args()
    output=Path(args.out);output.mkdir(parents=True,exist_ok=True)
    t0=time.perf_counter()
    if args.cached_input:
        cache=Path(args.cached_input)
        data=np.load(cache/'caracteristicas_reais.npz')
        X,y,groups=data['X'],data['y'],data['groups']
        previous=json.loads((cache/'resultados_m2.json').read_text())
        summary={k:previous[k] for k in ['csv_rows','url_price_pairs','conflicting_urls','distinct_images','group_col','image_col','price_col']}
    else:
        df=load_csv(args.csv,output)
        X,y,groups,summary=samples(df,args.max_images,args.workers,output)
    results=run_experiment(X,y,groups)
    results.update(summary)
    results['processing_mode']='cached_features' if args.cached_input else 'download_and_features'
    results['total_duration_seconds']=time.perf_counter()-t0
    graph(results,output/'curva_treino_validacao.png')
    paragraph=make_pdf(results,output/'curva_treino_validacao.png',output/'Entrega_M2.pdf')
    (output/'resultados_m2.json').write_text(json.dumps(results,indent=2,ensure_ascii=False),encoding='utf-8')
    (output/'paragrafo_m2.html').write_text(paragraph,encoding='utf-8')
    print('\nRESULTADO DO EXPERIMENTO:',flush=True)
    print(f'MAE val: US$ {results["val_mae_mean"][-1]:.2f} ± {results["val_mae_std"][-1]:.2f}',flush=True)
    print(f'Tempo de treinamento/avaliacao: {results["train_duration_seconds"]:.2f}s',flush=True)
    print(f'Tempo total: {results["total_duration_seconds"]:.1f}s',flush=True)
    print(f'PDF gerado: {output / "Entrega_M2.pdf"}',flush=True)

if __name__=='__main__':
    main()
