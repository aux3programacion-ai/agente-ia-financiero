import json, os, sys, time
import numpy as np
import pandas as pd
import yfinance as yf

DATA_DIR = os.environ.get('GITHUB_WORKSPACE', '.')
OUTPUT = os.path.join(DATA_DIR, 'Datos', 'sector_momentum.json')
os.makedirs(os.path.join(DATA_DIR, 'Datos'), exist_ok=True)

SECTOR_ETFS = {
    'XLF': 'Financials', 'XLE': 'Energy', 'XLK': 'Technology',
    'XLV': 'Healthcare', 'XLI': 'Industrials', 'XLP': 'Consumer Staples',
    'XLY': 'Consumer Discretionary', 'XLU': 'Utilities', 'XLRE': 'Real Estate',
    'KBE': 'Banks', 'SMH': 'Semiconductors', 'IBB': 'Biotech',
    'TAN': 'Solar/Energy', 'XAR': 'Aerospace', 'XLB': 'Materials',
    'XLC': 'Communication', 'XHB': 'Homebuilding', 'XRT': 'Retail',
    'XME': 'Metals/Mining', 'XOP': 'Oil/Gas'
}

def compute_momentum(close, periods=[21, 63, 126]):
    mom = {}
    for p in periods:
        if len(close) > p:
            mom[f'{p}d'] = round(float((close.iloc[-1] / close.iloc[-p-1] - 1) * 100), 2)
    mom['composite'] = round(np.mean([v for v in mom.values()]), 2) if mom else 0
    return mom

def main():
    print('[Sector Momentum] Evaluando momentum sectorial...')
    try:
        data = yf.download(list(SECTOR_ETFS.keys()), period='1y', interval='1d', progress=False, auto_adjust=True)
        if data is None or data.empty:
            print('[!] No data')
            return
        if isinstance(data.columns, pd.MultiIndex):
            close = data['Close']
        else:
            close = data
    except Exception as e:
        print(f'[!] Download error: {e}')
        return
    
    sectors = {}
    for etf, name in SECTOR_ETFS.items():
        if etf in close.columns:
            try:
                c = close[etf].dropna()
                if len(c) > 126:
                    mom = compute_momentum(c)
                    sectors[name] = {'etf': etf, 'momentum': mom}
            except:
                pass
    
    # Rank by composite momentum
    ranked = sorted(sectors.items(), key=lambda x: x[1]['momentum']['composite'], reverse=True)
    
    results = {
        'timestamp': time.strftime('%Y-%m-%dT%H:%M:%S'),
        'sectors': sectors,
        'ranking': [{'sector': s, 'score': d['momentum']['composite'], 'etf': d['etf']} for s, d in ranked],
        'top_3': [s for s, d in ranked[:3]],
        'bottom_3': [s for s, d in ranked[-3:]],
        'dispersion': round(np.std([d['momentum']['composite'] for _, d in ranked]), 2) if ranked else 0
    }
    
    with open(OUTPUT, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)
    
    print(f'  Top: {", ".join(results["top_3"])}')
    print(f'  Bottom: {", ".join(results["bottom_3"])}')
    print(f'  Dispersion: {results["dispersion"]:.1f}')
    print(f'[OK] Sector momentum guardado en {OUTPUT}')

if __name__ == '__main__':
    main()
