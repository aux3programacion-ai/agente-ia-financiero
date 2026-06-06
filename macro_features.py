#!/usr/bin/env python3
"""macro_features.py - Macro regime features from FRED, VIX, yields, DXY"""
import json, os, sys, urllib.request, time, datetime, warnings
import numpy as np, pandas as pd
warnings.filterwarnings('ignore')

DATA_DIR = os.environ.get('GITHUB_WORKSPACE', '.')
OUTPUT = os.path.join(DATA_DIR, 'Datos', 'macro_features.json')
os.makedirs(os.path.join(DATA_DIR, 'Datos'), exist_ok=True)

FRED_API_KEY = os.environ.get('FRED_API_KEY')

FRED_SERIES = {
    'fed_funds': 'FEDFUNDS',           # Effective Federal Funds Rate
    'yield_10y': 'DGS10',              # 10-Year Treasury Constant Maturity Rate
    'yield_2y': 'DGS2',                # 2-Year Treasury Constant Maturity Rate
    'yield_3m': 'DGS3MO',              # 3-Month Treasury Bill
    'dxy': 'DTWEXBGS',                 # Trade Weighted U.S. Dollar Index: Broad, Goods & Services
    'vix': 'VIXCLS',                   # CBOE Volatility Index (via FRED)
    'cpi': 'CPIAUCSL',                 # Consumer Price Index for All Urban Consumers
    'unemployment': 'UNRATE',          # Unemployment Rate
    'pmi': 'MANEMP',                   # Manufacturing Employment (proxy)
    'retail_sales': 'RSXFS',           # Retail Sales
    'industrial_prod': 'INDPRO',       # Industrial Production Index
    'housing_starts': 'HOUST',         # Housing Starts
}

def fetch_fred(series_id, limit=252):
    """Fetch series from FRED API."""
    if not FRED_API_KEY:
        return None
    try:
        url = f'https://api.stlouisfed.org/fred/series/observations?series_id={series_id}&api_key={FRED_API_KEY}&file_type=json&sort_order=desc&limit={limit}'
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read())
        obs = data.get('observations', [])
        if not obs:
            return None
        df = pd.DataFrame(obs)
        df['date'] = pd.to_datetime(df['date'])
        df['value'] = pd.to_numeric(df['value'], errors='coerce')
        df = df.dropna().set_index('date').sort_index()
        return df['value']
    except Exception as e:
        print(f'  [FRED] {series_id}: {e}')
        return None

def fetch_yahoo_macro():
    """Fetch VIX, DXY, yields from Yahoo Finance as fallback."""
    macro = {}
    tickers = {'vix': '^VIX', 'dxy': 'DX-Y.NYB', 'yield_10y': '^TNX', 'yield_2y': '^IRX'}
    try:
        import yfinance as yf
        data = yf.download(list(tickers.values()), period='1y', interval='1d', progress=False, auto_adjust=True)
        if data is not None and not data.empty:
            if isinstance(data.columns, pd.MultiIndex):
                close = data.xs('Close', axis=1, level=1)
            else:
                close = data
            for name, ticker in tickers.items():
                if ticker in close.columns:
                    macro[name] = close[ticker].dropna()
    except Exception as e:
        print(f'  [Yahoo Macro] {e}')
    return macro

def compute_features(fred_data, yahoo_data):
    """Compute derived macro features."""
    features = {}
    today = datetime.datetime.now().strftime('%Y-%m-%d')
    
    # Yield curve slope (10Y - 2Y)
    if 'yield_10y' in fred_data and 'yield_2y' in fred_data:
        y10 = fred_data['yield_10y'].iloc[-1] if len(fred_data['yield_10y']) else 0
        y2 = fred_data['yield_2y'].iloc[-1] if len(fred_data['yield_2y']) else 0
        features['yield_curve_10y_2y'] = round(y10 - y2, 2)
    elif 'yield_10y' in yahoo_data and 'yield_2y' in yahoo_data:
        y10 = yahoo_data['yield_10y'].iloc[-1] / 100  # TNX is in basis points
        y2 = yahoo_data['yield_2y'].iloc[-1] / 100
        features['yield_curve_10y_2y'] = round(y10 - y2, 2)
    
    # Yield curve slope (10Y - 3M)
    if 'yield_10y' in fred_data and 'yield_3m' in fred_data:
        y10 = fred_data['yield_10y'].iloc[-1] if len(fred_data['yield_10y']) else 0
        y3m = fred_data['yield_3m'].iloc[-1] if len(fred_data['yield_3m']) else 0
        features['yield_curve_10y_3m'] = round(y10 - y3m, 2)
    
    # Fed funds rate
    if 'fed_funds' in fred_data:
        features['fed_funds_rate'] = round(fred_data['fed_funds'].iloc[-1], 2)
    
    # VIX level and regime
    vix_val = None
    if 'vix' in fred_data:
        vix_val = fred_data['vix'].iloc[-1]
    elif 'vix' in yahoo_data:
        vix_val = yahoo_data['vix'].iloc[-1]
    if vix_val:
        features['vix_level'] = round(vix_val, 2)
        features['vix_regime'] = 'high' if vix_val > 30 else ('low' if vix_val < 15 else 'normal')
    
    # DXY (Dollar Index)
    dxy_val = None
    if 'dxy' in fred_data:
        dxy_val = fred_data['dxy'].iloc[-1]
    elif 'dxy' in yahoo_data:
        dxy_val = yahoo_data['dxy'].iloc[-1]
    if dxy_val:
        features['dxy_level'] = round(dxy_val, 2)
        # DXY momentum (20d)
        if 'dxy' in fred_data and len(fred_data['dxy']) > 20:
            dxy_20d_ago = fred_data['dxy'].iloc[-20]
            features['dxy_mom_20d'] = round((dxy_val - dxy_20d_ago) / dxy_20d_ago * 100, 2)
    
    # Real rate proxy (10Y - CPI YoY)
    if 'yield_10y' in fred_data and 'cpi' in fred_data:
        y10 = fred_data['yield_10y'].iloc[-1]
        cpi_latest = fred_data['cpi'].iloc[-1]
        cpi_year_ago = fred_data['cpi'].iloc[-13] if len(fred_data['cpi']) > 12 else cpi_latest
        cpi_yoy = (cpi_latest - cpi_year_ago) / cpi_year_ago * 100
        features['real_rate_10y'] = round(y10 - cpi_yoy, 2)
    
    # Credit spread proxy (using yield curve inversion as stress indicator)
    if 'yield_curve_10y_2y' in features:
        features['curve_inverted'] = 1 if features['yield_curve_10y_2y'] < 0 else 0
        features['curve_steepness'] = features['yield_curve_10y_2y']
    
    # Risk-on / risk-off composite
    risk_score = 0
    if features.get('vix_level', 20) < 20: risk_score += 1
    if features.get('yield_curve_10y_2y', 1) > 0.5: risk_score += 1
    if features.get('dxy_mom_20d', 0) < 0: risk_score += 1  # Dollar weakening = risk-on
    features['risk_on_score'] = risk_score  # 0-3
    
    return features

def main():
    print('[+] Fetching macro features...')
    
    fred_data = {}
    if FRED_API_KEY:
        print('  [FRED] Using API key')
        for name, series_id in FRED_SERIES.items():
            print(f'  Fetching {name} ({series_id})...', end=' ')
            series = fetch_fred(series_id)
            if series is not None:
                fred_data[name] = series
                print(f'OK ({len(series)} obs)')
            else:
                print('FAILED')
            time.sleep(0.1)  # Rate limit
    else:
        print('  [FRED] No API key, using Yahoo fallback')
    
    yahoo_data = fetch_yahoo_macro()
    
    features = compute_features(fred_data, yahoo_data)
    
    output = {
        'timestamp': datetime.datetime.now().strftime('%Y-%m-%dT%H:%M:%SZ'),
        'features': features,
        'source': 'FRED+Yahoo' if FRED_API_KEY else 'Yahoo'
    }
    
    with open(OUTPUT, 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f'[OK] Macro features saved: {len(features)} features')
    for k, v in features.items():
        print(f'  {k}: {v}')

if __name__ == '__main__':
    main()