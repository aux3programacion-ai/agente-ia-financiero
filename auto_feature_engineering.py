import json, os, sys, time, math
import numpy as np
import pandas as pd
import yfinance as yf
from collections import defaultdict

TICKERS_CORE = ['NVDA','MU','DELL','AVGO','DDOG','SMCI','SNOW','CRWD','NOW','TSM','ARM','OKTA','HPE','NTAP','CLS','AAPL','AMZN','GOOGL','META','MSFT','LLY','AMAT','LRCX','PANW','ORCL','HON','UBER','GE','COST','NEE']
DATA_DIR = os.environ.get('GITHUB_WORKSPACE', '.')
OUTPUT = os.path.join(DATA_DIR, 'Datos', 'auto_features.json')
os.makedirs(os.path.join(DATA_DIR, 'Datos'), exist_ok=True)

def compute_auto_features(prices, volumes, ticker):
    """Generate interaction, transform, and aggregate features automatically."""
    features = {}
    close = prices['Close'].values if isinstance(prices, pd.DataFrame) else prices
    vol = volumes.values if isinstance(volumes, pd.Series) else volumes
    
    if len(close) < 60:
        return features
    
    # Cross-features: interactions
    # return * vol
    ret = np.diff(close) / close[:-1]
    if len(ret) >= 20 and len(vol) >= 20:
        vol_norm = vol[-len(ret):] / np.mean(vol[-len(ret):])
        features['ret_vol_interaction'] = float(np.mean(ret[-20:] * vol_norm[-20:]))
        features['ret_vol_corr_20d'] = float(np.corrcoef(ret[-20:], vol_norm[-20:])[0,1]) if np.std(ret[-20:]) > 0 and np.std(vol_norm[-20:]) > 0 else 0.0
    
    # Price * volume profile
    if len(close) >= 30:
        close_norm = close / close.mean()
        vol_norm_full = vol / vol.mean() if len(vol) == len(close) else np.ones(len(close))
        features['price_vol_corr'] = float(np.corrcoef(close_norm[-30:], vol_norm_full[-30:])[0,1]) if np.std(close_norm[-30:]) > 0 and np.std(vol_norm_full[-30:]) > 0 else 0.0
    
    # Transform features
    # Skewness and kurtosis of returns
    if len(ret) >= 20:
        features['ret_skew_20d'] = float(pd.Series(ret[-20:]).skew())
        features['ret_kurt_20d'] = float(pd.Series(ret[-20:]).kurt())
        features['ret_skew_60d'] = float(pd.Series(ret[-60:]).skew()) if len(ret) >= 60 else 0.0
    
    # Rolling z-score of returns
    if len(ret) >= 20:
        r = ret[-20:]
        features['ret_zscore_20d'] = float((r[-1] - np.mean(r)) / max(np.std(r), 1e-8))
    
    # Volatility of volatility
    if len(ret) >= 100:
        vol_20 = pd.Series(ret).rolling(20, min_periods=10).std().values
        vol_20 = vol_20[~np.isnan(vol_20)]
        if len(vol_20) >= 20:
            features['vol_of_vol'] = float(np.std(vol_20[-20:]))
            features['vol_trend'] = float(np.mean(vol_20[-10:]) / max(np.mean(vol_20[-20:-10]), 1e-8))
    
    # Aggregate features across time scales
    for period, label in [(5, '1w'), (20, '1m'), (60, '3m')]:
        if len(ret) >= period:
            rp = ret[-period:]
            features[f'ret_mean_{label}'] = float(np.mean(rp))
            features[f'ret_std_{label}'] = float(np.std(rp))
            features[f'ret_sharpe_{label}'] = float(np.mean(rp) / max(np.std(rp), 1e-8))
            features[f'ret_max_{label}'] = float(np.max(rp))
            features[f'ret_min_{label}'] = float(np.min(rp))
    
    # Volatility clusters: high-vol days concentration
    if len(ret) >= 60:
        r = ret[-60:]
        high_vol = np.abs(r) > np.percentile(np.abs(ret), 80)
        features['high_vol_concentration_60d'] = float(np.mean(high_vol))
        # Serial correlation of absolute returns (vol clustering)
        abs_ret = np.abs(r)
        if len(abs_ret) >= 11:
            features['vol_clustering'] = float(np.corrcoef(abs_ret[:-1], abs_ret[1:])[0,1])
    
    # Consecutive gain/loss streaks
    if len(ret) >= 20:
        signs = np.sign(ret[-20:])
        streak = 1
        max_streak = 1
        for i in range(1, len(signs)):
            if signs[i] == signs[i-1]:
                streak += 1
                max_streak = max(max_streak, streak)
            else:
                streak = 1
        features['max_consecutive_streak_20d'] = int(max_streak)
        features['consecutive_direction'] = int(signs[-1])
    
    return features

def main():
    print('[Auto Features] Generating interaction/transform/aggregate features...')
    all_features = defaultdict(dict)
    for ticker in TICKERS_CORE:
        try:
            print(f'  {ticker}...', end=' ')
            df = yf.download(ticker, period='6mo', interval='1d', progress=False, auto_adjust=True)
            if df is None or df.empty:
                print('[NO DATA]')
                continue
            if isinstance(df.columns, pd.MultiIndex):
                df = df.xs(ticker, axis=1, level=0) if ticker in df.columns.get_level_values(0) else df
            close = df['Close'].dropna().values
            vol = df['Volume'].dropna().values if 'Volume' in df else np.ones(len(close))
            if len(close) < 60:
                print('[SHORT]')
                continue
            f = compute_auto_features(close, vol, ticker)
            all_features[ticker] = f
            print(f'{len(f)} features')
        except Exception as e:
            print(f'[!] {e}')
    
    output = {
        'timestamp': time.strftime('%Y-%m-%dT%H:%M:%S'),
        'tickers': dict(all_features)
    }
    with open(OUTPUT, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f'\n[OK] {len(all_features)} tickers, features saved to {OUTPUT}')

if __name__ == '__main__':
    main()
