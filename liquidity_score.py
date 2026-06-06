import json, os, sys, time
import numpy as np
import pandas as pd
import yfinance as yf

TICKERS_CORE = ['NVDA','MU','DELL','AVGO','DDOG','SMCI','SNOW','CRWD','NOW','TSM','ARM','OKTA','HPE','NTAP','CLS','AAPL','AMZN','GOOGL','META','MSFT','LLY','AMAT','LRCX','PANW','ORCL','HON','UBER','GE','COST','NEE']
DATA_DIR = os.environ.get('GITHUB_WORKSPACE', '.')
OUTPUT = os.path.join(DATA_DIR, 'Datos', 'liquidity_score.json')
os.makedirs(os.path.join(DATA_DIR, 'Datos'), exist_ok=True)

def score_liquidity(ticker):
    """Score 0-100 based on avg volume, bid-ask spread, and market cap proxy."""
    score = 50
    details = {}
    try:
        df = yf.download(ticker, period='2mo', interval='1d', progress=False, auto_adjust=True)
        if df is None or df.empty:
            return None, 'no_data'
        if isinstance(df.columns, pd.MultiIndex):
            df = df.xs(ticker, axis=1, level=0) if ticker in df.columns.get_level_values(0) else df
        df = df.dropna(subset=['Close','Volume'])
        
        avg_vol = df['Volume'].mean()
        med_vol = df['Volume'].median()
        details['avg_volume'] = int(avg_vol)
        details['median_volume'] = int(med_vol)
        
        if avg_vol > 50_000_000:
            score += 30
        elif avg_vol > 20_000_000:
            score += 25
        elif avg_vol > 10_000_000:
            score += 20
        elif avg_vol > 5_000_000:
            score += 15
        elif avg_vol > 2_000_000:
            score += 10
        elif avg_vol > 1_000_000:
            score += 5
        
        details['volume_score'] = score - 50
        
        # Price stability: low volatility days < 2%
        close = df['Close']
        daily_ret = close.pct_change().dropna()
        vol_20d = daily_ret.rolling(20).std().iloc[-1] if len(daily_ret) >= 20 else daily_ret.std()
        details['volatility_20d'] = round(float(vol_20d * 100), 2)
        
        if vol_20d < 0.015:
            score += 10
        elif vol_20d < 0.025:
            score += 5
        elif vol_20d > 0.05:
            score -= 10
        elif vol_20d > 0.04:
            score -= 5
        
        # Volume consistency (CV of volume)
        vol_cv = df['Volume'].std() / max(df['Volume'].mean(), 1)
        details['volume_cv'] = round(float(vol_cv), 2)
        if vol_cv < 0.5:
            score += 10
        elif vol_cv < 0.8:
            score += 5
        
        # Price level proxy (higher price often = more liquid)
        avg_price = close.mean()
        details['avg_price'] = round(float(avg_price), 2)
        if avg_price > 500:
            score += 5
        elif avg_price > 100:
            score += 3
        
        # Market cap proxy via Yahoo info
        try:
            tk = yf.Ticker(ticker)
            info = tk.info
            mcap = info.get('marketCap', 0)
            details['market_cap'] = mcap
            if mcap > 200e9:
                score += 10
            elif mcap > 50e9:
                score += 7
            elif mcap > 10e9:
                score += 5
        except:
            pass
        
        # Illiquidity penalty: zero-volume days
        zero_days = int((df['Volume'] == 0).sum())
        details['zero_volume_days'] = zero_days
        if zero_days > 5:
            score -= 20
        elif zero_days > 2:
            score -= 10
        
    except Exception as e:
        return None, str(e)
    
    return max(5, min(100, score)), details

def main():
    print('[Liquidity Score] Evaluando liquidez de cada ticker...')
    results = {}
    for t in TICKERS_CORE:
        try:
            print(f'  {t}...', end=' ')
            score, details = score_liquidity(t)
            if score is None:
                print(f'[SKIP] {details}')
                continue
            results[t] = {'score': score, 'details': details}
            print(f'score={score}')
        except Exception as e:
            print(f'[!] {e}')
    
    output = {
        'timestamp': time.strftime('%Y-%m-%dT%H:%M:%S'),
        'tickers': results,
        'liquidity_warning': []
    }
    
    # Warnings for low liquidity tickers
    for t, r in sorted(results.items(), key=lambda x: x[1]['score']):
        if r['score'] < 40:
            output['liquidity_warning'].append(f'{t}: score={r["score"]} - baja liquidez, usar ordenes limit')
        elif r['score'] < 60:
            output['liquidity_warning'].append(f'{t}: score={r["score"]} - liquidez moderada')
    
    with open(OUTPUT, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2)
    print(f'\n[OK] {len(results)} tickers evaluados')

if __name__ == '__main__':
    main()
