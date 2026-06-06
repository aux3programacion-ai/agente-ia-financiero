import json, os, sys, time, re, urllib.request

TICKERS_CORE = ['NVDA','MU','DELL','AVGO','DDOG','SMCI','SNOW','CRWD','NOW','TSM','ARM','OKTA','HPE','NTAP','CLS','AAPL','AMZN','GOOGL','META','MSFT','LLY','AMAT','LRCX','PANW','ORCL','HON','UBER','GE','COST','NEE']
DATA_DIR = os.environ.get('GITHUB_WORKSPACE', '.')
OUTPUT = os.path.join(DATA_DIR, 'Datos', 'short_interest.json')
os.makedirs(os.path.join(DATA_DIR, 'Datos'), exist_ok=True)

def fetch_short_interest_yahoo(ticker):
    try:
        tk = __import__('yfinance', fromlist=['Ticker']).Ticker(ticker)
        info = tk.info
        si = info.get('shortRatio', None) or info.get('shortPercentOfFloat', None)
        if si is not None:
            return {
                'short_ratio': round(info.get('shortRatio', 0), 2),
                'short_pct_float': round(info.get('shortPercentOfFloat', 0) * 100, 2) if info.get('shortPercentOfFloat') else None,
                'shares_short': info.get('sharesShort', 0),
                'days_to_cover': round(info.get('shortRatio', 0), 1)
            }
    except:
        pass
    return None

def fetch_short_interest_finviz(ticker):
    """Scrape Finviz for short interest."""
    try:
        url = f'https://finviz.com/quote.ashx?t={ticker}'
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'})
        with urllib.request.urlopen(req, timeout=10) as r:
            html = r.read().decode('utf-8', errors='replace')
        # Find Short Float in the table
        sf_match = re.search(r'Short Float[^0-9]*([\d.]+%)', html)
        si_match = re.search(r'Short Interest[^0-9]*([\d.]+[MK])', html)
        result = {}
        if sf_match:
            result['short_pct_float'] = sf_match.group(1)
        if si_match:
            result['short_interest'] = si_match.group(1)
        if result:
            return result
    except:
        pass
    return None

def main():
    import yfinance as yf
    print('[Short Interest] Obteniendo datos de cortos...')
    results = {}
    for t in TICKERS_CORE:
        try:
            print(f'  {t}...', end=' ')
            data = fetch_short_interest_yahoo(t)
            if data:
                results[t] = data
                print(f'SI%={data.get("short_pct_float","?")}% DTC={data.get("days_to_cover","?")}')
            else:
                data2 = fetch_short_interest_finviz(t)
                if data2:
                    results[t] = data2
                    print(f'Finviz: {data2.get("short_pct_float","?")}')
                else:
                    results[t] = {'note': 'No data'}
                    print('[no data]')
            time.sleep(0.5)
        except Exception as e:
            print(f'[!] {e}')
            results[t] = {'note': 'Error'}
    
    # Identify squeeze candidates
    squeeze_candidates = []
    for t, d in results.items():
        spf = d.get('short_pct_float', 0)
        if isinstance(spf, str):
            spf = float(spf.replace('%',''))
        dtc = d.get('days_to_cover', 0)
        if spf and spf > 20:
            squeeze_candidates.append({'ticker': t, 'short_pct_float': spf, 'days_to_cover': dtc})
    
    output = {
        'timestamp': time.strftime('%Y-%m-%dT%H:%M:%S'),
        'tickers': results,
        'squeeze_candidates': squeeze_candidates
    }
    with open(OUTPUT, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2)
    if squeeze_candidates:
        print(f'\n[!] Squeeze candidates: {", ".join(s["ticker"] for s in squeeze_candidates)}')
    print(f'[OK] Short interest guardado en {OUTPUT}')

if __name__ == '__main__':
    main()
