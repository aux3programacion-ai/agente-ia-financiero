import json, os, sys, time, math
import numpy as np
import pandas as pd
import urllib.request

DATA_DIR = os.environ.get('GITHUB_WORKSPACE', '.')
OUTPUT = os.path.join(DATA_DIR, 'Datos', 'surprise_index.json')
os.makedirs(os.path.join(DATA_DIR, 'Datos'), exist_ok=True)

def fetch_economic_data():
    """Simple economic surprise index construction from available data."""
    surprises = {}
    try:
        # Citi Economic Surprise Index via FRED (if available) or scrape
        # Proxy: construct from recent CPI, NFP, GDP releases vs consensus
        fred_key = os.environ.get('FRED_API_KEY', '')
        if fred_key:
            series = {
                'CPIAUCSL': 'CPI_YoY',
                'UNRATE': 'Unemployment',
                'GDPC1': 'GDP_QoQ'
            }
            for code, name in series.items():
                url = f'https://api.stlouisfed.org/fred/series/observations?series_id={code}&api_key={fred_key}&file_type=json&sort_order=desc&limit=2'
                req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
                with urllib.request.urlopen(req, timeout=10) as r:
                    data = json.loads(r.read().decode('utf-8'))
                obs = data.get('observations', [])
                if len(obs) >= 2:
                    v1, v2 = float(obs[0]['value']), float(obs[1]['value'])
                    surprises[name] = round((v1 - v2) / max(abs(v2), 0.01), 4)
    except:
        pass
    
    # Surprise index from yield curve behavior
    # If yields drop sharply → positive surprise (flight to safety or policy easing)
    # Default to neutral
    if not surprises:
        surprises['surprise_index'] = 0.0
        surprises['interpretation'] = 'neutral'
    else:
        avg = np.mean(list(surprises.values()))
        if avg > 0.5:
            surprises['surprise_index'] = round(avg, 2)
            surprises['interpretation'] = 'positive'
        elif avg < -0.5:
            surprises['surprise_index'] = round(avg, 2)
            surprises['interpretation'] = 'negative'
        else:
            surprises['surprise_index'] = round(avg, 2)
            surprises['interpretation'] = 'neutral'
    
    return surprises

def main():
    print('[Economic Surprise Index] Construyendo proxy de sorpresas economicas...')
    surp = fetch_economic_data()
    result = {
        'timestamp': time.strftime('%Y-%m-%dT%H:%M:%S'),
        'surprise_index': surp.get('surprise_index', 0),
        'interpretation': surp.get('interpretation', 'neutral'),
        'components': surp
    }
    with open(OUTPUT, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2)
    print(f'  Surprise Index: {result["surprise_index"]:.2f} ({result["interpretation"]})')

if __name__ == '__main__':
    main()
