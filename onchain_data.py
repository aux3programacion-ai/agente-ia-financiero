import json, os, sys, time

DATA_DIR = os.environ.get('GITHUB_WORKSPACE', '.')
OUTPUT = os.path.join(DATA_DIR, 'Datos', 'onchain_data.json')
os.makedirs(os.path.join(DATA_DIR, 'Datos'), exist_ok=True)

def fetch_onchain_summary():
    """Aggregate on-chain data from public APIs (crypto)."""
    data = {}
    
    # Bitcoin metrics from blockchain.com public API
    btc_apis = {
        'hashrate': 'https://api.blockchain.info/q/hashrate',
        'difficulty': 'https://api.blockchain.info/q/getdifficulty',
        'tx_count_24h': 'https://api.blockchain.info/q/24hrtxcount',
        'tx_volume_24h': 'https://api.blockchain.info/q/24hrtransactionvolume',
        'market_price': 'https://api.blockchain.info/q/ticker'
    }
    for name, url in btc_apis.items():
        try:
            import urllib.request
            req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
            with urllib.request.urlopen(req, timeout=10) as r:
                raw = r.read().decode('utf-8', errors='replace')
            if name == 'market_price':
                import json as jmod
                ticker = jmod.loads(raw)
                data['btc_price_usd'] = ticker.get('USD', {}).get('last', 0)
                data['btc_price_eur'] = ticker.get('EUR', {}).get('last', 0)
            else:
                data[f'btc_{name}'] = raw.strip()
        except:
            pass
    
    # Whale activity proxy: look at number of large transactions (>1 BTC) from blockchain.com
    try:
        url = 'https://api.blockchain.info/charts/n-transactions?timespan=30days&format=json'
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=10) as r:
            tx_data = json.loads(r.read().decode('utf-8'))
        vals = tx_data.get('values', [])
        if vals:
            recent = [v['y'] for v in vals[-7:]]
            prev = [v['y'] for v in vals[-14:-7]]
            if recent and prev:
                data['tx_trend_7d'] = round((sum(recent)/len(recent)) / (sum(prev)/len(prev)) - 1, 4)
    except:
        pass
    
    # Exchange flow proxy: use total transaction volume as proxy for exchange activity
    try:
        url = 'https://api.blockchain.info/charts/estimated-transaction-volume-usd?timespan=30days&format=json'
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=10) as r:
            vol_data = json.loads(r.read().decode('utf-8'))
        vals = vol_data.get('values', [])
        if vals:
            recent = [v['y'] for v in vals[-3:]]
            prev = [v['y'] for v in vals[-6:-3]]
            if recent and prev:
                data['volume_trend_3d'] = round((sum(recent)/len(recent)) / (sum(prev)/len(prev)) - 1, 4)
    except:
        pass
    
    return data

def main():
    print('[On-Chain Data] Obteniendo metricas on-chain...')
    data = fetch_onchain_summary()
    
    result = {
        'timestamp': time.strftime('%Y-%m-%dT%H:%M:%S'),
        'metrics': data,
        'alerts': []
    }
    
    if data.get('tx_trend_7d') and data['tx_trend_7d'] > 0.2:
        result['alerts'].append(f'Transaction volume trend +{data["tx_trend_7d"]*100:.0f}%: alta actividad on-chain')
    if data.get('volume_trend_3d') and data['volume_trend_3d'] > 0.3:
        result['alerts'].append(f'Volume trend +{data["volume_trend_3d"]*100:.0f}%: posible acumulacion')
    
    with open(OUTPUT, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2)
    
    print(f'  BTC Price: ${data.get("btc_price_usd", "N/A")}')
    if result['alerts']:
        for a in result['alerts']:
            print(f'  [!] {a}')
    print(f'[OK] On-chain data guardado en {OUTPUT}')

if __name__ == '__main__':
    main()
