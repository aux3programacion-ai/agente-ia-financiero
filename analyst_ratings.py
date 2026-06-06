import json, os, sys, time, re, urllib.request, random

TICKERS_CORE = ['NVDA','MU','DELL','AVGO','DDOG','SMCI','SNOW','CRWD','NOW','TSM','ARM','OKTA','HPE','NTAP','CLS','AAPL','AMZN','GOOGL','META','MSFT','LLY','AMAT','LRCX','PANW','ORCL','HON','UBER','GE','COST','NEE']
DATA_DIR = os.environ.get('GITHUB_WORKSPACE', '.')
OUTPUT = os.path.join(DATA_DIR, 'Datos', 'analyst_ratings.json')
os.makedirs(os.path.join(DATA_DIR, 'Datos'), exist_ok=True)

def fetch_tipranks(ticker):
    """Fetch analyst ratings summary from TipRanks."""
    try:
        url = f'https://www.tipranks.com/stocks/{ticker}/forecast'
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'})
        with urllib.request.urlopen(req, timeout=10) as r:
            html = r.read().decode('utf-8', errors='replace')
        # Extract consensus rating
        buy_match = re.search(r'(\d+)\s*Buy', html, re.I)
        hold_match = re.search(r'(\d+)\s*Hold', html, re.I)
        sell_match = re.search(r'(\d+)\s*Sell', html, re.I)
        buys = int(buy_match.group(1)) if buy_match else 0
        holds = int(hold_match.group(1)) if hold_match else 0
        sells = int(sell_match.group(1)) if sell_match else 0
        total = buys + holds + sells
        if total > 0:
            consensus_score = (buys * 100 + holds * 50) / total
            return {
                'buys': buys, 'holds': holds, 'sells': sells,
                'consensus_score': round(consensus_score, 1),
                'consensus': 'buy' if consensus_score > 65 else 'hold' if consensus_score > 35 else 'sell'
            }
    except:
        pass
    return None

def fetch_briefing(ticker):
    """Fallback: scrape Briefing.com upgrades/downgrades."""
    try:
        url = f'https://www.briefing.com/calendar/upgrades-downgrades'
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=10) as r:
            html = r.read().decode('utf-8', errors='replace')
        # Briefing embeds ticker in upgrade/downgrade lines
        pattern = re.compile(rf'{ticker}.*?(upgrad|downgrad|initiat|reiterat)', re.I)
        matches = pattern.findall(html)
        if matches:
            upgrades = sum(1 for m in matches if 'upgrad' in m.lower())
            downgrades = sum(1 for m in matches if 'downgrad' in m.lower())
            return {'upgrades': upgrades, 'downgrades': downgrades}
    except:
        pass
    return None

def main():
    print('[Analyst Ratings] Obteniendo consenso de analistas...')
    results = {}
    for t in TICKERS_CORE:
        try:
            print(f'  {t}...', end=' ')
            rating = fetch_tipranks(t)
            if rating:
                results[t] = rating
                print(f'consenso={rating["consensus"]} ({rating["consensus_score"]:.0f}/100)')
            else:
                # Fallback: briefing
                brief = fetch_briefing(t)
                if brief:
                    net = brief.get('upgrades', 0) - brief.get('downgrades', 0)
                    results[t] = {'upgrades': brief['upgrades'], 'downgrades': brief['downgrades'], 'net': net, 'consensus': 'buy' if net > 0 else 'sell' if net < 0 else 'hold', 'consensus_score': 50 + net * 5}
                    print(f'briefing: {brief}')
                else:
                    results[t] = {'consensus': 'neutral', 'consensus_score': 50, 'note': 'No data'}
                    print(f'[no data]')
            time.sleep(0.3)
        except Exception as e:
            print(f'[!] {e}')
            results[t] = {'consensus': 'neutral', 'consensus_score': 50}
    
    output = {'timestamp': time.strftime('%Y-%m-%dT%H:%M:%S'), 'tickers': results}
    with open(OUTPUT, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2)
    print(f'\n[OK] Ratings guardados en {OUTPUT}')

if __name__ == '__main__':
    main()
