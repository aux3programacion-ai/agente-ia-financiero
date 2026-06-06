import json, os, time, urllib.request, re, sys, math
from datetime import datetime

TICKERS_CORE = ['NVDA','MU','DELL','AVGO','DDOG','SMCI','SNOW','CRWD','NOW','TSM','ARM','OKTA','HPE','NTAP','CLS','AAPL','AMZN','GOOGL','META','MSFT','LLY','AMAT','LRCX','PANW','ORCL','HON','UBER','GE','COST','NEE']
DATA_DIR = os.environ.get('GITHUB_WORKSPACE', '.')
OUTPUT = os.path.join(DATA_DIR, 'Datos', 'earnings_quality.json')
os.makedirs(os.path.join(DATA_DIR, 'Datos'), exist_ok=True)

def fetch_sec_filing(ticker):
    """Fetch latest 10-K or 10-Q from SEC EDGAR."""
    try:
        cik_url = f'https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&CIK={ticker}&output=json'
        req = urllib.request.Request(cik_url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=10) as r:
            cik_data = json.loads(r.read().decode('utf-8'))
        cik_raw = cik_data.get('query', {}).get('results', {}).get('row', [])
        if not cik_raw:
            return None
        cik = str(cik_raw[0].get('CIK', '')).zfill(10)
        filings_url = f'https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json'
        req2 = urllib.request.Request(filings_url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req2, timeout=15) as r2:
            facts = json.loads(r2.read().decode('utf-8'))
        return facts.get('facts', {})
    except Exception:
        return None

def compute_earnings_quality(ticker):
    """Score 0-100 based on accruals ratio, earnings persistence, and revenue quality."""
    facts = fetch_sec_filing(ticker)
    if not facts:
        return None
    
    score = 50
    details = {}
    
    # Accruals ratio: (Net Income - CFO) / Total Assets
    try:
        us_gaap = facts.get('us-gaap', {})
        net_income_raw = us_gaap.get('NetIncomeLoss', {}).get('units', {}).get('USD', [])
        cfo_raw = us_gaap.get('NetCashProvidedByOperatingActivities', {}).get('units', {}).get('USD', [])
        assets_raw = us_gaap.get('Assets', {}).get('units', {}).get('USD', [])
        total_debt_raw = us_gaap.get('LongTermDebt', {}).get('units', {}).get('USD', [])
        revenue_raw = us_gaap.get('Revenues', {}).get('units', {}).get('USD', [])
    except KeyError:
        try:
            ifrs = facts.get('ifrs-full', {})
            net_income_raw = ifrs.get('ProfitLoss', {}).get('units', {}).get('USD', [])
            cfo_raw = ifrs.get('CashFlowsFromUsedInOperatingActivities', {}).get('units', {}).get('USD', [])
            assets_raw = ifrs.get('Assets', {}).get('units', {}).get('USD', [])
            total_debt_raw = ifrs.get('LongTermDebt', {}).get('units', {}).get('USD', [])
            revenue_raw = ifrs.get('Revenue', {}).get('units', {}).get('USD', [])
        except KeyError:
            return None
    
    def get_latest(arr):
        annual = [x for x in arr if x.get('fp') == 'FY' and x.get('frame') and 'CY' in x.get('frame','')]
        if annual:
            return sorted(annual, key=lambda x: x.get('end',''))[-1].get('val', 0)
        sorted_arr = sorted(arr, key=lambda x: x.get('end',''), reverse=True)
        return sorted_arr[0].get('val', 0) if sorted_arr else 0
    
    ni = get_latest(net_income_raw)
    cfo = get_latest(cfo_raw)
    assets = get_latest(assets_raw)
    total_debt = get_latest(total_debt_raw)
    revenue = get_latest(revenue_raw)
    
    if assets > 0:
        accruals = (ni - cfo) / assets
        if abs(accruals) < 0.02:
            score += 20
        elif abs(accruals) < 0.05:
            score += 10
        elif abs(accruals) > 0.15:
            score -= 20
        elif abs(accruals) > 0.10:
            score -= 10
        details['accruals_ratio'] = round(accruals, 4)
    
    # Revenue quality: CapEx / CFO (lower = higher quality)
    capex_raw = us_gaap.get('CapitalExpendituresIncurredButNotYetPaid', {}).get('units', {}).get('USD', [])
    if not capex_raw:
        capex_raw = us_gaap.get('PaymentsToAcquirePropertyPlantAndEquipment', {}).get('units', {}).get('USD', [])
    capex = get_latest(capex_raw) if capex_raw else 0
    if cfo > 0:
        capex_ratio = capex / cfo
        if capex_ratio < 0.3:
            score += 10
        elif capex_ratio < 0.5:
            score += 5
        elif capex_ratio > 0.8:
            score -= 10
        elif capex_ratio > 0.6:
            score -= 5
        details['capex_to_cfo'] = round(capex_ratio, 4)
    
    # Earnings persistence: smoothness of earnings growth
    if len(net_income_raw) >= 4:
        e_vals = []
        for x in sorted(net_income_raw, key=lambda x: x.get('end','')):
            if x.get('fp') in ('FY', None) or 'CY' in x.get('frame',''):
                e_vals.append(x.get('val', 0))
        if len(e_vals) >= 4:
            e_vals = e_vals[-8:]
            growths = []
            for i in range(1, len(e_vals)):
                if e_vals[i-1] > 0:
                    growths.append(e_vals[i] / e_vals[i-1] - 1)
            if growths:
                cv = np.std(growths) / max(abs(np.mean(growths)), 0.01)
                if cv < 0.3:
                    score += 10
                elif cv < 0.6:
                    score += 5
                elif cv > 1.5:
                    score -= 10
                elif cv > 1.0:
                    score -= 5
                details['earnings_cv'] = round(cv, 4)
    
    results = {
        'ticker': ticker,
        'quality_score': max(0, min(100, score)),
        'details': details,
        'timestamp': time.strftime('%Y-%m-%dT%H:%M:%S')
    }
    return results

def main():
    TICKERS = TICKERS_CORE
    results = {}
    for t in TICKERS:
        try:
            print(f'  {t}...', end=' ')
            r = compute_earnings_quality(t)
            if r:
                results[t] = r
                print(f'score={r["quality_score"]}')
            else:
                print('[SKIP] no data')
            time.sleep(0.5)
        except Exception as e:
            print(f'[!] {e}')
    with open(OUTPUT, 'w', encoding='utf-8') as f:
        json.dump({'timestamp': time.strftime('%Y-%m-%dT%H:%M:%S'), 'tickers': results}, f, indent=2, ensure_ascii=False)
    print(f'\n[OK] Earnings quality saved to {OUTPUT}')

if __name__ == '__main__':
    main()
