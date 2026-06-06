import json, os, sys, time, math
import numpy as np
from portafolio_utils import cargar_portafolio

TICKERS_CORE = ['NVDA','MU','DELL','AVGO','DDOG','SMCI','SNOW','CRWD','NOW','TSM','ARM','OKTA','HPE','NTAP','CLS','AAPL','AMZN','GOOGL','META','MSFT','LLY','AMAT','LRCX','PANW','ORCL','HON','UBER','GE','COST','NEE']
DATA_DIR = os.environ.get('GITHUB_WORKSPACE', '.')
OUTPUT = os.path.join(DATA_DIR, 'Datos', 'concentration_risk.json')
os.makedirs(os.path.join(DATA_DIR, 'Datos'), exist_ok=True)

SECTOR_MAP = {
    'NVDA':'Semiconductors','MU':'Semiconductors','AVGO':'Semiconductors','TSM':'Semiconductors','AMAT':'Semiconductors','LRCX':'Semiconductors','SMH':'Semiconductors',
    'DELL':'Hardware','HPE':'Hardware','NTAP':'Hardware','AAPL':'Hardware',
    'DDOG':'Software','SNOW':'Software','CRWD':'Software','NOW':'Software','OKTA':'Software','PANW':'Software','ORCL':'Software','MSFT':'Software',
    'SMCI':'Hardware','ARM':'Semiconductors','CLS':'Hardware',
    'AMZN':'E-Commerce','GOOGL':'Internet','META':'Internet','UBER':'Transportation',
    'LLY':'Pharma','HON':'Industrials','GE':'Industrials','COST':'Retail','NEE':'Utilities'
}

def main():
    print('[Concentration Risk] Evaluando concentracion del portafolio...')
    port_path = os.path.join(DATA_DIR, 'Datos', 'paper_trading.json')
    
    result = {
        'timestamp': time.strftime('%Y-%m-%dT%H:%M:%S'),
        'ticker_concentration': {},
        'sector_concentration': {},
        'alerts': [],
        'herfindahl_index': 0,
        'max_exposure_pct': 0
    }
    
    weights = {}
    if os.path.exists(port_path):
        try:
            pt = json.load(open(port_path))
            holdings = pt.get('holdings', {})
            total_val = sum(h.get('valor_costo', 0) for h in holdings.values())
            if total_val <= 0:
                total_val = sum(h.get('valor', 0) for h in holdings.values())
            if total_val > 0:
                for t, h in holdings.items():
                    w = h.get('valor_costo', h.get('valor', 0)) / total_val
                    weights[t.upper()] = w
        except Exception as e:
            print(f'[!] Error reading portfolio: {e}')
    
    if not weights:
        w = 1 / len(TICKERS_CORE)
        for t in TICKERS_CORE:
            weights[t] = w
    
    # Ticker concentration
    sorted_weights = sorted(weights.items(), key=lambda x: x[1], reverse=True)
    total_w = sum(weights.values()) or 1
    result['max_exposure_pct'] = round(sorted_weights[0][1] / total_w * 100, 1)
    
    top5_pct = sum(w for _, w in sorted_weights[:5]) / total_w * 100
    top10_pct = sum(w for _, w in sorted_weights[:10]) / total_w * 100
    result['top5_exposure_pct'] = round(top5_pct, 1)
    result['top10_exposure_pct'] = round(top10_pct, 1)
    
    # Herfindahl-Hirschman Index
    hhi = sum((w / total_w * 100) ** 2 for w in weights.values())
    result['herfindahl_index'] = round(hhi, 1)
    if hhi > 2500:
        result['alerts'].append(f'HHI={hhi:.0f} > 2500: portafolio altamente concentrado')
    elif hhi > 1500:
        result['alerts'].append(f'HHI={hhi:.0f} > 1500: portafolio moderadamente concentrado')
    
    # Sector concentration
    sector_weights = {}
    for t, w in weights.items():
        sector = SECTOR_MAP.get(t.upper(), 'Other')
        sector_weights[sector] = sector_weights.get(sector, 0) + w
    
    sorted_sectors = sorted(sector_weights.items(), key=lambda x: x[1], reverse=True)
    result['sector_concentration'] = {s: round(w / total_w * 100, 1) for s, w in sorted_sectors}
    
    max_sector = sorted_sectors[0] if sorted_sectors else ('None', 0)
    result['max_sector'] = max_sector[0]
    result['max_sector_pct'] = round(max_sector[1] / total_w * 100, 1)
    
    if max_sector[1] / total_w > 0.4:
        result['alerts'].append(f'Sector {max_sector[0]} con {max_sector[1]/total_w*100:.0f}% > 40%: alto riesgo sectorial')
    elif max_sector[1] / total_w > 0.25:
        result['alerts'].append(f'Sector {max_sector[0]} con {max_sector[1]/total_w*100:.0f}% > 25%: monitorear')
    
    # Individual ticker alerts
    for t, w in sorted_weights[:3]:
        pct = w / total_w * 100
        if pct > 15:
            result['alerts'].append(f'{t}: {pct:.1f}% > 15%: posicion muy grande')
    
    # Factor concentration (sector-level)
    sector_hhi = sum((w / total_w * 100) ** 2 for w in sector_weights.values())
    result['sector_hhi'] = round(sector_hhi, 1)
    
    with open(OUTPUT, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2)
    
    print(f'  HHI={result["herfindahl_index"]:.0f} | Max ticker={result["max_exposure_pct"]:.1f}% | Max sector={result["max_sector"]} ({result["max_sector_pct"]:.1f}%)')
    if result['alerts']:
        for a in result['alerts'][:3]:
            print(f'  [!] {a}')

if __name__ == '__main__':
    main()
