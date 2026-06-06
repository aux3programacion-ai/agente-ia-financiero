import json, os, sys, time
import numpy as np
import pandas as pd
import yfinance as yf

TICKERS_CORE = ['NVDA','MU','DELL','AVGO','DDOG','SMCI','SNOW','CRWD','NOW','TSM','ARM','OKTA','HPE','NTAP','CLS','AAPL','AMZN','GOOGL','META','MSFT','LLY','AMAT','LRCX','PANW','ORCL','HON','UBER','GE','COST','NEE']
DATA_DIR = os.environ.get('GITHUB_WORKSPACE', '.')
OUTPUT = os.path.join(DATA_DIR, 'Datos', 'drawdown_stop.json')
os.makedirs(os.path.join(DATA_DIR, 'Datos'), exist_ok=True)

def compute_drawdown(series):
    peak = series.expanding().max()
    dd = (series - peak) / peak
    return dd

def main():
    print('[Drawdown Stop] Evaluando drawdown y senales de parada...')
    port_path = os.path.join(DATA_DIR, 'Datos', 'paper_trading.json')
    signal = {
        'timestamp': time.strftime('%Y-%m-%dT%H:%M:%S'),
        'drawdown_pct': 0,
        'max_drawdown_30d': 0,
        'triggered': False,
        'action': 'none',
        'alerts': []
    }
    
    # Try to read portfolio value history
    if os.path.exists(port_path):
        try:
            pt = json.load(open(port_path))
            holdings = pt.get('holdings', {})
            total_value = sum(h.get('valor', 0) for h in holdings.values())
            cash = pt.get('cash', 0)
            equity = total_value + cash
            peak_key = 'peak_value'
            peak = pt.get(peak_key, equity)
            if equity > 0:
                dd = (equity - peak) / peak * 100 if peak > 0 else 0
                signal['drawdown_pct'] = round(dd, 2)
            # Rebuild value history from logs
            log = pt.get('log', [])
            if log:
                vals = [l.get('equity', 0) for l in log if l.get('equity')]
                if vals:
                    dd_series = np.array(vals) / np.maximum.accumulate(np.array(vals)) - 1
                    signal['max_drawdown_30d'] = round(float(np.min(dd_series[-30:])) * 100, 2) if len(dd_series) >= 30 else round(float(np.min(dd_series)) * 100, 2)
        except Exception as e:
            print(f'[!] Error leyendo portfolio: {e}')
    
    # Download SPY to compute market drawdown as fallback
    try:
        spy = yf.download('SPY', period='6mo', interval='1d', progress=False, auto_adjust=True)
        if spy is not None and not spy.empty:
            close = spy['Close'].dropna()
            dd_spy = compute_drawdown(close)
            current_market_dd = float(dd_spy.iloc[-1] * 100)
            max_market_dd = float(dd_spy.min() * 100)
            signal['market_drawdown_pct'] = round(current_market_dd, 2)
            signal['market_max_drawdown_6m'] = round(max_market_dd, 2)
    except Exception as e:
        print(f'[!] Error SPY: {e}')
    
    # Decision logic
    current_dd = abs(signal.get('drawdown_pct', 0))
    market_dd = abs(signal.get('market_drawdown_pct', 0))
    dd_used = max(current_dd, market_dd)
    
    if dd_used > 15:
        signal['triggered'] = True
        signal['action'] = 'reduce_exposure_50'
        signal['alerts'].append(f'Drawdown {dd_used:.1f}% > 15%: reducir exposicion 50%')
    elif dd_used > 10:
        signal['triggered'] = True
        signal['action'] = 'reduce_exposure_25'
        signal['alerts'].append(f'Drawdown {dd_used:.1f}% > 10%: reducir exposicion 25%')
    elif dd_used > 5:
        signal['action'] = 'monitor'
        signal['alerts'].append(f'Drawdown {dd_used:.1f}% > 5%: monitorear')
    else:
        signal['action'] = 'normal'
    
    with open(OUTPUT, 'w', encoding='utf-8') as f:
        json.dump(signal, f, indent=2)
    
    print(f'  Drawdown: {signal["drawdown_pct"]:.1f}% | Mercado: {signal.get("market_drawdown_pct",0):.1f}% | Accion: {signal["action"]}')
    if signal['alerts']:
        for a in signal['alerts']:
            print(f'    {a}')

if __name__ == '__main__':
    main()
