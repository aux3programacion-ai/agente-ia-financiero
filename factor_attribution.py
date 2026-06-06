import json, os, sys, time
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.linear_model import LinearRegression

TICKERS_CORE = ['NVDA','MU','DELL','AVGO','DDOG','SMCI','SNOW','CRWD','NOW','TSM','ARM','OKTA','HPE','NTAP','CLS','AAPL','AMZN','GOOGL','META','MSFT','LLY','AMAT','LRCX','PANW','ORCL','HON','UBER','GE','COST','NEE']
DATA_DIR = os.environ.get('GITHUB_WORKSPACE', '.')
OUTPUT = os.path.join(DATA_DIR, 'Datos', 'factor_attribution.json')
os.makedirs(os.path.join(DATA_DIR, 'Datos'), exist_ok=True)

FACTOR_ETFS = {
    'SPY': 'Market (Large Cap)',
    'IWM': 'Small Cap',
    'QQQ': 'Growth/Tech',
    'EFA': 'International',
    'AGG': 'Bonds',
    'TLT': 'Long Treasury',
    'XLF': 'Financials',
    'XLE': 'Energy',
    'XLK': 'Technology',
    'XLV': 'Healthcare',
    'XLI': 'Industrials',
    'XLP': 'Consumer Staples',
    'XLY': 'Consumer Discretionary',
    'XLU': 'Utilities',
    'XLRE': 'Real Estate',
    'KRE': 'Regional Banks',
    'SMH': 'Semiconductors',
    'IYR': 'REITs',
    'TLH': 'Treasury Long',
    'HYG': 'High Yield'
}

def main():
    print('[Factor Attribution] Regresando retornos contra factores...')
    lookback = 252
    
    # Download portfolio tickers + factors
    all_tickers = list(set(['SPY'] + list(FACTOR_ETFS.keys()) + TICKERS_CORE))
    try:
        data = yf.download(all_tickers, period='1y', interval='1d', progress=False, auto_adjust=True)
        if isinstance(data.columns, pd.MultiIndex):
            close = data['Close']
        else:
            close = data
    except Exception as e:
        print(f'[!] Download error: {e}')
        return
    
    # Portfolio weights from paper_trading or equal-weight
    port_path = os.path.join(DATA_DIR, 'Datos', 'paper_trading.json')
    weights = {}
    if os.path.exists(port_path):
        try:
            pt = json.load(open(port_path))
            holdings = pt.get('holdings', {})
            total_val = sum(h.get('valor', 0) for h in holdings.values()) or 1
            for t, h in holdings.items():
                weights[t.upper()] = h.get('valor', 0) / total_val
        except:
            pass
    if not weights:
        w = 1 / len(TICKERS_CORE)
        for t in TICKERS_CORE:
            weights[t] = w
    
    # Build portfolio return series
    available = [t for t in weights if t in close.columns]
    if not available:
        print('[!] No tickers available')
        return
    
    port_ret = pd.Series(0.0, index=close.index)
    for t in available:
        ret = close[t].pct_change()
        port_ret += ret * weights[t]
    port_ret = port_ret.dropna()
    
    # Download factors: Mkt-RF (SPY), SMB (IWM-SPY), HML, MOM
    factor_returns = {}
    # Market factor
    if 'SPY' in close:
        factor_returns['Market'] = close['SPY'].pct_change()
    # Size factor: IWM - SPY
    if 'IWM' in close and 'SPY' in close:
        factor_returns['Size (SMB)'] = close['IWM'].pct_change() - close['SPY'].pct_change()
    # Value factor proxy: IWD/IWF not available, use sector dispersion
    # Momentum: SPY 12m - 1m return
    if 'SPY' in close:
        spy_ret = close['SPY'].pct_change()
        factor_returns['Momentum'] = spy_ret.rolling(252).mean() - spy_ret.rolling(21).mean()
    # Quality: XLV + XLP (defensive) vs XLY (cyclical)
    if all(e in close for e in ['XLV','XLP','XLY']):
        defensive = (close['XLV'].pct_change() + close['XLP'].pct_change()) / 2
        cyclical = close['XLY'].pct_change()
        factor_returns['Quality'] = defensive - cyclical
    # Low Vol: XLU + XLP vs SPY
    if all(e in close for e in ['XLU','XLP','SPY']):
        low_vol = (close['XLU'].pct_change() + close['XLP'].pct_change()) / 2
        factor_returns['Low Vol'] = low_vol - close['SPY'].pct_change()
    
    df_factors = pd.DataFrame(factor_returns).dropna()
    aligned = port_ret.align(df_factors, join='inner')[0]
    df_factors = df_factors.loc[aligned.index]
    
    if len(df_factors) < 20:
        print(f'[!] Only {len(df_factors)} aligned observations')
        return
    
    X = df_factors.values
    y = aligned.values
    
    lr = LinearRegression()
    lr.fit(X, y)
    
    factor_loadings = {}
    for i, col in enumerate(df_factors.columns):
        factor_loadings[col] = round(float(lr.coef_[i]), 4)
    factor_loadings['Alpha (Intercept)'] = round(float(lr.intercept_), 6)
    factor_loadings['R_squared'] = round(float(lr.score(X, y)), 4)
    
    # Sector exposure
    sector_exposure = {}
    sector_map = {
        'XLF': ['Financials'], 'XLE': ['Energy'], 'XLK': ['Technology'],
        'XLV': ['Healthcare'], 'XLI': ['Industrials'], 'XLP': ['Staples'],
        'XLY': ['Discretionary'], 'XLU': ['Utilities'], 'SMH': ['Semiconductors']
    }
    for etf, sectors in sector_map.items():
        if etf in close:
            sec_ret = close[etf].pct_change()
            sec_ret_aligned = sec_ret.reindex(aligned.index).dropna()
            aligned_port = aligned.loc[sec_ret_aligned.index]
            if len(sec_ret_aligned) > 20:
                r2 = 1 - (np.var(aligned_port.values - sec_ret_aligned.values) / max(np.var(aligned_port.values), 1e-10))
                for s in sectors:
                    sector_exposure[s] = round(max(0, min(1, r2)), 4)
    
    result = {
        'timestamp': time.strftime('%Y-%m-%dT%H:%M:%S'),
        'factor_loadings': factor_loadings,
        'sector_exposure': sector_exposure,
        'n_observations': len(df_factors),
        'interpretation': {}
    }
    
    # Interpretation
    mkt_beta = factor_loadings.get('Market', 0)
    if mkt_beta > 1.2:
        result['interpretation']['market'] = 'Alto beta (>1.2): portafolio agresivo, amplifica movimientos del mercado'
    elif mkt_beta > 0.8:
        result['interpretation']['market'] = 'Beta neutral (0.8-1.2): portafolio sigue al mercado'
    else:
        result['interpretation']['market'] = f'Bajo beta ({mkt_beta}): portafolio defensivo'
    
    if factor_loadings.get('Size (SMB)', 0) > 0.2:
        result['interpretation']['size'] = 'Sesgo small-cap: mayor riesgo/retorno potencial'
    elif factor_loadings.get('Size (SMB)', 0) < -0.2:
        result['interpretation']['size'] = 'Sesgo large-cap: estabilidad relativa'
    
    if factor_loadings.get('Quality', 0) > 0.2:
        result['interpretation']['quality'] = 'Sesgo calidad/defensivo: menor volatilidad'
    
    if sector_exposure.get('Technology', 0) > 0.5 or sector_exposure.get('Semiconductors', 0) > 0.5:
        result['interpretation']['sector'] = 'Alta concentracion tech/semiconductores'
    
    with open(OUTPUT, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2)
    
    print(f'[OK] R²={factor_loadings["R_squared"]:.3f}, Beta={factor_loadings.get("Market", "N/A")}')
    print(f'  Factores: {", ".join(f"{k}={v}" for k,v in factor_loadings.items() if k not in ("Alpha (Intercept)","R_squared"))}')

if __name__ == '__main__':
    main()
