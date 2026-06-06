import json, os, sys, yfinance as yf, numpy as np, pandas as pd, time, math

from portafolio_utils import cargar_portafolio

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Datos')
os.makedirs(DATA_DIR, exist_ok=True)

TICKERS_CORE = ['NVDA','MU','DELL','AVGO','DDOG','SMCI','SNOW','CRWD','NOW','TSM','ARM','OKTA','HPE','NTAP','CLS','AAPL','AMZN','GOOGL','META','MSFT','LLY','AMAT','LRCX','PANW','ORCL','HON','UBER','GE','COST','NEE']

portfolio_tickers = cargar_portafolio(os.path.dirname(DATA_DIR))

all_tickers = list(dict.fromkeys(TICKERS_CORE + portfolio_tickers))
print(f"[+] Total tickers a procesar: {len(all_tickers)}")

print("[+] Descargando datos OHLCV (6 meses)...")
try:
    data = yf.download(all_tickers + ['^GSPC'], period='6mo', interval='1d', group_by='ticker', progress=False, auto_adjust=True)
except Exception as e:
    print(f"[!] Error en descarga yfinance: {e}")
    result = {
        "timestamp": pd.Timestamp.now().isoformat(),
        "tickers": {},
        "correlacion": {"tickers": all_tickers, "matriz": [[1.0 if i == j else 0.0 for j in range(len(all_tickers))] for i in range(len(all_tickers))]},
        "fecha_inicio": "",
        "dias": 0
    }
    output_path = os.path.join(DATA_DIR, 'analisis_riesgo.json')
    with open(output_path, 'w') as f:
        json.dump(result, f, indent=2)
    print(f"[!] Salida de emergencia guardada en {output_path}")
    sys.exit(1)

if isinstance(data.columns, pd.MultiIndex):
    close = data.xs('Close', axis=1, level=1)
    have_multi = True
else:
    close = data
    have_multi = False

close.columns = [str(c) for c in close.columns]
spx_col = None
for c in close.columns:
    if 'GSPC' in c or c.strip() == '^GSPC':
        spx_col = c
        break

if spx_col is None:
    print("[!] No se encontro ^GSPC en los datos")
    spx_col = '^GSPC'
    close[spx_col] = np.nan

returns = close.pct_change().dropna()

ticker_results = {}
valid_tickers_for_corr = []

for t in all_tickers:
    t_str = str(t)
    if t_str not in close.columns:
        print(f"  [~] {t}: sin datos, omitiendo")
        continue
    vals = close[t_str].dropna()
    rets = returns[t_str].dropna()
    if len(rets) < 5:
        print(f"  [~] {t}: pocos retornos ({len(rets)}), omitiendo")
        continue
    valid_tickers_for_corr.append(t_str)
    var_95 = float(np.percentile(rets, 5))
    var_99 = float(np.percentile(rets, 1))
    mean_daily = float(rets.mean())
    std_daily = float(rets.std())
    sharpe = float((mean_daily * 252 - 0.05) / (std_daily * math.sqrt(252))) if std_daily > 0 else 0.0
    vol_anual = float(std_daily * math.sqrt(252))
    cummax = vals.cummax()
    drawdown = (vals - cummax) / cummax
    max_dd = float(abs(drawdown.min()))
    spx_rets = returns[spx_col].dropna()
    common_idx = rets.index.intersection(spx_rets.index)
    if len(common_idx) > 5:
        r_common = rets.loc[common_idx]
        s_common = spx_rets.loc[common_idx]
        cov = float(np.cov(r_common, s_common)[0, 1])
        var_spx = float(np.var(s_common))
        beta = cov / var_spx if var_spx > 0 else 1.0
    else:
        beta = 1.0
    ticker_results[t_str] = {
        "var_95": round(abs(var_95), 4),
        "var_99": round(abs(var_99), 4),
        "sharpe": round(sharpe, 4),
        "max_drawdown": round(max_dd, 4),
        "vol_anual": round(vol_anual, 4),
        "beta": round(beta, 4)
    }
    print(f"  [OK] {t_str}: VaR95={abs(var_95):.4f} VaR99={abs(var_99):.4f} Sharpe={sharpe:.4f} DD={max_dd:.4f} Vol={vol_anual:.4f} Beta={beta:.4f}")

n = len(valid_tickers_for_corr)
if n > 1:
    corr_data = returns[valid_tickers_for_corr].corr()
    matriz = [[round(float(corr_data.iloc[i, j]), 4) for j in range(n)] for i in range(n)]
    
    # --- Correlation Regime Detection ---
    # Compute rolling average pairwise correlation
    rolling_corr = returns[valid_tickers_for_corr].rolling(60, min_periods=30).corr()
    # Get average pairwise correlation over time (upper triangle mean)
    n_assets = len(valid_tickers_for_corr)
    avg_corr_history = {}
    for idx in rolling_corr.index.levels[0] if isinstance(rolling_corr.index, pd.MultiIndex) else rolling_corr.index:
        try:
            if isinstance(rolling_corr.index, pd.MultiIndex):
                corr_slice = rolling_corr.loc[idx]
            else:
                corr_slice = rolling_corr
            if isinstance(corr_slice, pd.DataFrame) and len(corr_slice) == n_assets:
                triu = np.triu_indices(n_assets, k=1)
                avg_c = np.mean(corr_slice.values[triu])
                avg_corr_history[str(idx.date())] = round(float(avg_c), 4)
        except:
            pass
    
    current_avg_corr = np.mean([matriz[i][j] for i in range(n) for j in range(i+1, n)])
    recent_avg = current_avg_corr
    # Compare with 6 months ago
    corr_values = list(avg_corr_history.values())
    if len(corr_values) > 60:
        old_avg = np.mean(corr_values[:30])
        corr_regime_change = recent_avg - old_avg
    else:
        corr_regime_change = 0.0
    
    corr_regime = 'normal'
    if corr_regime_change > 0.15:
        corr_regime = 'high_correlation_crisis'
    elif corr_regime_change > 0.08:
        corr_regime = 'rising_correlation'
    elif corr_regime_change < -0.1:
        corr_regime = 'falling_correlation'
    
    print(f'  [Corr Regime] Avg pairwise corr={recent_avg:.3f}, change={corr_regime_change:.3f} -> {corr_regime}')
else:
    matriz = [[1.0]]
    avg_corr_history = {}
    corr_regime = 'unknown'
    recent_avg = 0

fecha_inicio = str(close.index[0].date()) if len(close) > 0 else ""
dias = len(returns)

result = {
    "timestamp": pd.Timestamp.now().isoformat(),
    "tickers": ticker_results,
    "correlacion": {
        "tickers": valid_tickers_for_corr,
        "matriz": matriz,
        "avg_pairwise_corr": round(float(recent_avg), 4) if n > 1 else 1.0,
        "corr_history": avg_corr_history,
        "corr_regime": corr_regime
    },
    "fecha_inicio": fecha_inicio,
    "dias": dias
}

output_path = os.path.join(DATA_DIR, 'analisis_riesgo.json')
with open(output_path, 'w') as f:
    json.dump(result, f, indent=2)
print(f"[+] Analisis de riesgo guardado en {output_path}")
