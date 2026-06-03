import json, os, sys, yfinance as yf, numpy as np, pandas as pd, time, math

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Datos')
os.makedirs(DATA_DIR, exist_ok=True)

TICKERS_CORE = ['NVDA','MU','DELL','AVGO','DDOG','SMCI','SNOW','CRWD','NOW','TSM','ARM','OKTA','HPE','NTAP','CLS','AAPL','AMZN','GOOGL','META','MSFT','LLY','AMAT','LRCX','PANW','ORCL','HON','UBER','GE','COST','NEE']

portfolio_path = os.path.join(DATA_DIR, 'portafolio_usuario.json')
portfolio_tickers = []
if os.path.exists(portfolio_path):
    with open(portfolio_path, 'r') as f:
        portfolio_tickers = json.load(f)
    if not isinstance(portfolio_tickers, list):
        portfolio_tickers = []

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
else:
    matriz = [[1.0]]

fecha_inicio = str(close.index[0].date()) if len(close) > 0 else ""
dias = len(returns)

result = {
    "timestamp": pd.Timestamp.now().isoformat(),
    "tickers": ticker_results,
    "correlacion": {
        "tickers": valid_tickers_for_corr,
        "matriz": matriz
    },
    "fecha_inicio": fecha_inicio,
    "dias": dias
}

output_path = os.path.join(DATA_DIR, 'analisis_riesgo.json')
with open(output_path, 'w') as f:
    json.dump(result, f, indent=2)
print(f"[+] Analisis de riesgo guardado en {output_path}")
