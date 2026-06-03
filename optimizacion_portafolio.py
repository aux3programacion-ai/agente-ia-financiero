import json, os, sys, yfinance as yf, numpy as np, pandas as pd, time, datetime, warnings, math

warnings.filterwarnings('ignore')

DATA_DIR = os.environ.get('GITHUB_WORKSPACE', '.')
OUTPUT_PATH = os.path.join(DATA_DIR, 'Datos', 'optimizacion_portafolio.json')
os.makedirs(os.path.join(DATA_DIR, 'Datos'), exist_ok=True)

TICKERS_CORE = ['NVDA','MU','DELL','AVGO','DDOG','SMCI','SNOW','CRWD','NOW','TSM','ARM','OKTA','HPE','NTAP','CLS','AAPL','AMZN','GOOGL','META','MSFT','LLY','AMAT','LRCX','PANW','ORCL','HON','UBER','GE','COST','NEE']

portfolio_path = os.path.join(DATA_DIR, 'Datos', 'portafolio_usuario.json')
portfolio_tickers = []
if os.path.exists(portfolio_path):
    with open(portfolio_path, 'r') as f:
        portfolio_tickers = json.load(f)
    if not isinstance(portfolio_tickers, list):
        portfolio_tickers = []

all_tickers = list(dict.fromkeys(TICKERS_CORE + portfolio_tickers))
print(f"[+] Tickers combinados ({len(all_tickers)}): {all_tickers}")

print("[+] Descargando datos historicos (1 ano)...")
try:
    data = yf.download(all_tickers + ['SPY'], period='1y', interval='1d', group_by='ticker', progress=False, auto_adjust=True)
except Exception as e:
    print(f"[!] Error descargando datos: {e}")
    sys.exit(1)

if data is None or data.empty:
    print("[!] No se obtuvieron datos")
    sys.exit(1)

if isinstance(data.columns, pd.MultiIndex):
    close = data.xs('Close', axis=1, level=1)
else:
    close = data

close.columns = [str(c).upper().strip() for c in close.columns]
print(f"[+] Columnas disponibles: {list(close.columns)}")

valid_tickers = [t for t in all_tickers if t in close.columns and close[t].dropna().shape[0] > 20]
if not valid_tickers:
    print("[!] No hay tickers validos con suficientes datos")
    sys.exit(1)

if len(valid_tickers) < len(all_tickers):
    print(f"[!] Se descartaron {len(all_tickers) - len(valid_tickers)} tickers sin datos suficientes")

close = close[valid_tickers].dropna()
print(f"[+] Tickers validos: {valid_tickers}, observaciones: {close.shape[0]}")

prices = close.copy()
log_returns = np.log(prices / prices.shift(1)).dropna()
annual_returns = log_returns.mean() * 252
cov_matrix = log_returns.cov() * 252
n_assets = len(valid_tickers)

rf = 0.05

print("[+] PARTE 1: Frontera Eficiente (Markowitz)")
print(f"    Generando 5000 portafolios aleatorios...")

n_portfolios = 5000
results = np.zeros((3, n_portfolios))
weight_records = []

np.random.seed(42)
for i in range(n_portfolios):
    w = np.random.random(n_assets)
    w /= w.sum()
    port_return = np.dot(w, annual_returns.values)
    port_vol = np.sqrt(np.dot(w.T, np.dot(cov_matrix.values, w)))
    sharpe = (port_return - rf) / port_vol
    results[0, i] = port_return
    results[1, i] = port_vol
    results[2, i] = sharpe
    weight_records.append(w)

sharpe_idx = np.argmax(results[2])
min_vol_idx = np.argmin(results[1])

max_sharpe_weights = weight_records[sharpe_idx]
min_vol_weights = weight_records[min_vol_idx]

max_sharpe_return = results[0, sharpe_idx]
max_sharpe_vol = results[1, sharpe_idx]
max_sharpe_ratio = results[2, sharpe_idx]

min_vol_return = results[0, min_vol_idx]
min_vol_vol = results[1, min_vol_idx]
min_vol_sharpe = results[2, min_vol_idx]

print(f"    Max Sharpe: ratio={max_sharpe_ratio:.4f}, retorno={max_sharpe_return:.4f}, vol={max_sharpe_vol:.4f}")
print(f"    Min Vol: retorno={min_vol_return:.4f}, vol={min_vol_vol:.4f}, sharpe={min_vol_sharpe:.4f}")

top10_idx = np.argsort(results[2])[-10:][::-1]
top10_portfolios = []
for idx in top10_idx:
    w = weight_records[idx]
    entry = {
        "pesos": {valid_tickers[j]: round(float(w[j]), 4) for j in range(n_assets)},
        "sharpe": round(float(results[2, idx]), 4),
        "retorno": round(float(results[0, idx]), 4),
        "vol": round(float(results[1, idx]), 4)
    }
    top10_portfolios.append(entry)

frontera_eficiente = {
    "max_sharpe": {
        "pesos": {valid_tickers[j]: round(float(max_sharpe_weights[j]), 4) for j in range(n_assets)},
        "sharpe": round(float(max_sharpe_ratio), 4),
        "retorno": round(float(max_sharpe_return), 4),
        "vol": round(float(max_sharpe_vol), 4)
    },
    "min_vol": {
        "pesos": {valid_tickers[j]: round(float(min_vol_weights[j]), 4) for j in range(n_assets)},
        "sharpe": round(float(min_vol_sharpe), 4),
        "retorno": round(float(min_vol_return), 4),
        "vol": round(float(min_vol_vol), 4)
    },
    "top10": top10_portfolios
}

print("[+] PARTE 2: Risk Parity")
print(f"    Calculando contribucion al riesgo...")

risk_parity_weights = np.ones(n_assets) / n_assets
portfolio_vol_initial = np.sqrt(np.dot(risk_parity_weights.T, np.dot(cov_matrix.values, risk_parity_weights)))

for iteration in range(1000):
    port_vol = np.sqrt(np.dot(risk_parity_weights.T, np.dot(cov_matrix.values, risk_parity_weights)))
    marginal = np.dot(cov_matrix.values, risk_parity_weights) / port_vol
    risk_contrib = risk_parity_weights * marginal
    total_risk = risk_contrib.sum()
    target_risk = total_risk / n_assets
    error = np.sum((risk_contrib - target_risk) ** 2)
    for i in range(n_assets):
        if risk_contrib[i] > target_risk:
            risk_parity_weights[i] *= 0.99
        else:
            risk_parity_weights[i] *= 1.01
    risk_parity_weights /= risk_parity_weights.sum()
    if error < 1e-8:
        break

port_vol_rp = np.sqrt(np.dot(risk_parity_weights.T, np.dot(cov_matrix.values, risk_parity_weights)))
marginal_rp = np.dot(cov_matrix.values, risk_parity_weights) / port_vol_rp
risk_contrib_final = risk_parity_weights * marginal_rp
risk_contrib_pct = risk_contrib_final / risk_contrib_final.sum()

print(f"    Convergencia en {iteration+1} iteraciones")

risk_parity = {
    "pesos": {valid_tickers[j]: round(float(risk_parity_weights[j]), 4) for j in range(n_assets)},
    "contribucion_riesgo": {valid_tickers[j]: round(float(risk_contrib_pct[j]), 4) for j in range(n_assets)},
    "vol": round(float(port_vol_rp), 4),
    "retorno": round(float(np.dot(risk_parity_weights, annual_returns.values)), 4)
}

print("[+] PARTE 3: Monte Carlo Simulation")
print(f"    Simulando 10,000 escenarios a 252 dias...")

opt_weights = max_sharpe_weights
port_mean = max_sharpe_return
port_vol = max_sharpe_vol

n_sims = 10000
n_days = 252
initial_value = 100000

np.random.seed(42)
sim_returns = np.random.normal(port_mean / 252, port_vol / np.sqrt(252), (n_sims, n_days))
sim_values = initial_value * np.exp(np.cumsum(sim_returns, axis=1))

final_values = sim_values[:, -1]
drawdowns = np.zeros(n_sims)
daily_peaks = np.maximum.accumulate(sim_values, axis=1)
drawdown_mat = (daily_peaks - sim_values) / daily_peaks
max_dd_per_sim = np.max(drawdown_mat, axis=1)

median_final = float(np.median(final_values))
p5_final = float(np.percentile(final_values, 5))
p95_final = float(np.percentile(final_values, 95))
prob_loss = float(np.mean(final_values < initial_value))

expected_max_dd = float(np.mean(max_dd_per_sim))

daily_mc_returns = sim_values[:, 1:] / sim_values[:, :-1] - 1
flat_returns = daily_mc_returns.flatten()
var_95 = float(np.percentile(flat_returns, 5))

print(f"    Mediana final: ${median_final:,.0f}")
print(f"    P5: ${p5_final:,.0f}, P95: ${p95_final:,.0f}")
print(f"    Prob. perdida: {prob_loss:.4f}")
print(f"    Max DD esperado: {expected_max_dd:.4f}")
print(f"    VaR 95%: {var_95:.4f}")

monte_carlo = {
    "simulaciones": n_sims,
    "dias": n_days,
    "mediana_final": round(median_final, 2),
    "p5": round(p5_final, 2),
    "p95": round(p95_final, 2),
    "prob_perdida": round(prob_loss, 4),
    "max_drawdown_esperado": round(expected_max_dd, 4),
    "var_95": round(abs(var_95), 4)
}

print("[+] PARTE 4: Stress Testing")
print(f"    Analizando periodos de crisis...")

crisis_periods = [
    ("2008 Financial Crisis", "2008-10-01", "2009-03-01"),
    ("2020 COVID Crash", "2020-02-15", "2020-04-15"),
    ("2022 Rate Hike", "2022-01-01", "2022-10-01")
]

stress_results = []

for crisis_name, start_date, end_date in crisis_periods:
    print(f"    Procesando: {crisis_name} ({start_date} a {end_date})")
    try:
        crisis_data = yf.download(valid_tickers, start=start_date, end=end_date, interval='1d', group_by='ticker', progress=False, auto_adjust=True)
        if crisis_data is None or crisis_data.empty:
            print(f"    [!] Sin datos para {crisis_name}, usando fallback")
            stress_results.append({
                "crisis": crisis_name.replace(" ", "_").lower().replace(" ", "_"),
                "retorno": 0.0,
                "max_drawdown": 0.0,
                "dias_recuperacion": 0
            })
            continue
        if isinstance(crisis_data.columns, pd.MultiIndex):
            crisis_close = crisis_data.xs('Close', axis=1, level=1)
        else:
            crisis_close = crisis_data
        crisis_close.columns = [str(c).upper().strip() for c in crisis_close.columns]
        valid_crisis = [t for t in valid_tickers if t in crisis_close.columns and crisis_close[t].dropna().shape[0] > 5]
        if not valid_crisis:
            print(f"    [!] Sin datos validos en crisis para {crisis_name}")
            stress_results.append({
                "crisis": crisis_name.replace(" ", "_").lower().replace(" ", "_"),
                "retorno": 0.0,
                "max_drawdown": 0.0,
                "dias_recuperacion": 0
            })
            continue
        crisis_prices = crisis_close[valid_crisis]
        crisis_returns = crisis_prices.pct_change().dropna()
        crisis_weighted = crisis_returns.dot(pd.Series({t: opt_weights[valid_tickers.index(t)] if t in valid_tickers else 0 for t in valid_crisis}))
        if len(crisis_weighted) == 0:
            stress_results.append({
                "crisis": crisis_name.replace(" ", "_").lower().replace(" ", "_"),
                "retorno": 0.0,
                "max_drawdown": 0.0,
                "dias_recuperacion": 0
            })
            continue
        cum_return = (1 + crisis_weighted).prod() - 1
        cum_series = (1 + crisis_weighted).cumprod()
        running_max = np.maximum.accumulate(cum_series)
        dd_series = (running_max - cum_series) / running_max
        max_dd = dd_series.max()
        recovery_days = 0
        peak_idx = np.argmax(cum_series.values)
        post_peak = cum_series.iloc[peak_idx:]
        for j in range(len(post_peak)):
            if post_peak.iloc[j] >= cum_series.iloc[peak_idx]:
                recovery_days = j
                break
        if recovery_days == 0 and max_dd > 0.01:
            recovery_days = len(post_peak)
        print(f"    Retorno: {cum_return:.4f}, Max DD: {max_dd:.4f}, Recuperacion: {recovery_days} dias")
        stress_results.append({
            "crisis": crisis_name.replace(" ", "_").lower().replace(" ", "_"),
            "retorno": round(float(cum_return), 4),
            "max_drawdown": round(float(max_dd), 4),
            "dias_recuperacion": int(recovery_days)
        })
    except Exception as e:
        print(f"    [!] Error en stress test {crisis_name}: {e}")
        stress_results.append({
            "crisis": crisis_name.replace(" ", "_").lower().replace(" ", "_"),
            "retorno": 0.0,
            "max_drawdown": 0.0,
            "dias_recuperacion": 0
        })

max_sharpe_ret = max_sharpe_return
max_sharpe_v = max_sharpe_vol
conclusion = ""
if max_sharpe_ret > 0.15:
    conclusion = f"El portafolio optimo maximiza retorno ajustado por riesgo con Sharpe de {max_sharpe_ratio:.2f}, priorizando crecimiento sobre estabilidad."
elif max_sharpe_ret > 0.10:
    conclusion = f"El portafolio optimo ofrece un balance atractivo entre riesgo y retorno con Sharpe de {max_sharpe_ratio:.2f}."
else:
    conclusion = f"El portafolio optimo prioriza estabilidad con volatilidad de {max_sharpe_v:.2f} y Sharpe de {max_sharpe_ratio:.2f}."

result = {
    "timestamp": pd.Timestamp.now().isoformat(),
    "tickers": valid_tickers,
    "frontera_eficiente": frontera_eficiente,
    "risk_parity": risk_parity,
    "monte_carlo": monte_carlo,
    "stress_test": stress_results,
    "peso_actual_sugerido": "max_sharpe",
    "conclusion": conclusion
}

with open(OUTPUT_PATH, 'w') as f:
    json.dump(result, f, indent=2)

print(f"\n[+] Resultados guardados en {OUTPUT_PATH}")
print(f"[+] Proceso completado exitosamente")
