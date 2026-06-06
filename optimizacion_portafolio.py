import json, os, sys, yfinance as yf, numpy as np, pandas as pd, time, datetime, warnings, math
from portafolio_utils import cargar_portafolio

warnings.filterwarnings('ignore')

DATA_DIR = os.environ.get('GITHUB_WORKSPACE', '.')
OUTPUT_PATH = os.path.join(DATA_DIR, 'Datos', 'optimizacion_portafolio.json')
os.makedirs(os.path.join(DATA_DIR, 'Datos'), exist_ok=True)

TICKERS_CORE = ['NVDA','MU','DELL','AVGO','DDOG','SMCI','SNOW','CRWD','NOW','TSM','ARM','OKTA','HPE','NTAP','CLS','AAPL','AMZN','GOOGL','META','MSFT','LLY','AMAT','LRCX','PANW','ORCL','HON','UBER','GE','COST','NEE']

# ============================================================
# REGIME-AWARE POSITION SIZING
# ============================================================
REGIME_CONFIG = {
    'ALCISTA':     {'max_position': 0.15, 'leverage': 1.0,  'risk_per_trade': 0.02, 'max_sector': 0.30, 'cash_reserve': 0.05},
    'LATERAL':     {'max_position': 0.08, 'leverage': 0.5,  'risk_per_trade': 0.01, 'max_sector': 0.25, 'cash_reserve': 0.15},
    'BAJISTA':     {'max_position': 0.05, 'leverage': 0.0,  'risk_per_trade': 0.005,'max_sector': 0.15, 'cash_reserve': 0.30},
    'INCIERTO':    {'max_position': 0.06, 'leverage': 0.3,  'risk_per_trade': 0.008,'max_sector': 0.20, 'cash_reserve': 0.20}
}

def get_regime_params():
    """Load regime from regimen_mercado.json and return sizing params."""
    regime_path = os.path.join(DATA_DIR, 'Datos', 'regimen_mercado.json')
    if os.path.exists(regime_path):
        try:
            with open(regime_path) as f:
                regime_data = json.load(f)
            regime = regime_data.get('regimen', 'INCIERTO')
            confianza = regime_data.get('confianza', 0.5)
            # Blend with INCIERTO if low confidence
            if confianza < 0.5:
                base = REGIME_CONFIG['INCIERTO']
                target = REGIME_CONFIG.get(regime, REGIME_CONFIG['INCIERTO'])
                return {k: base[k] * (1 - confianza) + target[k] * confianza for k in base}
            return REGIME_CONFIG.get(regime, REGIME_CONFIG['INCIERTO'])
        except:
            pass
    return REGIME_CONFIG['INCIERTO']

def apply_regime_sizing(weights, regime_params, portfolio_value=100000):
    """Apply regime-aware position sizing to portfolio weights."""
    max_pos = regime_params['max_position']
    leverage = regime_params['leverage']
    cash_reserve = regime_params['cash_reserve']
    
    # Cap individual positions
    capped = {k: min(v, max_pos) for k, v in weights.items()}
    total = sum(capped.values())
    if total > 0:
        # Renormalize and apply leverage + cash reserve
        invested = (1 - cash_reserve) * leverage
        scaled = {k: v / total * invested for k, v in capped.items()}
        return scaled
    return capped

portfolio_tickers = cargar_portafolio(DATA_DIR)

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

# ============================================================
# HIERARCHICAL RISK PARITY (HRP)
# ============================================================
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import squareform
from collections import OrderedDict

def get_hrp_weights(cov, corr):
    """Hierarchical Risk Parity: clustering + recursive bisection."""
    dist = ((1 - corr) / 2) ** 0.5
    link = linkage(squareform(dist), 'single')
    
    # Quasi-diagonalization (sort by cluster order)
    sorted_idx = __sort_by_cluster(link, list(range(len(cov))))
    sorted_cov = cov.iloc[sorted_idx, sorted_idx]
    
    # Recursive bisection
    weights = __recursive_bisection(sorted_cov, list(range(len(sorted_cov))))
    order_map = {sorted_idx[i]: weights[i] for i in range(len(weights))}
    return np.array([order_map[i] for i in range(n_assets)])

def __sort_by_cluster(link, items):
    """Sort items by hierarchical clustering order."""
    if len(items) <= 1:
        return items
    # Find the last merge
    n = len(items)
    if len(link) == 0:
        return items
    cluster = link[-1]
    i1, i2 = int(cluster[0]), int(cluster[1])
    # Map cluster indices back to original items
    left = __get_cluster_items(link, i1, items)
    right = __get_cluster_items(link, i2, items)
    return left + right

def __get_cluster_items(link, idx, items, n_original=None):
    """Recursively get all items in a cluster."""
    if n_original is None:
        n_original = len(items)
    if idx < n_original:
        return [items[idx]]
    cluster = link[idx - n_original]
    i1, i2 = int(cluster[0]), int(cluster[1])
    return __get_cluster_items(link, i1, items, n_original) + __get_cluster_items(link, i2, items, n_original)

def __recursive_bisection(cov, items):
    """Recursive bisection: split cluster, allocate variance inversely."""
    if len(items) <= 1:
        return [1.0]
    
    mid = len(items) // 2
    left = items[:mid]
    right = items[mid:]
    
    w_left = __cluster_variance(cov, left)
    w_right = __cluster_variance(cov, right)
    alpha = w_left / (w_left + w_right) if (w_left + w_right) > 0 else 0.5
    
    w_left_vec = __recursive_bisection(cov, left)
    w_right_vec = __recursive_bisection(cov, right)
    
    return [x * alpha for x in w_left_vec] + [x * (1 - alpha) for x in w_right_vec]

def __cluster_variance(cov, items):
    """Compute variance of a cluster."""
    w = np.ones(len(items)) / len(items)
    sub = cov.iloc[items, items].values
    return float(np.dot(w, np.dot(sub, w)))

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

# ============================================================
# HIERARCHICAL RISK PARITY (HRP) - mejora sobre Risk Parity
# ============================================================
print("[+] PARTE 2b: Hierarchical Risk Parity")
try:
    hrp_weights = get_hrp_weights(cov_matrix, log_returns.corr())
    hrp_weights = hrp_weights / hrp_weights.sum()
    hrp_return = float(np.dot(hrp_weights, annual_returns.values))
    hrp_vol = float(np.sqrt(np.dot(hrp_weights.T, np.dot(cov_matrix.values, hrp_weights))))
    hrp_sharpe = (hrp_return - rf) / hrp_vol if hrp_vol > 0 else 0
    hrp_marginal = np.dot(cov_matrix.values, hrp_weights) / hrp_vol
    hrp_risk_contrib = hrp_weights * hrp_marginal
    hrp_risk_contrib_pct = hrp_risk_contrib / hrp_risk_contrib.sum()
    
    hrp_result = {
        "pesos": {valid_tickers[j]: round(float(hrp_weights[j]), 4) for j in range(n_assets)},
        "contribucion_riesgo": {valid_tickers[j]: round(float(hrp_risk_contrib_pct[j]), 4) for j in range(n_assets)},
        "vol": round(hrp_vol, 4),
        "retorno": round(hrp_return, 4),
        "sharpe": round(hrp_sharpe, 4)
    }
    print(f"    HRP: retorno={hrp_return:.4f}, vol={hrp_vol:.4f}, sharpe={hrp_sharpe:.4f}")
except Exception as e:
    print(f"    [!] HRP fallo: {e}")
    hrp_result = None

# ============================================================
# TAX-AWARE REBALANCING
# ============================================================
print("[+] PARTE 2c: Tax-Aware Rebalancing")
tax_aware_result = {}
try:
    # Read current positions from paper_trading
    paper_path = os.path.join(DATA_DIR, 'Datos', 'paper_trading.json')
    current_holdings = {}
    if os.path.exists(paper_path):
        pt = json.load(open(paper_path))
        holdings = pt.get('holdings', {})
        for t, h in holdings.items():
            current_holdings[t.upper()] = {
                'cost_basis': h.get('valor_costo', 0),
                'current_value': h.get('valor', 0),
                'return_pct': (h.get('valor', 0) / max(h.get('valor_costo', 1), 0.01) - 1) * 100,
                'held_days': (datetime.datetime.now() - datetime.datetime.strptime(h.get('fecha_compra', time.strftime('%Y-%m-%d')), '%Y-%m-%d')).days if h.get('fecha_compra') else 0
            }
    
    target_weights = hrp_weights if hrp_result else max_sharpe_weights
    if n_assets == len(valid_tickers) and target_weights is not None:
        tax_aware_sells = []
        tax_cost_total = 0
        for i, t in enumerate(valid_tickers):
            target_w = target_weights[i] if i < len(target_weights) else 0
            current_w = current_holdings.get(t, {}).get('current_value', 0) / max(sum(h.get('current_value', 0) for h in current_holdings.values()), 1)
            
            if t in current_holdings and target_w < current_w:
                diff = current_w - target_w
                gain = current_holdings[t]['return_pct']
                held = current_holdings[t]['held_days']
                # Long-term vs short-term tax rate proxy
                tax_rate = 0.15 if held > 365 else 0.35
                tax_cost = diff * current_holdings[t]['current_value'] * tax_rate * gain / 100
                tax_cost_total += abs(tax_cost)
                tax_aware_sells.append({
                    'ticker': t,
                    'current_w': round(current_w * 100, 1),
                    'target_w': round(target_w * 100, 1),
                    'diff_pct': round(diff * 100, 1),
                    'gain_pct': round(gain, 1),
                    'held_days': held,
                    'tax_rate': tax_rate,
                    'tax_cost': round(abs(tax_cost), 2),
                    'tax_efficient': held > 365
                })
        
        tax_aware_result = {
            'current_tax_cost': round(tax_cost_total, 2),
            'sell_candidates': sorted(tax_aware_sells, key=lambda x: x['diff_pct'], reverse=True)[:10],
            'recommendation': 'sell_long_term_first' if any(s['held_days'] > 365 for s in tax_aware_sells) else 'hold'
        }
        print(f'    Tax cost: ${tax_cost_total:.2f} | {len(tax_aware_sells)} sell candidates')
except Exception as e:
    print(f'    [!] Tax-Aware Rebalancing fallo: {e}')

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

# ============================================================
# REGIME-AWARE POSITION SIZING
# ============================================================
regime_params = get_regime_params()
print(f"[+] PARTE 5: Regime-Aware Position Sizing")
print(f"    Regimen detectado: {regime_params}")

max_sharpe_scaled = apply_regime_sizing(frontera_eficiente['max_sharpe']['pesos'], regime_params)
min_vol_scaled = apply_regime_sizing(frontera_eficiente['min_vol']['pesos'], regime_params)
risk_parity_scaled = apply_regime_sizing(risk_parity['pesos'], regime_params)

print(f"    Max position: {regime_params['max_position']:.0%} | Leverage: {regime_params['leverage']:.1f}x | Cash reserve: {regime_params['cash_reserve']:.0%}")

regime_sizing = {
    "regimen": regime_params,
    "max_sharpe_regime": {
        "pesos": {k: round(v, 4) for k, v in max_sharpe_scaled.items()},
        "total_invested": round(sum(max_sharpe_scaled.values()), 4),
        "cash": round(regime_params['cash_reserve'], 4)
    },
    "min_vol_regime": {
        "pesos": {k: round(v, 4) for k, v in min_vol_scaled.items()},
        "total_invested": round(sum(min_vol_scaled.values()), 4),
        "cash": round(regime_params['cash_reserve'], 4)
    },
    "risk_parity_regime": {
        "pesos": {k: round(v, 4) for k, v in risk_parity_scaled.items()},
        "total_invested": round(sum(risk_parity_scaled.values()), 4),
        "cash": round(regime_params['cash_reserve'], 4)
    }
}

result = {
    "timestamp": pd.Timestamp.now().isoformat(),
    "tickers": valid_tickers,
    "frontera_eficiente": frontera_eficiente,
    "risk_parity": risk_parity,
    "hrp": hrp_result,
    "tax_aware": tax_aware_result,
    "monte_carlo": monte_carlo,
    "stress_test": stress_results,
    "regime_sizing": regime_sizing,
    "peso_actual_sugerido": "hrp" if hrp_result else "max_sharpe",
    "conclusion": conclusion
}

with open(OUTPUT_PATH, 'w') as f:
    json.dump(result, f, indent=2)

print(f"\n[+] Resultados guardados en {OUTPUT_PATH}")
print(f"[+] Proceso completado exitosamente")
