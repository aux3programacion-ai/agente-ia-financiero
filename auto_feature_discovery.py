import json, os, sys, time, re, math, traceback, textwrap
import numpy as np
import pandas as pd
import yfinance as yf

DATA_DIR = os.environ.get('GITHUB_WORKSPACE', '.')
OUTPUT_DIR = os.path.join(DATA_DIR, 'Datos')
os.makedirs(OUTPUT_DIR, exist_ok=True)

DISCOVERED_PATH = os.path.join(OUTPUT_DIR, 'auto_features_discovered.json')
FEATURE_LOG = os.path.join(OUTPUT_DIR, 'feature_discovery_log.json')

API_KEY = os.environ.get('OPENROUTER_KEY', '')

TICKERS_CORE = ['NVDA','MU','DELL','AVGO','DDOG','SMCI','SNOW','CRWD','NOW','TSM',
                'ARM','OKTA','HPE','NTAP','CLS','AAPL','AMZN','GOOGL','META','MSFT',
                'LLY','AMAT','LRCX','PANW','ORCL','HON','UBER','GE','COST','NEE']

EXISTING_FEATURES = [
    'return_1d','return_5d','return_20d','rsi_14','macd_hist','macd_hist_slope',
    'vol_ratio','vol_20d_profile','volatility_20d','bb_position',
    'sma50_dist_pct','sma200_dist_pct','rsi_div','atr_pct','roc_10d',
    'entropy_20','arima_resid','arima_resid_abs','autocorr_1','autocorr_5','hurst_approx',
    'opt_put_call_ratio','opt_iv_skew','opt_avg_iv','opt_oi_weighted_iv','opt_oi_ratio','opt_vol',
    'macro_yield_curve_10y_2y','macro_yield_curve_10y_3m','macro_fed_funds',
    'macro_vix','macro_dxy','macro_dxy_mom','macro_real_rate',
    'macro_curve_inverted','macro_risk_on'
]

def load_prediction_history():
    path = os.path.join(OUTPUT_DIR, 'aprendizaje.json')
    if not os.path.exists(path):
        return []
    try:
        return json.load(open(path)).get('predicciones', [])
    except:
        return []

def load_discovered_features():
    if os.path.exists(DISCOVERED_PATH):
        try:
            data = json.load(open(DISCOVERED_PATH))
            return data.get('features', []), data
        except:
            pass
    return [], {'features': [], 'version': 0}

def save_discovered_features(all_data, new_feature=None):
    if new_feature:
        all_data['features'].append(new_feature)
    all_data['version'] = all_data.get('version', 0) + 1
    all_data['updated'] = time.strftime('%Y-%m-%dT%H:%M:%S')
    with open(DISCOVERED_PATH, 'w') as f:
        json.dump(all_data, f, indent=2)

def detect_failing_tickers(predictions):
    """Find tickers with declining or poor accuracy using recent data."""
    ticker_stats = {}
    for p in predictions[-300:]:
        t = p.get('ticker', '')
        if t not in ticker_stats:
            ticker_stats[t] = []
        ticker_stats[t].append(p.get('acierto', False))
    
    failing = []
    for t, outcomes in ticker_stats.items():
        if len(outcomes) < 10:
            continue
        recent = outcomes[-10:]
        older = outcomes[:-10]
        recent_acc = sum(recent) / len(recent)
        older_acc = sum(older) / len(older) if older else 0.5
        decline = older_acc - recent_acc
        
        if recent_acc < 0.45 or decline > 0.15:
            failing.append({
                'ticker': t,
                'recent_accuracy': round(recent_acc, 3),
                'older_accuracy': round(older_acc, 3),
                'decline': round(decline, 3),
                'n_recent': len(recent),
                'n_total': len(outcomes)
            })
    
    return sorted(failing, key=lambda x: x['decline'], reverse=True)

def generate_feature_idea(ticker, context, modelo='openrouter/free'):
    """Ask LLM to write a feature that would help predict this ticker."""
    if not API_KEY:
        return None
    
    prompt = f'''Eres un quant researcher senior. Un modelo de ML esta fallando en predecir {ticker}.
Contexto: Precision reciente={context["recent_accuracy"]:.0%}, declive={context["decline"]:.1%}
Features actuales: {', '.join(EXISTING_FEATURES[:15])}...

INSTRUCCION: Escribe UNA funcion Python que calcule una NUEVA feature para mejorar la prediccion.
La funcion recibe: df (pandas DataFrame con columnas Open, High, Low, Close, Volume)
Devuelve: pd.Series con la feature calculada.

REQUISITOS:
- Usa SOLO numpy y pandas
- No acceso a red ni archivos
- No modifica df
- Returns pd.Series con el mismo index que df
- Feature name: nombre descriptivo

Ejemplo:
def sma_crossover_ratio(df):
    sma20 = df['Close'].rolling(20).mean()
    sma50 = df['Close'].rolling(50).mean()
    return (sma20 / sma50 - 1) * 100

Escribe SOLO codigo Python, sin markdown, sin explicaciones.'''
    
    try:
        import urllib.request
        payload = json.dumps({
            'model': modelo,
            'messages': [
                {'role': 'system', 'content': 'Eres un quant researcher. Escribe solo codigo Python.'},
                {'role': 'user', 'content': prompt}
            ],
            'temperature': 0.3,
            'max_tokens': 800
        }).encode('utf-8')
        req = urllib.request.Request(
            'https://openrouter.ai/api/v1/chat/completions',
            data=payload,
            headers={
                'Authorization': f'Bearer {API_KEY}',
                'Content-Type': 'application/json',
                'HTTP-Referer': 'https://github.com/agente-financiero',
                'X-Title': 'Auto Feature Discovery'
            },
            method='POST'
        )
        with urllib.request.urlopen(req, timeout=30) as r:
            response = json.loads(r.read())['choices'][0]['message']['content']
        
        # Extract Python code
        code = response.strip()
        if code.startswith('```'):
            m = re.search(r'```(?:python)?\s*([\s\S]*?)```', code)
            if m: code = m.group(1).strip()
        
        # Validate code structure
        if 'def ' not in code or 'return' not in code:
            return None
        
        # Extract function name
        fn_match = re.search(r'def\s+(\w+)\s*\(', code)
        if not fn_match:
            return None
        fn_name = fn_match.group(1)
        
        return {
            'ticker': ticker,
            'code': code,
            'function_name': fn_name,
            'generated_at': time.strftime('%Y-%m-%dT%H:%M:%S'),
            'context': context,
            'approved': False,
            'test_score': None
        }
    except Exception as e:
        print(f'    [!] LLM call failed: {e}')
        return None

def test_feature(ticker, feature_def):
    """Test a discovered feature against historical data. Returns (score, series_or_none)."""
    try:
        df = yf.download(ticker, period='1y', interval='1d', progress=False, auto_adjust=True)
        if df is None or df.empty:
            return -1, None
        if isinstance(df.columns, pd.MultiIndex):
            df = df.xs(ticker, axis=1, level=0) if ticker in df.columns.get_level_values(0) else df
        df = df.dropna(subset=['Close']).copy()
        
        if len(df) < 60:
            return -1, None
        
        # Execute the feature code in restricted namespace
        namespace = {'np': np, 'pd': pd, 'df': df}
        exec(textwrap.dedent(feature_def['code']), namespace)
        
        if feature_def['function_name'] not in namespace:
            return -1, None
        
        feature_series = namespace[feature_def['function_name']](df)
        if not isinstance(feature_series, pd.Series):
            return -1, None
        
        # Align with returns
        returns = df['Close'].pct_change()
        target = (df['Close'].shift(-20) > df['Close']).astype(int)
        
        combined = pd.DataFrame({
            'feature': feature_series,
            'target': target
        }).dropna()
        
        if len(combined) < 30:
            return -1, None
        
        # Score: correlation with future return direction
        from scipy.stats import pearsonr, spearmanr
        corr_p, _ = pearsonr(combined['feature'], combined['target'])
        
        # Also test predictive power via simple threshold
        feature_values = combined['feature']
        target_values = combined['target']
        median_val = feature_values.median()
        
        # If feature above median, how often is target 1?
        above_median = target_values[feature_values > median_val].mean() if (feature_values > median_val).sum() > 0 else 0.5
        below_median = target_values[feature_values <= median_val].mean() if (feature_values <= median_val).sum() > 0 else 0.5
        predictive_diff = abs(above_median - below_median)
        
        # Combined score: correlation + predictive power
        score = abs(corr_p) * 0.5 + predictive_diff * 0.5
        score = min(1, max(0, score))
        
        return score, feature_series
    except Exception as e:
        return -1, None

def main():
    print('[AutoFeature] Descubriendo nuevas features via LLM...')
    
    predictions = load_prediction_history()
    if len(predictions) < 30:
        print('  [!] Pocas predicciones, esperar mas datos')
        return
    
    existing_features, feature_db = load_discovered_features()
    existing_names = {f.get('function_name', '') for f in existing_features}
    print(f'  {len(existing_features)} features ya descubiertas')
    
    # Find failing tickers
    failing = detect_failing_tickers(predictions)
    if not failing:
        print('  [OK] Sin tickers con declive significativo')
        return
    
    print(f'  {len(failing)} tickers con declive:')
    for f in failing[:5]:
        print(f'    {f["ticker"]}: acc={f["recent_accuracy"]:.0%} (bajo {f["decline"]:.0%})')
    
    # Try to discover new features for top failing tickers
    MAX_NEW = 2
    new_count = 0
    
    for f in failing:
        if new_count >= MAX_NEW:
            break
        ticker = f['ticker']
        print(f'  Generando feature para {ticker}...')
        
        # Check not already tried recently
        already_tried = any(feat.get('ticker') == ticker for feat in existing_features[-5:])
        if already_tried:
            print(f'    Ya se intento recientemente, skip')
            continue
        
        # Generate feature via LLM
        feature_def = generate_feature_idea(ticker, f)
        if not feature_def:
            continue
        
        fn_name = feature_def['function_name']
        if fn_name in existing_names:
            print(f'    Feature {fn_name} ya existe')
            continue
        
        print(f'    Codigo generado: {fn_name}()')
        
        # Test against historical data
        score, series = test_feature(ticker, feature_def)
        
        if score > 0.55:
            # APPROVED
            feature_def['approved'] = True
            feature_def['test_score'] = round(score, 4)
            feature_def['approved_at'] = time.strftime('%Y-%m-%dT%H:%M:%S')
            save_discovered_features(feature_db, feature_def)
            new_count += 1
            print(f'    [APROBADA] score={score:.3f} -> guardada')
        elif score > 0:
            print(f'    [RECHAZADA] score={score:.3f} < 0.55')
            # Still save as rejected for tracking
            feature_def['approved'] = False
            feature_def['test_score'] = round(score, 4)
            save_discovered_features(feature_db, feature_def)
        else:
            print(f'    [ERROR] No se pudo probar')
    
    # Log
    log = []
    if os.path.exists(FEATURE_LOG):
        try: log = json.load(open(FEATURE_LOG))
        except: pass
    log.append({
        'timestamp': time.strftime('%Y-%m-%dT%H:%M:%S'),
        'failing_tickers': len(failing),
        'new_features': new_count,
        'total_features': len(feature_db['features']) if isinstance(feature_db, dict) else 0
    })
    log = log[-100:]
    with open(FEATURE_LOG, 'w') as f:
        json.dump(log, f, indent=2)
    
    print(f'[AutoFeature] OK: {new_count} nuevas features descubiertas')

if __name__ == '__main__':
    main()
