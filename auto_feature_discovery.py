import ast
import json
import os
import re
import textwrap
import time

import numpy as np
import pandas as pd
import yfinance as yf

from logger import get_module_logger

_log = get_module_logger('auto_feature_discovery')

DATA_DIR = os.environ.get('GITHUB_WORKSPACE', '.')
OUTPUT_DIR = os.path.join(DATA_DIR, 'Datos')
os.makedirs(OUTPUT_DIR, exist_ok=True)

DISCOVERED_PATH = os.path.join(OUTPUT_DIR, 'auto_features_discovered.json')
FEATURE_LOG = os.path.join(OUTPUT_DIR, 'feature_discovery_log.json')

if not os.environ.get('OPENROUTER_KEY'):
    env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '.env')
    if os.path.exists(env_path):
        for linea in open(env_path, encoding='utf-8', errors='replace'):
            linea = linea.strip()
            if linea and not linea.startswith('#') and '=' in linea:
                k, v = linea.split('=', 1)
                k = k.strip(); v = v.strip().strip('"').strip("'")
                if k and v:
                    os.environ.setdefault(k, v)

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

DANGEROUS_NODES = {
    ast.Import, ast.ImportFrom, ast.Exec, ast.Global,
    ast.AsyncFunctionDef, ast.AsyncWith, ast.AsyncFor,
}

DANGEROUS_CALLS = {
    'open', 'eval', 'exec', 'compile', 'getattr', 'setattr', 'delattr',
    'globals', 'locals', 'vars', 'dir', 'input', 'breakpoint',
    'os', 'subprocess', 'sys', 'shutil', 'pathlib', 'socket',
    'urllib', 'requests', 'http',
}

def _validate_feature_code(code: str) -> bool:
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return False
    for node in ast.walk(tree):
        if type(node) in DANGEROUS_NODES:
            return False
        if isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Name) and func.id in DANGEROUS_CALLS:
                return False
            if isinstance(func, ast.Attribute):
                if func.attr.startswith('_'):
                    return False
                if isinstance(func.value, ast.Name) and func.value.id in DANGEROUS_CALLS:
                    return False
        if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
            if node.value.id in DANGEROUS_CALLS:
                return False
    return True

def load_prediction_history():
    path = os.path.join(OUTPUT_DIR, 'aprendizaje.json')
    if not os.path.exists(path):
        return []
    try:
        return json.load(open(path)).get('predicciones', [])
    except Exception:
        return []

def load_discovered_features():
    if os.path.exists(DISCOVERED_PATH):
        try:
            data = json.load(open(DISCOVERED_PATH))
            return data.get('features', []), data
        except Exception:
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
        
        # Execute the feature code in restricted namespace with AST validation
        code = textwrap.dedent(feature_def['code'])
        if not _validate_feature_code(code):
            return -1, None
        SAFE_BUILTINS = {
            'abs': abs, 'min': min, 'max': max, 'len': len,
            'round': round, 'sum': sum, 'sorted': sorted,
            'range': range, 'enumerate': enumerate, 'zip': zip,
            'True': True, 'False': False, 'None': None,
            'int': int, 'float': float, 'str': str, 'bool': bool,
            'list': list, 'dict': dict, 'tuple': tuple, 'set': set,
        }
        namespace = {'__builtins__': SAFE_BUILTINS, 'np': np, 'pd': pd, 'df': df}
        exec(code, namespace)
        
        if feature_def['function_name'] not in namespace:
            return -1, None
        
        feature_series = namespace[feature_def['function_name']](df)
        if not isinstance(feature_series, pd.Series):
            return -1, None
        
        # Align with returns
        returns = df['Close'].pct_change()
        target = (df['Close'].shift(-20) > df['Close'].shift(-1)).astype(int)
        
        combined = pd.DataFrame({
            'feature': feature_series,
            'target': target
        }).dropna()
        
        if len(combined) < 30:
            return -1, None
        
        # Score: correlation with future return direction
        from scipy.stats import pearsonr
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
    except Exception:
        return -1, None

def main():
    _log.info("Descubriendo nuevas features via LLM...")
    
    predictions = load_prediction_history()
    if len(predictions) < 30:
        _log.warning("Pocas predicciones, esperar mas datos")
        return
    
    existing_features, feature_db = load_discovered_features()
    existing_names = {f.get('function_name', '') for f in existing_features}
    _log.info(f"{len(existing_features)} features ya descubiertas")
    
    failing = detect_failing_tickers(predictions)
    if not failing:
        _log.info("Sin tickers con declive significativo")
        return
    
    _log.info(f"{len(failing)} tickers con declive")
    for f in failing[:5]:
        _log.info(f'{f["ticker"]}: acc={f["recent_accuracy"]:.0%} (bajo {f["decline"]:.0%})')
    
    MAX_NEW = 2
    new_count = 0
    
    for f in failing:
        if new_count >= MAX_NEW:
            break
        ticker = f['ticker']
        _log.info(f"Generando feature para {ticker}...")
        
        already_tried = any(feat.get('ticker') == ticker for feat in existing_features[-5:])
        if already_tried:
            _log.info(f"Ya se intento recientemente para {ticker}, skip")
            continue
        
        feature_def = generate_feature_idea(ticker, f)
        if not feature_def:
            continue
        
        fn_name = feature_def['function_name']
        if fn_name in existing_names:
            _log.info(f"Feature {fn_name} ya existe")
            continue
        
        _log.info(f"Codigo generado: {fn_name}()")
        
        score, series = test_feature(ticker, feature_def)
        
        if score > 0.55:
            feature_def['approved'] = True
            feature_def['test_score'] = round(score, 4)
            feature_def['approved_at'] = time.strftime('%Y-%m-%dT%H:%M:%S')
            save_discovered_features(feature_db, feature_def)
            new_count += 1
            _log.info(f"[APROBADA] {fn_name} score={score:.3f}")
        elif score > 0:
            _log.info(f"[RECHAZADA] {fn_name} score={score:.3f} < 0.55")
            feature_def['approved'] = False
            feature_def['test_score'] = round(score, 4)
            save_discovered_features(feature_db, feature_def)
        else:
            _log.warning(f"[ERROR] No se pudo probar {fn_name}")
    
    log_entries = []
    if os.path.exists(FEATURE_LOG):
        try:
            log_entries = json.load(open(FEATURE_LOG))
        except Exception as e:
            _log.warning(f"Error loading feature log: {e}")
    log_entries.append({
        'timestamp': time.strftime('%Y-%m-%dT%H:%M:%S'),
        'failing_tickers': len(failing),
        'new_features': new_count,
        'total_features': len(feature_db['features']) if isinstance(feature_db, dict) else 0
    })
    log_entries = log_entries[-100:]
    with open(FEATURE_LOG, 'w') as f:
        json.dump(log_entries, f, indent=2)
    
    _log.info(f"OK: {new_count} nuevas features descubiertas")

if __name__ == '__main__':
    main()
