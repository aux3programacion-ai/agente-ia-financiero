import json, os, sys, yfinance as yf, numpy as np, pandas as pd, xgboost as xgb, time, datetime, warnings
warnings.filterwarnings('ignore')

DATA_DIR = os.environ.get('GITHUB_WORKSPACE', '.')

predicciones_hist = {}
calibracion = {}
try:
    with open(f'{DATA_DIR}/Datos/predicciones_hist.json', 'r') as f:
        predicciones_hist = json.load(f)
    print(f'[OK] predicciones_hist.json loaded ({len(predicciones_hist)} tickers)')
except Exception as e:
    print(f'[!] Could not load predicciones_hist.json: {e}')

try:
    with open(f'{DATA_DIR}/Datos/calibracion.json', 'r') as f:
        calibracion = json.load(f)
    print(f'[OK] calibracion.json loaded')
except Exception as e:
    print(f'[!] Could not load calibracion.json: {e}')

TICKERS_CORE = ['NVDA','MU','DELL','AVGO','DDOG','SMCI','SNOW','CRWD','NOW','TSM','ARM','OKTA','HPE','NTAP','CLS','AAPL','AMZN','GOOGL','META','MSFT','LLY','AMAT','LRCX','PANW','ORCL','HON','UBER','GE','COST','NEE']

def compute_rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.rolling(window=period, min_periods=period).mean()
    avg_loss = loss.rolling(window=period, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi

def compute_macd(series, fast=12, slow=26, signal=9):
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    return macd_line - signal_line

all_data = []
all_labels = []
ticker_results = {}

print(f'\n=== XGBoost Model Training ===\n')

for ticker in TICKERS_CORE:
    try:
        print(f'  Processing {ticker}...', end=' ')
        df = yf.download(ticker, period='1y', interval='1d', group_by='ticker', auto_adjust=True, progress=False, multi_level_index=False)
        if df is None or df.empty:
            print('[!] No data')
            continue
        if isinstance(df.columns, pd.MultiIndex):
            df = df.xs(ticker, level=0, axis=1) if ticker in df.columns.get_level_values(0) else df
        df = df.dropna(subset=['Close'])
        df = df.copy()
        df['return_1d'] = df['Close'].pct_change()
        df['return_5d'] = df['Close'].pct_change(5)
        df['return_20d'] = df['Close'].pct_change(20)
        df['rsi_14'] = compute_rsi(df['Close'], 14)
        df['macd_hist'] = compute_macd(df['Close'], 12, 26, 9)
        df['vol_ratio'] = df['Volume'] / df['Volume'].rolling(window=50, min_periods=20).mean()
        df['volatility_20d'] = df['return_1d'].rolling(window=20, min_periods=10).std()
        df['sma50'] = df['Close'].rolling(window=50, min_periods=20).mean()
        df['sma200'] = df['Close'].rolling(window=200, min_periods=50).mean()
        df['sma50_dist_pct'] = (df['Close'] - df['sma50']) / df['sma50'] * 100
        df['sma200_dist_pct'] = (df['Close'] - df['sma200']) / df['sma200'] * 100
        df['target'] = (df['Close'].shift(-20) > df['Close']).astype(int)

        feature_cols = ['return_1d','return_5d','return_20d','rsi_14','macd_hist','vol_ratio','volatility_20d','sma50_dist_pct','sma200_dist_pct']
        df_ml = df[feature_cols + ['target']].dropna()
        if len(df_ml) < 50:
            print(f'[!] Only {len(df_ml)} samples, skip')
            continue

        X = df_ml[feature_cols].values
        y = df_ml['target'].values
        n_train = int(len(X) * 0.8)
        X_train, X_test = X[:n_train], X[n_train:]
        y_train, y_test = y[:n_train], y[n_train:]

        model = xgb.XGBClassifier(n_estimators=200, max_depth=5, learning_rate=0.05, random_state=42, verbosity=0, use_label_encoder=False)
        model.fit(X_train, y_train)

        train_acc = model.score(X_train, y_train)
        test_acc = model.score(X_test, y_test)

        fi = model.feature_importances_
        fi_idx = np.argsort(fi)[::-1]
        top3 = [feature_cols[i] for i in fi_idx[:3]]

        latest = df_ml[feature_cols].iloc[-1:].values
        if len(latest) > 0:
            prob_up = model.predict_proba(latest)[0][1] * 100
        else:
            prob_up = 50.0

        if prob_up > 60:
            pred = 'alcista'
        elif prob_up < 40:
            pred = 'bajista'
        else:
            pred = 'neutral'

        ticker_results[ticker] = {
            'prob_up_20d': round(prob_up, 1),
            'features_top': top3,
            'prediccion': pred,
            'test_acc': round(test_acc, 4),
            'train_acc': round(train_acc, 4)
        }
        all_data.append(X_train)
        all_labels.append(y_train)

        print(f'[OK] prob={prob_up:.0f}% pred={pred} train_acc={train_acc:.2f} test_acc={test_acc:.2f}')

    except Exception as e:
        print(f'[!] Error: {e}')
        continue

global_fi = {}
if all_data:
    try:
        X_all = np.vstack(all_data)
        y_all = np.concatenate(all_labels)
        model_global = xgb.XGBClassifier(n_estimators=200, max_depth=5, learning_rate=0.05, random_state=42, verbosity=0, use_label_encoder=False)
        model_global.fit(X_all, y_all)
        fi_global = model_global.feature_importances_
        for i, col in enumerate(feature_cols):
            global_fi[col] = round(float(fi_global[i]), 4)
        global_fi = dict(sorted(global_fi.items(), key=lambda x: x[1], reverse=True))
    except Exception as e:
        print(f'[!] Global feature importance failed: {e}')

precision_test = round(np.mean([v['test_acc'] for v in ticker_results.values()]) if ticker_results else 0, 2)
precision_train = round(np.mean([v['train_acc'] for v in ticker_results.values()]) if ticker_results else 0, 2)
total_datos = sum(len(predicciones_hist.get(t, {}).get('predicciones', [])) for t in TICKERS_CORE if t in predicciones_hist)

output = {
    'timestamp': datetime.datetime.now().strftime('%Y-%m-%dT%H:%M:%SZ'),
    'precision_test': precision_test,
    'precision_train': precision_train,
    'tickers': {},
    'feature_importance_global': global_fi,
    'total_datos': total_datos
}

for tk, tr in ticker_results.items():
    output['tickers'][tk] = {
        'prob_up_20d': tr['prob_up_20d'],
        'features_top': tr['features_top'],
        'prediccion': tr['prediccion']
    }

try:
    os.makedirs(f'{DATA_DIR}/Datos', exist_ok=True)
    with open(f'{DATA_DIR}/Datos/modelo_xgboost.json', 'w') as f:
        class NpEncoder(json.JSONEncoder):
            def default(self, obj):
                if isinstance(obj, (np.integer, np.floating)): return float(obj)
                if isinstance(obj, np.bool_): return bool(obj)
                if isinstance(obj, np.ndarray): return obj.tolist()
                return super().default(obj)
        json.dump(output, f, indent=2, cls=NpEncoder)
    print(f'\n[OK] modelo_xgboost.json saved ({len(ticker_results)} tickers)')
except Exception as e:
    print(f'[!] Save failed: {e}')

print(f'\n=== XGBoost Summary ===')
print(f'  Tickers processed: {len(ticker_results)}/{len(TICKERS_CORE)}')
print(f'  Precision train: {precision_train}')
print(f'  Precision test:  {precision_test}')
print(f'  Total historical: {total_datos}')
print(f'  Top features: {list(global_fi.keys())[:5] if global_fi else "N/A"}')
for tk, tr in sorted(ticker_results.items()):
    print(f'  {tk}: prob_up={tr["prob_up_20d"]}% pred={tr["prediccion"]} feat={tr["features_top"]}')
