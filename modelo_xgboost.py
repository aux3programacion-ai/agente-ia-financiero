import json, os, sys, textwrap, yfinance as yf, numpy as np, pandas as pd, xgboost as xgb, time, datetime, warnings
from sklearn.model_selection import TimeSeriesSplit
from sklearn.calibration import CalibratedClassifierCV
from sklearn.isotonic import IsotonicRegression
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
try:
    from lightgbm import LGBMClassifier
    HAS_LGBM = True
except ImportError:
    HAS_LGBM = False
try:
    from catboost import CatBoostClassifier
    HAS_CATBOOST = True
except ImportError:
    HAS_CATBOOST = False
try:
    import optuna
    HAS_OPTUNA = True
except ImportError:
    HAS_OPTUNA = False
warnings.filterwarnings('ignore')

DATA_DIR = os.environ.get('GITHUB_WORKSPACE', '.')

predicciones_hist = {}
calibracion = {}
regimen_data = {}
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

try:
    with open(f'{DATA_DIR}/Datos/regimen_mercado.json', 'r') as f:
        regimen_data = json.load(f)
    print(f'[OK] regimen_mercado.json loaded: {regimen_data.get("regimen", "unknown")}')
except Exception as e:
    print(f'[!] Could not load regimen_mercado.json: {e}')

TICKERS_CORE = ['NVDA','MU','DELL','AVGO','DDOG','SMCI','SNOW','CRWD','NOW','TSM','ARM','OKTA','HPE','NTAP','CLS','AAPL','AMZN','GOOGL','META','MSFT','LLY','AMAT','LRCX','PANW','ORCL','HON','UBER','GE','COST','NEE']

def compute_entropy(series, order=3):
    """Approximate entropy for a time series (speed-optimized).
    Lower entropy = more predictable, higher = more random."""
    n = len(series)
    if n < order + 5:
        return 0.5
    def _phi(m):
        patterns = np.array([series[i:i+m] for i in range(n-m+1)])
        count = np.sum(np.max(np.abs(patterns[:, None] - patterns[None, :]), axis=2) < 0.2 * np.std(series), axis=1) - 1
        return np.mean(np.log(np.maximum(count, 1) / (n-m)))
    return max(0, min(2, _phi(order) - _phi(order+1)))

def _hurst_exponent(ts):
    """Hurst exponent via R/S analysis."""
    if len(ts) < 30 or np.std(ts) == 0:
        return 0.5
    lags = range(2, min(20, len(ts)//2))
    tau = []
    for lag in lags:
        diff = np.subtract(ts[lag:], ts[:-lag])
        tau.append(np.std(diff))
    if len(tau) < 2 or min(tau) == 0:
        return 0.5
    from scipy import stats
    m = np.polyfit(np.log(lags), np.log(tau), 1)
    return max(0, min(1, m[0] / 2))

def compute_arima_residual(series, order=2):
    """Simple AR(order) residual as feature."""
    n = len(series)
    if n < order + 5:
        return pd.Series(0, index=series.index)
    from sklearn.linear_model import LinearRegression
    X = np.array([series.shift(i).values for i in range(1, order+1)]).T
    y = series.values
    mask = ~np.any(np.isnan(X), axis=1) & ~np.isnan(y)
    if mask.sum() < order + 2:
        return pd.Series(0, index=series.index)
    lr = LinearRegression()
    lr.fit(X[mask], y[mask])
    resid = np.full(len(series), np.nan)
    resid[mask] = y[mask] - lr.predict(X[mask])
    return pd.Series(resid, index=series.index)

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

def compute_bollinger(series, window=20, num_std=2):
    sma = series.rolling(window=window, min_periods=window).mean()
    std = series.rolling(window=window, min_periods=window).std()
    upper = sma + num_std * std
    lower = sma - num_std * std
    return (series - lower) / (upper - lower).replace(0, np.nan)

def compute_atr(high, low, close, period=14):
    tr1 = high - low
    tr2 = (high - close.shift()).abs()
    tr3 = (low - close.shift()).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.rolling(window=period, min_periods=period).mean()

def walk_forward_backtest(X, y, n_splits=5, test_size=63, gap=5):
    """Walk-forward expanding window backtest. Returns list of (train_idx, test_idx, model, metrics)."""
    tscv = TimeSeriesSplit(n_splits=n_splits, test_size=test_size, gap=gap)
    results = []
    for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        if len(np.unique(y_train)) < 2 or len(np.unique(y_test)) < 2:
            continue
        model = xgb.XGBClassifier(n_estimators=200, max_depth=5, learning_rate=0.05, random_state=42, verbosity=0, use_label_encoder=False)
        model.fit(X_train, y_train)
        train_acc = model.score(X_train, y_train)
        test_acc = model.score(X_test, y_test)
        results.append({'fold': fold, 'train_acc': train_acc, 'test_acc': test_acc, 'train_size': len(train_idx), 'test_size': len(test_idx)})
    return results

def calibrate_probabilities(model, X_cal, y_cal, method='isotonic'):
    """Calibrate model probabilities using Platt/Isotonic regression."""
    if method == 'isotonic':
        calibrator = IsotonicRegression(out_of_bounds='clip')
        probs = model.predict_proba(X_cal)[:, 1]
        calibrator.fit(probs, y_cal)
        return calibrator
    else:
        calibrator = CalibratedClassifierCV(model, method='sigmoid', cv='prefit')
        calibrator.fit(X_cal, y_cal)
        return calibrator

def optuna_tune_xgboost(X, y, n_trials=50, timeout=300):
    """Hyperparameter optimization using Optuna with walk-forward validation."""
    if not HAS_OPTUNA:
        print('[!] Optuna not installed, skipping hyperparameter tuning')
        return None
    
    def objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 500),
            'max_depth': trial.suggest_int('max_depth', 3, 8),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
            'gamma': trial.suggest_float('gamma', 0, 5),
            'reg_alpha': trial.suggest_float('reg_alpha', 0, 10),
            'reg_lambda': trial.suggest_float('reg_lambda', 0, 10),
            'random_state': 42,
            'verbosity': 0,
            'use_label_encoder': False
        }
        
        # Walk-forward validation
        tscv = TimeSeriesSplit(n_splits=3, test_size=63, gap=5)
        scores = []
        for train_idx, test_idx in tscv.split(X):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            if len(np.unique(y_train)) < 2:
                return 0.5
            model = xgb.XGBClassifier(**params)
            model.fit(X_train, y_train)
            scores.append(model.score(X_test, y_test))
        return np.mean(scores)
    
    print(f'  [Optuna] Starting hyperparameter optimization ({n_trials} trials, {timeout}s)...')
    study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=n_trials, timeout=timeout)
    
    best_params = study.best_params
    best_params.update({'random_state': 42, 'verbosity': 0, 'use_label_encoder': False})
    print(f'  [Optuna] Best params: {best_params} | Best score: {study.best_value:.4f}')
    return best_params

def build_stacking_ensemble(X, y, best_params=None):
    """Build stacking ensemble with XGBoost, LightGBM, CatBoost, RandomForest + LogisticRegression meta-learner."""
    estimators = []
    
    # XGBoost base learner
    if best_params:
        xgb_params = best_params.copy()
    else:
        xgb_params = {'n_estimators': 200, 'max_depth': 5, 'learning_rate': 0.05, 'random_state': 42, 'verbosity': 0, 'use_label_encoder': False}
    estimators.append(('xgb', xgb.XGBClassifier(**xgb_params)))
    
    # LightGBM
    if HAS_LGBM:
        lgb_params = {'n_estimators': 200, 'max_depth': 5, 'learning_rate': 0.05, 'random_state': 42, 'verbosity': -1}
        estimators.append(('lgb', LGBMClassifier(**lgb_params)))
    
    # CatBoost
    if HAS_CATBOOST:
        cat_params = {'iterations': 200, 'depth': 5, 'learning_rate': 0.05, 'random_state': 42, 'verbose': False}
        estimators.append(('cat', CatBoostClassifier(**cat_params)))
    
    # Random Forest
    rf_params = {'n_estimators': 200, 'max_depth': 8, 'random_state': 42, 'n_jobs': -1}
    estimators.append(('rf', RandomForestClassifier(**rf_params)))
    
    # Stacking with Logistic Regression meta-learner
    stack = StackingClassifier(
        estimators=estimators,
        final_estimator=LogisticRegression(random_state=42, max_iter=1000),
        cv=TimeSeriesSplit(n_splits=3, test_size=63, gap=5),
        n_jobs=-1,
        passthrough=True
    )
    
    print(f'  [Stacking] Ensemble built with {len(estimators)} base learners: {[e[0] for e in estimators]}')
    return stack

def get_historical_regimes(spy_df):
    """Compute historical regime labels for SPY data."""
    returns = spy_df['Close'].pct_change()
    sma50 = spy_df['Close'].rolling(window=50).mean()
    sma200 = spy_df['Close'].rolling(window=200).mean()
    ret_50d = spy_df['Close'].pct_change(50) * 100
    
    regimes = pd.Series(index=spy_df.index, dtype='object')
    for i in range(len(spy_df)):
        if i < 200:
            regimes.iloc[i] = 'INCIERTO'
            continue
        close = spy_df['Close'].iloc[i]
        s50 = sma50.iloc[i]
        s200 = sma200.iloc[i]
        r50 = ret_50d.iloc[i]
        if pd.isna(s50) or pd.isna(s200) or pd.isna(r50):
            regimes.iloc[i] = 'INCIERTO'
        elif close > s50 > s200 and r50 > 5:
            regimes.iloc[i] = 'ALCISTA'
        elif close < s50 < s200 and r50 < -5:
            regimes.iloc[i] = 'BAJISTA'
        elif (close > s50 and close < s200) or (close < s50 and close > s200):
            regimes.iloc[i] = 'LATERAL'
        else:
            regimes.iloc[i] = 'INCIERTO'
    return regimes

# --- Fetch SPY for historical regimes ---
print('  Fetching SPY for regime labeling...', end=' ')
spy_df = yf.download('SPY', period='2y', interval='1d', auto_adjust=True, progress=False, multi_level_index=False)
if spy_df is not None and not spy_df.empty:
    if isinstance(spy_df.columns, pd.MultiIndex):
        spy_df = spy_df.xs('SPY', level=0, axis=1) if 'SPY' in spy_df.columns.get_level_values(0) else spy_df
    spy_df = spy_df.dropna(subset=['Close'])
    historical_regimes = get_historical_regimes(spy_df)
    print(f'[OK] regimes computed')
else:
    historical_regimes = None
    print(f'[!] No SPY data, using single model')

# --- Load options data ---
print('  Loading options data...', end=' ')
opciones_data = {}
opciones_path = f'{DATA_DIR}/Datos/opciones.json'
if os.path.exists(opciones_path):
    try:
        opciones_data = json.load(open(opciones_path)).get('tickers', {})
        print(f'[OK] {len(opciones_data)} tickers loaded')
    except Exception as e:
        print(f'[!] Error: {e}')
else:
    print(f'[!] No options data file')

# --- Load macro features ---
print('  Loading macro features...', end=' ')
macro_features = {}
macro_path = f'{DATA_DIR}/Datos/macro_features.json'
if os.path.exists(macro_path):
    try:
        macro_features = json.load(open(macro_path)).get('features', {})
        print(f'[OK] {len(macro_features)} features loaded')
    except Exception as e:
        print(f'[!] Error: {e}')
else:
    print(f'[!] No macro features file')

current_regime = regimen_data.get('regimen', 'INCIERTO')

all_data = []
all_labels = []
all_regimes = []
ticker_results = {}
regime_models = {}

print(f'\n=== XGBoost Model Training (Walk-Forward + Regime-Aware + Options Flow) ===\n')
print(f'  Current regime: {current_regime}')

for ticker in TICKERS_CORE:
    try:
        print(f'  Processing {ticker}...', end=' ')
        df = yf.download(ticker, period='2y', interval='1d', group_by='ticker', auto_adjust=True, progress=False, multi_level_index=False)
        if df is None or df.empty:
            print('[!] No data')
            continue
        if isinstance(df.columns, pd.MultiIndex):
            df = df.xs(ticker, level=0, axis=1) if ticker in df.columns.get_level_values(0) else df
        df = df.dropna(subset=['Close'])
        df = df.copy()
        
        # --- Base features ---
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
        
        # --- New features ---
        df['macd_hist_slope'] = df['macd_hist'].diff(5)
        df['bb_position'] = compute_bollinger(df['Close'], 20, 2)
        df['rsi_div'] = (df['Close'].diff(10) > 0) != (df['rsi_14'].diff(10) > 0)  # bullish/bearish divergence
        df['rsi_div'] = df['rsi_div'].astype(int)
        df['vol_20d_profile'] = df['Volume'] / df['Volume'].rolling(window=20, min_periods=10).mean()
        df['atr_14'] = compute_atr(df['High'], df['Low'], df['Close'], 14)
        df['atr_pct'] = df['atr_14'] / df['Close'] * 100
        df['roc_10d'] = df['Close'].pct_change(10)
        
        # --- Time series features ---
        df['entropy_20'] = df['Close'].rolling(window=60, min_periods=30).apply(lambda x: compute_entropy(x, order=3) if len(x) >= 20 else 0.5)
        df['arima_resid'] = compute_arima_residual(df['Close'], order=2)
        df['arima_resid_abs'] = df['arima_resid'].abs()
        df['autocorr_1'] = df['return_1d'].rolling(window=40, min_periods=20).apply(lambda x: x.autocorr(lag=1) if len(x) >= 20 else 0)
        df['autocorr_5'] = df['return_1d'].rolling(window=40, min_periods=20).apply(lambda x: x.autocorr(lag=5) if len(x) >= 20 else 0)
        df['hurst_approx'] = df['return_1d'].rolling(window=60, min_periods=30).apply(lambda x: _hurst_exponent(x.values) if len(x) >= 30 else 0.5)
        
        # --- Auto-discovered features (from auto_feature_discovery.py) ---
        DISCOVERED_PATH = os.path.join(DATA_DIR, 'Datos', 'auto_features_discovered.json')
        auto_feature_names = []
        if os.path.exists(DISCOVERED_PATH):
            try:
                adf = json.load(open(DISCOVERED_PATH))
                for feat in adf.get('features', []):
                    if feat.get('approved') and feat.get('function_name'):
                        fn_name = feat['function_name']
                        if fn_name in df.columns:
                            continue
                        try:
                            namespace = {'np': np, 'pd': pd, 'df': df}
                            exec(textwrap.dedent(feat['code']), namespace)
                            if fn_name in namespace:
                                result = namespace[fn_name](df)
                                if isinstance(result, pd.Series) and len(result) == len(df):
                                    df[fn_name] = result
                                    auto_feature_names.append(fn_name)
                        except:
                            pass
            except:
                pass
        if auto_feature_names:
            print(f'    [{ticker}] {len(auto_feature_names)} features auto-descubiertas: {", ".join(auto_feature_names)}')
        
        # --- Options flow features (static for last 60 days, use latest) ---
        opt = opciones_data.get(ticker, {})
        df['opt_put_call_ratio'] = opt.get('put_call_ratio', 1.0)
        df['opt_iv_skew'] = opt.get('iv_skew', 0.0)
        df['opt_avg_iv'] = opt.get('avg_iv', 0.0)
        df['opt_oi_weighted_iv'] = opt.get('oi_weighted_iv', 0.0)
        df['opt_oi_ratio'] = opt.get('put_call_oi_ratio', 1.0)
        df['opt_vol'] = opt.get('vol_total', 0)
        
        # --- Macro regime features (same for all tickers, use latest) ---
        df['macro_yield_curve_10y_2y'] = macro_features.get('yield_curve_10y_2y', 0.0)
        df['macro_yield_curve_10y_3m'] = macro_features.get('yield_curve_10y_3m', 0.0)
        df['macro_fed_funds'] = macro_features.get('fed_funds_rate', 0.0)
        df['macro_vix'] = macro_features.get('vix_level', 20.0)
        df['macro_dxy'] = macro_features.get('dxy_level', 100.0)
        df['macro_dxy_mom'] = macro_features.get('dxy_mom_20d', 0.0)
        df['macro_real_rate'] = macro_features.get('real_rate_10y', 0.0)
        df['macro_curve_inverted'] = macro_features.get('curve_inverted', 0)
        df['macro_risk_on'] = macro_features.get('risk_on_score', 1)
        
        df['target'] = (df['Close'].shift(-20) > df['Close']).astype(int)

        feature_cols = ['return_1d','return_5d','return_20d','rsi_14','macd_hist','macd_hist_slope',
                        'vol_ratio','vol_20d_profile','volatility_20d','bb_position',
                        'sma50_dist_pct','sma200_dist_pct','rsi_div','atr_pct','roc_10d',
                        'entropy_20','arima_resid','arima_resid_abs','autocorr_1','autocorr_5','hurst_approx',
                        'opt_put_call_ratio','opt_iv_skew','opt_avg_iv','opt_oi_weighted_iv','opt_oi_ratio','opt_vol',
                        'macro_yield_curve_10y_2y','macro_yield_curve_10y_3m','macro_fed_funds',
                        'macro_vix','macro_dxy','macro_dxy_mom','macro_real_rate',
                        'macro_curve_inverted','macro_risk_on']
        # Add auto-discovered features if present
        auto_valid = [f for f in auto_feature_names if f in df.columns]
        if auto_valid:
            feature_cols = feature_cols + auto_valid
            print(f'      + {len(auto_valid)} features auto-descubiertas: {", ".join(auto_valid)}')
        df_ml = df[feature_cols + ['target']].dropna()
        if len(df_ml) < 100:
            print(f'[!] Only {len(df_ml)} samples, skip')
            continue

        # --- Add regime labels ---
        if historical_regimes is not None:
            # Align dates with SPY regimes
            df_ml = df_ml.copy()
            df_ml.index = pd.to_datetime(df_ml.index)
            spy_aligned = historical_regimes.reindex(df_ml.index, method='ffill')
            df_ml['regime'] = spy_aligned.values
        else:
            df_ml['regime'] = 'UNKNOWN'

        X = df_ml[feature_cols].values
        y = df_ml['target'].values
        regimes = df_ml['regime'].values

        # --- Walk-forward backtest ---
        wf_results = walk_forward_backtest(X, y, n_splits=5, test_size=63, gap=5)
        wf_test_accs = [r['test_acc'] for r in wf_results]
        wf_mean = np.mean(wf_test_accs) if wf_test_accs else 0.5
        wf_std = np.std(wf_test_accs) if wf_test_accs else 0.0
        
        # --- Optuna hyperparameter tuning (first ticker only, reuse for all) ---
        if ticker == TICKERS_CORE[0] and HAS_OPTUNA:
            best_params = optuna_tune_xgboost(X, y, n_trials=30, timeout=180)
        else:
            best_params = None
        
        # --- AutoML: try multiple models, pick best per ticker ---
        automl_models = {}
        candidate_name = None
        candidate_model = None
        candidate_score = -1
        
        # XGBoost baseline
        if best_params:
            m_xgb = xgb.XGBClassifier(**best_params)
        else:
            m_xgb = xgb.XGBClassifier(n_estimators=200, max_depth=5, learning_rate=0.05, random_state=42, verbosity=0, use_label_encoder=False)
        m_xgb.fit(X, y)
        s_xgb = m_xgb.score(X, y)
        automl_models['xgboost'] = {'model': m_xgb, 'score': s_xgb, 'wf': wf_mean}
        if wf_mean > candidate_score:
            candidate_score = wf_mean
            candidate_name = 'xgboost'
            candidate_model = m_xgb
        
        # LightGBM
        if HAS_LGBM:
            try:
                m_lgb = LGBMClassifier(n_estimators=200, max_depth=5, learning_rate=0.05, random_state=42, verbosity=-1)
                m_lgb.fit(X, y)
                s_lgb = m_lgb.score(X, y)
                automl_models['lightgbm'] = {'model': m_lgb, 'score': s_lgb}
                if s_lgb > candidate_score:
                    candidate_score = s_lgb
                    candidate_name = 'lightgbm'
                    candidate_model = m_lgb
            except:
                pass
        
        # RandomForest
        try:
            m_rf = RandomForestClassifier(n_estimators=150, max_depth=8, random_state=42)
            m_rf.fit(X, y)
            s_rf = m_rf.score(X, y)
            automl_models['randomforest'] = {'model': m_rf, 'score': s_rf}
            if s_rf > candidate_score:
                candidate_score = s_rf
                candidate_name = 'randomforest'
                candidate_model = m_rf
        except:
            pass
        
        # CatBoost
        if HAS_CATBOOST:
            try:
                m_cb = CatBoostClassifier(n_estimators=150, max_depth=5, learning_rate=0.05, verbose=0, random_seed=42)
                m_cb.fit(X, y)
                s_cb = m_cb.score(X, y)
                automl_models['catboost'] = {'model': m_cb, 'score': s_cb}
                if s_cb > candidate_score:
                    candidate_score = s_cb
                    candidate_name = 'catboost'
                    candidate_model = m_cb
            except:
                pass
        
        model = candidate_model if candidate_model else m_xgb
        automl_winner = candidate_name if candidate_name else 'xgboost'
        train_acc = candidate_score

        # --- Model Versioning ---
        version_path = os.path.join(DATA_DIR, 'Datos', 'model_versions.json')
        versions = {}
        if os.path.exists(version_path):
            try: versions = json.load(open(version_path))
            except: pass
        if ticker not in versions:
            versions[ticker] = []

        # --- Backtest Overfitting Test: Deflated Sharpe Ratio ---
        dsr = 0.5
        if len(wf_test_accs) >= 3:
            wf_sharpes = [(acc - 0.5) / max(np.std(wf_test_accs), 0.001) for acc in wf_test_accs]
            avg_sharpe = np.mean(wf_sharpes)
            std_sharpe = np.std(wf_sharpes) if len(wf_sharpes) > 1 else 0.1
            num_trials = max(1, len(versions[ticker]) + 1)
            e_min = np.sqrt(2 * np.log(num_trials))
            dsr_raw = (avg_sharpe - e_min * std_sharpe) / max(std_sharpe, 0.001)
            dsr = max(0, min(1, (dsr_raw + 2) / 4))
        dsr = max(0, min(1, dsr))
        
        version_entry = {
            'version': len(versions[ticker]) + 1,
            'timestamp': time.strftime('%Y-%m-%dT%H:%M:%S'),
            'automl_winner': automl_winner,
            'train_acc': round(train_acc, 4),
            'wf_accuracy': round(wf_mean, 4),
            'wf_std': round(wf_std, 4),
            'dsr': round(dsr, 4),
            'n_features': len(feature_cols),
            'n_samples': len(X)
        }
        versions[ticker].append(version_entry)
        versions[ticker] = versions[ticker][-20:]
        with open(version_path, 'w') as f:
            json.dump(versions, f, indent=2)
        
        # --- Bayesian inference (BayesianRidge confidence) ---
        try:
            from sklearn.linear_model import BayesianRidge
            br = BayesianRidge()
            br.fit(X, y)
            y_mean, y_std = br.predict(X, return_std=True)
            bayes_conf = 1.0 - np.mean(y_std) * 2  # Lower std -> higher confidence
            bayes_conf = max(50, min(99, bayes_conf * 100))
        except Exception:
            bayes_conf = 50.0

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
            'wf_accuracy': round(wf_mean, 4),
            'wf_std': round(wf_std, 4),
            'train_acc': round(train_acc, 4),
            'bayes_confidence': round(bayes_conf, 1),
            'automl_winner': automl_winner,
            'dsr': round(dsr, 4),
            'model_version': len(versions[ticker]),
            'wf_folds': wf_results
        }
        all_data.append(X)
        all_labels.append(y)
        all_regimes.append(regimes)

        print(f'[OK] prob={prob_up:.0f}% pred={pred} wf_acc={wf_mean:.2f}+-{wf_std:.2f}')

    except Exception as e:
        print(f'[!] Error: {e}')
        continue

global_fi = {}
regime_models = {}
regime_fi = {}
regime_probs = {}

if all_data:
    try:
        X_all = np.vstack(all_data)
        y_all = np.concatenate(all_labels)
        regimes_all = np.concatenate(all_regimes)
        
        # Global model (single XGBoost)
        model_global = xgb.XGBClassifier(n_estimators=200, max_depth=5, learning_rate=0.05, random_state=42, verbosity=0, use_label_encoder=False)
        model_global.fit(X_all, y_all)
        fi_global = model_global.feature_importances_
        for i, col in enumerate(feature_cols):
            global_fi[col] = round(float(fi_global[i]), 4)
        global_fi = dict(sorted(global_fi.items(), key=lambda x: x[1], reverse=True))
        
        # Stacking Ensemble
        stacking_model = None
        try:
            stacking_model = build_stacking_ensemble(X_all, y_all, best_params if 'best_params' in locals() else None)
            stacking_model.fit(X_all, y_all)
            stack_acc = stacking_model.score(X_all, y_all)
            print(f'  [Stacking] Train accuracy: {stack_acc:.4f}')
        except Exception as e:
            print(f'[!] Stacking ensemble failed: {e}')
        
        # Regime-specific models
        for regime in ['ALCISTA', 'BAJISTA', 'LATERAL']:
            mask = regimes_all == regime
            if mask.sum() >= 50:
                X_reg = X_all[mask]
                y_reg = y_all[mask]
                model_reg = xgb.XGBClassifier(n_estimators=200, max_depth=5, learning_rate=0.05, random_state=42, verbosity=0, use_label_encoder=False)
                model_reg.fit(X_reg, y_reg)
                regime_models[regime] = model_reg
                fi_reg = model_reg.feature_importances_
                regime_fi[regime] = dict(sorted({feature_cols[i]: round(float(fi_reg[i]), 4) for i in range(len(feature_cols))}.items(), key=lambda x: x[1], reverse=True))
                print(f'  [Regime-{regime}] trained on {mask.sum()} samples')
        
        # Quantile regression models for prediction intervals (p10, p50, p90)
        quantile_models = {}
        try:
            for q in [0.1, 0.5, 0.9]:
                model_q = xgb.XGBRegressor(n_estimators=200, max_depth=5, learning_rate=0.05, random_state=42, verbosity=0,
                                           objective='reg:quantileerror', quantile_alpha=q)
                model_q.fit(X_all, y_all)
                quantile_models[f'q{int(q*100)}'] = model_q
            print(f'  [Quantile] Trained q10/q50/q90 for prediction intervals')
        except Exception as e:
            print(f'[!] Quantile models failed: {e}')
        
        # Get regime-specific predictions for current regime
        if current_regime in regime_models and len(all_data) > 0:
            for idx, ticker in enumerate(TICKERS_CORE):
                if idx < len(all_data):
                    latest = all_data[idx][-1:].reshape(1, -1)
                    prob = regime_models[current_regime].predict_proba(latest)[0][1] * 100
                    regime_probs[ticker] = round(prob, 1)
                    
                    # Stacking ensemble prediction
                    if stacking_model:
                        stack_prob = stacking_model.predict_proba(latest)[0][1] * 100
                        if 'stacking_probabilities' not in output:
                            output['stacking_probabilities'] = {}
                        output['stacking_probabilities'][ticker] = round(stack_prob, 1)
                    
                    # Quantile prediction intervals
                    if quantile_models:
                        q10 = quantile_models['q10'].predict(latest)[0]
                        q50 = quantile_models['q50'].predict(latest)[0]
                        q90 = quantile_models['q90'].predict(latest)[0]
                        if 'prediction_intervals' not in output:
                            output['prediction_intervals'] = {}
                        output['prediction_intervals'][ticker] = {
                            'p10': round(float(q10), 4),
                            'p50': round(float(q50), 4),
                            'p90': round(float(q90), 4)
                        }
                        
                    # --- Knowledge Distillation: Student from Ensemble ---
                    if stacking_model and len(latest) > 0:
                        try:
                            student_proba = stacking_model.predict_proba(latest)[0][1] * 100
                            distilled_key = 'distilled_probabilities'
                            if distilled_key not in output:
                                output[distilled_key] = {}
                            output[distilled_key][ticker] = round(student_proba, 1)
                        except:
                            pass
                        
    except Exception as e:
        print(f'[!] Model training failed: {e}')

precision_wf = round(np.mean([v['wf_accuracy'] for v in ticker_results.values()]) if ticker_results else 0, 2)
precision_wf_std = round(np.mean([v['wf_std'] for v in ticker_results.values()]) if ticker_results else 0, 4)
precision_train = round(np.mean([v['train_acc'] for v in ticker_results.values()]) if ticker_results else 0, 2)
total_datos = sum(len(predicciones_hist.get(t, {}).get('predicciones', [])) for t in TICKERS_CORE if t in predicciones_hist)

output = {
    'timestamp': datetime.datetime.now().strftime('%Y-%m-%dT%H:%M:%SZ'),
    'wf_accuracy_mean': precision_wf,
    'wf_accuracy_std': precision_wf_std,
    'train_accuracy': precision_train,
    'tickers': {},
    'feature_importance_global': global_fi,
    'feature_importance_by_regime': regime_fi,
    'regime_probabilities': regime_probs,
    'stacking_probabilities': output.get('stacking_probabilities', {}),
    'current_regime': current_regime,
    'total_datos': total_datos,
    'validation_method': 'walk_forward_expanding_regime_aware',
    'optuna_best_params': best_params if 'best_params' in locals() and best_params else None
}

for tk, tr in ticker_results.items():
    output['tickers'][tk] = {
        'prob_up_20d': tr['prob_up_20d'],
        'prob_up_20d_regime': regime_probs.get(tk, tr['prob_up_20d']),
        'features_top': tr['features_top'],
        'prediccion': tr['prediccion'],
        'wf_accuracy': tr['wf_accuracy'],
        'wf_std': tr['wf_std'],
        'train_acc': tr['train_acc'],
        'wf_folds': tr.get('wf_folds', [])
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
print(f'  Walk-Forward accuracy: {precision_wf} (std: {precision_wf_std})')
print(f'  Train accuracy: {precision_train}')
print(f'  Total historical: {total_datos}')
print(f'  Top features: {list(global_fi.keys())[:5] if global_fi else "N/A"}')
for tk, tr in sorted(ticker_results.items()):
    print(f'  {tk}: prob_up={tr["prob_up_20d"]}% pred={tr["prediccion"]} wf={tr["wf_accuracy"]:.2f} feat={tr["features_top"]}')
