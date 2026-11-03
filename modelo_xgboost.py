#!/usr/bin/env python3
"""
modelo_xgboost.py - ML pipeline: features, training, calibration, prediction.
Refactored: all logic in classes, no side effects at import time.
Usage:
    from modelo_xgboost import MLPipeline
    pipe = MLPipeline()
    pipe.run_all()
"""
import datetime
import json
import os
import textwrap
import time
import warnings
from typing import Any

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import BayesianRidge, LogisticRegression
from sklearn.model_selection import TimeSeriesSplit

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

try:
    from scipy import stats as scipy_stats
except ImportError:
    scipy_stats = None

from logger import get_module_logger

log = get_module_logger('modelo_xgboost')

warnings.filterwarnings('ignore')

DATA_DIR = os.environ.get('GITHUB_WORKSPACE', '.')


# ============================================================
# FEATURE BUILDERS
# ============================================================

class FeatureBuilder:
    RSI_PERIOD = 14
    MACD_FAST = 12
    MACD_SLOW = 26
    MACD_SIGNAL = 9
    BB_PERIOD = 20
    BB_STD = 2
    ATR_PERIOD = 14

    @staticmethod
    def compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
        delta = series.diff()
        gain = delta.clip(lower=0)
        loss = (-delta).clip(lower=0)
        avg_gain = gain.rolling(window=period, min_periods=period).mean()
        avg_loss = loss.rolling(window=period, min_periods=period).mean()
        rs = avg_gain / avg_loss.replace(0, np.nan)
        return 100 - (100 / (1 + rs))

    @staticmethod
    def compute_macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.Series:
        ema_fast = series.ewm(span=fast, adjust=False).mean()
        ema_slow = series.ewm(span=slow, adjust=False).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal, adjust=False).mean()
        return macd_line - signal_line

    @staticmethod
    def compute_bollinger(series: pd.Series, window: int = 20, num_std: int = 2) -> pd.Series:
        sma = series.rolling(window=window, min_periods=window).mean()
        std = series.rolling(window=window, min_periods=window).std()
        upper = sma + num_std * std
        lower = sma - num_std * std
        return (series - lower) / (upper - lower).replace(0, np.nan)

    @staticmethod
    def compute_atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
        tr1 = high - low
        tr2 = (high - close.shift()).abs()
        tr3 = (low - close.shift()).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        return tr.rolling(window=period, min_periods=period).mean()

    @staticmethod
    def compute_entropy(series: np.ndarray, order: int = 3) -> float:
        n = len(series)
        if n < order + 5:
            return 0.5
        patterns = np.array([series[i:i + order] for i in range(n - order + 1)])
        count = np.sum(np.max(np.abs(patterns[:, None] - patterns[None, :]), axis=2) < 0.2 * np.std(series), axis=1) - 1
        phi_m = np.mean(np.log(np.maximum(count, 1) / (n - order)))
        patterns2 = np.array([series[i:i + order + 1] for i in range(n - order)])
        count2 = np.sum(np.max(np.abs(patterns2[:, None] - patterns2[None, :]), axis=2) < 0.2 * np.std(series), axis=1) - 1
        phi_m1 = np.mean(np.log(np.maximum(count2, 1) / (n - order - 1)))
        return max(0, min(2, phi_m - phi_m1))

    @staticmethod
    def compute_hurst(ts: np.ndarray) -> float:
        if len(ts) < 30 or np.std(ts) == 0:
            return 0.5
        lags = range(2, min(20, len(ts) // 2))
        tau = []
        for lag in lags:
            diff = np.subtract(ts[lag:], ts[:-lag])
            tau.append(np.std(diff))
        if len(tau) < 2 or min(tau) == 0:
            return 0.5
        m = np.polyfit(np.log(list(lags)), np.log(tau), 1)
        return max(0, min(1, m[0] / 2))

    @staticmethod
    def compute_arima_residual(series: pd.Series, order: int = 2) -> pd.Series:
        from sklearn.linear_model import LinearRegression
        n = len(series)
        if n < order + 5:
            return pd.Series(0, index=series.index)
        X = np.array([series.shift(i).values for i in range(1, order + 1)]).T
        y = series.values
        mask = ~np.any(np.isnan(X), axis=1) & ~np.isnan(y)
        if mask.sum() < order + 2:
            return pd.Series(0, index=series.index)
        lr = LinearRegression()
        lr.fit(X[mask], y[mask])
        resid = np.full(len(series), np.nan)
        resid[mask] = y[mask] - lr.predict(X[mask])
        return pd.Series(resid, index=series.index)

    @classmethod
    def build_features(cls, df: pd.DataFrame, options_data: dict = None,
                        macro_features: dict = None,
                        auto_features: list[dict] = None,
                        spy_close: pd.Series = None,
                        short_data: dict = None,
                        sector_momentum: dict = None,
                        ticker: str = None,
                        earnings_data: dict = None) -> tuple[pd.DataFrame, list[str]]:
        df = df.copy()

        df['return_1d'] = df['Close'].pct_change()
        df['return_5d'] = df['Close'].pct_change(5)
        df['return_20d'] = df['Close'].pct_change(20)
        df['mom_12_1m'] = df['Close'].shift(21) / df['Close'].shift(252) - 1
        df['mom_6_1m'] = df['Close'].shift(21) / df['Close'].shift(126) - 1
        df['reversal_1m'] = df['Close'].shift(1) / df['Close'].shift(21) - 1
        df['dist_52w_high'] = df['Close'] / df['Close'].rolling(252).max() - 1
        df['rsi_14'] = cls.compute_rsi(df['Close'], 14)
        df['macd_hist'] = cls.compute_macd(df['Close'], 12, 26, 9)
        df['vol_ratio'] = df['Volume'] / df['Volume'].rolling(window=50, min_periods=20).mean()
        df['volatility_20d'] = df['return_1d'].rolling(window=20, min_periods=10).std()
        df['sma50'] = df['Close'].rolling(window=50, min_periods=20).mean()
        df['sma200'] = df['Close'].rolling(window=200, min_periods=50).mean()
        df['sma50_dist_pct'] = (df['Close'] - df['sma50']) / df['sma50'] * 100
        df['sma200_dist_pct'] = (df['Close'] - df['sma200']) / df['sma200'] * 100

        df['macd_hist_slope'] = df['macd_hist'].diff(5)
        df['bb_position'] = cls.compute_bollinger(df['Close'], 20, 2)
        df['rsi_div'] = ((df['Close'].diff(10) > 0) != (df['rsi_14'].diff(10) > 0)).astype(int)
        df['vol_20d_profile'] = df['Volume'] / df['Volume'].rolling(window=20, min_periods=10).mean()
        df['atr_14'] = cls.compute_atr(df['High'], df['Low'], df['Close'], 14)
        df['atr_pct'] = df['atr_14'] / df['Close'] * 100
        df['roc_10d'] = df['Close'].pct_change(10)

        if 'High' in df.columns and 'Low' in df.columns:
            df['entropy_20'] = df['Close'].rolling(window=60, min_periods=30).apply(
                lambda x: cls.compute_entropy(x.values, order=3) if len(x) >= 20 else 0.5, raw=False
            )
            df['arima_resid'] = cls.compute_arima_residual(df['Close'], order=2)
            df['arima_resid_abs'] = df['arima_resid'].abs()
            df['autocorr_1'] = df['return_1d'].rolling(window=40, min_periods=20).apply(
                lambda x: x.autocorr(lag=1) if len(x) >= 20 else 0, raw=False
            )
            df['autocorr_5'] = df['return_1d'].rolling(window=40, min_periods=20).apply(
                lambda x: x.autocorr(lag=5) if len(x) >= 20 else 0, raw=False
            )
            df['hurst_approx'] = df['return_1d'].rolling(window=60, min_periods=30).apply(
                lambda x: cls.compute_hurst(x.values) if len(x) >= 30 else 0.5, raw=False
            )
        else:
            for col in ['entropy_20', 'arima_resid', 'arima_resid_abs',
                         'autocorr_1', 'autocorr_5', 'hurst_approx']:
                df[col] = 0.0

        # opciones.json marca `disponible: false` con campos en null cuando no hay
        # datos de mercado. Esos campos NO son cero: son ausencia de dato, y se
        # sustituyen por el neutro histórico para no inyectar NaN al modelo.
        opt = options_data or {}
        opt_disponible = bool(opt.get('disponible', True)) and opt.get('put_call_ratio') is not None

        def opt_num(clave, neutro):
            if not opt_disponible:
                return neutro
            valor = opt.get(clave)
            return valor if valor is not None else neutro

        df['opt_put_call_ratio'] = opt_num('put_call_ratio', 1.0)
        df['opt_iv_skew'] = opt_num('iv_skew', 0.0)
        df['opt_avg_iv'] = opt_num('avg_iv', 0.0)
        df['opt_oi_weighted_iv'] = opt_num('oi_weighted_iv', 0.0)
        df['opt_oi_ratio'] = opt_num('put_call_oi_ratio', 1.0)
        df['opt_vol'] = opt_num('vol_total', 0)

        macro = macro_features or {}
        df['macro_yield_curve_10y_2y'] = macro.get('yield_curve_10y_2y', 0.0)
        df['macro_yield_curve_10y_3m'] = macro.get('yield_curve_10y_3m', 0.0)
        df['macro_fed_funds'] = macro.get('fed_funds_rate', 0.0)
        df['macro_vix'] = macro.get('vix_level', 20.0)
        df['macro_dxy'] = macro.get('dxy_level', 100.0)
        df['macro_dxy_mom'] = macro.get('dxy_mom_20d', 0.0)
        df['macro_real_rate'] = macro.get('real_rate_10y', 0.0)
        df['macro_curve_inverted'] = macro.get('curve_inverted', 0)
        df['macro_risk_on'] = macro.get('risk_on_score', 1)

        TICKERS_SECTOR = {t: s for s, tkrs in {
            'Semiconductores': ['NVDA','MU','AVGO','TSM','ARM'],
            'Servidores IA': ['DELL','SMCI','HPE'],
            'Software IA': ['DDOG','SNOW','NOW'],
            'Ciberseguridad': ['CRWD','PANW','OKTA'],
            'Almacenamiento': ['NTAP','CLS'],
            'Consumer Tech': ['AAPL','AMZN','GOOGL','META','MSFT'],
            'Farmaceutico': ['LLY'],
            'Semicon Equip': ['AMAT','LRCX'],
            'Cloud/Database': ['ORCL'],
            'Industrial': ['HON','GE'],
            'Movilidad/Tech': ['UBER'],
            'Consumo Defensivo': ['COST'],
            'Utilities/Energy': ['NEE']
        }.items() for t in tkrs}
        ticker_actual = ticker
        if short_data and ticker_actual and ticker_actual in short_data:
            si = short_data[ticker_actual]
            df['short_pct_float'] = si.get('short_pct_float', 0.0)
            df['days_to_cover'] = si.get('days_to_cover', si.get('short_ratio', 0.0))
        else:
            df['short_pct_float'] = 0.0
            df['days_to_cover'] = 0.0
        if sector_momentum and ticker_actual:
            sec = TICKERS_SECTOR.get(ticker_actual)
            sec_data = sector_momentum.get('sectors', {})
            comp = 0.0
            for name, info in sec_data.items():
                if sec and (sec.lower() in name.lower() or name.lower() in sec.lower()):
                    comp = info.get('momentum', {}).get('composite', 0.0)
                    break
            df['sector_mom_composite'] = comp
        else:
            df['sector_mom_composite'] = 0.0

        eps_surprise = 0.0
        if earnings_data and ticker_actual:
            info = earnings_data.get(ticker_actual, {}) if isinstance(earnings_data, dict) else {}
            er = (info.get('earnings') or {}) if isinstance(info, dict) else {}
            sp = str(er.get('surprise_pct', '') or '').replace('+', '').replace('%', '').strip()
            try:
                eps_surprise = float(sp) / 100.0
            except (ValueError, TypeError):
                eps_surprise = 0.0
        df['eps_surprise'] = eps_surprise

        horizon = 20
        fwd_ret = df['Close'].shift(-horizon) / df['Close'] - 1
        if spy_close is not None and len(spy_close) > 0:
            spy_fwd = spy_close.reindex(df.index).ffill().shift(-horizon) / spy_close.reindex(df.index).ffill() - 1
            df['target'] = (fwd_ret > spy_fwd).astype(int)
        else:
            df['target'] = (fwd_ret > 0).astype(int)
        df.loc[fwd_ret.isna(), 'target'] = np.nan

        r = df['Close'].pct_change()
        df['fwd_vol_20d'] = r.rolling(20).std().shift(-20) * np.sqrt(252)

        feature_cols = [
            'return_1d', 'return_5d', 'return_20d', 'rsi_14', 'macd_hist', 'macd_hist_slope',
            'mom_12_1m', 'mom_6_1m', 'reversal_1m', 'dist_52w_high',
            'short_pct_float', 'days_to_cover', 'sector_mom_composite', 'eps_surprise',
            'vol_ratio', 'vol_20d_profile', 'volatility_20d', 'bb_position',
            'sma50_dist_pct', 'sma200_dist_pct', 'rsi_div', 'atr_pct', 'roc_10d',
            'entropy_20', 'autocorr_1', 'autocorr_5', 'hurst_approx',
            'opt_put_call_ratio', 'opt_iv_skew', 'opt_avg_iv', 'opt_oi_weighted_iv', 'opt_oi_ratio', 'opt_vol',
            'macro_yield_curve_10y_2y', 'macro_yield_curve_10y_3m', 'macro_fed_funds',
            'macro_vix', 'macro_dxy', 'macro_dxy_mom', 'macro_real_rate',
            'macro_curve_inverted', 'macro_risk_on'
        ]

        if auto_features:
            valid = [f for f in auto_features if f.get('approved') and f.get('function_name')]
            fn_names = []
            for feat in valid:
                fn_name = feat['function_name']
                if fn_name in df.columns:
                    continue
                if cls._safe_apply_auto_feature(df, feat.get('code', ''), fn_name):
                    fn_names.append(fn_name)
            feature_cols.extend(fn_names)

        return df, feature_cols

    @staticmethod
    def _safe_apply_auto_feature(df: pd.DataFrame, code: str, fn_name: str) -> bool:
        SAFE_BUILTINS = {
            'abs': abs, 'min': min, 'max': max, 'len': len,
            'round': round, 'sum': sum, 'sorted': sorted,
            'range': range, 'enumerate': enumerate, 'zip': zip,
            'True': True, 'False': False, 'None': None,
            'int': int, 'float': float, 'str': str, 'bool': bool,
            'list': list, 'dict': dict, 'tuple': tuple, 'set': set,
        }
        namespace = {'__builtins__': SAFE_BUILTINS, 'np': np, 'pd': pd, 'df': df}
        try:
            exec(textwrap.dedent(code), namespace)
            if fn_name in namespace and callable(namespace[fn_name]):
                result = namespace[fn_name](df)
                if isinstance(result, pd.Series) and len(result) == len(df):
                    df[fn_name] = result
                    return True
        except Exception as e:
            log.warning(f"Auto-feature {fn_name} failed: {e}", feature=fn_name)
        return False


# ============================================================
# WALK-FORWARD VALIDATION
# ============================================================

class WalkForwardValidator:
    def __init__(self, n_splits: int = 3, test_size: int = 42, gap: int = 20):
        self.n_splits = n_splits
        self.test_size = test_size
        self.gap = gap

    def validate(self, X: np.ndarray, y: np.ndarray,
                 model_factory=None, params: dict = None) -> list[dict]:
        tscv = TimeSeriesSplit(n_splits=self.n_splits, test_size=self.test_size, gap=self.gap)
        results = []
        oos_proba, oos_y = [], []
        for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            if len(np.unique(y_train)) < 2 or len(np.unique(y_test)) < 2:
                continue
            if model_factory:
                model = model_factory(params)
            else:
                model = xgb.XGBClassifier(
                    n_estimators=200, max_depth=5, learning_rate=0.05,
                    random_state=42, verbosity=0, use_label_encoder=False
                )
            model.fit(X_train, y_train)
            train_acc = model.score(X_train, y_train)
            test_acc = model.score(X_test, y_test)
            brier = None
            if hasattr(model, 'predict_proba'):
                try:
                    from sklearn.metrics import brier_score_loss
                    proba = model.predict_proba(X_test)[:, 1]
                    brier = float(brier_score_loss(y_test, proba))
                    oos_proba.extend(proba.tolist())
                    oos_y.extend(y_test.tolist())
                except Exception:
                    brier = None
            results.append({
                'fold': fold, 'train_acc': train_acc, 'test_acc': test_acc,
                'test_brier': brier,
                'train_size': len(train_idx), 'test_size': len(test_idx)
            })
        if results and oos_proba and len(oos_proba) >= 30:
            try:
                from sklearn.isotonic import IsotonicRegression
                from sklearn.metrics import brier_score_loss
                iso = IsotonicRegression(out_of_bounds='clip')
                cal = iso.fit_transform(np.array(oos_proba), np.array(oos_y))
                results[-1]['brier_calib_global'] = float(brier_score_loss(np.array(oos_y), cal))
            except Exception:
                pass
        return results


# ============================================================
# CHRONOLOGICAL BLOCK CV (partition-compatible para Stacking)
# ============================================================

class _ChronologicalBlockCV:
    """CV cronologico que cubre todos los samples (particion completa).
    fold i: test = bloque contiguo i, train = resto. A diferencia de
    TimeSeriesSplit, permite cross_val_predict / StackingClassifier."""

    def __init__(self, n_splits: int = 3):
        self.n_splits = n_splits

    def get_n_splits(self, X=None, y=None, groups=None):
        return self.n_splits

    def split(self, X, y=None, groups=None):
        n = len(X)
        fold = max(1, n // self.n_splits)
        for i in range(self.n_splits):
            start = i * fold
            end = n if i == self.n_splits - 1 else (i + 1) * fold
            test_idx = np.arange(start, end)
            train_idx = np.concatenate([np.arange(0, start), np.arange(end, n)])
            yield train_idx, test_idx


# ============================================================
# CALIBRATION
# ============================================================

class ProbabilityCalibrator:
    @staticmethod
    def calibrate(model, X_cal: np.ndarray, y_cal: np.ndarray, method: str = 'isotonic'):
        if method == 'isotonic':
            calibrator = IsotonicRegression(out_of_bounds='clip')
            probs = model.predict_proba(X_cal)[:, 1]
            calibrator.fit(probs, y_cal)
            return calibrator
        else:
            calibrator = CalibratedClassifierCV(model, method='sigmoid', cv='prefit')
            calibrator.fit(X_cal, y_cal)
            return calibrator


# ============================================================
# AUTOML / HYPERPARAM OPT
# ============================================================

class AutoMLTrainer:
    def __init__(self, best_params: dict | None = None):
        self.best_params = best_params

    def optuna_tune(self, X: np.ndarray, y: np.ndarray,
                    n_trials: int = 50, timeout: int = 300) -> dict | None:
        if not HAS_OPTUNA:
            log.warning("Optuna not installed, skipping hyperparameter tuning")
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
                'random_state': 42, 'verbosity': 0, 'use_label_encoder': False
            }
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

        log.info(f"Optuna starting ({n_trials} trials, {timeout}s)...")
        study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=42))
        study.optimize(objective, n_trials=n_trials, timeout=timeout)

        best_params = study.best_params
        best_params.update({'random_state': 42, 'verbosity': 0, 'use_label_encoder': False})
        log.info(f"Optuna best params: {best_params}", score=study.best_value)
        self.best_params = best_params
        return best_params

    def build_stacking(self, X: np.ndarray, y: np.ndarray) -> StackingClassifier | None:
        estimators = []
        xgb_params = self.best_params or {
            'n_estimators': 200, 'max_depth': 5, 'learning_rate': 0.05,
            'random_state': 42, 'verbosity': 0, 'use_label_encoder': False
        }
        estimators.append(('xgb', xgb.XGBClassifier(**xgb_params)))

        if HAS_LGBM:
            estimators.append(('lgb', LGBMClassifier(
                n_estimators=200, max_depth=5, learning_rate=0.05, random_state=42, verbosity=-1
            )))
        if HAS_CATBOOST:
            estimators.append(('cat', CatBoostClassifier(
                iterations=200, depth=5, learning_rate=0.05, random_state=42, verbose=False
            )))

        estimators.append(('rf', RandomForestClassifier(
            n_estimators=200, max_depth=8, random_state=42, n_jobs=-1
        )))

        stack = StackingClassifier(
            estimators=estimators,
            final_estimator=LogisticRegression(random_state=42, max_iter=1000),
            cv=_ChronologicalBlockCV(n_splits=3),
            n_jobs=-1, passthrough=True
        )
        log.info(f"Stacking ensemble: {[e[0] for e in estimators]}")
        return stack

    def train_best_model(self, X: np.ndarray, y: np.ndarray) -> tuple[Any, str, float]:
        candidate_name = 'xgboost'
        candidate_model = None
        candidate_score = -1.0

        xgb_params = self.best_params or {
            'n_estimators': 200, 'max_depth': 5, 'learning_rate': 0.05,
            'random_state': 42, 'verbosity': 0, 'use_label_encoder': False
        }
        m_xgb = xgb.XGBClassifier(**xgb_params)
        m_xgb.fit(X, y)
        s_xgb = m_xgb.score(X, y)
        if s_xgb > candidate_score:
            candidate_score = s_xgb
            candidate_name = 'xgboost'
            candidate_model = m_xgb

        if HAS_LGBM:
            try:
                m_lgb = LGBMClassifier(n_estimators=200, max_depth=5, learning_rate=0.05, random_state=42, verbosity=-1)
                m_lgb.fit(X, y)
                s = m_lgb.score(X, y)
                if s > candidate_score:
                    candidate_score, candidate_name, candidate_model = s, 'lightgbm', m_lgb
            except Exception as e:
                log.warning(f"LightGBM failed: {e}")

        try:
            m_rf = RandomForestClassifier(n_estimators=150, max_depth=8, random_state=42)
            m_rf.fit(X, y)
            s = m_rf.score(X, y)
            if s > candidate_score:
                candidate_score, candidate_name, candidate_model = s, 'randomforest', m_rf
        except Exception as e:
            log.warning(f"RandomForest failed: {e}")

        if HAS_CATBOOST:
            try:
                m_cb = CatBoostClassifier(iterations=150, depth=5, learning_rate=0.05, verbose=0, random_seed=42)
                m_cb.fit(X, y)
                s = m_cb.score(X, y)
                if s > candidate_score:
                    candidate_score, candidate_name, candidate_model = s, 'catboost', m_cb
            except Exception as e:
                log.warning(f"CatBoost failed: {e}")

        return candidate_model or m_xgb, candidate_name, candidate_score


# ============================================================
# REGIME DETECTION
# ============================================================

class RegimeDetector:
    @staticmethod
    def get_historical_regimes(spy_df: pd.DataFrame) -> pd.Series:
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
            s50, s200, r50 = sma50.iloc[i], sma200.iloc[i], ret_50d.iloc[i]
            if pd.isna(s50) or pd.isna(s200) or pd.isna(r50):
                regimes.iloc[i] = 'INCIERTO'
            elif close > s50 > sma200.iloc[i] and r50 > 5:
                regimes.iloc[i] = 'ALCISTA'
            elif close < s50 < sma200.iloc[i] and r50 < -5:
                regimes.iloc[i] = 'BAJISTA'
            elif (close > s50 and close < sma200.iloc[i]) or (close < s50 and close > sma200.iloc[i]):
                regimes.iloc[i] = 'LATERAL'
            else:
                regimes.iloc[i] = 'INCIERTO'
        return regimes

    @staticmethod
    def compute_deflated_sharpe(wf_test_accs: list[float], num_versions: int) -> float:
        if len(wf_test_accs) < 3:
            return 0.5
        wf_sharpes = [(acc - 0.5) / max(np.std(wf_test_accs), 0.001) for acc in wf_test_accs]
        avg_sharpe = np.mean(wf_sharpes)
        std_sharpe = np.std(wf_sharpes) if len(wf_sharpes) > 1 else 0.1
        num_trials = max(1, num_versions)
        e_min = np.sqrt(2 * np.log(num_trials))
        dsr_raw = (avg_sharpe - e_min * std_sharpe) / max(std_sharpe, 0.001)
        return max(0.0, min(1.0, (dsr_raw + 2) / 4))


# ============================================================
# MAIN PIPELINE
# ============================================================

class MLPipeline:
    TICKERS_CORE = [
        'NVDA', 'MU', 'DELL', 'AVGO', 'DDOG', 'SMCI', 'SNOW', 'CRWD', 'NOW', 'TSM',
        'ARM', 'OKTA', 'HPE', 'NTAP', 'CLS', 'AAPL', 'AMZN', 'GOOGL', 'META', 'MSFT',
        'LLY', 'AMAT', 'LRCX', 'PANW', 'ORCL', 'HON', 'UBER', 'GE', 'COST', 'NEE'
    ]
    NEWS_WEIGHT = 4.0

    def __init__(self, data_dir: str = None):
        self.data_dir = data_dir or DATA_DIR
        self.fb = FeatureBuilder()
        self.wf = WalkForwardValidator()
        self.automl = AutoMLTrainer()
        self.rd = RegimeDetector()
        self.pc = ProbabilityCalibrator()
        self.ticker_results: dict[str, dict] = {}
        self.global_feature_importance: dict[str, float] = {}
        self.regime_feature_importance: dict[str, dict[str, float]] = {}
        self.regime_models: dict[str, Any] = {}
        self.regime_probs: dict[str, float] = {}
        self.output: dict[str, Any] = {}

    def _load_json(self, path: str) -> dict:
        full_path = os.path.join(self.data_dir, 'Datos', path)
        if os.path.exists(full_path):
            try:
                with open(full_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                log.warning(f"Error loading {path}: {e}")
        return {}

    def _save_json(self, path: str, data: dict):
        full_path = os.path.join(self.data_dir, 'Datos', path)
        os.makedirs(os.path.dirname(full_path), exist_ok=True)

        class NpEncoder(json.JSONEncoder):
            def default(self, obj):
                if isinstance(obj, (np.integer, np.floating)):
                    return float(obj)
                if isinstance(obj, np.bool_):
                    return bool(obj)
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                return super().default(obj)

        with open(full_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, cls=NpEncoder)

    def _load_regime(self) -> dict:
        return self._load_json('regimen_mercado.json')

    def _load_options(self) -> dict:
        data = self._load_json('opciones.json')
        return data.get('tickers', {})

    def _load_macro(self) -> dict:
        data = self._load_json('macro_features.json')
        return data.get('features', {})

    def _load_auto_features_config(self) -> list[dict]:
        data = self._load_json('auto_features_discovered.json')
        return data.get('features', [])

    def _load_predicciones(self) -> dict:
        return self._load_json('predicciones_hist.json')

    def _load_calibracion(self) -> dict:
        return self._load_json('calibracion.json')

    def fetch_spy_regimes(self) -> pd.Series | None:
        import yfinance as yf
        try:
            spy_df = yf.download('SPY', period='2y', interval='1d', auto_adjust=True, progress=False, multi_level_index=False)
            if spy_df is not None and not spy_df.empty:
                if isinstance(spy_df.columns, pd.MultiIndex):
                    spy_df = spy_df.xs('SPY', level=0, axis=1) if 'SPY' in spy_df.columns.get_level_values(0) else spy_df
                spy_df = spy_df.dropna(subset=['Close'])
                regimes = self.rd.get_historical_regimes(spy_df)
                log.info("SPY regimes computed")
                return regimes
        except Exception as e:
            log.warning(f"SPY fetch failed: {e}")
        return None

    def fetch_spy_close(self) -> pd.Series | None:
        import yfinance as yf
        try:
            spy_df = yf.download('SPY', period='2y', interval='1d', auto_adjust=True, progress=False, multi_level_index=False)
            if spy_df is not None and not spy_df.empty:
                if isinstance(spy_df.columns, pd.MultiIndex):
                    spy_df = spy_df.xs('SPY', level=0, axis=1) if 'SPY' in spy_df.columns.get_level_values(0) else spy_df
                spy_df = spy_df.dropna(subset=['Close'])
                s = spy_df['Close'].copy()
                s.index = pd.to_datetime(s.index)
                if getattr(s.index, 'tz', None) is not None:
                    s.index = s.index.tz_localize(None)
                return s
        except Exception as e:
            log.warning(f"SPY close fetch failed: {e}")
        return None

    def train_ticker(self, ticker: str, historical_regimes: pd.Series = None,
                     options_data: dict = None, macro_features: dict = None,
                     auto_features: list[dict] = None,
                     spy_close: pd.Series = None,
                     short_data: dict = None,
                     sector_momentum: dict = None,
                     earnings_data: dict = None,
                     news_data: dict = None) -> dict | None:
        import yfinance as yf

        try:
            log.info(f"Processing {ticker}")
            df = yf.download(ticker, period='2y', interval='1d', group_by='ticker',
                             auto_adjust=True, progress=False, multi_level_index=False)
            if df is None or df.empty:
                log.warning(f"{ticker}: no data")
                return None
            if isinstance(df.columns, pd.MultiIndex):
                df = df.xs(ticker, level=0, axis=1) if ticker in df.columns.get_level_values(0) else df
            df = df.dropna(subset=['Close']).copy()

            df, feature_cols = self.fb.build_features(df, options_data, macro_features, auto_features, spy_close=spy_close, short_data=short_data, sector_momentum=sector_momentum, ticker=ticker, earnings_data=earnings_data)

            df_ml = df[feature_cols + ['target']].dropna()
            if len(df_ml) < 100:
                log.warning(f"{ticker}: only {len(df_ml)} samples")
                return None

            if historical_regimes is not None:
                df_ml = df_ml.copy()
                df_ml.index = pd.to_datetime(df_ml.index)
                spy_aligned = historical_regimes.reindex(df_ml.index, method='ffill')
                df_ml['regime'] = spy_aligned.values
            else:
                df_ml['regime'] = 'UNKNOWN'

            X = df_ml[feature_cols].values
            y = df_ml['target'].values
            regimes_arr = df_ml['regime'].values

            wf_results = self.wf.validate(X, y)
            wf_test_accs = [r['test_acc'] for r in wf_results]
            wf_briers = [r['test_brier'] for r in wf_results if r.get('test_brier') is not None]
            wf_mean = np.mean(wf_test_accs) if wf_test_accs else 0.5
            wf_std = np.std(wf_test_accs) if wf_test_accs else 0.0
            wf_brier = float(np.mean(wf_briers)) if wf_briers else None
            wf_brier_calib = wf_results[-1].get('brier_calib_global') if wf_results else None

            model, automl_winner, train_acc = self.automl.train_best_model(X, y)
            bayes_conf = self._compute_bayesian_confidence(X, y)

            version_info = self._update_model_version(ticker, automl_winner, train_acc, wf_mean, wf_std, len(feature_cols), len(X))
            dsr = self.rd.compute_deflated_sharpe(wf_test_accs, version_info['version'])

            fi = model.feature_importances_
            fi_idx = np.argsort(fi)[::-1]
            top3 = [feature_cols[i] for i in fi_idx[:3]]

            latest = df_ml[feature_cols].iloc[-1:].values
            prob_up_modelo = model.predict_proba(latest)[0][1] * 100 if len(latest) > 0 else 50.0

            news_score = 0.0
            news_titulo = ''
            if news_data:
                sdata = news_data.get(ticker, {}) or {}
                sent = sdata.get('sentimiento', {}) if isinstance(sdata, dict) else {}
                if isinstance(sent, dict):
                    try:
                        news_score = float(sent.get('score', 0) or 0)
                    except (ValueError, TypeError):
                        news_score = 0.0
                    news_titulo = str(sent.get('tema_principal', '') or '')[:120]
            prob_up = float(np.clip(prob_up_modelo + self.NEWS_WEIGHT * news_score, 1.0, 99.0))

            pred = 'alcista' if prob_up > 60 else ('bajista' if prob_up < 40 else 'neutral')

            result = {
                'fwd_vol_20d_annual': round(float(df['fwd_vol_20d'].iloc[-21]), 4) if pd.notna(df['fwd_vol_20d'].iloc[-21]) else None,
                'prob_up_20d': round(prob_up, 1),
                'prob_up_20d_modelo': round(float(prob_up_modelo), 1),
                'news_score': round(float(news_score), 3),
                'news_ajuste': round(float(self.NEWS_WEIGHT * news_score), 2),
                'news_tema': news_titulo,
                'features_top': top3,
                'prediccion': pred,
                'wf_accuracy': round(wf_mean, 4),
                'wf_brier': round(wf_brier, 4) if wf_brier is not None else None,
                'wf_brier_calib': round(wf_brier_calib, 4) if wf_brier_calib is not None else None,
                'wf_std': round(wf_std, 4),
                'train_acc': round(train_acc, 4),
                'bayes_confidence': round(bayes_conf, 1),
                'automl_winner': automl_winner,
                'dsr': round(dsr, 4),
                'model_version': version_info['version'],
                'X': X, 'y': y, 'regimes': regimes_arr,
                'feature_cols': feature_cols,
                'model': model,
                'wf_folds': wf_results
            }
            self.ticker_results[ticker] = result
            log.info(f"{ticker}: prob={prob_up:.0f}% pred={pred} wf={wf_mean:.2f}", ticker=ticker)
            return result

        except Exception as e:
            log.error(f"{ticker}: {e}", ticker=ticker)
            return None

    @staticmethod
    def _compute_bayesian_confidence(X: np.ndarray, y: np.ndarray) -> float:
        try:
            br = BayesianRidge()
            br.fit(X, y)
            _, y_std = br.predict(X, return_std=True)
            conf = 1.0 - np.mean(y_std) * 2
            return max(50, min(99, conf * 100))
        except Exception:
            return 50.0

    def _update_model_version(self, ticker: str, automl_winner: str, train_acc: float,
                               wf_mean: float, wf_std: float, n_features: int, n_samples: int) -> dict:
        versions = self._load_json('model_versions.json')
        if ticker not in versions:
            versions[ticker] = []

        entry = {
            'version': len(versions[ticker]) + 1,
            'timestamp': time.strftime('%Y-%m-%dT%H:%M:%S'),
            'automl_winner': automl_winner,
            'train_acc': round(train_acc, 4),
            'wf_accuracy': round(wf_mean, 4),
            'wf_std': round(wf_std, 4),
            'n_features': n_features,
            'n_samples': n_samples
        }
        versions[ticker].append(entry)
        versions[ticker] = versions[ticker][-20:]
        self._save_json('model_versions.json', versions)
        return entry

    def train_global_models(self, historical_regimes: pd.Series = None):
        all_data = [r['X'] for r in self.ticker_results.values() if 'X' in r]
        all_labels = [r['y'] for r in self.ticker_results.values() if 'y' in r]
        all_regimes = [r['regimes'] for r in self.ticker_results.values() if 'regimes' in r]

        feature_cols_list = [r['feature_cols'] for r in self.ticker_results.values() if 'feature_cols' in r]
        feature_cols = feature_cols_list[0] if feature_cols_list else []

        if not all_data:
            return

        X_all = np.vstack(all_data)
        y_all = np.concatenate(all_labels)
        regimes_all = np.concatenate(all_regimes)

        model_global = xgb.XGBClassifier(n_estimators=200, max_depth=5, learning_rate=0.05, random_state=42, verbosity=0, use_label_encoder=False)
        model_global.fit(X_all, y_all)
        for i, col in enumerate(feature_cols):
            self.global_feature_importance[col] = round(float(model_global.feature_importances_[i]), 4)
        self.global_feature_importance = dict(sorted(self.global_feature_importance.items(), key=lambda x: x[1], reverse=True))

        try:
            stacking_model = self.automl.build_stacking(X_all, y_all)
            if stacking_model:
                stacking_model.fit(X_all, y_all)
                log.info(f"Stacking trained: acc={stacking_model.score(X_all, y_all):.4f}")
        except Exception as e:
            log.warning(f"Stacking failed: {e}")

        for regime in ['ALCISTA', 'BAJISTA', 'LATERAL']:
            mask = regimes_all == regime
            if mask.sum() >= 50:
                model_reg = xgb.XGBClassifier(n_estimators=200, max_depth=5, learning_rate=0.05, random_state=42, verbosity=0, use_label_encoder=False)
                model_reg.fit(X_all[mask], y_all[mask])
                self.regime_models[regime] = model_reg
                fi = model_reg.feature_importances_
                self.regime_feature_importance[regime] = dict(sorted(
                    {feature_cols[i]: round(float(fi[i]), 4) for i in range(len(feature_cols))}.items(),
                    key=lambda x: x[1], reverse=True
                ))
                log.info(f"Regime-{regime}: {mask.sum()} samples")

        current_regime = self._load_regime().get('regimen', 'INCIERTO')
        if current_regime in self.regime_models:
            for idx, ticker in enumerate(self.TICKERS_CORE):
                if ticker in self.ticker_results and 'X' in self.ticker_results[ticker]:
                    latest = self.ticker_results[ticker]['X'][-1:].reshape(1, -1)
                    prob = self.regime_models[current_regime].predict_proba(latest)[0][1] * 100
                    self.regime_probs[ticker] = round(prob, 1)

    def run_all(self, tickers: list[str] = None, run_optuna: bool = False):
        tickers = tickers or self.TICKERS_CORE
        historical_regimes = self.fetch_spy_regimes()
        spy_close = self.fetch_spy_close()
        short_data = self._load_json('short_interest.json').get('tickers', {})
        sector_momentum = self._load_json('sector_momentum.json')
        earnings_data = self._load_json('google_finance.json').get('tickers', {})
        news_data = self._load_json('noticias_recientes.json').get('por_ticker', {})
        options_data = self._load_options()
        macro_features = self._load_macro()
        auto_features = self._load_auto_features_config()
        current_regime = self._load_regime().get('regimen', 'INCIERTO')
        predicciones = self._load_predicciones()
        calibracion = self._load_calibracion()

        log.info(f"Pipeline starting | regime={current_regime} | tickers={len(tickers)}")

        for ticker in tickers:
            self.train_ticker(
                ticker, historical_regimes=historical_regimes,
                options_data=options_data.get(ticker, {}),
                macro_features=macro_features,
                auto_features=auto_features,
                spy_close=spy_close,
                short_data=short_data,
                sector_momentum=sector_momentum,
                earnings_data=earnings_data,
                news_data=news_data,
            )

        all_data = [r['X'] for r in self.ticker_results.values() if 'X' in r]
        if run_optuna and HAS_OPTUNA and all_data:
            X_all = np.vstack(all_data)
            y_all = np.concatenate([r['y'] for r in self.ticker_results.values() if 'y' in r])
            self.automl.optuna_tune(X_all, y_all, n_trials=30, timeout=180)

        self.train_global_models(historical_regimes)

        self._build_output(current_regime, predicciones)
        self._save_json('modelo_xgboost.json', self.output)
        self._log_sentiment_history(predicciones)
        self._print_summary()
        return self.output

    def _log_sentiment_history(self, predicciones: dict):
        """Guarda un registro diario de sentimiento por ticker.

        Sirve para construir, con semanas de datos reales, una serie temporal
        de sentimiento usable como feature real del modelo (hoy no se usa como
        feature porque solo existe una foto por dia).
        """
        from datetime import datetime, timezone
        try:
            path = os.path.join(self.data_dir, 'Datos', 'sentimiento_historico.json')
            hist = {}
            if os.path.exists(path):
                with open(path, encoding='utf-8') as f:
                    hist = json.load(f)
            hoy = datetime.now(timezone.utc).strftime('%Y-%m-%d')
            dia = hist.setdefault(hoy, {})
            for tk, tr in self.ticker_results.items():
                if tk not in predicciones:
                    continue
                dia[tk] = {
                    'score': tr.get('news_score', 0.0),
                    'ajuste': tr.get('news_ajuste', 0.0),
                    'tema': tr.get('news_tema', ''),
                }
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(hist, f, ensure_ascii=False, indent=1)
        except Exception as e:
            log.warning(f"No se pudo guardar historial de sentimiento: {e}")

    def _build_output(self, current_regime: str, predicciones: dict):
        wf_accs = [v['wf_accuracy'] for v in self.ticker_results.values()]
        total_datos = sum(
            len(predicciones.get(t, {}).get('predicciones', []))
            for t in self.TICKERS_CORE if t in predicciones
        )

        self.output = {
            'timestamp': datetime.datetime.now().strftime('%Y-%m-%dT%H:%M:%SZ'),
            'wf_accuracy_mean': round(np.mean(wf_accs), 2) if wf_accs else 0,
            'wf_accuracy_std': round(np.std(wf_accs), 4) if wf_accs else 0,
            'train_accuracy': round(np.mean([v['train_acc'] for v in self.ticker_results.values()]), 2) if self.ticker_results else 0,
            'tickers': {},
            'feature_importance_global': self.global_feature_importance,
            'feature_importance_by_regime': self.regime_feature_importance,
            'regime_probabilities': self.regime_probs,
            'current_regime': current_regime,
            'total_datos': total_datos,
            'validation_method': 'walk_forward_expanding_regime_aware',
        }

        for tk, tr in self.ticker_results.items():
            self.output['tickers'][tk] = {
                'prob_up_20d': tr['prob_up_20d'],
                'prob_up_20d_modelo': tr.get('prob_up_20d_modelo'),
                'news_score': tr.get('news_score'),
                'news_ajuste': tr.get('news_ajuste'),
                'prob_up_20d_regime': self.regime_probs.get(tk, tr['prob_up_20d']),
                'fwd_vol_20d_annual': tr.get('fwd_vol_20d_annual'),
                'features_top': tr['features_top'],
                'prediccion': tr['prediccion'],
                'wf_accuracy': tr['wf_accuracy'],
                'wf_brier': tr.get('wf_brier'),
                'wf_brier_calib': tr.get('wf_brier_calib'),
                'wf_std': tr['wf_std'],
                'train_acc': tr['train_acc'],
            }

    def _print_summary(self):
        log.info("=" * 40)
        log.info(f"Tickers: {len(self.ticker_results)}/{len(self.TICKERS_CORE)}")
        wf_accs = [v['wf_accuracy'] for v in self.ticker_results.values()]
        if wf_accs:
            log.info(f"Walk-Forward: {np.mean(wf_accs):.2f}")
        log.info(f"Top features: {list(self.global_feature_importance.keys())[:5]}")
        for tk, tr in sorted(self.ticker_results.items()):
            log.info(f"  {tk}: {tr['prob_up_20d']}% {tr['prediccion']} wf={tr['wf_accuracy']:.2f}")


# Backward-compatible module-level access
_pipeline: MLPipeline | None = None


def get_pipeline() -> MLPipeline:
    global _pipeline
    if _pipeline is None:
        _pipeline = MLPipeline()
    return _pipeline


if __name__ == '__main__':
    pipeline = MLPipeline()
    pipeline.run_all()
