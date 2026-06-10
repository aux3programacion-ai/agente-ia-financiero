#!/usr/bin/env python3
"""
online_learning.py - Online/incremental learning con River.
SGD con elasticnet penalty + ADWIN drift detection + rolling accuracy.
Reemplaza SGDClassifier estático con River para aprendizaje continuo.
"""
import json
import os
import time
import numpy as np
from collections import defaultdict, deque
from datetime import datetime, timezone
from pathlib import Path

try:
    from river import linear_model, optim, preprocessing, drift, metrics, ensemble
    RIVER_AVAILABLE = True
except ImportError:
    RIVER_AVAILABLE = False

from config.settings import get_setting

DATA_DIR = os.environ.get('GITHUB_WORKSPACE', '.')
OUTPUT_DIR = Path(DATA_DIR) / 'Datos'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OL_CONFIG = get_setting('online_learning', {})
ENABLED = OL_CONFIG.get('enabled', True)
ALGORITHM = OL_CONFIG.get('algoritmo', 'SGDClassifier')
LOSS = OL_CONFIG.get('loss', 'log_loss')
PENALTY = OL_CONFIG.get('penalty', 'elasticnet')
ALPHA = OL_CONFIG.get('alpha', 0.0001)
LEARNING_RATE = OL_CONFIG.get('learning_rate', 'adaptive')
ETA0 = OL_CONFIG.get('eta0', 0.01)
MAX_TICKERS = OL_CONFIG.get('max_tickers', 5)
ROLLING_WINDOW = OL_CONFIG.get('ventana_rolling_accuracy', 30)

DRIFT_ENABLED = OL_CONFIG.get('drift_detection', {}).get('enabled', False)
DRIFT_DELTA = OL_CONFIG.get('drift_detection', {}).get('delta', 0.001)

FEATURES_ORDER = [
    'rsi_14', 'macd_hist', 'vol_ratio', 'volatility_20d',
    'sma50_dist_pct', 'sma200_dist_pct', 'atr_pct',
    'ret_vol_interaction', 'ret_vol_corr_20d', 'price_vol_corr',
    'ret_skew_20d', 'ret_kurt_20d', 'ret_zscore_20d'
]


class OnlineLearner:
    def __init__(self, ticker: str):
        self.ticker = ticker
        self.model = None
        self.scaler = preprocessing.StandardScaler()
        self.drift_detector = drift.ADWIN(delta=DRIFT_DELTA) if DRIFT_ENABLED and RIVER_AVAILABLE else None
        self.rolling_accuracy = deque(maxlen=ROLLING_WINDOW)
        self.n_updates = 0
        self.n_correct = 0
        self._init_model()

    def _init_model(self):
        if not RIVER_AVAILABLE or not ENABLED:
            self.model = None
            return
        
        try:
            if ALGORITHM == 'SGDClassifier':
                lr_scheduler = None
                if LEARNING_RATE == 'adaptive':
                    try:
                        lr_scheduler = optim.schedulers.Adaptive(ETA0)
                    except AttributeError:
                        try:
                            lr_scheduler = optim.schedulers.Optimal(ETA0)
                        except AttributeError:
                            lr_scheduler = optim.schedulers.Constant(ETA0)
                else:
                    lr_scheduler = optim.schedulers.Constant(ETA0)
                
                self.model = linear_model.LogisticRegression(
                    optimizer=optim.SGD(lr=lr_scheduler),
                    loss=LOSS,
                    l1=0.5 if PENALTY == 'elasticnet' else 0,
                    l2=0.5 if PENALTY == 'elasticnet' else 1
                )
            elif ALGORITHM == 'HedgeClassifier':
                self.model = ensemble.HedgeClassifier([
                    linear_model.LogisticRegression(),
                    linear_model.PAClassifier(),
                    linear_model.ALMAClassifier()
                ])
            else:
                self.model = linear_model.LogisticRegression()
        except Exception:
            self.model = None

    def partial_fit(self, features: dict, target: int):
        if self.model is None:
            return
        
        x = {k: features.get(k, 0.0) for k in FEATURES_ORDER}
        x_scaled = self.scaler.learn_one(x).transform_one(x)
        
        y_pred = self.model.predict_one(x_scaled)
        y_proba = self.model.predict_proba_one(x_scaled)
        
        self.model.learn_one(x_scaled, target)
        
        correct = y_pred == target
        self.rolling_accuracy.append(1 if correct else 0)
        if correct:
            self.n_correct += 1
        self.n_updates += 1
        
        if self.drift_detector is not None:
            self.drift_detector.update(0 if correct else 1)
        
        return {
            'prediction': y_pred,
            'probability': y_proba.get(1, 0.5),
            'correct': correct,
            'rolling_accuracy': self.accuracy,
            'drift_detected': self.drift_detected
        }

    @property
    def accuracy(self) -> float:
        if not self.rolling_accuracy:
            return 0.5
        return sum(self.rolling_accuracy) / len(self.rolling_accuracy)

    @property
    def drift_detected(self) -> bool:
        if self.drift_detector is None:
            return False
        return self.drift_detector.drift_detected

    def get_state(self) -> dict:
        return {
            'ticker': self.ticker,
            'n_updates': self.n_updates,
            'n_correct': self.n_correct,
            'rolling_accuracy': self.accuracy,
            'drift_detected': self.drift_detected,
            'window_size': len(self.rolling_accuracy)
        }

    def should_retrain(self, threshold: float = 0.45) -> bool:
        return self.accuracy < threshold and self.n_updates > ROLLING_WINDOW


class OnlineLearningManager:
    def __init__(self):
        self.learners: dict[str, OnlineLearner] = {}
        self.perf_path = OUTPUT_DIR / 'online_learning_perf.json'
        self._load_perf()

    def _load_perf(self):
        if self.perf_path.exists():
            try:
                self.performance = json.loads(self.perf_path.read_text())
            except:
                self.performance = {'history': [], 'drift_events': []}
        else:
            self.performance = {'history': [], 'drift_events': []}

    def _save_perf(self):
        self.performance['history'] = self.performance['history'][-500:]
        self.performance['drift_events'] = self.performance['drift_events'][-100:]
        self.perf_path.write_text(json.dumps(self.performance, indent=2))

    def get_or_create(self, ticker: str) -> OnlineLearner:
        if ticker not in self.learners:
            if len(self.learners) >= MAX_TICKERS:
                # Reemplazar el peor performer
                worst = min(self.learners.items(), key=lambda x: x[1].accuracy)
                del self.learners[worst[0]]
                print(f'[Online] Reemplazado {worst[0]} (acc={worst[1].accuracy:.2f}) por {ticker}')
            self.learners[ticker] = OnlineLearner(ticker)
        return self.learners[ticker]

    def update(self, ticker: str, features: dict, target: int) -> dict:
        learner = self.get_or_create(ticker)
        result = learner.partial_fit(features, target)
        
        if result and result['drift_detected']:
            event = {
                'timestamp': datetime.now(timezone.utc).isoformat() + 'Z',
                'ticker': ticker,
                'accuracy_before': result['rolling_accuracy'],
                'n_updates': learner.n_updates
            }
            self.performance['drift_events'].append(event)
            print(f'[Online] Drift detectado en {ticker} (acc={result["rolling_accuracy"]:.2f})')
            # Reset learner after drift
            self.learners[ticker] = OnlineLearner(ticker)
        
        self.performance['history'].append({
            'timestamp': datetime.now(timezone.utc).isoformat() + 'Z',
            'ticker': ticker,
            'accuracy': learner.accuracy,
            'n_updates': learner.n_updates
        })
        self._save_perf()
        
        return result

    def predict_one(self, ticker: str, features: dict) -> dict:
        if ticker not in self.learners:
            return {'prediction': None, 'probability': 0.5, 'rolling_accuracy': 0.5}
        
        learner = self.learners[ticker]
        if learner.model is None:
            return {'prediction': None, 'probability': 0.5, 'rolling_accuracy': learner.accuracy}
        
        x = {k: features.get(k, 0.0) for k in FEATURES_ORDER}
        x_scaled = learner.scaler.transform_one(x)
        
        return {
            'prediction': learner.model.predict_one(x_scaled),
            'probability': learner.model.predict_proba_one(x_scaled).get(1, 0.5),
            'rolling_accuracy': learner.accuracy,
            'drift_detected': learner.drift_detected
        }

    def summary(self) -> dict:
        active = sum(1 for l in self.learners.values() if l.n_updates > 0)
        avg_acc = np.mean([l.accuracy for l in self.learners.values()]) if self.learners else 0
        return {
            'enabled': ENABLED and RIVER_AVAILABLE,
            'river_available': RIVER_AVAILABLE,
            'n_learners': len(self.learners),
            'active_learners': active,
            'avg_rolling_accuracy': round(float(avg_acc), 4),
            'drift_events_total': len(self.performance.get('drift_events', [])),
            'total_updates': sum(l.n_updates for l in self.learners.values()),
            'learners': {t: l.get_state() for t, l in self.learners.items()}
        }


_manager_instance = None

def get_online_learning_manager() -> OnlineLearningManager:
    global _manager_instance
    if _manager_instance is None:
        _manager_instance = OnlineLearningManager()
    return _manager_instance


def update_online(ticker: str, features: dict, target: int) -> dict:
    return get_online_learning_manager().update(ticker, features, target)


def predict_online(ticker: str, features: dict) -> dict:
    return get_online_learning_manager().predict_one(ticker, features)


if __name__ == '__main__':
    print(f'[OnlineLearning] River available: {RIVER_AVAILABLE}')
    print(f'[OnlineLearning] Config: algorithm={ALGORITHM}, drift={DRIFT_ENABLED}')
    
    if RIVER_AVAILABLE:
        olm = get_online_learning_manager()
        dummy_features = {k: np.random.randn() for k in FEATURES_ORDER}
        for i in range(20):
            olm.update('NVDA', dummy_features, 1 if np.random.rand() > 0.4 else 0)
        print(json.dumps(olm.summary(), indent=2))
    else:
        print('[OnlineLearning] Instalar River: pip install river')