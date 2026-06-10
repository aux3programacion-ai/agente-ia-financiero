#!/usr/bin/env python3
"""
calibration.py - Calibración de probabilidades isotónica/Platt mediante
sklearn CalibratedClassifierCV. Reemplaza el factor heurístico actual.
Integra con model_store.py para persistencia.
"""
import json
import os
import time
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import brier_score_loss, log_loss

from config.settings import get_setting
from model_store import get_model_store

DATA_DIR = os.environ.get('GITHUB_WORKSPACE', '.')
OUTPUT_DIR = Path(DATA_DIR) / 'Datos'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CAL_CONFIG = get_setting('ml.calibracion', {})
METODO = CAL_CONFIG.get('metodo', 'isotonic')
MIN_MUESTRAS = CAL_CONFIG.get('min_muestras_calibracion', 50)
HOLDOUT_RATIO = CAL_CONFIG.get('holdout_ratio', 0.2)


class ProbabilityCalibrator:
    def __init__(self, metodo: str = METODO):
        """
        metodo: 'isotonic' (no paramétrico) o 'platt' (sigmoid/logistic)
        """
        self.metodo = metodo
        self.calibrator = None
        self.is_fitted = False
        self.brier_before = None
        self.brier_after = None
        self.calibration_error = None

    def fit(self, y_true: np.ndarray, y_prob: np.ndarray, X: Optional[np.ndarray] = None):
        """
        Entrena calibrador isotónico/Platt.
        
        Args:
            y_true: Labels reales (0/1)
            y_prob: Probabilidades sin calibrar
            X: Features originales (opcional, para Platt)
        """
        if len(y_true) < MIN_MUESTRAS:
            print(f'[Calibracion] Pocas muestras ({len(y_true)} < {MIN_MUESTRAS}), saltando')
            return self

        self.brier_before = float(brier_score_loss(y_true, y_prob))

        if self.metodo == 'isotonic':
            self.calibrator = IsotonicRegression(out_of_bounds='clip')
            self.calibrator.fit(y_prob, y_true)
        elif self.metodo == 'platt':
            from sklearn.linear_model import LogisticRegression
            if X is not None:
                # Platt scaling con features
                self.calibrator = LogisticRegression(C=1.0, class_weight='balanced')
                self.calibrator.fit(X, y_true)
            else:
                # Platt scaling solo con log-odds
                log_odds = np.clip(np.log(y_prob / (1 - y_prob + 1e-12)), -10, 10)
                self.calibrator = LogisticRegression(C=1.0)
                self.calibrator.fit(log_odds.reshape(-1, 1), y_true)
        else:
            raise ValueError(f'Método desconocido: {metodo}')

        self.is_fitted = True

        # Evaluar mejora
        y_calib = self.predict(y_prob, X)
        self.brier_after = float(brier_score_loss(y_true, y_calib))

        # Calibration error (ECE - Expected Calibration Error)
        prob_true, prob_pred = calibration_curve(y_true, y_calib, n_bins=10)
        self.calibration_error = float(np.mean(np.abs(prob_true - prob_pred)))

        return self

    def predict(self, y_prob: np.ndarray, X: Optional[np.ndarray] = None) -> np.ndarray:
        if not self.is_fitted or self.calibrator is None:
            return y_prob
        
        if self.metodo == 'isotonic':
            return self.calibrator.predict(y_prob)
        elif self.metodo == 'platt':
            if X is not None:
                return self.calibrator.predict_proba(X)[:, 1]
            else:
                log_odds = np.clip(np.log(y_prob / (1 - y_prob + 1e-12)), -10, 10)
                return self.calibrator.predict_proba(log_odds.reshape(-1, 1))[:, 1]

    def predict_single(self, prob: float, features: Optional[Dict] = None) -> float:
        """Calibra una probabilidad individual."""
        arr = np.array([prob])
        return float(self.predict(arr)[0])

    def get_summary(self) -> Dict:
        return {
            'metodo': self.metodo,
            'is_fitted': self.is_fitted,
            'brier_before': self.brier_before,
            'brier_after': self.brier_after,
            'brier_improvement': (self.brier_before - self.brier_after) if self.brier_before is not None else None,
            'calibration_error': self.calibration_error
        }


class CalibrationManager:
    def __init__(self):
        self.calibrators: Dict[str, ProbabilityCalibrator] = {}
        self.calib_path = OUTPUT_DIR / 'calibracion_real.json'
        self._load()

    def _load(self):
        if self.calib_path.exists():
            try:
                data = json.loads(self.calib_path.read_text())
                self.calibrators = {
                    k: self._deserialize(v) for k, v in data.get('calibrators', {}).items()
                }
            except:
                pass

    def _save(self):
        data = {
            'timestamp': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
            'calibrators': {
                k: {
                    'metodo': c.metodo,
                    'is_fitted': c.is_fitted,
                    'brier_before': c.brier_before,
                    'brier_after': c.brier_after,
                    'calibration_error': c.calibration_error
                }
                for k, c in self.calibrators.items()
            },
            'global_brier': self.brier_score()
        }
        self.calib_path.write_text(json.dumps(data, indent=2))

    def _deserialize(self, data: dict) -> ProbabilityCalibrator:
        c = ProbabilityCalibrator(metodo=data.get('metodo', 'isotonic'))
        # Reconstruimos el calibrador
        c.is_fitted = data.get('is_fitted', False)
        c.brier_before = data.get('brier_before')
        c.brier_after = data.get('brier_after')
        c.calibration_error = data.get('calibration_error')
        return c

    def get_or_create(self, key: str, metodo: str = METODO) -> ProbabilityCalibrator:
        if key not in self.calibrators:
            self.calibrators[key] = ProbabilityCalibrator(metodo=metodo)
        return self.calibrators[key]

    def calibrate(
        self,
        key: str,
        y_true: np.ndarray,
        y_prob: np.ndarray,
        X: Optional[np.ndarray] = None,
        save: bool = True
    ) -> ProbabilityCalibrator:
        cal = self.get_or_create(key)
        cal.fit(y_true, y_prob, X)
        if save:
            self._save()
        return cal

    def brier_score(self) -> float:
        scores = []
        for c in self.calibrators.values():
            if c.brier_after is not None:
                scores.append(c.brier_after)
        return float(np.mean(scores)) if scores else 0.0

    def calibrate_confidences(self, predictions: List[Dict]) -> List[Dict]:
        """
        Aplica calibración isotónica a todas las predicciones en la bitácora.
        Retorna predicciones calibradas.
        """
        calibrated = []
        for p in predictions:
            ticker = p.get('ticker', '')
            prob = p.get('probabilidad', 50) / 100.0
            cal = self.calibrators.get(ticker)
            if cal and cal.is_fitted:
                prob_cal = cal.predict_single(prob)
                p['probabilidad_calibrada'] = round(prob_cal * 100, 1)
                p['probabilidad_original'] = round(prob * 100, 1)
                p['calibracion_aplicada'] = True
                p['metodo_calibracion'] = cal.metodo
            calibrated.append(p)
        return calibrated


_calibration_manager = None


def get_calibration_manager() -> CalibrationManager:
    global _calibration_manager
    if _calibration_manager is None:
        _calibration_manager = CalibrationManager()
    return _calibration_manager


def calibrate_probabilities(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    metodo: str = METODO
) -> Tuple[ProbabilityCalibrator, np.ndarray]:
    cal = ProbabilityCalibrator(metodo=metodo)
    cal.fit(y_true, y_prob)
    y_calib = cal.predict(y_prob)
    return cal, y_calib


if __name__ == '__main__':
    # Test
    print('[Calibracion] Test con datos sintéticos...')
    np.random.seed(42)
    y_true = np.random.binomial(1, 0.3, 1000)
    y_prob_overconfident = np.clip(y_true + np.random.randn(1000) * 0.5, 0.05, 0.95)
    
    for metodo in ['isotonic', 'platt']:
        cal, y_cal = calibrate_probabilities(y_true, y_prob_overconfident, metodo=metodo)
        print(f'  {metodo}: Brier before={cal.brier_before:.4f}, after={cal.brier_after:.4f}')
    
    # Manager test
    cm = get_calibration_manager()
    cm.calibrate('NVDA', y_true[:500], y_prob_overconfident[:500])
    cm.calibrate('AAPL', y_true[500:], y_prob_overconfident[500:])
    print(f'  Global Brier: {cm.brier_score():.4f}')