#!/usr/bin/env python3
"""
regime_models.py - Entrena modelos XGBoost separados por régimen de mercado.
Cada régimen (ALCISTA, BAJISTA, LATERAL) tiene su propio modelo especializado.
"""
import json
import os
import time
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

from config.settings import get_setting
from model_store import get_model_store
from walkforward_validator import walk_forward_train_test, cross_validate_model_predictive_power

DATA_DIR = os.environ.get('GITHUB_WORKSPACE', '.')
OUTPUT_DIR = Path(DATA_DIR) / 'Datos'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

REGIME_CONFIG = get_setting('ml.regime_models', {})
MIN_MUESTRAS = REGIME_CONFIG.get('min_muestras_por_regimen', 50)
REGIMENES = REGIME_CONFIG.get('regimenes', ['ALCISTA', 'BAJISTA', 'LATERAL'])
XGB_PARAMS = get_setting('ml.xgboost', {})

TICKERS_CORE = get_setting('tickers.core', [])


class RegimeModelManager:
    def __init__(self):
        self.models: Dict[str, Any] = {}
        self.results: Dict[str, Dict] = {}
        self.store = get_model_store()

    def prepare_regime_data(
        self,
        features_df: pd.DataFrame,
        target_series: pd.Series,
        regimes_series: pd.Series
    ) -> Dict[str, Tuple[pd.DataFrame, pd.Series]]:
        """Prepara datos separados por régimen."""
        regime_data = {}
        for regime in REGIMENES:
            mask = regimes_series == regime
            count = mask.sum()
            if count >= MIN_MUESTRAS:
                regime_data[regime] = (
                    features_df.loc[mask],
                    target_series.loc[mask]
                )
                print(f'  [RegimeModels] {regime}: {count} muestras')
        return regime_data

    def train_regime_models(
        self,
        features_df: pd.DataFrame,
        target_series: pd.Series,
        regimes_series: pd.Series,
        walk_forward: bool = True,
        **wf_kwargs
    ) -> Dict[str, Dict]:
        """
        Entrena modelo XGBoost por cada régimen con suficiente data.
        
        Returns:
            Dict {regimen: {metrics, version, walkforward}}
        """
        from xgboost import XGBClassifier
        
        regime_data = self.prepare_regime_data(features_df, target_series, regimes_series)
        
        for regime, (X_reg, y_reg) in regime_data.items():
            print(f'[RegimeModels] Entrenando modelo {regime}...')
            
            if walk_forward:
                def model_fn(X, y, params):
                    return XGBClassifier(**{**XGB_PARAMS, **params}, verbosity=0).fit(X, y)
                
                wf_result = walk_forward_train_test(
                    X=X_reg, y=y_reg, model_fn=model_fn,
                    model_name=f'xgb_regime_{regime}',
                    regime=regime,
                    **wf_kwargs
                )
                self.results[regime] = wf_result
                print(f'  {regime}: OOS Acc={wf_result["oos_accuracy"]:.3f}, AUC={wf_result["oos_auc_roc"]:.3f}')
            else:
                model = XGBClassifier(**XGB_PARAMS, verbosity=0)
                model.fit(X_reg, y_reg)
                
                from sklearn.metrics import accuracy_score, roc_auc_score
                y_pred = model.predict(X_reg)
                y_proba = model.predict_proba(X_reg)[:, 1]
                acc = accuracy_score(y_reg, y_proba > 0.5)
                auc = roc_auc_score(y_reg, y_proba)
                
                wf_result = self.store.save_model(
                    model=model,
                    name=f'xgb_regime_{regime}',
                    regime=regime,
                    params=XGB_PARAMS,
                    metrics={'accuracy': float(acc), 'auc_roc': float(auc), 'n_samples': len(X_reg)},
                    feature_names=list(X_reg.columns)
                )
                self.results[regime] = wf_result
        
        return self.results

    def predict_regime(
        self,
        features: np.ndarray,
        regime: str,
        use_probability: bool = True
    ) -> np.ndarray:
        """Predice usando el modelo del régimen específico."""
        if regime not in self.models:
            try:
                loaded = self.store.load_latest(f'xgb_regime_{regime}', regime)
                self.models[regime] = loaded['model']
            except:
                return np.full(len(features) if hasattr(features, '__len__') else 1, 0.5)
        
        model = self.models[regime]
        if use_probability and hasattr(model, 'predict_proba'):
            return model.predict_proba(features)[:, 1]
        return model.predict(features)

    def ensemble_regime_predictions(
        self,
        features: np.ndarray,
        regimes_probs: Dict[str, float],
        fallback_model: Any = None
    ) -> np.ndarray:
        """
        Ensemble ponderado: cada régimen contribuye según su probabilidad.
        
        Args:
            features: Features de entrada
            regimes_probs: Dict {regimen: probabilidad} del HMM/clasificador
            fallback_model: Modelo global si no hay régimen específico
        
        Returns:
            Predicción ponderada
        """
        predictions = []
        weights = []
        
        for regime, prob in regimes_probs.items():
            try:
                pred = self.predict_regime(features, regime)
                predictions.append(pred)
                weights.append(prob)
            except:
                continue
        
        if not predictions:
            if fallback_model is not None:
                return fallback_model.predict_proba(features)[:, 1] if hasattr(fallback_model, 'predict_proba') else fallback_model.predict(features)
            return np.full(len(features) if hasattr(features, '__len__') else 1, 0.5)
        
        weights = np.array(weights) / sum(weights)
        return np.average(np.array(predictions), axis=0, weights=weights)

    def get_regime_performance(self) -> Dict:
        """Retorna performance de cada modelo por régimen."""
        perf = {}
        for regime in REGIMENES:
            try:
                loaded = self.store.load_latest(f'xgb_regime_{regime}', regime)
                perf[regime] = loaded['metadata']['metrics']
            except:
                perf[regime] = {'error': 'no model'}
        return perf


def train_regime_models_from_history(
    predictions_file: str = 'aprendizaje.json',
    features_file: str = 'auto_features.json',
    regime_file: str = 'regimen_mercado.json'
) -> Dict:
    """
    Entrena modelos por régimen desde archivos históricos.
    """
    pred_path = OUTPUT_DIR / predictions_file
    feat_path = OUTPUT_DIR / features_file
    regime_path = OUTPUT_DIR / regime_file
    
    if not all(p.exists() for p in [pred_path, feat_path, regime_path]):
        return {'error': 'Faltan archivos de datos'}
    
    with open(pred_path) as f:
        pred_data = json.load(f)
    with open(feat_path) as f:
        feat_data = json.load(f)
    with open(regime_path) as f:
        regime_data = json.load(f)
    
    predictions = pred_data.get('predicciones', [])
    features = feat_data.get('tickers', {})
    regimes = regime_data.get('historial', {})
    
    # Construir DataFrame
    records = []
    for p in predictions:
        ticker = p.get('ticker', '')
        fecha = p.get('fecha', '')[:10]
        if ticker in features and fecha in regimes:
            feat = features[ticker]
            record = {'ticker': ticker, 'fecha': fecha, 'target': 1 if p.get('acierto') else 0, 'regimen': regimes[fecha]}
            for k, v in feat.items():
                if isinstance(v, (int, float)):
                    record[k] = v
            records.append(record)
    
    if len(records) < 100:
        return {'error': f'Pocos registros: {len(records)}'}
    
    df = pd.DataFrame(records)
    feature_cols = [c for c in df.columns if c not in ('ticker', 'fecha', 'target', 'regimen')]
    
    manager = RegimeModelManager()
    results = manager.train_regime_models(
        features_df=df[feature_cols],
        target_series=df['target'],
        regimes_series=df['regimen'],
        walk_forward=True,
        n_splits=3,
        test_size_dias=42
    )
    
    return results


if __name__ == '__main__':
    print('[RegimeModels] Test...')
    manager = RegimeModelManager()
    perf = manager.get_regime_performance()
    print(f'Modelos disponibles: {list(perf.keys())}')
    
    if not perf:
        print('  No hay modelos entrenados. Entrenando con datos sintéticos...')
        np.random.seed(42)
        n = 500
        X = pd.DataFrame({
            'feature1': np.random.randn(n),
            'feature2': np.random.randn(n),
            'feature3': np.random.randn(n),
        }, index=pd.date_range('2023-01-01', periods=n, freq='B'))
        y = pd.Series(
            (X['feature1'] * 0.3 + X['feature2'] * 0.2 + np.random.randn(n) * 0.1 > 0).astype(float),
            index=X.index
        )
        regimes = pd.Series(
            np.random.choice(['ALCISTA', 'BAJISTA', 'LATERAL'], n),
            index=X.index
        )
        
        results = manager.train_regime_models(X, y, regimes, walk_forward=True, n_splits=2, test_size_dias=21)
        print(json.dumps({k: {'oos_accuracy': v.get('oos_accuracy')} for k, v in results.items()}, indent=2))