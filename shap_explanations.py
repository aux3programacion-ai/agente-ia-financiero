#!/usr/bin/env python3
"""
shap_explanations.py - SHAP explanations para cada predicción.
Explicabilidad regulatoria, debug de features, identificación de sesgos.
"""
import json
import os
import time
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

from config.settings import get_setting

DATA_DIR = os.environ.get('GITHUB_WORKSPACE', '.')
OUTPUT_DIR = Path(DATA_DIR) / 'Datos'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

EXP_CONFIG = get_setting('explicabilidad', {})
SHAP_ENABLED = EXP_CONFIG.get('shap', {}).get('enabled', True)
MAX_MUESTRAS = EXP_CONFIG.get('shap', {}).get('max_muestras', 500)
PLOT_TOP_N = EXP_CONFIG.get('shap', {}).get('plot_top_n', 10)


class ShapExplainer:
    def __init__(self, model, feature_names: List[str], background_data: Optional[np.ndarray] = None):
        self.model = model
        self.feature_names = feature_names
        self.background_data = background_data
        self.explainer = None
        self.shap_values = None
        self.base_value = None

    def _build_explainer(self, X: np.ndarray):
        if not SHAP_AVAILABLE:
            return
        
        try:
            if hasattr(self.model, 'get_booster'):
                # XGBoost
                self.explainer = shap.TreeExplainer(self.model)
            elif hasattr(self.model, 'coef_'):
                # Linear models
                if self.background_data is not None:
                    self.explainer = shap.LinearExplainer(self.model, self.background_data)
                else:
                    self.explainer = shap.LinearExplainer(self.model, X[:100])
            elif hasattr(self.model, 'predict'):
                # Generic
                if self.background_data is not None:
                    self.explainer = shap.KernelExplainer(self.model.predict, self.background_data)
                else:
                    self.explainer = shap.KernelExplainer(self.model.predict, X[:50])
            else:
                self.explainer = None
        except Exception as e:
            print(f'[Shap] Error creating explainer: {e}')
            self.explainer = None

    def explain(self, X: np.ndarray, sample: bool = True) -> Dict[str, Any]:
        """
        Calcula SHAP values para las predicciones.
        
        Args:
            X: Features matrix
            sample: Si True, usa subset de datos para performance
            
        Returns:
            Dict con shap_values, base_value, feature_importance
        """
        if not SHAP_AVAILABLE or not SHAP_ENABLED:
            return {'error': 'SHAP no disponible'}
        
        if sample and len(X) > MAX_MUESTRAS:
            idx = np.random.choice(len(X), MAX_MUESTRAS, replace=False)
            X = X[idx]
        
        self._build_explainer(X)
        if self.explainer is None:
            return {'error': 'No se pudo crear explainer'}
        
        try:
            shap_values = self.explainer.shap_values(X)
            
            if isinstance(shap_values, list):
                shap_values = shap_values[1] if len(shap_values) > 1 else shap_values[0]
            
            self.shap_values = shap_values
            self.base_value = self.explainer.expected_value
            
            if isinstance(self.base_value, list):
                self.base_value = self.base_value[1] if len(self.base_value) > 1 else self.base_value[0]
            
            feature_importance = self._compute_feature_importance(shap_values, X)
            
            return {
                'shap_values': shap_values.tolist() if hasattr(shap_values, 'tolist') else shap_values,
                'base_value': float(self.base_value),
                'feature_importance': feature_importance,
                'top_features': feature_importance[:PLOT_TOP_N],
                'n_features': len(self.feature_names),
                'n_explanations': len(X)
            }
        except Exception as e:
            return {'error': str(e)}

    def _compute_feature_importance(self, shap_values: np.ndarray, X: np.ndarray) -> List[Dict]:
        """Feature importance basada en |SHAP| mean."""
        mean_abs_shap = np.mean(np.abs(shap_values), axis=0)
        total = mean_abs_shap.sum()
        
        importance = []
        for i, name in enumerate(self.feature_names):
            if i < len(mean_abs_shap):
                importance.append({
                    'feature': name,
                    'importance': float(mean_abs_shap[i]),
                    'importance_pct': float(mean_abs_shap[i] / total) if total > 0 else 0,
                    'direction': 'positive' if np.mean(shap_values[:, i]) > 0 else 'negative'
                })
        
        importance.sort(key=lambda x: x['importance'], reverse=True)
        return importance

    def explain_single(self, x: np.ndarray) -> Dict[str, float]:
        """Explica una predicción individual."""
        if self.explainer is None:
            self._build_explainer(x.reshape(1, -1))
        
        try:
            shap_val = self.explainer.shape_values(x.reshape(1, -1))
            if isinstance(shap_val, list):
                shap_val = shap_val[1] if len(shap_val) > 1 else shap_val[0]
            
            return {
                'base_value': float(self.base_value),
                'prediction': float(self.base_value + shap_val.sum()),
                'feature_contributions': {
                    name: float(shap_val[0][i])
                    for i, name in enumerate(self.feature_names)
                    if i < len(shap_val[0])
                }
            }
        except Exception as e:
            return {'error': str(e)}

    def get_dependence_plot_data(self, feature_idx: int, X: np.ndarray) -> Dict:
        """Datos para SHAP dependence plot."""
        if self.shap_values is None:
            return {}
        
        return {
            'feature_values': X[:, feature_idx].tolist(),
            'shap_values': self.shap_values[:, feature_idx].tolist(),
            'feature_name': self.feature_names[feature_idx] if feature_idx < len(self.feature_names) else f'feature_{feature_idx}'
        }

    def get_summary_plot_data(self, top_n: int = 10) -> Dict:
        """Datos para SHAP summary plot."""
        if self.shap_values is None:
            return {}
        
        importance = self._compute_feature_importance(self.shap_values, None)
        top = importance[:top_n]
        
        return {
            'base_value': float(self.base_value),
            'top_features': top,
            'shap_range': {
                'min': float(self.shap_values.min()),
                'max': float(self.shap_values.max())
            }
        }

    def save_explanations(self, X: np.ndarray, output_name: str = 'shap_explanations'):
        """Guarda SHAP explanations en JSON."""
        result = self.explain(X)
        
        output_path = OUTPUT_DIR / f'{output_name}.json'
        with open(output_path, 'w') as f:
            json.dump(result, f, indent=2)
        
        print(f'[Shap] Explanations guardadas en {output_path}')
        return result


def explain_model_predictions(
    model,
    feature_names: List[str],
    X: np.ndarray,
    output_name: str = 'shap_explanations'
) -> Dict:
    """Función conveniente para explicar modelo completo."""
    explainer = ShapExplainer(model, feature_names)
    return explainer.save_explanations(X, output_name)


def get_top_features(model, feature_names: List[str], X: np.ndarray, n: int = 10) -> List[str]:
    """Retorna top N features por SHAP importance."""
    explainer = ShapExplainer(model, feature_names)
    result = explainer.explain(X)
    
    if 'feature_importance' in result:
        return [f['feature'] for f in result['feature_importance'][:n]]
    return feature_names[:n]


def detect_feature_drift(
    X_production: np.ndarray,
    X_reference: np.ndarray,
    feature_names: List[str]
) -> Dict[str, float]:
    """
    Detecta drift en distribución de features usando SHAP.
    Retorna PSI (Population Stability Index) por feature.
    """
    drift_scores = {}
    
    for i, name in enumerate(feature_names):
        if i >= X_production.shape[1] or i >= X_reference.shape[1]:
            continue
        
        prod = X_production[:, i]
        ref = X_reference[:, i]
        
        # PSI calculation
        bins = np.linspace(min(prod.min(), ref.min()), max(prod.max(), ref.max()), 11)
        
        prod_bins = np.histogram(prod, bins=bins, density=True)[0] + 1e-10
        ref_bins = np.histogram(ref, bins=bins, density=True)[0] + 1e-10
        
        prod_pct = prod_bins / prod_bins.sum()
        ref_pct = ref_bins / ref_bins.sum()
        
        psi = np.sum((prod_pct - ref_pct) * np.log(prod_pct / ref_pct))
        drift_scores[name] = float(psi)
    
    return drift_scores


if __name__ == '__main__':
    print(f'[Shap] SHAP available: {SHAP_AVAILABLE}')
    
    if SHAP_AVAILABLE:
        from xgboost import XGBClassifier
        
        np.random.seed(42)
        n = 500
        X = np.random.randn(n, 5)
        y = ((X[:, 0] * 0.3 + X[:, 1] * 0.2 + np.random.randn(n) * 0.05) > 0).astype(int)
        feature_names = ['feature_a', 'feature_b', 'feature_c', 'feature_d', 'feature_e']
        
        model = XGBClassifier(n_estimators=50, max_depth=3, verbosity=0)
        model.fit(X, y)
        
        result = explain_model_predictions(model, feature_names, X)
        if 'feature_importance' in result:
            print('Top features:')
            for f in result['feature_importance'][:5]:
                print(f'  {f["feature"]}: {f["importance_pct"]:.1%}')
    else:
        print('[Shap] Instalar: pip install shap')