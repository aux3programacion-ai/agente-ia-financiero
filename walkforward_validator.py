#!/usr/bin/env python3
"""
walkforward_validator.py - Walk-forward validation real para modelos ML.
Implementa expansion window con embargo (gap) entre train/test.
Genera métricas out-of-sample y detecta overfitting.
"""
import json
import os
import time
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple, Callable
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

from config.settings import get_setting
from model_store import get_model_store

DATA_DIR = os.environ.get('GITHUB_WORKSPACE', '.')
OUTPUT_DIR = Path(DATA_DIR) / 'Datos'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

WF_CONFIG = get_setting('ml.walk_forward', {})
N_SPLITS = WF_CONFIG.get('n_splits', 5)
TEST_SIZE_DIAS = WF_CONFIG.get('test_size_dias', 63)
GAP_DIAS = WF_CONFIG.get('gap_dias', 5)
MIN_TRAIN_SIZE = WF_CONFIG.get('min_train_size', 100)

XGB_PARAMS = get_setting('ml.xgboost', {})


@dataclass
class WFSplit:
    train_start: datetime
    train_end: datetime
    test_start: datetime
    test_end: datetime
    split_idx: int


def generate_walk_forward_splits(
    dates: pd.DatetimeIndex,
    n_splits: int = N_SPLITS,
    test_size_dias: int = TEST_SIZE_DIAS,
    gap_dias: int = GAP_DIAS,
    min_train_size: int = MIN_TRAIN_SIZE
) -> List[WFSplit]:
    """
    Genera splits expansion window con embargo (gap).
    
    Split 0: train=[0..T], test=[T+gap..T+gap+test_size]
    Split 1: train=[0..T+test_size+gap], test=[T+test_size+gap+gap..T+2*(test_size+gap)]
    """
    sorted_dates = sorted(dates)
    total = len(sorted_dates)
    step = test_size_dias + gap_dias
    
    min_start = min_train_size
    splits = []
    
    for i in range(n_splits):
        train_end_idx = min_start + i * step - 1
        if train_end_idx >= total - test_size_dias - gap_dias:
            break
        
        test_start_idx = train_end_idx + gap_dias + 1
        test_end_idx = test_start_idx + test_size_dias - 1
        
        if test_end_idx >= total:
            break
        
        splits.append(WFSplit(
            train_start=sorted_dates[0],
            train_end=sorted_dates[train_end_idx],
            test_start=sorted_dates[test_start_idx],
            test_end=sorted_dates[test_end_idx],
            split_idx=i
        ))
    
    return splits


def walk_forward_train_test(
    X: pd.DataFrame,
    y: pd.Series,
    model_fn: Callable,
    n_splits: int = N_SPLITS,
    test_size_dias: int = TEST_SIZE_DIAS,
    gap_dias: int = GAP_DIAS,
    min_train_size: int = MIN_TRAIN_SIZE,
    retrain_each_split: bool = True,
    model_params: Optional[Dict] = None,
    model_name: str = 'xgboost',
    regime: str = 'global'
) -> Dict[str, Any]:
    """
    Walk-forward validation completa.
    
    Args:
        X: Features (index=datetimes)
        y: Target (index=datetimes)
        model_fn: Callable que entrena modelo: model_fn(X_train, y_train, params) -> model
        n_splits: Número de splits
        test_size_dias: Días por ventana de test
        gap_dias: Embargo entre train/test
        retrain_each_split: Si True, re-entrena en cada split
        model_params: Parámetros del modelo
        model_name: Nombre para model store
        regime: Régimen de mercado
        
    Returns:
        Dict con resultados, métricas, predicciones OOS
    """
    if model_params is None:
        model_params = {}
    
    dates = X.index
    splits = generate_walk_forward_splits(dates, n_splits, test_size_dias, gap_dias, min_train_size)
    
    if not splits:
        return {'error': 'No se pudieron generar splits', 'splits': 0}
    
    results = []
    all_oos_predictions = []
    all_oos_targets = []
    models_saved = []
    store = get_model_store()
    
    for split in splits:
        mask_train = (dates >= split.train_start) & (dates <= split.train_end)
        mask_test = (dates >= split.test_start) & (dates <= split.test_end)
        
        X_train = X.loc[mask_train]
        y_train = y.loc[mask_train]
        X_test = X.loc[mask_test]
        y_test = y.loc[mask_test]
        
        if len(X_train) < min_train_size or len(X_test) < 10:
            continue
        
        model = model_fn(X_train, y_train, model_params)
        
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else y_pred
        
        from sklearn.metrics import (
            accuracy_score, precision_score, recall_score, f1_score,
            roc_auc_score, log_loss, brier_score_loss
        )
        
        y_test_bin = (y_test > 0).astype(int)
        y_pred_bin = (y_pred > 0).astype(int)
        
        metrics = {
            'split': split.split_idx,
            'train_start': split.train_start.isoformat(),
            'train_end': split.train_end.isoformat(),
            'test_start': split.test_start.isoformat(),
            'test_end': split.test_end.isoformat(),
            'n_train': len(X_train),
            'n_test': len(X_test),
            'accuracy': float(accuracy_score(y_test_bin, y_pred_bin)),
            'precision': float(precision_score(y_test_bin, y_pred_bin, zero_division=0)),
            'recall': float(recall_score(y_test_bin, y_pred_bin, zero_division=0)),
            'f1': float(f1_score(y_test_bin, y_pred_bin, zero_division=0)),
            'auc_roc': float(roc_auc_score(y_test_bin, y_proba)),
            'log_loss': float(log_loss(y_test_bin, y_proba)),
            'brier': float(brier_score_loss(y_test_bin, y_proba))
        }
        
        model_version = None
        if retrain_each_split:
            metrics_train = {
                'train_accuracy': float(accuracy_score(
                    (y_train > 0).astype(int),
                    (model.predict(X_train) > 0).astype(int)
                ))
            }
            metrics.update(metrics_train)
            
            model_version = store.save_model(
                model=model,
                name=model_name,
                regime=f'{regime}_split{split.split_idx}',
                params=model_params,
                metrics=metrics,
                feature_names=list(X.columns)
            )
            models_saved.append({
                'split': split.split_idx,
                'version': model_version,
                'test_accuracy': metrics['accuracy']
            })
        
        results.append(metrics)
        all_oos_predictions.extend(y_proba.tolist())
        all_oos_targets.extend(y_test_bin.tolist())
    
    if not results:
        return {'error': 'No splits con datos suficientes', 'splits': 0}
    
    # Métricas agregadas OOS
    from sklearn.metrics import accuracy_score, roc_auc_score, brier_score_loss
    oos_accuracy = float(accuracy_score(all_oos_targets, [1 if p > 0.5 else 0 for p in all_oos_predictions]))
    oos_auc = float(roc_auc_score(all_oos_targets, all_oos_predictions))
    oos_brier = float(brier_score_loss(all_oos_targets, all_oos_predictions))
    
    accuracies = [r['accuracy'] for r in results]
    
    summary = {
        'n_splits': len(results),
        'n_total_oos': len(all_oos_targets),
        'oos_accuracy': oos_accuracy,
        'oos_auc_roc': oos_auc,
        'oos_brier_score': oos_brier,
        'mean_split_accuracy': float(np.mean(accuracies)),
        'std_split_accuracy': float(np.std(accuracies)),
        'min_split_accuracy': float(min(accuracies)),
        'max_split_accuracy': float(max(accuracies)),
        'accuracy_stability': float(np.std(accuracies) / max(np.mean(accuracies), 0.01)),
        'overfitting_ratio': _detect_overfitting(results),
        'splits': results,
        'models_saved': models_saved,
        'timestamp': datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z'),
        'config': {
            'n_splits': n_splits,
            'test_size_dias': test_size_dias,
            'gap_dias': gap_dias,
            'min_train_size': min_train_size
        }
    }
    
    # Guardar resultados
    output_path = OUTPUT_DIR / f'walkforward_{model_name}_{regime}.json'
    with open(output_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f'[WalkForward] {model_name}/{regime}: {len(results)} splits, '
          f'OOS Acc={oos_accuracy:.3f}, AUC={oos_auc:.3f}, '
          f'Brier={oos_brier:.4f}')
    
    return summary


def _detect_overfitting(results: List[Dict]) -> float:
    """
    Detecta overfitting: diferencia entre train y test accuracy.
    >0.15 = overfitting severo, >0.08 = moderado.
    """
    ratios = []
    for r in results:
        train_acc = r.get('train_accuracy', 0)
        test_acc = r.get('accuracy', 0)
        if train_acc > 0:
            ratios.append((train_acc - test_acc) / train_acc)
    
    if not ratios:
        return 0.0
    return float(np.mean(ratios))


def cross_validate_model_predictive_power(
    features_df: pd.DataFrame,
    target_series: pd.Series,
    model_name: str = 'xgboost',
    regimes_series: Optional[pd.Series] = None,
    **kwargs
) -> Dict:
    """
    Walk-forward validation por régimen de mercado.
    Si regimes_series se provee, corre walk-forward separado por régimen.
    """
    if regimes_series is not None:
        all_results = {}
        common_idx = features_df.index.intersection(regimes_series.index)
        features_df = features_df.loc[common_idx]
        target_series = target_series.loc[common_idx]
        regimes_series = regimes_series.loc[common_idx]
        
        for regime in ['ALCISTA', 'BAJISTA', 'LATERAL', 'INCIERTO']:
            mask = regimes_series == regime
            if mask.sum() < 50:
                continue
            
            X_reg = features_df.loc[mask]
            y_reg = target_series.loc[mask]
            
            from xgboost import XGBClassifier
            def model_fn(X, y, params):
                return XGBClassifier(**{**XGB_PARAMS, **params}, verbosity=0).fit(X, y)
            
            result = walk_forward_train_test(
                X=X_reg, y=y_reg, model_fn=model_fn,
                model_name=model_name, regime=regime, **kwargs
            )
            all_results[regime] = result
        
        return all_results
    else:
        from xgboost import XGBClassifier
        def model_fn(X, y, params):
            return XGBClassifier(**{**XGB_PARAMS, **params}, verbosity=0).fit(X, y)
        
        return walk_forward_train_test(
            X=features_df, y=target_series, model_fn=model_fn,
            model_name=model_name, **kwargs
        )


def load_walkforward_results(
    model_name: str = 'xgboost',
    regime: str = 'global'
) -> Optional[Dict]:
    path = OUTPUT_DIR / f'walkforward_{model_name}_{regime}.json'
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return None


if __name__ == '__main__':
    print('[WalkForward] Test con datos simulados...')
    np.random.seed(42)
    dates = pd.date_range('2022-01-01', '2024-12-31', freq='B')
    X = pd.DataFrame({
        'feature1': np.random.randn(len(dates)),
        'feature2': np.random.randn(len(dates)),
        'feature3': np.random.randn(len(dates)),
    }, index=dates)
    y = pd.Series(
        (X['feature1'] * 0.3 + X['feature2'] * 0.2 + np.random.randn(len(dates)) * 0.1 > 0).astype(float),
        index=dates
    )
    
    from xgboost import XGBClassifier
    def model_fn(X, y, params):
        return XGBClassifier(**{**XGB_PARAMS, **params}, n_estimators=50, verbosity=0).fit(X, y)
    
    result = walk_forward_train_test(X, y, model_fn, n_splits=3, test_size_dias=42)
    print(f'OOS Accuracy: {result["oos_accuracy"]:.3f}')
    print(f'OOS AUC: {result["oos_auc_roc"]:.3f}')
    print(f'Overfitting ratio: {result["overfitting_ratio"]:.3f}')