#!/usr/bin/env python3
"""
hyperparameter_optimizer.py - AutoML con Optuna.
Optimizacion automatica de hiperparametros con pruning,
early stopping, integracion con model_store y walk_forward.
"""
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Callable, Tuple
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
import json, os, warnings, inspect
from pathlib import Path

try:
    import optuna
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False

from config.settings import get_setting
try:
    from model_store import ModelStore
except ImportError:
    ModelStore = None

try:
    from persistent_db import db
except ImportError:
    db = None

DATA_DIR = os.environ.get('GITHUB_WORKSPACE', '.')
OUTPUT_DIR = Path(DATA_DIR) / 'Datos' / 'optuna'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


@dataclass
class TrialResult:
    trial_id: int; params: Dict; metric: float; duration: float
    model_path: Optional[str] = None; pruning: bool = False


@dataclass
class StudySummary:
    study_name: str; n_trials: int; best_params: Dict
    best_metric: float; improvement: float; duration: float
    trials: List[TrialResult] = field(default_factory=list)


def sugerir_params(trial, search_space: Dict) -> Dict:
    params = {}
    for nombre, config in search_space.items():
        tipo = config.get('tipo', 'float')
        if tipo == 'float':
            if 'log' in config and config['log']:
                params[nombre] = trial.suggest_float(
                    nombre, config['min'], config['max'], log=True)
            else:
                params[nombre] = trial.suggest_float(
                    nombre, config['min'], config['max'])
        elif tipo == 'int':
            params[nombre] = trial.suggest_int(
                nombre, config['min'], config['max'],
                log=config.get('log', False))
        elif tipo == 'categorical':
            params[nombre] = trial.suggest_categorical(
                nombre, config['choices'])
    return params


ESPACIOS_XGBOOST = {
    'n_estimators': {'tipo': 'int', 'min': 50, 'max': 500},
    'max_depth': {'tipo': 'int', 'min': 3, 'max': 12},
    'learning_rate': {'tipo': 'float', 'min': 0.01, 'max': 0.3, 'log': True},
    'subsample': {'tipo': 'float', 'min': 0.5, 'max': 1.0},
    'colsample_bytree': {'tipo': 'float', 'min': 0.3, 'max': 1.0},
    'min_child_weight': {'tipo': 'int', 'min': 1, 'max': 10},
    'gamma': {'tipo': 'float', 'min': 0, 'max': 5},
    'reg_alpha': {'tipo': 'float', 'min': 1e-8, 'max': 10, 'log': True},
    'reg_lambda': {'tipo': 'float', 'min': 1e-8, 'max': 10, 'log': True},
}

ESPACIOS_LIGHTGBM = {
    'n_estimators': {'tipo': 'int', 'min': 50, 'max': 500},
    'num_leaves': {'tipo': 'int', 'min': 15, 'max': 127},
    'max_depth': {'tipo': 'int', 'min': 3, 'max': 15},
    'learning_rate': {'tipo': 'float', 'min': 0.01, 'max': 0.3, 'log': True},
    'subsample': {'tipo': 'float', 'min': 0.5, 'max': 1.0},
    'colsample_bytree': {'tipo': 'float', 'min': 0.3, 'max': 1.0},
    'min_child_samples': {'tipo': 'int', 'min': 5, 'max': 50},
    'reg_alpha': {'tipo': 'float', 'min': 1e-8, 'max': 10, 'log': True},
    'reg_lambda': {'tipo': 'float', 'min': 1e-8, 'max': 10, 'log': True},
}

ESPACIOS_RANDOMFOREST = {
    'n_estimators': {'tipo': 'int', 'min': 50, 'max': 500},
    'max_depth': {'tipo': 'int', 'min': 3, 'max': 25},
    'min_samples_split': {'tipo': 'int', 'min': 2, 'max': 20},
    'min_samples_leaf': {'tipo': 'int', 'min': 1, 'max': 20},
    'max_features': {'tipo': 'categorical', 'choices': ['sqrt', 'log2', None]},
}

ESPACIOS_RED = {
    'hidden_layers': {'tipo': 'int', 'min': 1, 'max': 3},
    'hidden_units': {'tipo': 'int', 'min': 16, 'max': 256, 'log': True},
    'learning_rate': {'tipo': 'float', 'min': 1e-4, 'max': 1e-2, 'log': True},
    'dropout': {'tipo': 'float', 'min': 0.0, 'max': 0.5},
    'batch_size': {'tipo': 'int', 'min': 16, 'max': 128},
    'epochs': {'tipo': 'int', 'min': 10, 'max': 100},
}


class HyperparameterOptimizer:
    def __init__(self, study_name: str = 'agente_financiero',
                 storage: Optional[str] = None,
                 direction: str = 'maximize'):
        self.study_name = study_name
        self.storage = storage or f'sqlite:///{OUTPUT_DIR}/optuna_{study_name}.db'
        self.direction = direction
        self.study: Optional[optuna.Study] = None
        self.trials: List[TrialResult] = []
        self.baseline_metric: Optional[float] = None

    def crear_estudio(self, n_startup: int = 5, n_ei: int = 3) -> 'optuna.Study':
        if not OPTUNA_AVAILABLE:
            raise ImportError('optuna no instalado. pip install optuna')
        sampler = optuna.samplers.TPESampler(
            n_startup_trials=n_startup, n_ei_candidates=n_ei)
        pruner = optuna.pruners.MedianPruner(
            n_startup_trials=5, n_warmup_steps=20)
        self.study = optuna.create_study(
            study_name=self.study_name, storage=self.storage,
            direction=self.direction, sampler=sampler,
            pruner=pruner, load_if_exists=True)
        return self.study

    def optimizar(self, objective_fn: Callable, search_space: Dict,
                  n_trials: int = 50, timeout: Optional[int] = None,
                  n_jobs: int = 1, baseline_fn: Optional[Callable] = None) -> StudySummary:
        if baseline_fn:
            self.baseline_metric = baseline_fn()
        if not self.study:
            self.crear_estudio()

        import time as _time
        def wrapper(trial):
            params = sugerir_params(trial, search_space)
            t0 = _time.time()
            try:
                result = objective_fn(params, trial)
                if isinstance(result, tuple):
                    metric, extra = result[0], result[1]
                else:
                    metric = result
                duration = _time.time() - t0
                self.trials.append(TrialResult(
                    trial_id=trial.number, params=params,
                    metric=float(metric), duration=duration))
                return metric
            except Exception as e:
                warnings.warn(f'Trial {trial.number} fallo: {e}')
                raise optuna.TrialPruned(f'Trial fallo: {e}')

        self.study.optimize(wrapper, n_trials=n_trials,
                            timeout=timeout, n_jobs=n_jobs)
        return self.resumen()

    def resumen(self) -> StudySummary:
        best = None
        if self.study:
            try:
                best = self.study.best_trial
            except ValueError:
                pass
        return StudySummary(
            study_name=self.study_name,
            n_trials=len(self.trials),
            best_params=best.params if best else {},
            best_metric=best.value if best else 0.0,
            improvement=(best.value / self.baseline_metric - 1)
            if (best and self.baseline_metric) else 0.0,
            duration=getattr(best, 'duration_seconds', 0.0) or 0.0,
            trials=self.trials)

    def importancia_parametros(self) -> Dict[str, float]:
        if not self.study or len(self.trials) < 10:
            return {}
        return optuna.importance.get_param_importances(self.study)

    def generar_reporte(self, X_names: List[str]) -> str:
        s = self.resumen()
        report = f'=== Optimizacion: {s.study_name} ===\n'
        report += f'Trials: {s.n_trials}\n'
        report += f'Mejor metrica: {s.best_metric:.4f}\n'
        if s.improvement:
            report += f'Mejora vs baseline: {s.improvement:+.1%}\n'
        report += f'Mejores parametros:\n'
        for k, v in sorted(s.best_params.items()):
            report += f'  {k}: {v}\n'
        if self.study:
            imp = self.importancia_parametros()
            if imp:
                report += 'Importancia:\n'
                for k, v in sorted(imp.items(), key=lambda x: -x[1])[:5]:
                    report += f'  {k}: {v:.3f}\n'
        path = OUTPUT_DIR / f'{s.study_name}_report.txt'
        path.write_text(report, encoding='utf-8')
        if db:
            db.guardar_metrica('optuna', 'best_metric', s.best_metric,
                               {'study': s.study_name, 'n_trials': s.n_trials})
        return str(path)


class OptimizadorXGBoost:
    def __init__(self, X: pd.DataFrame, y: pd.Series,
                 cv_fn: Optional[Callable] = None,
                 metric_fn: Optional[Callable] = None):
        self.X = X
        self.y = y
        self.cv_fn = cv_fn or self._default_cv
        self.metric_fn = metric_fn or self._default_metric
        self.X_names = list(X.columns)

    def _default_cv(self, X, y, params) -> float:
        try:
            import xgboost as xgb
            dtrain = xgb.DMatrix(X, label=y)
            cv_result = xgb.cv(params, dtrain, nfold=3,
                               num_boost_round=50, early_stopping_rounds=10,
                               verbose_eval=False)
            return float(cv_result['test-auc-mean'].iloc[-1])
        except Exception:
            from sklearn.model_selection import cross_val_score
            from xgboost import XGBClassifier
            p = {k: v for k, v in params.items()
                 if k not in ('verbosity', 'n_estimators')}
            model = XGBClassifier(**p, verbosity=0, n_estimators=100)
            scores = cross_val_score(model, X, y, cv=3,
                                     scoring='roc_auc')
            return float(scores.mean())

    def _default_metric(self, y_true, y_pred):
        from sklearn.metrics import roc_auc_score
        return float(roc_auc_score(y_true, y_pred))

    def objetivo(self, params: Dict, trial) -> float:
        params_xgb = {k: v for k, v in params.items()
                      if k in ESPACIOS_XGBOOST}
        params_xgb['verbosity'] = 0
        params_xgb['random_state'] = 42
        try:
            score = self.cv_fn(self.X, self.y, params_xgb)
        except Exception:
            from sklearn.model_selection import cross_val_score
            from xgboost import XGBClassifier
            model = XGBClassifier(**params_xgb, verbosity=0, n_estimators=50)
            scores = cross_val_score(model, self.X, self.y, cv=3,
                                     scoring='roc_auc')
            score = float(scores.mean())
        return score

    def optimizar(self, n_trials: int = 50, timeout: Optional[int] = None,
                  study_name: str = 'xgboost_default') -> StudySummary:
        opt = HyperparameterOptimizer(study_name=study_name)
        return opt.optimizar(
            self.objetivo, ESPACIOS_XGBOOST,
            n_trials=n_trials, timeout=timeout,
            baseline_fn=lambda: self._default_metric(
                self.y, np.full_like(self.y, self.y.mean())))


class OptimizadorLightGBM:
    def __init__(self, X: pd.DataFrame, y: pd.Series,
                 cv_fn: Optional[Callable] = None):
        self.X = X
        self.y = y
        self.cv_fn = cv_fn or self._default_cv

    def _default_cv(self, X, y, params) -> float:
        try:
            import lightgbm as lgb
            dtrain = lgb.Dataset(X, label=y)
            cv_result = lgb.cv(params, dtrain, nfold=3,
                               num_boost_round=50, early_stopping_rounds=10,
                               verbose_eval=False)
            return float(max(cv_result['auc-mean']))
        except Exception:
            from sklearn.model_selection import cross_val_score
            from lightgbm import LGBMClassifier
            model = LGBMClassifier(**params, verbose=-1, n_estimators=50)
            scores = cross_val_score(model, X, y, cv=3, scoring='roc_auc')
            return float(scores.mean())

    def objetivo(self, params: Dict, trial) -> float:
        params_lgb = {k: v for k, v in params.items()
                      if k in ESPACIOS_LIGHTGBM}
        params_lgb['verbose'] = -1
        params_lgb['random_state'] = 42
        return self.cv_fn(self.X, self.y, params_lgb)

    def optimizar(self, n_trials: int = 50, timeout: Optional[int] = None,
                  study_name: str = 'lightgbm_default') -> StudySummary:
        opt = HyperparameterOptimizer(study_name=study_name)
        return opt.optimizar(self.objetivo, ESPACIOS_LIGHTGBM,
                             n_trials=n_trials, timeout=timeout)


class AutoMLPipeline:
    def __init__(self):
        self.optimizadores: Dict[str, Any] = {}
        self.resultados: Dict[str, StudySummary] = {}
        self.mejor_modelo: Optional[str] = None
        self.mejor_metrica: float = 0.0
        self.mejores_params: Dict = {}

    def comparar_modelos(self, X: pd.DataFrame, y: pd.Series,
                         modelos: Optional[List[str]] = None,
                         n_trials: int = 30,
                         timeout: Optional[int] = 300) -> Dict[str, StudySummary]:
        modelos = modelos or ['xgboost', 'lightgbm', 'randomforest']
        for nombre in modelos:
            if nombre == 'xgboost':
                opt = OptimizadorXGBoost(X, y)
            elif nombre == 'lightgbm':
                opt = OptimizadorLightGBM(X, y)
            elif nombre == 'randomforest':
                continue
            else:
                continue
            resumen = opt.optimizar(n_trials=n_trials // len(modelos),
                                    timeout=timeout,
                                    study_name=f'{nombre}_{datetime.now().strftime("%Y%m%d")}')
            self.resultados[nombre] = resumen
            self.optimizadores[nombre] = opt
            if resumen.best_metric > self.mejor_metrica:
                self.mejor_metrica = resumen.best_metric
                self.mejor_modelo = nombre
                self.mejores_params = resumen.best_params
        return self.resultados

    def generar_reporte_final(self) -> str:
        lines = ['=== AutoML Report ===', f'Fecha: {datetime.now().isoformat()}', '']
        for nombre, res in sorted(self.resultados.items(),
                                  key=lambda x: x[1].best_metric, reverse=True):
            lines.append(f'{nombre}: best={res.best_metric:.4f} '
                         f'trials={res.n_trials}')
            if res.improvement:
                lines.append(f'  mejora: {res.improvement:+.1%}')
        lines.append(f'\nMejor modelo: {self.mejor_modelo} '
                     f'({self.mejor_metrica:.4f})')
        path = OUTPUT_DIR / f'automl_report_{datetime.now().strftime("%Y%m%d")}.txt'
        Path(path).write_text('\n'.join(lines), encoding='utf-8')
        return str(path)


try:
    import xgboost
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False

if __name__ == '__main__':
    np.random.seed(42)
    n = 500
    X = pd.DataFrame({f'f{i}': np.random.randn(n) for i in range(10)})
    y = (X['f0'] + X['f1'] * 0.5 > 0).astype(int)
    if XGB_AVAILABLE:
        opt = OptimizadorXGBoost(X, y)
        res = opt.optimizar(n_trials=5)
        print(f'Best: {res.best_metric:.4f} params={res.best_params}')
        path = opt.optimizadores['xgboost_default'].generar_reporte(list(X.columns))
        print(f'Reporte: {path}')
