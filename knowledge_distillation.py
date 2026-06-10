#!/usr/bin/env python3
"""
knowledge_distillation.py - Knowledge distillation: teacher ensemble -> student model.
Permite inferencia rápida preservando precisión del ensemble.
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

DATA_DIR = os.environ.get('GITHUB_WORKSPACE', '.')
OUTPUT_DIR = Path(DATA_DIR) / 'Datos'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DIST_CONFIG = get_setting('ml.distillation', {})
ENABLED = DIST_CONFIG.get('enabled', True)
STUDENT_MODEL = DIST_CONFIG.get('student_model', 'xgboost_small')
XGB_PARAMS = get_setting('ml.xgboost', {})


class KnowledgeDistiller:
    def __init__(self, teacher_models: List[Any] = None):
        self.teacher_models = teacher_models or []
        self.student_model = None
        self.distillation_loss = None
        self.performance_gap = None
        self.store = get_model_store()

    def add_teacher(self, model):
        self.teacher_models.append(model)

    def generate_soft_targets(
        self, X: np.ndarray, temperature: float = 2.0
    ) -> np.ndarray:
        """
        Genera soft targets promediando predicciones del ensemble teacher.
        
        Args:
            X: Features
            temperature: Temperatura para suavizar probabilidades
            
        Returns:
            Soft targets promedio del ensemble
        """
        if not self.teacher_models:
            return None
        
        predictions = []
        for model in self.teacher_models:
            if hasattr(model, 'predict_proba'):
                pred = model.predict_proba(X)[:, 1]
            elif hasattr(model, 'predict'):
                pred = model.predict(X)
            else:
                continue
            
            # Aplicar temperatura
            pred = np.clip(pred, 1e-7, 1 - 1e-7)
            logits = np.log(pred / (1 - pred))
            pred_soft = 1 / (1 + np.exp(-logits / temperature))
            predictions.append(pred_soft)
        
        if not predictions:
            return None
        
        return np.mean(predictions, axis=0)

    def distill(
        self,
        X: np.ndarray,
        y_hard: Optional[np.ndarray] = None,
        temperature: float = 2.0,
        alpha: float = 0.7,
        student_params: Optional[Dict] = None,
        save_model: bool = True,
        model_name: str = 'student_xgboost',
        regime: str = 'global'
    ) -> Dict:
        """
        Entrena student model usando soft targets del teacher.
        
        Args:
            X: Features de entrenamiento
            y_hard: Labels reales (opcional, para hard loss)
            temperature: Temperatura para soft targets
            alpha: Peso entre hard loss y distillation loss
            student_params: Parámetros del modelo student
            save_model: Guardar en model store
            model_name: Nombre en model store
        
        Returns:
            Dict con métricas de distillación
        """
        if student_params is None:
            student_params = {
                'n_estimators': 50,
                'max_depth': 3,
                'learning_rate': 0.1,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'verbosity': 0,
                **{k: v for k, v in XGB_PARAMS.items() 
                   if k not in ('n_estimators', 'max_depth', 'learning_rate')}
            }
        
        # Generar soft targets
        soft_targets = self.generate_soft_targets(X, temperature)
        if soft_targets is None:
            return {'error': 'No hay teachers para generar soft targets'}
        
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import accuracy_score, log_loss
        
        X_train, X_val, y_soft_train, y_soft_val = train_test_split(
            X, soft_targets, test_size=0.2, random_state=42
        )
        
        if y_hard is not None:
            _, _, y_hard_train, y_hard_val = train_test_split(
                X, y_hard, test_size=0.2, random_state=42
            )
        
        from xgboost import XGBClassifier
        from sklearn.ensemble import RandomForestClassifier
        
        if 'small' in STUDENT_MODEL.lower():
            student = XGBClassifier(**student_params, verbosity=0)
        elif 'randomforest' in STUDENT_MODEL.lower():
            student = RandomForestClassifier(
                n_estimators=50, max_depth=5, random_state=42
            )
        else:
            student = XGBClassifier(**student_params, verbosity=0)
        
        # Hard targets binarios de soft targets
        y_train_distill = (y_soft_train > 0.5).astype(int)
        student.fit(X_train, y_train_distill)
        
        # Evaluar
        y_pred = student.predict(X_val)
        y_proba = student.predict_proba(X_val)[:, 1]
        
        teacher_acc = accuracy_score((y_soft_val > 0.5).astype(int), y_soft_val.round())
        student_acc = accuracy_score((y_soft_val > 0.5).astype(int), y_pred)
        
        metrics = {
            'temperature': temperature,
            'alpha': alpha,
            'n_teachers': len(self.teacher_models),
            'n_samples': len(X),
            'teacher_accuracy': float(teacher_acc),
            'student_accuracy': float(student_acc),
            'accuracy_gap': float(teacher_acc - student_acc),
            'relative_accuracy': float(student_acc / max(teacher_acc, 0.01)),
            'student_log_loss': float(log_loss((y_soft_val > 0.5).astype(int), y_proba)),
            'n_student_params': student.get_params() if hasattr(student, 'get_params') else {},
            'student_size': sum(p.nbytes for p in student.get_booster().get_dump() if hasattr(student, 'get_booster')) 
                           if hasattr(student, 'get_booster') else 0
        }
        
        self.student_model = student
        self.performance_gap = metrics['accuracy_gap']
        
        if save_model and metrics['accuracy_gap'] < 0.05:
            self.store.save_model(
                model=student,
                name=model_name,
                regime=regime,
                params=student_params,
                metrics=metrics,
                feature_names=None
            )
        
        print(f'[Distillation] Teacher={teacher_acc:.3f}, Student={student_acc:.3f}, '
              f'Gap={metrics["accuracy_gap"]:.3f}')
        return metrics

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.student_model is None:
            raise ValueError('Student model no entrenado')
        
        if hasattr(self.student_model, 'predict_proba'):
            return self.student_model.predict_proba(X)[:, 1]
        return self.student_model.predict(X)

    def compress_ensemble(
        self,
        models: List[Tuple[str, Any]],
        X: np.ndarray,
        y: Optional[np.ndarray] = None,
        max_teachers: int = 3
    ) -> Dict:
        """
        Comprime un ensemble grande seleccionando los mejores teachers.
        """
        from sklearn.metrics import accuracy_score
        
        scored = []
        for name, model in models:
            if hasattr(model, 'predict'):
                y_pred = model.predict(X)
                if hasattr(model, 'predict_proba'):
                    y_proba = model.predict_proba(X)[:, 1]
                else:
                    y_proba = y_pred
                
                if y is not None:
                    acc = accuracy_score(y, y_pred)
                else:
                    acc = 0.5
                
                scored.append((name, model, acc, y_proba))
        
        scored.sort(key=lambda x: x[2], reverse=True)
        top_teachers = scored[:max_teachers]
        
        self.teacher_models = [m for _, m, _, _ in top_teachers]
        
        result = self.distill(X, y)
        result['selected_teachers'] = [s[0] for s in top_teachers]
        result['skipped_teachers'] = [s[0] for s in scored[max_teachers:]]
        
        return result


def distill_xgboost_ensemble(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    n_teachers: int = 5
) -> Dict:
    """
    Entrena ensemble de teachers XGBoost con diferentes semillas,
    luego destila a un student pequeño.
    """
    from xgboost import XGBClassifier
    from sklearn.metrics import accuracy_score
    
    teachers = []
    for i in range(n_teachers):
        model = XGBClassifier(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.05,
            random_state=42 + i,
            subsample=0.8,
            colsample_bytree=0.8,
            verbosity=0
        )
        model.fit(X_train, y_train)
        teachers.append(model)
        acc = accuracy_score(y_val, model.predict(X_val))
        print(f'  Teacher {i+1}: acc={acc:.3f}')
    
    distiller = KnowledgeDistiller(teachers)
    result = distiller.compress_ensemble(
        [(f'teacher_{i}', m) for i, m in enumerate(teachers)],
        np.vstack([X_train, X_val]),
        np.hstack([y_train, y_val]),
        max_teachers=min(3, n_teachers)
    )
    
    return result


if __name__ == '__main__':
    print('[Distillation] Test con datos sintéticos...')
    np.random.seed(42)
    n = 1000
    X = np.random.randn(n, 5)
    y = ((X[:, 0] * 0.3 + X[:, 1] * 0.2 + np.random.randn(n) * 0.1) > 0).astype(int)
    
    result = distill_xgboost_ensemble(X[:700], y[:700], X[700:], y[700:], n_teachers=3)
    print(json.dumps({k: v for k, v in result.items() if isinstance(v, (int, float))}, indent=2))