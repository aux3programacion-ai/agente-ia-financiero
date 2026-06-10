#!/usr/bin/env python3
"""
meta_learning.py - Meta-learning / Few-shot adaptation.
MAML (Model-Agnostic Meta-Learning) para adaptación ultrarrápida
a nuevos regímenes de mercado con <10 días de datos.
"""
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple, Callable
from dataclasses import dataclass, field
from datetime import datetime
import json
import os
from pathlib import Path

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.nn import functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from config.settings import get_setting

DATA_DIR = os.environ.get('GITHUB_WORKSPACE', '.')
OUTPUT_DIR = Path(DATA_DIR) / 'Datos' / 'meta_learning'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


class RapidAdaptationNetwork(nn.Module if TORCH_AVAILABLE else object):
    """Red neuronal base para adaptación rápida vía MAML."""

    def __init__(self, input_dim: int, hidden_dim: int = 64, output_dim: int = 1):
        if not TORCH_AVAILABLE:
            return
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
            nn.Sigmoid(),
        )

    def forward(self, x) -> 'torch.Tensor':
        return self.net(x)

    def clone(self):
        clone = RapidAdaptationNetwork(
            self.net[0].in_features,
            self.net[0].out_features,
            self.net[-2].out_features
        )
        clone.load_state_dict(self.state_dict())
        return clone


class MAMLLearner:
    """Model-Agnostic Meta-Learning para adaptación a regímenes.

    Entrena en múltiples tareas (regímenes históricos) para aprender
    una inicialización de pesos que se adapte rápidamente a nuevos regímenes
    con pocos pasos de gradiente.
    """

    def __init__(self, input_dim: int, hidden_dim: int = 64,
                 inner_lr: float = 0.01, outer_lr: float = 0.001,
                 inner_steps: int = 5):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.inner_lr = inner_lr
        self.outer_lr = outer_lr
        self.inner_steps = inner_steps
        self.trained = False

        if TORCH_AVAILABLE:
            self.model = RapidAdaptationNetwork(input_dim, hidden_dim)
            self.meta_optimizer = optim.Adam(self.model.parameters(), lr=outer_lr)
        else:
            self.model = None
            self.meta_optimizer = None

    def _inner_update(self, model, x,
                      y):
        """Un paso de adaptación interna (few-shot)."""
        cloned = model.clone()
        logits = cloned(x).squeeze()
        loss = F.binary_cross_entropy(logits, y, reduction='mean')
        grad = torch.autograd.grad(loss, cloned.parameters(), create_graph=True)
        for p, g in zip(cloned.parameters(), grad):
            p.data = p.data - self.inner_lr * g
        return cloned

    def meta_train_step(self, tasks: List[Tuple]) -> float:
        """Un paso de meta-entrenamiento en un batch de tareas.

        Cada tarea: (x_support, y_support, x_query, y_query)
        """
        if not TORCH_AVAILABLE or self.model is None:
            return 0.0

        meta_loss = 0.0
        n_tasks = len(tasks)

        for x_s, y_s, x_q, y_q in tasks:
            adapted = self.model.clone()
            for _ in range(self.inner_steps):
                adapted = self._inner_update(adapted, x_s, y_s)

            pred = adapted(x_q).squeeze()
            task_loss = F.binary_cross_entropy(pred, y_q, reduction='mean')
            meta_loss = meta_loss + task_loss

        meta_loss = meta_loss / n_tasks
        self.meta_optimizer.zero_grad()
        meta_loss.backward()
        self.meta_optimizer.step()

        return meta_loss.item()

    def adapt(self, x_support: np.ndarray, y_support: np.ndarray,
              steps: int = 10):
        """Adapta el modelo a un nuevo régimen con pocos ejemplos."""
        if not TORCH_AVAILABLE or self.model is None:
            return None

        adapted = self.model.clone()
        adapted.train()

        x_t = torch.FloatTensor(x_support)
        y_t = torch.FloatTensor(y_support)

        inner_opt = optim.SGD(adapted.parameters(), lr=self.inner_lr)

        for _ in range(steps):
            inner_opt.zero_grad()
            pred = adapted(x_t).squeeze()
            loss = F.binary_cross_entropy(pred, y_t, reduction='mean')
            loss.backward()
            inner_opt.step()

        return adapted

    def predict(self, model, x: np.ndarray) -> np.ndarray:
        if not TORCH_AVAILABLE or model is None:
            return np.zeros(len(x))
        model.eval()
        with torch.no_grad():
            x_t = torch.FloatTensor(x)
            pred = model(x_t).squeeze().numpy()
        return pred


class FewShotRegimeAdapter:
    """Adaptación few-shot a nuevos regímenes de mercado."""

    def __init__(self, feature_cols: List[str] = None):
        self.feature_cols = feature_cols or [
            'rsi_14', 'macd_hist', 'vol_ratio', 'volatility_20d',
            'sma50_dist_pct', 'atr_pct', 'returns_5d', 'returns_20d'
        ]
        self.maml: Optional[MAMLLearner] = None
        self.adapted_model = None
        self.regime_history: List[Dict] = []
        self.adaptation_log: List[Dict] = []

    def prepare_tasks(self, data: pd.DataFrame, regime_col: str = 'regime',
                      target_col: str = 'forward_return',
                      n_support: int = 20) -> List:
        """Divide datos históricos en tareas por régimen para meta-entrenamiento."""
        feature_data = data[self.feature_cols].values
        target_data = (data[target_col] > 0).astype(float).values

        tasks = []
        for regime in data[regime_col].unique():
            mask = data[regime_col] == regime
            idx = np.where(mask)[0]
            if len(idx) < n_support * 2:
                continue

            np.random.shuffle(idx)
            support_idx = idx[:n_support]
            query_idx = idx[n_support:2 * n_support]

            x_s = feature_data[support_idx]
            y_s = target_data[support_idx]
            x_q = feature_data[query_idx]
            y_q = target_data[query_idx]

            if TORCH_AVAILABLE:
                tasks.append((
                    torch.FloatTensor(x_s),
                    torch.FloatTensor(y_s),
                    torch.FloatTensor(x_q),
                    torch.FloatTensor(y_q),
                ))

        return tasks

    def meta_train(self, data: pd.DataFrame, regime_col: str = 'regime',
                   target_col: str = 'forward_return',
                   n_epochs: int = 100, n_support: int = 20) -> Dict:
        """Entrena el meta-modelo en todos los regímenes históricos."""
        n_features = len(self.feature_cols)
        self.maml = MAMLLearner(input_dim=n_features)

        tasks = self.prepare_tasks(data, regime_col, target_col, n_support)
        if not tasks:
            return {'error': 'No hay suficientes datos para crear tareas'}

        losses = []
        for epoch in range(n_epochs):
            np.random.shuffle(tasks)
            batch = tasks[:min(16, len(tasks))]
            loss = self.maml.meta_train_step(batch)
            losses.append(loss)

            if (epoch + 1) % 20 == 0:
                print(f'[MetaLearning] Epoch {epoch+1}/{n_epochs}, Loss: {loss:.4f}')

        self.trained = True
        return {'final_loss': losses[-1] if losses else 0, 'n_tasks': len(tasks)}

    def adapt_to_new_regime(self, x_new: pd.DataFrame, y_new: pd.Series,
                            regime_name: str = 'nuevo') -> Dict:
        """Adapta el modelo a un nuevo régimen con pocos datos."""
        if self.maml is None or not self.maml.trained:
            return {'error': 'Meta-modelo no entrenado. Ejecuta meta_train primero.'}

        feature_data = x_new[self.feature_cols].values
        target_data = (y_new > 0).astype(float).values

        self.adapted_model = self.maml.adapt(feature_data, target_data, steps=10)

        preds = self.maml.predict(self.adapted_model, feature_data)
        accuracy = float(np.mean((preds > 0.5) == target_data))

        record = {
            'regime': regime_name,
            'n_samples': len(feature_data),
            'accuracy': accuracy,
            'timestamp': datetime.now().isoformat(),
        }
        self.adaptation_log.append(record)

        return record

    def predict(self, x: pd.DataFrame) -> np.ndarray:
        if self.adapted_model is None or self.maml is None:
            return np.zeros(len(x))
        return self.maml.predict(self.adapted_model, x[self.feature_cols].values)

    def save_state(self):
        state = {
            'adaptation_log': self.adaptation_log,
            'regime_history': self.regime_history,
        }
        path = OUTPUT_DIR / 'few_shot_state.json'
        path.write_text(json.dumps(state, indent=2), encoding='utf-8')

    def load_state(self):
        path = OUTPUT_DIR / 'few_shot_state.json'
        if path.exists():
            state = json.loads(path.read_text())
            self.adaptation_log = state.get('adaptation_log', [])
            self.regime_history = state.get('regime_history', [])


class MetaLearningPipeline:
    """Pipeline completo de meta-learning."""

    def __init__(self):
        self.adapter = FewShotRegimeAdapter()
        self.results: Dict[str, Any] = {}

    def run(self, data: pd.DataFrame, regime_col: str = 'regime',
            target_col: str = 'forward_return',
            n_epochs: int = 100) -> Dict:
        train_result = self.adapter.meta_train(data, regime_col, target_col, n_epochs)
        self.results['train'] = train_result

        if 'error' in train_result:
            return self.results

        self.adapter.save_state()
        return self.results


if __name__ == '__main__':
    print('[MetaLearning] Ejecutando pipeline...')
    np.random.seed(42)

    n = 2000
    n_features = 8
    data = pd.DataFrame(
        np.random.randn(n, n_features),
        columns=['rsi_14', 'macd_hist', 'vol_ratio', 'volatility_20d',
                 'sma50_dist_pct', 'atr_pct', 'returns_5d', 'returns_20d']
    )
    data['regime'] = np.random.choice(['ALCISTA', 'BAJISTA', 'LATERAL'], n)
    data['forward_return'] = np.random.randn(n) * 0.02

    pipeline = MetaLearningPipeline()
    results = pipeline.run(data, n_epochs=30)
    print(f'Resultados: {results}')