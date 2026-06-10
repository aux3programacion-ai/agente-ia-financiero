#!/usr/bin/env python3
"""
mlops_pipeline.py - MLOps completo.
Pipeline automatico, versionado, deteccion de drift,
rollback, A/B testing, logging de decisiones.
"""
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
import json, os, hashlib, pickle
from pathlib import Path

try:
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

from config.settings import get_setting

DATA_DIR = os.environ.get('GITHUB_WORKSPACE', '.')
OUTPUT_DIR = Path(DATA_DIR) / 'Datos' / 'mlops'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


@dataclass
class ModelVersion:
    model_id: str; version: int; path: str; metrics: Dict
    features: List[str]; created: str; status: str = 'staging'


@dataclass
class DataVersion:
    dataset_id: str; version: int; path: str; rows: int
    columns: List[str]; hash: str; created: str


@dataclass
class DecisionLog:
    timestamp: str; ticker: str; signal: str; confidence: float
    model_version: int; features_used: Dict; outcome: Optional[float] = None
    correct: Optional[bool] = None


class DataVersioner:
    def __init__(self):
        self.versions: List[DataVersion] = []

    def snapshot(self, data: pd.DataFrame, dataset_id: str) -> DataVersion:
        ver = len([v for v in self.versions if v.dataset_id == dataset_id]) + 1
        data_hash = hashlib.md5(pd.util.hash_pandas_object(data).values).hexdigest()
        path = OUTPUT_DIR / f'{dataset_id}_v{ver}.csv'
        data.to_csv(path, index=False)
        dv = DataVersion(dataset_id=dataset_id, version=ver, path=str(path),
            rows=len(data), columns=list(data.columns), hash=data_hash,
            created=datetime.now().isoformat())
        self.versions.append(dv)
        return dv

    def load(self, dataset_id: str, version: Optional[int] = None) -> pd.DataFrame:
        candidates = [v for v in self.versions if v.dataset_id == dataset_id]
        if version:
            candidates = [v for v in candidates if v.version == version]
        if not candidates:
            return pd.DataFrame()
        return pd.read_csv(candidates[-1].path)

    def get_history(self, dataset_id: str) -> List[DataVersion]:
        return [v for v in self.versions if v.dataset_id == dataset_id]


class DriftDetector:
    def __init__(self, threshold=0.05, window=100):
        self.threshold = threshold
        self.window = window
        self.reference_stats: Dict[str, Dict] = {}
        self.drift_log: List[Dict] = []

    def set_reference(self, data: pd.DataFrame, name: str = 'reference'):
        self.reference_stats[name] = {
            'means': data.mean().to_dict(), 'stds': data.std().to_dict(),
            'n': len(data), 'timestamp': datetime.now().isoformat()}

    def detect(self, data: pd.DataFrame, name: str = 'current') -> Dict:
        if not self.reference_stats:
            return {'drift_detected': False, 'message': 'No reference set'}
        ref = list(self.reference_stats.values())[-1]
        drifted_features = []
        for col in data.columns:
            if col in ref['means']:
                z_score = abs(data[col].mean() - ref['means'][col]) / max(ref['stds'].get(col, 0.001), 0.001)
                if z_score > 3.0:
                    drifted_features.append({'feature': col, 'z_score': float(z_score)})
        result = {'drift_detected': len(drifted_features) > 0,
                  'n_drifted': len(drifted_features), 'features': drifted_features,
                  'timestamp': datetime.now().isoformat()}
        self.drift_log.append(result)
        return result

    def get_log(self) -> List[Dict]:
        return self.drift_log[-50:]


class ModelRegistry:
    def __init__(self):
        self.models: Dict[str, List[ModelVersion]] = {}
        self.production: Dict[str, ModelVersion] = {}

    def register(self, model_id: str, path: str, metrics: Dict,
                 features: List[str]) -> ModelVersion:
        versions = self.models.get(model_id, [])
        ver = len(versions) + 1
        mv = ModelVersion(model_id=model_id, version=ver, path=path,
            metrics=metrics, features=features,
            created=datetime.now().isoformat(), status='staging')
        versions.append(mv)
        self.models[model_id] = versions
        return mv

    def promote(self, model_id: str, version: int):
        versions = self.models.get(model_id, [])
        for v in versions:
            if v.version == version:
                v.status = 'production'
                self.production[model_id] = v
                break

    def rollback(self, model_id: str) -> Optional[ModelVersion]:
        if model_id in self.production:
            prod = self.production[model_id]
            prod.status = 'rolled_back'
            versions = self.models.get(model_id, [])
            older = [v for v in versions if v.version < prod.version and v.status == 'staging']
            if older:
                self.production[model_id] = older[-1]
                older[-1].status = 'production'
                return older[-1]
        return None


class ABTester:
    def __init__(self):
        self.trials: Dict[str, Dict] = {}
        self.results: Dict[str, List] = {}

    def start_trial(self, trial_id: str, model_a: str, model_b: str,
                    traffic_split: float = 0.5, min_samples: int = 100):
        self.trials[trial_id] = {'model_a': model_a, 'model_b': model_b,
            'traffic_split': traffic_split, 'min_samples': min_samples,
            'results_a': [], 'results_b': [], 'start': datetime.now().isoformat()}

    def record(self, trial_id: str, model: str, correct: bool):
        if trial_id not in self.trials:
            return
        if model not in self.results:
            self.results[model] = []
        self.results[model].append(correct)

    def evaluate(self, trial_id: str) -> Dict:
        if trial_id not in self.trials:
            return {'error': 'Trial not found'}
        t = self.trials[trial_id]
        ra = self.results.get(t['model_a'], [])
        rb = self.results.get(t['model_b'], [])
        if len(ra) < t['min_samples'] or len(rb) < t['min_samples']:
            return {'status': 'running', 'n_a': len(ra), 'n_b': len(rb)}
        acc_a = np.mean(ra) if ra else 0
        acc_b = np.mean(rb) if rb else 0
        winner = t['model_a'] if acc_a > acc_b else t['model_b']
        return {'status': 'completed', 'winner': winner,
                'accuracy_a': float(acc_a), 'accuracy_b': float(acc_b),
                'n_a': len(ra), 'n_b': len(rb)}


class MLOpsPipeline:
    def __init__(self):
        self.data_versioner = DataVersioner()
        self.drift = DriftDetector()
        self.registry = ModelRegistry()
        self.ab = ABTester()
        self.logs: List[DecisionLog] = []

    def log_decision(self, ticker: str, signal: str, confidence: float,
                     model_version: int, features: Dict):
        dl = DecisionLog(timestamp=datetime.now().isoformat(), ticker=ticker,
            signal=signal, confidence=confidence, model_version=model_version,
            features_used=features)
        self.logs.append(dl)
        return dl

    def update_outcome(self, idx: int, outcome: float, correct: bool):
        if 0 <= idx < len(self.logs):
            self.logs[idx].outcome = outcome
            self.logs[idx].correct = correct

    def get_dashboard(self) -> Dict:
        n_logs = len(self.logs)
        correct = sum(1 for l in self.logs if l.correct is True)
        incorrect = sum(1 for l in self.logs if l.correct is False)
        return {
            'total_decisions': n_logs,
            'accuracy': float(correct / max(n_logs, 1)),
            'correct': correct,
            'incorrect': incorrect,
            'models_registered': sum(len(v) for v in self.registry.models.values()),
            'models_in_production': len(self.registry.production),
            'drift_events': len(self.drift.drift_log),
            'data_snapshots': len(self.data_versioner.versions)}

    def save_state(self):
        path = OUTPUT_DIR / 'mlops_state.json'
        state = {'logs': [asdict(l) for l in self.logs[-1000:]],
                 'n_logs': len(self.logs)}
        path.write_text(json.dumps(state, indent=2), encoding='utf-8')

    def report(self) -> str:
        d = self.get_dashboard()
        report = f'MLOps Dashboard\n'
        report += f'Decisiones: {d["total_decisions"]} '
        report += f'(Precision: {d["accuracy"]:.1%})\n'
        report += f'Modelos registrados: {d["models_registered"]}\n'
        report += f'En produccion: {d["models_in_production"]}\n'
        report += f'Eventos de drift: {d["drift_events"]}\n'
        path = OUTPUT_DIR / 'mlops_report.txt'
        path.write_text(report, encoding='utf-8')
        return str(path)


if __name__ == '__main__':
    mlops = MLOpsPipeline()
    mlops.log_decision('NVDA', 'COMPRA', 0.78, 3, {'rsi': 65, 'macd': 1.2})
    mlops.log_decision('AAPL', 'VENTA', 0.65, 3, {'rsi': 45, 'macd': -0.5})
    mlops.update_outcome(0, 0.02, True)
    mlops.update_outcome(1, -0.01, False)
    data = pd.DataFrame(np.random.randn(200, 5), columns=['f1', 'f2', 'f3', 'f4', 'f5'])
    snap = mlops.data_versioner.snapshot(data, 'features')
    mlops.drift.set_reference(data.iloc[:100])
    drift_result = mlops.drift.detect(data.iloc[100:])
    print(f'Dashboard: {json.dumps(mlops.get_dashboard(), indent=2)}')
    print(f'Drift: {drift_result["drift_detected"]}')
    print(mlops.report())
