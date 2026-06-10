#!/usr/bin/env python3
"""
model_store.py - Persistencia y versionado de modelos ML.
Guarda/carga modelos XGBoost, ensemble, calibradores, scalers.
Integra con MLflow para experiment tracking.
"""
import os
import json
import joblib
import hashlib
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
import numpy as np

try:
    import mlflow
    import mlflow.sklearn
    import mlflow.xgboost
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False

from config.settings import get_setting

DATA_DIR = os.environ.get('GITHUB_WORKSPACE', '.')
MODEL_DIR = Path(DATA_DIR) / 'Datos' / 'models'
MODEL_DIR.mkdir(parents=True, exist_ok=True)

MLFLOW_TRACKING_URI = get_setting('ml.mlflow.tracking_uri', 'file:./mlruns')
MLFLOW_EXPERIMENT = get_setting('ml.mlflow.experiment', 'agente-financiero')


class ModelStore:
    def __init__(self, use_mlflow: bool = True):
        self.model_dir = MODEL_DIR
        self.use_mlflow = use_mlflow and MLFLOW_AVAILABLE
        if self.use_mlflow:
            mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
            mlflow.set_experiment(MLFLOW_EXPERIMENT)
        self.metadata_file = self.model_dir / 'model_metadata.json'
        self._load_metadata()

    def _load_metadata(self):
        if self.metadata_file.exists():
            with open(self.metadata_file) as f:
                self.metadata = json.load(f)
        else:
            self.metadata = {'models': {}, 'versions': {}}

    def _save_metadata(self):
        with open(self.metadata_file, 'w') as f:
            json.dump(self.metadata, f, indent=2)

    def _model_hash(self, model, params: Dict) -> str:
        """Genera hash único del modelo basado en parámetros y coeficientes."""
        if hasattr(model, 'get_booster'):
            booster = model.get_booster()
            model_bytes = booster.save_raw('json')
        elif hasattr(model, 'coef_'):
            model_bytes = str(model.coef_).encode()
        else:
            model_bytes = joblib.dumps(model)
        param_str = json.dumps(params, sort_keys=True)
        return hashlib.sha256(model_bytes + param_str.encode()).hexdigest()[:16]

    def save_model(
        self,
        model: Any,
        name: str,
        regime: str = 'global',
        version: Optional[str] = None,
        params: Optional[Dict] = None,
        metrics: Optional[Dict] = None,
        feature_names: Optional[List[str]] = None,
        scaler: Any = None,
        calibrator: Any = None
    ) -> str:
        """
        Guarda modelo con versionado semántico.
        Retorna version_id (ej: v1.2.3).
        """
        if version is None:
            existing = self.metadata['versions'].get(f'{name}_{regime}', [])
            major = len(existing) + 1
            version = f'v{major}.0.0'
        
        version_dir = self.model_dir / name / regime / version
        version_dir.mkdir(parents=True, exist_ok=True)

        model_path = version_dir / 'model.pkl'
        joblib.dump(model, model_path)

        artifacts = {'model': 'model.pkl'}
        
        if scaler is not None:
            scaler_path = version_dir / 'scaler.pkl'
            joblib.dump(scaler, scaler_path)
            artifacts['scaler'] = 'scaler.pkl'
        
        if calibrator is not None:
            cal_path = version_dir / 'calibrator.pkl'
            joblib.dump(calibrator, cal_path)
            artifacts['calibrator'] = 'calibrator.pkl'
        
        if feature_names:
            with open(version_dir / 'features.json', 'w') as f:
                json.dump(feature_names, f)
            artifacts['features'] = 'features.json'

        model_hash = self._model_hash(model, params or {})
        
        record = {
            'name': name,
            'regime': regime,
            'version': version,
            'hash': model_hash,
            'params': params or {},
            'metrics': metrics or {},
            'feature_names': feature_names or [],
            'artifacts': artifacts,
            'created_at': datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z'),
            'size_bytes': model_path.stat().st_size
        }

        key = f'{name}_{regime}'
        if key not in self.metadata['models']:
            self.metadata['models'][key] = {}
        self.metadata['models'][key][version] = record
        
        if key not in self.metadata['versions']:
            self.metadata['versions'][key] = []
        self.metadata['versions'][key].append(version)
        
        self._save_metadata()

        if self.use_mlflow:
            self._log_mlflow(name, regime, version, version_dir, params, metrics)

        print(f'[ModelStore] Guardado: {name}/{regime}/{version} (hash={model_hash})')
        return version

    def _log_mlflow(self, name, regime, version, version_dir, params, metrics):
        try:
            with mlflow.start_run(run_name=f'{name}_{regime}_{version}'):
                mlflow.log_params(params or {})
                mlflow.log_metrics(metrics or {})
                mlflow.log_param('regime', regime)
                mlflow.log_param('model_name', name)
                mlflow.log_param('version', version)
                for art_name, art_path in (version_dir / 'model.pkl',).items():
                    mlflow.log_artifact(str(version_dir / 'model.pkl'))
                if (version_dir / 'scaler.pkl').exists():
                    mlflow.log_artifact(str(version_dir / 'scaler.pkl'))
                if (version_dir / 'calibrator.pkl').exists():
                    mlflow.log_artifact(str(version_dir / 'calibrator.pkl'))
        except Exception as e:
            print(f'[ModelStore] MLflow log failed: {e}')

    def load_model(
        self,
        name: str,
        regime: str = 'global',
        version: Optional[str] = None
    ) -> Tuple[Any, Dict]:
        """
        Carga modelo. Si version=None, carga el más reciente.
        Retorna (model, metadata).
        """
        key = f'{name}_{regime}'
        if key not in self.metadata['models']:
            raise ValueError(f'Modelo no encontrado: {name}/{regime}')
        
        versions = self.metadata['models'][key]
        if version is None:
            version = self.metadata['versions'][key][-1]
        
        if version not in versions:
            raise ValueError(f'Versión no encontrada: {version}')
        
        record = versions[version]
        version_dir = self.model_dir / name / regime / version
        
        model = joblib.load(version_dir / 'model.pkl')
        
        scaler = None
        if 'scaler' in record['artifacts']:
            scaler = joblib.load(version_dir / 'scaler.pkl')
        
        calibrator = None
        if 'calibrator' in record['artifacts']:
            calibrator = joblib.load(version_dir / 'calibrator.pkl')
        
        feature_names = None
        if 'features' in record['artifacts']:
            with open(version_dir / 'features.json') as f:
                feature_names = json.load(f)
        
        return {
            'model': model,
            'scaler': scaler,
            'calibrator': calibrator,
            'feature_names': feature_names,
            'metadata': record
        }

    def load_latest(self, name: str, regime: str = 'global') -> Tuple[Any, Dict]:
        """Carga la versión más reciente."""
        return self.load_model(name, regime, version=None)

    def list_models(self) -> Dict:
        """Lista todos los modelos guardados."""
        return self.metadata['models']

    def list_versions(self, name: str, regime: str = 'global') -> List[str]:
        key = f'{name}_{regime}'
        return self.metadata['versions'].get(key, [])

    def delete_model(self, name: str, regime: str = 'global', version: Optional[str] = None):
        """Elimina modelo (versión específica o todas)."""
        key = f'{name}_{regime}'
        if key not in self.metadata['models']:
            return
        
        if version is None:
            shutil.rmtree(self.model_dir / name / regime)
            del self.metadata['models'][key]
            del self.metadata['versions'][key]
        else:
            if version in self.metadata['models'][key]:
                shutil.rmtree(self.model_dir / name / regime / version)
                del self.metadata['models'][key][version]
                self.metadata['versions'][key].remove(version)
        
        self._save_metadata()

    def promote_to_production(self, name: str, regime: str = 'global', version: str = None):
        """Marca una versión como 'production' (symlink latest -> version)."""
        if version is None:
            version = self.metadata['versions'][f'{name}_{regime}'][-1]
        
        latest_link = self.model_dir / name / regime / 'latest'
        if latest_link.exists() or latest_link.is_symlink():
            latest_link.unlink()
        
        target = self.model_dir / name / regime / version
        latest_link.symlink_to(target, target_is_directory=True)
        
        self.metadata['models'][f'{name}_{regime}'][version]['production'] = True
        self._save_metadata()
        print(f'[ModelStore] Promovido a production: {name}/{regime}/{version}')

    def load_production(self, name: str, regime: str = 'global') -> Tuple[Any, Dict]:
        """Carga modelo marcado como production."""
        latest_link = self.model_dir / name / regime / 'latest'
        if not latest_link.exists():
            return self.load_latest(name, regime)
        
        version = latest_link.resolve().name
        return self.load_model(name, regime, version)

    def compare_versions(self, name: str, regime: str, v1: str, v2: str) -> Dict:
        """Compara métricas de dos versiones."""
        key = f'{name}_{regime}'
        m1 = self.metadata['models'][key].get(v1, {}).get('metrics', {})
        m2 = self.metadata['models'][key].get(v2, {}).get('metrics', {})
        
        diff = {}
        all_keys = set(m1.keys()) | set(m2.keys())
        for k in all_keys:
            diff[k] = {
                v1: m1.get(k),
                v2: m2.get(k),
                'delta': (m2.get(k, 0) - m1.get(k, 0)) if k in m1 and k in m2 else None
            }
        return diff


def get_model_store() -> ModelStore:
    """Singleton accessor."""
    if not hasattr(get_model_store, '_instance'):
        get_model_store._instance = ModelStore()
    return get_model_store._instance


if __name__ == '__main__':
    store = get_model_store()
    print('Modelos guardados:')
    for k, v in store.list_models().items():
        print(f'  {k}: {list(v.keys())}')