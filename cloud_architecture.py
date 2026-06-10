#!/usr/bin/env python3
"""
cloud_architecture.py - Cloud-native architecture setup.
Ray (distributed compute), Feast (feature store), Prefect/Airflow (orchestration),
Kubernetes/Docker deployment configs.
"""
import json
import os
import sys
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime

try:
    import ray
    RAY_AVAILABLE = True
except ImportError:
    RAY_AVAILABLE = False

try:
    from prefect import flow, task
    PREFECT_AVAILABLE = True
except ImportError:
    PREFECT_AVAILABLE = False

from config.settings import get_setting

DATA_DIR = os.environ.get('GITHUB_WORKSPACE', '.')
OUTPUT_DIR = Path(DATA_DIR) / 'Datos' / 'cloud'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


class RayCluster:
    """Distributed compute with Ray for parallel backtesting and training."""
    
    def __init__(self, address: str = 'auto', num_cpus: int = None, num_gpus: int = None):
        self.address = address
        self.num_cpus = num_cpus or os.cpu_count() or 4
        self.num_gpus = num_gpus or 0
        self.initialized = False
    
    def start(self):
        if not RAY_AVAILABLE:
            print('[Ray] Not available. pip install ray')
            return False
        try:
            if not ray.is_initialized():
                ray.init(address=self.address, num_cpus=self.num_cpus, num_gpus=self.num_gpus,
                         ignore_reinit_error=True)
                print(f'[Ray] Started: {ray.cluster_resources()}')
            self.initialized = True
            return True
        except Exception as e:
            print(f'[Ray] Start failed: {e}')
            return False
    
    def stop(self):
        if RAY_AVAILABLE and ray.is_initialized():
            ray.shutdown()
            self.initialized = False
    
    @staticmethod
    def _backtest_batch(tickers: List[str], start: str, end: str) -> Dict:
        from backtest_engine import run_backtest
        results = {}
        for ticker in tickers:
            results[ticker] = run_backtest(tickers=[ticker], start_date=start, end_date=end)
        return results
    
    def parallel_backtest(self, tickers: List[str], start: str, end: str,
                          batch_size: int = 10) -> List[Any]:
        if not self.initialized:
            self.start()
        
        if RAY_AVAILABLE:
            batches = [tickers[i:i+batch_size] for i in range(0, len(tickers), batch_size)]
            futures = [ray.remote(self._backtest_batch).remote(batch, start, end) for batch in batches]
            return ray.get(futures)
        else:
            return [self._backtest_batch(batch, start, end) for batch in 
                    [tickers[i:i+batch_size] for i in range(0, len(tickers), batch_size)]]
    
    @staticmethod
    def _train_single(config: Dict) -> Dict:
        from xgboost import XGBClassifier
        import numpy as np
        
        X = np.random.randn(1000, 10)
        y = (X[:, 0] > 0).astype(int)
        model = XGBClassifier(**config.get('params', {}), verbosity=0)
        model.fit(X, y)
        return {'accuracy': float((model.predict(X) == y).mean()), 'config': config}
    
    def distributed_hyperparameter_search(self, param_grid: List[Dict]) -> List[Dict]:
        if not self.initialized:
            self.start()
        
        if RAY_AVAILABLE:
            futures = [ray.remote(self._train_single).remote(config) for config in param_grid]
            return ray.get(futures)
        else:
            return [self._train_single(config) for config in param_grid]


class FeastFeatureStore:
    """Feast feature store integration for production feature serving."""
    
    def __init__(self, repo_path: str = None):
        self.repo_path = repo_path or str(OUTPUT_DIR / 'feast_repo')
        self.fs = None
    
    def setup_repo(self):
        """Create a Feast feature repository."""
        repo = Path(self.repo_path)
        repo.mkdir(parents=True, exist_ok=True)
        
        # feature_store.yaml
        (repo / 'feature_store.yaml').write_text(f"""
project: agente_financiero
registry: data/registry.db
provider: local
online_store:
    type: sqlite
    path: data/online_store.db
offline_store:
    type: file
""")
        
        # Feature definitions
        (repo / 'features.py').write_text("""
from datetime import timedelta
from feast import Entity, FeatureView, Field, FileSource
from feast.types import Float32, Int32, String

ticker = Entity(name='ticker', join_keys=['ticker'])

tecnicos_source = FileSource(
    path='Datos/analisis_tecnico.parquet',
    timestamp_field='timestamp',
    created_timestamp_column='created'
)

tecnicos_fv = FeatureView(
    name='indicadores_tecnicos',
    entities=[ticker],
    ttl=timedelta(days=1),
    schema=[
        Field(name='rsi_14', dtype=Float32),
        Field(name='macd_hist', dtype=Float32),
        Field(name='vol_ratio', dtype=Float32),
        Field(name='volatility_20d', dtype=Float32),
        Field(name='sma50_dist_pct', dtype=Float32),
        Field(name='atr_pct', dtype=Float32),
    ],
    source=tecnicos_source,
)

features_source = FileSource(
    path='Datos/auto_features.parquet',
    timestamp_field='timestamp',
)

features_fv = FeatureView(
    name='auto_features',
    entities=[ticker],
    ttl=timedelta(days=1),
    schema=[
        Field(name='ret_vol_interaction', dtype=Float32),
        Field(name='ret_vol_corr_20d', dtype=Float32),
        Field(name='ret_skew_20d', dtype=Float32),
        Field(name='vol_of_vol', dtype=Float32),
        Field(name='ret_sharpe_1m', dtype=Float32),
    ],
    source=features_source,
)
""")
        
        print(f'[Feast] Repo created at {repo}')
        return str(repo)
    
    def serve(self):
        """Start Feast feature server."""
        try:
            from feast import FeatureStore
            self.fs = FeatureStore(repo_path=self.repo_path)
            print('[Feast] Feature store ready')
            return self.fs
        except Exception as e:
            print(f'[Feast] Error: {e}')
            return None


class PrefectPipeline:
    """Prefect-based pipeline orchestration."""
    
    @staticmethod
    def full_pipeline_flow():
        """Complete pipeline (runs as Prefect DAG if available)."""
        if not PREFECT_AVAILABLE:
            print('[Prefect] Not available. pip install prefect')
            return {'status': 'prefect not available'}
        
        from prefect import flow, task
        
        @flow(name='agente_financiero_completo')
        def _inner_flow():
            @task
            def step_tecnicos():
                from analisis_tecnico import obtener_tecnicos
                return obtener_tecnicos()
            
            @task
            def step_social():
                import analisis_social
                return analisis_social.resultados
            
            @task
            def step_riesgo():
                import analisis_riesgo
                return analisis_riesgo.result
            
            @task
            def step_features(tecnicos):
                return {'status': 'ok', 'tecnicos_shape': len(tecnicos) if tecnicos else 0}
            
            @task
            def step_aprendizaje():
                from aprendizaje import calibracion
                return calibracion
            
            @task
            def step_ml():
                from walkforward_validator import generate_walk_forward_splits
                import pandas as pd
                dates = pd.date_range('2022-01-01', '2024-12-31', freq='B')
                splits = generate_walk_forward_splits(dates)
                return {'n_splits': len(splits)}
            
            @task
            def step_mas():
                from multi_agent_system import get_multi_agent_system
                mas = get_multi_agent_system()
                return {'agents': list(mas.agents.keys())}
            
            @task
            def step_backtest(results):
                from backtest_engine import run_backtest
                from datetime import datetime, timedelta
                return run_backtest(
                    start_date=(datetime.now() - timedelta(days=730)).strftime('%Y-%m-%d'),
                    end_date=datetime.now().strftime('%Y-%m-%d')
                )
            
            @task
            def generate_report(all_results):
                return {'status': 'completed', 'steps': len(all_results)}
            
            t1 = step_tecnicos()
            t2 = step_social()
            t3 = step_riesgo()
            t4 = step_features(t1)
            t5 = step_aprendizaje()
            t6 = step_ml()
            t7 = step_mas()
            t8 = step_backtest([t4, t5, t6, t7])
            report = generate_report([t1, t2, t3, t4, t5, t6, t7, t8])
            
            return report
        
        return _inner_flow()
    
    @staticmethod
    def run():
        if not PREFECT_AVAILABLE:
            print('[Prefect] Not available. pip install prefect')
            return None
        return PrefectPipeline.full_pipeline_flow()


class DockerInfrastructure:
    """Generate Docker and Docker Compose files."""
    
    @staticmethod
    def generate_dockerfile() -> str:
        content = """FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \\
    build-essential gcc \\
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["python", "circuito_completo.py", "--quick"]
"""
        path = Path(DATA_DIR) / 'Dockerfile'
        path.write_text(content)
        print(f'[Docker] Dockerfile generated at {path}')
        return str(path)
    
    @staticmethod
    def generate_docker_compose() -> str:
        content = """version: '3.8'

services:
  agente-financiero:
    build: .
    container_name: agente-financiero
    volumes:
      - ./Datos:/app/Datos
      - ./config:/app/config
    env_file:
      - .env
    environment:
      - GITHUB_WORKSPACE=/app
      - PARALLEL_WORKERS=8
      - YF_CACHE_TTL=3600
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "python", "-c", "import sys; sys.exit(0)"]
      interval: 30s
      timeout: 10s
      retries: 3

  mlflow:
    image: ghcr.io/mlflow/mlflow:v2.10.0
    container_name: mlflow-server
    ports:
      - "5000:5000"
    volumes:
      - ./mlruns:/mlruns
    command: mlflow server --host 0.0.0.0 --port 5000 --backend-store-uri sqlite:///mlruns/mlflow.db --default-artifact-root ./mlruns

  feast:
    build: .
    container_name: feast-server
    ports:
      - "6566:6566"
    volumes:
      - ./Datos:/app/Datos
    command: feast serve -h 0.0.0.0 -p 6566

  prefect:
    image: prefecthq/prefect:3-latest
    container_name: prefect-server
    ports:
      - "4200:4200"
    command: prefect server start --host 0.0.0.0

volumes:
  model_data:
"""
        path = Path(DATA_DIR) / 'docker-compose.yml'
        path.write_text(content)
        print(f'[Docker] docker-compose.yml generated at {path}')
        return str(path)
    
    @staticmethod
    def generate_kubernetes() -> str:
        content = """apiVersion: apps/v1
kind: Deployment
metadata:
  name: agente-financiero
spec:
  replicas: 1
  selector:
    matchLabels:
      app: agente-financiero
  template:
    metadata:
      labels:
        app: agente-financiero
    spec:
      containers:
      - name: main
        image: agente-financiero:latest
        env:
        - name: GITHUB_WORKSPACE
          value: "/app"
        - name: PARALLEL_WORKERS
          value: "16"
        volumeMounts:
        - name: data
          mountPath: /app/Datos
        resources:
          requests:
            memory: "2Gi"
            cpu: "1"
          limits:
            memory: "8Gi"
            cpu: "4"
      volumes:
      - name: data
        persistentVolumeClaim:
          claimName: agente-data-pvc
---
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: agente-data-pvc
spec:
  accessModes:
    - ReadWriteOnce
  resources:
    requests:
      storage: 50Gi
---
apiVersion: v1
kind: Service
metadata:
  name: agente-financiero-svc
spec:
  selector:
    app: agente-financiero
  ports:
  - port: 8000
    targetPort: 8000
  type: ClusterIP
"""
        path = Path(DATA_DIR) / 'k8s-deployment.yaml'
        path.write_text(content)
        print(f'[K8s] Kubernetes deployment generated at {path}')
        return str(path)


class CloudOrchestrator:
    """Manage cloud infrastructure setup."""
    
    @staticmethod
    def setup_all():
        DockerInfrastructure.generate_dockerfile()
        DockerInfrastructure.generate_docker_compose()
        DockerInfrastructure.generate_kubernetes()
        
        feast = FeastFeatureStore()
        feast.setup_repo()
        
        ray_cluster = RayCluster()
        ray_cluster.start()
        
        print('\n[Cloud] Infrastructure ready!')
        print('  Docker: docker-compose up -d')
        print('  MLflow: http://localhost:5000')
        print('  Prefect: http://localhost:4200')
        print('  Feast: http://localhost:6566')
        print('  K8s: kubectl apply -f k8s-deployment.yaml')
    
    @staticmethod
    def generate_requirements() -> str:
        extras = """
# Cloud & Distributed
ray>=2.9.0
prefect>=3.0.0
feast>=0.40.0
apache-beam>=2.55.0

# MLflow
mlflow>=2.10.0

# Monitoring
prometheus-client>=0.19.0
structlog>=24.1.0

# Kafka streaming
confluent-kafka>=2.3.0
"""
        path = Path(DATA_DIR) / 'requirements-cloud.txt'
        path.write_text(extras.strip())
        print(f'[Cloud] requirements-cloud.txt generated')
        return str(path)


if __name__ == '__main__':
    print('[Cloud] Setting up infrastructure...')
    
    CloudOrchestrator.setup_all()
    CloudOrchestrator.generate_requirements()
    
    if PREFECT_AVAILABLE:
        print('\n[Cloud] Running Prefect flow...')
        result = PrefectPipeline.run()
        print(f'  Result: {result}')
    
    if RAY_AVAILABLE:
        print('\n[Cloud] Testing Ray cluster...')
        ray_cluster = RayCluster(num_cpus=2)
        ray_cluster.start()
        
        param_grid = [{'params': {'n_estimators': n, 'max_depth': d}}
                     for n in [10, 50] for d in [3, 5]]
        results = ray_cluster.distributed_hyperparameter_search(param_grid)
        print(f'  Ray results: {len(results)} configs tested')
        ray_cluster.stop()