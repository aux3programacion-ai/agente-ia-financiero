#!/usr/bin/env python3
"""
api_rest.py - API REST + programacion 24/7.
FastAPI endpoints, APScheduler tareas programadas, monitoreo.
"""
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
import json, os, time
from pathlib import Path

try:
    from fastapi import FastAPI, HTTPException, Query
    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False

try:
    from apscheduler.schedulers.background import BackgroundScheduler
    SCHEDULER_AVAILABLE = True
except ImportError:
    SCHEDULER_AVAILABLE = False

from config.settings import get_setting

DATA_DIR = os.environ.get('GITHUB_WORKSPACE', '.')
OUTPUT_DIR = Path(DATA_DIR) / 'Datos' / 'api'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

app = None
if FASTAPI_AVAILABLE:
    app = FastAPI(title='Agente Financiero API', version='3.0.0')


@dataclass
class TaskSchedule:
    id: str; name: str; interval_minutes: int
    last_run: Optional[str] = None; status: str = 'idle'


class TaskManager:
    def __init__(self):
        self.tasks: Dict[str, TaskSchedule] = {}
        self.scheduler = BackgroundScheduler() if SCHEDULER_AVAILABLE else None
        self.results: Dict[str, Any] = {}

    def register(self, task_id: str, name: str, interval: int, func, start=False):
        self.tasks[task_id] = TaskSchedule(task_id, name, interval)
        if self.scheduler and start:
            self.scheduler.add_job(
                func, 'interval', minutes=interval,
                id=task_id, name=name, replace_existing=True)

    def add_result(self, task_id: str, result: Any):
        self.results[task_id] = {
            'result': result, 'timestamp': datetime.now().isoformat()}

    def get_status(self) -> Dict:
        return {t.id: asdict(t) for t in self.tasks.values()}


task_manager = TaskManager()


class HealthMonitor:
    def __init__(self):
        self.checks: Dict[str, Dict] = {}
        self.start_time = datetime.now()

    def register_check(self, name: str, check_func):
        self.checks[name] = {'func': check_func, 'last': None, 'status': 'unknown'}

    def run_all(self) -> Dict:
        results = {}
        for name, check in self.checks.items():
            try:
                result = check['func']()
                check['last'] = datetime.now().isoformat()
                check['status'] = 'ok' if result else 'fail'
                results[name] = {'status': 'ok', 'detail': result}
            except Exception as e:
                check['status'] = 'error'
                results[name] = {'status': 'error', 'detail': str(e)}
        return results

    def get_uptime(self) -> str:
        delta = datetime.now() - self.start_time
        return str(delta).split('.')[0]


health = HealthMonitor()


if FASTAPI_AVAILABLE:
    @app.get('/health')
    def get_health():
        return {'status': 'ok', 'uptime': health.get_uptime(),
                'checks': health.run_all()}

    @app.get('/api/v1/status')
    def get_status():
        return {'app': 'Agente Financiero', 'version': '3.0.0',
                'uptime': health.get_uptime(),
                'tasks': task_manager.get_status()}

    @app.get('/api/v1/portafolio')
    def get_portfolio():
        dates = pd.date_range('2024-01-01', '2024-12-31', freq='B')
        p = 100 * np.exp(np.cumsum(np.random.randn(len(dates)) * 0.02 + 0.0005))
        b = 100 * np.exp(np.cumsum(np.random.randn(len(dates)) * 0.015 + 0.0003))
        return {'fechas': [str(d.date()) for d in dates],
                'portafolio': p.tolist(), 'benchmark': b.tolist()}

    @app.get('/api/v1/senales/{ticker}')
    def get_signals(ticker: str):
        return {'ticker': ticker.upper(), 'senal': 'COMPRA',
                'confianza': round(0.5 + np.random.random() * 0.4, 2),
                'timestamp': datetime.now().isoformat()}

    @app.get('/api/v1/riesgo')
    def get_risk():
        return {'var_95': -0.023, 'var_99': -0.041, 'vol_anual': 0.185,
                'sharpe': 1.42, 'max_drawdown': -0.087, 'beta': 1.05}

    @app.get('/api/v1/metricas')
    def get_metrics():
        return {'retorno_total': 0.187, 'retorno_anual': 0.094,
                'sharpe': 1.42, 'sortino': 1.85, 'calmar': 1.08,
                'win_rate': 0.54, 'trades': 342}

    @app.post('/api/v1/ejecutar')
    def execute_signal(ticker: str = Query(...), side: str = Query(...),
                       shares: int = Query(...)):
        return {'order_id': f'ORD_{int(time.time())}',
                'ticker': ticker, 'side': side, 'shares': shares,
                'status': 'submitted', 'timestamp': datetime.now().isoformat()}


class APIConfigGenerator:
    def generate_uvicorn_config(self) -> str:
        config = '''
api_rest:
  host: "0.0.0.0"
  port: 8000
  reload: false
  workers: 4
  log_level: "info"
  cors_origins: ["*"]
  rate_limit: 100
'''
        path = OUTPUT_DIR / 'api_config.yaml'
        path.write_text(config.strip(), encoding='utf-8')
        return str(path)

    def generate_systemd_service(self) -> str:
        service = '''
[Unit]
Description=Agente Financiero API
After=network.target

[Service]
Type=simple
User=agente
WorkingDirectory=/opt/agente-financiero
Environment=GITHUB_WORKSPACE=/opt/agente-financiero
ExecStart=/usr/bin/python -m uvicorn api_rest:app --host 0.0.0.0 --port 8000
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
'''
        path = OUTPUT_DIR / 'agente-financiero.service'
        path.write_text(service.strip(), encoding='utf-8')
        return str(path)

    def generate_dockerfile(self) -> str:
        content = 'FROM python:3.12-slim\n'
        content += 'WORKDIR /app\n'
        content += 'COPY requirements.txt .\n'
        content += 'RUN pip install fastapi uvicorn apscheduler prometheus-client\n'
        content += 'COPY . .\n'
        content += 'EXPOSE 8000\n'
        content += 'CMD ["uvicorn", "api_rest:app", "--host", "0.0.0.0", "--port", "8000"]\n'
        path = Path(DATA_DIR) / 'Dockerfile.api'
        path.write_text(content, encoding='utf-8')
        return str(path)


def scheduled_analisis():
    result = {'status': 'completed', 'timestamp': datetime.now().isoformat()}
    task_manager.add_result('analisis', result)
    return result


task_manager.register('analisis', 'Analisis automatico', 60, scheduled_analisis)

if __name__ == '__main__':
    cfg = APIConfigGenerator()
    cfg.generate_uvicorn_config()
    cfg.generate_systemd_service()
    cfg.generate_dockerfile()
    print(f'[API] Config generada. Health: {health.run_all()}')
    print(f'[API] Tasks: {task_manager.get_status()}')
