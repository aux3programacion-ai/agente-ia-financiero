#!/usr/bin/env python3
"""
api_rest.py - API REST + programacion 24/7.
FastAPI endpoints, APScheduler tareas programadas, monitoreo.
"""
import json
import os
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from logger import get_module_logger

log = get_module_logger('api_rest')

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


DATA_DIR = os.environ.get('GITHUB_WORKSPACE', '.')
OUTPUT_DIR = Path(DATA_DIR) / 'Datos' / 'api'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def _load_json(path: str) -> dict | None:
    p = Path(DATA_DIR) / 'Datos' / path
    if p.exists():
        try:
            return json.loads(p.read_text(encoding='utf-8'))
        except Exception as e:
            log.error(f"Error loading {path}: {e}")
    return None


app = None
if FASTAPI_AVAILABLE:
    app = FastAPI(title='Agente Financiero API', version='3.0.0')


@dataclass
class TaskSchedule:
    id: str; name: str; interval_minutes: int
    last_run: str | None = None; status: str = 'idle'


class TaskManager:
    def __init__(self):
        self.tasks: dict[str, TaskSchedule] = {}
        self.scheduler = BackgroundScheduler() if SCHEDULER_AVAILABLE else None
        self.results: dict[str, Any] = {}

    def register(self, task_id: str, name: str, interval: int, func, start=False):
        self.tasks[task_id] = TaskSchedule(task_id, name, interval)
        if self.scheduler and start:
            self.scheduler.add_job(
                func, 'interval', minutes=interval,
                id=task_id, name=name, replace_existing=True)

    def add_result(self, task_id: str, result: Any):
        self.results[task_id] = {
            'result': result, 'timestamp': datetime.now().isoformat()}

    def get_status(self) -> dict:
        return {t.id: asdict(t) for t in self.tasks.values()}


task_manager = TaskManager()


class HealthMonitor:
    def __init__(self):
        self.checks: dict[str, dict] = {}
        self.start_time = datetime.now()

    def register_check(self, name: str, check_func):
        self.checks[name] = {'func': check_func, 'last': None, 'status': 'unknown'}

    def run_all(self) -> dict:
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
                log.error(f"Health check {name} failed: {e}")
        return results

    def get_uptime(self) -> str:
        delta = datetime.now() - self.start_time
        return str(delta).split('.')[0]


health = HealthMonitor()


class PortfolioDataProvider:
    def get_portfolio(self) -> dict:
        data = _load_json('optimizacion_portafolio.json')
        if data:
            return data
        return {'error': 'No portfolio data available', 'tickers': {}}

    def get_signals(self, ticker: str) -> dict:
        ia_data = _load_json('analisis_ia.json')
        if ia_data and 'probabilidades' in ia_data:
            probs = ia_data['probabilidades']
            if ticker in probs and isinstance(probs[ticker], dict):
                p = probs[ticker]
                prob = p.get('probabilidad', 50)
                direction = 'COMPRA' if prob > 60 else ('VENTA' if prob < 40 else 'NEUTRAL')
                return {
                    'ticker': ticker.upper(),
                    'senal': direction,
                    'probabilidad': prob,
                    'confianza': p.get('confianza', 50),
                    'precio_objetivo': p.get('precio_objetivo_30d'),
                    'timestamp': datetime.now().isoformat()
                }
        return {'ticker': ticker.upper(), 'senal': 'SIN_DATOS', 'confianza': 0}

    def get_risk(self) -> dict:
        risk_data = _load_json('analisis_riesgo.json')
        if risk_data:
            return risk_data
        return {'error': 'No risk data available'}

    def get_metrics(self) -> dict:
        opt_data = _load_json('optimizacion_portafolio.json')
        xgb_data = _load_json('modelo_xgboost.json')
        metrics: dict[str, Any] = {}
        if xgb_data:
            metrics['wf_accuracy_mean'] = xgb_data.get('wf_accuracy_mean', 0)
            metrics['wf_accuracy_std'] = xgb_data.get('wf_accuracy_std', 0)
            metrics['train_accuracy'] = xgb_data.get('train_accuracy', 0)
            metrics['current_regime'] = xgb_data.get('current_regime', 'UNKNOWN')
            metrics['n_tickers'] = len(xgb_data.get('tickers', {}))
        if opt_data:
            for k in ['retorno_total', 'retorno_anual', 'sharpe', 'sortino', 'calmar', 'max_drawdown']:
                if k in opt_data:
                    metrics[k] = opt_data[k]
        if metrics:
            return metrics
        return {'error': 'No metrics data available'}


_portfolio_provider = PortfolioDataProvider()


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
        return _portfolio_provider.get_portfolio()

    @app.get('/api/v1/senales/{ticker}')
    def get_signals(ticker: str):
        return _portfolio_provider.get_signals(ticker.upper())

    @app.get('/api/v1/riesgo')
    def get_risk():
        return _portfolio_provider.get_risk()

    @app.get('/api/v1/metricas')
    def get_metrics():
        return _portfolio_provider.get_metrics()

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
    log.info("API config generated", checks=health.run_all())
    log.info("Tasks registered", tasks=task_manager.get_status())
