#!/usr/bin/env python3
"""
riesgos_tiempo_real.py - Gestion de riesgos en tiempo real.
Limites por estrategia/activo/mercado, circuit breakers,
monitoreo de concentracion, apalancamiento, VAR en vivo.
"""
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
import json, os, threading
from pathlib import Path

from config.settings import get_setting

DATA_DIR = os.environ.get('GITHUB_WORKSPACE', '.')
OUTPUT_DIR = Path(DATA_DIR) / 'Datos' / 'riesgos'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


@dataclass
class RiskLimit:
    nombre: str; tipo: str; limite: float; actual: float
    unidad: str = '%'; activo: bool = True; excedido: bool = False

    def check(self) -> bool:
        self.excedido = self.actual > self.limite if 'max' in self.tipo else self.actual < self.limite
        return self.excedido


@dataclass
class RiskAlert:
    timestamp: str; severity: str; category: str; mensaje: str
    valor: float; limite: float; suggested_action: str = ''


@dataclass
class CircuitBreakerEvent:
    timestamp: str; reason: str; action_taken: str
    affected_assets: List[str]; duration_minutes: int


class RealTimeVAR:
    def __init__(self, window=252, confidence=0.95):
        self.window = window
        self.confidence = confidence

    def compute(self, returns: np.ndarray, portfolio_value: float) -> Dict:
        if len(returns) < 10:
            return {'var_95': 0, 'var_99': 0, 'cvar_95': 0, 'method': 'insufficient_data'}
        historical_var_95 = np.percentile(returns, (1 - self.confidence) * 100)
        historical_var_99 = np.percentile(returns, 1)
        cvar = returns[returns <= historical_var_95].mean()
        parametric_var = np.mean(returns) - 1.645 * np.std(returns)
        return {
            'var_95': float(historical_var_95 * portfolio_value),
            'var_99': float(historical_var_99 * portfolio_value),
            'cvar_95': float(cvar * portfolio_value) if not np.isnan(cvar) else 0,
            'parametric_var_95': float(parametric_var * portfolio_value),
            'method': 'historical'} | {'portfolio_value': portfolio_value}


class LimitManager:
    def __init__(self):
        self.limits: Dict[str, RiskLimit] = {}
        self.alerts: List[RiskAlert] = []

    def add_limit(self, name: str, tipo: str, limite: float, unidad='%'):
        self.limits[name] = RiskLimit(nombre=name, tipo=tipo, limite=limite,
                                       actual=0.0, unidad=unidad)

    def update(self, name: str, value: float):
        if name in self.limits:
            self.limits[name].actual = value
            if self.limits[name].check():
                self.alerts.append(RiskAlert(
                    timestamp=datetime.now().isoformat(), severity='alta',
                    category=name, mensaje=f'{name} excedido: {value:.2f} > {self.limits[name].limite:.2f}',
                    valor=value, limite=self.limits[name].limite,
                    suggested_action=f'Reducir {name}'))

    def get_exceeded(self) -> List[RiskLimit]:
        return [l for l in self.limits.values() if l.excedido]


class CircuitBreaker:
    def __init__(self):
        self.events: List[CircuitBreakerEvent] = []
        self.active_breaks: Dict[str, datetime] = {}
        self.default_duration = 30

    def trigger(self, reason: str, assets: List[str],
                duration: Optional[int] = None) -> CircuitBreakerEvent:
        dur = duration or self.default_duration
        event = CircuitBreakerEvent(timestamp=datetime.now().isoformat(),
            reason=reason, action_taken='PAUSE_ALL',
            affected_assets=assets, duration_minutes=dur)
        self.events.append(event)
        for a in assets:
            self.active_breaks[a] = datetime.now() + timedelta(minutes=dur)
        return event

    def is_paused(self, asset: str) -> bool:
        if asset not in self.active_breaks:
            return False
        if datetime.now() > self.active_breaks[asset]:
            del self.active_breaks[asset]
            return False
        return True

    def get_active(self) -> List[str]:
        now = datetime.now()
        return [a for a, t in self.active_breaks.items() if now < t]


class RiskDashboard:
    def __init__(self):
        self.var = RealTimeVAR()
        self.limits = LimitManager()
        self.circuit = CircuitBreaker()
        self._setup_default_limits()

    def _setup_default_limits(self):
        for name, tipo, lim in [
            ('apalancamiento', 'max', 2.0), ('concentracion_single', 'max', 0.25),
            ('concentracion_sector', 'max', 0.40), ('drawdown', 'max', 0.20),
            ('var_95_pct', 'max', 0.03), ('exposure_cash', 'min', 0.05),
            ('max_single_pnl', 'max', 0.10), ('max_daily_loss', 'max', 0.05)]:
            self.limits.add_limit(name, tipo, lim)

    def evaluate(self, positions: Dict[str, float], prices: Dict[str, float],
                 returns: pd.DataFrame, capital: float) -> Dict:
        total_exposure = sum(abs(p) * prices.get(t, 0) for t, p in positions.items())
        leverage = total_exposure / max(capital, 1)

        self.limits.update('apalancamiento', leverage)
        for ticker, pos in positions.items():
            concentration = abs(pos * prices.get(ticker, 0)) / max(capital, 1)
            self.limits.update('concentracion_single', max(
                self.limits.limits.get('concentracion_single', RiskLimit('','',0,0)).actual,
                concentration))

        if len(returns) > 0:
            var_result = self.var.compute(returns.values.flatten(), capital)
            var_pct = var_result['var_95'] / max(capital, 1)
            self.limits.update('var_95_pct', abs(var_pct))
        else:
            var_result = {'var_95': 0, 'var_99': 0, 'cvar_95': 0}

        exceeded = self.limits.get_exceeded()
        if exceeded:
            for exc in exceeded:
                self.circuit.trigger(f'{exc.nombre} excedido: {exc.actual:.2f}',
                    list(positions.keys()))

        return {
            'leverage': float(leverage),
            'total_exposure': float(total_exposure),
            'capital': float(capital),
            'var': var_result,
            'limits_exceeded': [asdict(l) for l in exceeded],
            'circuit_breakers_active': self.circuit.get_active(),
            'alerts': [asdict(a) for a in self.limits.alerts[-10:]]}

    def report(self) -> str:
        report = '=== RISK DASHBOARD ===\n'
        report += f'Generated: {datetime.now().isoformat()}\n\n'
        for l in self.limits.limits.values():
            flag = 'EXCEDIDO' if l.excedido else 'OK'
            report += f'{l.nombre}: {l.actual:.4f} / {l.limite:.4f} [{flag}]\n'
        active = self.circuit.get_active()
        if active:
            report += f'\nCIRCUIT BREAKERS ACTIVE: {active}\n'
        report += f'\nAlerts: {len(self.limits.alerts)}\n'
        path = OUTPUT_DIR / 'risk_report.txt'
        path.write_text(report, encoding='utf-8')
        return str(path)


if __name__ == '__main__':
    np.random.seed(42)
    rd = RiskDashboard()
    positions = {'NVDA': 1000, 'AAPL': 500, 'MSFT': 750, 'GOOGL': 300}
    prices = {'NVDA': 150, 'AAPL': 180, 'MSFT': 350, 'GOOGL': 140}
    dates = pd.date_range('2024-01-01', '2024-12-31', freq='B')
    returns = pd.DataFrame({t: np.random.randn(len(dates)) * 0.02 for t in positions},
                           index=dates)
    result = rd.evaluate(positions, prices, returns, capital=500000)
    print(f"Leverage: {result['leverage']:.2f}")
    print(f"VAR95: ${result['var']['var_95']:,.0f}")
    print(f"Limits exceeded: {len(result['limits_exceeded'])}")
    print(f"Active breaks: {result['circuit_breakers_active']}")
    print(rd.report())
