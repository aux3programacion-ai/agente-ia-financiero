#!/usr/bin/env python3
"""cppi.py - Seguros de portafolio: CPPI y OBPI."""
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
import json, os
from pathlib import Path
from datetime import datetime
from config.settings import get_setting
DATA_DIR = os.environ.get("GITHUB_WORKSPACE", ".")
OUTPUT_DIR = Path(DATA_DIR) / "Datos" / "cppi"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

@dataclass
class CPPIResult:
    fecha: str; valor_portafolio: float; valor_riesgo: float
    valor_seguro: float; floor: float; cusion: float; exposicion: float

class CPPI:
    def __init__(self, capital_inicial=100000, floor_pct=0.85, multiplier=3.0, rebalance_freq=5):
        self.capital_inicial = capital_inicial
        self.floor_pct = floor_pct
        self.multiplier = multiplier
        self.rebalance_freq = rebalance_freq
        self.portfolio_value = capital_inicial
        self.risk_value = capital_inicial * 0.5
        self.safe_value = capital_inicial * 0.5
        self.floor = capital_inicial * floor_pct
        self.history = []

    def rebalance(self, risk_return, safe_return):
        self.risk_value *= (1 + risk_return)
        self.safe_value *= (1 + safe_return)
        self.portfolio_value = self.risk_value + self.safe_value
        self.floor *= (1 + 0.02 / 252)
        cushion = self.portfolio_value - self.floor
        target_risk = min(self.multiplier * cushion, self.portfolio_value)
        target_risk = max(0, target_risk)
        target_safe = self.portfolio_value - target_risk
        self.risk_value = target_risk
        self.safe_value = target_safe
        rec = CPPIResult(
            datetime.now().isoformat(),
            float(self.portfolio_value), float(self.risk_value),
            float(self.safe_value), float(self.floor),
            float(cushion), float(target_risk / self.portfolio_value if self.portfolio_value > 0 else 0)
        )
        self.history.append(rec)
        return rec

    def run_simulation(self, n_days=252, risk_vol=0.20, risk_mu=0.08, safe_rate=0.03):
        np.random.seed(42)
        for i in range(n_days):
            r_r = np.random.normal(risk_mu / 252, risk_vol / np.sqrt(252))
            r_s = safe_rate / 252
            self.rebalance(r_r, r_s)
        return pd.DataFrame([{'fecha': h.fecha, 'portafolio': h.valor_portafolio,
            'riesgo': h.valor_riesgo, 'seguro': h.valor_seguro,
            'floor': h.floor, 'cushion': h.cusion} for h in self.history])

class OBPI:
    def __init__(self, capital=100000, floor_pct=0.90, T=1.0, r=0.05, sigma=0.20):
        self.capital = capital
        self.floor = capital * floor_pct
        self.T = T; self.r = r; self.sigma = sigma

    def compute_allocation(self, S=100):
        from scipy.stats import norm
        d1 = (np.log(S / self.floor) + (self.r + 0.5 * self.sigma ** 2) * self.T) / (self.sigma * np.sqrt(self.T))
        delta = norm.cdf(d1)
        stocks = delta * self.capital / S
        bonds = self.capital - stocks * S
        return {'stocks': float(stocks), 'bonds': float(bonds), 'delta': float(delta),
            'stocks_value': float(stocks * S), 'bond_value': float(bonds)}

class PortfolioInsurance:
    def __init__(self):
        self.cppi = None
        self.obpi = None

    def compare_strategies(self, capital=100000, n_days=252):
        self.cppi = CPPI(capital_inicial=capital)
        cppi_result = self.cppi.run_simulation(n_days=n_days)
        final = cppi_result.iloc[-1] if len(cppi_result) > 0 else None
        return {'cppi_final': float(final['portafolio']) if final is not None else capital,
            'cppi_floor': float(final['floor']) if final is not None else capital * 0.85,
            'n_days': n_days}

if __name__ == '__main__':
    pi = PortfolioInsurance()
    r = pi.compare_strategies()
    print(f"CPPI final: ${r['cppi_final']:.2f}")
