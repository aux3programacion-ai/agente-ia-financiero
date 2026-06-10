#!/usr/bin/env python3
"""
backtest_alta_fidelidad.py - Simulador multi-activo de alta fidelidad.
Slippage realista, latencia, restricciones de capital/regulatorias, costos.
"""
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
import json, os
from pathlib import Path

from config.settings import get_setting

DATA_DIR = os.environ.get('GITHUB_WORKSPACE', '.')
OUTPUT_DIR = Path(DATA_DIR) / 'Datos' / 'backtest_hifi'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


@dataclass
class FillSimulation:
    timestamp: str; ticker: str; side: str; shares: int
    price: float; slippage_bps: float; latency_ms: float
    partial_fill: bool = False; fill_pct: float = 1.0


@dataclass
class ConstraintViolation:
    tipo: str; descripcion: str; severidad: str
    valor_actual: float; limite: float


class SlippageModel:
    def __init__(self, fixed_bps=0.5, market_impact_pct=0.1, spread_pct=0.05):
        self.fixed_bps = fixed_bps
        self.market_impact_pct = market_impact_pct
        self.spread_pct = spread_pct

    def compute_slippage(self, shares: int, price: float, volume: int,
                         side: str, volatility: float) -> float:
        participation = shares / max(volume, 1)
        market_impact = self.market_impact_pct * np.sqrt(participation) * volatility
        spread_cost = self.spread_pct / 2
        total_bps = self.fixed_bps + market_impact * 10000 + spread_cost * 10000
        return total_bps * (1 if side == 'buy' else -1)


class LatencySimulator:
    def __init__(self, base_ms=50, jitter_ms=20, network_drops=0.001):
        self.base_ms = base_ms
        self.jitter_ms = jitter_ms
        self.network_drops = network_drops

    def simulate(self) -> Tuple[float, bool]:
        latency = self.base_ms + np.random.exponential(self.jitter_ms)
        dropped = np.random.random() < self.network_drops
        return latency, dropped


class CapitalConstraints:
    def __init__(self, max_capital=1_000_000, max_leverage=2.0,
                 max_concentration=0.25, min_cash=0.05):
        self.max_capital = max_capital
        self.max_leverage = max_leverage
        self.max_concentration = max_concentration
        self.min_cash = min_cash
        self.capital = max_capital
        self.cash = max_capital
        self.positions: Dict[str, float] = {}
        self.violations: List[ConstraintViolation] = []

    def check_order(self, ticker: str, side: str, shares: int,
                    price: float) -> List[ConstraintViolation]:
        violations = []
        cost = shares * price
        if side == 'buy' and cost > self.cash:
            violations.append(ConstraintViolation('capital_insuficiente',
                f'Compra ${cost:.0f} > caja ${self.cash:.0f}', 'alta', cost, self.cash))
        total_exposure = sum(abs(v) for v in self.positions.values())
        new_exposure = total_exposure + cost if side == 'buy' else total_exposure - cost
        if new_exposure > self.capital * self.max_leverage:
            violations.append(ConstraintViolation('apalancamiento',
                f'Exposicion ${new_exposure:.0f} > limite ${self.capital*self.max_leverage:.0f}',
                'alta', new_exposure, self.capital * self.max_leverage))
        ticker_exposure = abs(shares * price) + abs(self.positions.get(ticker, 0))
        if ticker_exposure > self.capital * self.max_concentration:
            violations.append(ConstraintViolation('concentracion',
                f'{ticker}: ${ticker_exposure:.0f} > limite ${self.capital*self.max_concentration:.0f}',
                'media', ticker_exposure, self.capital * self.max_concentration))
        return violations

    def execute(self, ticker: str, side: str, shares: int, price: float):
        cost = shares * price
        if side == 'buy':
            self.cash -= cost
            self.positions[ticker] = self.positions.get(ticker, 0) + shares
        else:
            self.cash += cost * 0.999
            self.positions[ticker] = self.positions.get(ticker, 0) - shares
            if abs(self.positions.get(ticker, 0)) < 0.001:
                del self.positions[ticker]


class RegulatoryConstraints:
    def __init__(self, max_daily_turnover=0.3, min_holding_days=1,
                 max_single_ticker_pnl=0.15):
        self.max_daily_turnover = max_daily_turnover
        self.min_holding_days = min_holding_days
        self.max_single_ticker_pnl = max_single_ticker_pnl

    def check(self, ticker: str, side: str, shares: int, price: float,
              portfolio_value: float, positions: Dict) -> List[ConstraintViolation]:
        violations = []
        turnover = (shares * price) / max(portfolio_value, 1)
        if turnover > self.max_daily_turnover:
            violations.append(ConstraintViolation('facturacion',
                f'Rotacion {turnover:.1%} > limite {self.max_daily_turnover:.0%}',
                'baja', turnover, self.max_daily_turnover))
        return violations


class HighFidelityBacktest:
    def __init__(self, initial_capital=1_000_000):
        self.slippage = SlippageModel()
        self.latency = LatencySimulator()
        self.capital = CapitalConstraints(max_capital=initial_capital)
        self.regulatory = RegulatoryConstraints()
        self.fills: List[FillSimulation] = []
        self.violations: List[ConstraintViolation] = []

    def execute_order(self, ticker: str, side: str, shares: int,
                      price_data: pd.DataFrame, price_col='close',
                      volume_col='volume', vol_col='volatility') -> Dict:
        row = price_data.iloc[0]
        price = float(row[price_col])
        volume = int(row.get(volume_col, 100000))
        vol = float(row.get(vol_col, 0.02))

        latency_ms, dropped = self.latency.simulate()
        if dropped:
            return {'status': 'dropped', 'reason': 'network_drop'}

        slippage = self.slippage.compute_slippage(shares, price, volume, side, vol)
        fill_price = price * (1 + slippage / 10000)

        cap_violations = self.capital.check_order(ticker, side, shares, fill_price)
        reg_violations = self.regulatory.check(ticker, side, shares, fill_price,
                                                 self.capital.capital, self.capital.positions)
        all_violations = cap_violations + reg_violations
        self.violations.extend(all_violations)

        severe = [v for v in all_violations if v.severidad == 'alta']
        if severe:
            return {'status': 'rejected', 'reason': severe[0].descripcion,
                    'violations': [asdict(v) for v in severe]}

        self.capital.execute(ticker, side, shares, fill_price)

        fill = FillSimulation(
            timestamp=datetime.now().isoformat(), ticker=ticker, side=side,
            shares=shares, price=fill_price, slippage_bps=slippage,
            latency_ms=latency_ms)
        self.fills.append(fill)

        return {'status': 'filled', 'fill_price': fill_price,
                'slippage_bps': slippage, 'latency_ms': latency_ms,
                'shares': shares, 'cost': shares * fill_price}

    def run_backtest(self, signals: pd.DataFrame, price_data: pd.DataFrame,
                     ticker_col='ticker', signal_col='signal',
                     shares_col='shares') -> Dict:
        results = []
        for idx in range(len(signals)):
            sig = signals.iloc[idx]
            ticker = sig[ticker_col]
            side = sig[signal_col]
            shares = int(sig[shares_col])
            ticker_data = price_data[price_data.index == ticker] if ticker in price_data.index else price_data
            result = self.execute_order(ticker, side, shares, ticker_data)
            results.append(result)
        return {'n_orders': len(signals), 'filled': sum(1 for r in results if r['status'] == 'filled'),
                'rejected': sum(1 for r in results if r['status'] == 'rejected'),
                'dropped': sum(1 for r in results if r['status'] == 'dropped'),
                'total_cost': sum(r.get('cost', 0) for r in results),
                'avg_slippage': np.mean([r.get('slippage_bps', 0) for r in results if 'slippage_bps' in r]),
                'capital_remaining': self.capital.capital,
                'cash_remaining': self.capital.cash,
                'violations': len(self.violations)}

    def report(self) -> str:
        report = f"Backtest Hi-Fi Report\n"
        report += f"Capital inicial: ${self.capital.max_capital:,.0f}\n"
        report += f"Capital final: ${self.capital.capital + self.capital.cash:,.0f}\n"
        report += f"Violaciones: {len(self.violations)}\n"
        report += f"Fills: {len(self.fills)}\n"
        path = OUTPUT_DIR / 'backtest_hifi_report.txt'
        path.write_text(report, encoding='utf-8')
        return str(path)


if __name__ == '__main__':
    np.random.seed(42)
    n = 100
    data = pd.DataFrame({'close': 100 + np.cumsum(np.random.randn(n) * 0.1),
        'volume': np.random.randint(10000, 1000000, n),
        'volatility': np.random.rand(n) * 0.02 + 0.01})
    signals = pd.DataFrame({'ticker': ['NVDA'] * 20, 'signal': ['buy'] * 20,
        'shares': np.random.randint(100, 1000, 20)})
    bt = HighFidelityBacktest()
    r = bt.run_backtest(signals, data)
    print(f"Filled: {r['filled']}/{r['n_orders']}, Slippage: {r['avg_slippage']:.2f} bps")
    print(bt.report())
