#!/usr/bin/env python3
"""ejecucion_algoritmica.py - Ejecucion algoritmica VWAP/TWAP/Implementation Shortfall."""
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import json, os
from pathlib import Path
from config.settings import get_setting
DATA_DIR = os.environ.get("GITHUB_WORKSPACE", ".")
OUTPUT_DIR = Path(DATA_DIR) / "Datos" / "ejecucion"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

@dataclass
class ExecutionOrder:
    ticker: str; side: str; total_shares: int; executed_shares: int
    price_limit: Optional[float]; start_time: datetime; algo: str
    status: str = 'pending'; fill_price: Optional[float] = None
    slippage_bps: Optional[float] = None

class VWAPExecutor:
    def __init__(self, target_pct=0.1, min_chunk=100):
        self.target_pct = target_pct
        self.min_chunk = min_chunk

    def execute(self, ticker: str, side: str, shares: int,
                price_data: pd.DataFrame, price_col='close', volume_col='volume') -> ExecutionOrder:
        order = ExecutionOrder(ticker, side, shares, 0, None, datetime.now(), 'VWAP')
        total_vol = price_data[volume_col].sum() if volume_col in price_data.columns else 1
        executed = 0; total_cost = 0.0
        for idx in range(len(price_data)):
            if executed >= shares: break
            row = price_data.iloc[idx]
            expected_vol = self.target_pct * total_vol / len(price_data)
            chunk_vol = max(self.min_chunk, int(expected_vol))
            chunk_shares = min(chunk_vol, shares - executed)
            if chunk_shares <= 0: continue
            price = float(row[price_col]) if price_col in price_data.columns else float(row.iloc[0])
            cost = chunk_shares * price
            total_cost += cost; executed += chunk_shares
        avg_price = total_cost / executed if executed > 0 else 0
        mid_price = float(price_data[price_col].iloc[0]) if price_col in price_data.columns else float(price_data.iloc[0, 0])
        slippage_bps = (avg_price / mid_price - 1) * 10000 if mid_price > 0 else 0
        order.executed_shares = executed; order.status = 'filled'
        order.fill_price = avg_price; order.slippage_bps = slippage_bps
        return order

class TWAPExecutor:
    def __init__(self, n_slices=10):
        self.n_slices = n_slices

    def execute(self, ticker: str, side: str, shares: int,
                price_data: pd.DataFrame, price_col='close') -> ExecutionOrder:
        order = ExecutionOrder(ticker, side, shares, 0, None, datetime.now(), 'TWAP')
        chunk = shares // self.n_slices
        executed = 0; total_cost = 0.0
        for i in range(self.n_slices):
            if executed >= shares: break
            idx = min(i * len(price_data) // self.n_slices, len(price_data) - 1)
            price = float(price_data.iloc[idx][price_col]) if price_col in price_data.columns else float(price_data.iloc[idx, 0])
            this_chunk = min(chunk, shares - executed)
            total_cost += this_chunk * price; executed += this_chunk
        avg_price = total_cost / executed if executed > 0 else 0
        mid = float(price_data.iloc[0][price_col]) if price_col in price_data.columns else float(price_data.iloc[0, 0])
        order.executed_shares = executed; order.status = 'filled'
        order.fill_price = avg_price
        order.slippage_bps = (avg_price / mid - 1) * 10000 if mid > 0 else 0
        return order

class ImplementationShortfallExecutor:
    def __init__(self, urgency=0.5, participation_rate=0.1):
        self.urgency = urgency; self.participation_rate = participation_rate

    def execute(self, ticker: str, side: str, shares: int,
                price_data: pd.DataFrame, price_col='close', volume_col='volume') -> ExecutionOrder:
        order = ExecutionOrder(ticker, side, shares, 0, None, datetime.now(), 'ImplementationShortfall')
        arrival_price = float(price_data.iloc[0][price_col]) if price_col in price_data.columns else float(price_data.iloc[0, 0])
        executed = 0; total_cost = 0.0; market_impact = 0.0
        for idx in range(len(price_data)):
            if executed >= shares: break
            row = price_data.iloc[idx]
            price = float(row[price_col]) if price_col in price_data.columns else float(row.iloc[0])
            vol = float(row[volume_col]) if volume_col in price_data.columns else 1000
            chunk = max(1, int(vol * self.participation_rate))
            chunk = min(chunk, shares - executed)
            impact = price * chunk * self.urgency * 0.0001
            fill_price = price + (impact / chunk) if side == 'buy' else price - (impact / chunk)
            total_cost += chunk * fill_price; market_impact += impact; executed += chunk
        avg_price = total_cost / executed if executed > 0 else 0
        shortfall = (avg_price - arrival_price) / arrival_price * 10000 if side == 'buy' else (arrival_price - avg_price) / arrival_price * 10000
        order.executed_shares = executed; order.status = 'filled'
        order.fill_price = avg_price; order.slippage_bps = shortfall
        return order

class ExecutionManager:
    def __init__(self):
        self.executors = {
            'VWAP': VWAPExecutor(), 'TWAP': TWAPExecutor(),
            'ImplementationShortfall': ImplementationShortfallExecutor()
        }
        self.orders: List[ExecutionOrder] = []

    def execute(self, algo: str, ticker: str, side: str, shares: int,
                price_data: pd.DataFrame) -> ExecutionOrder:
        exec_cls = self.executors.get(algo)
        if not exec_cls:
            raise ValueError(f"Algoritmo desconocido: {algo}")
        order = exec_cls.execute(ticker, side, shares, price_data)
        self.orders.append(order)
        return order

    def compare_algos(self, ticker: str, side: str, shares: int,
                      price_data: pd.DataFrame) -> pd.DataFrame:
        results = []
        for name in self.executors:
            order = self.execute(name, ticker, side, shares, price_data)
            results.append({'algo': name, 'fill_price': order.fill_price,
                'executed': order.executed_shares, 'slippage_bps': order.slippage_bps})
        return pd.DataFrame(results)

if __name__ == '__main__':
    np.random.seed(42)
    n = 100
    data = pd.DataFrame({
        'close': 100 + np.cumsum(np.random.randn(n) * 0.1),
        'volume': np.random.randint(1000, 100000, n),
    })
    em = ExecutionManager()
    df = em.compare_algos('NVDA', 'buy', 10000, data)
    print(df.to_string())
