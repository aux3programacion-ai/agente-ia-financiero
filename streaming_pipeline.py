#!/usr/bin/env python3
"""streaming_pipeline.py - Arquitectura event-driven para streaming en tiempo real."""
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import json, os, time, threading
from collections import deque
from pathlib import Path
from config.settings import get_setting
DATA_DIR = os.environ.get("GITHUB_WORKSPACE", ".")
OUTPUT_DIR = Path(DATA_DIR) / "Datos" / "streaming"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

@dataclass
class MarketEvent:
    ticker: str; price: float; volume: int; timestamp: datetime
    event_type: str = 'trade'; bid: Optional[float] = None; ask: Optional[float] = None

@dataclass
class SignalEvent:
    ticker: str; signal: str; confidence: float; timestamp: datetime
    source: str = 'analysis'

class DataBuffer:
    def __init__(self, maxlen=1000):
        self.buffers: Dict[str, deque] = {}
        self.maxlen = maxlen
    def append(self, ticker: str, event: MarketEvent):
        if ticker not in self.buffers:
            self.buffers[ticker] = deque(maxlen=self.maxlen)
        self.buffers[ticker].append(event)
    def get(self, ticker: str, n=10) -> List[MarketEvent]:
        if ticker not in self.buffers: return []
        return list(self.buffers[ticker])[-n:]
    def latest(self, ticker: str) -> Optional[MarketEvent]:
        if ticker not in self.buffers or not self.buffers[ticker]: return None
        return self.buffers[ticker][-1]

class StreamProcessor:
    def __init__(self):
        self.buffer = DataBuffer()
        self.features: Dict[str, pd.Series] = {}
        self.processors: Dict[str, Callable] = {}
    def register_processor(self, name: str, fn: Callable):
        self.processors[name] = fn
    def on_event(self, event: MarketEvent):
        self.buffer.append(event.ticker, event)
        for name, fn in self.processors.items():
            try:
                result = fn(event, self.buffer)
                if result:
                    self.features[f"{name}_{event.ticker}"] = result
            except Exception as e:
                pass
    def get_latest_features(self, ticker: str) -> Dict:
        return {k: v for k, v in self.features.items() if ticker in k}

class SimpleBacktestStream:
    def __init__(self, data: pd.DataFrame, ticker_col='ticker', price_col='close',
                 volume_col='volume', speed=1):
        self.data = data; self.ticker_col = ticker_col; self.price_col = price_col
        self.volume_col = volume_col; self.speed = speed
        self.processor = StreamProcessor()
        self.listeners: List[Callable] = []
        self.running = False
    def add_listener(self, fn: Callable):
        self.listeners.append(fn)
    def start(self):
        self.running = True
        for idx in range(len(self.data)):
            if not self.running: break
            row = self.data.iloc[idx]
            event = MarketEvent(
                ticker=str(row[self.ticker_col]) if self.ticker_col in self.data.columns else 'SYMBOL',
                price=float(row[self.price_col]), volume=int(row.get(self.volume_col, 0)),
                timestamp=datetime.now(), event_type='trade')
            self.processor.on_event(event)
            for listener in self.listeners:
                listener(event)
            time.sleep(1.0 / self.speed)
    def stop(self):
        self.running = False

class RealtimeAggregator:
    def __init__(self, window_sec=60):
        self.window_sec = window_sec
        self.windows: Dict[str, deque] = {}
    def add(self, ticker: str, price: float, volume: int):
        now = datetime.now()
        if ticker not in self.windows:
            self.windows[ticker] = deque()
        self.windows[ticker].append({'time': now, 'price': price, 'volume': volume})
        while self.windows[ticker] and (now - self.windows[ticker][0]['time']).total_seconds() > self.window_sec:
            self.windows[ticker].popleft()
    def vwap(self, ticker: str) -> Optional[float]:
        if ticker not in self.windows or not self.windows[ticker]: return None
        total_vol = sum(w['volume'] for w in self.windows[ticker])
        if total_vol == 0: return None
        return sum(w['price'] * w['volume'] for w in self.windows[ticker]) / total_vol
    def flow_imbalance(self, ticker: str) -> Optional[float]:
        if ticker not in self.windows or len(self.windows[ticker]) < 2: return None
        prices = [w['price'] for w in self.windows[ticker]]
        direction = np.diff(prices)
        buy_vol = sum(self.windows[ticker][i+1]['volume'] for i, d in enumerate(direction) if d > 0)
        sell_vol = sum(self.windows[ticker][i+1]['volume'] for i, d in enumerate(direction) if d < 0)
        total = buy_vol + sell_vol
        return (buy_vol - sell_vol) / total if total > 0 else 0.0

class StreamingPipeline:
    def __init__(self):
        self.processor = StreamProcessor()
        self.aggregator = RealtimeAggregator()
        self.stream = None
    def run_backtest(self, data: pd.DataFrame, speed=10) -> Dict:
        self.stream = SimpleBacktestStream(data, speed=speed)
        self.stream.add_listener(self._on_event)
        self.stream.start()
        return {'events_processed': len(data), 'status': 'completed'}
    def _on_event(self, event: MarketEvent):
        self.processor.on_event(event)
        self.aggregator.add(event.ticker, event.price, event.volume)

if __name__ == '__main__':
    print('[Streaming] Probando pipeline...')
    n = 100
    data = pd.DataFrame({
        'ticker': ['NVDA'] * n + ['AAPL'] * n,
        'close': 100 + np.cumsum(np.random.randn(n*2) * 0.5),
        'volume': np.random.randint(100, 10000, n*2),
    })
    pipe = StreamingPipeline()
    result = pipe.run_backtest(data, speed=100)
    print(f"Procesados: {result['events_processed']}")
    vwap = pipe.aggregator.vwap('NVDA')
    print(f"VWAP NVDA: {vwap:.2f}" if vwap else "Sin datos")
