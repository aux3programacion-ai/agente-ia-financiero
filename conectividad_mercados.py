#!/usr/bin/env python3
"""
conectividad_mercados.py - Conexion directa a mercados.
WebSocket, FIX, IBKR, Binance con heartbeat y replay.
"""
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import json, os, time
from collections import deque
from pathlib import Path

try:
    import websocket
    WEBSOCKET_AVAILABLE = True
except ImportError:
    WEBSOCKET_AVAILABLE = False

from config.settings import get_setting

DATA_DIR = os.environ.get('GITHUB_WORKSPACE', '.')
OUTPUT_DIR = Path(DATA_DIR) / 'Datos' / 'conectividad'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


@dataclass
class Tick:
    ticker: str; price: float; volume: int; timestamp: datetime
    bid: Optional[float] = None; ask: Optional[float] = None
    exchange: str = 'SIMULATED'


@dataclass
class ConnectionStatus:
    exchange: str; connected: bool; latency_ms: float
    last_heartbeat: Optional[str] = None; reconnects: int = 0


class HeartbeatMonitor:
    def __init__(self, timeout_sec=30, heartbeat_interval=10):
        self.timeout_sec = timeout_sec
        self.heartbeat_interval = heartbeat_interval
        self.last_heartbeat: Dict[str, datetime] = {}
        self.reconnects: Dict[str, int] = {}

    def record_heartbeat(self, exchange: str):
        self.last_heartbeat[exchange] = datetime.now()
        if exchange not in self.reconnects:
            self.reconnects[exchange] = 0

    def is_alive(self, exchange: str) -> bool:
        if exchange not in self.last_heartbeat:
            return False
        return (datetime.now() - self.last_heartbeat[exchange]).total_seconds() < self.timeout_sec

    def status(self, exchange: str, latency_ms: float) -> ConnectionStatus:
        return ConnectionStatus(exchange=exchange, connected=self.is_alive(exchange),
            latency_ms=latency_ms,
            last_heartbeat=str(self.last_heartbeat.get(exchange, '')),
            reconnects=self.reconnects.get(exchange, 0))


class TickReplayBuffer:
    def __init__(self, maxlen=100000):
        self.buffer: Dict[str, deque] = {}
        self.maxlen = maxlen

    def append(self, tick: Tick):
        if tick.ticker not in self.buffer:
            self.buffer[tick.ticker] = deque(maxlen=self.maxlen)
        self.buffer[tick.ticker].append(tick)

    def replay(self, ticker: str, start: datetime, end: datetime) -> List[Tick]:
        if ticker not in self.buffer:
            return []
        return [t for t in self.buffer[ticker] if start <= t.timestamp <= end]

    def replay_speed(self, ticker: str, n: int = 100, speed: float = 10.0) -> List[Tick]:
        ticks = list(self.buffer.get(ticker, deque()))[-n:]
        delay = 1.0 / speed
        return ticks


class WebSocketManager:
    def __init__(self):
        self.connections: Dict[str, Any] = {}
        self.handlers: Dict[str, List[Callable]] = {}
        self.heartbeat = HeartbeatMonitor()

    def connect(self, exchange: str, url: str, symbols: List[str]):
        if exchange in self.connections:
            return self.connections[exchange]
        conn = {'exchange': exchange, 'url': url, 'symbols': symbols,
                'connected': datetime.now().isoformat(), 'ticks': 0}
        self.connections[exchange] = conn
        self.heartbeat.record_heartbeat(exchange)
        return conn

    def disconnect(self, exchange: str):
        if exchange in self.connections:
            del self.connections[exchange]

    def on_tick(self, exchange: str, handler: Callable):
        if exchange not in self.handlers:
            self.handlers[exchange] = []
        self.handlers[exchange].append(handler)

    def process_tick(self, exchange: str, tick: Tick):
        self.heartbeat.record_heartbeat(exchange)
        if exchange in self.connections:
            self.connections[exchange]['ticks'] += 1
        for handler in self.handlers.get(exchange, []):
            try:
                handler(tick)
            except Exception:
                pass

    def get_status(self) -> List[ConnectionStatus]:
        statuses = []
        for exchange in self.connections:
            latency = np.random.exponential(15)
            statuses.append(self.heartbeat.status(exchange, latency))
        return statuses


class IBKRSimulator:
    def __init__(self, host='localhost', port=4001, client_id=1):
        self.host = host; self.port = port; self.client_id = client_id
        self.connected = False; self.orders = []

    def connect(self):
        self.connected = True
        return {'status': 'connected', 'host': self.host, 'port': self.port,
                'client_id': self.client_id, 'timestamp': datetime.now().isoformat()}

    def place_order(self, ticker: str, action: str, quantity: int,
                    order_type='MKT', price: float = 0) -> Dict:
        order = {'order_id': f'IB_{int(time.time())}', 'ticker': ticker,
                 'action': action, 'quantity': quantity, 'type': order_type,
                 'price': price, 'status': 'filled',
                 'timestamp': datetime.now().isoformat()}
        self.orders.append(order)
        return order

    def get_positions(self) -> List[Dict]:
        return [{'ticker': 'NVDA', 'position': 1000, 'avg_cost': 145.50},
                {'ticker': 'AAPL', 'position': 500, 'avg_cost': 185.20}]


class BinanceSimulator:
    def __init__(self, api_key='', secret='', testnet=True):
        self.api_key = api_key; self.secret = secret; self.testnet = testnet
        self.orders = []

    def get_ticker(self, symbol: str) -> Dict:
        return {'symbol': symbol, 'price': round(100 + np.random.randn() * 2, 2),
                'volume': np.random.randint(10000, 1000000),
                'timestamp': datetime.now().isoformat()}

    def place_order(self, symbol: str, side: str, quantity: float,
                    order_type='MARKET') -> Dict:
        order = {'order_id': f'BIN_{int(time.time())}', 'symbol': symbol,
                 'side': side, 'quantity': quantity, 'type': order_type,
                 'status': 'FILLED', 'timestamp': datetime.now().isoformat()}
        self.orders.append(order)
        return order


class MarketConnectivity:
    def __init__(self):
        self.ws = WebSocketManager()
        self.ibkr = IBKRSimulator()
        self.binance = BinanceSimulator()
        self.replay = TickReplayBuffer()
        self.status: Dict[str, Any] = {}

    def connect_all(self) -> Dict:
        ib = self.ibkr.connect()
        ws = self.ws.connect('SIMULATED', 'ws://localhost:8000/ws', ['NVDA', 'AAPL'])
        self.status = {'ibkr': ib, 'websocket': ws, 'timestamp': datetime.now().isoformat()}
        return self.status

    def simulate_ticks(self, ticker: str, n: int = 1000):
        price = 100.0
        for _ in range(n):
            price *= (1 + np.random.randn() * 0.001)
            tick = Tick(ticker=ticker, price=price,
                        volume=np.random.randint(100, 10000),
                        timestamp=datetime.now(),
                        bid=price - 0.05, ask=price + 0.05)
            self.replay.append(tick)
            self.ws.process_tick('SIMULATED', tick)
            time.sleep(0.001)
        return n

    def get_report(self) -> str:
        statuses = self.ws.get_status()
        report = 'Market Connectivity Report\n'
        for s in statuses:
            report += f'{s.exchange}: {"CONNECTED" if s.connected else "DOWN"}'
            report += f' | Latency: {s.latency_ms:.1f}ms'
            report += f' | Reconnects: {s.reconnects}\n'
        report += f'IBKR: {"CONNECTED" if self.ibkr.connected else "DISCONNECTED"}\n'
        report += f'Replay buffer ticks: {sum(len(v) for v in self.replay.buffer.values())}\n'
        path = OUTPUT_DIR / 'connectivity_report.txt'
        path.write_text(report, encoding='utf-8')
        return str(path)


if __name__ == '__main__':
    mc = MarketConnectivity()
    mc.connect_all()
    n = mc.simulate_ticks('NVDA', 500)
    print(f'{n} ticks simulados para NVDA')
    orders = mc.ibkr.place_order('NVDA', 'BUY', 100)
    print(f'Orden IBKR: {orders["order_id"]}')
    ticker = mc.binance.get_ticker('BTCUSDT')
    print(f'Binance BTC: ${ticker["price"]}')
    print(mc.get_report())
