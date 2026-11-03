#!/usr/bin/env python3
"""Script auxiliar para escribir modulos del sistema."""
import os

BASE = r'C:\Users\LAZARO LANDERO\Claro drive\AGENTE FINANCIERO'

def write_cppi():
    content = r'''#!/usr/bin/env python3
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
'''
    with open(os.path.join(BASE, 'cppi.py'), 'w', encoding='utf-8') as f:
        f.write(content)
    print('CPPI done')

def write_genetic():
    content = r'''#!/usr/bin/env python3
"""genetic_strategies.py - Estrategias evolutivas con programacion genetica."""
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass, field
from enum import Enum
import json, os, random, copy
from pathlib import Path
from config.settings import get_setting
DATA_DIR = os.environ.get("GITHUB_WORKSPACE", ".")
OUTPUT_DIR = Path(DATA_DIR) / "Datos" / "genetic"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

class NodeType(Enum):
    OPERATOR = 1; INDICATOR = 2; CONSTANT = 3; VARIABLE = 4

@dataclass
class Node:
    type: NodeType; value: str
    left: Optional['Node'] = None; right: Optional['Node'] = None

OPERATORS = ['+', '-', '*', '/', '>', '<']
INDICATORS = ['rsi_14', 'macd_hist', 'vol_ratio', 'volatility_20d',
    'sma50_dist_pct', 'atr_pct', 'returns_5d', 'returns_20d',
    'volume', 'close', 'sma_20', 'sma_50', 'bb_width']

def random_node(max_depth=3) -> Node:
    if max_depth <= 0:
        if random.random() < 0.5:
            return Node(NodeType.INDICATOR, random.choice(INDICATORS))
        return Node(NodeType.CONSTANT, str(round(random.uniform(0.1, 5.0), 2)))
    t = random.choice(['op', 'indicator', 'constant'])
    if t == 'op':
        return Node(NodeType.OPERATOR, random.choice(OPERATORS),
            random_node(max_depth - 1), random_node(max_depth - 1))
    elif t == 'indicator':
        return Node(NodeType.INDICATOR, random.choice(INDICATORS))
    return Node(NodeType.CONSTANT, str(round(random.uniform(0.1, 5.0), 2)))

def evaluate_tree(node: Node, data: pd.Series) -> float:
    if node.type == NodeType.CONSTANT: return float(node.value)
    if node.type == NodeType.INDICATOR: return float(data.get(node.value, 0))
    left = evaluate_tree(node.left, data)
    right = evaluate_tree(node.right, data)
    if node.value == '+': return left + right
    if node.value == '-': return left - right
    if node.value == '*': return left * right
    if node.value == '/': return left / (right + 1e-8)
    if node.value == '>': return 1.0 if left > right else 0.0
    if node.value == '<': return 1.0 if left < right else 0.0
    return 0.0

@dataclass
class Strategy:
    name: str; entry: Node; exit_node: Optional[Node] = None
    fitness: float = 0.0; params: Dict = field(default_factory=dict)

class GeneticStrategyOptimizer:
    def __init__(self, pop_size=50, generations=20, mutation_rate=0.2,
                 crossover_rate=0.6, max_depth=4):
        self.pop_size = pop_size; self.generations = generations
        self.mutation_rate = mutation_rate; self.crossover_rate = crossover_rate
        self.max_depth = max_depth; self.population: List[Strategy] = []
        self.best: Optional[Strategy] = None; self.history = []

    def _random_strategy(self, i):
        return Strategy(f"strat_{i}", random_node(self.max_depth))

    def evaluate(self, strategy: Strategy, data: pd.DataFrame, target='forward_return') -> float:
        signals = []
        for idx in range(len(data)):
            sig = evaluate_tree(strategy.entry, data.iloc[idx])
            signals.append(sig)
        signals = np.array(signals)
        if signals.std() == 0: return -1.0
        fwd = data[target].values[:len(signals)]
        sr = signals * fwd
        sharpe = np.sqrt(252) * sr.mean() / (sr.std() + 1e-8)
        wr = (sr > 0).mean()
        return sharpe * 0.6 + wr * 0.4

    def _tournament(self, fitness, k=3):
        idxs = random.sample(range(len(fitness)), min(k, len(fitness)))
        return max(idxs, key=lambda i: fitness[i])

    def _crossover(self, s1, s2):
        child = copy.deepcopy(s1)
        if child.entry.left and s2.entry.left:
            child.entry.left.value = s2.entry.left.value
            child.entry.left.type = s2.entry.left.type
        return Strategy(f"child_{random.randint(0,9999)}", child.entry)

    def _mutate(self, s):
        m = copy.deepcopy(s)
        def _mut(n):
            if random.random() < self.mutation_rate:
                if n.type == NodeType.CONSTANT:
                    n.value = str(round(random.uniform(0.1, 5.0), 2))
                elif n.type == NodeType.INDICATOR:
                    n.value = random.choice(INDICATORS)
                elif n.type == NodeType.OPERATOR:
                    n.value = random.choice(OPERATORS)
            if n.left: _mut(n.left)
            if n.right: _mut(n.right)
        _mut(m.entry)
        return m

    def run(self, data: pd.DataFrame, target='forward_return') -> Dict:
        self.population = [self._random_strategy(i) for i in range(self.pop_size)]
        for gen in range(self.generations):
            fitness = [self.evaluate(s, data, target) for s in self.population]
            best_idx = int(np.argmax(fitness))
            self.best = copy.deepcopy(self.population[best_idx])
            self.best.fitness = fitness[best_idx]
            self.history.append({'gen': gen, 'best_fitness': float(fitness[best_idx]),
                'mean_fitness': float(np.mean(fitness))})
            new_pop = [self.best]
            while len(new_pop) < self.pop_size:
                if random.random() < self.crossover_rate:
                    i1 = self._tournament(fitness); i2 = self._tournament(fitness)
                    child = self._crossover(self.population[i1], self.population[i2])
                else:
                    i = self._tournament(fitness)
                    child = self._mutate(self.population[i])
                new_pop.append(child)
            self.population = new_pop[:self.pop_size]
            if (gen+1) % 5 == 0:
                print(f'[Genetic] Gen {gen+1}, Best: {self.history[-1]["best_fitness"]:.4f}')
        self.best.fitness = self.evaluate(self.best, data, target)
        result = {'best_fitness': float(self.best.fitness),
            'generations': self.generations, 'pop_size': self.pop_size,
            'history': self.history}
        path = OUTPUT_DIR / 'genetic_result.json'
        path.write_text(json.dumps(result, indent=2, default=str), encoding='utf-8')
        return result

if __name__ == '__main__':
    np.random.seed(42); random.seed(42)
    n = 500
    data = pd.DataFrame({k: np.random.randn(n) for k in INDICATORS})
    data['forward_return'] = np.random.randn(n) * 0.02 + 0.0003
    opt = GeneticStrategyOptimizer(pop_size=20, generations=10)
    r = opt.run(data)
    print(f"Best fitness: {r['best_fitness']:.4f}")
'''
    with open(os.path.join(BASE, 'genetic_strategies.py'), 'w', encoding='utf-8') as f:
        f.write(content)
    print('Genetic done')

def write_dashboard():
    # Build content with careful quoting
    lines = []
    lines.append('#!/usr/bin/env python3')
    lines.append('"""dashboard_app.py - Dashboard interactivo con Streamlit."""')
    lines.append('import numpy as np')
    lines.append('import pandas as pd')
    lines.append('from typing import Dict, Any, List')
    lines.append('from datetime import datetime')
    lines.append('import json, os')
    lines.append('from pathlib import Path')
    lines.append('from config.settings import get_setting')
    lines.append('DATA_DIR = os.environ.get("GITHUB_WORKSPACE", ".")')
    lines.append('OUTPUT_DIR = Path(DATA_DIR) / "Datos" / "dashboard"')
    lines.append('OUTPUT_DIR.mkdir(parents=True, exist_ok=True)')
    lines.append('')
    lines.append('class DashboardDataProvider:')
    lines.append('    def load_portfolio_data(self) -> pd.DataFrame:')
    lines.append('        path = Path(DATA_DIR) / "Datos" / "backtest_results.parquet"')
    lines.append('        if path.exists(): return pd.read_parquet(path)')
    lines.append('        dates = pd.date_range("2023-01-01", "2024-12-31", freq="B")')
    lines.append('        n = len(dates)')
    lines.append('        p = 100 * np.exp(np.cumsum(np.random.randn(n) * 0.02 + 0.0005))')
    lines.append('        b = 100 * np.exp(np.cumsum(np.random.randn(n) * 0.015 + 0.0003))')
    lines.append('        df = pd.DataFrame({"fecha": dates, "portafolio": p, "benchmark": b})')
    lines.append('        df["retorno"] = df["portafolio"].pct_change()')
    lines.append('        df["retorno_bm"] = df["benchmark"].pct_change()')
    lines.append('        return df')
    lines.append('    def load_signals_data(self) -> pd.DataFrame:')
    lines.append('        dates = pd.date_range("2024-01-01", "2024-12-31", freq="B")')
    lines.append('        return pd.DataFrame({"fecha": dates,')
    lines.append('            "NVDA": np.random.randn(len(dates))*0.5+0.2,')
    lines.append('            "AAPL": np.random.randn(len(dates))*0.5-0.1,')
    lines.append('            "MSFT": np.random.randn(len(dates))*0.5+0.1})')
    lines.append('    def load_risk_data(self) -> Dict:')
    lines.append('        return {"var_95": -0.023, "var_99": -0.041, "vol_anual": 0.185,')
    lines.append('            "sharpe": 1.42, "max_drawdown": -0.087, "beta": 1.05}')
    lines.append('    def load_agent_performance(self) -> pd.DataFrame:')
    lines.append('        return pd.DataFrame({"agente": ["tecnico","fundamental","macro","sentimiento","riesgo"],')
    lines.append('            "precision": [0.62, 0.55, 0.58, 0.61, 0.53],')
    lines.append('            "peso": [1.2, 0.9, 1.1, 1.0, 0.8]})')
    lines.append('    def load_performance_metrics(self) -> Dict:')
    lines.append('        return {"retorno_total": 0.187, "retorno_anual": 0.094, "volatilidad": 0.185,')
    lines.append('            "sharpe": 1.42, "sortino": 1.85, "calmar": 1.08, "win_rate": 0.54, "trades": 342}')
    lines.append('')
    lines.append('')
    lines.append('class DashboardGenerator:')
    lines.append('    def __init__(self):')
    lines.append('        self.dp = DashboardDataProvider()')
    lines.append('    def generate_summary(self) -> Dict:')
    lines.append('        perf = self.dp.load_performance_metrics()')
    lines.append('        risk = self.dp.load_risk_data()')
    lines.append('        return {"fecha": datetime.now().isoformat(), "rendimiento": perf, "riesgo": risk}')
    lines.append('    def generate_html_report(self) -> str:')
    lines.append('        s = self.generate_summary()')
    lines.append('        rend = s.get("rendimiento", {})')
    lines.append('        riesgo = s.get("riesgo", {})')
    lines.append('        html = "<html><body>"')
    lines.append('        html += "<h1>Dashboard Financiero</h1>"')
    lines.append('        fecha = s.get("fecha", "")')
    lines.append('        tot_ret = rend.get("retorno_total", 0)')
    lines.append('        shrp = rend.get("sharpe", 0)')
    lines.append('        v95 = riesgo.get("var_95", 0)')
    lines.append('        html = "<html><body>"')
    lines.append('        html += f"<p>Generado: {fecha}</p>"')
    lines.append('        html += f"<p>Retorno: {tot_ret*100:.1f}%</p>"')
    lines.append('        html += f"<p>Sharpe: {shrp:.2f}</p>"')
    lines.append('        html += f"<p>VaR95: {v95*100:.1f}%</p>"')
    lines.append('        html += "</body></html>"')
    lines.append('        path = OUTPUT_DIR / "dashboard.html"')
    lines.append('        path.write_text(html, encoding="utf-8")')
    lines.append('        return str(path)')
    lines.append('')
    lines.append('')
    lines.append('class DashboardAPI:')
    lines.append('    def __init__(self):')
    lines.append('        self.dp = DashboardDataProvider()')
    lines.append('        self.gen = DashboardGenerator()')
    lines.append('    def get_summary(self) -> Dict: return self.gen.generate_summary()')
    lines.append('    def get_portfolio_data(self) -> pd.DataFrame: return self.dp.load_portfolio_data()')
    lines.append('    def get_risk_data(self) -> Dict: return self.dp.load_risk_data()')
    lines.append('    def get_agent_data(self) -> pd.DataFrame: return self.dp.load_agent_performance()')
    lines.append('    def generate_report(self) -> str: return self.gen.generate_html_report()')
    lines.append('')
    lines.append('')
    lines.append('if __name__ == "__main__":')
    lines.append('    api = DashboardAPI()')
    lines.append('    s = api.get_summary()')
    lines.append('    tr = s["rendimiento"]["retorno_total"]')
    lines.append('    print(f"Retorno: {tr*100:.1f}%")')
    lines.append('    r = api.generate_report()')
    lines.append('    print(f"Reporte: {r}")')
    content = '\n'.join(lines)
    with open(os.path.join(BASE, 'dashboard_app.py'), 'w', encoding='utf-8') as f:
        f.write(content)
    print('Dashboard done')

def write_streaming():
    content = r'''#!/usr/bin/env python3
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
'''
    with open(os.path.join(BASE, 'streaming_pipeline.py'), 'w', encoding='utf-8') as f:
        f.write(content)
    print('Streaming done')

def write_ejecucion():
    content = r'''#!/usr/bin/env python3
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
'''
    with open(os.path.join(BASE, 'ejecucion_algoritmica.py'), 'w', encoding='utf-8') as f:
        f.write(content)
    print('Ejecucion done')

if __name__ == '__main__':
    write_cppi()
    write_genetic()
    write_dashboard()
    write_streaming()
    write_ejecucion()
