#!/usr/bin/env python3
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
