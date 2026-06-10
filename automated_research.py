#!/usr/bin/env python3
"""
automated_research.py - Automated alpha research and hypothesis generation.
Alpha factory with combinatorial alpha expressions, hypothesis generator
using LLM-style reasoning, and signal decay analysis.
"""
import numpy as np
import pandas as pd
from itertools import combinations
from typing import Dict, Any, List, Callable, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path

from config.settings import get_setting

DATA_DIR = os.environ.get('GITHUB_WORKSPACE', '.')
OUTPUT_DIR = Path(DATA_DIR) / 'Datos' / 'automated_research'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ─── Alpha Primitives ─────────────────────────────────────────────────────

def returns(close: pd.Series, period: int = 1) -> pd.Series:
    return close.pct_change(period)

def log_returns(close: pd.Series) -> pd.Series:
    return np.log(close / close.shift(1))

def sma(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window).mean()

def ema(series: pd.Series, window: int) -> pd.Series:
    return series.ewm(span=window, adjust=False).mean()

def rank(series: pd.Series) -> pd.Series:
    return series.rank(pct=True)

def stddev(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window).std()

def correlation(x: pd.Series, y: pd.Series, window: int) -> pd.Series:
    return x.rolling(window).corr(y)

def covariance(x: pd.Series, y: pd.Series, window: int) -> pd.Series:
    return x.rolling(window).cov(y)

def zscore(series: pd.Series, window: int) -> pd.Series:
    return (series - series.rolling(window).mean()) / series.rolling(window).std()

def max_series(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window).max()

def min_series(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window).min()

def delta(series: pd.Series, period: int) -> pd.Series:
    return series - series.shift(period)

def ts_sum(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window).sum()

def decay_linear(series: pd.Series, window: int) -> pd.Series:
    weights = np.arange(1, window + 1) / (window * (window + 1) / 2)
    return series.rolling(window).apply(
        lambda x: np.dot(x, weights), raw=True)

# ─── Alpha Factory ────────────────────────────────────────────────────────

ALPHA_PRIMITIVES = {
    'returns': returns,
    'log_returns': log_returns,
    'sma': lambda s: sma(s, 20),
    'ema': lambda s: ema(s, 20),
    'rank': rank,
    'stddev': lambda s: stddev(s, 20),
    'zscore': lambda s: zscore(s, 20),
    'max': lambda s: max_series(s, 20),
    'min': lambda s: min_series(s, 20),
    'delta': lambda s: delta(s, 5),
    'ts_sum': lambda s: ts_sum(s, 10),
    'decay': lambda s: decay_linear(s, 10),
}

OPERATORS = {
    '+': lambda a, b: a + b,
    '-': lambda a, b: a - b,
    '*': lambda a, b: a * b,
    '/': lambda a, b: a / b.replace(0, np.nan),
    'rank': lambda a, _: rank(a),
    'zscore': lambda a, _: zscore(a, 20),
    'abs': lambda a, _: a.abs(),
    'neg': lambda a, _: -a,
    'square': lambda a, _: a ** 2,
    'sqrt': lambda a, _: np.sqrt(a.abs()),
    'sign': lambda a, _: np.sign(a),
}


@dataclass
class AlphaExpression:
    name: str
    expression: str
    primitive: str
    operator: str = '+'
    secondary: Optional[str] = None
    params: Dict[str, Any] = field(default_factory=dict)
    
    def compute(self, data: pd.DataFrame) -> pd.Series:
        p1 = ALPHA_PRIMITIVES.get(self.primitive, lambda s: s)
        s1 = p1(data['Close'] if 'Close' in data else data.iloc[:, 0])
        
        if self.secondary and self.operator in OPERATORS:
            p2 = ALPHA_PRIMITIVES.get(self.secondary, lambda s: s)
            s2 = p2(data['Close'] if 'Close' in data else data.iloc[:, 0])
            op = OPERATORS.get(self.operator, lambda a, _: a)
            return op(s1, s2)
        
        op = OPERATORS.get(self.operator, lambda a, _: a)
        return op(s1, None)


class AlphaFactory:
    """Combinatorial alpha expression generator."""
    
    def __init__(self):
        self.primitives = list(ALPHA_PRIMITIVES.keys())
        self.operators = ['+', '-', '*', '/']
        self.generated: Dict[str, AlphaExpression] = {}
    
    def generate_random(self, n: int = 100) -> List[AlphaExpression]:
        alphas = []
        for i in range(n):
            p1, p2 = np.random.choice(self.primitives, 2, replace=False)
            op = np.random.choice(self.operators)
            name = f'alpha_{len(self.generated) + 1}'
            expr = AlphaExpression(
                name=name,
                expression=f'{p1} {op} {p2}',
                primitive=p1, operator=op, secondary=p2,
            )
            self.generated[name] = expr
            alphas.append(expr)
        return alphas
    
    def generate_exhaustive(self, max_combos: int = 500) -> List[AlphaExpression]:
        alphas = []
        count = 0
        for p1, p2 in combinations(self.primitives, 2):
            for op in self.operators:
                if count >= max_combos:
                    break
                name = f'alpha_{count + 1}'
                expr = AlphaExpression(
                    name=name,
                    expression=f'{p1} {op} {p2}',
                    primitive=p1, operator=op, secondary=p2,
                )
                self.generated[name] = expr
                alphas.append(expr)
                count += 1
            if count >= max_combos:
                break
        return alphas
    
    def generate_by_template(self, template: str, params_list: List[Dict]) -> List[AlphaExpression]:
        alphas = []
        for i, params in enumerate(params_list):
            name = f'alpha_{len(self.generated) + 1}'
            expr_str = template.format(**params)
            expr = AlphaExpression(
                name=name,
                expression=expr_str,
                primitive=params.get('primitive', 'returns'),
                operator=params.get('operator', '+'),
                secondary=params.get('secondary'),
                params=params,
            )
            self.generated[name] = expr
            alphas.append(expr)
        return alphas


class AlphaSelector:
    """Evaluate and select best alphas by IC, Sharpe, turnover."""
    
    def __init__(self):
        self.results: Dict[str, Dict] = {}
    
    def evaluate_alpha(self, expr: AlphaExpression, data: pd.DataFrame,
                       forward_returns: pd.Series) -> Dict:
        sig = expr.compute(data)
        valid = sig.notna() & forward_returns.notna()
        
        if valid.sum() < 10:
            return {'alpha': expr.name, 'ic': 0, 'sharpe': 0, 'turnover': 0, 'n': 0}
        
        sig_aligned = sig[valid]
        ret_aligned = forward_returns[valid]
        
        ic = sig_aligned.corr(ret_aligned)
        
        daily_ret = sig_aligned * np.sign(ret_aligned)
        sharpe = np.sqrt(252) * daily_ret.mean() / (daily_ret.std() + 1e-8)
        
        turnover = np.abs(sig_aligned.diff()).mean() / (sig_aligned.abs().mean() + 1e-8)
        
        result = {
            'alpha': expr.name,
            'expression': expr.expression,
            'ic': float(ic),
            'sharpe': float(sharpe),
            'turnover': float(turnover),
            'n': int(valid.sum()),
        }
        self.results[expr.name] = result
        return result
    
    def evaluate_all(self, alphas: List[AlphaExpression], data: pd.DataFrame,
                     forward_period: int = 1) -> pd.DataFrame:
        close = data['Close'] if 'Close' in data else data.iloc[:, 0]
        fwd_ret = close.pct_change(forward_period).shift(-forward_period)
        
        results = []
        for expr in alphas:
            r = self.evaluate_alpha(expr, data, fwd_ret)
            results.append(r)
        
        df = pd.DataFrame(results)
        if len(df) > 0:
            df['ic_rank'] = df['ic'].rank(pct=True)
            df['score'] = df['ic'].abs() - 0.5 * df['turnover'] + 0.3 * df['sharpe'].abs()
        return df
    
    def select_top(self, df: pd.DataFrame, n: int = 10) -> pd.DataFrame:
        return df.nlargest(n, 'score')


class HypothesisGenerator:
    """Generate market hypotheses from data patterns."""
    
    def __init__(self):
        self.hypotheses: List[Dict] = []
    
    def from_correlation(self, df_corr: pd.DataFrame, target: str) -> List[Dict]:
        """Generate hypotheses from correlation matrix."""
        corr = df_corr[target].drop(target).sort_values(ascending=False)
        
        for feature, val in corr.items():
            if abs(val) < 0.3:
                continue
            direction = 'positivo' if val > 0 else 'negativo'
            h = {
                'id': hashlib.md5(f'{target}:{feature}:{val}'.encode()).hexdigest()[:8],
                'hypothesis': f'{feature} tiene correlación {direction} ({val:.2f}) con {target}',
                'target': target,
                'feature': feature,
                'correlation': float(val),
                'strength': abs(val),
                'method': 'correlation',
                'timestamp': datetime.now().isoformat(),
            }
            self.hypotheses.append(h)
        return self.hypotheses[-len(corr):]
    
    def from_ic_results(self, df_ic: pd.DataFrame) -> List[Dict]:
        for _, row in df_ic.iterrows():
            h = {
                'id': hashlib.md5(f'ic:{row["alpha"]}'.encode()).hexdigest()[:8],
                'hypothesis': f'Alpha "{row["expression"]}" tiene IC={row["ic"]:.3f} y Sharpe={row["sharpe"]:.2f}',
                'alpha': row['alpha'],
                'expression': row.get('expression', ''),
                'ic': float(row['ic']),
                'sharpe': float(row.get('sharpe', 0)),
                'strength': abs(float(row['ic'])) * 0.7 + abs(float(row.get('sharpe', 0))) * 0.3,
                'method': 'ic_analysis',
                'timestamp': datetime.now().isoformat(),
            }
            self.hypotheses.append(h)
        return self.hypotheses[-len(df_ic):]
    
    def from_regime_change(self, data: pd.DataFrame, window: int = 60) -> List[Dict]:
        close = data['Close'] if 'Close' in data else data.iloc[:, 0]
        returns = close.pct_change()
        
        vol_current = returns.tail(window).std()
        vol_prior = returns.iloc[-2*window:-window].std() if len(returns) > 2*window else vol_current
        
        change = (vol_current / vol_prior - 1) * 100
        if abs(change) > 20:
            direction = 'incremento' if change > 0 else 'disminución'
            h = {
                'id': hashlib.md5(f'regime:{change}'.encode()).hexdigest()[:8],
                'hypothesis': f'Volatilidad {direction} en {change:.1f}% sugiere cambio de régimen',
                'target': 'volatilidad',
                'change_pct': float(change),
                'strength': min(abs(change) / 100, 1.0),
                'method': 'regime_analysis',
                'timestamp': datetime.now().isoformat(),
            }
            self.hypotheses.append(h)
            return [h]
        return []
    
    def get_best(self, n: int = 5) -> List[Dict]:
        scored = sorted(self.hypotheses, key=lambda h: h.get('strength', 0), reverse=True)
        return scored[:n]
    
    def save(self):
        path = OUTPUT_DIR / 'hypotheses.json'
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(self.hypotheses, f, indent=2, ensure_ascii=False)
        print(f'[AutoResearch] {len(self.hypotheses)} hypotheses saved')


class SignalDecayAnalyzer:
    """Analyze signal decay and optimal rebalance frequency."""
    
    def __init__(self):
        self.decay_rates: Dict[str, float] = {}
    
    def compute_decay(self, signal: pd.Series, forward_returns: pd.Series,
                      max_lag: int = 20) -> Dict:
        ics = {}
        for lag in range(0, max_lag + 1):
            sig_lag = signal.shift(lag)
            valid = sig_lag.notna() & forward_returns.notna()
            if valid.sum() > 10:
                ic = sig_lag[valid].corr(forward_returns[valid])
                ics[lag] = float(ic)
            else:
                ics[lag] = 0.0
        
        if len(ics) > 2:
            lags = list(range(len(ics)))
            ic_vals = list(ics.values())
            half_life = None
            max_ic = max(abs(v) for v in ic_vals)
            if max_ic > 0:
                for lag, ic_val in ics.items():
                    if abs(ic_val) < max_ic / 2:
                        half_life = lag
                        break
            
            return {
                'ics_by_lag': ics,
                'half_life': half_life,
                'decay_rate': float(np.polyfit(lags, np.abs(ic_vals), 1)[0]),
                'optimal_lag': max(lags, key=lambda l: abs(ics[l])),
            }
        return {'ics_by_lag': ics, 'half_life': None, 'decay_rate': 0, 'optimal_lag': 0}


class AutomatedResearch:
    """Main automated research pipeline."""
    
    def __init__(self):
        self.factory = AlphaFactory()
        self.selector = AlphaSelector()
        self.hypothesis_gen = HypothesisGenerator()
        self.decay_analyzer = SignalDecayAnalyzer()
    
    def run_pipeline(self, data: pd.DataFrame, n_alphas: int = 100,
                     n_select: int = 10) -> Dict:
        close = data['Close'] if 'Close' in data else data.iloc[:, 0]
        fwd_ret = close.pct_change().shift(-1)
        
        alphas = self.factory.generate_random(n_alphas)
        df_ic = self.selector.evaluate_all(alphas, data)
        
        top = self.selector.select_top(df_ic, n_select)
        
        self.hypothesis_gen.from_ic_results(top)
        
        if len(data.columns) > 1:
            col = data.select_dtypes(include=[np.number]).columns[0]
            n_cols = min(10, len(data.select_dtypes(include=[np.number]).columns))
            top_corr = data.select_dtypes(include=[np.number]).iloc[:, :n_cols].corr()
            self.hypothesis_gen.from_correlation(top_corr, top_corr.columns[0])
        
        decay_results = {}
        for _, row in top.iterrows():
            expr = self.factory.generated.get(row['alpha'])
            if expr:
                sig = expr.compute(data)
                decay_results[row['alpha']] = self.decay_analyzer.compute_decay(sig, fwd_ret)
        
        self.hypothesis_gen.save()
        
        return {
            'n_alphas_generated': len(alphas),
            'top_alphas': top.to_dict('records'),
            'hypotheses': self.hypothesis_gen.get_best(10),
            'signal_decay': decay_results,
            'ic_summary': {
                'mean_ic': float(df_ic['ic'].mean()),
                'max_ic': float(df_ic['ic'].max()),
                'ic_std': float(df_ic['ic'].std()),
            },
        }
    
    def generate_report(self) -> str:
        report_path = OUTPUT_DIR / 'research_report.md'
        report = f"# Automated Research Report\n"
        report += f"Generated: {datetime.now().isoformat()}\n\n"
        report += f"## Generated Alphas: {len(self.factory.generated)}\n"
        report += f"## Selected Alphas: {len(self.selector.results)}\n"
        report += f"## Hypotheses: {len(self.hypothesis_gen.hypotheses)}\n\n"
        
        best = self.hypothesis_gen.get_best(5)
        if best:
            report += "### Top Hypotheses\n"
            for h in best:
                report += f"- [{h.get('strength', 0):.2f}] {h['hypothesis']}\n"
        
        report_path.write_text(report, encoding='utf-8')
        return str(report_path)


if __name__ == '__main__':
    print('[AutoResearch] Running alpha research pipeline...')
    np.random.seed(42)
    
    dates = pd.date_range('2020-01-01', '2024-12-31', freq='B')
    n = len(dates)
    close = 100 * np.exp(np.cumsum(np.random.randn(n) * 0.02))
    data = pd.DataFrame({'Close': close}, index=dates)
    
    research = AutomatedResearch()
    results = research.run_pipeline(data, n_alphas=50, n_select=10)
    
    print(f"Alphas: {results['n_alphas_generated']}")
    print(f"Mean IC: {results['ic_summary']['mean_ic']:.4f}")
    print(f"Best IC: {results['ic_summary']['max_ic']:.4f}")
    print(f"Hypotheses: {len(results['hypotheses'])}")
    
    report = research.generate_report()
    print(f"Report: {report}")