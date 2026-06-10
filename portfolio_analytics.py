#!/usr/bin/env python3
"""
portfolio_analytics.py - Advanced portfolio analytics.
Barra risk factor decomposition, Brinson performance attribution,
Black-Litterman model, risk budgeting, and scenario analysis.
"""
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import json
import os
from pathlib import Path
from scipy.optimize import minimize

from config.settings import get_setting

DATA_DIR = os.environ.get('GITHUB_WORKSPACE', '.')
OUTPUT_DIR = Path(DATA_DIR) / 'Datos' / 'portfolio_analytics'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ─── Barra Risk Model ────────────────────────────────────────────────────

BARRA_STYLE_FACTORS = [
    'value', 'momentum', 'size', 'volatility', 'quality',
    'growth', 'liquidity', 'leverage', 'dividend_yield', 'sentiment',
]


@dataclass
class FactorExposure:
    factor: str
    exposure: float
    contribution: float  # risk contribution


class BarraRiskModel:
    """Barra-style factor risk decomposition."""
    
    def __init__(self, style_factors: List[str] = None):
        self.style_factors = style_factors or BARRA_STYLE_FACTORS
        self.factor_returns: Optional[pd.DataFrame] = None
        self.factor_cov: Optional[np.ndarray] = None
        self.specific_risk: Optional[pd.Series] = None
    
    def estimate_factor_returns(self, returns: pd.DataFrame,
                                exposures: pd.DataFrame) -> pd.DataFrame:
        """Cross-sectional regression: R = X*f + e."""
        factor_rets = []
        for t in returns.index:
            if t in exposures.index:
                y = returns.loc[t].values
                X = exposures.loc[t].values
                X = np.column_stack([X, np.ones(len(X))])
                try:
                    beta = np.linalg.lstsq(X, y, rcond=None)[0]
                    factor_rets.append(beta[:-1])  # exclude intercept
                except np.linalg.LinAlgError:
                    factor_rets.append(np.zeros(len(self.style_factors)))
            else:
                factor_rets.append(np.zeros(len(self.style_factors)))
        
        self.factor_returns = pd.DataFrame(
            factor_rets, index=returns.index, columns=self.style_factors)
        return self.factor_returns
    
    def factor_covariance(self, window: int = 60) -> np.ndarray:
        if self.factor_returns is None or len(self.factor_returns) < window:
            n = len(self.style_factors)
            self.factor_cov = np.eye(n) * 0.01
            return self.factor_cov
        
        self.factor_cov = self.factor_returns.tail(window).cov().values
        return self.factor_cov
    
    def risk_decomposition(self, weights: np.ndarray,
                           exposures: pd.DataFrame,
                           specific_risk: pd.Series) -> Dict[str, Any]:
        """Decompose portfolio risk into factor + specific."""
        if self.factor_cov is None:
            self.factor_covariance()
        
        n_assets = len(weights)
        n_factors = len(self.style_factors)
        
        # Factor risk: w' * B * F * B' * w
        if len(exposures) >= n_assets:
            B = exposures.values[:n_assets, :n_factors]
        else:
            B = np.eye(n_assets, n_factors) * 0.1
        
        factor_risk = weights @ B @ self.factor_cov @ B.T @ weights
        specific_risk_val = np.sum(weights ** 2 * specific_risk.values[:n_assets] ** 2)
        total_risk = np.sqrt(factor_risk + specific_risk_val)
        
        # Marginal contributions
        marginal_factor = 2 * (B @ self.factor_cov @ B.T) @ weights / (2 * total_risk + 1e-8)
        marginal_specific = 2 * specific_risk.values[:n_assets] * weights / (2 * total_risk + 1e-8)
        
        factor_contrib = weights * marginal_factor
        specific_contrib = weights * marginal_specific
        
        # Per-factor decomposition
        factor_exposures = []
        for i, factor in enumerate(self.style_factors[:B.shape[1]]):
            contrib = weights @ (B[:, i:i+1] * self.factor_cov[i, i]) @ (B[:, i:i+1].T) @ weights
            factor_exposures.append(FactorExposure(
                factor=factor,
                exposure=float(np.abs(weights).dot(np.abs(B[:, i]))) / (n_assets + 1e-8),
                contribution=float(contrib / (total_risk ** 2 + 1e-8)),
            ))
        
        return {
            'total_risk': float(total_risk),
            'factor_risk': float(np.sqrt(factor_risk)),
            'specific_risk': float(np.sqrt(specific_risk_val)),
            'factor_risk_pct': float(factor_risk / (total_risk ** 2 + 1e-8)),
            'factor_exposures': [{
                'factor': fe.factor,
                'exposure': fe.exposure,
                'contribution': fe.contribution,
            } for fe in factor_exposures],
            'marginal_contributions': {
                'mean_factor': float(np.mean(marginal_factor)),
                'mean_specific': float(np.mean(marginal_specific)),
            },
        }
    
    def risk_report(self, weights: np.ndarray, exposures: pd.DataFrame,
                    specific_risk: pd.Series) -> str:
        dec = self.risk_decomposition(weights, exposures, specific_risk)
        report = "# Barra Risk Report\n\n"
        report += f"Generated: {datetime.now().isoformat()}\n\n"
        report += f"## Total Risk: {dec['total_risk']:.4%}\n"
        report += f"- Factor Risk: {dec['factor_risk']:.4%} ({dec['factor_risk_pct']:.1%} of total)\n"
        report += f"- Specific Risk: {dec['specific_risk']:.4%} ({1-dec['factor_risk_pct']:.1%} of total)\n\n"
        report += "## Factor Exposures\n"
        for fe in dec['factor_exposures']:
            report += f"- {fe['factor']}: exposure={fe['exposure']:.3f}, contribution={fe['contribution']:.2%}\n"
        
        (OUTPUT_DIR / 'barra_report.md').write_text(report, encoding='utf-8')
        return report


# ─── Brinson Attribution ────────────────────────────────────────────────

class BrinsonAttribution:
    """Brinson decomposition: allocation + selection + interaction effects."""
    
    def __init__(self):
        self.results: Dict[str, float] = {}
    
    def decompose(self, portfolio_weights: pd.Series,
                  benchmark_weights: pd.Series,
                  portfolio_returns: pd.Series,
                  benchmark_returns: pd.Series) -> Dict[str, float]:
        """Brinson, Hood, Beebower (1986) attribution."""
        alloc_effect = ((portfolio_weights - benchmark_weights) * 
                        (benchmark_returns)).sum()
        
        selec_effect = (benchmark_weights * 
                        (portfolio_returns - benchmark_returns)).sum()
        
        inter_effect = ((portfolio_weights - benchmark_weights) * 
                        (portfolio_returns - benchmark_returns)).sum()
        
        total_excess = (portfolio_weights * portfolio_returns).sum() - \
                       (benchmark_weights * benchmark_returns).sum()
        
        self.results = {
            'allocation_effect': float(alloc_effect),
            'selection_effect': float(selec_effect),
            'interaction_effect': float(inter_effect),
            'total_excess_return': float(total_excess),
            'allocation_pct': float(alloc_effect / (total_excess + 1e-8)),
            'selection_pct': float(selec_effect / (total_excess + 1e-8)),
            'interaction_pct': float(inter_effect / (total_excess + 1e-8)),
        }
        return self.results
    
    def multi_period(self, portfolio_df: pd.DataFrame,
                     benchmark_df: pd.DataFrame,
                     returns_df: pd.DataFrame) -> pd.DataFrame:
        """Multi-period attribution."""
        periods = []
        for t in returns_df.index:
            if t in portfolio_df.index and t in benchmark_df.index:
                pd_ret = returns_df.loc[t] if t in returns_df.index else pd.Series(0, index=returns_df.columns)
                bm_ret = returns_df.loc[t] if t in returns_df.index else pd.Series(0, index=returns_df.columns)
                r = self.decompose(
                    portfolio_df.loc[t], benchmark_df.loc[t],
                    pd_ret, bm_ret)
                r['period'] = t
                periods.append(r)
        return pd.DataFrame(periods)


# ─── Black-Litterman Model ──────────────────────────────────────────────

class BlackLitterman:
    """Black-Litterman portfolio optimization with views."""
    
    def __init__(self, risk_aversion: float = 2.5, tau: float = 0.05):
        self.risk_aversion = risk_aversion
        self.tau = tau
        self.implied_returns: Optional[np.ndarray] = None
        self.posterior_returns: Optional[np.ndarray] = None
        self.posterior_cov: Optional[np.ndarray] = None
    
    def implied_equilibrium_returns(self, cap_weights: np.ndarray,
                                    cov_matrix: np.ndarray,
                                    market_ret: float) -> np.ndarray:
        """Reverse-engineer returns from market cap weights."""
        sigma = cov_matrix
        pi = self.risk_aversion * sigma @ cap_weights
        self.implied_returns = pi
        return pi
    
    def add_views(self, P: np.ndarray, Q: np.ndarray,
                  omega: Optional[np.ndarray] = None,
                  sigma: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
        """Combine prior with views for posterior."""
        if sigma is None:
            sigma = np.eye(len(P[0])) * 0.01
        if omega is None:
            omega = np.eye(len(Q)) * 0.01
        
        n = sigma.shape[0]
        tau_sigma = self.tau * sigma
        
        # Posterior mean
        M1 = np.linalg.inv(tau_sigma)
        M2 = P.T @ np.linalg.inv(omega) @ P
        post_cov = np.linalg.inv(M1 + M2)
        post_mean = post_cov @ (np.linalg.inv(tau_sigma) @ self.implied_returns + 
                                P.T @ np.linalg.inv(omega) @ Q)
        
        self.posterior_returns = post_mean
        self.posterior_cov = post_cov + tau_sigma
        return post_mean, self.posterior_cov
    
    def optimize(self, returns: np.ndarray, cov: np.ndarray,
                 constraints: Optional[List] = None) -> np.ndarray:
        """Mean-variance optimization."""
        n = len(returns)
        
        def neg_utility(w):
            port_ret = w @ returns
            port_risk = w @ cov @ w
            return -(port_ret - 0.5 * self.risk_aversion * port_risk)
        
        w0 = np.ones(n) / n
        
        bounds = [(0, 1) for _ in range(n)]
        constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]
        
        result = minimize(neg_utility, w0, method='SLSQP',
                          bounds=bounds, constraints=constraints)
        return result.x if result.success else w0


# ─── Risk Budgeting ─────────────────────────────────────────────────────

class RiskBudgeting:
    """Risk parity and risk budgeting portfolios."""
    
    @staticmethod
    def risk_parity_weights(cov: np.ndarray) -> np.ndarray:
        """Equal risk contribution weights."""
        n = cov.shape[0]
        
        def risk_contribution(w):
            sigma = np.sqrt(w @ cov @ w)
            mrc = cov @ w / sigma
            rc = w * mrc
            target = sigma / n
            return np.sum((rc - target) ** 2)
        
        w0 = np.ones(n) / n
        bounds = [(0, 1) for _ in range(n)]
        constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]
        
        result = minimize(risk_contribution, w0, method='SLSQP',
                          bounds=bounds, constraints=constraints)
        return result.x if result.success else w0
    
    @staticmethod
    def risk_budget_weights(cov: np.ndarray, budgets: np.ndarray) -> np.ndarray:
        """Target risk contribution weights."""
        n = cov.shape[0]
        budgets = budgets / budgets.sum()
        
        def risk_contribution_error(w):
            sigma = np.sqrt(w @ cov @ w)
            mrc = cov @ w / sigma
            rc = w * mrc
            return np.sum((rc / (sigma + 1e-8) - budgets) ** 2)
        
        w0 = np.ones(n) / n
        bounds = [(0, 1) for _ in range(n)]
        constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]
        
        result = minimize(risk_contribution_error, w0, method='SLSQP',
                          bounds=bounds, constraints=constraints)
        return result.x if result.success else w0


# ─── Scenario Analysis ─────────────────────────────────────────────────

class ScenarioAnalysis:
    """Stress testing and scenario analysis."""
    
    def __init__(self):
        self.scenarios = {
            'market_crash': {'equity': -0.20, 'bond': 0.02, 'commodity': -0.10, 'cash': 0.0},
            'rate_hike': {'equity': -0.05, 'bond': -0.03, 'commodity': 0.05, 'cash': 0.02},
            'inflation_surge': {'equity': -0.08, 'bond': -0.05, 'commodity': 0.15, 'cash': 0.0},
            'recession': {'equity': -0.15, 'bond': 0.05, 'commodity': -0.08, 'cash': 0.01},
            'bull_market': {'equity': 0.15, 'bond': -0.02, 'commodity': 0.05, 'cash': 0.0},
            'stagflation': {'equity': -0.12, 'bond': -0.08, 'commodity': 0.08, 'cash': 0.02},
            'liquidity_crisis': {'equity': -0.25, 'bond': 0.01, 'commodity': -0.15, 'cash': 0.005},
            'tech_bubble': {'equity': 0.10, 'bond': -0.04, 'commodity': -0.02, 'cash': 0.0},
        }
    
    def run(self, weights: np.ndarray, asset_classes: List[str]) -> pd.DataFrame:
        results = []
        for scenario, impacts in self.scenarios.items():
            impact_vec = np.array([impacts.get(ac, 0) for ac in asset_classes])
            port_impact = weights @ impact_vec
            
            var_95 = np.percentile(impact_vec, 5)
            cvar_95 = impact_vec[impact_vec <= var_95].mean() if (impact_vec <= var_95).any() else var_95
            
            results.append({
                'scenario': scenario,
                'portfolio_impact': float(port_impact),
                'max_asset_impact': float(np.min(impact_vec)),
                'var_95': float(var_95),
                'cvar_95': float(cvar_95),
            })
        
        df = pd.DataFrame(results).sort_values('portfolio_impact')
        path = OUTPUT_DIR / 'scenario_analysis.csv'
        df.to_csv(path, index=False)
        return df


# ─── Risk Metrics ───────────────────────────────────────────────────────

class RiskMetrics:
    """Comprehensive risk metrics."""
    
    @staticmethod
    def compute_all(returns: pd.Series) -> Dict[str, float]:
        rf = 0.05 / 252
        excess = returns - rf
        cum = (1 + returns).cumprod()
        peak = cum.expanding().max()
        dd = (cum - peak) / peak
        
        ann_ret = (1 + returns.mean()) ** 252 - 1
        ann_vol = returns.std() * np.sqrt(252)
        sharpe = excess.mean() / (excess.std() + 1e-8) * np.sqrt(252)
        sortino = excess.mean() / (excess[excess < 0].std() + 1e-8) * np.sqrt(252)
        max_dd = dd.min()
        calmar = ann_ret / (abs(max_dd) + 1e-8)
        
        var_95 = np.percentile(returns, 5)
        cvar_95 = returns[returns <= var_95].mean()
        
        skew = returns.skew()
        kurt = returns.kurt()
        
        # Tail ratio
        tail_ratio = returns.quantile(0.95) / abs(returns.quantile(0.05))
        
        # Win rate
        win_rate = (returns > 0).mean()
        
        return {
            'annualized_return': float(ann_ret),
            'annualized_volatility': float(ann_vol),
            'sharpe_ratio': float(sharpe),
            'sortino_ratio': float(sortino),
            'max_drawdown': float(max_dd),
            'calmar_ratio': float(calmar),
            'var_95': float(var_95),
            'cvar_95': float(cvar_95),
            'skewness': float(skew),
            'kurtosis': float(kurt),
            'tail_ratio': float(tail_ratio),
            'win_rate': float(win_rate),
        }


# ─── Main Orchestrator ─────────────────────────────────────────────────

class PortfolioAnalytics:
    """Complete portfolio analytics pipeline."""
    
    def __init__(self):
        self.barra = BarraRiskModel()
        self.brinson = BrinsonAttribution()
        self.bl = BlackLitterman()
        self.risk_budget = RiskBudgeting()
        self.scenario = ScenarioAnalysis()
        self.risk_metrics = RiskMetrics()
    
    def full_report(self, returns: pd.DataFrame, weights: np.ndarray,
                    benchmark_returns: pd.Series = None,
                    exposures: pd.DataFrame = None,
                    specific_risk: pd.Series = None) -> Dict:
        results = {}
        
        if exposures is not None and specific_risk is not None:
            results['barra'] = self.barra.risk_decomposition(weights, exposures, specific_risk)
        
        if benchmark_returns is not None:
            port_ret = returns @ weights if isinstance(returns, pd.DataFrame) else returns
            bm_ret = benchmark_returns
            results['brinson'] = self.brinson.decompose(
                pd.Series(weights, index=returns.columns[:len(weights)]),
                pd.Series(np.ones(len(weights)) / len(weights), index=returns.columns[:len(weights)]),
                pd.Series(port_ret.mean(), index=returns.columns[:len(weights)]) if hasattr(port_ret, 'mean') else port_ret,
                bm_ret)
        
        if isinstance(returns, pd.DataFrame):
            results['risk_metrics'] = {
                col: self.risk_metrics.compute_all(returns[col])
                for col in returns.columns[:min(5, len(returns.columns))]
            }
        
        asset_classes = ['equity'] * len(weights)
        results['scenarios'] = self.scenario.run(weights, asset_classes).to_dict('records')
        
        (OUTPUT_DIR / 'portfolio_report.json').write_text(
            json.dumps(results, indent=2, default=str), encoding='utf-8')
        
        return results


if __name__ == '__main__':
    print('[PortfolioAnalytics] Running analytics...')
    np.random.seed(42)
    
    n = 1000
    n_assets = 5
    returns = pd.DataFrame({
        f'Asset_{i}': np.random.randn(n) * 0.02 + 0.0005
        for i in range(n_assets)
    }, index=pd.date_range('2022-01-01', periods=n, freq='B'))
    
    weights = np.ones(n_assets) / n_assets
    
    exposures = pd.DataFrame(
        np.random.randn(n_assets, len(BARRA_STYLE_FACTORS)),
        columns=BARRA_STYLE_FACTORS)
    
    specific_risk = pd.Series(np.random.rand(n_assets) * 0.01)
    
    analytics = PortfolioAnalytics()
    results = analytics.full_report(returns, weights, exposures=exposures, 
                                    specific_risk=specific_risk)
    
    print(f"Barra total risk: {results.get('barra', {}).get('total_risk', 0):.4%}")
    print(f"Scenarios: {len(results.get('scenarios', []))}")
    print(f"Risk metrics: {len(results.get('risk_metrics', {}))} assets")