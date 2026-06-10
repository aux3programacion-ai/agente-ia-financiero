#!/usr/bin/env python3
"""
causal_reasoning.py - Causal inference for financial markets.
DoWhy + EconML: identifies causal relationships between features and returns.
Counterfactual simulation: "what would have happened if X had been different?"
"""
import json
import os
import time
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, asdict

try:
    import dowhy
    from dowhy import CausalModel
    DOWHY_AVAILABLE = True
except ImportError:
    DOWHY_AVAILABLE = False

try:
    from econml.dml import CausalForestDML, LinearDML
    from econml.dr import DRLearner
    ECONML_AVAILABLE = True
except ImportError:
    ECONML_AVAILABLE = False

from config.settings import get_setting

DATA_DIR = os.environ.get('GITHUB_WORKSPACE', '.')
OUTPUT_DIR = Path(DATA_DIR) / 'Datos' / 'causal'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


@dataclass
class CausalEffect:
    treatment: str
    outcome: str
    effect: float
    confidence_interval: Tuple[float, float]
    p_value: float
    method: str
    significant: bool
    description: str = ''


@dataclass
class CausalGraph:
    nodes: List[str]
    edges: List[Tuple[str, str]]
    graph_dot: str = ''


class CausalGraphBuilder:
    """Build causal graphs for financial markets."""
    
    DEFAULT_GRAPH = """
    digraph {
        # Macro → Market
        fed_rate -> yield_curve;
        yield_curve -> market_regime;
        vix -> market_regime;
        dxy -> sector_returns;
        
        # Market → Sector
        market_regime -> sector_returns;
        macro_sentiment -> sector_returns;
        
        # Sector → Ticker
        sector_returns -> ticker_returns;
        analyst_ratings -> ticker_returns;
        earnings_surprise -> ticker_returns;
        
        # Ticker level
        ticker_returns -> volatility;
        ticker_returns -> volume;
        volatility -> rsi;
        rsi -> ticker_returns;
        
        # Confounders
        inflation -> fed_rate;
        inflation -> consumer_spending;
        consumer_spending -> sector_returns;
        unemployment -> fed_rate;
        unemployment -> consumer_spending;
        
        # Risk
        volatility -> var_95;
        correlation -> portfolio_risk;
        market_regime -> correlation;
    }
    """
    
    @classmethod
    def get_default_graph(cls) -> CausalGraph:
        return cls.from_dot(cls.DEFAULT_GRAPH)
    
    @classmethod
    def from_dot(cls, dot_string: str) -> CausalGraph:
        import re
        nodes = set()
        edges = []
        for line in dot_string.split('\n'):
            line = line.strip()
            arrow = re.search(r'(\w+)\s*->\s*(\w+)', line)
            if arrow:
                src, dst = arrow.group(1), arrow.group(2)
                nodes.add(src)
                nodes.add(dst)
                edges.append((src, dst))
        return CausalGraph(list(nodes), edges, dot_string)
    
    @classmethod
    def ticker_graph(cls, ticker: str) -> CausalGraph:
        dot = f"""
        digraph {{
            macro_sentiment -> {ticker}_returns;
            sector_sentiment -> {ticker}_returns;
            {ticker}_rsi -> {ticker}_returns;
            {ticker}_macd -> {ticker}_returns;
            {ticker}_volume -> {ticker}_returns;
            {ticker}_volatility -> {ticker}_returns;
            {ticker}_returns -> {ticker}_volatility;
            market_regime -> {ticker}_returns;
            analyst_rating -> {ticker}_returns;
            earnings_surprise -> {ticker}_returns;
        }}
        """
        return cls.from_dot(dot)


class CausalAnalyzer:
    def __init__(self):
        self.results = {}
        self.graphs = {}
        self.cache_path = OUTPUT_DIR / 'causal_results.json'
        self._load()
    
    def _load(self):
        if self.cache_path.exists():
            try:
                self.results = json.loads(self.cache_path.read_text())
            except:
                self.results = {'analyses': []}
        else:
            self.results = {'analyses': [], 'causal_graphs': []}
    
    def _save(self):
        self.results['analyses'] = self.results['analyses'][-200:]
        self.cache_path.write_text(json.dumps(self.results, indent=2, default=str))
    
    def estimate_ate(self, data: pd.DataFrame, treatment: str, outcome: str,
                     common_causes: List[str], instruments: List[str] = None,
                     graph_dot: str = None) -> CausalEffect:
        """
        Estimate Average Treatment Effect using DoWhy.
        
        Args:
            data: DataFrame with all variables
            treatment: Name of treatment variable
            outcome: Name of outcome variable
            common_causes: List of confounders
            instruments: List of instrumental variables
            graph_dot: DOT graph string (optional)
        """
        if not DOWHY_AVAILABLE:
            return self._estimate_linear(data, treatment, outcome, common_causes)
        
        try:
            if graph_dot is None:
                graph_dot = f"""
                digraph {{
                    {'; '.join(f'{c} -> {treatment}; {c} -> {outcome}' for c in common_causes)}
                    {treatment} -> {outcome};
                    {'; '.join(f'{i} -> {treatment}' for i in (instruments or []))}
                }}
                """
            
            model = CausalModel(
                data=data,
                treatment=treatment,
                outcome=outcome,
                common_causes=common_causes,
                instruments=instruments or [],
                graph=graph_dot
            )
            
            identified = model.identify_effect(proceed_when_unidentifiable=True)
            
            estimate = model.estimate_effect(identified, 
                method_name='backdoor.linear_regression',
                target_units='ate',
                confidence_level=0.95
            )
            
            effect = CausalEffect(
                treatment=treatment,
                outcome=outcome,
                effect=float(estimate.value),
                confidence_interval=(float(estimate.get_confidence_interval()[0]),
                                    float(estimate.get_confidence_interval()[1])),
                p_value=float(estimate.test_statistics().get('p_value', [1.0])[0]) if hasattr(estimate, 'test_statistics') else 0.05,
                method='dowhy_linear_regression',
                significant=abs(float(estimate.value)) > 0.01
            )
            
            refute = model.refute_estimate(identified, estimate, 'random_common_cause')
            effect.description = f'ATE={effect.effect:.4f} [{effect.confidence_interval[0]:.4f}, {effect.confidence_interval[1]:.4f}], p={effect.p_value:.4f}'
            
            record = asdict(effect)
            record['refutation'] = str(refute)
            record['timestamp'] = datetime.now(timezone.utc).isoformat()
            self.results['analyses'].append(record)
            self._save()
            
            return effect
            
        except Exception as e:
            print(f'[Causal] DoWhy failed: {e}')
            return self._estimate_linear(data, treatment, outcome, common_causes)
    
    def _estimate_linear(self, data, treatment, outcome, common_causes) -> CausalEffect:
        """Fallback: linear regression with controls."""
        try:
            import statsmodels.api as sm
            
            X = data[[treatment] + common_causes]
            X = sm.add_constant(X)
            y = data[outcome]
            
            model = sm.OLS(y, X).fit()
            
            coef = model.params[treatment]
            ci = model.conf_int().loc[treatment]
            
            return CausalEffect(
                treatment=treatment,
                outcome=outcome,
                effect=float(coef),
                confidence_interval=(float(ci[0]), float(ci[1])),
                p_value=float(model.pvalues[treatment]),
                method='linear_regression_controls',
                significant=model.pvalues[treatment] < 0.05
            )
        except ImportError:
            from sklearn.linear_model import LinearRegression
            X = data[[treatment] + common_causes].values
            y = data[outcome].values
            model = LinearRegression().fit(X, y)
            coef = model.coef_[0]
            return CausalEffect(
                treatment=treatment, outcome=outcome,
                effect=float(coef),
                confidence_interval=(float(coef - 0.1), float(coef + 0.1)),
                p_value=0.05,
                method='sklearn_linear_regression',
                significant=abs(coef) > 0.01
            )
    
    def estimate_cate(self, data: pd.DataFrame, treatment: str, outcome: str,
                      features: List[str], common_causes: List[str]) -> Dict:
        """
        Estimate Conditional Average Treatment Effect using CausalForest.
        Gives heterogeneous treatment effects (who benefits most?).
        """
        if not ECONML_AVAILABLE:
            return {'error': 'EconML not available. pip install econml'}
        
        try:
            est = CausalForestDML(
                model_y=LinearDML(model_y='linear', model_t='linear'),
                model_t='linear',
                discrete_treatment=False,
                cv=3
            )
            
            est.fit(y=data[outcome], T=data[treatment], 
                    X=data[features], W=data[common_causes])
            
            cate = est.effect(data[features])
            ate = float(np.mean(cate))
            
            return {
                'ate': ate,
                'cate_mean': float(np.mean(cate)),
                'cate_std': float(np.std(cate)),
                'cate_min': float(np.min(cate)),
                'cate_max': float(np.max(cate)),
                'n_samples': len(data),
                'features': features,
                'top_features_heterogeneity': self._find_heterogeneity(est, data, features)
            }
        except Exception as e:
            return {'error': str(e)}
    
    def _find_heterogeneity(self, est, data, features) -> List[Dict]:
        """Find which features drive heterogeneity in treatment effects."""
        results = []
        for i, feat in enumerate(features):
            feat_values = data[feat]
            try:
                cate_by_feat = est.effect(data[features].values, T0=0, T1=1)
                low_mask = feat_values <= feat_values.median()
                high_mask = feat_values > feat_values.median()
                
                cate_low = float(np.mean(cate_by_feat[low_mask]))
                cate_high = float(np.mean(cate_by_feat[high_mask]))
                
                results.append({
                    'feature': feat,
                    'cate_low': cate_low,
                    'cate_high': cate_high,
                    'diff': cate_high - cate_low
                })
            except:
                continue
        
        results.sort(key=lambda x: abs(x['diff']), reverse=True)
        return results[:5]
    
    def counterfactual(self, data: pd.DataFrame, treatment: str, outcome: str,
                       common_causes: List[str], treatment_value: float = 0.0) -> Dict:
        """
        Counterfactual: what would outcome be if treatment = treatment_value?
        """
        if not DOWHY_AVAILABLE:
            return self._counterfactual_linear(data, treatment, outcome, common_causes, treatment_value)
        
        try:
            graph_dot = f"""
            digraph {{
                {'; '.join(f'{c} -> {treatment}; {c} -> {outcome}' for c in common_causes)}
                {treatment} -> {outcome};
            }}
            """
            
            model = CausalModel(
                data=data, treatment=treatment, outcome=outcome,
                common_causes=common_causes, graph=graph_dot
            )
            identified = model.identify_effect()
            
            cf = model.counterfactual(
                identified,
                {treatment: lambda df: treatment_value},
                data
            )
            
            actual = data[outcome].values
            counterfactual = cf.values
            
            return {
                'treatment_changed_to': treatment_value,
                'actual_mean': float(np.mean(actual)),
                'counterfactual_mean': float(np.mean(counterfactual)),
                'impact': float(np.mean(counterfactual) - np.mean(actual)),
                'n_samples': len(data),
                'method': 'dowhy'
            }
        except Exception as e:
            return self._counterfactual_linear(data, treatment, outcome, common_causes, treatment_value)
    
    def _counterfactual_linear(self, data, treatment, outcome, common_causes, treatment_value) -> Dict:
        try:
            import statsmodels.api as sm
            X = data[[treatment] + common_causes]
            X = sm.add_constant(X)
            y = data[outcome]
            model = sm.OLS(y, X).fit()
            
            actual_mean = float(np.mean(y))
            counterfactual_X = X.copy()
            counterfactual_X[treatment] = treatment_value
            cf_preds = model.predict(counterfactual_X)
            
            return {
                'treatment_changed_to': treatment_value,
                'actual_mean': actual_mean,
                'counterfactual_mean': float(np.mean(cf_preds)),
                'impact': float(np.mean(cf_preds) - actual_mean),
                'method': 'linear_regression',
                'coef_treatment': float(model.params[treatment])
            }
        except ImportError:
            from sklearn.linear_model import LinearRegression
            X = data[[treatment] + common_causes].values
            y = data[outcome].values
            model = LinearRegression().fit(X, y)
            actual_mean = float(np.mean(y))
            cf_X = X.copy()
            cf_X[:, 0] = treatment_value
            cf_mean = float(np.mean(model.predict(cf_X)))
            return {
                'treatment_changed_to': treatment_value,
                'actual_mean': actual_mean,
                'counterfactual_mean': cf_mean,
                'impact': cf_mean - actual_mean,
                'method': 'sklearn_regression',
                'coef_treatment': float(model.coef_[0])
            }
    
    def market_structure_analysis(self, data: pd.DataFrame) -> Dict:
        """Analyze market structure: what drives ticker returns?"""
        features = [c for c in data.columns if c not in ['returns', 'date', 'ticker']]
        if len(features) < 2:
            return {'error': 'Need at least 2 features'}
        
        results = {}
        for feat in features[:10]:
            causes = [f for f in features[:5] if f != feat]
            effect = self.estimate_ate(data, feat, 'returns', causes[:3])
            results[feat] = {
                'effect': effect.effect,
                'p_value': effect.p_value,
                'significant': effect.significant,
                'ci': list(effect.confidence_interval)
            }
        
        return {
            'features_analyzed': list(results.keys()),
            'significant_drivers': [k for k, v in results.items() if v.get('significant')],
            'results': results,
            'timestamp': datetime.now(timezone.utc).isoformat()
        }


def analyze_causal_impact(ticker: str, feature_name: str, data: pd.DataFrame = None) -> Dict:
    """Quick causal impact analysis for a ticker-feature pair."""
    analyzer = CausalAnalyzer()
    
    if data is None:
        np.random.seed(42)
        n = 200
        data = pd.DataFrame({
            'rsi_14': np.random.uniform(20, 80, n),
            'macd_hist': np.random.randn(n) * 2,
            'volume_ratio': np.random.uniform(0.3, 3, n),
            'volatility': np.random.uniform(0.1, 0.6, n),
            'returns': np.random.randn(n) * 0.02,
            'market_return': np.random.randn(n) * 0.015,
            'vix': np.random.uniform(12, 35, n)
        })
    
    effect = analyzer.estimate_ate(
        data=data,
        treatment=feature_name,
        outcome='returns',
        common_causes=['market_return', 'vix'],
        graph_dot=CausalGraphBuilder.ticker_graph(ticker).graph_dot
    )
    
    return asdict(effect)


def build_causal_graph(ticker: str) -> Dict:
    """Build and return causal graph for a ticker."""
    graph = CausalGraphBuilder.ticker_graph(ticker)
    return {'ticker': ticker, 'nodes': graph.nodes, 'edges': graph.edges}


if __name__ == '__main__':
    np.random.seed(42)
    n = 500
    data = pd.DataFrame({
        'rsi_14': np.random.uniform(20, 80, n),
        'macd_hist': np.random.randn(n) * 2,
        'volume_ratio': np.random.uniform(0.3, 3, n),
        'volatility': np.random.uniform(0.1, 0.6, n),
        'vix': np.random.uniform(12, 35, n),
        'market_return': np.random.randn(n) * 0.015,
        'returns': np.random.randn(n) * 0.02
    })
    
    analyzer = CausalAnalyzer()
    
    print('[Causal] Estimating ATE...')
    effect = analyzer.estimate_ate(
        data, treatment='rsi_14', outcome='returns',
        common_causes=['market_return', 'vix']
    )
    print(f'  ATE: {effect.effect:.4f} [{effect.confidence_interval[0]:.4f}, {effect.confidence_interval[1]:.4f}]')
    print(f'  p={effect.p_value:.4f} significant={effect.significant}')
    
    print('\n[Causal] Counterfactual (what if rsi_14=30?)...')
    cf = analyzer.counterfactual(data, 'rsi_14', 'returns', ['market_return', 'vix'], 30)
    print(f'  Actual: {cf["actual_mean"]:.4f} -> Counterfactual: {cf["counterfactual_mean"]:.4f}')
    print(f'  Impact: {cf["impact"]:.4f}')
    
    print('\n[Causal] Market structure analysis...')
    structure = analyzer.market_structure_analysis(data)
    print(f'  Significant drivers: {structure["significant_drivers"]}')
    for feat, res in structure.get('results', {}).items():
        print(f'    {feat}: effect={res["effect"]:.4f} p={res["p_value"]:.4f}')