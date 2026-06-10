#!/usr/bin/env python3
"""
stress_test.py - Stress testing automático de portfolios.
Simula performance histórica en crisis (2008, 2020, 2022) y escenarios sintéticos.
"""
import json
import os
import time
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta, timezone
from typing import Dict, Any, List, Optional

from config.settings import get_setting

DATA_DIR = os.environ.get('GITHUB_WORKSPACE', '.')
OUTPUT_DIR = Path(DATA_DIR) / 'Datos'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

ST_CONFIG = get_setting('portfolio.stress_test', {})
PERIODOS = ST_CONFIG.get('periodos', [])
MC_CONFIG = get_setting('portfolio.monte_carlo', {})
N_SIM = MC_CONFIG.get('n_simulaciones', 10000)
N_DIAS = MC_CONFIG.get('n_dias', 252)
VALOR_INICIAL = MC_CONFIG.get('valor_inicial', 100000)
RF = MC_CONFIG.get('rf', 0.05)


class StressTester:
    def __init__(self):
        self.results_path = OUTPUT_DIR / 'stress_test_results.json'
        self._load()

    def _load(self):
        if self.results_path.exists():
            try:
                self.results = json.loads(self.results_path.read_text())
            except:
                self.results = {'stress_tests': [], 'monte_carlo': [], 'timestamp': None}
        else:
            self.results = {'stress_tests': [], 'monte_carlo': [], 'timestamp': None}

    def _save(self):
        self.results_path.write_text(json.dumps(self.results, indent=2))

    def run_historical_stress(
        self,
        tickers: List[str],
        weights: Optional[Dict[str, float]] = None
    ) -> List[Dict]:
        """Ejecuta stress tests en periodos históricos de crisis."""
        import yfinance as yf
        
        stress_results = []
        for periodo in PERIODOS:
            nombre = periodo.get('nombre', 'unknown')
            inicio = periodo.get('inicio')
            fin = periodo.get('fin')
            
            if not inicio or not fin:
                continue
            
            try:
                data = yf.download(tickers, start=inicio, end=fin, progress=False, auto_adjust=True)
                if data.empty:
                    continue
                
                if isinstance(data.columns, pd.MultiIndex):
                    close = data['Close']
                else:
                    close = data
                
                returns = close.pct_change().dropna()
                
                if weights:
                    weighted_ret = returns[list(weights.keys())].dot(
                        pd.Series(weights)[list(weights.keys())]
                    )
                else:
                    weighted_ret = returns.mean(axis=1)
                
                cum_ret = (1 + weighted_ret).cumprod()
                max_dd = (cum_ret / cum_ret.cummax() - 1).min()
                
                sharpe = (weighted_ret.mean() * 252 - RF) / (weighted_ret.std() * np.sqrt(252)) if weighted_ret.std() > 0 else 0
                cagr = (1 + weighted_ret.sum()) ** (252 / len(weighted_ret)) - 1
                ann_vol = weighted_ret.std() * np.sqrt(252)
                
                result = {
                    'periodo': nombre,
                    'inicio': inicio,
                    'fin': fin,
                    'dias': len(weighted_ret),
                    'total_return': float(cum_ret.iloc[-1] - 1),
                    'cagr': float(cagr),
                    'vol_anual': float(ann_vol),
                    'sharpe': float(sharpe),
                    'max_drawdown': float(max_dd),
                    'worst_day': float(weighted_ret.min()),
                    'best_day': float(weighted_ret.max()),
                    'neg_days_pct': float((weighted_ret < 0).mean()),
                    'timestamp': datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z')
                }
                stress_results.append(result)
                print(f'[StressTest] {nombre}: ret={result["total_return"]:.1%}, DD={result["max_drawdown"]:.1%}, Sharpe={result["sharpe"]:.2f}')
                
            except Exception as e:
                print(f'[StressTest] Error en {nombre}: {e}')
        
        self.results['stress_tests'] = stress_results
        self.results['timestamp'] = datetime.utcnow().isoformat() + 'Z'
        self._save()
        return stress_results

    def run_monte_carlo(
        self,
        expected_return: float = 0.10,
        expected_vol: float = 0.20,
        initial_capital: float = VALOR_INICIAL,
        n_simulations: int = N_SIM,
        n_days: int = N_DIAS,
        stress_factor: float = 1.0
    ) -> Dict:
        """
        Simulación Monte Carlo de portfolio.
        
        Args:
            stress_factor: >1 para escenarios de estrés (ej: 2.0 duplica vol)
        """
        np.random.seed(42)
        mu = expected_return / 252 * stress_factor
        sigma = expected_vol / np.sqrt(252) * stress_factor
        
        simulations = np.zeros((n_simulations, n_days))
        for i in range(n_simulations):
            daily_ret = np.random.normal(mu, sigma, n_days)
            simulations[i] = initial_capital * (1 + daily_ret).cumprod()
        
        final_values = simulations[:, -1]
        portfolio_values = simulations[-1, :]
        
        var_95 = float(np.percentile(final_values, 5))
        var_99 = float(np.percentile(final_values, 1))
        cvar_95 = float(final_values[final_values <= var_95].mean()) if (final_values <= var_95).sum() > 0 else var_95
        prob_loss = float((final_values < initial_capital).mean())
        median_final = float(np.median(final_values))
        mean_final = float(np.mean(final_values))
        
        worst_path = float(simulations[np.argmin(final_values), -1])
        best_path = float(simulations[np.argmax(final_values), -1])
        
        result = {
            'n_simulations': n_simulations,
            'n_days': n_days,
            'initial_capital': initial_capital,
            'stress_factor': stress_factor,
            'expected_return': expected_return,
            'expected_vol': expected_vol,
            'median_final': median_final,
            'mean_final': mean_final,
            'var_95': var_95,
            'var_99': var_99,
            'cvar_95': cvar_95,
            'prob_loss': prob_loss,
            'worst_path': worst_path,
            'best_path': best_path,
            'expected_final': initial_capital * (1 + expected_return),
            'downside_ratio': float((mean_final - var_95) / (mean_final - initial_capital)) if mean_final != initial_capital else 0,
            'timestamp': datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z')
        }
        
        self.results['monte_carlo'].append(result)
        self._save()
        
        print(f'[MonteCarlo] {n_simulations} sims x {n_days}d: '
              f'median=${median_final:,.0f}, VaR95=${var_95:,.0f}, '
              f'P(loss)={prob_loss:.1%}')
        return result

    def run_custom_scenario(self, name: str, tickers: List[str], 
                           crash_pct: float = -0.20,
                           recovery_days: int = 60) -> Dict:
        """Scenario sintético: crash súbito + recuperación."""
        import yfinance as yf
        
        end = datetime.now()
        start = end - timedelta(days=252)
        
        data = yf.download(tickers, start=start.strftime('%Y-%m-%d'),
                          end=end.strftime('%Y-%m-%d'), progress=False, auto_adjust=True)
        
        if isinstance(data.columns, pd.MultiIndex):
            close = data['Close']
        else:
            close = data
        
        prices = close.iloc[-1]
        
        crash_prices = prices * (1 + crash_pct)
        
        scenario = {
            'name': name,
            'pre_crash_value': prices.to_dict(),
            'crash_value': crash_prices.to_dict(),
            'crash_pct': crash_pct,
            'impact_per_ticker': {t: float(crash_prices[t] / prices[t] - 1) for t in tickers if t in prices},
            'timestamp': datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z')
        }
        
        print(f'[Scenario] {name}: crash {crash_pct:.0%}, recovery {recovery_days}d')
        return scenario

    def get_summary(self) -> Dict:
        return {
            'n_stress_periods': len(self.results.get('stress_tests', [])),
            'n_mc_simulations': len(self.results.get('monte_carlo', [])),
            'last_updated': self.results.get('timestamp'),
            'worst_stress': min(self.results.get('stress_tests', []), 
                                key=lambda x: x.get('total_return', 0),
                                default={}),
            'best_stress': max(self.results.get('stress_tests', []),
                               key=lambda x: x.get('total_return', 0),
                               default={})
        }


_stress_tester = None


def get_stress_tester() -> StressTester:
    global _stress_tester
    if _stress_tester is None:
        _stress_tester = StressTester()
    return _stress_tester


def run_full_stress_test(tickers: List[str], weights: Optional[Dict[str, float]] = None) -> Dict:
    """Ejecuta todos los stress tests + Monte Carlo."""
    st = get_stress_tester()
    stress = st.run_historical_stress(tickers, weights)
    mc = st.run_monte_carlo()
    return {
        'historical_stress': stress,
        'monte_carlo': mc,
        'summary': st.get_summary()
    }


if __name__ == '__main__':
    print('[StressTest] Test con tickers core...')
    TICKERS = ['NVDA','AAPL','MSFT','GOOGL','AMZN']
    st = get_stress_tester()
    
    # Stress histórico
    st.run_historical_stress(TICKERS)
    
    # Monte Carlo base
    st.run_monte_carlo(expected_return=0.12, expected_vol=0.22)
    
    # Monte Carlo estrés (vol x3)
    st.run_monte_carlo(expected_return=0.12, expected_vol=0.22, stress_factor=3.0)
    
    print(json.dumps(st.get_summary(), indent=2))