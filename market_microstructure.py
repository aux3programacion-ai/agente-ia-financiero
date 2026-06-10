#!/usr/bin/env python3
"""
market_microstructure.py - Market microstructure analysis.
Order flow imbalance, VPIN, tick data analysis, bid-ask spread estimator,
Kyle's lambda, and trade classification.
"""
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import json
import os
from pathlib import Path

from config.settings import get_setting

DATA_DIR = os.environ.get('GITHUB_WORKSPACE', '.')
OUTPUT_DIR = Path(DATA_DIR) / 'Datos' / 'microstructure'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ─── Order Imbalance ──────────────────────────────────────────────────────

class OrderFlowImbalance:
    """Calculate order flow imbalance from tick data."""
    
    def __init__(self, volume_weighted: bool = True):
        self.volume_weighted = volume_weighted
    
    def from_ticks(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Expects DataFrame with columns: price, volume, side (buy/sell/aggressive)
        or alternative: trade price, bid, ask, volume.
        """
        result = df.copy()
        
        if 'side' in df.columns:
            buy_vol = df[df['side'].isin(['buy', 'aggressive_buy'])]['volume'].sum()
            sell_vol = df[df['side'].isin(['sell', 'aggressive_sell'])]['volume'].sum()
            result['of_imbalance'] = (buy_vol - sell_vol) / (buy_vol + sell_vol + 1e-8)
        elif all(c in df.columns for c in ['price', 'bid', 'ask']):
            mid = (df['bid'] + df['ask']) / 2
            result['trade_direction'] = np.where(df['price'] > mid, 1,
                                                  np.where(df['price'] < mid, -1, 0))
            result['signed_volume'] = result['trade_direction'] * df['volume']
            result['of_imbalance'] = result['signed_volume'] / (df['volume'].rolling(100).sum() + 1e-8)
        else:
            result['of_imbalance'] = 0.0
        
        return result
    
    def aggregate_bars(self, df: pd.DataFrame, freq: str = '5min') -> pd.DataFrame:
        df = self.from_ticks(df)
        grouped = df.resample(freq, on='timestamp' if 'timestamp' in df.columns else None)
        if grouped.empty:
            df['_bucket'] = pd.cut(range(len(df)), bins=max(1, len(df)//10), labels=False)
            grouped = df.groupby('_bucket')
        
        agg = grouped.agg({
            'of_imbalance': 'mean',
            'price': ['last', 'min', 'max', 'std'],
            'volume': 'sum',
        } if 'volume' in df.columns else {
            'of_imbalance': 'mean',
        })
        
        agg.columns = ['_'.join(c) if isinstance(c, tuple) else c for c in agg.columns]
        return agg


class VPIN:
    """Volume-synchronized Probability of Informed Trading."""
    
    def __init__(self, n_buckets: int = 50, bucket_volume: Optional[int] = None):
        self.n_buckets = n_buckets
        self.bucket_volume = bucket_volume
    
    def compute(self, df: pd.DataFrame, volume_col: str = 'volume',
                price_col: str = 'price') -> pd.DataFrame:
        if len(df) < self.n_buckets:
            return pd.DataFrame({'vpin': [0]})
        
        # Volume buckets
        if self.bucket_volume is None:
            total_vol = df[volume_col].sum()
            self.bucket_volume = int(total_vol / self.n_buckets)
        
        df = df.copy()
        df['cum_vol'] = df[volume_col].cumsum()
        df['bucket'] = (df['cum_vol'] / self.bucket_volume).astype(int)
        
        # Price direction per bucket
        bucketed = df.groupby('bucket').agg({
            price_col: ['first', 'last'],
            volume_col: 'sum',
        })
        bucketed.columns = ['_'.join(c) for c in bucketed.columns]
        bucketed['direction'] = np.sign(bucketed[f'{price_col}_last'] - bucketed[f'{price_col}_first'])
        bucketed['signed_vol'] = bucketed['direction'] * bucketed[f'{volume_col}_sum']
        
        # VPIN = sum(|signed_vol|) / total_vol over rolling window
        window = min(self.n_buckets, len(bucketed) // 2)
        bucketed['abs_signed_vol_sum'] = bucketed['signed_vol'].abs().rolling(window).sum()
        bucketed['total_vol_sum'] = bucketed[f'{volume_col}_sum'].rolling(window).sum()
        bucketed['vpin'] = bucketed['abs_signed_vol_sum'] / (bucketed['total_vol_sum'] + 1e-8)
        
        return bucketed


# ─── Spread Estimators ────────────────────────────────────────────────────

class SpreadEstimator:
    """Estimate bid-ask spread from trade data."""
    
    @staticmethod
    def roll_estimator(prices: pd.Series) -> float:
        """Roll (1984) spread estimator: sp = 2 * sqrt(-cov(dP, dP_{t-1}))."""
        dp = prices.diff().dropna()
        cov = dp.shift(1).cov(dp)
        if cov < 0:
            return float(2 * np.sqrt(-cov))
        return 0.0
    
    @staticmethod
    def corwin_schultz(prices: pd.Series, highs: pd.Series, lows: pd.Series,
                       period: int = 1) -> pd.Series:
        """Corwin-Schultz high-low spread estimator."""
        h_l = np.log(highs / lows)
        h_l_2 = h_l ** 2
        h_l_next = h_l.shift(-period)
        h_l_next_2 = h_l_2.shift(-period)
        
        beta = h_l * h_l_next
        gamma = (h_l + h_l_next) ** 2
        
        alpha = (np.sqrt(2 * beta) - np.sqrt(beta)) / (3 - 2 * np.sqrt(2))
        alpha = alpha - np.sqrt(gamma / (3 - 2 * np.sqrt(2)))
        
        spread = 2 * (np.exp(alpha) - 1) / (1 + np.exp(alpha))
        return spread.clip(lower=0)
    
    @staticmethod
    def effective_spread(trade_price: pd.Series, mid: pd.Series) -> pd.Series:
        """Effective half-spread: 2 * |trade_price - mid| / mid."""
        return 2 * (trade_price - mid).abs() / mid


class KyleLambda:
    """Kyle's lambda - price impact coefficient."""
    
    def __init__(self, window: int = 100):
        self.window = window
    
    def compute(self, df: pd.DataFrame, price_col: str = 'price',
                volume_col: str = 'volume', signed: bool = True) -> pd.Series:
        df = df.copy()
        df['ret'] = df[price_col].pct_change()
        df['signed_vol'] = df[volume_col] * np.sign(df['ret'])
        
        df['lambda'] = df['signed_vol'].rolling(self.window).corr(df['ret'])
        df['lambda'] = df['lambda'].fillna(0)
        return df['lambda']


class TradeClassification:
    """Classify trades as buyer/seller initiated."""
    
    @staticmethod
    def quote_rule(price: pd.Series, bid: pd.Series, ask: pd.Series,
                   aggressiveness: float = 0.5) -> pd.Series:
        """Lee-Ready quote rule."""
        mid = (bid + ask) / 2
        direction = pd.Series(np.zeros(len(price)), index=price.index)
        direction[price > mid + aggressiveness * (ask - mid)] = 1
        direction[price < mid - aggressiveness * (mid - bid)] = -1
        direction[(price == bid) | (price == ask)] = 0
        return direction
    
    @staticmethod
    def tic_rule(price: pd.Series, uptick_threshold: float = 0.0) -> pd.Series:
        """Tick test: classify by price change."""
        dp = price.diff()
        direction = pd.Series(np.zeros(len(price)), index=price.index)
        direction[dp > uptick_threshold] = 1
        direction[dp < -uptick_threshold] = -1
        direction.iloc[0] = 0
        direction = direction.replace(0).ffill().fillna(0)
        return direction


# ─── Liquidity Measures ──────────────────────────────────────────────────

class LiquidityMeasures:
    """Amihud illiquidity, turnover, and depth measures."""
    
    @staticmethod
    def amihud(ret: pd.Series, volume: pd.Series, window: int = 20) -> pd.Series:
        """Amihud illiquidity ratio: |return| / volume."""
        illiq = (ret.abs() / (volume + 1e-8)).rolling(window).mean()
        return illiq * 1e6  # scaled for readability
    
    @staticmethod
    def turnover(volume: pd.Series, shares_outstanding: pd.Series) -> pd.Series:
        """Turnover ratio."""
        return volume / (shares_outstanding + 1e-8)
    
    @staticmethod
    def quoted_spread(bid: pd.Series, ask: pd.Series) -> pd.Series:
        """Quoted relative spread."""
        return (ask - bid) / ((ask + bid) / 2)


# ─── Main Analysis Class ────────────────────────────────────────────────

class MarketMicrostructureAnalyzer:
    """Complete market microstructure analysis pipeline."""
    
    def __init__(self):
        self.ofi = OrderFlowImbalance()
        self.vpin = VPIN()
        self.spread = SpreadEstimator()
        self.kyle = KyleLambda()
        self.classifier = TradeClassification()
        self.liquidity = LiquidityMeasures()
        self.results: Dict[str, Any] = {}
    
    def analyze(self, df: pd.DataFrame) -> Dict[str, Any]:
        results = {}
        
        if all(c in df.columns for c in ['bid', 'ask']):
            results['quoted_spread'] = self.liquidity.quoted_spread(df['bid'], df['ask']).mean()
            results['effective_spread'] = self.spread.effective_spread(
                df.get('price', (df['bid']+df['ask'])/2),
                (df['bid']+df['ask'])/2
            ).mean() if 'price' in df else 0
        
        if 'price' in df and 'volume' in df:
            results['amihud'] = self.liquidity.amihud(
                df['price'].pct_change(), df['volume']
            ).mean()
            results['roll_spread'] = self.spread.roll_estimator(df['price'])
        
        if 'price' in df:
            ofi_results = self.ofi.from_ticks(df)
            results['ofi_mean'] = float(ofi_results['of_imbalance'].mean())
            results['ofi_std'] = float(ofi_results['of_imbalance'].std())
            
            vpin_df = self.vpin.compute(df)
            results['vpin_mean'] = float(vpin_df['vpin'].mean()) if 'vpin' in vpin_df else 0
            results['vpin_max'] = float(vpin_df['vpin'].max()) if 'vpin' in vpin_df else 0
        
        if 'price' in df:
            kyle_series = self.kyle.compute(df)
            results['kyle_lambda'] = float(kyle_series.mean() if hasattr(kyle_series, 'mean') else 0.0)
        
        self.results = results
        
        path = OUTPUT_DIR / 'microstructure_results.json'
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2)
        
        return results
    
    def generate_market_quality_report(self) -> str:
        report_path = OUTPUT_DIR / 'market_quality_report.md'
        report = "# Market Microstructure Report\n\n"
        report += f"Generated: {datetime.now().isoformat()}\n\n"
        if self.results:
            report += "## Key Metrics\n"
            for k, v in self.results.items():
                report += f"- **{k}**: {v:.6f}\n"
        report_path.write_text(report, encoding='utf-8')
        return str(report_path)


if __name__ == '__main__':
    print('[Microstructure] Running analysis...')
    np.random.seed(42)
    n = 10000
    mid = 100 + np.cumsum(np.random.randn(n) * 0.1)
    spread = 0.05 + np.random.rand(n) * 0.02
    df = pd.DataFrame({
        'price': mid + (np.random.rand(n) - 0.5) * spread,
        'bid': mid - spread / 2,
        'ask': mid + spread / 2,
        'volume': np.random.randint(100, 10000, n),
        'side': np.random.choice(['buy', 'sell', 'aggressive_buy', 'aggressive_sell'], n),
    })
    
    analyzer = MarketMicrostructureAnalyzer()
    results = analyzer.analyze(df)
    for k, v in results.items():
        print(f"  {k}: {v:.6f}")
    report = analyzer.generate_market_quality_report()
    print(f"Report: {report}")