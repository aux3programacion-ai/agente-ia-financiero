#!/usr/bin/env python3
"""
multi_asset_rl.py - Multi-asset portfolio optimization with RL.
FinRL-style environment for portfolio allocation, PPO/SAC agents
for multi-asset rebalancing, with transaction costs and slippage.
"""
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple, Union
from dataclasses import dataclass, field
from datetime import datetime
import json
import os
from pathlib import Path

try:
    import gymnasium as gym
    import torch
    import torch.nn as nn
    import torch.optim as optim
    GYM_AVAILABLE = True
    TORCH_AVAILABLE = True
except ImportError:
    GYM_AVAILABLE = False
    TORCH_AVAILABLE = False

from config.settings import get_setting

DATA_DIR = os.environ.get('GITHUB_WORKSPACE', '.')
OUTPUT_DIR = Path(DATA_DIR) / 'Datos' / 'multi_asset_rl'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


class MultiAssetPortfolioEnv(gym.Env if GYM_AVAILABLE else object):
    """Multi-asset portfolio allocation environment."""
    
    metadata = {'render_modes': ['human']}
    
    def __init__(self, prices: pd.DataFrame, returns: pd.DataFrame,
                 features: Optional[pd.DataFrame] = None,
                 window: int = 60, n_assets: int = 10,
                 transaction_cost: float = 0.001,
                 slippage: float = 0.0005, max_leverage: float = 1.0,
                 risk_free_rate: float = 0.05,
                 render_mode: Optional[str] = None):
        super().__init__()
        
        self.prices = prices
        self.returns = returns.fillna(0)
        self.features = features
        self.window = window
        self.n_assets = min(n_assets, len(prices.columns)) if prices is not None else n_assets
        self.transaction_cost = transaction_cost
        self.slippage = slippage
        self.max_leverage = max_leverage
        self.risk_free_rate = risk_free_rate
        self.render_mode = render_mode
        
        if prices is not None:
            self.n_assets = min(n_assets, len(prices.columns))
            self.assets = list(prices.columns[:self.n_assets])
        
        n_features = 0
        if features is not None:
            n_features = len(features.columns)
        
        self.state_dim = self.n_assets * 3 + 2 + n_features
        self.action_dim = self.n_assets + 1  # weights + cash
        
        if GYM_AVAILABLE:
            self.observation_space = gym.spaces.Box(
                -np.inf, np.inf, shape=(self.state_dim,), dtype=np.float32)
            self.action_space = gym.spaces.Box(
                0, 1, shape=(self.action_dim,), dtype=np.float32)
        
        self.reset()
    
    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None):
        if GYM_AVAILABLE:
            super().reset(seed=seed)
        np.random.seed(seed)
        
        self.current_step = self.window
        self.portfolio = np.ones(self.action_dim) / self.action_dim
        self.portfolio_history = [self.portfolio.copy()]
        self.returns_history = []
        self.portfolio_value = 1.0
        self.done = False
        
        return self._get_state(), {}
    
    def _get_state(self) -> np.ndarray:
        if self.features is not None:
            feats = self.features.iloc[self.current_step].values[:self.state_dim - self.n_assets * 3 - 2]
            if len(feats) < self.state_dim - self.n_assets * 3 - 2:
                feats = np.pad(feats, (0, self.state_dim - self.n_assets * 3 - 2 - len(feats)))
        else:
            feats = np.zeros(self.state_dim - self.n_assets * 3 - 2)
        
        rets = self.returns.iloc[self.current_step].values[:self.n_assets]
        window_rets = self.returns.iloc[max(0, self.current_step-20):self.current_step].mean().values[:self.n_assets]
        vol = self.returns.iloc[max(0, self.current_step-20):self.current_step].std().values[:self.n_assets]
        
        state = np.concatenate([
            rets.astype(np.float32),
            window_rets.astype(np.float32),
            vol.astype(np.float32),
            [self.portfolio_value],
            [np.mean(rets) / (np.mean(vol) + 1e-8)],
            feats.astype(np.float32),
        ])
        
        return state[:self.state_dim].astype(np.float32)
    
    def step(self, action: np.ndarray):
        action = np.clip(action, 0, 1)
        action = action / (action.sum() + 1e-8) * self.max_leverage
        
        returns = self.returns.iloc[self.current_step].values[:self.n_assets]
        cash_return = 0.0001  # risk-free proxy
        asset_returns = np.concatenate([returns, [cash_return]])
        
        actual_rets = asset_returns[:len(action)]
        portfolio_ret = np.dot(self.portfolio[:len(action)], actual_rets)
        
        turnover = np.sum(np.abs(action[:len(self.portfolio)] - self.portfolio[:len(action)]))
        cost = turnover * (self.transaction_cost + self.slippage)
        net_return = portfolio_ret - cost
        
        self.portfolio_value *= (1 + net_return)
        self.portfolio = action.copy()
        
        self.returns_history.append(net_return)
        self.portfolio_history.append(self.portfolio.copy())
        self.current_step += 1
        
        if self.current_step >= len(self.returns) - 1:
            self.done = True
        
        # Reward: Sharpe-like with drawdown penalty
        returns_arr = np.array(self.returns_history)
        if len(returns_arr) > 5:
            sharpe = np.mean(returns_arr) / (np.std(returns_arr) + 1e-8) * np.sqrt(252)
            drawdown = (1 - self.portfolio_value / np.maximum(
                np.max([1.0] + [v for v in [self.portfolio_value] if v > 0]), 1e-8))
            reward = sharpe - 2 * drawdown
        else:
            reward = net_return * 100
        
        if self.done:
            final_value = self.portfolio_value
            years = len(self.returns_history) / 252
            total_ret = final_value - 1
            ann_ret = (final_value ** (1 / max(years, 0.1)) - 1) * 100
            reward += ann_ret / 10
        
        return self._get_state(), float(reward), self.done, False, {
            'portfolio_value': self.portfolio_value,
            'return': net_return,
            'turnover': turnover,
            'cost': cost,
        }
    
    def render(self):
        if self.render_mode == 'human':
            print(f'Step: {self.current_step}, Value: {self.portfolio_value:.4f}, '
                  f'Weights: {self.portfolio[:3]}...')
    
    def get_performance(self) -> Dict:
        returns_arr = np.array(self.returns_history)
        n = len(returns_arr)
        if n == 0:
            return {'total_return': 0, 'sharpe': 0, 'max_drawdown': 0}
        
        total_ret = np.prod(1 + returns_arr) - 1
        sharpe = np.mean(returns_arr) / (np.std(returns_arr) + 1e-8) * np.sqrt(252)
        
        cum = np.cumprod(1 + returns_arr)
        running_max = np.maximum.accumulate(cum)
        dd = (cum - running_max) / running_max
        max_dd = np.min(dd)
        
        return {
            'total_return': float(total_ret),
            'sharpe': float(sharpe),
            'max_drawdown': float(max_dd),
            'volatility': float(np.std(returns_arr) * np.sqrt(252)),
            'n_trades': n,
        }


class PortfolioPolicyNetwork:
    """Policy network for portfolio allocation."""
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 128):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.net = None
        if TORCH_AVAILABLE:
            self.net = nn.Sequential(
                nn.Linear(state_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, action_dim),
                nn.Softmax(dim=-1),
            )
    
    def forward(self, state: 'torch.Tensor') -> 'torch.Tensor':
        if not TORCH_AVAILABLE or self.net is None:
            return __import__('torch').zeros(1)
        return self.net(state)


class MultiAssetRLAgent:
    """RL agent for multi-asset portfolio optimization."""
    
    def __init__(self, state_dim: int, action_dim: int,
                 algorithm: str = 'ppo', lr: float = 3e-4,
                 gamma: float = 0.99):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.algorithm = algorithm
        self.gamma = gamma
        self.trained = False
        
        if TORCH_AVAILABLE:
            self.policy = PortfolioPolicyNetwork(state_dim, action_dim)
            self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        else:
            self.policy = None
            self.optimizer = None
    
    def act(self, state: np.ndarray, deterministic: bool = True) -> np.ndarray:
        if not TORCH_AVAILABLE:
            return np.ones(self.action_dim) / self.action_dim
        with torch.no_grad():
            state_t = torch.FloatTensor(state).unsqueeze(0)
            probs = self.policy(state_t).squeeze(0).numpy()
        return probs
    
    def train(self, env: MultiAssetPortfolioEnv, n_episodes: int = 100,
              eval_interval: int = 10) -> Dict:
        if not TORCH_AVAILABLE:
            print('[MultiAssetRL] PyTorch not available.')
            return {'episodes': 0, 'final_sharpe': 0}
        
        episode_rewards = []
        best_reward = -np.inf
        
        for episode in range(n_episodes):
            state, _ = env.reset()
            done = False
            total_reward = 0
            log_probs = []
            rewards = []
            states = []
            
            while not done:
                state_t = torch.FloatTensor(state).unsqueeze(0)
                probs = self.policy(state_t)
                dist = torch.distributions.Categorical(probs)
                action_idx = dist.sample()
                
                action = np.zeros(self.action_dim)
                action[action_idx.item() % self.action_dim] = 1.0
                
                next_state, reward, done, _, _ = env.step(action)
                
                states.append(state)
                log_probs.append(dist.log_prob(action_idx))
                rewards.append(reward)
                state = next_state
                total_reward += reward
            
            episode_rewards.append(total_reward)
            
            # PPO-style update (simplified)
            returns = []
            R = 0
            for r in reversed(rewards):
                R = r + self.gamma * R
                returns.insert(0, R)
            returns = torch.FloatTensor(returns)
            
            # Normalize returns
            if returns.std() > 0:
                returns = (returns - returns.mean()) / (returns.std() + 1e-8)
            
            log_probs_t = torch.stack(log_probs)
            policy_loss = -(log_probs_t * returns.detach()).mean()
            
            self.optimizer.zero_grad()
            policy_loss.backward()
            self.optimizer.step()
            
            if (episode + 1) % eval_interval == 0:
                avg_reward = np.mean(episode_rewards[-eval_interval:])
                print(f'[MultiAssetRL] Ep {episode+1}, Avg Reward: {avg_reward:.2f}')
                
                if avg_reward > best_reward:
                    best_reward = avg_reward
                    self.trained = True
        
        self.trained = True
        
        # Evaluate
        perf = env.get_performance()
        return {
            'episodes': n_episodes,
            'final_sharpe': perf['sharpe'],
            'final_return': perf['total_return'],
            'max_drawdown': perf['max_drawdown'],
        }


class MultiAssetRLOptimizer:
    """High-level multi-asset RL optimizer."""
    
    def __init__(self, algorithm: str = 'ppo'):
        self.algorithm = algorithm
        self.env = None
        self.agent = None
    
    def run(self, prices: pd.DataFrame, features: Optional[pd.DataFrame] = None,
            n_assets: int = 5, window: int = 60, n_episodes: int = 50) -> Dict:
        returns = prices.pct_change().dropna()
        
        # Align
        common_idx = returns.index.intersection(prices.index[window:])
        returns = returns.loc[common_idx]
        prices = prices.loc[common_idx]
        
        if features is not None:
            features = features.loc[features.index.intersection(common_idx)]
        
        n_assets = min(n_assets, len(prices.columns))
        
        self.env = MultiAssetPortfolioEnv(
            prices=prices, returns=returns, features=features,
            window=window, n_assets=n_assets,
            transaction_cost=0.001, slippage=0.0005)
        
        self.agent = MultiAssetRLAgent(
            state_dim=self.env.state_dim,
            action_dim=self.env.action_dim,
            algorithm=self.algorithm)
        
        if TORCH_AVAILABLE:
            train_results = self.agent.train(self.env, n_episodes=n_episodes)
        else:
            train_results = {
                'episodes': 0, 'final_sharpe': 0, 'final_return': 0, 'max_drawdown': 0
            }
        
        perf = self.env.get_performance()
        final = {**train_results, 'env_performance': perf}
        
        (OUTPUT_DIR / 'optimization_result.json').write_text(
            json.dumps(final, indent=2), encoding='utf-8')
        
        return final


if __name__ == '__main__':
    print('[MultiAssetRL] Running portfolio optimization...')
    np.random.seed(42)
    
    n = 1500
    n_assets = 10
    dates = pd.date_range('2022-01-01', periods=n, freq='B')
    
    prices = pd.DataFrame({
        f'Asset_{i}': 100 * np.exp(np.cumsum(
            np.random.randn(n) * 0.02 + 0.0003 * (i % 3)))
        for i in range(n_assets)
    }, index=dates)
    
    features = pd.DataFrame({
        'vol_ma': np.random.randn(n),
        'rsi': np.random.rand(n) * 100,
    }, index=dates)
    
    optimizer = MultiAssetRLOptimizer(algorithm='ppo')
    result = optimizer.run(prices, features, n_assets=5, n_episodes=20)
    
    print(f"Sharpe: {result.get('final_sharpe', 0):.3f}")
    print(f"Return: {result.get('final_return', 0):.3%}")
    print(f"Drawdown: {result.get('max_drawdown', 0):.2%}")
    print(f"Episodes: {result.get('episodes', 0)}")