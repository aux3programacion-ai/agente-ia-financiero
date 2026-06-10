#!/usr/bin/env python3
"""
rl_position_sizing.py - Reinforcement Learning para position sizing.
Gym environment + PPO/SAC agent para optimizar pesos de portafolio.
State: market regime, portfolio stats, ticker features, risk metrics.
Action: position weights per ticker.
Reward: Sharpe ratio + drawdown penalty.
"""
import json
import os
import time
import math
import numpy as np
from pathlib import Path
from collections import deque
from typing import Dict, Any, List, Optional, Tuple, Callable
from dataclasses import dataclass

try:
    import gymnasium as gym
    from gymnasium import spaces
    GYM_AVAILABLE = True
except ImportError:
    GYM_AVAILABLE = False

try:
    from stable_baselines3 import PPO, SAC, A2C, DDPG
    from stable_baselines3.common.vec_env import DummyVecEnv
    from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold
    from stable_baselines3.common.monitor import Monitor
    SB3_AVAILABLE = True
except ImportError:
    SB3_AVAILABLE = False

from config.settings import get_setting
from model_store import get_model_store

DATA_DIR = os.environ.get('GITHUB_WORKSPACE', '.')
OUTPUT_DIR = Path(DATA_DIR) / 'Datos' / 'rl_sizing'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TICKERS = get_setting('tickers.core', [])
REGIME_SIZING = get_setting('portfolio.regime_sizing', {})
MC_CONFIG = get_setting('portfolio.monte_carlo', {})
N_ASSETS = len(TICKERS) if TICKERS else 10


class PortfolioEnv(gym.Env if GYM_AVAILABLE else object):
    """
    Gym environment for portfolio position sizing.
    
    State space:
    - Price returns (N assets, last 20 days)
    - Technical indicators (N assets, 5 features each)
    - Portfolio state (cash %, current weights, PnL)
    - Market regime (one-hot: 4 regimes)
    - Risk metrics (VaR, volatility, correlation)
    
    Action space:
    - Continuous weights per asset (-1.0 to 1.0, including short)
    - Cash reserve (0.0 to 0.5)
    
    Reward: Sharpe ratio over rebalance window - drawdown_penalty
    """
    
    def __init__(self,
                 price_data: Optional[np.ndarray] = None,
                 feature_data: Optional[np.ndarray] = None,
                 tickers: List[str] = None,
                 initial_capital: float = 100000.0,
                 lookback: int = 20,
                 rebalance_freq: int = 5,
                 transaction_cost: float = 0.001,
                 max_position: float = 0.15,
                 max_leverage: float = 1.0,
                 window_length: int = 504,
                 seed: int = 42):
        
        super().__init__()
        self.tickers = tickers or TICKERS[:N_ASSETS]
        self.n_assets = len(self.tickers)
        self.initial_capital = initial_capital
        self.lookback = lookback
        self.rebalance_freq = rebalance_freq
        self.transaction_cost = transaction_cost
        self.max_position = max_position
        self.max_leverage = max_leverage
        self.window_length = window_length
        self.seed = seed
        
        # Price data: [time, assets]
        if price_data is not None:
            self.price_data = price_data
        else:
            self.price_data = self._generate_prices()
        
        self.n_timesteps = len(self.price_data)
        
        # Feature data: [time, assets, features]
        if feature_data is not None:
            self.feature_data = feature_data
        else:
            self.feature_data = np.zeros((self.n_timesteps, self.n_assets, 1))
        
        self.n_features = self.feature_data.shape[2] if feature_data is not None else 1
        
        state_dim = (self.n_assets * self.lookback) + (self.n_assets * self.n_features) + self.n_assets + 4 + 4
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(state_dim,), dtype=np.float32)
        self.action_space = spaces.Box(low=-self.max_position, high=self.max_position,
                                       shape=(self.n_assets + 1,), dtype=np.float32)
        
        self.reset()
    
    def _generate_prices(self) -> np.ndarray:
        np.random.seed(self.seed)
        n = 1000
        prices = np.zeros((n, self.n_assets))
        prices[0] = np.random.uniform(50, 500, self.n_assets)
        
        for t in range(1, n):
            ret = np.random.randn(self.n_assets) * 0.015 + 0.0005
            prices[t] = prices[t-1] * (1 + ret)
        
        return prices
    
    def _get_observation(self, t: int) -> np.ndarray:
        returns = self.price_data[max(0, t-self.lookback):t]
        if len(returns) < self.lookback:
            pad = np.zeros((self.lookback - len(returns), self.n_assets))
            returns = np.vstack([pad, returns])
        
        ret_features = returns.flatten()
        
        tech_features = self.feature_data[t].flatten()
        
        weights_flat = self.current_weights.flatten()
        
        regime_onehot = np.zeros(4)
        if self.current_regime == 'ALCISTA':
            regime_onehot[0] = 1
        elif self.current_regime == 'BAJISTA':
            regime_onehot[1] = 1
        elif self.current_regime == 'LATERAL':
            regime_onehot[2] = 1
        else:
            regime_onehot[3] = 1
        
        risk_vec = np.array([
            self.var_95,
            self.volatility,
            self.sharpe_window,
            self.max_drawdown_window
        ])
        
        return np.concatenate([ret_features, tech_features, weights_flat, regime_onehot, risk_vec]).astype(np.float32)
    
    def _calculate_reward(self, new_prices: np.ndarray) -> float:
        """Reward: portfolio return - transaction cost - drawdown penalty."""
        old_value = self.portfolio_value
        new_weights = self.current_weights
        
        asset_returns = (new_prices - self.prev_prices) / self.prev_prices
        
        portfolio_return = np.dot(new_weights, asset_returns)
        
        turnover = np.sum(np.abs(new_weights - self.prev_weights))
        tc = turnover * self.transaction_cost
        
        net_return = portfolio_return - tc
        
        self.portfolio_value *= (1 + net_return)
        
        self.returns_window.append(net_return)
        self.equity_window.append(self.portfolio_value)
        
        if len(self.returns_window) > 20:
            self.returns_window.popleft()
            self.equity_window.popleft()
        
        sharpe = 0.0
        if len(self.returns_window) >= 10:
            mu = np.mean(self.returns_window) * 252
            sigma = np.std(self.returns_window) * np.sqrt(252)
            sharpe = mu / (sigma + 1e-8)
        
        eq = np.array(self.equity_window)
        running_max = np.maximum.accumulate(eq)
        drawdowns = (eq - running_max) / running_max
        current_dd = abs(min(0, drawdowns[-1])) if len(drawdowns) > 0 else 0
        max_dd_window = abs(min(0, np.min(drawdowns))) if len(drawdowns) > 0 else 0
        
        dd_penalty = current_dd * 2.0 + max_dd_window * 1.0
        
        reward = sharpe * 0.01 - dd_penalty
        
        if self.portfolio_value < self.initial_capital * 0.5:
            reward -= 5.0
        
        self.sharpe_window = sharpe
        self.max_drawdown_window = max_dd_window
        
        return float(reward)
    
    def reset(self, seed: Optional[int] = None):
        super().reset(seed=seed)
        self.current_step = np.random.randint(self.lookback + 10, self.n_timesteps - self.window_length)
        self.portfolio_value = self.initial_capital
        self.cash = self.initial_capital
        self.current_weights = np.zeros(self.n_assets)
        self.prev_weights = np.zeros(self.n_assets)
        self.prev_prices = self.price_data[self.current_step]
        self.current_regime = np.random.choice(['ALCISTA', 'BAJISTA', 'LATERAL', 'INCIERTO'])
        self.var_95 = 0.02
        self.volatility = 0.01
        self.sharpe_window = 0.0
        self.max_drawdown_window = 0.0
        
        self.returns_window = deque(maxlen=20)
        self.equity_window = deque(maxlen=20)
        self.equity_window.append(self.portfolio_value)
        self.actions_taken = []
        
        return self._get_observation(self.current_step), {}
    
    def step(self, action: np.ndarray):
        """Execute action and return next state, reward, done, info."""
        weights = action[:self.n_assets]
        cash_reserve = abs(action[-1])
        
        total_long = max(0, np.sum(weights[weights > 0]))
        total_short = min(0, np.sum(weights[weights < 0]))
        total_exposure = total_long + abs(total_short)
        
        if total_exposure > self.max_leverage:
            scale = self.max_leverage / max(total_exposure, 1e-8)
            weights *= scale
        
        if cash_reserve > 0.5:
            cash_reserve = 0.5
        
        self.current_weights = weights * (1 - cash_reserve)
        self.prev_prices = self.price_data[self.current_step]
        
        self.current_step += 1
        done = self.current_step >= self.n_timesteps - 1
        
        new_prices = self.price_data[self.current_step]
        reward = self._calculate_reward(new_prices) if not done else 0.0
        
        self.prev_weights = self.current_weights.copy()
        self.current_regime = np.random.choice(['ALCISTA', 'BAJISTA', 'LATERAL', 'INCIERTO'])
        
        current_ret = (new_prices - self.prev_prices) / self.prev_prices
        self.volatility = float(np.std(current_ret)) if len(current_ret) > 0 else 0.01
        self.var_95 = float(np.percentile(current_ret, 5)) if len(current_ret) > 0 else -0.02
        
        self.actions_taken.append({
            'step': self.current_step,
            'portfolio_value': self.portfolio_value,
            'weights': weights.tolist(),
            'reward': reward,
            'regime': self.current_regime
        })
        
        obs = self._get_observation(self.current_step)
        truncated = False
        
        return obs, reward, done, truncated, {'portfolio_value': self.portfolio_value}
    
    def render(self):
        pass


class RLPositionSizing:
    def __init__(self, 
                 algo: str = 'PPO',
                 total_timesteps: int = 50000,
                 learning_rate: float = 3e-4,
                 n_steps: int = 2048,
                 batch_size: int = 64,
                 gamma: float = 0.99,
                 gae_lambda: float = 0.95,
                 clip_range: float = 0.2,
                 ent_coef: float = 0.01,
                 tensorboard_log: Optional[str] = None):
        
        self.algo = algo
        self.total_timesteps = total_timesteps
        self.learning_rate = learning_rate
        self.model = None
        self.env = None
        self.store = get_model_store()
        self.tensorboard_log = tensorboard_log or str(OUTPUT_DIR / 'tensorboard')
        
        self.hyperparams = {
            'algo': algo,
            'total_timesteps': total_timesteps,
            'learning_rate': learning_rate,
            'n_steps': n_steps,
            'batch_size': batch_size,
            'gamma': gamma,
            'gae_lambda': gae_lambda,
            'clip_range': clip_range,
            'ent_coef': ent_coef
        }
    
    def create_env(self, price_data: Optional[np.ndarray] = None, **env_kwargs):
        if not GYM_AVAILABLE:
            raise ImportError('Install gymnasium: pip install gymnasium')
        
        env = PortfolioEnv(price_data=price_data, **env_kwargs)
        self.env = Monitor(env)
        return self.env
    
    def train(self, env=None, save: bool = True) -> Dict:
        if not SB3_AVAILABLE:
            raise ImportError('Install stable-baselines3: pip install stable-baselines3')
        if not GYM_AVAILABLE:
            raise ImportError('Install gymnasium: pip install gymnasium')
        
        if env is not None:
            self.env = Monitor(env)
        if self.env is None:
            self.create_env()
        
        vec_env = DummyVecEnv([lambda: self.env])
        
        if self.algo == 'PPO':
            self.model = PPO(
                'MlpPolicy', vec_env,
                learning_rate=self.learning_rate,
                n_steps=self.hyperparams['n_steps'],
                batch_size=self.hyperparams['batch_size'],
                gamma=self.hyperparams['gamma'],
                gae_lambda=self.hyperparams['gae_lambda'],
                clip_range=self.hyperparams['clip_range'],
                ent_coef=self.hyperparams['ent_coef'],
                verbose=1,
                tensorboard_log=self.tensorboard_log
            )
        elif self.algo == 'SAC':
            self.model = SAC(
                'MlpPolicy', vec_env,
                learning_rate=self.learning_rate,
                buffer_size=100000,
                batch_size=self.hyperparams['batch_size'],
                gamma=self.hyperparams['gamma'],
                verbose=1,
                tensorboard_log=self.tensorboard_log
            )
        elif self.algo == 'A2C':
            self.model = A2C(
                'MlpPolicy', vec_env,
                learning_rate=self.learning_rate,
                n_steps=self.hyperparams['n_steps'],
                gamma=self.hyperparams['gamma'],
                gae_lambda=self.hyperparams['gae_lambda'],
                verbose=1,
                tensorboard_log=self.tensorboard_log
            )
        elif self.algo == 'DDPG':
            self.model = DDPG(
                'MlpPolicy', vec_env,
                learning_rate=self.learning_rate,
                buffer_size=100000,
                batch_size=self.hyperparams['batch_size'],
                gamma=self.hyperparams['gamma'],
                verbose=1,
                tensorboard_log=self.tensorboard_log
            )
        else:
            raise ValueError(f'Unknown algorithm: {self.algo}')
        
        start = time.time()
        callback = StopTrainingOnRewardThreshold(reward_threshold=50, verbose=1)
        
        self.model.learn(
            total_timesteps=self.total_timesteps,
            callback=callback,
            progress_bar=True
        )
        
        elapsed = time.time() - start
        
        if save:
            version = self.store.save_model(
                model=self.model.policy,
                name=f'rl_{self.algo.lower()}_sizing',
                regime='global',
                params=self.hyperparams,
                metrics={
                    'total_timesteps': self.total_timesteps,
                    'training_time_s': round(elapsed, 2),
                    'reward_threshold': 50
                }
            )
            model_path = OUTPUT_DIR / f'rl_{self.algo.lower()}_sizing.zip'
            self.model.save(str(model_path))
        
        print(f'[RL] {self.algo} trained: {self.total_timesteps} steps in {elapsed:.1f}s')
        
        return {
            'algo': self.algo,
            'total_timesteps': self.total_timesteps,
            'training_time_s': round(elapsed, 2),
            'model_path': str(model_path) if save else None
        }
    
    def load(self, path: Optional[str] = None) -> bool:
        if path is None:
            path = OUTPUT_DIR / f'rl_{self.algo.lower()}_sizing.zip'
        
        if not Path(path).exists():
            return False
        
        try:
            if self.algo == 'PPO':
                self.model = PPO.load(path)
            elif self.algo == 'SAC':
                self.model = SAC.load(path)
            elif self.algo == 'A2C':
                self.model = A2C.load(path)
            else:
                return False
            return True
        except Exception as e:
            print(f'[RL] Load failed: {e}')
            return False
    
    def predict_weights(self, observation: np.ndarray) -> Dict[str, float]:
        if self.model is None:
            raise ValueError('Model not loaded')
        
        action, _ = self.model.predict(observation, deterministic=True)
        n_assets = len(action) - 1
        
        ticker_weights = {self.env.get_attr('tickers')[0][i] if hasattr(self.env, 'get_attr') else f'asset_{i}': 
                         float(action[i]) for i in range(n_assets)}
        cash_reserve = float(action[-1])
        
        return {
            'weights': ticker_weights,
            'cash_reserve': cash_reserve,
            'n_positions': sum(1 for w in ticker_weights.values() if abs(w) > 0.01),
            'total_exposure': sum(abs(w) for w in ticker_weights.values()),
            'leverage': sum(abs(w) for w in ticker_weights.values())
        }
    
    def create_wrapper(self, env) -> Callable:
        """Returns a function that takes market state and returns position sizes."""
        def sizing_fn(market_state: Dict) -> Dict[str, float]:
            obs = env.reset()[0]
            return self.predict_weights(obs)
        return sizing_fn


class RLSizer:
    """Convenient wrapper for training and inference."""
    
    def __init__(self, algo='PPO', total_timesteps=20000, **kwargs):
        self.impl = RLPositionSizing(algo=algo, total_timesteps=total_timesteps, **kwargs)
        self.algo = algo
    
    def train(self, env=None, **kwargs):
        """Train the RL agent."""
        return self.impl.train(env=env, **kwargs)
    
    def predict(self, obs):
        """Predict weights for given observation."""
        return self.impl.predict_weights(obs)
    
    def save(self, path=None):
        if path is None:
            path = OUTPUT_DIR / f'rl_{self.algo.lower()}_sizer.zip'
        if self.impl.model:
            self.impl.model.save(str(path))
        return path


def train_rl_sizer(algo: str = 'PPO', timesteps: int = 20000, tickers: List[str] = None) -> RLSizer:
    if tickers is None:
        tickers = TICKERS[:10]
    
    sizer = RLSizer(algo=algo, total_timesteps=timesteps)
    env = PortfolioEnv(tickers=tickers, n_assets=len(tickers))
    result = sizer.train(env)
    
    return sizer


if __name__ == '__main__':
    print(f'[RL] Gym available: {GYM_AVAILABLE}, SB3 available: {SB3_AVAILABLE}')
    
    if not GYM_AVAILABLE:
        print('[RL] Install: pip install gymnasium')
    if not SB3_AVAILABLE:
        print('[RL] Install: pip install stable-baselines3')
    
    if GYM_AVAILABLE:
        env = PortfolioEnv(tickers=TICKERS[:5], n_assets=5, window_length=200)
        obs, _ = env.reset()
        print(f'  Observation dim: {obs.shape}')
        print(f'  Action dim: {env.action_space.shape[0]}')
        
        if SB3_AVAILABLE:
            sizer = train_rl_sizer(algo='PPO', timesteps=5000, tickers=TICKERS[:5])
            weights = sizer.predict(obs)
            print(f'  Predicted weights: {weights}')
            sizer.save()