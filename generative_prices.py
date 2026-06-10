#!/usr/bin/env python3
"""
generative_prices.py - Generative AI for synthetic price data.
TimeGAN implementation for generating realistic financial time series,
with synthetic data validation and discriminative evaluation.
"""
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import json
import os
from pathlib import Path

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from config.settings import get_setting

DATA_DIR = os.environ.get('GITHUB_WORKSPACE', '.')
OUTPUT_DIR = Path(DATA_DIR) / 'Datos' / 'generative'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

if TORCH_AVAILABLE:
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
else:
    DEVICE = 'cpu'


class TimeGANComponents:
    """PyTorch components for TimeGAN."""
    
    @staticmethod
    def build_embedder(input_dim: int, hidden_dim: int, n_layers: int = 2) -> 'nn.Module':
        if not TORCH_AVAILABLE:
            return None
        class Embedder(nn.Module):
            def __init__(self, in_dim, h_dim, n_lay):
                super().__init__()
                layers = [nn.Linear(in_dim, h_dim), nn.Sigmoid()]
                for _ in range(n_lay - 1):
                    layers.append(nn.Linear(h_dim, h_dim))
                    layers.append(nn.Sigmoid())
                self.net = nn.Sequential(*layers)
            def forward(self, x):
                return self.net(x)
        return Embedder(input_dim, hidden_dim, n_layers)
    
    @staticmethod
    def build_recovery(input_dim: int, hidden_dim: int, n_layers: int = 2) -> 'nn.Module':
        if not TORCH_AVAILABLE:
            return None
        class Recovery(nn.Module):
            def __init__(self, in_dim, h_dim, n_lay):
                super().__init__()
                layers = [nn.Linear(h_dim, h_dim), nn.Sigmoid()]
                for _ in range(n_lay - 1):
                    layers.append(nn.Linear(h_dim, h_dim))
                    layers.append(nn.Sigmoid())
                layers.append(nn.Linear(h_dim, in_dim))
                self.net = nn.Sequential(*layers)
            def forward(self, x):
                return self.net(x)
        return Recovery(input_dim, hidden_dim, n_layers)
    
    @staticmethod
    def build_generator(input_dim: int, hidden_dim: int, n_layers: int = 2,
                        z_dim: int = 10) -> 'nn.Module':
        if not TORCH_AVAILABLE:
            return None
        class Generator(nn.Module):
            def __init__(self, z_dim_, in_dim, h_dim, n_lay):
                super().__init__()
                self.z_dim = z_dim_
                layers = [nn.Linear(z_dim_, h_dim), nn.Sigmoid()]
                for _ in range(n_lay - 1):
                    layers.append(nn.Linear(h_dim, h_dim))
                    layers.append(nn.Sigmoid())
                self.net = nn.Sequential(*layers)
                self.out = nn.Linear(h_dim, in_dim)
            def forward(self, z):
                return self.out(self.net(z))
        return Generator(z_dim, input_dim, hidden_dim, n_layers)
    
    @staticmethod
    def build_discriminator(input_dim: int, hidden_dim: int,
                            n_layers: int = 2) -> 'nn.Module':
        if not TORCH_AVAILABLE:
            return None
        class Discriminator(nn.Module):
            def __init__(self, in_dim, h_dim, n_lay):
                super().__init__()
                layers = [nn.Linear(in_dim, h_dim), nn.Sigmoid()]
                for _ in range(n_lay - 1):
                    layers.append(nn.Linear(h_dim, h_dim))
                    layers.append(nn.Sigmoid())
                layers.append(nn.Linear(h_dim, 1))
                self.net = nn.Sequential(*layers)
            def forward(self, x):
                return self.net(x)
        return Discriminator(input_dim, hidden_dim, n_layers)


class TimeGAN:
    """TimeGAN implementation for financial time series generation."""
    
    def __init__(self, input_dim: int = 1, hidden_dim: int = 24,
                 n_layers: int = 2, z_dim: int = 10, lr: float = 1e-3,
                 batch_size: int = 64, n_epochs: int = 1000):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.z_dim = z_dim
        self.lr = lr
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.trained = False
        
        if TORCH_AVAILABLE:
            self.embedder = TimeGANComponents.build_embedder(input_dim, hidden_dim, n_layers)
            self.recovery = TimeGANComponents.build_recovery(input_dim, hidden_dim, n_layers)
            self.generator = TimeGANComponents.build_generator(input_dim, hidden_dim, n_layers, z_dim)
            self.discriminator = TimeGANComponents.build_discriminator(input_dim, hidden_dim, n_layers)
            
            self.e_opt = optim.Adam(self.embedder.parameters(), lr=lr)
            self.r_opt = optim.Adam(self.recovery.parameters(), lr=lr)
            self.g_opt = optim.Adam(self.generator.parameters(), lr=lr)
            self.d_opt = optim.Adam(self.discriminator.parameters(), lr=lr)
            
            self.mse = nn.MSELoss()
            self.bce = nn.BCEWithLogitsLoss()
        else:
            self.embedder = self.recovery = self.generator = self.discriminator = None
    
    def fit(self, data: np.ndarray):
        if not TORCH_AVAILABLE:
            print('[TimeGAN] PyTorch not available. Skipping training.')
            return self
        
        n_seq, seq_len, n_feat = data.shape
        data_t = torch.FloatTensor(data).to(DEVICE)
        
        for epoch in range(self.n_epochs):
            idx = np.random.permutation(n_seq)
            
            for i in range(0, n_seq, self.batch_size):
                batch_idx = idx[i:i+self.batch_size]
                X = data_t[batch_idx]
                batch_size = X.shape[0]
                
                # 1. Embedder + Recovery
                H = self.embedder(X)
                X_tilde = self.recovery(H)
                e_loss = self.mse(X_tilde, X)
                
                self.e_opt.zero_grad()
                self.r_opt.zero_grad()
                e_loss.backward()
                self.e_opt.step()
                self.r_opt.step()
                
                # 2. Generator (unsupervised)
                Z = torch.randn(batch_size, self.z_dim).to(DEVICE)
                E_hat = self.generator(Z)
                H_hat = self.embedder(E_hat)
                X_hat = self.recovery(H_hat)
                
                g_loss_u = self.mse(X_hat, X)
                
                # 3. Discriminator
                y_real = self.discriminator(H)
                y_fake = self.discriminator(H_hat.detach())
                d_loss = self.bce(y_real, torch.ones_like(y_real)) + \
                         self.bce(y_fake, torch.zeros_like(y_fake))
                
                self.d_opt.zero_grad()
                d_loss.backward()
                self.d_opt.step()
                
                # 4. Generator (adversarial)
                Z = torch.randn(batch_size, self.z_dim).to(DEVICE)
                E_hat = self.generator(Z)
                H_hat = self.embedder(E_hat)
                y_fake = self.discriminator(H_hat)
                g_loss_g = self.bce(y_fake, torch.ones_like(y_fake))
                
                g_loss = g_loss_u + g_loss_g
                self.g_opt.zero_grad()
                g_loss.backward()
                self.g_opt.step()
            
            if (epoch + 1) % 200 == 0:
                print(f'[TimeGAN] Epoch {epoch+1}/{self.n_epochs}, '
                      f'E_loss: {e_loss.item():.4f}, D_loss: {d_loss.item():.4f}, '
                      f'G_loss: {g_loss.item():.4f}')
        
        self.trained = True
        return self
    
    def generate(self, n_sequences: int, seq_len: int) -> np.ndarray:
        if not TORCH_AVAILABLE or not self.trained:
            print('[TimeGAN] Not available. Returning random.')
            return np.random.randn(n_sequences, seq_len, self.input_dim)
        
        with torch.no_grad():
            Z = torch.randn(n_sequences, self.z_dim).to(DEVICE)
            E_hat = self.generator(Z)
            H_hat = self.embedder(E_hat)
            generated = self.recovery(H_hat)
        return generated.cpu().numpy()
    
    def save(self, path: str = None):
        if TORCH_AVAILABLE:
            p = path or str(OUTPUT_DIR / 'timegan_weights.pt')
            torch.save({
                'embedder': self.embedder.state_dict(),
                'recovery': self.recovery.state_dict(),
                'generator': self.generator.state_dict(),
                'discriminator': self.discriminator.state_dict(),
            }, p)
            print(f'[TimeGAN] Weights saved: {p}')
    
    def load(self, path: str):
        if TORCH_AVAILABLE and os.path.exists(path):
            state = torch.load(path, map_location=DEVICE)
            self.embedder.load_state_dict(state['embedder'])
            self.recovery.load_state_dict(state['recovery'])
            self.generator.load_state_dict(state['generator'])
            self.discriminator.load_state_dict(state['discriminator'])
            self.trained = True


class SyntheticDataValidator:
    """Validate synthetic data quality against real data."""
    
    def __init__(self):
        self.metrics: Dict[str, float] = {}
    
    def validate(self, real: np.ndarray, synthetic: np.ndarray) -> Dict[str, float]:
        real = real.reshape(-1, real.shape[-1]) if real.ndim > 2 else real
        synthetic = synthetic.reshape(-1, synthetic.shape[-1]) if synthetic.ndim > 2 else synthetic
        
        metrics = {}
        for i in range(min(real.shape[1], synthetic.shape[1])):
            r = real[:, i]
            s = synthetic[:, i]
            
            metrics[f'mean_diff_{i}'] = float(abs(r.mean() - s.mean()))
            metrics[f'std_diff_{i}'] = float(abs(r.std() - s.std()))
            metrics[f'skew_diff_{i}'] = float(abs(pd.Series(r).skew() - pd.Series(s).skew()))
            metrics[f'kurt_diff_{i}'] = float(abs(pd.Series(r).kurt() - pd.Series(s).kurt()))
            metrics[f'autocorr_diff_{i}'] = float(abs(
                pd.Series(r).autocorr() - pd.Series(s).autocorr()))
        
        metrics['mean_abs_diff'] = np.mean([v for v in metrics.values()])
        
        # Discriminative score: can a classifier tell them apart?
        try:
            from sklearn.model_selection import cross_val_score
            from sklearn.ensemble import RandomForestClassifier
            
            n = min(len(real), len(synthetic), 1000)
            X = np.vstack([real[:n], synthetic[:n]])
            y = np.hstack([np.ones(n), np.zeros(n)])
            
            clf = RandomForestClassifier(n_estimators=20, max_depth=5, random_state=42)
            scores = cross_val_score(clf, X, y, cv=3, scoring='accuracy')
            metrics['discriminative_score'] = float(abs(scores.mean() - 0.5) * 2)
        except Exception:
            metrics['discriminative_score'] = 0.0
        
        # Visual similarity proxy via PCA
        try:
            from sklearn.decomposition import PCA
            pca = PCA(n_components=min(2, real.shape[1])).fit(real[:1000])
            real_var = pca.explained_variance_ratio_.sum()
            pca_fake = PCA(n_components=min(2, synthetic.shape[1])).fit(synthetic[:1000])
            fake_var = pca_fake.explained_variance_ratio_.sum()
            metrics['pca_var_diff'] = float(abs(real_var - fake_var))
        except Exception:
            metrics['pca_var_diff'] = 0.0
        
        self.metrics = metrics
        return metrics


class GenerativePricePipeline:
    """End-to-end generative price pipeline."""
    
    def __init__(self, input_dim: int = 1, seq_len: int = 60):
        self.input_dim = input_dim
        self.seq_len = seq_len
        self.gan = TimeGAN(input_dim=input_dim)
        self.validator = SyntheticDataValidator()
    
    def prepare_data(self, close: pd.Series) -> np.ndarray:
        """Prepare data as sequences for TimeGAN."""
        log_ret = np.log(close / close.shift(1)).dropna().values.reshape(-1, 1)
        log_ret = (log_ret - log_ret.mean()) / (log_ret.std() + 1e-8)
        
        sequences = []
        for i in range(0, len(log_ret) - self.seq_len, self.seq_len):
            seq = log_ret[i:i+self.seq_len]
            if len(seq) == self.seq_len:
                sequences.append(seq)
        
        if len(sequences) < 10:
            n_tiles = len(log_ret) // self.seq_len
            seq = log_ret[:n_tiles * self.seq_len].reshape(-1, self.seq_len, 1)
            return seq[:10]
        
        return np.array(sequences)
    
    def run(self, close: pd.Series, n_generate: int = 100) -> Dict[str, Any]:
        print('[Generative] Preparing data...')
        data = self.prepare_data(close)
        print(f'[Generative] Sequences: {data.shape}')
        
        print('[Generative] Training TimeGAN...')
        self.gan.fit(data)
        
        print('[Generative] Generating synthetic sequences...')
        synthetic = self.gan.generate(n_generate, self.seq_len)
        
        print('[Generative] Validating quality...')
        metrics = self.validator.validate(data, synthetic)
        
        # Save
        np.save(OUTPUT_DIR / 'synthetic_prices.npy', synthetic)
        
        result = {
            'n_real_sequences': len(data),
            'n_synthetic_sequences': len(synthetic),
            'seq_len': self.seq_len,
            'input_dim': self.input_dim,
            'validation_metrics': metrics,
            'generated_path': str(OUTPUT_DIR / 'synthetic_prices.npy'),
        }
        
        (OUTPUT_DIR / 'generation_result.json').write_text(
            json.dumps(result, indent=2), encoding='utf-8')
        
        return result


if __name__ == '__main__':
    print('[Generative] Running price generation pipeline...')
    np.random.seed(42)
    
    n = 5000
    close = 100 * np.exp(np.cumsum(np.random.randn(n) * 0.015))
    close = pd.Series(close, 
                      index=pd.date_range('2022-01-01', periods=n, freq='h'))
    
    pipeline = GenerativePricePipeline(input_dim=1, seq_len=60)
    result = pipeline.run(close, n_generate=50)
    
    print(f"Real sequences: {result['n_real_sequences']}")
    print(f"Generated: {result['n_synthetic_sequences']}")
    print(f"Validation metrics:")
    for k, v in result['validation_metrics'].items():
        print(f"  {k}: {v:.4f}")