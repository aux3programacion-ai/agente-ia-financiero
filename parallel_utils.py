#!/usr/bin/env python3
"""
parallel_utils.py - Utilidades de paralelización para descarga y procesamiento de tickers.
Reduce drásticamente el tiempo de ejecución de los pipelines.
"""
import time
import os
from pathlib import Path
from typing import List, Any, Callable, Dict, Optional

try:
    from concurrent.futures import ThreadPoolExecutor, as_completed, ProcessPoolExecutor
    PARALLEL_AVAILABLE = True
except ImportError:
    PARALLEL_AVAILABLE = False

DATA_DIR = os.environ.get('GITHUB_WORKSPACE', '.')
OUTPUT_DIR = Path(DATA_DIR) / 'Datos'

# Detectar si estamos en GitHub Actions (menos workers)
_MAX_WORKERS = int(os.environ.get('PARALLEL_WORKERS', '10'))
if 'GITHUB_ACTIONS' in os.environ:
    _MAX_WORKERS = min(_MAX_WORKERS, 5)


def parallel_map(
    func: Callable,
    items: List[Any],
    max_workers: int = _MAX_WORKERS,
    use_processes: bool = False,
    show_progress: bool = True,
    timeout_per_item: int = 60,
    desc: str = 'Processing'
) -> List[Any]:
    """
    Ejecuta función en paralelo sobre items.
    
    Args:
        func: Función a ejecutar
        items: Lista de argumentos
        max_workers: Número máximo de workers
        use_processes: Usar ProcessPoolExecutor vs ThreadPoolExecutor
        show_progress: Mostrar barra de progreso
        timeout_per_item: Timeout por item en segundos
        desc: Descripción para log

    Returns:
        Lista de resultados (mismo orden que items)
    """
    if not PARALLEL_AVAILABLE:
        return [func(item) for item in items]
    
    Executor = ProcessPoolExecutor if use_processes else ThreadPoolExecutor
    results = [None] * len(items)
    completed = 0
    errors = 0
    
    with Executor(max_workers=max_workers) as executor:
        future_to_idx = {
            executor.submit(func, item): i 
            for i, item in enumerate(items)
        }
        
        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            try:
                results[idx] = future.result(timeout=timeout_per_item)
                completed += 1
                if show_progress and (completed % max(1, len(items) // 10) == 0 or completed == len(items)):
                    print(f'  [{desc}] {completed}/{len(items)} completados ({errors} errores)')
            except Exception as e:
                results[idx] = None
                errors += 1
                if show_progress:
                    print(f'  [!] Item {items[idx] if idx < len(items) else idx}: {str(e)[:80]}')
    
    if errors > 0:
        print(f'  [Parallel] {completed} ok, {errors} errores de {len(items)}')
    else:
        print(f'  [Parallel] {completed}/{len(items)} completados en {desc}')
    
    return results


def parallel_download_yfinance(
    tickers: List[str],
    download_func: Callable,
    max_workers: int = _MAX_WORKERS,
    **kwargs
) -> Dict[str, Any]:
    """
    Descarga datos de yfinance en paralelo.
    
    Args:
        tickers: Lista de tickers
        download_func: Función que descarga un ticker (ej: lambda t: yf.download(t, ...))
        max_workers: Número de workers
    
    Returns:
        Dict {ticker: resultado}
    """
    if not PARALLEL_AVAILABLE:
        return {t: download_func(t) for t in tickers}
    
    results = {}
    errors = []
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_ticker = {
            executor.submit(download_func, ticker): ticker 
            for ticker in tickers
        }
        
        for future in as_completed(future_to_ticker):
            ticker = future_to_ticker[future]
            try:
                results[ticker] = future.result(timeout=120)
            except Exception as e:
                results[ticker] = None
                errors.append((ticker, str(e)))
    
    if errors:
        print(f'[Parallel] Errores: {len(errors)}/{len(tickers)}')
        for t, e in errors[:5]:
            print(f'  {t}: {e[:80]}')
    
    return results


def parallel_process_tickers(
    tickers: List[str],
    process_func: Callable[[str], Any],
    max_workers: int = _MAX_WORKERS,
    delay_between: float = 0.0,
    **kwargs
) -> Dict[str, Any]:
    """
    Procesa tickers en paralelo con delay opcional entre requests.
    
    Args:
        tickers: Lista de tickers
        process_func: Función que recibe ticker y retorna resultado
        max_workers: Workers paralelos
        delay_between: Delay entre requests (para rate limiting)
    """
    if delay_between > 0:
        # Con delay, usar ejecución secuencial con ThreadPool para rate limiting
        def rate_limited_func(t):
            result = process_func(t)
            if delay_between > 0:
                time.sleep(delay_between)
            return result
        return parallel_map(rate_limited_func, tickers, max_workers=min(max_workers, 2), **kwargs)
    
    return parallel_map(process_func, tickers, max_workers=max_workers, **kwargs)


def parallel_gather_technicals(tickers: List[str], max_workers: int = _MAX_WORKERS) -> Dict:
    """
    Descarga indicadores técnicos en paralelo para todos los tickers.
    """
    import yfinance as yf
    import numpy as np
    
    def fetch_tecnicos(ticker):
        try:
            df = yf.download(ticker, period='1y', interval='1d', progress=False, auto_adjust=True)
            if df is None or df.empty or len(df) < 30:
                return ticker, None
            
            close = df['Close'].values.flatten()
            actual = float(close[-1])
            ma50 = float(np.mean(close[-50:])) if len(close) >= 50 else actual
            ma200 = float(np.mean(close[-200:])) if len(close) >= 200 else actual
            
            returns = np.diff(close) / close[:-1]
            volatility = float(np.std(returns[-20:])) if len(returns) >= 20 else 0
            
            max_dd = 0
            if len(close) > 0:
                peak = np.maximum.accumulate(close)
                dd = (close - peak) / peak
                max_dd = float(np.min(dd))
            
            return ticker, {
                'precio': actual,
                'ma50': float(ma50),
                'ma200': float(ma200),
                'volatility_20d': volatility,
                'max_drawdown_1y': max_dd,
                'n_days': len(close)
            }
        except Exception as e:
            return ticker, None
    
    results = parallel_map(fetch_tecnicos, tickers, max_workers=max_workers, desc='Tecnicos')
    return {r[0]: r[1] for r in results if r[1] is not None}


def parallel_gather_features(tickers: List[str], max_workers: int = _MAX_WORKERS) -> Dict:
    """
    Calcula features en paralelo para todos los tickers (auto_feature_engineering).
    """
    from auto_feature_engineering import compute_auto_features
    import yfinance as yf
    
    def fetch_and_compute(ticker):
        try:
            df = yf.download(ticker, period='6mo', interval='1d', progress=False, auto_adjust=True)
            if df is None or df.empty:
                return ticker, None
            if isinstance(df.columns, pd.MultiIndex):
                df = df.xs(ticker, axis=1, level=0)
            close = df['Close'].dropna().values
            vol = df['Volume'].dropna().values if 'Volume' in df else None
            features = compute_auto_features(close, vol, ticker)
            return ticker, features
        except Exception as e:
            return ticker, None
    
    import pandas as pd
    results = parallel_map(fetch_and_compute, tickers, max_workers=max_workers, desc='Features')
    return {r[0]: r[1] for r in results if r[1] is not None}


if __name__ == '__main__':
    test_tickers = ['NVDA', 'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'TSLA']
    print(f'[Parallel] Test con {len(test_tickers)} tickers ({_MAX_WORKERS} workers)')
    
    start = time.time()
    results = parallel_gather_technicals(test_tickers)
    elapsed = time.time() - start
    print(f'[Parallel] {len(results)} tickers en {elapsed:.1f}s')
    for t, r in results.items():
        print(f'  {t}: ${r["precio"]:.2f}, vol={r["volatility_20d"]:.4f}')