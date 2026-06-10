#!/usr/bin/env python3
"""
feature_store.py - Feature store con formato Parquet particionado por fecha y ticker.
Evita recalcular features, permite time-travel y auditoría de features.
"""
import json
import os
import time
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Union

from config.settings import get_setting

DATA_DIR = os.environ.get('GITHUB_WORKSPACE', '.')
STORE_DIR = Path(DATA_DIR) / 'Datos' / 'feature_store'
STORE_DIR.mkdir(parents=True, exist_ok=True)

FS_CONFIG = get_setting('features.feature_store', {})
ENABLED = FS_CONFIG.get('enabled', True)
FORMAT = FS_CONFIG.get('formato', 'parquet')
PARTITIONED = FS_CONFIG.get('particionado', ['fecha', 'ticker'])

try:
    import pyarrow.parquet as pq
    import pyarrow as pa
    PARQUET_AVAILABLE = True
except ImportError:
    PARQUET_AVAILABLE = False


class FeatureStore:
    def __init__(self, store_dir: Path = STORE_DIR):
        self.store_dir = store_dir
        self.meta_path = store_dir / 'feature_registry.json'
        self._load_registry()
        self._feature_cache = {}

    def _load_registry(self):
        if self.meta_path.exists():
            try:
                self.registry = json.loads(self.meta_path.read_text())
            except:
                self.registry = {'features': {}, 'versions': {}}
        else:
            self.registry = {'features': {}, 'versions': {}}

    def _save_registry(self):
        self.meta_path.write_text(json.dumps(self.registry, indent=2))

    def _partition_path(self, fecha: str, ticker: str) -> Path:
        year, month = fecha[:4], fecha[5:7]
        return self.store_dir / f'year={year}' / f'month={month}' / f'ticker={ticker}'

    def store_features(
        self,
        ticker: str,
        features: Dict[str, float],
        timestamp: Optional[str] = None,
        metadata: Optional[Dict] = None
    ) -> bool:
        if not ENABLED:
            return False
        
        if timestamp is None:
            timestamp = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
        
        fecha = timestamp[:10]
        part_path = self._partition_path(fecha, ticker)
        part_path.mkdir(parents=True, exist_ok=True)

        record = {
            'ticker': ticker,
            'timestamp': timestamp,
            'fecha': fecha,
            'features': features,
            'metadata': metadata or {}
        }

        format = FORMAT
        if format == 'parquet' and PARQUET_AVAILABLE:
            file_path = part_path / f'{timestamp.replace(":", "-")}.parquet'
            try:
                df = pd.DataFrame([{
                    'ticker': ticker,
                    'timestamp': timestamp,
                    'fecha': fecha,
                    **features,
                    'metadata': json.dumps(metadata or {})
                }])
                df.to_parquet(file_path, index=False)
            except Exception as e:
                print(f'[FeatureStore] Error parquet: {e}')
                format = 'json'
        
        if format == 'json':
            file_path = part_path / f'{timestamp.replace(":", "-")}.json'
            with open(file_path, 'w') as f:
                json.dump(record, f)
        
        for fname in features:
            if fname not in self.registry['features']:
                self.registry['features'][fname] = {
                    'first_seen': timestamp,
                    'n_records': 0,
                    'tickers': []
                }
            self.registry['features'][fname]['n_records'] += 1
            if ticker not in self.registry['features'][fname]['tickers']:
                self.registry['features'][fname]['tickers'].append(ticker)
        self._save_registry()
        return True

    def load_features(
        self,
        ticker: Optional[str] = None,
        fecha_inicio: Optional[str] = None,
        fecha_fin: Optional[str] = None,
        feature_names: Optional[List[str]] = None,
        as_dataframe: bool = True
    ) -> Union[pd.DataFrame, List[Dict]]:
        if not self.store_dir.exists():
            return pd.DataFrame() if as_dataframe else []
        
        if not PARQUET_AVAILABLE:
            return self._load_json(ticker, fecha_inicio, fecha_fin, feature_names, as_dataframe)
        
        try:
            import pyarrow.parquet as pq
            
            files = list(self.store_dir.rglob('*.parquet'))
            if not files:
                return self._load_json(ticker, fecha_inicio, fecha_fin, feature_names, as_dataframe)
            
            tables = []
            for f in files:
                try:
                    table = pq.read_table(f)
                    tables.append(table)
                except:
                    continue
            
            if not tables:
                return pd.DataFrame() if as_dataframe else []
            
            import pyarrow as pa
            combined = pa.concat_tables(tables)
            df = combined.to_pandas()
            
            if ticker:
                df = df[df['ticker'] == ticker]
            if fecha_inicio:
                df = df[df['fecha'] >= fecha_inicio]
            if fecha_fin:
                df = df[df['fecha'] <= fecha_fin]
            
            return df if as_dataframe else df.to_dict('records')
        except Exception as e:
            print(f'[FeatureStore] Error reading parquet: {e}')
            return self._load_json(ticker, fecha_inicio, fecha_fin, feature_names, as_dataframe)

    def _load_json(self, ticker, fecha_inicio, fecha_fin, feature_names, as_dataframe):
        records = []
        for path in self.store_dir.rglob('*.json'):
            try:
                with open(path) as f:
                    rec = json.load(f)
                if ticker and rec.get('ticker') != ticker:
                    continue
                if fecha_inicio and rec.get('fecha', '') < fecha_inicio:
                    continue
                if fecha_fin and rec.get('fecha', '') > fecha_fin:
                    continue
                
                if feature_names:
                    rec['features'] = {k: v for k, v in rec['features'].items() if k in feature_names}
                records.append(rec)
            except:
                continue
        
        if as_dataframe:
            rows = []
            for r in records:
                row = {'ticker': r['ticker'], 'timestamp': r['timestamp'], 'fecha': r['fecha']}
                if 'features' in r:
                    row.update(r['features'])
                rows.append(row)
            return pd.DataFrame(rows) if rows else pd.DataFrame()
        return records

    def get_feature_history(
        self,
        ticker: str,
        feature_name: str,
        days: int = 252
    ) -> pd.Series:
        """Retorna serie histórica de una feature específica."""
        df = self.load_features(ticker=ticker, as_dataframe=True)
        if df.empty or feature_name not in df.columns:
            return pd.Series(dtype=float)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values('timestamp')
        cutoff = datetime.now() - timedelta(days=days)
        df = df[df['timestamp'] >= cutoff]
        return df.set_index('timestamp')[feature_name]

    def list_features(self) -> Dict[str, Any]:
        return self.registry.get('features', {})

    def get_feature_stats(self, ticker: str, days: int = 252) -> Dict:
        df = self.load_features(ticker=ticker, as_dataframe=True)
        if df.empty:
            return {}
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        stats = {}
        for col in numeric_cols:
            series = df[col].dropna()
            if len(series) > 0:
                stats[col] = {
                    'mean': float(series.mean()),
                    'std': float(series.std()),
                    'min': float(series.min()),
                    'max': float(series.max()),
                    'last': float(series.iloc[-1]),
                    'n': len(series)
                }
        return stats

    def purge_old(self, days: int = 365):
        """Elimina features más antiguas que days."""
        cutoff = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
        for path in self.store_dir.rglob('*.json'):
            if path.suffix == '.json':
                try:
                    with open(path) as f:
                        rec = json.load(f)
                    if rec.get('fecha', '') < cutoff:
                        path.unlink()
                except:
                    continue
        
        for path in self.store_dir.rglob('*.parquet'):
            try:
                df = pd.read_parquet(path)
                if 'fecha' in df.columns:
                    df_old = df[df['fecha'] < cutoff]
                    if len(df_old) > 0:
                        path.unlink()
            except:
                continue
        
        print(f'[FeatureStore] Purged records before {cutoff}')


_feature_store = None


def get_feature_store() -> FeatureStore:
    global _feature_store
    if _feature_store is None:
        _feature_store = FeatureStore()
    return _feature_store


def store_features_batch(
    ticker: str,
    features_df: pd.DataFrame,
    timestamp_col: str = 'timestamp'
) -> int:
    """Guarda batch de features (un DataFrame)."""
    store = get_feature_store()
    count = 0
    for idx, row in features_df.iterrows():
        ts = str(row.get(timestamp_col, idx))
        feat_dict = row.drop(labels=[timestamp_col], errors='ignore').to_dict()
        feat_dict = {k: float(v) if isinstance(v, (int, float, np.number)) else v 
                     for k, v in feat_dict.items()}
        if store.store_features(ticker, feat_dict, timestamp=ts):
            count += 1
    return count


if __name__ == '__main__':
    print(f'[FeatureStore] Parquet available: {PARQUET_AVAILABLE}')
    fs = get_feature_store()
    fs.store_features('NVDA', {'rsi_14': 65.2, 'macd_hist': 1.5, 'vol_ratio': 1.2})
    fs.store_features('NVDA', {'rsi_14': 62.1, 'macd_hist': 0.8, 'vol_ratio': 0.9})
    fs.store_features('AAPL', {'rsi_14': 45.0, 'macd_hist': -0.5, 'vol_ratio': 1.5})
    print(f'Features registradas: {list(fs.list_features().keys())}')
    print(f'NVDA RSI history: {fs.get_feature_history("NVDA", "rsi_14")}')