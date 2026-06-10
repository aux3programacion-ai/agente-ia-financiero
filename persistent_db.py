#!/usr/bin/env python3
"""
persistent_db.py - Base de datos SQLite centralizada.
CRUD para modelos, caracteristicas, decisiones, ordenes, riesgos, logs.
"""
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
import json, os, sqlite3, threading
from pathlib import Path

from config.settings import get_setting

DATA_DIR = os.environ.get('GITHUB_WORKSPACE', '.')
DB_PATH = Path(DATA_DIR) / 'Datos' / 'agente.db'
DB_PATH.parent.mkdir(parents=True, exist_ok=True)

_local = threading.local()


def get_conn() -> sqlite3.Connection:
    if not hasattr(_local, 'conn') or _local.conn is None:
        _local.conn = sqlite3.connect(str(DB_PATH))
        _local.conn.row_factory = sqlite3.Row
        _local.conn.execute('PRAGMA journal_mode=WAL')
        _local.conn.execute('PRAGMA foreign_keys=ON')
    return _local.conn


class DatabaseManager:
    def __init__(self, db_path: Optional[Path] = None):
        self.db_path = db_path or DB_PATH
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._init_schema()

    def _get_conn(self) -> sqlite3.Connection:
        return sqlite3.connect(str(self.db_path))

    def _init_schema(self):
        conn = self._get_conn()
        try:
            conn.executescript('''
                CREATE TABLE IF NOT EXISTS modelos (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    model_id TEXT NOT NULL, version INTEGER NOT NULL,
                    nombre TEXT, regime TEXT, path TEXT,
                    metrics TEXT, features TEXT, created TEXT,
                    status TEXT DEFAULT 'staging',
                    UNIQUE(model_id, version)
                );
                CREATE TABLE IF NOT EXISTS decisiones (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT, ticker TEXT, senial TEXT,
                    confianza REAL, model_version INTEGER,
                    features_used TEXT, outcome REAL,
                    correcto INTEGER, created TEXT DEFAULT (datetime('now'))
                );
                CREATE TABLE IF NOT EXISTS ordenes (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    order_id TEXT UNIQUE, ticker TEXT, lado TEXT,
                    shares INTEGER, filled_shares INTEGER DEFAULT 0,
                    precio REAL, status TEXT, broker TEXT,
                    slippage_bps REAL, created TEXT,
                    updated TEXT DEFAULT (datetime('now'))
                );
                CREATE TABLE IF NOT EXISTS portafolio (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    fecha TEXT, ticker TEXT, cantidad INTEGER,
                    precio_compra REAL, valor_actual REAL,
                    pnl_realizado REAL DEFAULT 0,
                    pnl_no_realizado REAL DEFAULT 0
                );
                CREATE TABLE IF NOT EXISTS riesgos (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT, ticker TEXT, tipo TEXT,
                    valor REAL, limite REAL, excedido INTEGER,
                    detalles TEXT
                );
                CREATE TABLE IF NOT EXISTS caracteristicas (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT, ticker TEXT,
                    nombre TEXT, valor REAL
                );
                CREATE TABLE IF NOT EXISTS metricas (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT, categoria TEXT,
                    nombre TEXT, valor REAL,
                    etiquetas TEXT
                );
                CREATE TABLE IF NOT EXISTS snapshots_datos (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    dataset_id TEXT, version INTEGER,
                    path TEXT, filas INTEGER, columnas TEXT,
                    hash TEXT, created TEXT,
                    UNIQUE(dataset_id, version)
                );
                CREATE INDEX IF NOT EXISTS idx_decisiones_ticker
                    ON decisiones(ticker);
                CREATE INDEX IF NOT EXISTS idx_ordenes_ticker
                    ON ordenes(ticker);
                CREATE INDEX IF NOT EXISTS idx_riesgos_ticker
                    ON riesgos(ticker);
                CREATE INDEX IF NOT EXISTS idx_caracteristicas_ticker
                    ON caracteristicas(ticker, timestamp);
            ''')
            conn.commit()
        finally:
            conn.close()

    def ejecutar(self, sql: str, params: tuple = ()) -> sqlite3.Cursor:
        conn = self._get_conn()
        try:
            cur = conn.execute(sql, params)
            conn.commit()
            return cur
        finally:
            conn.close()

    def consultar(self, sql: str, params: tuple = ()) -> List[Dict]:
        conn = self._get_conn()
        try:
            conn.row_factory = sqlite3.Row
            cur = conn.execute(sql, params)
            return [dict(r) for r in cur.fetchall()]
        finally:
            conn.close()

    def consultar_df(self, sql: str, params: tuple = ()) -> pd.DataFrame:
        conn = self._get_conn()
        try:
            return pd.read_sql_query(sql, conn, params=params)
        finally:
            conn.close()

    def guardar_modelo(self, model_id: str, nombre: str, regime: str,
                       path: str, metrics: Dict, features: List[str]) -> int:
        version = len(self.consultar(
            'SELECT id FROM modelos WHERE model_id=?', (model_id,))) + 1
        self.ejecutar(
            'INSERT INTO modelos (model_id,version,nombre,regime,path,metrics,features,created) '
            'VALUES (?,?,?,?,?,?,?,?)',
            (model_id, version, nombre, regime, path,
             json.dumps(metrics), json.dumps(features),
             datetime.now().isoformat()))
        return version

    def promover_modelo(self, model_id: str, version: int):
        self.ejecutar(
            'UPDATE modelos SET status=? WHERE model_id=? AND version=?',
            ('rolled_back', model_id, version - 1))
        self.ejecutar(
            'UPDATE modelos SET status=? WHERE model_id=? AND version=?',
            ('production', model_id, version))

    def obtener_modelo(self, model_id: str, version: Optional[int] = None) -> Optional[Dict]:
        if version:
            rows = self.consultar(
                'SELECT * FROM modelos WHERE model_id=? AND version=? ORDER BY version DESC',
                (model_id, version))
        else:
            rows = self.consultar(
                'SELECT * FROM modelos WHERE model_id=? ORDER BY version DESC LIMIT 1',
                (model_id,))
        if rows:
            r = rows[0]
            r['metrics'] = json.loads(r['metrics']) if isinstance(r['metrics'], str) else r['metrics']
            r['features'] = json.loads(r['features']) if isinstance(r['features'], str) else r['features']
            return r
        return None

    def listar_modelos(self) -> List[Dict]:
        return self.consultar(
            'SELECT model_id, MAX(version) as version, nombre, status, created '
            'FROM modelos GROUP BY model_id ORDER BY created DESC')

    def guardar_decision(self, ticker: str, senial: str, confianza: float,
                         model_version: int, features: Dict):
        self.ejecutar(
            'INSERT INTO decisiones (timestamp,ticker,senial,confianza,model_version,features_used) '
            'VALUES (?,?,?,?,?,?)',
            (datetime.now().isoformat(), ticker, senial, confianza,
             model_version, json.dumps(features)))

    def actualizar_resultado(self, decision_id: int, outcome: float, correcto: bool):
        self.ejecutar(
            'UPDATE decisiones SET outcome=?, correcto=? WHERE id=?',
            (outcome, int(correcto), decision_id))

    def obtener_decisiones(self, ticker: Optional[str] = None,
                           limite: int = 100) -> List[Dict]:
        if ticker:
            return self.consultar(
                'SELECT * FROM decisiones WHERE ticker=? ORDER BY id DESC LIMIT ?',
                (ticker, limite))
        return self.consultar(
            'SELECT * FROM decisiones ORDER BY id DESC LIMIT ?', (limite,))

    def guardar_orden(self, order_id: str, ticker: str, lado: str,
                      shares: int, precio: float, broker: str = 'paper',
                      slippage_bps: float = 0.0):
        self.ejecutar(
            'INSERT INTO ordenes (order_id,ticker,lado,shares,precio,broker,slippage_bps,created,status) '
            'VALUES (?,?,?,?,?,?,?,?,?)',
            (order_id, ticker, lado, shares, precio, broker,
             slippage_bps, datetime.now().isoformat(), 'filled'))

    def obtener_ordenes(self, ticker: Optional[str] = None,
                        limite: int = 100) -> List[Dict]:
        if ticker:
            return self.consultar(
                'SELECT * FROM ordenes WHERE ticker=? ORDER BY id DESC LIMIT ?',
                (ticker, limite))
        return self.consultar(
            'SELECT * FROM ordenes ORDER BY id DESC LIMIT ?', (limite,))

    def guardar_riesgo(self, ticker: str, tipo: str, valor: float,
                       limite: float, excedido: bool, detalles: str = ''):
        self.ejecutar(
            'INSERT INTO riesgos (timestamp,ticker,tipo,valor,limite,excedido,detalles) '
            'VALUES (?,?,?,?,?,?,?)',
            (datetime.now().isoformat(), ticker, tipo, valor, limite,
             int(excedido), detalles))

    def obtener_riesgos(self, ticker: Optional[str] = None,
                        limite: int = 100) -> List[Dict]:
        if ticker:
            return self.consultar(
                'SELECT * FROM riesgos WHERE ticker=? ORDER BY id DESC LIMIT ?',
                (ticker, limite))
        return self.consultar(
            'SELECT * FROM riesgos ORDER BY id DESC LIMIT ?', (limite,))

    def guardar_caracteristica(self, ticker: str, nombre: str, valor: float):
        self.ejecutar(
            'INSERT INTO caracteristicas (timestamp,ticker,nombre,valor) VALUES (?,?,?,?)',
            (datetime.now().isoformat(), ticker, nombre, valor))

    def obtener_historial_caracteristicas(self, ticker: str, nombre: str,
                                          dias: int = 30) -> pd.DataFrame:
        desde = (datetime.now() - timedelta(days=dias)).isoformat()
        return self.consultar_df(
            'SELECT timestamp, valor FROM caracteristicas '
            'WHERE ticker=? AND nombre=? AND timestamp>=? ORDER BY timestamp',
            (ticker, nombre, desde))

    def guardar_metrica(self, categoria: str, nombre: str, valor: float,
                        etiquetas: Optional[Dict] = None):
        self.ejecutar(
            'INSERT INTO metricas (timestamp,categoria,nombre,valor,etiquetas) VALUES (?,?,?,?,?)',
            (datetime.now().isoformat(), categoria, nombre, valor,
             json.dumps(etiquetas or {})))

    def obtener_dashboard(self) -> Dict:
        n_modelos = self.consultar(
            'SELECT COUNT(DISTINCT model_id) as n FROM modelos')[0]['n']
        en_prod = self.consultar(
            "SELECT COUNT(*) as n FROM modelos WHERE status='production'")[0]['n']
        n_decisiones = self.consultar(
            'SELECT COUNT(*) as n FROM decisiones')[0]['n']
        correctas = self.consultar(
            'SELECT COUNT(*) as n FROM decisiones WHERE correcto=1')[0]['n']
        n_ordenes = self.consultar(
            'SELECT COUNT(*) as n FROM ordenes')[0]['n']
        n_riesgos = self.consultar(
            'SELECT COUNT(*) as n FROM riesgos')[0]['n']
        return {
            'modelos_unicos': n_modelos,
            'modelos_produccion': en_prod,
            'decisiones': n_decisiones,
            'precision': float(correctas / max(n_decisiones, 1)),
            'ordenes': n_ordenes,
            'eventos_riesgo': n_riesgos,
            'db_path': str(self.db_path),
            'tamano_mb': round(self.db_path.stat().st_size / 1e6, 2) if self.db_path.exists() else 0,
        }

    def snapshot_tabla(self, tabla: str, dataset_id: str) -> str:
        df = self.consultar_df(f'SELECT * FROM {tabla} LIMIT 50000')
        path = self.db_path.parent / f'{dataset_id}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
        df.to_csv(path, index=False)
        self.ejecutar(
            'INSERT INTO snapshots_datos (dataset_id,version,path,filas,columnas,hash,created) '
            'VALUES (?,?,?,?,?,?,?)',
            (dataset_id, 1, str(path), len(df), json.dumps(list(df.columns)),
             str(hash(str(df.values.tobytes()))), datetime.now().isoformat()))
        return str(path)

    def close(self):
        pass


db = DatabaseManager()

if __name__ == '__main__':
    db.guardar_modelo('xgboost_default', 'XGBoost Base', 'ALCISTA',
                      '/tmp/model.pkl', {'acc': 0.82}, ['rsi', 'macd'])
    db.guardar_decision('NVDA', 'COMPRA', 0.78, 1, {'rsi': 65})
    db.guardar_orden('ORD_001', 'NVDA', 'COMPRA', 100, 150.0)
    db.guardar_riesgo('NVDA', 'apalancamiento', 0.8, 1.0, False)
    db.guardar_caracteristica('NVDA', 'rsi_14', 65.0)
    print(json.dumps(db.obtener_dashboard(), indent=2))
