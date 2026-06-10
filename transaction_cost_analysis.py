#!/usr/bin/env python3
"""
transaction_cost_analysis.py - Analisis de costos de transaccion (TCA).
Slippage real vs esperado, market impact, timing cost,
oportunity cost, y reportes post-trade.
"""
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
import json, os
from pathlib import Path

from config.settings import get_setting

DATA_DIR = os.environ.get('GITHUB_WORKSPACE', '.')
OUTPUT_DIR = Path(DATA_DIR) / 'Datos' / 'tca'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

try:
    from persistent_db import db
except Exception:
    db = None


@dataclass
class TradeRecord:
    trade_id: str; ticker: str; lado: str; shares: int
    precio_esperado: float; precio_real: float
    timestamp_decision: str; timestamp_envio: str; timestamp_lleno: str
    volumen_diario: float = 0.0; volatilidad: float = 0.0
    spread_bps: float = 0.0; participacion: float = 0.0


@dataclass
class CostoDesglose:
    slippage_bps: float; market_impact_bps: float
    timing_cost_bps: float; oportunity_cost_bps: float
    spread_cost_bps: float; comision_bps: float
    total_bps: float; total_dinero: float


@dataclass
class TCAResult:
    trade: TradeRecord; costos: CostoDesglose
    senal_ejecucion: str; calidad: str


class CalculatorTCA:
    def calcular_slippage(self, trade: TradeRecord) -> float:
        if trade.precio_esperado <= 0:
            return 0.0
        if trade.lado == 'COMPRA':
            return (trade.precio_real - trade.precio_esperado) / trade.precio_esperado * 10000
        return (trade.precio_esperado - trade.precio_real) / trade.precio_esperado * 10000

    def calcular_market_impact(self, trade: TradeRecord) -> float:
        if trade.volumen_diario <= 0 or trade.volatilidad <= 0:
            return 0.0
        impact = (trade.participacion ** 0.5) * trade.volatilidad * 10000 * 0.5
        return float(impact)

    def calcular_timing_cost(self, trade: TradeRecord) -> float:
        try:
            dt_dec = datetime.fromisoformat(trade.timestamp_decision)
            dt_env = datetime.fromisoformat(trade.timestamp_envio)
            delay_min = max((dt_env - dt_dec).total_seconds() / 60, 0)
            if delay_min <= 1:
                return 0.0
            return float(min(delay_min * 0.1, 5.0))
        except Exception:
            return 0.0

    def calcular_oportunity_cost(self, trade: TradeRecord) -> float:
        try:
            dt_env = datetime.fromisoformat(trade.timestamp_envio)
            dt_lleno = datetime.fromisoformat(trade.timestamp_lleno)
            delay_min = max((dt_lleno - dt_env).total_seconds() / 60, 0)
            if delay_min <= 0.5:
                return 0.0
            return float(min(delay_min * 0.05, 2.0))
        except Exception:
            return 0.0

    def calcular_spread_cost(self, trade: TradeRecord) -> float:
        return trade.spread_bps / 2.0

    def calcular_todo(self, trade: TradeRecord) -> CostoDesglose:
        slippage = self.calcular_slippage(trade)
        impact = self.calcular_market_impact(trade)
        timing = self.calcular_timing_cost(trade)
        oportunity = self.calcular_oportunity_cost(trade)
        spread = self.calcular_spread_cost(trade)
        comision = 1.0
        total = slippage + impact + timing + oportunity + spread + comision
        total_dinero = total / 10000 * trade.precio_real * trade.shares
        return CostoDesglose(
            slippage_bps=round(slippage, 2),
            market_impact_bps=round(impact, 2),
            timing_cost_bps=round(timing, 2),
            oportunity_cost_bps=round(oportunity, 2),
            spread_cost_bps=round(spread, 2),
            comision_bps=round(comision, 2),
            total_bps=round(total, 2),
            total_dinero=round(total_dinero, 2))


class TCAAnalyzer:
    def __init__(self):
        self.calculator = CalculatorTCA()
        self.trades: List[TradeRecord] = []
        self.resultados: List[TCAResult] = []

    def analizar_trade(self, trade: TradeRecord) -> TCAResult:
        costos = self.calculator.calcular_todo(trade)
        if costos.total_bps <= 15:
            calidad = 'excelente'
            senal = '👍'
        elif costos.total_bps <= 30:
            calidad = 'buena'
            senal = '✓'
        elif costos.total_bps <= 60:
            calidad = 'regular'
            senal = '⚠'
        else:
            calidad = 'mala'
            senal = '🔴'
        self.trades.append(trade)
        result = TCAResult(trade=trade, costos=costos,
                           senal_ejecucion=senal, calidad=calidad)
        self.resultados.append(result)
        if db:
            try:
                db.guardar_metrica('tca', 'slippage_bps', costos.slippage_bps,
                                   {'ticker': trade.ticker, 'trade_id': trade.trade_id})
                db.guardar_metrica('tca', 'total_cost_bps', costos.total_bps,
                                   {'ticker': trade.ticker, 'trade_id': trade.trade_id})
                db.guardar_orden(trade.trade_id, trade.ticker, trade.lado,
                                 trade.shares, trade.precio_real, 'tca',
                                 costos.total_bps)
            except Exception:
                pass
        return result

    def analizar_lote(self, trades: List[TradeRecord]) -> List[TCAResult]:
        return [self.analizar_trade(t) for t in trades]

    def crear_trade_desde_orden(self, order_id: str, ticker: str, lado: str,
                                 shares: int, precio_esperado: float,
                                 precio_real: float, timestamp_decision: str,
                                 timestamp_envio: str, timestamp_lleno: str,
                                 volumen_diario: float = 0,
                                 volatilidad: float = 0.02,
                                 spread_bps: float = 10,
                                 participacion: float = 0.01) -> TradeRecord:
        return TradeRecord(
            trade_id=order_id, ticker=ticker, lado=lado, shares=shares,
            precio_esperado=precio_esperado, precio_real=precio_real,
            timestamp_decision=timestamp_decision,
            timestamp_envio=timestamp_envio,
            timestamp_lleno=timestamp_lleno,
            volumen_diario=volumen_diario, volatilidad=volatilidad,
            spread_bps=spread_bps, participacion=participacion)

    def resumen_estadistico(self) -> Dict:
        if not self.resultados:
            return {}
        df = self._a_dataframe()
        return {
            'total_trades': len(df),
            'slippage_promedio': float(df['slippage_bps'].mean()),
            'slippage_mediano': float(df['slippage_bps'].median()),
            'slippage_std': float(df['slippage_bps'].std()),
            'costo_total_promedio': float(df['total_bps'].mean()),
            'mejor_calidad': float(df['total_bps'].min()),
            'peor_calidad': float(df['total_bps'].max()),
            'trades_excelentes': int((df['total_bps'] <= 15).sum()),
            'trades_buenos': int(((df['total_bps'] > 15) & (df['total_bps'] <= 30)).sum()),
            'trades_regulares': int(((df['total_bps'] > 30) & (df['total_bps'] <= 60)).sum()),
            'trades_malos': int((df['total_bps'] > 60).sum()),
            'desglose_promedio': {
                'slippage': float(df['slippage_bps'].mean()),
                'market_impact': float(df['market_impact_bps'].mean()),
                'timing': float(df['timing_cost_bps'].mean()),
                'oportunity': float(df['oportunity_cost_bps'].mean()),
                'spread': float(df['spread_cost_bps'].mean()),
                'comision': float(df['comision_bps'].mean()),
            },
            'total_costos_dinero': float(df['total_dinero'].sum()),
        }

    def resumen_por_ticker(self) -> pd.DataFrame:
        if not self.resultados:
            return pd.DataFrame()
        df = self._a_dataframe()
        return df.groupby('ticker').agg(
            trades=('total_bps', 'count'),
            slippage_prom=('slippage_bps', 'mean'),
            costo_prom=('total_bps', 'mean'),
            costo_total=('total_dinero', 'sum'),
        ).round(2).reset_index()

    def _a_dataframe(self) -> pd.DataFrame:
        rows = []
        for r in self.resultados:
            row = asdict(r.costos)
            row['ticker'] = r.trade.ticker
            row['lado'] = r.trade.lado
            row['shares'] = r.trade.shares
            row['calidad'] = r.calidad
            rows.append(row)
        return pd.DataFrame(rows)

    def generar_reporte(self) -> str:
        res = self.resumen_estadistico()
        if not res:
            return 'Sin datos TCA'
        lines = ['=== Reporte TCA ===',
                 f'Fecha: {datetime.now().isoformat()}', '']
        lines.append(f'Trades analizados: {res["total_trades"]}')
        lines.append(f'Slippage promedio: {res["slippage_promedio"]:.2f} bps')
        lines.append(f'Costo total promedio: {res["costo_total_promedio"]:.2f} bps')
        lines.append(f'Costos totales: ${res["total_costos_dinero"]:.2f}')
        lines.append('')
        lines.append('Desglose promedio (bps):')
        for k, v in res['desglose_promedio'].items():
            lines.append(f'  {k}: {v:.2f}')
        lines.append('')
        lines.append(f'Excelentes (<15): {res["trades_excelentes"]}')
        lines.append(f'Buenos (15-30): {res["trades_buenos"]}')
        lines.append(f'Regulares (30-60): {res["trades_regulares"]}')
        lines.append(f'Malos (>60): {res["trades_malos"]}')
        path = OUTPUT_DIR / f'tca_report_{datetime.now().strftime("%Y%m%d")}.txt'
        Path(path).write_text('\n'.join(lines), encoding='utf-8')
        return str(path)


class SimuladorTCA:
    def __init__(self, seed: int = 42):
        self.rng = np.random.default_rng(seed)

    def simular_trade(self, ticker: str, lado: str, shares: int,
                      precio_base: float = 100.0) -> TradeRecord:
        ahora = datetime.now()
        delay_decision = self.rng.exponential(2)
        delay_ejecucion = self.rng.exponential(1)
        precio_esperado = precio_base * (1 + self.rng.normal(0, 0.001))
        slippage = self.rng.exponential(15)
        direccion = 1 if lado == 'COMPRA' else -1
        precio_real = precio_esperado + direccion * slippage / 10000 * precio_esperado
        timestamp_decision = (ahora - timedelta(minutes=delay_decision)).isoformat()
        timestamp_envio = ahora.isoformat()
        timestamp_lleno = (ahora + timedelta(seconds=delay_ejecucion)).isoformat()
        return TradeRecord(
            trade_id=f'TCA_{uuid4().hex[:8]}' if __import__('uuid', fromlist=['uuid4']).uuid4 else f'TCA_{self.rng.integers(100000)}',
            ticker=ticker, lado=lado, shares=shares,
            precio_esperado=round(precio_esperado, 2),
            precio_real=round(precio_real, 2),
            timestamp_decision=timestamp_decision,
            timestamp_envio=timestamp_envio,
            timestamp_lleno=timestamp_lleno,
            volumen_diario=float(self.rng.exponential(5e6)),
            volatilidad=float(self.rng.exponential(0.02)),
            spread_bps=float(self.rng.exponential(10)),
            participacion=float(self.rng.uniform(0.001, 0.05)))

    def simular_lote(self, tickers: List[str], n_por_ticker: int = 10) -> List[TradeRecord]:
        from uuid import uuid4
        trades = []
        for ticker in tickers:
            for _ in range(n_por_ticker):
                lado = self.rng.choice(['COMPRA', 'VENTA'])
                shares = int(self.rng.integers(100, 5000))
                trade = self.simular_trade(ticker, lado, shares)
                trade.trade_id = f'TCA_{uuid4().hex[:8]}'
                trades.append(trade)
        return trades


try:
    from uuid import uuid4
except ImportError:
    uuid4 = None

if __name__ == '__main__':
    sim = SimuladorTCA()
    trades = sim.simular_lote(['NVDA', 'AAPL', 'MSFT'], 5)
    analyzer = TCAAnalyzer()
    results = analyzer.analizar_lote(trades)
    res = analyzer.resumen_estadistico()
    print(f'Trades: {res["total_trades"]}')
    print(f'Slippage prom: {res["slippage_promedio"]:.2f} bps')
    print(f'Costo total: {res["costo_total_promedio"]:.2f} bps')
    print(analyzer.resumen_por_ticker().to_string())
    path = analyzer.generar_reporte()
    print(f'Reporte: {path}')
