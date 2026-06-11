#!/usr/bin/env python3
"""trading_orchestrator.py - Puente entre brokers, ejecucion algoritmica, TCA y paper trading."""
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
import json, os, time
from pathlib import Path

from config.settings import get_setting
DATA_DIR = os.environ.get('GITHUB_WORKSPACE', '.')
OUTPUT_DIR = Path(DATA_DIR) / 'Datos' / 'orchestrator'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

try:
    from broker_interface import GestorBrokers, LadoOrden, TipoOrden, BrokerPaper
    BROKER_AVAILABLE = True
except Exception:
    BROKER_AVAILABLE = False

try:
    from ejecucion_algoritmica import ExecutionManager, VWAPExecutor, TWAPExecutor, ImplementationShortfallExecutor
    EJEC_AVAILABLE = True
except Exception:
    EJEC_AVAILABLE = False

try:
    from transaction_cost_analysis import TCAAnalyzer, SimuladorTCA
    TCA_AVAILABLE = True
except Exception:
    TCA_AVAILABLE = False

try:
    from paper_trading import PaperTradingManager, get_paper_trading_manager
    PAPER_AVAILABLE = True
except Exception:
    PAPER_AVAILABLE = False


class TradingOrchestrator:
    def __init__(self, capital: float = 100000):
        self.capital = capital
        self.gestor = GestorBrokers().crear_brokers_default(capital) if BROKER_AVAILABLE else None
        self.exec_mgr = ExecutionManager() if EJEC_AVAILABLE else None
        self.tca = TCAAnalyzer() if TCA_AVAILABLE else None
        self.paper = get_paper_trading_manager() if PAPER_AVAILABLE else None
        self.ordenes_ejecutadas = []

    def iniciar_paper_trading(self, tickers: Optional[List[str]] = None):
        if not self.paper:
            print('[Orch] paper_trading no disponible')
            return
        self.paper.start()
        if tickers:
            prices = {t: 100.0 + hash(t) % 5000 / 100 for t in tickers}
            self.paper.broker.set_price_provider(lambda t, p=prices: p.get(t, 100.0))
        print(f'[Orch] Paper trading iniciado con ${self.capital:,.0f}')

    def detener_paper_trading(self):
        if self.paper:
            self.paper.stop()

    def rl_position_sizing(self, senales_path: str = None) -> Dict[str, float]:
        """Carga pesos RL desde analisis_ia.json para position sizing."""
        if senales_path is None:
            senales_path = os.path.join(DATA_DIR, 'Datos', 'analisis_ia.json')
        pesos = {}
        try:
            from learning_engine import RLSizingBridge
            rl = RLSizingBridge()
            if os.path.exists(senales_path):
                ia = json.load(open(senales_path))
                probs = ia.get('rl_position_sizing', {})
                if probs:
                    return probs
            # Fallback: compute from probabilidades
            if os.path.exists(senales_path):
                ia = json.load(open(senales_path))
                pbs = ia.get('probabilidades', {})
                senales = {t: d.get('probabilidad', 50) for t, d in pbs.items() if isinstance(d, dict)}
                precios_rl = {t: d.get('precio_objetivo_30d', 100) for t, d in pbs.items() if isinstance(d, dict)}
                pesos = rl.predecir_pesos(senales, precios_rl, self.capital)
        except Exception as e:
            print(f'[RL Sizing] Error: {e}')
        return pesos

    def ejecutar_senal(self, ticker: str, lado: str, shares: int = None, algo: str = 'VWAP', usar_broker: bool = True):
        resultados = {}
        if shares is None:
            pesos_rl = self.rl_position_sizing()
            peso = pesos_rl.get(ticker, 0.05)
            precio = 100.0
            try:
                ia = json.load(open(os.path.join(DATA_DIR, 'Datos', 'analisis_ia.json')))
                pbs = ia.get('probabilidades', {})
                if ticker in pbs and isinstance(pbs[ticker], dict):
                    precio = pbs[ticker].get('precio_objetivo_30d', 100)
            except:
                pass
            shares = max(1, int(self.capital * peso / precio))
            print(f'[RL Sizing] {ticker}: peso={peso:.3f} shares={shares}')

        if usar_broker and self.gestor:
            lado_enum = LadoOrden.COMPRA if lado.upper() == 'COMPRA' else LadoOrden.VENTA
            orden = self.gestor.enviar_orden(ticker, lado_enum, shares)
            resultados['broker'] = {
                'order_id': orden.order_id if orden else None,
                'estado': orden.estado.value if orden else 'fallo',
                'precio': orden.precio_lleno if orden else 0,
                'slippage': orden.slippage_bps if orden else 0
            }

        if self.exec_mgr:
            n = 100
            price_data = pd.DataFrame({
                'close': 100 + np.cumsum(np.random.randn(n) * 0.1),
                'volume': np.random.randint(1000, 100000, n),
            })
            try:
                order = self.exec_mgr.execute(algo, ticker, 'buy' if lado.upper() == 'COMPRA' else 'sell', shares, price_data)
                resultados['ejecucion'] = {
                    'algo': algo,
                    'fill_price': order.fill_price,
                    'slippage_bps': order.slippage_bps
                }
            except Exception as e:
                resultados['ejecucion'] = {'error': str(e)}

        if self.tca:
            trade = self.tca.crear_trade_desde_orden(
                order_id=f"ORD_{ticker}_{int(time.time())}",
                ticker=ticker, lado=lado, shares=shares,
                precio_esperado=100.0, precio_real=resultados.get('broker', {}).get('precio', 100.0),
                timestamp_decision=datetime.now().isoformat(),
                timestamp_envio=datetime.now().isoformat(),
                timestamp_lleno=datetime.now().isoformat()
            )
            tca_result = self.tca.analizar_trade(trade)
            resultados['tca'] = {
                'total_bps': tca_result.costos.total_bps,
                'calidad': tca_result.calidad
            }

        if self.paper and self.paper.active:
            signal = {'direction': 'buy' if lado.upper() == 'COMPRA' else 'sell', 'confidence': 70, 'price': 100.0}
            self.paper.execute_signal(ticker, signal)

        self.ordenes_ejecutadas.append({
            'timestamp': datetime.now().isoformat(),
            'ticker': ticker, 'lado': lado, 'shares': shares,
            'resultados': resultados
        })
        self._guardar_estado()
        return resultados

    def reporte_general(self) -> Dict:
        report = {
            'timestamp': datetime.now().isoformat(),
            'capital': self.capital,
            'n_ordenes': len(self.ordenes_ejecutadas)
        }
        if self.gestor:
            try:
                res = self.gestor.obtener_resumen()
                if res:
                    report['broker'] = {
                        'valor_portafolio': res.valor_portafolio,
                        'cash': res.cash_disponible,
                        'pnl': res.pnl_total,
                        'retorno': res.retorno_total
                    }
            except Exception:
                pass
        if self.tca:
            try:
                report['tca'] = self.tca.resumen_estadistico()
            except Exception:
                pass
        return report

    def _guardar_estado(self):
        state = {
            'ordenes': self.ordenes_ejecutadas[-100:],
            'reporte': self.reporte_general()
        }
        path = OUTPUT_DIR / 'orchestrator_state.json'
        path.write_text(json.dumps(state, indent=2, default=str), encoding='utf-8')

    def generar_reporte_tca(self) -> str:
        if self.tca:
            return self.tca.generar_reporte()
        return ''


_orch = None

def get_orchestrator(capital: float = 100000) -> TradingOrchestrator:
    global _orch
    if _orch is None:
        _orch = TradingOrchestrator(capital)
    return _orch


if __name__ == '__main__':
    print('=== Trading Orchestrator ===')
    orch = get_orchestrator(100000)
    orch.iniciar_paper_trading(['NVDA', 'AAPL', 'MSFT'])

    for ticker in ['NVDA', 'AAPL']:
        r = orch.ejecutar_senal(ticker, 'COMPRA', 100)
        print(f"\n{ticker}:")
        for k, v in r.items():
            print(f"  {k}: {v}")

    report = orch.reporte_general()
    print(f"\n=== Reporte General ===")
    print(f"Ordenes: {report['n_ordenes']}")
    if 'broker' in report:
        print(f"Portafolio: ${report['broker']['valor_portafolio']:.2f}")
    if 'tca' in report:
        print(f"TCA trades: {report['tca'].get('total_trades', 0)}")

    tca_path = orch.generar_reporte_tca()
    if tca_path:
        print(f"Reporte TCA: {tca_path}")

    orch.detener_paper_trading()
