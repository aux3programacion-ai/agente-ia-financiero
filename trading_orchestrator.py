#!/usr/bin/env python3
"""trading_orchestrator.py - Puente entre brokers, ejecucion algoritmica, TCA y paper trading."""
import json
import os
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from config.settings import get_setting
from logger import get_module_logger

_log = get_module_logger('trading_orchestrator')
DATA_DIR = os.environ.get('GITHUB_WORKSPACE', '.')
OUTPUT_DIR = Path(DATA_DIR) / 'Datos' / 'orchestrator'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

try:
    from broker_interface import GestorBrokers, LadoOrden
    BROKER_AVAILABLE = True
except Exception:
    BROKER_AVAILABLE = False

try:
    from ejecucion_algoritmica import (
        ExecutionManager,
    )
    EJEC_AVAILABLE = True
except Exception:
    EJEC_AVAILABLE = False

try:
    from transaction_cost_analysis import TCAAnalyzer
    TCA_AVAILABLE = True
except Exception:
    TCA_AVAILABLE = False

try:
    from paper_trading import get_paper_trading_manager
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

    def iniciar_paper_trading(self, tickers: list[str] | None = None):
        if not self.paper:
            _log.warning("paper_trading no disponible")
            return
        self.paper.start()
        if tickers:
            prices = {t: 100.0 + hash(t) % 5000 / 100 for t in tickers}
            self.paper.broker.set_price_provider(lambda t, p=prices: p.get(t, 100.0))
        _log.info(f"Paper trading iniciado con ${self.capital:,.0f}")

    def detener_paper_trading(self):
        if self.paper:
            self.paper.stop()

    def rl_position_sizing(self, senales_path: str = None) -> dict[str, float]:
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
            _log.error(f"RL Sizing: {e}")
        return pesos

    def verificar_riesgo(self, ticker: str, shares: int, precio: float) -> bool:
        """Verifica reglas de riesgo antes de ejecutar."""
        config = get_setting('risk_management')
        if not config:
            return True
        
        report = self.reporte_general()
        total_portafolio = report.get('broker', {}).get('valor_portafolio', self.capital)
        
        # 1. Verificar límite de exposición
        valor_posicion = shares * precio
        if (valor_posicion / max(total_portafolio, 1.0)) > config.get('max_exposure_per_ticker_pct', 0.10):
            _log.error(f"Riesgo rechazado: Exposición excedida en {ticker}")
            return False
            
        # 2. Verificar límite de pérdida diaria
        pnl_diario = report.get('broker', {}).get('pnl', 0)
        if pnl_diario < -config.get('daily_loss_limit_usd', 500):
            if config.get('kill_switch_enabled', True):
                _log.critical("¡KILL SWITCH ACTIVADO! Límite de pérdida diaria alcanzado.")
                self.detener_paper_trading()
                return False
                
        return True

    def _precio_actual(self, ticker: str) -> float | None:
        """
        Precio real del ticker. Devuelve None si no hay ninguna fuente disponible:
        preferimos no operar a inventar un precio.
        """
        precios_path = os.path.join(DATA_DIR, 'Datos', 'precios_reales.json')
        if os.path.exists(precios_path):
            try:
                with open(precios_path) as f:
                    precio = json.load(f).get('precios', {}).get(ticker, {}).get('price')
                if precio:
                    return float(precio)
            except (OSError, ValueError, TypeError):
                pass

        try:
            import yfinance as yf
            hist = yf.Ticker(ticker).history(period='5d')
            if hist is not None and not hist.empty:
                return float(hist['Close'].iloc[-1])
        except Exception as e:
            _log.warning(f"No se pudo obtener precio de {ticker}: {str(e)[:80]}")

        # Solo se usa el proveedor del broker si fue configurado explicitamente:
        # su valor por defecto devuelve 100.0, que no es un precio real.
        broker = getattr(self.paper, 'broker', None) if self.paper else None
        if broker is not None and hasattr(broker, '_price_provider'):
            try:
                precio = float(broker._price_provider(ticker))
                if precio > 0:
                    return precio
            except Exception:
                pass
        return None

    def ejecutar_senal(self, ticker: str, lado: str, shares: int = None, algo: str = 'VWAP', usar_broker: bool = True):
        resultados = {}
        precio = self._precio_actual(ticker)
        if precio is None:
            return {"error": f"Sin precio de mercado para {ticker}; no se ejecuta la orden"}

        if shares is None:
            pesos = self.rl_position_sizing()
            peso = pesos.get(ticker, 0.0)
            if not peso:
                peso = get_setting('risk_management', default={}).get('peso_maximo_por_ticker', 0.10)
            valor = min(self.capital * peso, self.capital)
            shares = max(1, int(valor / precio))
            _log.info(f"RL Sizing {ticker}: peso={peso:.3f} precio={precio:.2f} shares={shares}")

        # --- APLICAR REGLAS DE RIESGO ---
        if not self.verificar_riesgo(ticker, shares, precio):
            return {"error": "Riesgo rechazado por política de seguridad"}

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

    def reporte_general(self) -> dict:
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
            except Exception as e:
                _log.warning(f"Broker summary failed: {e}")
        if self.tca:
            try:
                report['tca'] = self.tca.resumen_estadistico()
            except Exception as e:
                _log.warning(f"TCA summary failed: {e}")
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
    _log.info("=== Trading Orchestrator ===")
    orch = get_orchestrator(100000)
    orch.iniciar_paper_trading(['NVDA', 'AAPL', 'MSFT'])

    for ticker in ['NVDA', 'AAPL']:
        r = orch.ejecutar_senal(ticker, 'COMPRA', 100)
        _log.info(f"{ticker}: {r}")

    report = orch.reporte_general()
    _log.info(f"Ordenes: {report['n_ordenes']}")
    if 'broker' in report:
        _log.info(f"Portafolio: ${report['broker']['valor_portafolio']:.2f}")
    if 'tca' in report:
        _log.info(f"TCA trades: {report['tca'].get('total_trades', 0)}")

    tca_path = orch.generar_reporte_tca()
    if tca_path:
        _log.info(f"Reporte TCA: {tca_path}")

    orch.detener_paper_trading()
