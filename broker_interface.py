#!/usr/bin/env python3
"""
broker_interface.py - Interfaz unificada de brokers.
Clase abstracta base + implementaciones para PaperBroker,
IBKR, Alpaca, Binance. Intercambiables sin cambiar el codigo de trading.
"""
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple, Callable
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from enum import Enum
import json, os, time, uuid, threading
from pathlib import Path
from abc import ABC, abstractmethod

try:
    import alpaca_trade_api as alpaca_api
    ALPACA_AVAILABLE = True
except ImportError:
    ALPACA_AVAILABLE = False

try:
    from ib_insync import IB, Stock, MarketOrder, LimitOrder
    IB_AVAILABLE = True
except ImportError:
    IB_AVAILABLE = False

from config.settings import get_setting

DATA_DIR = os.environ.get('GITHUB_WORKSPACE', '.')
OUTPUT_DIR = Path(DATA_DIR) / 'Datos' / 'broker'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

try:
    from persistent_db import db
except Exception:
    db = None


class LadoOrden(Enum):
    COMPRA = 'COMPRA'
    VENTA = 'VENTA'


class TipoOrden(Enum):
    MERCADO = 'MERCADO'
    LIMITE = 'LIMITE'
    STOP = 'STOP'
    STOP_LIMITE = 'STOP_LIMITE'


class EstadoOrden(Enum):
    PENDIENTE = 'pendiente'
    ENVIADA = 'enviada'
    LLENADA = 'llena'
    PARCIAL = 'parcial'
    CANCELADA = 'cancelada'
    RECHAZADA = 'rechazada'
    EXPIRADA = 'expirada'


@dataclass
class Orden:
    order_id: str; ticker: str; lado: LadoOrden; tipo: TipoOrden
    shares_solicitadas: int; shares_llenas: int = 0
    precio_limite: Optional[float] = None
    precio_lleno: Optional[float] = None
    estado: EstadoOrden = EstadoOrden.PENDIENTE
    broker: str = 'paper'
    slippage_bps: float = 0.0
    comision: float = 0.0
    created: str = ''
    updated: str = ''
    error: Optional[str] = None


@dataclass
class Posicion:
    ticker: str; cantidad: int; precio_promedio: float
    valor_actual: float; pnl_no_realizado: float = 0.0
    pnl_realizado: float = 0.0


@dataclass
class ResumenCuenta:
    capital: float; valor_portafolio: float; cash_disponible: float
    posiciones: List[Posicion]; pnl_total: float
    retorno_total: float; broker: str


class BrokerBase(ABC):
    def __init__(self, nombre: str = 'base'):
        self.nombre = nombre
        self.ordenes: Dict[str, Orden] = {}
        self.posiciones: Dict[str, Posicion] = {}
        self.capital: float = 0.0
        self._lock = threading.RLock()
        self._conectado = False

    @abstractmethod
    def conectar(self) -> bool:
        pass

    @abstractmethod
    def desconectar(self):
        pass

    @property
    def conectado(self) -> bool:
        return self._conectado

    @abstractmethod
    def enviar_orden(self, ticker: str, lado: LadoOrden, shares: int,
                     tipo: TipoOrden = TipoOrden.MERCADO,
                     precio_limite: Optional[float] = None) -> Orden:
        pass

    @abstractmethod
    def cancelar_orden(self, order_id: str) -> bool:
        pass

    @abstractmethod
    def obtener_posiciones(self) -> List[Posicion]:
        pass

    @abstractmethod
    def obtener_resumen(self) -> ResumenCuenta:
        pass

    @abstractmethod
    def obtener_precio(self, ticker: str) -> float:
        pass

    def _registrar_orden(self, orden: Orden):
        with self._lock:
            self.ordenes[orden.order_id] = orden
        if db:
            try:
                db.guardar_orden(orden.order_id, orden.ticker,
                                 orden.lado.value, orden.shares_solicitadas,
                                 orden.precio_lleno or 0, self.nombre,
                                 orden.slippage_bps)
            except Exception:
                pass

    def historial_ordenes(self, ticker: Optional[str] = None,
                          limite: int = 50) -> List[Orden]:
        with self._lock:
            ords = list(self.ordenes.values())
        if ticker:
            ords = [o for o in ords if o.ticker == ticker]
        return sorted(ords, key=lambda x: x.created, reverse=True)[:limite]


class BrokerPaper(BrokerBase):
    def __init__(self, capital_inicial: float = 100000,
                 comision: float = 0.0,
                 proveedor_precio: Optional[Callable] = None):
        super().__init__('paper')
        self.capital = capital_inicial
        self._cash = capital_inicial
        self._comision = comision
        self._proveedor_precio = proveedor_precio or (
            lambda t: 100 + np.random.random() * 50)

    def conectar(self) -> bool:
        self._conectado = True
        return True

    def desconectar(self):
        self._conectado = False

    def obtener_precio(self, ticker: str) -> float:
        return self._proveedor_precio(ticker)

    def enviar_orden(self, ticker: str, lado: LadoOrden, shares: int,
                     tipo: TipoOrden = TipoOrden.MERCADO,
                     precio_limite: Optional[float] = None) -> Orden:
        order_id = f'PAPER_{uuid.uuid4().hex[:12]}'
        precio = precio_limite or self.obtener_precio(ticker)
        costo = shares * precio + self._comision
        with self._lock:
            if lado == LadoOrden.COMPRA and costo > self._cash:
                orden = Orden(order_id=order_id, ticker=ticker, lado=lado,
                              tipo=tipo, shares_solicitadas=shares,
                              estado=EstadoOrden.RECHAZADA,
                              broker=self.nombre, error='Fondos insuficientes',
                              created=datetime.now().isoformat())
                self._registrar_orden(orden)
                return orden
            orden = Orden(order_id=order_id, ticker=ticker, lado=lado,
                          tipo=tipo, shares_solicitadas=shares,
                          shares_llenas=shares, precio_lleno=precio,
                          estado=EstadoOrden.LLENADA,
                          broker=self.nombre, comision=self._comision,
                          created=datetime.now().isoformat())
            if lado == LadoOrden.COMPRA:
                self._cash -= costo
                if ticker in self.posiciones:
                    p = self.posiciones[ticker]
                    total_shares = p.cantidad + shares
                    total_costo = p.cantidad * p.precio_promedio + shares * precio
                    p.cantidad = total_shares
                    p.precio_promedio = total_costo / total_shares
                else:
                    self.posiciones[ticker] = Posicion(
                        ticker=ticker, cantidad=shares,
                        precio_promedio=precio, valor_actual=precio)
            else:
                self._cash += shares * precio - self._comision
                if ticker in self.posiciones:
                    p = self.posiciones[ticker]
                    pnl = shares * (precio - p.precio_promedio)
                    p.cantidad -= shares
                    p.pnl_realizado += pnl
                    if p.cantidad <= 0:
                        del self.posiciones[ticker]
            orden.updated = datetime.now().isoformat()
            self._registrar_orden(orden)
        return orden

    def cancelar_orden(self, order_id: str) -> bool:
        with self._lock:
            if order_id in self.ordenes:
                self.ordenes[order_id].estado = EstadoOrden.CANCELADA
                return True
        return False

    def obtener_posiciones(self) -> List[Posicion]:
        with self._lock:
            for p in self.posiciones.values():
                precio = self.obtener_precio(p.ticker)
                p.valor_actual = precio
                p.pnl_no_realizado = p.cantidad * (precio - p.precio_promedio)
            return list(self.posiciones.values())

    def obtener_resumen(self) -> ResumenCuenta:
        pos = self.obtener_posiciones()
        valor_pos = sum(p.cantidad * p.valor_actual for p in pos)
        portafolio = self._cash + valor_pos
        pnl_total = portafolio - self.capital
        return ResumenCuenta(
            capital=self.capital, valor_portafolio=portafolio,
            cash_disponible=self._cash, posiciones=pos,
            pnl_total=pnl_total,
            retorno_total=pnl_total / self.capital if self.capital else 0,
            broker=self.nombre)


class BrokerAlpaca(BrokerBase):
    def __init__(self, api_key: str = '', secret_key: str = '',
                 paper_url: str = 'https://paper-api.alpaca.markets'):
        super().__init__('alpaca')
        self.api_key = api_key or os.environ.get('ALPACA_API_KEY', '')
        self.secret_key = secret_key or os.environ.get('ALPACA_SECRET_KEY', '')
        self.paper_url = paper_url or os.environ.get('ALPACA_PAPER_URL',
                                                      'https://paper-api.alpaca.markets')
        self._api = None

    def conectar(self) -> bool:
        if not ALPACA_AVAILABLE:
            return False
        try:
            self._api = alpaca_api.REST(self.api_key, self.secret_key,
                                        self.paper_url, api_version='v2')
            self._conectado = True
            acc = self._api.get_account()
            self.capital = float(acc.equity)
            return True
        except Exception as e:
            self._conectado = False
            return False

    def desconectar(self):
        self._api = None
        self._conectado = False

    def obtener_precio(self, ticker: str) -> float:
        if self._api:
            try:
                trade = self._api.get_latest_trade(ticker)
                return float(trade.price)
            except Exception:
                pass
        return 100.0

    def enviar_orden(self, ticker: str, lado: LadoOrden, shares: int,
                     tipo: TipoOrden = TipoOrden.MERCADO,
                     precio_limite: Optional[float] = None) -> Orden:
        order_id = f'ALPACA_{uuid.uuid4().hex[:12]}'
        if not self._api:
            return Orden(order_id=order_id, ticker=ticker, lado=lado,
                         tipo=tipo, shares_solicitadas=shares,
                         estado=EstadoOrden.RECHAZADA,
                         broker=self.nombre, error='No conectado',
                         created=datetime.now().isoformat())
        try:
            side = 'buy' if lado == LadoOrden.COMPRA else 'sell'
            alpaca_order = self._api.submit_order(
                symbol=ticker, qty=shares, side=side,
                type='market', time_in_force='day')
            orden = Orden(order_id=order_id, ticker=ticker, lado=lado,
                          tipo=tipo, shares_solicitadas=shares,
                          shares_llenas=int(alpaca_order.filled_qty or shares),
                          precio_lleno=float(alpaca_order.filled_avg_price or 0),
                          estado=EstadoOrden.LLENADA,
                          broker=self.nombre,
                          created=datetime.now().isoformat())
            self._registrar_orden(orden)
            return orden
        except Exception as e:
            return Orden(order_id=order_id, ticker=ticker, lado=lado,
                         tipo=tipo, shares_solicitadas=shares,
                         estado=EstadoOrden.RECHAZADA,
                         broker=self.nombre, error=str(e),
                         created=datetime.now().isoformat())

    def cancelar_orden(self, order_id: str) -> bool:
        return False

    def obtener_posiciones(self) -> List[Posicion]:
        if not self._api:
            return []
        try:
            alpaca_pos = self._api.list_positions()
            return [Posicion(ticker=p.symbol, cantidad=int(p.qty),
                             precio_promedio=float(p.avg_entry_price),
                             valor_actual=float(p.current_price),
                             pnl_no_realizado=float(p.unrealized_pl))
                    for p in alpaca_pos]
        except Exception:
            return []

    def obtener_resumen(self) -> ResumenCuenta:
        if not self._api:
            return ResumenCuenta(capital=0, valor_portafolio=0,
                                 cash_disponible=0, posiciones=[],
                                 pnl_total=0, retorno_total=0, broker=self.nombre)
        try:
            acc = self._api.get_account()
            equity = float(acc.equity)
            cash = float(acc.cash)
            return ResumenCuenta(capital=equity, valor_portafolio=equity,
                                 cash_disponible=cash,
                                 posiciones=self.obtener_posiciones(),
                                 pnl_total=float(acc.unrealized_pl),
                                 retorno_total=float(acc.equity) / 100000 - 1,
                                 broker=self.nombre)
        except Exception:
            return ResumenCuenta(capital=0, valor_portafolio=0,
                                 cash_disponible=0, posiciones=[],
                                 pnl_total=0, retorno_total=0, broker=self.nombre)


class BrokerIBKR(BrokerBase):
    def __init__(self, host: str = '127.0.0.1', port: int = 7497,
                 client_id: int = 1):
        super().__init__('ibkr')
        self.host = host
        self.port = port
        self.client_id = client_id
        self._ib = None

    def conectar(self) -> bool:
        if not IB_AVAILABLE:
            return False
        try:
            self._ib = IB()
            self._ib.connect(self.host, self.port, clientId=self.client_id)
            self._conectado = True
            return True
        except Exception:
            self._conectado = False
            return False

    def desconectar(self):
        if self._ib:
            try:
                self._ib.disconnect()
            except Exception:
                pass
        self._conectado = False

    def obtener_precio(self, ticker: str) -> float:
        if self._ib:
            try:
                contract = Stock(ticker, 'SMART', 'USD')
                self._ib.qualifyContracts(contract)
                tick = self._ib.reqMktData(contract, '', False, False)
                self._ib.sleep(1)
                return tick.last or tick.close or 100.0
            except Exception:
                pass
        return 100.0

    def enviar_orden(self, ticker: str, lado: LadoOrden, shares: int,
                     tipo: TipoOrden = TipoOrden.MERCADO,
                     precio_limite: Optional[float] = None) -> Orden:
        order_id = f'IBKR_{uuid.uuid4().hex[:12]}'
        if not self._ib:
            return Orden(order_id=order_id, ticker=ticker, lado=lado,
                         tipo=tipo, shares_solicitadas=shares,
                         estado=EstadoOrden.RECHAZADA,
                         broker=self.nombre, error='No conectado',
                         created=datetime.now().isoformat())
        try:
            contract = Stock(ticker, 'SMART', 'USD')
            self._ib.qualifyContracts(contract)
            action = 'BUY' if lado == LadoOrden.COMPRA else 'SELL'
            if tipo == TipoOrden.LIMITE and precio_limite:
                order = LimitOrder(action, shares, precio_limite)
            else:
                order = MarketOrder(action, shares)
            trade = self._ib.placeOrder(contract, order)
            self._ib.sleep(1)
            estado = EstadoOrden.LLENADA if trade.orderStatus.status == 'Filled' else EstadoOrden.ENVIADA
            orden = Orden(order_id=order_id, ticker=ticker, lado=lado,
                          tipo=tipo, shares_solicitadas=shares,
                          shares_llenas=int(trade.orderStatus.filled or shares),
                          precio_lleno=float(trade.orderStatus.avgFillPrice or 0),
                          estado=estado, broker=self.nombre,
                          created=datetime.now().isoformat())
            self._registrar_orden(orden)
            return orden
        except Exception as e:
            return Orden(order_id=order_id, ticker=ticker, lado=lado,
                         tipo=tipo, shares_solicitadas=shares,
                         estado=EstadoOrden.RECHAZADA,
                         broker=self.nombre, error=str(e),
                         created=datetime.now().isoformat())

    def cancelar_orden(self, order_id: str) -> bool:
        return False

    def obtener_posiciones(self) -> List[Posicion]:
        if not self._ib:
            return []
        try:
            portfolio = self._ib.portfolio()
            return [Posicion(ticker=p.contract.symbol,
                             cantidad=int(p.position),
                             precio_promedio=float(p.avgCost),
                             valor_actual=float(p.marketValue) / abs(p.position) if p.position else 0,
                             pnl_no_realizado=float(p.unrealizedPNL))
                    for p in portfolio]
        except Exception:
            return []

    def obtener_resumen(self) -> ResumenCuenta:
        if not self._ib:
            return ResumenCuenta(capital=0, valor_portafolio=0,
                                 cash_disponible=0, posiciones=[],
                                 pnl_total=0, retorno_total=0, broker=self.nombre)
        try:
            account = self._ib.accountSummary()
            summary = {item.tag: item.value for item in account}
            equity = float(summary.get('NetLiquidation', 0))
            cash = float(summary.get('CashBalance', 0))
            return ResumenCuenta(capital=equity, valor_portafolio=equity,
                                 cash_disponible=cash,
                                 posiciones=self.obtener_posiciones(),
                                 pnl_total=float(summary.get('UnrealizedPnL', 0)),
                                 retorno_total=0, broker=self.nombre)
        except Exception:
            return ResumenCuenta(capital=0, valor_portafolio=0,
                                 cash_disponible=0, posiciones=[],
                                 pnl_total=0, retorno_total=0, broker=self.nombre)


class GestorBrokers:
    def __init__(self):
        self.brokers: Dict[str, BrokerBase] = {}
        self._activo: Optional[str] = None

    def registrar(self, nombre: str, broker: BrokerBase):
        self.brokers[nombre] = broker

    def activar(self, nombre: str) -> bool:
        if nombre not in self.brokers:
            return False
        if self.brokers[nombre].conectar():
            self._activo = nombre
            return True
        return False

    def activo(self) -> Optional[BrokerBase]:
        return self.brokers.get(self._activo)

    def crear_brokers_default(self, capital_inicial: float = 100000) -> 'GestorBrokers':
        self.registrar('paper', BrokerPaper(capital_inicial=capital_inicial))
        if ALPACA_AVAILABLE:
            self.registrar('alpaca', BrokerAlpaca())
        if IB_AVAILABLE:
            self.registrar('ibkr', BrokerIBKR())
        self.activar('paper')
        return self

    def enviar_orden(self, ticker: str, lado: LadoOrden, shares: int,
                     tipo: TipoOrden = TipoOrden.MERCADO) -> Optional[Orden]:
        b = self.activo()
        if not b:
            return None
        return b.enviar_orden(ticker, lado, shares, tipo)

    def obtener_resumen(self) -> Optional[ResumenCuenta]:
        b = self.activo()
        if not b:
            return None
        return b.obtener_resumen()

    def obtener_precio(self, ticker: str) -> float:
        b = self.activo()
        if not b:
            return 0.0
        return b.obtener_precio(ticker)

    def cambiar_broker(self, nombre: str, capital: float = 100000) -> bool:
        if nombre not in self.brokers:
            if nombre == 'paper':
                self.registrar('paper', BrokerPaper(capital_inicial=capital))
            else:
                return False
        return self.activar(nombre)

    def estado_conexiones(self) -> Dict[str, bool]:
        return {n: b.conectado for n, b in self.brokers.items()}


gestor = GestorBrokers()

if __name__ == '__main__':
    g = GestorBrokers()
    g.crear_brokers_default(100000)
    o = g.enviar_orden('NVDA', LadoOrden.COMPRA, 10)
    print(f'Orden: {o.estado.value} precio={o.precio_lleno}')
    res = g.obtener_resumen()
    print(f'Capital: ${res.valor_portafolio:.2f}')
    print(f'Conexiones: {g.estado_conexiones()}')
