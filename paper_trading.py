#!/usr/bin/env python3
"""
paper_trading.py - Paper trading engine for IBKR and Alpaca.
Simulates real trading with live market data, order management,
position tracking, P&L reporting. Supports multiple brokers.
"""
import json
import os
import time
import uuid
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional, Tuple, Callable
from dataclasses import dataclass, field, asdict
from enum import Enum

from config.settings import get_setting

DATA_DIR = os.environ.get('GITHUB_WORKSPACE', '.')
OUTPUT_DIR = Path(DATA_DIR) / 'Datos' / 'paper_trading'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

ALPACA_KEY = os.environ.get('ALPACA_API_KEY', '')
ALPACA_SECRET = os.environ.get('ALPACA_SECRET_KEY', '')
ALPACA_PAPER = os.environ.get('ALPACA_PAPER_URL', 'https://paper-api.alpaca.markets')

IBKR_HOST = os.environ.get('IBKR_HOST', '127.0.0.1')
IBKR_PORT = int(os.environ.get('IBKR_PORT', '7497'))  # 7497 = paper, 7496 = live
IBKR_CLIENT_ID = int(os.environ.get('IBKR_CLIENT_ID', '1'))


class OrderSide(Enum):
    BUY = 'buy'
    SELL = 'sell'

class OrderType(Enum):
    MARKET = 'market'
    LIMIT = 'limit'
    STOP = 'stop'
    TRAILING_STOP = 'trailing_stop'

class OrderStatus(Enum):
    PENDING = 'pending'
    SUBMITTED = 'submitted'
    FILLED = 'filled'
    PARTIALLY_FILLED = 'partially_filled'
    CANCELLED = 'cancelled'
    REJECTED = 'rejected'
    EXPIRED = 'expired'


@dataclass
class Order:
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    ticker: str = ''
    side: OrderSide = OrderSide.BUY
    order_type: OrderType = OrderType.MARKET
    shares: int = 0
    price: float = 0.0
    stop_price: float = 0.0
    trail_percent: float = 0.0
    status: OrderStatus = OrderStatus.PENDING
    created_at: str = ''
    filled_at: str = ''
    filled_price: float = 0.0
    filled_shares: int = 0
    commission: float = 0.0
    reason: str = ''
    metadata: Dict = field(default_factory=dict)

    def to_dict(self):
        d = asdict(self)
        d['side'] = self.side.value
        d['order_type'] = self.order_type.value
        d['status'] = self.status.value
        return d


@dataclass
class Position:
    ticker: str = ''
    shares: int = 0
    avg_entry: float = 0.0
    current_price: float = 0.0
    pnl_unrealized: float = 0.0
    pnl_realized: float = 0.0
    pnl_pct: float = 0.0
    cost_basis: float = 0.0
    market_value: float = 0.0
    entry_date: str = ''
    sector: str = ''

    def update_market_value(self, price: float):
        self.current_price = price
        self.market_value = self.shares * price
        self.pnl_unrealized = self.market_value - self.cost_basis
        self.pnl_pct = (self.pnl_unrealized / self.cost_basis * 100) if self.cost_basis > 0 else 0.0


class PaperBroker:
    def __init__(self, initial_capital: float = 100000.0, commission: float = 0.0):
        self.capital = initial_capital
        self.initial_capital = initial_capital
        self.commission = commission
        self.positions: Dict[str, Position] = {}
        self.orders: List[Order] = []
        self.trade_history: List[Dict] = []
        self.equity_curve: List[Dict] = []
        self.account_id = f'paper_{uuid.uuid4().hex[:6]}'
        self.start_time = datetime.now(timezone.utc).isoformat()

    @property
    def portfolio_value(self) -> float:
        return self.capital + sum(p.market_value for p in self.positions.values())

    @property
    def total_pnl(self) -> float:
        return self.portfolio_value - self.initial_capital

    @property
    def total_pnl_pct(self) -> float:
        return (self.total_pnl / self.initial_capital * 100) if self.initial_capital > 0 else 0.0

    @property
    def buying_power(self) -> float:
        return self.capital * 2  # 2:1 margin

    @property
    def n_positions(self) -> int:
        return len([p for p in self.positions.values() if p.shares != 0])

    def submit_order(self,
                     ticker: str,
                     side: OrderSide,
                     shares: int,
                     order_type: OrderType = OrderType.MARKET,
                     price: float = 0.0,
                     stop_price: float = 0.0,
                     trail_percent: float = 0.0,
                     **kwargs) -> Order:
        if shares <= 0:
            raise ValueError('Shares must be positive')

        if side == OrderSide.BUY:
            cost = shares * (price or self._get_price(ticker))
            if cost > self.buying_power:
                shares = int(self.buying_power / (price or self._get_price(ticker)))
                if shares <= 0:
                    return Order(ticker=ticker, status=OrderStatus.REJECTED,
                                 reason='Insufficient buying power')

        order = Order(
            ticker=ticker.upper(),
            side=side,
            order_type=order_type,
            shares=shares,
            price=price,
            stop_price=stop_price,
            trail_percent=trail_percent,
            created_at=datetime.now(timezone.utc).isoformat(),
            **kwargs
        )
        self.orders.append(order)
        self._process_order(order)
        return order

    def _process_order(self, order: Order):
        fill_price = self._simulate_fill(order)
        if fill_price is None:
            order.status = OrderStatus.REJECTED
            order.reason = 'Could not fill'
            return

        order.filled_price = fill_price
        order.filled_shares = order.shares
        order.filled_at = datetime.now(timezone.utc).isoformat()
        order.status = OrderStatus.FILLED
        order.commission = max(1.0, order.shares * self.commission)

        if order.side == OrderSide.BUY:
            self._execute_buy(order)
        else:
            self._execute_sell(order)

        self.trade_history.append(order.to_dict())
        self._record_snapshot(order.ticker)

    def _simulate_fill(self, order: Order) -> Optional[float]:
        price = self._get_price(order.ticker)
        if price is None or price <= 0:
            return None

        # Market impact: 5bps slippage
        slippage = price * 0.0005
        if order.side == OrderSide.BUY:
            return price + slippage
        return price - slippage

    def _execute_buy(self, order: Order):
        cost = order.filled_price * order.filled_shares + order.commission
        if cost > self.capital:
            return

        self.capital -= cost

        if order.ticker in self.positions:
            pos = self.positions[order.ticker]
            total_cost = pos.cost_basis + cost
            total_shares = pos.shares + order.filled_shares
            pos.avg_entry = total_cost / total_shares if total_shares > 0 else 0
            pos.shares = total_shares
            pos.cost_basis = total_cost
        else:
            self.positions[order.ticker] = Position(
                ticker=order.ticker,
                shares=order.filled_shares,
                avg_entry=order.filled_price,
                cost_basis=cost,
                entry_date=datetime.now(timezone.utc).strftime('%Y-%m-%d')
            )

    def _execute_sell(self, order: Order):
        if order.ticker not in self.positions:
            return

        pos = self.positions[order.ticker]
        sell_shares = min(order.filled_shares, pos.shares)
        proceeds = order.filled_price * sell_shares - order.commission
        self.capital += proceeds

        realized_pnl = (order.filled_price - pos.avg_entry) * sell_shares
        pos.pnl_realized += realized_pnl
        pos.shares -= sell_shares
        pos.cost_basis = pos.cost_basis * (pos.shares / (pos.shares + sell_shares)) if (pos.shares + sell_shares) > 0 else 0

        if pos.shares <= 0:
            del self.positions[order.ticker]

    def _get_price(self, ticker: str) -> float:
        return getattr(self, '_price_provider', lambda t: 100.0)(ticker)

    def set_price_provider(self, provider: Callable[[str], float]):
        self._price_provider = provider

    def update_prices(self, prices: Dict[str, float]):
        for ticker, price in prices.items():
            if ticker in self.positions:
                self.positions[ticker].update_market_value(price)
        self._record_snapshot('prices_update')

    def _record_snapshot(self, trigger: str = ''):
        self.equity_curve.append({
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'portfolio_value': round(self.portfolio_value, 2),
            'cash': round(self.capital, 2),
            'n_positions': self.n_positions,
            'total_pnl': round(self.total_pnl, 2),
            'total_pnl_pct': round(self.total_pnl_pct, 2),
            'trigger': trigger
        })

    def cancel_order(self, order_id: str) -> bool:
        for o in self.orders:
            if o.id == order_id and o.status in (OrderStatus.PENDING, OrderStatus.SUBMITTED):
                o.status = OrderStatus.CANCELLED
                return True
        return False

    def close_position(self, ticker: str) -> Optional[Order]:
        if ticker not in self.positions:
            return None
        pos = self.positions[ticker]
        return self.submit_order(ticker, OrderSide.SELL, pos.shares)

    def close_all(self) -> List[Order]:
        orders = []
        for ticker in list(self.positions.keys()):
            o = self.close_position(ticker)
            if o:
                orders.append(o)
        return orders

    def get_account_summary(self) -> Dict:
        return {
            'account_id': self.account_id,
            'broker': 'paper',
            'initial_capital': self.initial_capital,
            'cash': round(self.capital, 2),
            'portfolio_value': round(self.portfolio_value, 2),
            'total_pnl': round(self.total_pnl, 2),
            'total_pnl_pct': round(self.total_pnl_pct, 2),
            'n_positions': self.n_positions,
            'buying_power': round(self.buying_power, 2),
            'n_orders': len(self.orders),
            'n_trades': len(self.trade_history),
            'started_at': self.start_time,
            'positions': {t: {
                'shares': p.shares,
                'avg_entry': round(p.avg_entry, 2),
                'current_price': round(p.current_price, 2),
                'cost_basis': round(p.cost_basis, 2),
                'market_value': round(p.market_value, 2),
                'pnl_unrealized': round(p.pnl_unrealized, 2),
                'pnl_pct': round(p.pnl_pct, 2),
                'entry_date': p.entry_date
            } for t, p in self.positions.items() if p.shares > 0}
        }

    def save_state(self):
        state = {
            'account_id': self.account_id,
            'initial_capital': self.initial_capital,
            'capital': self.capital,
            'start_time': self.start_time,
            'positions': {t: asdict(p) for t, p in self.positions.items()},
            'trade_history': self.trade_history[-500:],
            'equity_curve': self.equity_curve[-1000:],
            'updated_at': datetime.now(timezone.utc).isoformat()
        }
        path = OUTPUT_DIR / f'account_{self.account_id}.json'
        with open(path, 'w') as f:
            json.dump(state, f, indent=2)
        return path

    def load_state(self, account_id: str) -> bool:
        path = OUTPUT_DIR / f'account_{account_id}.json'
        if not path.exists():
            return False
        
        with open(path) as f:
            state = json.load(f)
        
        self.account_id = state['account_id']
        self.initial_capital = state['initial_capital']
        self.capital = state['capital']
        self.start_time = state['start_time']
        self.positions = {t: Position(**p) for t, p in state.get('positions', {}).items()}
        self.trade_history = state.get('trade_history', [])
        self.equity_curve = state.get('equity_curve', [])
        return True


class AlpacaBroker(PaperBroker):
    def __init__(self, api_key: str = ALPACA_KEY, secret_key: str = ALPACA_SECRET,
                 paper: bool = True, initial_capital: float = 100000.0):
        super().__init__(initial_capital)
        self.api_key = api_key
        self.secret_key = secret_key
        self.base_url = ALPACA_PAPER if paper else 'https://api.alpaca.markets'
        self.live = not paper
        self._client = None

    @property
    def client(self):
        if self._client is None and self.api_key and self.secret_key:
            try:
                from alpaca.trading.client import TradingClient
                self._client = TradingClient(self.api_key, self.secret_key, paper=not self.live)
            except ImportError:
                print('[Alpaca] Install: pip install alpaca-py')
            except Exception as e:
                print(f'[Alpaca] Connection failed: {e}')
        return self._client

    def submit_order(self, ticker, side, shares, order_type=OrderType.MARKET,
                     price=0.0, stop_price=0.0, trail_percent=0.0, **kwargs) -> Order:
        if self.client:
            try:
                from alpaca.trading.requests import MarketOrderRequest, LimitOrderRequest
                from alpaca.trading.enums import OrderSide as AlpacaSide, OrderType as AlpacaType, TimeInForce
                
                alpaca_side = AlpacaSide.BUY if side == OrderSide.BUY else AlpacaSide.SELL
                
                if order_type == OrderType.MARKET:
                    req = MarketOrderRequest(
                        symbol=ticker.upper(),
                        qty=shares,
                        side=alpaca_side,
                        time_in_force=TimeInForce.DAY
                    )
                else:
                    req = LimitOrderRequest(
                        symbol=ticker.upper(),
                        qty=shares,
                        side=alpaca_side,
                        limit_price=price,
                        time_in_force=TimeInForce.DAY
                    )
                
                alpaca_order = self.client.submit_order(req)
                return self._alpaca_to_order(alpaca_order)
            except Exception as e:
                print(f'[Alpaca] Order failed: {e}')
        
        return super().submit_order(ticker, side, shares, order_type, price, stop_price, trail_percent)

    def _alpaca_to_order(self, alpaca_order) -> Order:
        return Order(
            id=alpaca_order.id[:8] if hasattr(alpaca_order, 'id') else str(uuid.uuid4())[:8],
            ticker=alpaca_order.symbol if hasattr(alpaca_order, 'symbol') else '',
            side=OrderSide.BUY if getattr(alpaca_order, 'side', '').lower() == 'buy' else OrderSide.SELL,
            shares=int(getattr(alpaca_order, 'qty', 0)),
            status=OrderStatus.FILLED if getattr(alpaca_order, 'status', '') == 'filled' else OrderStatus.SUBMITTED,
            filled_shares=int(getattr(alpaca_order, 'filled_qty', 0)),
            filled_price=float(getattr(alpaca_order, 'filled_avg_price', 0)),
            created_at=str(getattr(alpaca_order, 'created_at', '')),
            filled_at=str(getattr(alpaca_order, 'filled_at', ''))
        )

    def get_positions(self) -> List[Dict]:
        if not self.client:
            return []
        try:
            positions = self.client.get_all_positions()
            return [{
                'ticker': p.symbol,
                'shares': int(p.qty),
                'avg_entry': float(p.avg_entry_price),
                'current_price': float(p.current_price),
                'pnl_unrealized': float(p.unrealized_pl),
                'pnl_pct': float(p.unrealized_plpc) * 100,
                'market_value': float(p.market_value)
            } for p in positions]
        except Exception as e:
            print(f'[Alpaca] Get positions failed: {e}')
            return []

    def get_account(self) -> Dict:
        if not self.client:
            return super().get_account_summary()
        try:
            acct = self.client.get_account()
            return {
                'cash': float(acct.cash),
                'portfolio_value': float(acct.portfolio_value),
                'buying_power': float(acct.buying_power),
                'day_trade_count': int(acct.daytrade_count),
                'status': acct.status
            }
        except:
            return super().get_account_summary()


class IBKRBroker(PaperBroker):
    def __init__(self, host: str = IBKR_HOST, port: int = IBKR_PORT,
                 client_id: int = IBKR_CLIENT_ID, initial_capital: float = 100000.0):
        super().__init__(initial_capital)
        self.host = host
        self.port = port
        self.client_id = client_id
        self._app = None
        self._connected = False

    @property
    def app(self):
        if self._app is None:
            try:
                from ib_insync import IB, Stock
                self._app = IB()
                self._app.connect(self.host, self.port, clientId=self.client_id)
                self._connected = True
                print(f'[IBKR] Connected to TWS/Gateway at {self.host}:{self.port}')
            except ImportError:
                print('[IBKR] Install: pip install ib_insync')
            except Exception as e:
                print(f'[IBKR] Connection failed: {e}')
        return self._app

    def submit_order(self, ticker, side, shares, order_type=OrderType.MARKET,
                     price=0.0, stop_price=0.0, trail_percent=0.0, **kwargs) -> Order:
        if self.app and self._connected:
            try:
                from ib_insync import Stock, MarketOrder, LimitOrder, StopOrder
                contract = Stock(ticker.upper(), 'SMART', 'USD')
                
                ib_side = 'BUY' if side == OrderSide.BUY else 'SELL'
                if order_type == OrderType.MARKET:
                    ib_order = MarketOrder(ib_side, shares)
                elif order_type == OrderType.LIMIT:
                    ib_order = LimitOrder(ib_side, shares, price)
                elif order_type == OrderType.STOP:
                    ib_order = StopOrder(ib_side, shares, stop_price)
                else:
                    ib_order = MarketOrder(ib_side, shares)
                
                trade = self.app.placeOrder(contract, ib_order)
                time.sleep(0.5)
                
                return Order(
                    ticker=ticker.upper(),
                    side=side,
                    shares=shares,
                    status=OrderStatus.SUBMITTED,
                    created_at=datetime.now(timezone.utc).isoformat()
                )
            except Exception as e:
                print(f'[IBKR] Order failed: {e}')
        
        return super().submit_order(ticker, side, shares, order_type, price, stop_price, trail_percent)

    def get_positions(self) -> List[Dict]:
        if not self.app:
            return []
        try:
            positions = self.app.positions()
            return [{
                'ticker': p.contract.symbol,
                'shares': int(p.position),
                'avg_entry': float(p.avgCost),
                'current_price': float(p.marketPrice) if hasattr(p, 'marketPrice') else 0
            } for p in positions]
        except:
            return []

    def disconnect(self):
        if self.app:
            self.app.disconnect()
            self._connected = False


class PaperTradingManager:
    def __init__(self):
        self.broker = PaperBroker()
        self.broker_type = 'paper'
        self.strategies = {}
        self.active = False
        self.scheduler = None

    def connect_alpaca(self, api_key: str = ALPACA_KEY, secret: str = ALPACA_SECRET) -> AlpacaBroker:
        self.broker = AlpacaBroker(api_key, secret)
        self.broker_type = 'alpaca'
        return self.broker

    def connect_ibkr(self, host: str = IBKR_HOST, port: int = IBKR_PORT) -> IBKRBroker:
        self.broker = IBKRBroker(host, port)
        self.broker_type = 'ibkr'
        return self.broker

    def execute_signal(self, ticker: str, signal: Dict) -> Optional[Order]:
        direction = signal.get('direction', 'hold')
        confidence = signal.get('confidence', 50)
        price = signal.get('price', 0)
        
        if direction == 'hold' or confidence < 30:
            return None
        
        position = self.broker.positions.get(ticker)
        current_shares = position.shares if position else 0
        portfolio_value = self.broker.portfolio_value
        max_position_value = portfolio_value * 0.12
        
        if direction == 'buy':
            target_value = max_position_value * (confidence / 100)
            target_shares = int(target_value / price) if price > 0 else 0
            shares_to_buy = max(0, target_shares - current_shares)
            
            if shares_to_buy > 0:
                return self.broker.submit_order(
                    ticker=ticker,
                    side=OrderSide.BUY,
                    shares=min(shares_to_buy, 200),
                    order_type=OrderType.MARKET
                )
        elif direction == 'sell':
            if current_shares > 0:
                sell_pct = min(confidence / 100, 1.0)
                shares_to_sell = int(current_shares * sell_pct)
                if shares_to_sell > 0:
                    return self.broker.submit_order(
                        ticker=ticker,
                        side=OrderSide.SELL,
                        shares=shares_to_sell,
                        order_type=OrderType.MARKET
                    )
        return None

    def apply_portfolio_weights(self, weights: Dict[str, float], prices: Dict[str, float]):
        """Apply target portfolio weights (rebalance)."""
        portfolio_value = self.broker.portfolio_value
        current = self.broker.positions
        
        for ticker, target_weight in weights.items():
            if ticker not in prices:
                continue
            price = prices[ticker]
            target_value = portfolio_value * target_weight
            current_shares = current.get(ticker, Position()).shares if ticker in current else 0
            current_value = current_shares * price
            diff = target_value - current_value
            
            if abs(diff) < portfolio_value * 0.005:
                continue
            
            shares = int(diff / price)
            if shares > 0:
                self.broker.submit_order(ticker, OrderSide.BUY, abs(shares))
            elif shares < 0:
                self.broker.submit_order(ticker, OrderSide.SELL, abs(shares))

    def generate_report(self) -> Dict:
        summary = self.broker.get_account_summary()
        
        if len(self.broker.equity_curve) >= 2:
            import numpy as np
            values = [e['portfolio_value'] for e in self.broker.equity_curve]
            returns = np.diff(values) / values[:-1]
            if len(returns) > 1:
                ann_return = (values[-1] / values[0]) ** (252 / len(returns)) - 1
                ann_vol = np.std(returns) * np.sqrt(252)
                sharpe = (ann_return - 0.05) / ann_vol if ann_vol > 0 else 0
                cummax = np.maximum.accumulate(values)
                drawdown = (np.array(values) - cummax) / cummax
                max_dd = abs(float(min(drawdown)))
                
                summary['metrics'] = {
                    'annualized_return_pct': round(ann_return * 100, 2),
                    'annualized_vol_pct': round(ann_vol * 100, 2),
                    'sharpe_ratio': round(sharpe, 3),
                    'max_drawdown_pct': round(max_dd * 100, 2),
                    'win_rate_pct': round(float(np.mean(returns > 0)) * 100, 1),
                    'n_trades': len(self.broker.trade_history)
                }
        
        report_path = OUTPUT_DIR / 'paper_trading_report.json'
        with open(report_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        return summary

    def run_scheduled(self, signal_fn: Callable, interval_minutes: int = 60):
        """Run paper trading loop. Call this periodically."""
        if not self.active:
            return
        
        signals = signal_fn()
        for ticker, signal in signals.items():
            self.execute_signal(ticker, signal)
        
        self.broker.save_state()
        self.generate_report()

    def start(self):
        self.active = True
        print(f'[PaperTrading] Started with {self.broker_type} broker, '
              f'capital=${self.broker.initial_capital:,.0f}')

    def stop(self):
        self.active = False
        if hasattr(self.broker, 'disconnect'):
            self.broker.disconnect()
        self.broker.save_state()
        print(f'[PaperTrading] Stopped. Final value=${self.broker.portfolio_value:,.2f}')


_manager = None

def get_paper_trading_manager() -> PaperTradingManager:
    global _manager
    if _manager is None:
        _manager = PaperTradingManager()
    return _manager


if __name__ == '__main__':
    manager = get_paper_trading_manager()
    
    def dummy_signal():
        return {t: {'direction': 'buy', 'confidence': 65, 'price': 100 + i * 10}
                for i, t in enumerate(['NVDA', 'AAPL', 'MSFT'][:3])}
    
    manager.start()
    manager.broker.set_price_provider(lambda t: 150.0 if t == 'NVDA' else 180.0 if t == 'AAPL' else 350.0)
    
    for ticker, signal in dummy_signal().items():
        order = manager.execute_signal(ticker, signal)
        if order:
            print(f'  {order.side.value.upper()} {order.shares} {order.ticker} @ ${order.filled_price:.2f}')
    
    manager.broker.update_prices({'NVDA': 155.0, 'AAPL': 182.0, 'MSFT': 345.0})
    
    report = manager.generate_report()
    print(f'\nPortfolio: ${report["portfolio_value"]:,.2f} | PnL: ${report["total_pnl"]:,.2f} ({report["total_pnl_pct"]:.2f}%)')
    
    manager.stop()