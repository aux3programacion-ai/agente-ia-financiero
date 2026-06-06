#!/usr/bin/env python3
"""broker_api.py - Paper-to-Live Bridge: IBKR/Alpaca execution, position sync, risk checks."""

import json, os, sys, time, datetime, warnings, threading
warnings.filterwarnings('ignore')

DATA_DIR = os.environ.get('GITHUB_WORKSPACE', '.')
MODE = os.environ.get('BROKER_MODE', 'paper')  # paper | live
BROKER = os.environ.get('BROKER', 'ibkr')       # ibkr | alpaca

# IBKR paper defaults
IBKR_HOST = os.environ.get('IBKR_HOST', '127.0.0.1')
IBKR_PORT = int(os.environ.get('IBKR_PORT', 7497))   # 7497=paper, 7496=live
IBKR_CLIENT_ID = int(os.environ.get('IBKR_CLIENT_ID', 1))

# Alpaca defaults
ALPACA_KEY = os.environ.get('ALPACA_KEY', '')
ALPACA_SECRET = os.environ.get('ALPACA_SECRET', '')
ALPACA_PAPER = True if MODE == 'paper' else False

class BrokerError(Exception):
    pass

class BrokerAPI:
    def __init__(self):
        self.connected = False
        self.broker = BROKER
        self.mode = MODE
        self.client = None
        self.positions = {}
        self.account_info = {}
        self._connect()

    def _connect(self):
        if self.broker == 'ibkr':
            self._connect_ibkr()
        elif self.broker == 'alpaca':
            self._connect_alpaca()
        else:
            raise BrokerError(f'Unknown broker: {self.broker}')

    def _connect_ibkr(self):
        try:
            from ib_insync import IB, Stock, MarketOrder, LimitOrder, Trade
            self.IB = IB
            self.Stock = Stock
            self.MarketOrder = MarketOrder
            self.LimitOrder = LimitOrder
            self.ib = IB()
            port_label = 'paper' if self.mode == 'paper' else 'live'
            print(f'[Broker] Conectando IBKR {port_label} @ {IBKR_HOST}:{IBKR_PORT}...')
            self.ib.connect(IBKR_HOST, IBKR_PORT, clientId=IBKR_CLIENT_ID, timeout=10)
            self.connected = True
            self._sync_positions_ibkr()
            self._sync_account_ibkr()
            print(f'[Broker] IBKR conectado. Cash: ${self.account_info.get("cash",0):.2f}')
        except ImportError:
            raise BrokerError('ib_insync not installed. pip install ib_insync')
        except Exception as e:
            raise BrokerError(f'IBKR connect failed: {e}')

    def _connect_alpaca(self):
        try:
            from alpaca.trading.client import TradingClient
            from alpaca.trading.requests import MarketOrderRequest, LimitOrderRequest
            from alpaca.trading.enums import OrderSide, TimeInForce, OrderType
            self.AlpacaClient = TradingClient
            self.MarketOrderRequest = MarketOrderRequest
            self.LimitOrderRequest = LimitOrderRequest
            self.OrderSide = OrderSide
            self.TimeInForce = TimeInForce
            self.OrderType_ = OrderType
            self.alpaca = TradingClient(ALPACA_KEY, ALPACA_SECRET, paper=ALPACA_PAPER)
            self.connected = True
            self._sync_positions_alpaca()
            self._sync_account_alpaca()
            print(f'[Broker] Alpaca conectado. Cash: ${self.account_info.get("cash",0):.2f}')
        except ImportError:
            raise BrokerError('alpaca-py not installed. pip install alpaca-py')
        except Exception as e:
            raise BrokerError(f'Alpaca connect failed: {e}')

    def _sync_positions_ibkr(self):
        try:
            self.ib.reqPositions()
            time.sleep(1)
            self.positions = {}
            for p in self.ib.positions():
                ticker = p.contract.symbol
                self.positions[ticker] = {
                    'qty': p.position,
                    'avg_price': float(p.avgCost) if p.avgCost else 0,
                    'market_price': 0.0,
                    'market_value': 0.0
                }
            # Get market prices
            contracts = [self.Stock(t) for t in self.positions]
            if contracts:
                self.ib.reqMarketDataType(1)  # Live or frozen
                ticks = self.ib.reqTickers(*contracts)
                for t in ticks:
                    ticker = t.contract.symbol
                    if ticker in self.positions:
                        self.positions[ticker]['market_price'] = float(t.marketPrice() or 0)
                        self.positions[ticker]['market_value'] = self.positions[ticker]['qty'] * self.positions[ticker]['market_price']
            print(f'[Broker] {len(self.positions)} posiciones sincronizadas')
        except Exception as e:
            print(f'[!] IBKR position sync: {e}')

    def _sync_account_ibkr(self):
        try:
            acc = self.ib.accountSummary()
            summary = {item.tag: item.value for item in acc}
            self.account_info = {
                'cash': float(summary.get('TotalCashValue', 0)),
                'equity': float(summary.get('NetLiquidation', 0)),
                'buying_power': float(summary.get('BuyingPower', 0)),
                'gross_positions': float(summary.get('GrossPositionValue', 0)),
                'init_margin': float(summary.get('InitMarginReq', 0)),
                'maint_margin': float(summary.get('MaintMarginReq', 0)),
                'unrealized_pnl': float(summary.get('UnrealizedPnL', 0)),
                'realized_pnl': float(summary.get('RealizedPnL', 0))
            }
        except Exception as e:
            print(f'[!] IBKR account sync: {e}')

    def _sync_positions_alpaca(self):
        try:
            positions = self.alpaca.get_all_positions()
            self.positions = {}
            for p in positions:
                ticker = p.symbol
                self.positions[ticker] = {
                    'qty': float(p.qty),
                    'avg_price': float(p.avg_entry_price),
                    'market_price': float(p.current_price),
                    'market_value': float(p.market_value)
                }
            print(f'[Broker] {len(self.positions)} posiciones sincronizadas')
        except Exception as e:
            print(f'[!] Alpaca position sync: {e}')

    def _sync_account_alpaca(self):
        try:
            acc = self.alpaca.get_account()
            self.account_info = {
                'cash': float(acc.cash),
                'equity': float(acc.equity),
                'buying_power': float(acc.buying_power),
                'gross_positions': float(acc.long_market_value + acc.short_market_value),
                'init_margin': float(acc.initial_margin),
                'maint_margin': float(acc.maintenance_margin),
                'unrealized_pnl': float(acc.unrealized_pl),
                'realized_pnl': float(acc.realized_pl)
            }
        except Exception as e:
            print(f'[!] Alpaca account sync: {e}')

    def sync(self):
        if self.broker == 'ibkr':
            self._sync_positions_ibkr()
            self._sync_account_ibkr()
        else:
            self._sync_positions_alpaca()
            self._sync_account_alpaca()
        return self.account_info, self.positions

    def place_order(self, ticker, qty, side, order_type='market', limit_price=None):
        if qty <= 0:
            raise BrokerError(f'Invalid qty: {qty}')
        if self.broker == 'ibkr':
            return self._place_ibkr(ticker, qty, side, order_type, limit_price)
        else:
            return self._place_alpaca(ticker, qty, side, order_type, limit_price)

    def _place_ibkr(self, ticker, qty, side, order_type, limit_price):
        contract = self.Stock(ticker)
        if order_type == 'market':
            order = self.MarketOrder('BUY' if side == 'buy' else 'SELL', qty)
        elif order_type == 'limit' and limit_price:
            order = self.LimitOrder('BUY' if side == 'buy' else 'SELL', qty, limit_price)
        else:
            raise BrokerError(f'Unsupported order type: {order_type}')
        trade = self.ib.placeOrder(contract, order)
        print(f'[Orden] {side.upper()} {qty} {ticker} @ {order_type}')
        return trade.order.orderId

    def _place_alpaca(self, ticker, qty, side, order_type, limit_price):
        side_enum = self.OrderSide.BUY if side == 'buy' else self.OrderSide.SELL
        if order_type == 'market':
            req = self.MarketOrderRequest(
                symbol=ticker,
                qty=qty,
                side=side_enum,
                time_in_force=self.TimeInForce.DAY
            )
        elif order_type == 'limit' and limit_price:
            req = self.LimitOrderRequest(
                symbol=ticker,
                limit_price=limit_price,
                qty=qty,
                side=side_enum,
                time_in_force=self.TimeInForce.DAY
            )
        else:
            raise BrokerError(f'Unsupported order type: {order_type}')
        order = self.alpaca.submit_order(req)
        print(f'[Orden] {side.upper()} {qty} {ticker} @ {order_type}')
        return order.id

    def cancel_order(self, order_id):
        if self.broker == 'ibkr':
            order = self.ib.order(order_id)
            if order:
                self.ib.cancelOrder(order)
        else:
            self.alpaca.cancel_order_by_id(order_id)
        print(f'[Orden] Cancelada: {order_id}')

    def get_orders(self):
        if self.broker == 'ibkr':
            trades = self.ib.trades()
            return [{'id': t.order.orderId, 'ticker': t.contract.symbol, 'qty': t.order.totalQuantity,
                     'side': 'BUY' if t.order.action == 'BUY' else 'SELL', 'status': t.orderStatus.status}
                    for t in trades if t.orderStatus.status in ('Submitted', 'PreSubmitted', 'Filled')]
        else:
            orders = self.alpaca.get_orders()
            return [{'id': o.id, 'ticker': o.symbol, 'qty': float(o.qty),
                     'side': o.side, 'status': o.status} for o in orders if o.status in ('new', 'accepted', 'filled', 'partially_filled')]

    def disconnect(self):
        if self.broker == 'ibkr' and self.connected:
            self.ib.disconnect()
            print('[Broker] IBKR desconectado')
        self.connected = False


# ============================================================
# EXECUTION ENGINE (VWAP/TWAP slicing)
# ============================================================
class ExecutionEngine:
    def __init__(self, broker_api, max_slice_qty=200):
        self.broker = broker_api
        self.max_slice_qty = max_slice_qty

    def execute_fills(self, positions, prices, regime_params, capital):
        """Execute buy/sell signals with TWAP slicing and risk checks."""
        cash = self.broker.account_info.get('cash', 0)
        buying_power = self.broker.account_info.get('buying_power', 0)
        max_per_position = regime_params.get('max_position', 0.08)
        risk_per_trade = regime_params.get('risk_per_trade', 0.01)

        orders = []
        for ticker, weight in positions.items():
            if weight <= 0:
                continue
            price = prices.get(ticker, {}).get('price', 0)
            if price <= 0:
                print(f'  [!] {ticker}: sin precio')
                continue

            target_value = weight * capital
            target_qty = int(target_value / price)
            if target_qty <= 0:
                continue

            current = self.broker.positions.get(ticker, {})
            current_qty = int(current.get('qty', 0))
            delta = target_qty - current_qty

            if abs(delta) < 1:
                continue
            if delta > 0:
                cost = delta * price
                if cost > buying_power * 0.95:
                    print(f'  [!] {ticker}: insuficiente buying power (${cost:.0f} > ${buying_power:.0f})')
                    continue
                if abs(delta * price / capital) > max_per_position:
                    delta = int(max_per_position * capital / price)
                    print(f'  [!] {ticker}: ajustado por max_position ({max_per_position:.0%})')

                side = 'buy'
            else:
                delta = abs(delta)
                side = 'sell'

            order_id = self._twap_slice(ticker, delta, side, price)
            if order_id:
                orders.append({'ticker': ticker, 'qty': delta, 'side': side, 'order_id': order_id})
        return orders

    def _twap_slice(self, ticker, total_qty, side, price, n_slices=3):
        """TWAP: split large orders into smaller slices over 3 intervals."""
        if total_qty <= self.max_slice_qty:
            return self.broker.place_order(ticker, total_qty, side)

        slice_qty = total_qty // n_slices
        order_ids = []
        for i in range(n_slices):
            qty = slice_qty if i < n_slices - 1 else total_qty - i * slice_qty
            if qty <= 0:
                break
            oid = self.broker.place_order(ticker, qty, side)
            order_ids.append(oid)
            if i < n_slices - 1:
                time.sleep(30)  # 30s between slices
        return order_ids[-1] if order_ids else None


# ============================================================
# RECONCILIATION
# ============================================================
def reconcile_positions(local_positions, broker_positions):
    """Compare local vs broker positions, report divergences."""
    alerts = []
    all_tickers = set(local_positions.keys()) | set(broker_positions.keys())
    for t in all_tickers:
        local_qty = local_positions.get(t, {}).get('qty', 0)
        broker_qty = broker_positions.get(t, {}).get('qty', 0)
        if abs(local_qty - broker_qty) > 1:
            alerts.append({
                'ticker': t,
                'local_qty': local_qty,
                'broker_qty': broker_qty,
                'diff': local_qty - broker_qty
            })
    if alerts:
        for a in alerts:
            print(f'[RECONCILIATION] {a["ticker"]}: local={a["local_qty"]:.0f} broker={a["broker_qty"]:.0f}')
    return alerts


# ============================================================
# CLI ENTRY POINT
# ============================================================
if __name__ == '__main__':
    print('=== Broker API / Paper-to-Live Bridge ===')
    print(f'  Mode: {MODE.upper()} | Broker: {BROKER.upper()}')

    try:
        broker = BrokerAPI()
    except BrokerError as e:
        print(f'[!] {e}')
        sys.exit(1)

    print(f'\n[Account]')
    for k, v in broker.account_info.items():
        print(f'  {k}: {v}')
    print(f'\n[Positions]')
    for t, p in sorted(broker.positions.items()):
        print(f'  {t}: {p["qty"]:.0f} @ ${p["avg_price"]:.2f} = ${p["market_value"]:.0f}')

    broker.disconnect()
