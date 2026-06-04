import json
import os
import sys
import time
import datetime
from portafolio_utils import cargar_portafolio, cargar_portafolio_cantidades

DATA_DIR = os.environ.get('GITHUB_WORKSPACE', '.')

PAPER_FILE = os.path.join(DATA_DIR, 'Datos', 'paper_trading.json')
IA_FILE = os.path.join(DATA_DIR, 'Datos', 'analisis_ia.json')
PORTFOLIO_FILE = os.path.join(DATA_DIR, 'Datos', 'portafolio_usuario.json')
PRICES_FILE = os.path.join(DATA_DIR, 'Datos', 'precios_reales.json')
OPTIM_FILE = os.path.join(DATA_DIR, 'Datos', 'optimizacion_portafolio.json')
HIST_FILE = os.path.join(DATA_DIR, 'Datos', 'predicciones_hist.json')

INITIAL_CAPITAL = 100000.0
FEE_RATE = 0.001
BUY_THRESHOLD = 60
SELL_THRESHOLD = 40


def load_json(path, default=None):
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"  No se pudo cargar {path}: {e}")
        return default


def save_json(path, data):
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        print(f"  Guardado: {path}")
    except Exception as e:
        print(f"  Error guardando {path}: {e}")


def get_earliest_date(hist):
    earliest = datetime.date.today().isoformat()
    try:
        for ticker, data in hist.items():
            for pred in data.get('predicciones', []):
                fd = pred.get('fecha', '')
                if fd and fd < earliest:
                    earliest = fd
    except Exception:
        pass
    return earliest


def classify_signal(prob):
    if prob is None:
        return 'neutral'
    if prob > BUY_THRESHOLD:
        return 'buy'
    if prob < SELL_THRESHOLD:
        return 'sell'
    return 'neutral'


def compute_max_drawdown(equity_curve):
    if not equity_curve:
        return 0.0
    peak = equity_curve[0]
    max_dd = 0.0
    for val in equity_curve:
        if val > peak:
            peak = val
        dd = (peak - val) / peak if peak > 0 else 0
        if dd > max_dd:
            max_dd = dd
    return max_dd


def run():
    print("=== Paper Trading Simulator ===")

    state = load_json(PAPER_FILE)
    ia_data = load_json(IA_FILE, {})
    portfolio_tickers = cargar_portafolio(DATA_DIR)
    portfolio_cantidades = cargar_portafolio_cantidades(DATA_DIR)
    prices_data = load_json(PRICES_FILE, {})
    optim_data = load_json(OPTIM_FILE)
    hist_data = load_json(HIST_FILE, {})

    probabilities = ia_data.get('probabilidades', {}) if ia_data else {}
    prices = prices_data.get('precios', {}) if prices_data else {}

    now_str = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    today_str = datetime.datetime.now().strftime('%Y-%m-%d')

    if state is None:
        start_date = get_earliest_date(hist_data)
        state = {
            'timestamp': now_str,
            'capital_inicial': INITIAL_CAPITAL,
            'efectivo': INITIAL_CAPITAL,
            'valor_portafolio': 0.0,
            'valor_total': INITIAL_CAPITAL,
            'retorno_total': 0.0,
            'max_drawdown': 0.0,
            'win_rate': 0.0,
            'total_trades': 0,
            'trades_ganadores': 0,
            'trades_perdedores': 0,
            'fecha_inicio': start_date,
            'posiciones': {},
            'historial_trades': [],
            'equity_curve': [INITIAL_CAPITAL]
        }
        print(f"  Nuevo portafolio creado. Capital inicial: ${INITIAL_CAPITAL:,.2f}")
    else:
        state['timestamp'] = now_str
        print(f"  Estado cargado. Efectivo: ${state.get('efectivo', 0):,.2f}")

    if not probabilities:
        print("  No hay probabilidades. Guardando estado vacio.")
        save_json(PAPER_FILE, state)
        tv = state.get('valor_total', INITIAL_CAPITAL)
        rt = state.get('retorno_total', 0)
        tt = state.get('total_trades', 0)
        wr = state.get('win_rate', 0) * 100
        dd = state.get('max_drawdown', 0) * 100
        print(f"  Paper Trading: ${tv:,.0f} ({rt:+.0%}) | {tt} trades | Win rate {wr:.0f}% | Drawdown {dd:.0f}%")
        return

    old_positions = state.get('posiciones', {})
    trades = state.get('historial_trades', [])
    total_trades = state.get('total_trades', 0)
    winners = state.get('trades_ganadores', 0)
    losers = state.get('trades_perdedores', 0)
    equity_curve = state.get('equity_curve', [INITIAL_CAPITAL])
    cash = state.get('efectivo', INITIAL_CAPITAL)

    buy_candidates = []
    for ticker, prob_data in probabilities.items():
        prob = prob_data.get('probabilidad')
        if prob is None:
            continue
        signal = classify_signal(prob)
        if signal == 'buy':
            conf = prob_data.get('confianza', 50)
            buy_candidates.append((ticker, prob, conf))

    total_confidence = sum(c[2] for c in buy_candidates) if buy_candidates else 1

    hedged_prices = {}
    for ticker in old_positions:
        if ticker in prices:
            hedged_prices[ticker] = prices[ticker]['price']
        elif ticker in probabilities:
            hedged_prices[ticker] = probabilities[ticker].get('precio_objetivo_30d', 0)
        else:
            hedged_prices[ticker] = old_positions[ticker].get('precio_compra', 0)

    for ticker in buy_candidates:
        t = ticker[0]
        if t in prices:
            hedged_prices[t] = prices[t]['price']
        elif t in probabilities:
            hedged_prices[t] = probabilities[t].get('precio_objetivo_30d', 0)

    new_positions = {}
    for ticker in old_positions:
        pos = dict(old_positions[ticker])
        if ticker in probabilities:
            prob = probabilities[ticker].get('probabilidad', 50)
            signal = classify_signal(prob)
            if signal == 'sell':
                sell_price = hedged_prices.get(ticker, pos.get('precio_actual', pos['precio_compra']))
                notional = pos['cantidad'] * sell_price
                fee = notional * FEE_RATE
                pnl = (sell_price - pos['precio_compra']) * pos['cantidad'] - fee
                cash += notional - fee
                total_trades += 1
                if pnl > 0:
                    winners += 1
                else:
                    losers += 1
                trade_record = {
                    'ticker': ticker,
                    'tipo': 'venta',
                    'fecha': today_str,
                    'precio': round(sell_price, 2),
                    'cantidad': pos['cantidad'],
                    'comision': round(fee, 2),
                    'pnl': round(pnl, 2)
                }
                trades.append(trade_record)
                print(f"  VENDIDO {ticker}: {pos['cantidad']} @ ${sell_price:.2f} | PnL: ${pnl:.2f}")
                continue
            elif signal == 'buy':
                new_positions[ticker] = pos
                continue
        new_positions[ticker] = pos

    current_tickers = set(new_positions.keys())
    for ticker, prob, conf in buy_candidates:
        if ticker not in current_tickers:
            price = hedged_prices.get(ticker, 100.0)
            if price <= 0:
                price = 100.0

            # Si el usuario especifico cantidad para este ticker, usarla
            user_qty = portfolio_cantidades.get(ticker, 0)
            if user_qty > 0:
                quantity = user_qty
            else:
                weight = conf / total_confidence
                total_buy_candidates = len(buy_candidates)
                equal_weight = 1.0 / total_buy_candidates if total_buy_candidates > 0 else 0
                blended_weight = (equal_weight * 0.5) + (weight * 0.5)
                alloc = cash * blended_weight
                quantity = int(alloc / price) if price > 0 else 0

            if quantity <= 0:
                continue
            notional = quantity * price
            fee = notional * FEE_RATE
            cost = notional + fee
            if cost > cash:
                if user_qty > 0:
                    print(f"  [!] {ticker}: Efectivo insuficiente para comprar {quantity} acciones a ${price:.2f} (necesitas ${cost:.2f}, tienes ${cash:.2f})")
                    continue
                quantity = int((cash - fee) / price) if price > 0 else 0
                if quantity <= 0:
                    continue
                notional = quantity * price
                fee = notional * FEE_RATE
                cost = notional + fee
            cash -= cost
            total_trades += 1
            user_tag = ' (cantidad usuario)' if user_qty > 0 else ''
            new_positions[ticker] = {
                'cantidad': quantity,
                'precio_compra': round(price, 2),
                'precio_actual': round(price, 2),
                'valor': round(notional, 2),
                'pnl': 0.0,
                'pct': 0.0,
                'fecha_compra': today_str,
                'cantidad_usuario': user_qty > 0
            }
            trade_record = {
                'ticker': ticker,
                'tipo': 'compra',
                'fecha': today_str,
                'precio': round(price, 2),
                'cantidad': quantity,
                'comision': round(fee, 2)
            }
            trades.append(trade_record)
            print(f"  COMPRADO {ticker}: {quantity} @ ${price:.2f} (fee: ${fee:.2f}){user_tag}")

    portfolio_value = 0.0
    for ticker, pos in new_positions.items():
        current_price = hedged_prices.get(ticker, pos.get('precio_actual', pos['precio_compra']))
        pnl = (current_price - pos['precio_compra']) * pos['cantidad']
        valor = current_price * pos['cantidad']
        pct = ((current_price / pos['precio_compra']) - 1) * 100 if pos['precio_compra'] > 0 else 0
        pos['precio_actual'] = round(current_price, 2)
        pos['valor'] = round(valor, 2)
        pos['pnl'] = round(pnl, 2)
        pos['pct'] = round(pct, 2)
        portfolio_value += valor

    total_value = cash + portfolio_value
    total_return = (total_value - INITIAL_CAPITAL) / INITIAL_CAPITAL

    equity_curve.append(total_value)
    if len(equity_curve) > 500:
        equity_curve = equity_curve[-500:]
    max_dd = compute_max_drawdown(equity_curve)

    win_rate = winners / total_trades if total_trades > 0 else 0

    state.update({
        'timestamp': now_str,
        'efectivo': round(cash, 2),
        'valor_portafolio': round(portfolio_value, 2),
        'valor_total': round(total_value, 2),
        'retorno_total': round(total_return, 4),
        'max_drawdown': round(max_dd, 4),
        'win_rate': round(win_rate, 4),
        'total_trades': total_trades,
        'trades_ganadores': winners,
        'trades_perdedores': losers,
        'posiciones': new_positions,
        'historial_trades': trades,
        'equity_curve': equity_curve
    })

    save_json(PAPER_FILE, state)

    print(f"  Paper Trading: ${total_value:,.0f} ({total_return:+.0%}) | {total_trades} trades | Win rate {win_rate:.0%} | Drawdown {max_dd:.0%}")


if __name__ == '__main__':
    run()
