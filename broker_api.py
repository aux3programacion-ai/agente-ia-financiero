import json
import os
import sys
import time
import urllib.request
import base64

DATA_DIR = os.environ.get('GITHUB_WORKSPACE', '.')
STATUS_FILE = os.path.join(DATA_DIR, 'Datos', 'broker_status.json')
PAPER_FILE = os.path.join(DATA_DIR, 'Datos', 'paper_trading.json')


def load_json(path, default=None):
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception:
        return default


def save_json(path, data):
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    except Exception as e:
        print(f"  Error guardando {path}: {e}")


class BrokerAPI:
    def get_positions(self):
        raise NotImplementedError

    def get_account_info(self):
        raise NotImplementedError

    def place_order(self, ticker, quantity, order_type, side):
        raise NotImplementedError

    def get_order_status(self, order_id):
        raise NotImplementedError

    def get_market_hours(self):
        raise NotImplementedError


class SimulatedBroker(BrokerAPI):
    def __init__(self):
        self.name = "simulado"
        self.connected = True

    def get_positions(self):
        state = load_json(PAPER_FILE, {})
        pos = state.get('posiciones', {})
        result = []
        for ticker, p in pos.items():
            result.append({
                'ticker': ticker,
                'cantidad': p.get('cantidad', 0),
                'precio_actual': p.get('precio_actual', 0),
                'valor': p.get('valor', 0),
                'pnl': p.get('pnl', 0),
                'pct': p.get('pct', 0)
            })
        return result

    def get_account_info(self):
        state = load_json(PAPER_FILE, {})
        return {
            'broker': 'simulado',
            'efectivo': state.get('efectivo', 0),
            'valor_portafolio': state.get('valor_portafolio', 0),
            'valor_total': state.get('valor_total', 0),
            'retorno_total': state.get('retorno_total', 0),
            'total_trades': state.get('total_trades', 0),
            'win_rate': state.get('win_rate', 0),
            'max_drawdown': state.get('max_drawdown', 0)
        }

    def place_order(self, ticker, quantity, order_type, side):
        return {
            'ticker': ticker,
            'cantidad': quantity,
            'tipo': order_type,
            'lado': side,
            'status': 'simulado',
            'mensaje': 'Orden simulada ejecutada en paper_trading.json'
        }

    def get_order_status(self, order_id):
        return {'order_id': order_id, 'status': 'filled', 'filled_qty': 0, 'mensaje': 'Simulado'}

    def get_market_hours(self):
        return {'market': 'simulado', 'is_open': True, 'next_open': '', 'next_close': ''}


class AlpacaBroker(BrokerAPI):
    def __init__(self):
        self.api_key = os.environ.get('ALPACA_KEY', '')
        self.api_secret = os.environ.get('ALPACA_SECRET', '')
        self.base_url = 'https://paper-api.alpaca.markets'
        self.name = 'alpaca'
        self.connected = False
        if self.api_key and self.api_secret:
            self.connected = True

    def _headers(self):
        auth_str = f"{self.api_key}:{self.api_secret}"
        encoded = base64.b64encode(auth_str.encode()).decode()
        return {
            'Authorization': f'Basic {encoded}',
            'Content-Type': 'application/json'
        }

    def _request(self, method, path, data=None):
        url = f"{self.base_url}{path}"
        req = urllib.request.Request(url, data=data, headers=self._headers(), method=method)
        try:
            with urllib.request.urlopen(req, timeout=15) as resp:
                return json.loads(resp.read().decode())
        except urllib.error.HTTPError as e:
            body = e.read().decode() if e.fp else ''
            return {'error': str(e), 'body': body}
        except Exception as e:
            return {'error': str(e)}

    def get_positions(self):
        if not self.connected:
            raise NotImplementedError("Alpaca integration requires API key")
        return self._request('GET', '/v2/positions')

    def get_account_info(self):
        if not self.connected:
            raise NotImplementedError("Alpaca integration requires API key")
        result = self._request('GET', '/v2/account')
        if 'error' in result:
            return {'error': result['error'], 'conectado': False}
        return {
            'broker': 'alpaca',
            'efectivo': float(result.get('cash', 0)),
            'buying_power': float(result.get('buying_power', 0)),
            'portfolio_value': float(result.get('portfolio_value', 0)),
            'status': result.get('status', ''),
            'conectado': True
        }

    def place_order(self, ticker, quantity, order_type, side):
        if not self.connected:
            raise NotImplementedError("Alpaca integration requires API key")
        body = json.dumps({
            'symbol': ticker,
            'qty': str(quantity),
            'type': order_type,
            'side': side,
            'time_in_force': 'day'
        }).encode()
        return self._request('POST', '/v2/orders', data=body)

    def get_order_status(self, order_id):
        if not self.connected:
            raise NotImplementedError("Alpaca integration requires API key")
        return self._request('GET', f'/v2/orders/{order_id}')

    def get_market_hours(self):
        if not self.connected:
            raise NotImplementedError("Alpaca integration requires API key")
        return self._request('GET', '/v2/clock')


def get_broker():
    alpaca_key = os.environ.get('ALPACA_KEY', '')
    alpaca_secret = os.environ.get('ALPACA_SECRET', '')
    if alpaca_key and alpaca_secret:
        return AlpacaBroker()
    return SimulatedBroker()


def run():
    print("=== Broker API Interface ===")

    broker = get_broker()
    now_str = time.strftime('%Y-%m-%d %H:%M:%S')

    if isinstance(broker, AlpacaBroker):
        print("  Broker: Alpaca (real)")
        acct = broker.get_account_info()
        if 'error' in acct:
            print(f"  Error conectando a Alpaca: {acct.get('error')}")
            status = {
                'timestamp': now_str,
                'broker': 'alpaca',
                'conectado': False,
                'mensaje': f"Error: {acct.get('error')}",
                'modo': 'error'
            }
        else:
            print(f"  Conectado. Cash: ${acct.get('efectivo', 0):,.2f}")
            print(f"  Portfolio: ${acct.get('portfolio_value', 0):,.2f}")
            status = {
                'timestamp': now_str,
                'broker': 'alpaca',
                'conectado': True,
                'mensaje': 'Conectado a Alpaca API',
                'modo': 'real',
                'cuenta': acct
            }
    else:
        print("  Broker: Simulado (paper_trading.json)")
        info = broker.get_account_info()
        print(f"  Efectivo: ${info.get('efectivo', 0):,.2f}")
        print(f"  Valor total: ${info.get('valor_total', 0):,.2f}")
        print(f"  Retorno: {info.get('retorno_total', 0):+.2%}")
        print(f"  Trades: {info.get('total_trades', 0)}")
        status = {
            'timestamp': now_str,
            'broker': 'ninguno',
            'conectado': False,
            'mensaje': 'Broker API disponible. Configura ALPACA_KEY y ALPACA_SECRET para conexion real.',
            'modo': 'paper_trading'
        }

    save_json(STATUS_FILE, status)
    print(f"  Status guardado en {STATUS_FILE}")


if __name__ == '__main__':
    run()
