import json, os

def cargar_portafolio(data_dir='.'):
    """
    Lee portafolio_usuario.json y retorna lista de tickers (strings).
    Acepta ambos formatos:
      - Antiguo: ["AMZN", "NVDA"]
      - Nuevo:   [{"ticker": "AMZN", "cantidad": 10}, {"ticker": "NVDA"}]
    """
    ruta = os.path.join(data_dir, 'Datos', 'portafolio_usuario.json')
    tickers = []
    if os.path.exists(ruta):
        try:
            data = json.load(open(ruta, 'r'))
            if isinstance(data, list):
                for item in data:
                    if isinstance(item, str):
                        t = item.strip().upper()
                        if t:
                            tickers.append(t)
                    elif isinstance(item, dict):
                        t = item.get('ticker', '').strip().upper()
                        if t:
                            tickers.append(t)
        except Exception:
            pass
    return tickers


def cargar_portafolio_cantidades(data_dir='.'):
    """
    Lee portafolio_usuario.json y retorna dict {ticker: cantidad}.
    cantidad es 0 si no se especifico.
    Acepta ambos formatos:
      - Antiguo: ["AMZN"] -> {"AMZN": 0}
      - Nuevo:   [{"ticker": "AMZN", "cantidad": 10}] -> {"AMZN": 10}
    """
    ruta = os.path.join(data_dir, 'Datos', 'portafolio_usuario.json')
    resultado = {}
    if os.path.exists(ruta):
        try:
            data = json.load(open(ruta, 'r'))
            if isinstance(data, list):
                for item in data:
                    if isinstance(item, str):
                        t = item.strip().upper()
                        if t:
                            resultado[t] = 0
                    elif isinstance(item, dict):
                        t = item.get('ticker', '').strip().upper()
                        c = item.get('cantidad', 0)
                        if t:
                            try:
                                resultado[t] = max(0, int(c))
                            except (ValueError, TypeError):
                                resultado[t] = 0
        except Exception:
            pass
    return resultado
