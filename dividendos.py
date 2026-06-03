import json
import os
import sys
import time
import datetime

TICKERS_CORE = ['NVDA','MU','DELL','AVGO','DDOG','SMCI','SNOW','CRWD','NOW','TSM','ARM','OKTA','HPE','NTAP','CLS','AAPL','AMZN','GOOGL','META','MSFT','LLY','AMAT','LRCX','PANW','ORCL','HON','UBER','GE','COST','NEE']

DATA_DIR = os.environ.get('GITHUB_WORKSPACE', '.')
OUTPUT = os.path.join(DATA_DIR, 'Datos', 'dividendos.json')
os.makedirs(os.path.join(DATA_DIR, 'Datos'), exist_ok=True)

def cargar_portafolio():
    try:
        ruta = os.path.join(DATA_DIR, 'Datos', 'portafolio_usuario.json')
        with open(ruta, 'r') as f:
            return json.load(f)
    except:
        return []

def tickers_a_procesar():
    portafolio = cargar_portafolio()
    combinados = list(TICKERS_CORE)
    for t in portafolio:
        t = t.strip().upper()
        if t and t not in combinados:
            combinados.append(t)
    return combinados[:30]

def determinar_frecuencia(dividends):
    if len(dividends) < 2:
        return 'unknown'
    fechas = dividends.index.tolist()
    intervalos = []
    for i in range(1, min(len(fechas), 13)):
        diff = (fechas[-i] - fechas[-i-1]).days
        if 0 < diff < 365:
            intervalos.append(diff)
    if not intervalos:
        return 'unknown'
    promedio = sum(intervalos) / len(intervalos)
    if promedio < 35:
        return 'monthly'
    elif promedio < 55:
        return 'bimonthly'
    elif promedio < 100:
        return 'quarterly'
    elif promedio < 200:
        return 'semi-annual'
    else:
        return 'annual'

def procesar_yfinance(ticker):
    try:
        import yfinance as yf
        yt = yf.Ticker(ticker)
        dividends = yt.dividends

        if dividends is None or dividends.empty:
            return None

        ultimos = dividends.tail(20)
        if ultimos.empty:
            return None

        hoy = datetime.datetime.now()
        ultimo_pago = ultimos.iloc[-1]
        ultima_fecha = ultimos.index[-1]

        precio_actual = None
        try:
            hist = yt.history(period='5d')
            if not hist.empty:
                precio_actual = hist['Close'].iloc[-1]
        except:
            pass

        if precio_actual is None or precio_actual == 0:
            return None

        div_recientes = ultimos.tail(4)
        if div_recientes.empty:
            return None
        pago_anual = float(div_recientes.sum())

        yield_value = round(pago_anual / float(precio_actual), 4)

        frecuencia = determinar_frecuencia(ultimos)

        crecimiento_5yr = 0.0
        if len(ultimos) >= 10:
            try:
                fechas_5yr = ultimos.index[-1] - datetime.timedelta(days=365*5)
                hace_5 = ultimos[ultimos.index >= fechas_5yr]
                if len(hace_5) >= 4:
                    pago_actual = hace_5.tail(4).sum()
                    pago_anterior = hace_5.head(4).sum()
                    if pago_anterior > 0:
                        crecimiento_5yr = round((pago_actual - pago_anterior) / pago_anterior, 4)
            except:
                pass

        ex_date = ''
        pay_date = ''
        try:
            calendar = yt.calendar
            if calendar is not None and not calendar.empty:
                if 'Ex-Dividend Date' in calendar.index:
                    ex_date = str(calendar.loc['Ex-Dividend Date'].iloc[0])[:10]
                if 'Dividend Date' in calendar.index:
                    pay_date = str(calendar.loc['Dividend Date'].iloc[0])[:10]
        except:
            pass

        if not ex_date:
            ex_date = ultima_fecha.strftime('%Y-%m-%d')

        return {
            "yield": yield_value,
            "ex_date": ex_date,
            "pay_date": pay_date,
            "frecuencia": frecuencia,
            "anual_por_accion": round(pago_anual, 4),
            "crecimiento_5yr": crecimiento_5yr
        }
    except:
        return None

def main():
    tickers = tickers_a_procesar()
    print(f"[!] Procesando dividendos para {len(tickers)} tickers...")

    resultados = {}
    for t in tickers:
        try:
            print(f"  {t}...", end=' ')
            data = procesar_yfinance(t)
            if data is None:
                print(f"[!] sin dividendos o error")
            else:
                print(f"[OK] yield: {data['yield']*100:.2f}%, freq: {data['frecuencia']}")
                resultados[t] = data
            time.sleep(0.3)
        except Exception as e:
            print(f"[!] error: {e}")

    salida = {
        "timestamp": datetime.datetime.now().isoformat(),
        "tickers": resultados
    }

    with open(OUTPUT, 'w', encoding='utf-8') as f:
        json.dump(salida, f, ensure_ascii=False, indent=2)

    print(f"\n[OK] Dividendos guardados en {OUTPUT}")
    for t, d in sorted(resultados.items(), key=lambda x: x[1].get('yield', 0), reverse=True):
        print(f"  {t}: yield {d['yield']*100:.2f}% | anual ${d['anual_por_accion']:.2f} | {d['frecuencia']}")

if __name__ == '__main__':
    main()
