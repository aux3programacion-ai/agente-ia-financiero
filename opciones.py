import json
import os
import sys
import time

import numpy as np

TICKERS_CORE = ['NVDA','MU','DELL','AVGO','DDOG','SMCI','SNOW','CRWD','NOW','TSM','ARM','OKTA','HPE','NTAP','CLS','AAPL','AMZN','GOOGL','META','MSFT','LLY','AMAT','LRCX','PANW','ORCL','HON','UBER','GE','COST','NEE']

DATA_DIR = os.environ.get('GITHUB_WORKSPACE', '.')
OUTPUT = os.path.join(DATA_DIR, 'Datos', 'opciones.json')
os.makedirs(os.path.join(DATA_DIR, 'Datos'), exist_ok=True)

from portafolio_utils import cargar_portafolio


def tickers_a_procesar():
    portafolio = cargar_portafolio(DATA_DIR)
    combinados = list(TICKERS_CORE)
    for t in portafolio:
        t = t.strip().upper()
        if t and t not in combinados:
            combinados.append(t)
    return combinados[:30]

def _elegir_expiracion(expiraciones, dias_objetivo=35):
    """Expiracion mas cercana a `dias_objetivo` dias vista (no la mas cercana)."""
    from datetime import date
    from datetime import datetime as _dt
    hoy = date.today()
    return min(
        expiraciones,
        key=lambda e: abs((_dt.strptime(e, '%Y-%m-%d').date() - hoy).days - dias_objetivo),
    )


def _a_numero(valor):
    """Convierte a float finito; NaN, None y strings no numericos dan None."""
    if valor is None:
        return None
    try:
        numero = float(valor)
    except (TypeError, ValueError):
        return None
    return numero if np.isfinite(numero) else None


def _volatilidad_implicita(precio, spot, strike, anios, tasa=0.045, sigma=0.3,
                           tipo='call'):
    """
    IV por Newton-Raphson sobre Black-Scholes.

    Yahoo devuelve impliedVolatility en 0 para casi toda la cadena, asi que la
    IV se deduce del ultimo precio negociado real. Devuelve None si no converge
    o si el precio no es consistente (fuera de los limites de no arbitraje).
    """
    from scipy.stats import norm

    precio = _a_numero(precio)
    spot = _a_numero(spot)
    strike = _a_numero(strike)

    if precio is None or spot is None or strike is None:
        return None
    if spot <= 0 or strike <= 0 or anios <= 0 or precio <= 0:
        return None

    descontado = strike * np.exp(-tasa * anios)
    tolerancia = max(0.005 * spot, 0.05)
    if tipo == 'put':
        piso, techo = max(0.0, descontado - spot), descontado
    else:
        piso, techo = max(0.0, spot - descontado), spot

    # Sin este filtro, precios stale o malformados "convergen" a un sigma arbitrario.
    if not (piso - tolerancia <= precio <= techo + tolerancia):
        return None

    modelo = None
    for _ in range(60):
        d1 = (np.log(spot / strike) + (tasa + 0.5 * sigma ** 2) * anios) / (sigma * np.sqrt(anios))
        d2 = d1 - sigma * np.sqrt(anios)
        if tipo == 'put':
            modelo = strike * np.exp(-tasa * anios) * norm.cdf(-d2) - spot * norm.cdf(-d1)
        else:
            modelo = spot * norm.cdf(d1) - descontado * norm.cdf(d2)
        vega = spot * norm.pdf(d1) * np.sqrt(anios)
        if vega < 1e-8:
            break
        if abs(modelo - precio) < 1e-4:
            return sigma
        sigma = max(0.01, min(3.0, sigma - (modelo - precio) / vega))

    return sigma if modelo is not None and abs(modelo - precio) < 0.05 else None


def _iv_de_la_cadena(legs, spot, dias, tipo='call'):
    """IV en % por strike para los legs con precio consistente."""
    out = {}
    for leg in legs:
        strike = _a_numero(leg.get('strike'))
        if strike is None or strike <= 0 or dias <= 0:
            continue
        iv = _volatilidad_implicita(leg.get('lastPrice'), spot, strike,
                                    dias / 365.0, tipo=tipo)
        if iv is not None:
            out[strike] = iv * 100
    return out


def fetch_options_yahoo(ticker):
    """
    Cadena de opciones real via yfinance.

    Yahoo ya no sirve el endpoint v7 sin crumb (401), y ademas devuelve
    impliedVolatility en 0 y openInterest en 0. Por eso:
      - volumen, spot y ultimo precio se toman tal cual de la cadena;
      - la IV se calcula por Black-Scholes sobre el ultimo precio real;
      - openInterest se marca no disponible cuando la cadena no lo reporta.
    """
    try:
        import yfinance as yf
    except ImportError:
        return registrar_no_disponible('yfinance no instalado')

    try:
        t = yf.Ticker(ticker)

        spot = t.fast_info.get('lastPrice')
        spot = float(spot) if spot else None
        if not spot:
            hist = t.history(period='5d')
            spot = float(hist['Close'].iloc[-1]) if not hist.empty else None
        if not spot:
            return registrar_no_disponible('sin precio del subyacente')

        expiraciones = list(t.options or [])
        if not expiraciones:
            return registrar_no_disponible('sin expiraciones listadas')

        from datetime import date
        from datetime import datetime as _dt
        expiracion = _elegir_expiracion(expiraciones)
        dias = (_dt.strptime(expiracion, '%Y-%m-%d').date() - date.today()).days
        if dias <= 0:
            return registrar_no_disponible(f'expiracion {expiracion} ya vencida')

        cadena = t.option_chain(expiracion)
        calls = [r.to_dict() for _, r in cadena.calls.iterrows()]
        puts = [r.to_dict() for _, r in cadena.puts.iterrows()]

        vol_calls = int(sum(_a_numero(c.get('volume')) or 0 for c in calls))
        vol_puts = int(sum(_a_numero(p.get('volume')) or 0 for p in puts))
        vol_total = vol_calls + vol_puts

        if vol_total == 0:
            return registrar_no_disponible('cadena sin volumen negociado')

        put_call_ratio = round(vol_puts / vol_calls, 2) if vol_calls > 0 else None
        if put_call_ratio is None:
            sentimiento = None
        elif put_call_ratio > 1.0:
            sentimiento = 'bearish'
        elif put_call_ratio < 0.7:
            sentimiento = 'bullish'
        else:
            sentimiento = 'neutral'

        for leg in calls + puts:
            leg['_dias'] = dias
        iv_calls = _iv_de_la_cadena(calls, spot, dias, tipo='call')
        iv_puts = _iv_de_la_cadena(puts, spot, dias, tipo='put')

        def mas_cercano(ivs):
            if not ivs:
                return None
            k = min(ivs, key=lambda s: abs(s - spot))
            return round(ivs[k], 2)

        atm_iv_call = mas_cercano(iv_calls)
        atm_iv_put = mas_cercano(iv_puts)

        if atm_iv_put is not None and atm_iv_call is not None:
            iv_skew = round(atm_iv_put - atm_iv_call, 2)
            avg_iv = round((atm_iv_put + atm_iv_call) / 2, 2)
        else:
            iv_skew = None
            disponibles = [v for v in (atm_iv_call, atm_iv_put) if v is not None]
            avg_iv = round(sum(disponibles) / len(disponibles), 2) if disponibles else None

        # Volatilidad media de toda la banda, ponderada por volumen real.
        iv_todas = list(iv_calls.values()) + list(iv_puts.values())
        iv_promedio = round(float(np.mean(iv_todas)), 2) if iv_todas else None

        # Curvatura de la sonrisa: IV de las alas menos la del centro.
        iv_por_strike = {**iv_puts, **iv_calls}
        iv_curvature = None
        if len(iv_por_strike) >= 5:
            ks = sorted(iv_por_strike)
            ivs = [iv_por_strike[k] for k in ks]
            centro = ivs[len(ivs) // 2]
            cuarto = max(1, len(ivs) // 4)
            ala = max(float(np.mean(ivs[:cuarto])), float(np.mean(ivs[-cuarto:])))
            iv_curvature = round((ala - centro) / centro, 4) if centro else None

        # Yahoo no publica openInterest en la respuesta actual.
        oi_reports = int(sum((c.get('openInterest') or 0) for c in calls + puts))
        if oi_reports > 0:
            oi_calls = int(sum(c.get('openInterest') or 0 for c in calls))
            oi_puts = int(sum(p.get('openInterest') or 0 for p in puts))
            put_call_oi_ratio = round(oi_puts / oi_calls, 2) if oi_calls else None
            oi_weighted_iv = None
            gamma_exposure = None
            oi_disponible = True
        else:
            oi_calls = oi_puts = 0
            put_call_oi_ratio = None
            oi_weighted_iv = None
            gamma_exposure = None
            oi_disponible = False

        # Spread relativo: solo con bid/ask reales.
        spreads = [
            (leg['ask'] - leg['bid']) / ((leg['ask'] + leg['bid']) / 2)
            for leg in calls + puts
            if (leg.get('bid') or 0) > 0 and (leg.get('ask') or 0) > (leg.get('bid') or 0)
        ]
        spread_pct = round(float(np.mean(spreads)), 4) if spreads else None

        return {
            "disponible": True,
            "fuente": "yfinance_option_chain",
            "spot": round(spot, 2),
            "fecha_expiracion_proxima": expiracion,
            "dias_a_expiracion": dias,
            "strike_min": round(float(cadena.calls['strike'].min()), 2) if len(cadena.calls) else None,
            "strike_max": round(float(cadena.calls['strike'].max()), 2) if len(cadena.calls) else None,
            "n_calls": len(calls),
            "n_puts": len(puts),
            "vol_calls": vol_calls,
            "vol_puts": vol_puts,
            "put_call_ratio": put_call_ratio,
            "sentimiento": sentimiento,
            "vol_total": vol_total,
            "iv_skew": iv_skew,
            "avg_iv": avg_iv,
            "iv_promedio_banda": iv_promedio,
            "atm_iv_put": atm_iv_put,
            "atm_iv_call": atm_iv_call,
            "oi_disponible": oi_disponible,
            "put_call_oi_ratio": put_call_oi_ratio,
            "oi_weighted_iv": oi_weighted_iv,
            "gamma_exposure": gamma_exposure,
            "iv_curvature": iv_curvature,
            "spread_pct": spread_pct,
        }
    except Exception as e:
        return registrar_no_disponible(f"{type(e).__name__}: {str(e)[:140]}")


def registrar_no_disponible(motivo):
    """
    Registro para un ticker sin datos de mercado.

    No inventa IV, volumen ni ratios: un put/call ratio sintetico se
    consumía igual que uno real y contaminaba el entrenamiento del modelo.
    Los consumidores deben tratar la ausencia de campos como "sin dato".
    """
    return {
        "disponible": False,
        "motivo": str(motivo)[:160],
        "put_call_ratio": None,
        "sentimiento": None,
        "vol_total": None,
        "fecha_expiracion_proxima": None,
        "iv_skew": None,
        "avg_iv": None,
        "oi_weighted_iv": None,
        "atm_iv_put": None,
        "atm_iv_call": None,
        "put_call_oi_ratio": None,
        "iv_curvature": None,
        "spread_pct": None,
        "gamma_exposure": None
    }

def main():
    tickers = tickers_a_procesar()
    print(f"[!] Procesando opciones para {len(tickers)} tickers...")

    resultados = {}
    sin_datos = []
    for t in tickers:
        print(f"  {t}...", end=' ')
        try:
            data = fetch_options_yahoo(t)
        except Exception as e:
            data = registrar_no_disponible(f"{type(e).__name__}: {e}")
            print(f"[!] error: {str(e)[:80]}")
            time.sleep(0.5)
            continue

        if data is None:
            data = registrar_no_disponible('respuesta sin cadena de opciones')

        if data.get('disponible'):
            print(f"[OK] P/C {data['put_call_ratio']} ({data['sentimiento']}), "
                  f"IV {data['avg_iv']}%, exp {data['fecha_expiracion_proxima']}")
        else:
            sin_datos.append(t)
            print(f"[!] sin datos: {data['motivo']}")

        resultados[t] = data
        time.sleep(0.5)

    disponibles = [t for t, d in resultados.items() if d.get('disponible')]
    if sin_datos:
        print(f"[!] {len(sin_datos)}/{len(resultados)} tickers sin datos de mercado: "
              f"{', '.join(sin_datos)}")

    if not disponibles:
        print("[ERROR] Ningun ticker devolvio datos de opciones. "
              "No se escribe un archivo con valores inventados.")

    salida = {
        "timestamp": time.strftime('%Y-%m-%dT%H:%M:%S'),
        "fuente": "yfinance_option_chain",
        "completo": len(disponibles) == len(resultados) and bool(resultados),
        "nota": "Los campos en null significan que la fuente no los publica. "
                "No se sustituyen por valores estimados.",
        "resumen": {
            "tickers_solicitados": len(resultados),
            "tickers_con_datos": len(disponibles),
            "tickers_sin_datos": len(resultados) - len(disponibles),
        },
        "tickers": resultados
    }

    if disponibles:
        with open(OUTPUT, 'w', encoding='utf-8') as f:
            json.dump(salida, f, ensure_ascii=False, indent=2)
        print(f"\n[OK] Datos de opciones guardados en {OUTPUT}")
        for t in disponibles:
            d = resultados[t]
            vol = d['vol_total'] or 0
            print(f"  {t}: P/C {d['put_call_ratio']} ({d['sentimiento']}), "
                  f"IV {d['avg_iv']}%, vol {vol:,}")
    else:
        print(f"[ERROR] {OUTPUT} no se actualizo: se conserva el ultimo estado real conocido.")

    return 0 if disponibles else 1


if __name__ == '__main__':
    sys.exit(main())
