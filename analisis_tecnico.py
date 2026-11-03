#!/usr/bin/env python3
"""
analisis_tecnico.py - Calcula indicadores tecnicos + regimen de mercado
para inyectar contexto al AI. Fuente: yfinance (gratis, sin API key).
"""
import json
import os
from datetime import datetime, timezone

import pandas as pd

TICKERS = ['NVDA','MU','DELL','AVGO','DDOG','SMCI','SNOW','CRWD','NOW','TSM',
           'ARM','OKTA','HPE','NTAP','CLS','AAPL','AMZN','GOOGL','META','MSFT',
           'LLY','AMAT','LRCX','PANW','ORCL','HON','UBER','GE','COST','NEE']

DATA_DIR = os.environ.get('GITHUB_WORKSPACE', '.')
OUTPUT = os.path.join(DATA_DIR, 'Datos', 'analisis_tecnico.json')
os.makedirs(os.path.join(DATA_DIR, 'Datos'), exist_ok=True)

def _ema(valores, periodo):
    """
    Media exponencial con alpha = 2/(periodo+1) y semilla en el primer valor.

    Misma convencion que pandas `ewm(span=periodo, adjust=False)`, para que
    este modulo y modelo_xgboost devuelvan el mismo numero.
    """
    if not valores:
        return []
    alpha = 2.0 / (periodo + 1)
    ema = [float(valores[0])]
    for v in valores[1:]:
        ema.append(float(v) * alpha + ema[-1] * (1 - alpha))
    return ema


def calc_rsi(precios, periodo=14):
    """
    RSI de Wilder, acotado a [0, 100].

    Antes usaba medias simples de los ultimos N cambios, que no es RSI.
    """
    if len(precios) < periodo + 1:
        return 50

    difs = [precios[i] - precios[i - 1] for i in range(1, len(precios))]
    media_gan = max(difs[0], 0.0)
    media_per = max(-difs[0], 0.0)

    for d in difs[1:]:
        media_gan = (media_gan * (periodo - 1) + max(d, 0.0)) / periodo
        media_per = (media_per * (periodo - 1) + max(-d, 0.0)) / periodo

    if media_per == 0:
        return 100 if media_gan > 0 else 50
    rs = media_gan / media_per
    return round(100 - (100 / (1 + rs)), 1)


def calc_macd(precios, rapida=12, lenta=26, senal=9):
    """
    MACD estandar: EMA(rapida) - EMA(lenta), senal EMA del MACD e histograma.

    Antes la senal era `macd * 0.3` y las "EMA" eran medias simples, lo que no
    es un MACD y distortionaba el regimen de todos los tickers aguas abajo.
    """
    if len(precios) < lenta:
        return {'macd': 0, 'senal': 0, 'histograma': 0}

    # Ambas EMA se calculan sobre la serie completa y luego se alinean por fecha.
    ema_rapida = _ema(precios, rapida)
    ema_lenta = _ema(precios, lenta)

    macd = [c - l for c, l in zip(ema_rapida[-lenta:], ema_lenta[-lenta:])]
    linea_senal = _ema(macd, senal)

    return {
        'macd': round(macd[-1], 2),
        'senal': round(linea_senal[-1], 2),
        'histograma': round(macd[-1] - linea_senal[-1], 2),
    }

def calc_atr(precios_cierre, highs, lows, periodo=14):
    if len(precios_cierre) < periodo:
        return 0
    trs = []
    for i in range(-periodo, 0):
        hl = highs[i] - lows[i]
        hc = abs(highs[i] - precios_cierre[i-1]) if i > -len(precios_cierre) else 0
        lc = abs(lows[i] - precios_cierre[i-1]) if i > -len(precios_cierre) else 0
        trs.append(max(hl, hc, lc))
    return round(sum(trs) / periodo, 2)

def obtener_tecnicos():
    try:
        import yfinance as yf
    except ImportError:
        print('[!] yfinance no instalado')
        return None

    resultado = {'timestamp': datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ'),
                 'regimen_mercado': 'desconocido', 'tecnicos': {}}

    # Fetch mercado primero (SPY como proxy del S&P 500)
    try:
        spy = yf.download('SPY', period='1y', interval='1d', progress=False)
        # yfinance devuelve MultiIndex (campo, ticker) incluso para un solo simbolo.
        if isinstance(spy.columns, pd.MultiIndex):
            spy = spy.xs('Close', axis=1, level=0)
            # Tras el xs queda una sola columna (el ticker), no 'Close'.
            spy_close = spy.iloc[:, 0].to_numpy()
        else:
            spy_close = spy['Close'].to_numpy()
        if spy is not None and not spy.empty and len(spy) > 50:
            spy_ma50_val = round(float(spy_close[-50:].mean()), 2)
            spy_ma200_val = round(float(spy_close[-200:].mean()) if len(spy_close) >= 200
                                  else spy_ma50_val, 2)
            spy_actual = round(float(spy_close[-1]), 2)
            spy_ma50_trend = round((spy_close[-1] - spy_ma50_val) / spy_ma50_val * 100, 1) if spy_ma50_val else 0

            if spy_ma50_val > spy_ma200_val:
                if spy_ma50_trend > 3: regimen = 'alcista-fuerte'
                elif spy_ma50_trend > 1: regimen = 'alcista'
                elif spy_ma50_trend > -1: regimen = 'alcista-debil'
                else: regimen = 'alcista'
            elif spy_ma50_val < spy_ma200_val:
                if spy_ma50_trend < -3: regimen = 'bajista-fuerte'
                elif spy_ma50_trend < -1: regimen = 'bajista'
                else: regimen = 'bajista-debil'
            else:
                regimen = 'lateral'

            resultado['regimen_mercado'] = regimen
            resultado['spy'] = {
                'precio': spy_actual,
                'ma50': spy_ma50_val,
                'ma200': spy_ma200_val,
                'tendencia_pct': spy_ma50_trend,
                'descripcion': f'S&P 500 (SPY) en regimen {regimen.upper()}. '
                               f'Precio: ${spy_actual}, MA50: ${spy_ma50_val}, MA200: ${spy_ma200_val}'
            }
            print(f'[Tecnico] Regimen mercado: {regimen} (SPY ${spy_actual})')
    except Exception as e:
        print(f'[!] Error obteniendo SPY: {e}')
        resultado['regimen_mercado'] = 'desconocido'

    # Fetch datos tecnicos por ticker
    for t in TICKERS:
        try:
            df = yf.download(t, period='1y', interval='1d', progress=False)
            # yfinance devuelve MultiIndex (campo, ticker) incluso para un solo simbolo.
            if df is not None and isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            if df is None or df.empty or len(df) < 30:
                resultado['tecnicos'][t] = {'error': 'datos insuficientes'}
                continue

            close = df['Close'].values.flatten()
            high = df['High'].values.flatten()
            low = df['Low'].values.flatten()
            volume = df['Volume'].values.flatten()
            actual = round(float(close[-1]), 2)
            close_list = [float(x) for x in close]
            high_list = [float(x) for x in high]
            low_list = [float(x) for x in low]

            # MA50, MA200
            ma50 = round(float(close[-50:].mean()) if len(close) >= 50 else actual, 2)
            ma200 = round(float(close[-200:].mean()) if len(close) >= 200 else actual, 2)
            precio_ma50 = round((actual - ma50) / ma50 * 100, 1) if ma50 else 0
            precio_ma200 = round((actual - ma200) / ma200 * 100, 1) if ma200 else 0

            # Tendencia
            if actual > ma50 > ma200: tendencia = 'uptrend'
            elif actual < ma50 < ma200: tendencia = 'downtrend'
            elif actual > ma50: tendencia = 'ligera-alcista'
            elif actual < ma50: tendencia = 'ligera-bajista'
            else: tendencia = 'lateral'

            # RSI
            rsi = calc_rsi(close_list, 14)
            senial_rsi = 'sobrecompra' if rsi > 70 else 'sobreventa' if rsi < 30 else 'neutral'

            # MACD
            macd = calc_macd(close_list)
            senial_macd = 'alcista' if macd['histograma'] > 0 else 'bajista' if macd['histograma'] < 0 else 'neutral'

            # ATR%
            atr = calc_atr(close_list, high_list, low_list)
            atr_pct = round(atr / actual * 100, 2) if actual else 0

            # Volumen relativo
            vol_actual = float(volume[-5:].mean()) if len(volume) >= 5 else float(volume[-1])
            vol_hist = float(volume[-21:].mean()) if len(volume) >= 21 else vol_actual
            vol_ratio = round(vol_actual / vol_hist, 2) if vol_hist else 1

            # Soportes y resistencias simples
            soporte20 = round(float(close[-20:].min()), 2) if len(close) >= 20 else actual * 0.95
            resistencia20 = round(float(close[-20:].max()), 2) if len(close) >= 20 else actual * 1.05
            soporte50 = round(float(close[-50:].min()), 2) if len(close) >= 50 else actual * 0.9
            distancia_soporte = round((actual - soporte20) / actual * 100, 1)
            distancia_resistencia = round((resistencia20 - actual) / actual * 100, 1)

            resultado['tecnicos'][t] = {
                'precio': actual,
                'ma50': ma50,
                'ma200': ma200,
                'pct_sobre_ma50': precio_ma50,
                'pct_sobre_ma200': precio_ma200,
                'tendencia': tendencia,
                'rsi': rsi,
                'senial_rsi': senial_rsi,
                'macd': macd['histograma'],
                'senial_macd': senial_macd,
                'atr_pct': atr_pct,
                'vol_ratio': vol_ratio,
                'soporte_20d': soporte20,
                'resistencia_20d': resistencia20,
                'dist_soporte_pct': distancia_soporte,
                'dist_resistencia_pct': distancia_resistencia
            }
        except Exception as e:
            resultado['tecnicos'][t] = {'error': str(e)[:60]}

    # Resumen general
    uptrends = sum(1 for t in TICKERS if resultado['tecnicos'].get(t, {}).get('tendencia') == 'uptrend')
    downtrends = sum(1 for t in TICKERS if resultado['tecnicos'].get(t, {}).get('tendencia') == 'downtrend')
    resultado['resumen'] = {
        'uptrend_count': uptrends,
        'downtrend_count': downtrends,
        'sobrecompra_count': sum(1 for t in TICKERS if resultado['tecnicos'].get(t, {}).get('senial_rsi') == 'sobrecompra'),
        'sobreventa_count': sum(1 for t in TICKERS if resultado['tecnicos'].get(t, {}).get('senial_rsi') == 'sobreventa'),
        'macd_alcista_count': sum(1 for t in TICKERS if resultado['tecnicos'].get(t, {}).get('senial_macd') == 'alcista'),
        'macd_bajista_count': sum(1 for t in TICKERS if resultado['tecnicos'].get(t, {}).get('senial_macd') == 'bajista')
    }

    with open(OUTPUT, 'w') as f:
        json.dump(resultado, f, indent=2, ensure_ascii=False)
    print(f'[OK] Tecnicos: {len(resultado["tecnicos"])} tickers | Regimen: {resultado["regimen_mercado"]} | '
          f'Uptrends: {uptrends}/{len(TICKERS)} | MACD+: {resultado["resumen"]["macd_alcista_count"]}')
    return resultado

if __name__ == '__main__':
    obtener_tecnicos()
