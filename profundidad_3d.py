#!/usr/bin/env python3
"""
profundidad_3d.py - Genera el JSON de liquidez reconstruida para el dashboard 3D.

Reconstruye un "order book" modelado para acciones US a partir de datos publicos
(yfinance intradia), igual que hace UltraDepth para NASDAQ:
  - velas 5m del ultimo dia y medio (OHLCV)
  - perfil de volumen por nivel de precio -> POC, VAH, VAL, VWAP
  - muros BID/ASK por nivel (liquidez estatica reconstruida)
  - ballenas (clusters de volumen anormalmente grande)
  - footprint: delta comprador/vendedor por nivel
  - zonas S/R por toques

Salida: Reportes/profundidad_3d_data.json  (+ HTML que lo lee via fetch)
Uso:    python profundidad_3d.py [TICKER]
"""
import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

REPO = Path(__file__).parent
REPORTES = REPO / 'Reportes'
REPORTES.mkdir(exist_ok=True)

N_NIVELES = 60          # niveles de precio de la pared
N_VELAS = 78            # ~1 dia y medio de velas 5m en horario US
TAM_TICK = 60           # ultimo tick tape
N_ZONAS = 6             # zonas S/R por toques


def _descargar(ticker):
    """Descarga velas 5m recientes; si falla, aplica fallback sintetico."""
    import yfinance as yf
    h = yf.Ticker(ticker).history(period='2d', interval='5m')
    if h.empty or len(h) < 10:
        raise RuntimeError(f'sin datos intradia para {ticker}')
    h = h.tail(N_VELAS)
    return h


def _fallback(ticker):
    """Velas sinteticas en caso de rate-limit: la escena 3D siempre renderiza."""
    rng = np.random.default_rng(abs(hash(ticker)) % 2**31)
    n = N_VELAS
    base = 100 + (abs(hash(ticker)) % 300)
    precio = np.full(n, base + rng.uniform(-2, 2))
    drift = rng.normal(0, base * 0.0007, n).cumsum()
    precio += drift
    cierre = precio
    apertura = np.roll(cierre, 1) + rng.normal(0, base * 0.0004, n)
    apertura[0] = precio[0] - rng.uniform(0, 0.3)
    spread = np.abs(rng.normal(base * 0.004, base * 0.001, n))
    alto = np.maximum(apertura, cierre) + spread * rng.uniform(0.3, 1.0, n)
    bajo = np.minimum(apertura, cierre) - spread * rng.uniform(0.3, 1.0, n)
    vol = np.abs(rng.normal(base * 4000, base * 900, n))
    import pandas as pd
    ts = pd.date_range(end=datetime.now(), periods=n, freq='5min')
    h = pd.DataFrame({'Open': apertura, 'High': alto, 'Low': bajo,
                      'Close': cierre, 'Volume': vol}, index=ts)
    h.attrs['fallback'] = True
    return h


def _tick(v0):
    """Redondeo a tick de 1 centavo."""
    return round(v0 * 100) / 100.0


def _perfil(h, precio_min, precio_max):
    """Perfil de volumen por nivel + POC/VAH/VAL/VWAP."""
    paso = (precio_max - precio_min) / N_NIVELES
    cent = np.clip(((h['Close'] - precio_min) / paso).astype(int), 0, N_NIVELES - 1)
    vol_nivel = np.zeros(N_NIVELES)
    for i, v in zip(cent, h['Volume'].values):
        vol_nivel[i] += v

    total = vol_nivel.sum()
    poc_idx = int(vol_nivel.argmax()) if total > 0 else N_NIVELES // 2

    # value area 70%: expandir alrededor del POC hasta cubrir 70% del volumen
    va = {poc_idx}
    acc = vol_nivel[poc_idx]
    lo, hi = poc_idx, poc_idx
    while acc < 0.70 * total and (lo > 0 or hi < N_NIVELES - 1):
        abajo = vol_nivel[lo - 1] if lo > 0 else -1
        arriba = vol_nivel[hi + 1] if hi < N_NIVELES - 1 else -1
        if abajo >= arriba:
            lo -= 1
            acc += max(abajo, 0)
            va.add(lo)
        else:
            hi += 1
            acc += max(arriba, 0)
            va.add(hi)

    vwap = float((h['Close'] * h['Volume']).sum() / max(h['Volume'].sum(), 1))
    return {
        'vol_nivel': vol_nivel.tolist(),
        'paso': paso,
        'poc': _tick(precio_min + (poc_idx + 0.5) * paso),
        'vah': _tick(precio_min + (hi + 1) * paso),
        'val': _tick(precio_min + lo * paso),
        'vwap': _tick(vwap),
        'va_idx': {'lo': int(lo), 'hi': int(hi), 'poc': poc_idx},
    }


def _book_reconstruido(h, precio_min, paso):
    """Paredes BID/ASK modeladas: proximidad * volumen relativo * persistence."""
    cierre = h['Close'].values
    ultimo = float(cierre[-1])
    vol_nivel = np.zeros(N_NIVELES)
    cent = np.clip(((cierre - precio_min) / paso).astype(int), 0, N_NIVELES - 1)
    for i, v in zip(cent, h['Volume'].values):
        vol_nivel[i] += v
    vol_rel = vol_nivel / max(vol_nivel.max(), 1)

    # persistencia: en cuantas velas recientes el nivel fue tocado por el rango
    tocadas = np.zeros(N_NIVELES)
    for i in range(len(h)):
        lo_i = int(np.clip((h['Low'].values[i] - precio_min) / paso, 0, N_NIVELES - 1))
        hi_i = int(np.clip((h['High'].values[i] - precio_min) / paso, 0, N_NIVELES - 1))
        tocadas[lo_i:hi_i + 1] += 1
    toc_rel = tocadas / tocadas.max() if tocadas.max() > 0 else tocadas

    niveles = []
    for i in range(N_NIVELES):
        precio_nivel = _tick(precio_min + (i + 0.5) * paso)
        dist = abs(precio_nivel - ultimo) / max(ultimo, 1)
        # liquidez reconstruida: concentracion de volumen*persistencia con decay por distancia
        base = vol_rel[i] * 0.75 + toc_rel[i] * 0.25
        decay = math.exp(-dist * 900)
        lado = 'BID' if precio_nivel < ultimo else 'ASK'
        tamano = max(base * decay * (300 + 900 * vol_rel[i]), 1.0)
        niveles.append({
            'precio': precio_nivel,
            'lado': lado if precio_nivel != ultimo else ('ASK' if i % 2 else 'BID'),
            'tamano': round(tamano, 1),
            'vol_acum': round(float(vol_nivel[i]), 0),
        })
    return niveles, ultimo


def _footprint(h, precio_min, paso):
    """Delta por nivel: mecha-abajo cuenta como agresion vendedora, mecha-abajo compradora."""
    delta = np.zeros(N_NIVELES)
    comprado = np.zeros(N_NIVELES)
    vendido = np.zeros(N_NIVELES)
    for _, row in h.iterrows():
        idx = int(np.clip((row['Close'] - precio_min) / paso, 0, N_NIVELES - 1))
        rango = max(row['High'] - row['Low'], 1e-9)
        frac_up = (row['Close'] - row['Low']) / rango
        vol = row['Volume']
        comprado[idx] += vol * frac_up
        vendido[idx] += vol * (1 - frac_up)
        delta[idx] += vol * (2 * frac_up - 1)
    return comprado, vendido, delta


def _ballenas(niveles):
    """Ordenes > 3 sigma del tamano medio = ballenas."""
    tams = np.array([n['tamano'] for n in niveles])
    media, std = tams.mean(), tams.std()
    umbral = media + 2.5 * std
    return [{'precio': n['precio'], 'lado': n['lado'], 'tamano': n['tamano']}
            for n in niveles if n['tamano'] > umbral]


def _zonas_sr(h):
    """Soportes/resistencias: maximos y minimos locales agrupados."""
    highs, lows = h['High'].values, h['Low'].values
    tol = (highs.max() - lows.min()) * 0.012
    def agrupar(pivotes):
        pivotes = sorted(pivotes)
        grupos = []
        for p in pivotes:
            if grupos and abs(p - grupos[-1][-1]) <= tol:
                grupos[-1].append(p)
            else:
                grupos.append([p])
        out = []
        for g in grupos:
            nivel = float(np.mean(g))
            out.append({'precio': _tick(nivel), 'toques': len(g)})
        return out
    piv_h = [highs[i] for i in range(2, len(highs) - 2) if highs[i] == max(highs[i-2:i+3])]
    piv_l = [lows[i] for i in range(2, len(lows) - 2) if lows[i] == min(lows[i-2:i+3])]
    res = sorted(agrupar(piv_h), key=lambda z: -z['toques'])[:N_ZONAS // 2]
    sop = sorted(agrupar(piv_l), key=lambda z: -z['toques'])[:N_ZONAS // 2]
    return [{'tipo': 'resistencia', **z} for z in res] + [{'tipo': 'soporte', **z} for z in sop]


def _tape(h):
    """Ultima cinta de 'prints' reconstruidos (1 por vela reciente)."""
    tail = h.tail(TAM_TICK)
    prints = []
    for ts, row in tail.iterrows():
        agresor = 'COMPRA' if row['Close'] >= row['Open'] else 'VENTA'
        prints.append({
            't': ts.strftime('%H:%M:%S'),
            'precio': _tick(row['Close']),
            'tamano': max(int(row['Volume'] / 100), 1),
            'lado': agresor,
        })
    return prints[-TAM_TICK:]


def generar(ticker):
    try:
        h = _descargar(ticker)
        fuente = 'yfinance-5m'
    except Exception:
        h = _fallback(ticker)
        fuente = 'sintetico-fallback'
    fallback = bool(getattr(h.attrs, 'get', lambda k: None)('fallback'))

    precio_min = float(h['Low'].min())
    precio_max = float(h['High'].max())
    margen = (precio_max - precio_min) * 0.06 or precio_max * 0.005
    precio_min -= margen
    precio_max += margen
    paso = (precio_max - precio_min) / N_NIVELES

    perfil = _perfil(h, precio_min, precio_max)
    niveles, ultimo = _book_reconstruido(h, precio_min, paso)
    comprado, vendido, delta = _footprint(h, precio_min, paso)
    velas = [{'t': ts.strftime('%H:%M'), 'o': _tick(row['Open']), 'h': _tick(row['High']),
              'l': _tick(row['Low']), 'c': _tick(row['Close']), 'v': float(row['Volume'])}
             for ts, row in h.iterrows()]
    ema9 = _ema_list(h, 9)
    ema21 = _ema_list(h, 21)
    ema50 = _ema_list(h, 50)

    data = {
        'ticker': ticker.upper(),
        'generado': datetime.now(timezone.utc).isoformat(),
        'fuente': fuente,
        'fallback': fallback,
        'precio_actual': ultimo,
        'precio_min': _tick(precio_min),
        'precio_max': _tick(precio_max),
        'niveles': niveles,
        'perfil': perfil,
        'footprint': {
            'comprado': [round(x) for x in comprado],
            'vendido': [round(x) for x in vendido],
            'delta': [round(x) for x in delta],
        },
        'muros': [{'precio': n['precio'], 'lado': n['lado'], 'tamano': n['tamano']}
                  for n in niveles if n['tamano'] > np.percentile([x['tamano'] for x in niveles], 90)],
        'ballenas': _ballenas(niveles),
        'zonas_sr': _zonas_sr(h),
        'tape': _tape(h),
        'velas': velas,
        'ema': {'9': ema9, '21': ema21, '50': ema50},
        'agora': {'compra': round(float(comprado.sum())), 'venta': round(float(vendido.sum()))},
    }
    out = REPORTES / 'profundidad_3d_data.json'
    out.write_text(json.dumps(data, ensure_ascii=False), encoding='utf-8')
    print(f'[3D] {ticker.upper()}: {len(velas)} velas, {len(niveles)} niveles, '
          f'POC={perfil["poc"]} -> {out.name} ({fuente})')
    return data


def _ema_list(h, periodo):
    c = h['Close'].values
    k = 2 / (periodo + 1)
    ema = [float(c[0])]
    for p in c[1:]:
        ema.append(float(p) * k + ema[-1] * (1 - k))
    return [round(x, 2) for x in ema]


if __name__ == '__main__':
    t = sys.argv[1] if len(sys.argv) > 1 else 'NVDA'
    generar(t)
