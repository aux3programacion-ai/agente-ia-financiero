#!/usr/bin/env python3
"""calibracion_real.py - ¿La "confianza" del modelo significa algo?

Mide, con las predicciones YA RESUELTAS del propio sistema
(Datos/predicciones_hist.json), qué porcentaje subió de verdad en cada tramo
de probabilidad declarada, y calcula el Brier score (0.25 = tirar una moneda).

Lo usan servidor_consulta.py (análisis en vivo) y consulta_accion.py (ciclo
diario) para mostrar la misma curva en Reportes/consulta.html.
"""
import json
from pathlib import Path

_ALCISTA = ('up', 'alcista', 'subir', 'alza')
_CACHE = {'mtime': None, 'datos': None}


def es_alcista(direccion):
    d = str(direccion or '').lower()
    return any(a in d for a in _ALCISTA)


def calibracion_real(datos_dir):
    """Curva de calibración de las predicciones resueltas, o None si no hay datos.

    Devuelve un dict con: evaluadas, tickers, aciertos_pct, brier,
    skill_brier_pct, desde, hasta, tramos[{rango,decia,real,n}] y
    por_confianza[{rango,aciertos,n}]. Se cachea hasta que el archivo cambie.
    """
    datos_dir = Path(datos_dir)
    p = datos_dir / 'predicciones_hist.json'
    if not p.exists():
        return None
    try:
        mt = p.stat().st_mtime
    except OSError:
        mt = None
    if _CACHE['mtime'] == mt and _CACHE['datos'] is not None:
        return _CACHE['datos']
    try:
        crudo = json.loads(p.read_text(encoding='utf-8'))
    except Exception:
        return None

    filas = []
    for _, v in crudo.items():
        for r in v.get('predicciones', []):
            if r.get('acertada') is None or r.get('probabilidad') is None:
                continue
            filas.append({
                'prob': float(r['probabilidad']),
                'acerto': bool(r['acertada']),
                'alcista': es_alcista(r.get('direccion')),
                'fecha': r.get('fecha'),
                'confianza': r.get('confianza'),
            })
    if not filas:
        return None

    n = len(filas)
    aciertos = sum(1 for f in filas if f['acerto'])
    brier = 0.0
    for f in filas:
        p_up = f['prob'] / 100 if f['alcista'] else 1 - f['prob'] / 100
        subio = 1 if (f['alcista'] == f['acerto']) else 0
        brier += (p_up - subio) ** 2
    brier /= n

    tramos = []
    for lo, hi in ((50, 55), (55, 60), (60, 65), (65, 70), (70, 80), (80, 101)):
        g = [f for f in filas if lo <= f['prob'] < hi]
        if not g:
            continue
        real = sum(1 for f in g if f['alcista'] == f['acerto'])
        tramos.append({'rango': '%d-%d%%' % (lo, min(hi, 100)),
                       'decia': round(sum(f['prob'] for f in g) / len(g), 1),
                       'real': round(100 * real / len(g), 1), 'n': len(g)})

    conf_tramos = []
    for lo, hi in ((0, 55), (55, 60), (60, 70), (70, 80), (80, 101)):
        g = [f for f in filas if f['confianza'] is not None and lo <= float(f['confianza']) < hi]
        if not g:
            continue
        a = sum(1 for f in g if f['acerto'])
        conf_tramos.append({'rango': '%d-%d' % (lo, min(hi, 100)),
                            'aciertos': round(100 * a / len(g), 1), 'n': len(g)})

    fechas = [f['fecha'] for f in filas if f['fecha']]
    datos = {
        'evaluadas': n,
        'tickers': len(crudo),
        'aciertos_pct': round(100 * aciertos / n, 1),
        'brier': round(brier, 4),
        'azar_brier': 0.25,
        'skill_brier_pct': round((1 - brier / 0.25) * 100, 1),
        'desde': min(fechas) if fechas else None,
        'hasta': max(fechas) if fechas else None,
        'tramos': tramos,
        'por_confianza': conf_tramos,
    }
    _CACHE.update(mtime=mt, datos=datos)
    return datos
