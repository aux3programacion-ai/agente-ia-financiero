#!/usr/bin/env python3
"""
consulta_accion.py - Genera Reportes/consulta_data.json

Consolida por ticker la "sensacion del mercado":
  probabilidad IA (analisis_ia.json) + noticias con sentimiento
  (noticias_recientes.json) + tecnicos (analisis_tecnico.json)
  + precision historica (calibracion.json / modelo_xgboost.json)

La app Reportes/consulta.html solo lee este JSON.
Uso:  python consulta_accion.py            # tickers del analisis_ia
      python consulta_accion.py AAPL TSLA  # fuerza tickers concretos
"""
import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

REPO = Path(__file__).parent
DATOS = REPO / 'Datos'
REPORTES = REPO / 'Reportes'
REPORTES.mkdir(exist_ok=True)

from calibracion_real import calibracion_real  # noqa: E402


def _leer(nombre):
    p = DATOS / nombre
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text(encoding='utf-8'))
    except Exception:
        return {}


def _sentimiento_por_ticker(noticias):
    """noticias_recientes.json -> {ticker: {sentimiento, score, impacto, resumen, titulares[]}}"""
    out = {}
    por_t = noticias.get('por_ticker', {})
    if isinstance(por_t, dict):
        for t, v in por_t.items():
            if not isinstance(v, dict):
                continue
            items = v.get('noticias', v.get('items', []))
            titulares = [i.get('titulo', '') for i in items[:5] if isinstance(i, dict)]
            # Titulares en espanol del ciclo (noticias_es.py) y tono medido con
            # lexico local, que sirve de respaldo cuando la IA no responde.
            titulares_es = v.get('titulares_es') or []
            s = v.get('sentimiento_ia') or v.get('sentimiento_es') or v.get('sentimiento')
            if isinstance(s, dict):
                titulares_es = titulares_es or (s.get('titulares_es') or [])
                base = titulares_es or titulares
                out[t.upper()] = {
                    'sentimiento': s.get('sentimiento', 'neutral'),
                    'score': float(s.get('score', 0) or 0),
                    'impacto': s.get('impacto', 'medio'),
                    'tema': s.get('tema_principal', ''),
                    'resumen': s.get('resumen', ''),
                    'titulares': titulares,
                    'titulares_es': titulares_es,
                    'idioma_titulares': 'es' if titulares_es else 'en',
                    'fuente_analisis': s.get('fuente_analisis', ''),
                    'num_noticias': int(len(base)),
                }
            else:
                out[t.upper()] = {'sentimiento': 'neutral', 'score': 0.0,
                                  'impacto': 'bajo', 'tema': '', 'resumen': '',
                                  'titulares': titulares,
                                  'titulares_es': titulares_es,
                                  'idioma_titulares': 'es' if titulares_es else 'en',
                                  'fuente_analisis': '',
                                  'num_noticias': int(len(titulares_es or titulares))}
    return out


def _rendimiento_emitido(ticker, cal, xgb):
    """Precision historica de predicciones emitidas (aprendizaje.py)."""
    prec = cal.get('precision_global')
    prec_pond = cal.get('precision_ponderada')
    n_eval = cal.get('total_evaluado', 0)
    xgb_acc = xgb.get('wf_accuracy_mean')
    return {
        'precision_direccion': round(float(prec) * 100, 1) if isinstance(prec, (int, float)) and prec <= 1 else (round(float(prec), 1) if isinstance(prec, (int, float)) else None),
        'precision_ponderada': round(float(prec_pond) * 100, 1) if isinstance(prec_pond, (int, float)) and prec_pond <= 1 else (round(float(prec_pond), 1) if isinstance(prec_pond, (int, float)) else None),
        'evaluados': int(n_eval),
        'backtest_wf_accuracy': round(float(xgb_acc) * 100) if isinstance(xgb_acc, (int, float)) else None,
    }


def _consenso(p_ia, p_sent_score, w_ia=0.70, w_sent=0.30):
    """Mezcla la probabilidad IA con el sesgo del sentimiento de noticias."""
    p_comb = w_ia * p_ia + w_sent * p_sent_score * 100 * 0.5 + w_sent * 50 * 0.5
    return max(2.0, min(98.0, p_comb))


def construir_ticker(t, p_ia_row, sent, tecnicos, regimen, rendimiento):
    p_ia = p_ia_row.get('probabilidad', 50)
    conf = p_ia_row.get('confianza', 50)
    targets = {k: p_ia_row.get(f'precio_objetivo_{k}') for k in ('30d', '3m', '6m', '1y')}
    tec = tecnicos.get(t, {}) if isinstance(tecnicos, dict) else {}

    senales = []
    # Senal tecnica
    rsi = tec.get('rsi')
    if isinstance(rsi, (int, float)):
        if rsi < 30:
            senales.append(('tecnico', 'RSI sobrevendido (' + str(round(rsi)) + ') -> sesgo de rebote al alza', +1))
        elif rsi > 70:
            senales.append(('tecnico', 'RSI sobrecomprado (' + str(round(rsi)) + ') -> riesgo de correccion', -1))
        else:
            senales.append(('tecnico', 'RSI neutro (' + str(round(rsi)) + ')', 0))
    trend_raw = tec.get('tendencia')
    if isinstance(trend_raw, str):
        d = +1 if 'alcis' in trend_raw.lower() or 'alcista' in trend_raw.lower() else (-1 if 'baja' in trend_raw.lower() else 0)
        senales.append(('tecnico', 'Tendencia: ' + trend_raw, d))

    # Senal noticias
    if sent and sent.get('num_noticias'):
        sc = sent['score']
        if abs(sc) >= 0.3:
            senales.append(('noticias',
                f"Sentimiento de noticias {sent['sentimiento']} (score {sc:+.1f}, impacto {sent['impacto']}): "
                + (sent['resumen'] or sent['tema'] or (sent['titulares'][0] if sent['titulares'] else '')),
                +1 if sc > 0 else -1))
        elif sent['titulares']:
            senales.append(('noticias', 'Noticias neutrales: ' + sent['titulares'][0], 0))

    # Peso del sentimiento en el consenso
    p_sent = 0.5 + (sent['score'] or 0) * 0.5 if sent else 0.5
    p_final = _consenso(float(p_ia), p_sent if sent else 0.5)

    if p_final >= 60:
        veredicto = 'MAYOR PROBABILIDAD DE SUBIR'
    elif p_final <= 40:
        veredicto = 'MAYOR PROBABILIDAD DE BAJAR'
    else:
        veredicto = 'RANGO / SIN SESGO CLARO'

    return {
        'ticker': t.upper(),
        'prob_subir': round(p_final, 1),
        'prob_bajar': round(100 - p_final, 1),
        'veredicto': veredicto,
        'confianza_ia': conf,
        'precision_historica': rendimiento,
        'sentimiento_noticias': sent or None,
        'senales_explicacion': [{'fuente': f, 'texto': txt, 'direccion': d} for f, txt, d in senales],
        'targets': {k: (round(v, 2) if isinstance(v, (int, float)) else None) for k, v in targets.items()},
        'regimen_mercado': regimen,
        'rsi': rsi if isinstance(rsi, (int, float)) else None,
        'tendencia': trend_raw if isinstance(trend_raw, str) else None,
    }


def main():
    ia = _leer('analisis_ia.json')
    noticias = _leer('noticias_recientes.json')
    tec_doc = _leer('analisis_tecnico.json')
    cal = _leer('calibracion.json')
    xgb = _leer('modelo_xgboost.json')

    probs = ia.get('probabilidades', {}) if ia else {}
    sent_map = _sentimiento_por_ticker(noticias)
    tecnicos = tec_doc.get('tecnicos', {})
    regimen = tec_doc.get('regimen_mercado', 'desconocido')

    pedir = [a.upper() for a in sys.argv[1:]]
    tickers = list(dict.fromkeys(list(probs.keys()) + pedir)) if probs else pedir

    out = {'generado': datetime.now(timezone.utc).isoformat(),
           'regimen_mercado': regimen,
           'resumen_mercado': ia.get('resumen_mercado', ''),
           'calibracion_real': calibracion_real(DATOS),
           'tickers': []}

    for t in tickers:
        row = probs.get(t) or {}
        if not row and t not in pedir:
            continue
        if not row:
            row = {'probabilidad': 50, 'confianza': 20,
                   'analisis': 'Sin analisis IA todavia para este ticker; consulta basada solo en noticias/tecnicos.',
                   'mercado': 'US'}
        out['tickers'].append(construir_ticker(
            t, row, sent_map.get(t), tecnicos, regimen,
            _rendimiento_emitido(t, cal, xgb)))

    dst = REPORTES / 'consulta_data.json'
    dst.write_text(json.dumps(out), encoding='utf-8')
    print(f'[consulta] {len(out["tickers"])} tickers -> {dst.name}')


if __name__ == '__main__':
    main()
