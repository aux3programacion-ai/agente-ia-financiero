#!/usr/bin/env python3
"""
aprendizaje.py - Sistema de aprendizaje autonomo.
Evalua predicciones pasadas vs resultados reales y calibra
probabilidades usando ajuste bayesiano.
"""
import json, os, time, math

DATA_DIR = os.environ.get('GITHUB_WORKSPACE', '.')
HIST_PATH = os.path.join(DATA_DIR, 'Datos', 'predicciones_hist.json')
PRECIOS_PATH = os.path.join(DATA_DIR, 'Datos', 'precios_reales.json')
IA_PATH = os.path.join(DATA_DIR, 'Datos', 'analisis_ia.json')
CALIB_PATH = os.path.join(DATA_DIR, 'Datos', 'calibracion.json')
os.makedirs(os.path.join(DATA_DIR, 'Datos'), exist_ok=True)

TICKERS = ['NVDA','MU','DELL','AVGO','DDOG','SMCI','SNOW','CRWD','NOW','TSM',
           'ARM','OKTA','HPE','NTAP','CLS','AAPL','AMZN','GOOGL','META','MSFT',
           'LLY','AMAT','LRCX','PANW','ORCL','HON','UBER','GE','COST','NEE']

# --- Cargar base historica ---
hist = {}
if os.path.exists(HIST_PATH):
    try:
        with open(HIST_PATH) as f:
            hist = json.load(f)
    except Exception as e:
        print(f'[!] Error cargando historico: {e}')

for t in TICKERS:
    if t not in hist:
        hist[t] = {'predicciones': [], 'total': 0, 'aciertos': 0, 'precision': 0.5}

# --- Cargar precios actuales ---
precios = {}
if os.path.exists(PRECIOS_PATH):
    try:
        pd = json.load(open(PRECIOS_PATH)).get('precios', {})
        for t in TICKERS:
            p = pd.get(t, {})
            if p: precios[t] = {'price': p.get('price',0), 'change': p.get('change',0)}
    except Exception as e:
        print(f'[!] Error precios: {e}')

# --- Evaluar predicciones pendientes ---
evaluados = 0
for t in TICKERS:
    preds = hist[t]['predicciones']
    for p in preds:
        if p.get('resultado') is None and t in precios:
            cambio_real = precios[t]['change']
            p['resultado'] = 'up' if cambio_real >= 0 else 'down'
            p['precio_real'] = precios[t]['price']
            p['acertada'] = p['direccion'] == p['resultado']
            if p['acertada']:
                hist[t]['aciertos'] += 1
            hist[t]['total'] += 1
            evaluados += 1

# Recalcular precision
for t in TICKERS:
    total = hist[t]['total']
    aciertos = hist[t]['aciertos']
    hist[t]['precision'] = round(aciertos / total, 4) if total > 0 else 0.5

# --- Calcular promedios globales ---
total_global = sum(hist[t]['total'] for t in TICKERS)
aciertos_global = sum(hist[t]['aciertos'] for t in TICKERS)
precision_global = round(aciertos_global / total_global, 4) if total_global > 0 else 0.5

# --- Guardar historico actualizado ---
with open(HIST_PATH, 'w') as f:
    json.dump(hist, f, indent=2, ensure_ascii=False)

# --- Generar calibracion bayesiana ---
calibracion = {
    'timestamp': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
    'precision_global': precision_global,
    'total_evaluado': total_global,
    'aciertos_global': aciertos_global,
    'evaluados_este_ciclo': evaluados,
    'factores': {}
}

for t in TICKERS:
    h = hist[t]
    total = h['total']
    precision = h['precision']

    # Ajuste bayesiano: combinamos precision del ticker con precision global
    # Si hay pocos datos (< 5), pesa mas la precision global
    peso_local = min(total / 5, 1.0)
    precision_ajustada = precision * peso_local + precision_global * (1 - peso_local)

    # Factor de calibracion: precision_ajustada / 0.5 (0.5 = baseline sin info)
    factor = precision_ajustada / 0.5 if precision_ajustada > 0 else 1.0

    calibracion['factores'][t] = {
        'total': total,
        'aciertos': h['aciertos'],
        'precision': precision,
        'precision_ajustada': round(precision_ajustada, 4),
        'factor': round(factor, 4),
        'peso_local': round(peso_local, 2)
    }

with open(CALIB_PATH, 'w') as f:
    json.dump(calibracion, f, indent=2, ensure_ascii=False)

# --- Aplicar calibracion a analisis actual ---
if os.path.exists(IA_PATH):
    try:
        ia = json.load(open(IA_PATH))
        for t in TICKERS:
            if t in ia.get('probabilidades', {}):
                prob = ia['probabilidades'][t]['probabilidad']
                conf = ia['probabilidades'][t]['confianza']
                factor = calibracion['factores'][t]['factor']
                prob_calib = max(5, min(95, round(prob * factor)))
                conf_calib = max(5, min(95, round(conf * (0.5 + 0.5 * factor))))
                ia['probabilidades'][t]['probabilidad_original'] = prob
                ia['probabilidades'][t]['confianza_original'] = conf
                ia['probabilidades'][t]['probabilidad'] = prob_calib
                ia['probabilidades'][t]['confianza'] = conf_calib
                ia['probabilidades'][t]['factor_calibracion'] = round(factor, 4)
        ia['calibracion_aplicada'] = True
        ia['precision_global'] = precision_global
        ia['total_evaluado'] = total_global
        json.dump(ia, open(IA_PATH, 'w'), indent=2, ensure_ascii=False)
        print(f'[OK] Calibracion aplicada a {len(TICKERS)} tickers')
    except Exception as e:
        print(f'[!] Error aplicando calibracion: {e}')

print(f'[OK] Precision global: {precision_global:.1%} ({aciertos_global}/{total_global})')
print(f'[OK] Evaluados este ciclo: {evaluados} predicciones')
