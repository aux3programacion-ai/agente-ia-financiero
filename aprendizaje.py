#!/usr/bin/env python3
"""
aprendizaje.py - Sistema de aprendizaje autonomo multi-factor.
Evaluacion temporal ponderada, calibracion por sector/rango/modelo,
decaimiento exponencial, y retroalimentacion estructurada al AI.
"""
import json, os, time, math
from datetime import datetime, timezone

DATA_DIR = os.environ.get('GITHUB_WORKSPACE', '.')
HIST_PATH = os.path.join(DATA_DIR, 'Datos', 'predicciones_hist.json')
PRECIOS_PATH = os.path.join(DATA_DIR, 'Datos', 'precios_reales.json')
IA_PATH = os.path.join(DATA_DIR, 'Datos', 'analisis_ia.json')
CALIB_PATH = os.path.join(DATA_DIR, 'Datos', 'calibracion.json')
NEWS_PATH = os.path.join(DATA_DIR, 'Datos', 'noticias_recientes.json')
os.makedirs(os.path.join(DATA_DIR, 'Datos'), exist_ok=True)

TICKERS = ['NVDA','MU','DELL','AVGO','DDOG','SMCI','SNOW','CRWD','NOW','TSM',
           'ARM','OKTA','HPE','NTAP','CLS','AAPL','AMZN','GOOGL','META','MSFT',
           'LLY','AMAT','LRCX','PANW','ORCL','HON','UBER','GE','COST','NEE']

SECTOR_MAP = {
    'Semiconductores': ['NVDA','MU','AVGO','TSM','ARM'],
    'Servidores IA': ['DELL','SMCI','HPE'],
    'Software IA': ['DDOG','SNOW','NOW'],
    'Ciberseguridad': ['CRWD','PANW','OKTA'],
    'Almacenamiento': ['NTAP','CLS'],
    'Consumer Tech': ['AAPL','AMZN','GOOGL','META','MSFT'],
    'Farmaceutico': ['LLY'],
    'Semicon Equip': ['AMAT','LRCX'],
    'Cloud/Database': ['ORCL'],
    'Industrial': ['HON','GE'],
    'Movilidad/Tech': ['UBER'],
    'Consumo Defensivo': ['COST'],
    'Utilities/Energy': ['NEE']
}

RANGOS_PROB = [(0,40),(40,50),(50,55),(55,60),(60,65),(65,70),(70,80),(80,100)]

def peso_temporal(dias_antiguedad):
    """Decaimiento exponencial: los mas recientes pesan mas. Half-life 14 dias."""
    return math.exp(-dias_antiguedad * math.log(2) / 14)

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

# --- Cargar sentimiento de noticias ---
if os.path.exists(NEWS_PATH):
    try:
        nd = json.load(open(NEWS_PATH)).get('por_ticker', {})
    except:
        nd = {}
else:
    nd = {}

# --- Evaluar predicciones pendientes ---
hoy = datetime.now(timezone.utc).strftime('%Y-%m-%d')
evaluados = 0
for t in TICKERS:
    preds = hist[t]['predicciones']
    for p in preds:
        if p.get('resultado') is None and t in precios:
            cambio_real = precios[t]['change']
            p['resultado'] = 'up' if cambio_real >= 0 else 'down'
            p['precio_real'] = precios[t]['price']
            p['acertada'] = p['direccion'] == p['resultado']
            p['fecha_evaluacion'] = hoy
            if p['acertada']:
                hist[t]['aciertos'] += 1
            hist[t]['total'] += 1
            evaluados += 1

# --- Verificar targets a 30 dias ---
targets_verificados = 0
targets_acertados = 0
error_pct_total = 0
targets_con_error = 0
for t in TICKERS:
    for p in hist[t]['predicciones']:
        if p.get('precio_objetivo_30d') and p.get('precio_real') and not p.get('target_verificado'):
            try:
                dias = (datetime.now(timezone.utc) - datetime.strptime(p['fecha'], '%Y-%m-%d')).days
            except:
                dias = 0
            if dias >= 30 and p['precio_real'] and p['precio_objetivo_30d'] > 0:
                real = p['precio_real']
                objetivo = p['precio_objetivo_30d']
                error_pct = abs(real - objetivo) / real
                error_pct_total += error_pct
                targets_con_error += 1
                # Acierto: target dentro del 10% del precio real
                acerto_target = error_pct <= 0.10
                p['target_verificado'] = True
                p['target_error_pct'] = round(error_pct * 100, 1)
                p['target_acertado'] = acerto_target
                targets_verificados += 1
                if acerto_target:
                    targets_acertados += 1

precision_target_30d = round(targets_acertados / targets_verificados, 4) if targets_verificados > 0 else None
error_promedio_target = round(error_pct_total / targets_con_error * 100, 1) if targets_con_error > 0 else None

# --- Calcular precision PONDERADA por tiempo ---
for t in TICKERS:
    preds = hist[t]['predicciones']
    peso_total = 0
    aciertos_pond = 0
    for p in preds:
        if p.get('acertada') is not None:
            try:
                dias = (datetime.now(timezone.utc) - datetime.strptime(p['fecha'], '%Y-%m-%d')).days
            except:
                dias = 30
            w = peso_temporal(dias)
            peso_total += w
            if p['acertada']:
                aciertos_pond += w
    if peso_total > 0:
        hist[t]['precision_ponderada'] = round(aciertos_pond / peso_total, 4)
    else:
        hist[t]['precision_ponderada'] = 0.5

# --- Recalcular precision simple ---
for t in TICKERS:
    total = hist[t]['total']
    aciertos = hist[t]['aciertos']
    hist[t]['precision'] = round(aciertos / total, 4) if total > 0 else 0.5

# --- Calcular promedios globales ---
total_global = sum(hist[t]['total'] for t in TICKERS)
aciertos_global = sum(hist[t]['aciertos'] for t in TICKERS)
precision_global = round(aciertos_global / total_global, 4) if total_global > 0 else 0.5

# Precision ponderada global
peso_total_global = 0
aciertos_pond_global = 0
for t in TICKERS:
    for p in hist[t]['predicciones']:
        if p.get('acertada') is not None:
            try:
                dias = (datetime.now(timezone.utc) - datetime.strptime(p['fecha'], '%Y-%m-%d')).days
            except:
                dias = 30
            w = peso_temporal(dias)
            peso_total_global += w
            if p['acertada']:
                aciertos_pond_global += w
precision_pond_global = round(aciertos_pond_global / peso_total_global, 4) if peso_total_global > 0 else 0.5

# --- Guardar historico actualizado ---
with open(HIST_PATH, 'w') as f:
    json.dump(hist, f, indent=2, ensure_ascii=False)

# ============================================================
# CALIBRACION MULTI-FACTOR
# ============================================================

# --- 1. Calibracion por SECTOR ---
sector_stats = {}
for sector, tickers in SECTOR_MAP.items():
    total_s = 0; aciertos_s = 0
    peso_s = 0; aciertos_pond_s = 0
    for t in tickers:
        for p in hist[t]['predicciones']:
            if p.get('acertada') is not None:
                total_s += 1
                if p['acertada']: aciertos_s += 1
    sector_stats[sector] = {
        'total': total_s,
        'aciertos': aciertos_s,
        'precision': round(aciertos_s / total_s, 4) if total_s > 0 else 0.5
    }

# --- 2. Calibracion por RANGO DE PROBABILIDAD ---
rango_stats = {}
for (lo, hi) in RANGOS_PROB:
    total_r = 0; aciertos_r = 0
    for t in TICKERS:
        for p in hist[t]['predicciones']:
            if p.get('acertada') is not None:
                prob = p.get('probabilidad', 50)
                if lo <= prob < hi:
                    total_r += 1
                    if p['acertada']: aciertos_r += 1
    rango_stats[f'{lo}-{hi}'] = {
        'total': total_r,
        'aciertos': aciertos_r,
        'precision': round(aciertos_r / total_r, 4) if total_r > 0 else None
    }

# --- 3. Calibracion por MODELO IA ---
modelo_stats = {}
for t in TICKERS:
    for p in hist[t]['predicciones']:
        if p.get('acertada') is not None:
            mod = p.get('modelo_usado', 'desconocido')
            if mod not in modelo_stats:
                modelo_stats[mod] = {'total': 0, 'aciertos': 0}
            modelo_stats[mod]['total'] += 1
            if p['acertada']:
                modelo_stats[mod]['aciertos'] += 1
for m in modelo_stats:
    modelo_stats[m]['precision'] = round(modelo_stats[m]['aciertos'] / modelo_stats[m]['total'], 4)

# --- 3b. Precision por MODELO + TICKER (granular) ---
modelo_ticker_stats = {}
for t in TICKERS:
    modelo_ticker_stats[t] = {}
    for p in hist[t]['predicciones']:
        if p.get('acertada') is not None:
            mod = p.get('modelo_usado', 'desconocido')
            if mod not in modelo_ticker_stats[t]:
                modelo_ticker_stats[t][mod] = {'total': 0, 'aciertos': 0}
            modelo_ticker_stats[t][mod]['total'] += 1
            if p['acertada']:
                modelo_ticker_stats[t][mod]['aciertos'] += 1
    for m in modelo_ticker_stats[t]:
        ms = modelo_ticker_stats[t][m]
        ms['precision'] = round(ms['aciertos'] / ms['total'], 4) if ms['total'] > 0 else 0.5

# --- 3c. Correlacion NOTICIA → PRECIO ---
news_correlacion = {'positivo': {'total': 0, 'acertadas': 0, 'precision': None},
                     'negativo': {'total': 0, 'acertadas': 0, 'precision': None},
                     'neutral': {'total': 0, 'acertadas': 0, 'precision': None}}
for t in TICKERS:
    for p in hist[t]['predicciones']:
        ns = p.get('news_sentimiento')
        acertada = p.get('acertada')
        if ns and acertada is not None and ns in news_correlacion:
            news_correlacion[ns]['total'] += 1
            if acertada:
                news_correlacion[ns]['acertadas'] += 1
for k in news_correlacion:
    nc = news_correlacion[k]
    nc['precision'] = round(nc['acertadas'] / nc['total'], 4) if nc['total'] > 0 else None

# Precision de noticias: cuando sentimiento es positivo y prediccion es up, que tan seguido acierta?
precision_noticia_alcista = 0
total_noticia_alcista = 0
for t in TICKERS:
    for p in hist[t]['predicciones']:
        if p.get('news_sentimiento') == 'positivo' and p.get('direccion') == 'up' and p.get('acertada') is not None:
            total_noticia_alcista += 1
            if p['acertada']:
                precision_noticia_alcista += 1
prec_noticia_alcista = round(precision_noticia_alcista / total_noticia_alcista, 4) if total_noticia_alcista > 0 else None

# --- 4. Calibracion por ticker con temporal weighting ---
calibracion = {
    'timestamp': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
    'precision_global': precision_global,
    'precision_ponderada': precision_pond_global,
    'total_evaluado': total_global,
    'aciertos_global': aciertos_global,
    'evaluados_este_ciclo': evaluados,
    'precision_target_30d': precision_target_30d,
    'error_promedio_target_pct': error_promedio_target,
    'targets_verificados': targets_verificados,
    'sectores': sector_stats,
    'rangos_probabilidad': rango_stats,
    'modelos_ia': modelo_stats,
    'modelo_por_ticker': modelo_ticker_stats,
    'correlacion_noticias': news_correlacion,
    'prec_noticia_alcista': prec_noticia_alcista,
    'factores': {}
}

for t in TICKERS:
    h = hist[t]
    total = h['total']
    precision = h['precision']
    precision_pond = h['precision_ponderada']

    # Factor base: ponderada por tiempo
    peso_local = min(total / 5, 1.0)
    precision_combinada = precision_pond * peso_local + precision_pond_global * (1 - peso_local)

    # Ajuste por sector
    sector = None
    for s, tkrs in SECTOR_MAP.items():
        if t in tkrs:
            sector = s
            break
    prec_sector = sector_stats.get(sector, {}).get('precision', 0.5) if sector else 0.5

    # Factor final: promedio ponderado entre ticker, sector, y global
    w_ticker = min(total / 8, 0.6)
    w_sector = 0.2
    w_global = 1.0 - w_ticker - w_sector
    precision_final = (precision_combinada * w_ticker +
                       prec_sector * w_sector +
                       precision_pond_global * w_global)

    factor = precision_final / 0.5 if precision_final > 0 else 1.0

    calibracion['factores'][t] = {
        'total': total,
        'aciertos': h['aciertos'],
        'precision': precision,
        'precision_ponderada': precision_pond,
        'precision_final': round(precision_final, 4),
        'factor': round(factor, 4),
        'peso_local': round(peso_local, 2),
        'sector': sector,
        'precision_sector': prec_sector
    }

with open(CALIB_PATH, 'w') as f:
    json.dump(calibracion, f, indent=2, ensure_ascii=False)

# ============================================================
# APLICAR CALIBRACION y generar retroalimentacion para el AI
# ============================================================
if os.path.exists(IA_PATH):
    try:
        ia = json.load(open(IA_PATH))
        feedback_lines = [
            f'[APRENDIZAJE] Precision global: {precision_pond_global:.1%} ponderada ({precision_global:.1%} simple)',
            f'[APRENDIZAJE] Evaluados: {total_global} predicciones ({evaluados} este ciclo)',
            '',
            '[SECTORES CON MEJOR PRECISION]:'
        ]
        sectores_ordenados = sorted(sector_stats.items(), key=lambda x: x[1]['precision'], reverse=True)
        for s, st in sectores_ordenados[:5]:
            if st['total'] > 0:
                feedback_lines.append(f'  {s}: {st["precision"]:.0%} ({st["aciertos"]}/{st["total"]})')

        feedback_lines.append('')
        feedback_lines.append('[RANGOS DE PROBABILIDAD]:')
        for (lo, hi) in RANGOS_PROB:
            rs = rango_stats.get(f'{lo}-{hi}', {})
            if rs.get('total', 0) > 0:
                rp = rs['precision']
                feedback_lines.append(f'  {lo}-{hi}%: {rp:.0%} ({rs["aciertos"]}/{rs["total"]})')

        feedback_lines.append('')
        feedback_lines.append('[PRECISION POR TICKER (ponderada)]:')
        for t in TICKERS:
            f = calibracion['factores'][t]
            if f['total'] > 0:
                feedback_lines.append(f'  {t}: {f["precision_ponderada"]:.0%} ({f["aciertos"]}/{f["total"]})')

        feedback_lines.append('')
        feedback_lines.append('[MEJORES MODELOS IA]:')
        mejores_modelos = sorted(modelo_stats.items(), key=lambda x: x[1]['precision'], reverse=True)[:5]
        for m, ms in mejores_modelos:
            if ms['total'] >= 3:
                feedback_lines.append(f'  {m}: {ms["precision"]:.0%} ({ms["aciertos"]}/{ms["total"]})')

        feedback_lines.append('')
        if precision_target_30d is not None:
            feedback_lines.append(f'[TARGETS 30d] Precision: {precision_target_30d:.0%} ({targets_acertados}/{targets_verificados} dentro del 10%) | Error prom: {error_promedio_target}%')
        else:
            feedback_lines.append('[TARGETS 30d] Aun sin datos suficientes para evaluar')
        if prec_noticia_alcista is not None:
            feedback_lines.append(f'[NOTICIAS] Precision cuando noticia positiva + prediccion alcista: {prec_noticia_alcista:.0%} ({precision_noticia_alcista}/{total_noticia_alcista})')
        for k, nc in news_correlacion.items():
            if nc['total'] > 0:
                feedback_lines.append(f'  Sentimiento {k}: precision {nc["precision"]:.0%} ({nc["acertadas"]}/{nc["total"]})')

        feedback_text = '\n'.join(feedback_lines)
        ia['feedback_aprendizaje'] = feedback_text
        ia['precision_ponderada'] = precision_pond_global
        ia['precision_global'] = precision_global
        ia['total_evaluado'] = total_global

        # Aplicar calibracion a probabilidades
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
                ia['probabilidades'][t]['precision_historica'] = calibracion['factores'][t]['precision_ponderada']

        ia['calibracion_aplicada'] = True
        json.dump(ia, open(IA_PATH, 'w'), indent=2, ensure_ascii=False)
        print(f'[OK] Calibracion multi-factor aplicada a {len(TICKERS)} tickers')
    except Exception as e:
        print(f'[!] Error aplicando calibracion: {e}')

print(f'[OK] Precision global ponderada: {precision_pond_global:.1%} ({aciertos_global}/{total_global})')
print(f'[OK] Evaluados este ciclo: {evaluados}')
print(f'[OK]{len(modelo_stats)} modelos tracking | {len([r for r in rango_stats.values() if r["total"]>0])} rangos calibrados')
