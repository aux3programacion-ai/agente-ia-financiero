#!/usr/bin/env python3
"""
servidor_consulta.py - API de análisis bajo demanda para CUALQUIER acción de EE.UU.

GET /api/analisis?ticker=LULU  ->  análisis completo en vivo:
  - historial 3 meses (yfinance) -> técnicos (RSI, tendencia, ATR, vs media 50/200)
  - noticias del ticker en ESPAÑOL (Yahoo Finanzas ES / Google Noticias es) +
    sentimiento con IA (OpenRouter) o léxico financiero español como respaldo
  - probabilidades combinadas con la calibración histórica del sistema
Uso:  python servidor_consulta.py          # http://127.0.0.1:8010
"""
import json
import math
import os
import re
import sys
import time
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

REPO = Path(__file__).parent
DATOS = REPO / 'Datos'
sys.path.insert(0, str(REPO))

from calibracion_real import calibracion_real  # noqa: E402  (tras fijar sys.path)

app = FastAPI(title='Agente Financiero · Consulta on-demand')
app.add_middleware(CORSMiddleware, allow_origins=['*'],
                   allow_methods=['*'], allow_headers=['*'])

# ---------------------------------------------------------------- utilidades

def _leer(nombre):
    p = DATOS / nombre
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text(encoding='utf-8'))
    except Exception:
        return {}


def _validar_ticker(t):
    if not re.fullmatch(r'[A-Z][A-Z0-9.\-^=]{0,9}', t.upper()):
        raise HTTPException(400, 'Ticker inválido')
    return t.upper()


def _obtener_historial(ticker):
    import yfinance as yf
    h = yf.Ticker(ticker).history(period='3mo', interval='1d')
    if h.empty or len(h) < 40:
        raise HTTPException(404, f'Sin historial suficiente para {ticker} (¿ticker correcto?)')
    return h


def _tecnico(h, regimen='desconocido'):
    c = h['Close']
    rsi = _rsi(c)
    ema50 = float(c.tail(50).mean())
    ema200 = float(c.mean()) if len(c) < 200 else float(c.tail(200).mean())
    ultimo = float(c.iloc[-1])
    serie = c.tail(20).values
    if ultimo > ema50 > ema200 or _slope(serie) > 0.15:
        tendencia = 'uptrend'
    elif ultimo < ema50 < ema200 or _slope(serie) < -0.15:
        tendencia = 'downtrend'
    else:
        tendencia = 'lateral'
    atr = float((h['High'] - h['Low']).tail(14).mean())
    return {
        'precio': round(ultimo, 2),
        'rsi': round(rsi, 1),
        'tendencia': tendencia,
        'atr': round(atr, 2),
        'vs_media50': round((ultimo / ema50 - 1) * 100, 2),
        'vs_media200': round((ultimo / ema200 - 1) * 100, 2),
        'retorno_1m': round((ultimo / float(c.iloc[-21]) - 1) * 100, 2) if len(c) > 21 else None,
        'retorno_3m': round((ultimo / float(c.iloc[0]) - 1) * 100, 2),
    }


def _rsi(c, periodo=14):
    d = c.diff().dropna()
    gan = d.clip(lower=0).tail(periodo).mean()
    per = (-d.clip(upper=0)).tail(periodo).mean()
    if per == 0:
        return 100.0
    rs = gan / per
    return 100 - 100 / (1 + rs)


def _slope(serie):
    x = np.arange(len(serie))
    y = np.asarray(serie, dtype=float)
    if y.std() == 0:
        return 0.0
    return float(np.polyfit(x, y, 1)[0] / max(y.mean(), 1e-9) * 10)


# Palabras genéricas que no sirven para decidir si una noticia habla del ticker
_TOK_GENERICOS = {
    'inc', 'corp', 'ltd', 'plc', 'sa', 'nv', 'ag', 'group', 'grupo', 'company',
    'compania', 'holdings', 'holding', 'international', 'technologies', 'tech',
    'class', 'clase', 'stock', 'tokenized', 'coinbase', 'plc', 'the', 'and',
    'health', 'healthcare', 'athletica', 'brands', 'motors', 'systems', 'energy',
}
_NOMBRES_CACHE = {}


def _nombre_empresa(ticker):
    """Nombre corto de la empresa según yfinance, para buscar/filtrar noticias
    en español que de verdad sean del ticker. Cacheado por proceso."""
    if ticker in _NOMBRES_CACHE:
        return _NOMBRES_CACHE[ticker]
    nombre = ''
    try:
        import yfinance as yf
        info = yf.Ticker(ticker).info or {}
        nombre = (info.get('shortName') or info.get('longName') or '').strip()
    except Exception:
        nombre = ''
    _NOMBRES_CACHE[ticker] = nombre
    return nombre


def _tokens_nombre(nombre):
    return [w for w in re.findall(r'[a-z]{4,}', _sin_acentos(nombre))
            if w not in _TOK_GENERICOS]


# Palabras que delatan una noticia financiera (sin acentos, en minúsculas)
_PAL_FINANZAS = (
    'accion acciones bolsa mercado cotizacion cotiza resultados beneficio beneficios '
    'ingresos prevision previsiones precio precios objetivo objetivos valor valores '
    'cae caen cayo sube suben subio alza recorta recorta recortes rebaja rebajan '
    'trimestre dividendo dividendos ganancias perdidas ventas ingresos inversion '
    'inversores inversor riesgo multa investigacion demanda despido stock stocks share '
    'shares earnings revenue price market quarter rally drop drops gain gains loss '
    'losses analyst analysts downgrade upgrade guidance ipo etf'
).split()
_RE_FIN = re.compile(r'\b(' + '|'.join(_PAL_FINANZAS) + r')\b')
_PAL_ES = ('de', 'la', 'el', 'los', 'las', 'del', 'en', 'que', 'y', 'para', 'con',
           'por', 'se', 'al', 'un', 'una', 'su', 'sus', 'tras', 'ante', 'mas', 'o')
_RE_ES = re.compile(r'\b(' + '|'.join(_PAL_ES) + r')\b')


def _relevante(titulo, ticker, nombre):
    """¿El titular habla realmente de este ticker? Las fuentes en español
    devuelven ruido genérico para tickers poco cubiertos (p. ej. "Epic Games"
    para TOST), así que se exige el símbolo, o el nombre de la empresa
    acompañado de contexto financiero (evita falsos positivos del tipo
    "Jelly Toast se lanza en India")."""
    t = _sin_acentos(titulo)
    sim = _sin_acentos(ticker).lower()
    if re.search(r'\b' + re.escape(sim) + r'\b', t):
        return True
    if not any(w in t for w in _tokens_nombre(nombre)):
        return False
    return bool(_RE_FIN.search(t))


def _es_espanol(titulo):
    """Heurística: ¿el titular está en español? (las fuentes "es" devuelven
    a veces enlaces en inglés)."""
    return bool(_RE_ES.search(_sin_acentos(titulo)))


def _dedup(lista, n=5):
    vistos, out = set(), []
    for t in lista:
        k = _sin_acentos(t)[:70]
        if t and k not in vistos:
            vistos.add(k)
            out.append(t)
    return out[:n]


def _noticias(ticker):
    """Titulares del ticker priorizando FUENTES EN ESPAÑOL (Yahoo Finanzas ES y
    Google Noticias es-419 con el nombre real de la empresa), para que las
    noticias ya lleguen en español sin depender de traducir con IA. Sólo se
    aceptan titulares que mencionen el ticker o la empresa. Si no hay nada en
    español, usa la fuente en inglés (que la IA traduce o VADER puntúa).
    Devuelve {'es': [...], 'en': [...], 'nombre': str}."""
    nombre = _nombre_empresa(ticker)

    def _filtra(lista):
        return [t for t in lista if _relevante(t, ticker, nombre) and _es_espanol(t)]

    es = _filtra(n['titulo'] for n in _rss(
        f'https://es.finance.yahoo.com/rss/headline?s={ticker}', 6))
    if len(es) < 3:
        # edición España (suele traer titulares en español) + edición es-419
        consultas = []
        if nombre:
            consultas.append((f'"{nombre}" acciones', 'es', 'ES', 'ES:es'))
        consultas.append((f'{ticker} acciones', 'es-419', 'US', 'US:es-419'))
        for q, hl, gl, ceid in consultas:
            url = ('https://news.google.com/rss/search?q=' + urllib.parse.quote(q)
                   + f'&hl={hl}&gl={gl}&ceid={ceid}')
            es += _filtra(n['titulo'] for n in _rss(url, 6))
            if len(es) >= 3:
                break

    en = [n['titulo'] for n in _rss(f'https://finance.yahoo.com/rss/headline?s={ticker}', 5)]
    if not en:
        q = urllib.parse.quote(f'{ticker} stock')
        en = [n['titulo'] for n in _rss(
            f'https://news.google.com/rss/search?q={q}&hl=en-US&gl=US&ceid=US:en', 5)]
    return {'es': _dedup(es), 'en': _dedup(en), 'nombre': nombre}


def _rss(url, n):
    try:
        import xml.etree.ElementTree as ET
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=10) as resp:
            tree = ET.fromstring(resp.read())
        out = []
        for it in tree.iter('item'):
            titulo = (it.findtext('title') or '').strip()
            if titulo:
                out.append({'titulo': titulo,
                            'fecha': (it.findtext('pubDate') or '')[:24]})
            if len(out) >= n:
                break
        return out
    except Exception:
        return []


# Modelos gratuitos de OpenRouter, en orden de preferencia. Los modelos :free
# son volátiles (responden vacío, 429 o 403 según el enrutado), por eso se
# prueban en cadena hasta obtener JSON válido. El primero está medido: ~4 s y
# JSON correcto en pruebas repetidas; 'openrouter/free' queda de último recurso.
IA_TIMEOUT_S = 10  # presupuesto por modelo gratuito (son volátiles: a veces 429 o vacío)

# El plan gratuito de OpenRouter permite 50 peticiones/día a modelos :free. Cuando
# se agota (429 "free-models-per-day") NO tiene sentido seguir probando modelos:
# se corta, se recuerda el reinicio y la noticia en español se resuelve sin IA.
_IA_AGOTADA = {'agotado': False, 'reset': None}

MODELOS_IA = [
    'nvidia/nemotron-3-super-120b-a12b:free',
    'nvidia/nemotron-3-ultra-550b-a55b:free',
    'nvidia/nemotron-3.5-lightning:free',
    'dots-studio/dots-3-note-preview:free',
    'openrouter/free',
]


def _sentimiento(ticker, nots):
    """Sentimiento de las noticias del ticker.

    Los titulares se piden ya en español (RSS en español). El análisis fino
    (score + tema + resumen) lo intenta la IA de OpenRouter con un presupuesto
    de tiempo acotado; si la IA falla o responde vacío, se cae a un léxico
    financiero español local (instantáneo y determinista).
    Los titulares en español se muestran SIEMPRE que existan, con o sin IA."""
    es, en = (nots or {}).get('es') or [], (nots or {}).get('en') or []
    base = es or en
    if not base:
        return None
    txts = '\n'.join('- ' + t[:200] for t in (es[:4] or en[:4]))
    ia = _analisis_ia_noticias(ticker, txts, ya_es=bool(es))
    if ia is not None:
        r = float(ia.get('score', 0.0))
        titulos_es = ia.get('titulares_es') or es
        resumen = ia.get('resumen', '')
        tema = ia.get('tema', '')
        fuente_txt = 'IA OpenRouter · ' + str(ia.get('modelo_ia', 'modelo free'))
    elif es:
        r, coinc = _sentimiento_lexico_es(es)
        titulos_es = es
        tono = 'positivo' if r > 0.15 else 'negativo' if r < -0.15 else 'neutral'
        if _IA_AGOTADA['agotado']:
            motivo = 'cupo gratuito de IA agotado'
            if _IA_AGOTADA['reset']:
                motivo += ' hasta %s UTC' % _IA_AGOTADA['reset']
        else:
            motivo = 'IA no disponible ahora'
        resumen = ('Tono %s (%+.2f) medido con el léxico financiero español sobre %d coincidencias. '
                   'Titular de referencia: «%s»' % (tono, r, coinc, es[0][:150]))
        tema = 'noticias del ticker en español (sin IA: %s)' % motivo
        fuente_txt = 'léxico financiero español (%d coincidencias; %s)' % (coinc, motivo)
    else:
        r = _sentimiento_vader(en[:4])
        titulos_es = []
        resumen = ''
        tema = ''
        fuente_txt = 'VADER (titulares en inglés; sin IA disponible)'
    return {
        'sentimiento': ('positivo' if r > 0.15 else 'negativo' if r < -0.15 else 'neutral'),
        'score': round(float(r), 2),
        'titulares': en[:5] or es[:5],
        'titulares_es': [t for t in titulos_es[:5]],
        'idioma_titulares': 'es' if es else 'en',
        'num_noticias': len(base),
        'tema': tema,
        'resumen': resumen,
        'impacto': 'medio' if abs(r) > 0.3 else 'bajo',
        'fuente_analisis': fuente_txt,
        'modelo_ia': ia.get('modelo_ia') if ia is not None else None,
    }


# Léxico financiero en español: respaldo determinista cuando la IA no responde,
# para que el sentimiento siga siendo útil con titulares en español (VADER solo
# entiende inglés).
_LEX_POS = (
    'sube suben subio subieron alza alzas gana ganan gano ganancias beneficio beneficios '
    'supera superan supero record records maxima maximos maximo mejora mejoran mejores '
    'crece crecen crecimiento dispara disparan impulsa impulsan recupera recuperacion '
    'rebote rebota optimista positivo aprueba acuerdo expansion fuerte solidos solido '
    'eleva elevan previsiones contrato adjudica inversión inversion demanda duplica '
    'recompras dividendo subida subidas avanza avanzan acelera'
).split()
_LEX_NEG = (
    'cae caen cayo cayeron baja bajan bajo bajaron desploma desploman desplome hunde hundio '
    'pierde pierden perdida perdidas recorta recortan recorte decepciona decepcionante '
    'debil debiles preocupacion incertidumbre riesgo riesgos investigacion investiga multa '
    'sancion despido despidos cierra cierre quiebra suspende advertencia alerta rebaja '
    'rebajan minimos minimo correccion correcciones temor temores pesimista negativo '
    'problemas quejas retraso retrasos arancel aranceles inflacion huelga fraude '
    'demanda-judicial desplome recorta-previsiones venta-masiva cautela presiona presion '
    'crisis pierde-valor recorta-objetivo rebaja-objetivo'
).split()


def _sin_acentos(t):
    import unicodedata
    return ''.join(c for c in unicodedata.normalize('NFD', t.lower())
                   if unicodedata.category(c) != 'Mn')


def _sentimiento_lexico_es(titulares):
    """Score en [-1, 1] por conteo de palabras positivas/negativas del titular.
    Devuelve (score, número de coincidencias)."""
    pos = neg = 0
    for t in titulares[:5]:
        w = set(re.findall(r'[a-z\-]+', _sin_acentos(t)))
        pos += len(w.intersection(_LEX_POS))
        neg += len(w.intersection(_LEX_NEG))
    tot = pos + neg
    if tot == 0:
        return 0.0, 0
    return max(-1.0, min(1.0, (pos - neg) / tot)), tot


def _analisis_ia_noticias(ticker, textos, ya_es=False):
    """Un solo paso a la IA: sentimiento + tema + resumen (y traducción al
    español solo si los titulares vienen en inglés). Prueba varios modelos
    gratuitos en cadena y devuelve dict, o None si ninguno responde."""
    key = os.environ.get('OPENROUTER_KEY')
    if not key:
        env = REPO / '.env'
        if env.exists():
            for ln in env.read_text(encoding='utf-8', errors='replace').splitlines():
                ln = ln.strip()
                if ln and not ln.startswith('#') and '=' in ln:
                    k, v = ln.split('=', 1)
                    os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))
    key = os.environ.get('OPENROUTER_KEY')
    if not key:
        return None
    mensajes = [
        {'role': 'system', 'content': 'Eres un analista financiero experto que trabaja en español. Responde SOLO JSON válido.'},
        {'role': 'user', 'content':
            f'Estas son noticias recientes de {ticker}. Analiza la sensación de mercado y responde SOLO este JSON:\n'
            '{"score": <número -1.0 a 1.0>, "tema": "<tema principal en español>", '
            '"resumen": "<1 oración en español sobre qué dicen las noticias y cómo pueden mover la acción>", '
            '"titulares_es": ["<titular 1 traducido al español>", ...]}\n\n'
            + ('Los titulares YA están en español: devuélvelos en "titulares_es" tal cual, sin traducir.\n\n'
               if ya_es else
               'Traduce TODOS los titulares al español, conservando el sentido financiero.\n\n') + textos},
    ]
    def _pedir(modelo):
        payload = json.dumps({'model': modelo, 'messages': mensajes,
                              'temperature': 0.1, 'max_tokens': 600}).encode()
        req = urllib.request.Request(
            'https://openrouter.ai/api/v1/chat/completions', data=payload,
            headers={'Authorization': f'Bearer {key}', 'Content-Type': 'application/json',
                     'HTTP-Referer': 'https://github.com/aux3programacion-ai/agente-ia-financiero',
                     'X-Title': 'Agente IA Financiero'}, method='POST')
        try:
            with urllib.request.urlopen(req, timeout=IA_TIMEOUT_S) as resp:
                raw = json.loads(resp.read())['choices'][0]['message'].get('content') or ''
        except urllib.request.HTTPError as e:
            # 429 "free-models-per-day" = cupo gratuito diario agotado (50/día)
            if e.code == 429:
                cuerpo = ''
                try:
                    cuerpo = e.read().decode('utf-8', 'replace')
                except Exception:
                    pass
                if 'free-models-per-day' in cuerpo:
                    _IA_AGOTADA['agotado'] = True
                    try:
                        ms = int(e.headers.get('X-RateLimit-Reset'))
                        _IA_AGOTADA['reset'] = datetime.fromtimestamp(
                            ms / 1000, timezone.utc).strftime('%Y-%m-%d %H:%M')
                    except Exception:
                        _IA_AGOTADA['reset'] = None
            raise
        if not raw.strip():
            raise ValueError('respuesta vacía')
        m = re.search(r'\{[\s\S]*\}', raw.strip())
        if not m:
            raise ValueError('sin JSON en la respuesta')
        d = json.loads(m.group(0))
        if not isinstance(d.get('score'), (int, float)):
            raise ValueError('JSON sin campo score')
        if isinstance(d.get('titulares_es'), list):
            d['titulares_es'] = [str(x) for x in d['titulares_es']][:5]
        else:
            d['titulares_es'] = []
        d['modelo_ia'] = modelo
        return d

    # Se prueban los modelos en orden: basta con uno que conteste (así se gastan
    # pocas peticiones del cupo gratuito). Si el cupo diario ya se agotó, se corta.
    for modelo in MODELOS_IA:
        try:
            return _pedir(modelo)
        except Exception as e:
            print(f'[IA] {ticker}: {modelo} falló ({str(e)[:80]})', flush=True)
            if _IA_AGOTADA['agotado']:
                print('[IA] cupo diario de modelos gratuitos agotado; sigo sin IA', flush=True)
                break
    print(f'[IA] {ticker}: ningún modelo devolvió análisis; uso léxico español / VADER', flush=True)
    return None


def _sentimiento_vader(titulos):
    try:
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
        s = SentimentIntensityAnalyzer()
        return float(np.mean([s.polarity_scores(t)['compound'] for t in titulos if len(t) > 8] or [0]))
    except Exception:
        return 0.0


# ---------------------------------------------------------------- core

def analizar(ticker):
    ticker = _validar_ticker(ticker)
    h = _obtener_historial(ticker)

    tec_doc = _leer('analisis_tecnico.json')
    cal = _leer('calibracion.json')
    regimen = tec_doc.get('regimen_mercado', 'desconocido')
    tec = _tecnico(h, regimen)

    nots = _noticias(ticker)
    sent = _sentimiento(ticker, nots)
    # Etiqueta honesta de lo que REALMENTE se usó (no de lo que está configurado)
    if not sent:
        via_noticias = 'sin noticias disponibles'
    elif sent.get('modelo_ia'):
        via_noticias = 'IA OpenRouter (' + str(sent['modelo_ia']) + ')'
    elif sent.get('idioma_titulares') == 'es':
        via_noticias = 'léxico financiero español (IA no disponible)'
    else:
        via_noticias = 'VADER sobre titulares en inglés'

    # base técnica -> probabilidad previa
    p = 50.0
    razones = []
    if tec['tendencia'] == 'uptrend':
        p += 6; razones.append(('tecnico', 'Tendencia alcista (vs media50 %+.1f%%)' % tec['vs_media50'], +1))
    elif tec['tendencia'] == 'downtrend':
        p -= 6; razones.append(('tecnico', 'Tendencia bajista (vs media50 %.1f%%)' % tec['vs_media50'], -1))
    if tec['rsi'] < 30:
        p += 4; razones.append(('tecnico', 'RSI sobrevendido (%.0f): posible rebote' % tec['rsi'], +1))
    elif tec['rsi'] > 70:
        p -= 4; razones.append(('tecnico', 'RSI sobrecomprado (%.0f): riesgo de corrección' % tec['rsi'], -1))
    else:
        razones.append(('tecnico', 'RSI neutral (%.0f)' % tec['rsi'], 0))
    if tec['retorno_1m'] is not None:
        if tec['retorno_1m'] > 8:
            p += 3; razones.append(('tecnico', 'Momentum fuerte 1 mes (%+.1f%%)' % tec['retorno_1m'], +1))
        elif tec['retorno_1m'] < -8:
            p -= 3; razones.append(('tecnico', 'Corrección 1 mes (%+.1f%%)' % tec['retorno_1m'], -1))

    # sentimiento de noticias
    if sent:
        sc = sent['score']
        if abs(sc) >= 0.15:
            delta = (8 if sent['impacto'] == 'medio' else 5) * (1 if sc > 0 else -1)
            p += delta
            primer_titular = (sent.get('titulares_es') or sent.get('titulares') or [''])[0]
            razones.append(('noticias',
                'Sensación de noticias %s (score %+.2f): %s' % (sent['sentimiento'], sc,
                    primer_titular[:120]), +1 if sc > 0 else -1))

    # calibración del sistema: desplaza hacia su precisión histórica
    precision = cal.get('precision_global', 0.52)
    p = 50 + (p - 50) * max(precision, 0.45) / 0.55
    p = max(3.0, min(97.0, p))

    veredicto = ('MAYOR PROBABILIDAD DE SUBIR' if p >= 60 else
                 'MAYOR PROBABILIDAD DE BAJAR' if p <= 40 else
                 'RANGO / SIN SESGO CLARO')

    return {
        'ticker': ticker,
        'generado': datetime.now(timezone.utc).isoformat(),
        'modo': 'en-vivo-bajo-demanda',
        'regimen_mercado': regimen,
        'prob_subir': round(p, 1),
        'prob_bajar': round(100 - p, 1),
        'veredicto': veredicto,
        'confianza_ia': min(80, max(25, int(55 + abs(p - 50)))),
        'precision_historica': {
            'precision_direccion': round(precision * 100, 1) if isinstance(precision, (int, float)) else None,
            'evaluados': cal.get('total_evaluado', 0),
        },
        'sentimiento_noticias': sent,
        'senales_explicacion': [{'fuente': f, 'texto': txt, 'direccion': d} for f, txt, d in razones],
        'tecnico': tec,
        'calibracion_real': calibracion_real(DATOS),
        'fuente': 'yfinance-1d + noticias RSS en español + ' + via_noticias,
    }


@app.get('/api/analisis')
def api_analisis(ticker: str = Query(...)):
    return analizar(ticker)


@app.get('/')
def root():
    return {'ok': True, 'endpoints': ['/api/analisis?ticker=AAPL'], 'ts': datetime.now().isoformat()}


if __name__ == '__main__':
    try:
        puerto = int(os.environ.get('PORT') or 8010)
    except (ValueError, TypeError):
        puerto = 8010
    if puerto <= 0:
        puerto = 8010
    print(f'[API] Consulta on-demand en http://127.0.0.1:{puerto}', flush=True)
    uvicorn.run(app, host='127.0.0.1', port=puerto, log_level='warning')
