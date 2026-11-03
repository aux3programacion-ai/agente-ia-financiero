#!/usr/bin/env python3
"""
noticias_es.py - Titulares de noticias en ESPANOL por ticker (sin IA).

El ciclo diario traia solo titulares en ingles (Yahoo/Google News en-US) y la
pagina publicada los mostraba en ingles. Este modulo consigue los titulares ya
en espanol, sin depender de traducir con IA:

  - Yahoo Finanzas ES  (es.finance.yahoo.com/rss/headline?s=TICKER)
  - Google Noticias ES (edicion Espana y es-419) buscando el nombre real de la
    empresa, no solo el simbolo.

Filtros para que "espanol" no signifique "otro tema": el titular debe mencionar
el simbolo o el nombre de la empresa, si entra por nombre debe traer contexto
financiero, y debe estar realmente en espanol (las fuentes "es" cuelan enlaces
en ingles). Ademas se puntua el tono con un lexico financiero espanol local, que
es determinista y no consume cupo de IA.

Uso:
    from noticias_es import enriquecer
    datos = enriquecer('AAPL')       # {'titulares_es': [...], 'nombre': ...,
                                     #  'sentimiento_es': {...} | None}
Sin librerias externas (yfinance solo si esta instalado, para el nombre).
"""
import json
import os
import re
import time
import unicodedata
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET

# Palabras genericas: no sirven para decidir si una noticia habla del ticker
_TOK_GENERICOS = {
    'inc', 'corp', 'ltd', 'plc', 'sa', 'nv', 'ag', 'group', 'grupo', 'company',
    'compania', 'holdings', 'holding', 'international', 'technologies', 'tech',
    'class', 'clase', 'stock', 'tokenized', 'coinbase', 'the', 'and',
    'health', 'healthcare', 'athletica', 'brands', 'motors', 'systems', 'energy',
}
_NOMBRES_CACHE = {}

# Palabras que delatan una noticia financiera (sin acentos, en minusculas)
_PAL_FINANZAS = (
    'accion acciones bolsa mercado cotizacion cotiza resultados beneficio beneficios '
    'ingresos prevision previsiones precio precios objetivo objetivos valor valores '
    'cae caen cayo sube suben subio alza recorta recortes rebaja rebajan '
    'trimestre dividendo dividendos ganancias perdidas ventas inversion '
    'inversores inversor riesgo multa investigacion demanda despido stock stocks share '
    'shares earnings revenue price market quarter rally drop drops gain gains loss '
    'losses analyst analysts downgrade upgrade guidance ipo etf'
).split()
_RE_FIN = re.compile(r'\b(' + '|'.join(_PAL_FINANZAS) + r')\b')
_PAL_ES = ('de', 'la', 'el', 'los', 'las', 'del', 'en', 'que', 'y', 'para', 'con',
           'por', 'se', 'al', 'un', 'una', 'su', 'sus', 'tras', 'ante', 'mas', 'o')
_RE_ES = re.compile(r'\b(' + '|'.join(_PAL_ES) + r')\b')

# Lexico financiero en espanol: respaldo determinista del tono
_LEX_POS = (
    'sube suben subio subieron alza alzas gana ganan gano ganancias beneficio beneficios '
    'supera superan supero record records maxima maximos maximo mejora mejoran mejores '
    'crece crecen crecimiento dispara disparan impulsa impulsan recupera recuperacion '
    'rebote rebota optimista positivo aprueba acuerdo expansion fuerte solidos solido '
    'eleva elevan previsiones contrato adjudica inversion demanda duplica '
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

TIMEOUT_RSS = 10


def _sin_acentos(t):
    return ''.join(c for c in unicodedata.normalize('NFD', str(t).lower())
                   if unicodedata.category(c) != 'Mn')


def _nombre_empresa(ticker):
    """Nombre corto de la empresa segun yfinance (opcional) para buscar/filtrar."""
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


def _rss(url, n):
    """Titulares de un feed RSS. Devuelve [] ante cualquier fallo de red."""
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=TIMEOUT_RSS) as resp:
            tree = ET.fromstring(resp.read())
        out = []
        for it in tree.iter('item'):
            titulo = (it.findtext('title') or '').strip()
            if titulo:
                out.append(titulo)
            if len(out) >= n:
                break
        return out
    except Exception:
        return []


def _relevante(titulo, ticker, nombre):
    """Espanol + relevante para el ticker.

    Las fuentes en espanol devuelven ruido generico para tickers poco cubiertos
    (p. ej. "Epic Games Store te regala un juego" para TOST), asi que se exige el
    simbolo, o el nombre de la empresa acompanado de contexto financiero (evita
    falsos positivos del tipo "Jelly Toast se lanza en India")."""
    t = _sin_acentos(titulo)
    sim = _sin_acentos(ticker).lower()
    if re.search(r'\b' + re.escape(sim) + r'\b', t):
        return True
    if not any(w in t for w in _tokens_nombre(nombre)):
        return False
    return bool(_RE_FIN.search(t))


def _es_espanol(titulo):
    return bool(_RE_ES.search(_sin_acentos(titulo)))


def _dedup(lista, n=5):
    vistos, out = set(), []
    for t in lista:
        k = _sin_acentos(t)[:70]
        if t and k not in vistos:
            vistos.add(k)
            out.append(t)
    return out[:n]


def titulares_es(ticker, max_n=5):
    """Titulares en espanol del ticker, ya filtrados por relevancia e idioma."""
    ticker = str(ticker).upper().strip()
    nombre = _nombre_empresa(ticker)

    def _filtra(lista):
        return [t for t in lista if _relevante(t, ticker, nombre) and _es_espanol(t)]

    es = _filtra(_rss(f'https://es.finance.yahoo.com/rss/headline?s={ticker}', 6))
    if len(es) < 3:
        consultas = []
        if nombre:
            consultas.append((f'"{nombre}" acciones', 'es', 'ES', 'ES:es'))
        consultas.append((f'{ticker} acciones', 'es-419', 'US', 'US:es-419'))
        for q, hl, gl, ceid in consultas:
            url = ('https://news.google.com/rss/search?q=' + urllib.parse.quote(q)
                   + f'&hl={hl}&gl={gl}&ceid={ceid}')
            es += _filtra(_rss(url, 6))
            if len(es) >= 3:
                break
    return _dedup(es, max_n)


def score_lexico(titulares):
    """Tono en [-1, 1] por conteo de palabras del lexico. (score, coincidencias)."""
    pos = neg = 0
    for t in (titulares or [])[:5]:
        w = set(re.findall(r'[a-z\-]+', _sin_acentos(t)))
        pos += len(w.intersection(_LEX_POS))
        neg += len(w.intersection(_LEX_NEG))
    tot = pos + neg
    if tot == 0:
        return 0.0, 0
    return max(-1.0, min(1.0, (pos - neg) / tot)), tot


def sentimiento_es(titulares):
    """Sentimiento (tono, score, impacto, tema, resumen) de titulares en espanol."""
    titulares = [t for t in (titulares or []) if t]
    if not titulares:
        return None
    r, coinc = score_lexico(titulares)
    tono = 'positivo' if r > 0.15 else 'negativo' if r < -0.15 else 'neutral'
    resumen = ('Tono %s (%+.2f) medido con el lexico financiero espanol sobre %d '
               'coincidencias. Titular de referencia: «%s»'
               % (tono, r, coinc, titulares[0][:150]))
    return {
        'sentimiento': tono,
        'score': round(float(r), 2),
        'impacto': 'medio' if abs(r) > 0.3 else 'bajo',
        'tema': 'noticias del ticker en espanol',
        'tema_principal': 'noticias del ticker en espanol',
        'resumen': resumen,
        'fuente_analisis': 'lexico financiero espanol (%d coincidencias)' % coinc,
        'titulares_es': titulares[:5],
        'idioma_titulares': 'es',
    }


def enriquecer(ticker, max_n=5):
    """Anade a la noticia del ticker sus titulares en espanol y el tono medido.

    Devuelve dict con 'titulares_es' (lista, puede ser vacia) y 'sentimiento_es'
    (dict o None). No lanza excepciones: si falla la red, devuelve vacio."""
    try:
        titulares = titulares_es(ticker, max_n=max_n)
    except Exception as e:
        print(f'  [Noticias ES {ticker}] {e}')
        titulares = []
    return {'titulares_es': titulares,
            'sentimiento_es': sentimiento_es(titulares)}


def main():
    import sys
    for t in sys.argv[1:] or ['AAPL']:
        d = enriquecer(t)
        print(f'--- {t.upper()} ({len(d["titulares_es"])} titulares en espanol)')
        for x in d['titulares_es']:
            print('   ', x)
        print('    sentimiento:', json.dumps(d['sentimiento_es'], ensure_ascii=False))
        time.sleep(0.2)


if __name__ == '__main__':
    main()
