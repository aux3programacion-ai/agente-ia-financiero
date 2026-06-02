#!/usr/bin/env python3
"""
noticias_mercado.py - Obtiene noticias frescas por ticker via RSS gratuito
(Yahoo Finance RSS + Google News RSS) y las estructura para el AI.
Sin API key, sin librerias externas.
"""
import json, os, urllib.request, time, re
from datetime import datetime, timezone
from xml.etree import ElementTree

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

DATA_DIR = os.environ.get('GITHUB_WORKSPACE', '.')
OUT_DIR = os.path.join(DATA_DIR, 'Datos')
os.makedirs(OUT_DIR, exist_ok=True)

NEWS_PATH = os.path.join(OUT_DIR, 'noticias_recientes.json')
RESUMEN_PATH = os.path.join(OUT_DIR, 'News_Feed_Resumen.txt')
SEEN_PATH = os.path.join(OUT_DIR, 'noticias_vistas.json')

def fetch_rss(url, max_items=3):
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=10) as resp:
            raw = resp.read()
        root = ElementTree.fromstring(raw)
        ns = {'atom': 'http://www.w3.org/2005/Atom',
              'dc': 'http://purl.org/dc/elements/1.1/',
              'content': 'http://purl.org/rss/1.0/modules/content/'}
        items = []
        for entry in root.iter('item'):
            title = entry.findtext('title', '')
            link = entry.findtext('link', '')
            pubdate = entry.findtext('pubDate', '')
            desc = entry.findtext('description', '')
            if title:
                items.append({
                    'titulo': title.strip(),
                    'link': link.strip(),
                    'fecha': pubdate.strip(),
                    'descripcion': desc.strip()
                })
            if len(items) >= max_items:
                break
        return items
    except Exception as e:
        print(f'  [RSS] Error: {e}')
        return []

def extraer_sentimiento_ia(ticker, noticias):
    if not noticias:
        return None
    try:
        api_key = os.environ.get('OPENROUTER_KEY')
        if not api_key:
            return None
        textos = '\n'.join(f'- {n["titulo"]}' for n in noticias[:3])
        prompt = f'''Analiza estas noticias de {ticker} y responde SOLO este JSON:

Noticias:
{textos}

{{
  "sentimiento": "positivo|negativo|neutral",
  "score": 0.0,
  "impacto": "alto|medio|bajo",
  "tema_principal": "texto corto",
  "resumen": "1 oracion"
}}

Score: -1.0 (muy negativo) a +1.0 (muy positivo).'''
        payload = json.dumps({
            'model': 'openrouter/free',
            'messages': [
                {'role': 'system', 'content': 'Eres un analista de noticias financieras. Respondes SOLO JSON valido.'},
                {'role': 'user', 'content': prompt}
            ],
            'temperature': 0.1,
            'max_tokens': 300
        }).encode('utf-8')
        req = urllib.request.Request(
            'https://openrouter.ai/api/v1/chat/completions',
            data=payload,
            headers={
                'Authorization': f'Bearer {api_key}',
                'Content-Type': 'application/json',
                'HTTP-Referer': 'https://github.com/aux3programacion-ai/agente-ia-financiero',
                'X-Title': 'Agente IA Financiero'
            },
            method='POST')
        with urllib.request.urlopen(req, timeout=20) as resp:
            raw = json.loads(resp.read())['choices'][0]['message']['content']
        raw = raw.strip()
        if raw.startswith('```'):
            m = re.search(r'```(?:json)?\s*([\s\S]*?)```', raw)
            if m: raw = m.group(1).strip()
        inicio = raw.find('{'); fin = raw.rfind('}')
        if inicio != -1 and fin != -1:
            raw = raw[inicio:fin+1]
        return json.loads(raw)
    except Exception as e:
        print(f'  [Sentimiento IA] Error: {e}')
        return None

def main():
    print('[Noticias] Obteniendo noticias frescas de mercado...')

    # Cargar noticias ya vistas para deduplicar
    vistas = set()
    if os.path.exists(SEEN_PATH):
        try:
            vistas = set(json.load(open(SEEN_PATH)))
        except: pass

    noticias_por_ticker = {}
    total_nuevas = 0

    for t in TICKERS:
        yahoo_url = f'https://finance.yahoo.com/rss/headline?s={t}'
        items = fetch_rss(yahoo_url, max_items=3)
        if not items:
            time.sleep(0.3)
            items = fetch_rss(f'https://news.google.com/rss/search?q={t}+stock&hl=en-US&gl=US', max_items=2)

        nuevas = []
        for item in items:
            tid = item['titulo'][:80]
            if tid not in vistas:
                vistas.add(tid)
                nuevas.append(item)
                total_nuevas += 1

        noticias_por_ticker[t] = nuevas if nuevas else items[:1]

        # Analisis de sentimiento IA para este ticker (si hay noticias nuevas)
        sent = extraer_sentimiento_ia(t, noticias_por_ticker[t])
        noticias_por_ticker[t] = {
            'noticias': noticias_por_ticker[t],
            'sentimiento': sent
        }
        time.sleep(0.2)

    # Construir estructura de salida por sector
    por_sector = {}
    for sector, tickers in SECTOR_MAP.items():
        notis_sector = []
        for t in tickers:
            data = noticias_por_ticker.get(t, {'noticias': [], 'sentimiento': None})
            for n in data['noticias']:
                notis_sector.append({'ticker': t, **n})
        por_sector[sector] = notis_sector

    output = {
        'timestamp': datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ'),
        'total_tickers': len(TICKERS),
        'total_noticias_nuevas': total_nuevas,
        'por_ticker': noticias_por_ticker,
        'por_sector': por_sector
    }

    with open(NEWS_PATH, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    # Guardar set de vistas
    with open(SEEN_PATH, 'w', encoding='utf-8') as f:
        json.dump(list(vistas), f, indent=2, ensure_ascii=False)

    # Actualizar resumen TXT legible
    ahora = datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')
    lines = [f'NEWS FEED - {ahora}', f'{total_nuevas} noticias nuevas | {len(TICKERS)} tickers escaneados', '']
    for t in TICKERS:
        data = noticias_por_ticker.get(t, {'noticias': [], 'sentimiento': None})
        notis = data['noticias']
        sent = data.get('sentimiento')
        if notis:
            for n in notis[:2]:
                lines.append(f'[{t}] {n["titulo"][:120]}')
        if sent:
            lines.append(f'       Sentimiento: {sent.get("sentimiento","?")} (score:{sent.get("score","?")})')
    lines.append('')
    lines.append('--- Generado por Agente IA Financiero | No constituye asesoramiento ---')

    with open(RESUMEN_PATH, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))

    print(f'[OK] {total_nuevas} noticias nuevas para {len(TICKERS)} tickers')
    print(f'[OK] Noticias guardadas en noticias_recientes.json')

if __name__ == '__main__':
    main()
