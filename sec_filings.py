import json
import os
import sys
import urllib.request
import re
import time
import html
import xml.etree.ElementTree as ET

TICKERS_CORE = ['NVDA','MU','DELL','AVGO','DDOG','SMCI','SNOW','CRWD','NOW','TSM','ARM','OKTA','HPE','NTAP','CLS','AAPL','AMZN','GOOGL','META','MSFT','LLY','AMAT','LRCX','PANW','ORCL','HON','UBER','GE','COST','NEE']

DATA_DIR = os.environ.get('GITHUB_WORKSPACE', '.')
OUTPUT = os.path.join(DATA_DIR, 'Datos', 'sec_filings.json')
os.makedirs(os.path.join(DATA_DIR, 'Datos'), exist_ok=True)

def cargar_portafolio():
    try:
        ruta = os.path.join(DATA_DIR, 'Datos', 'portafolio_usuario.json')
        with open(ruta, 'r') as f:
            return json.load(f)
    except:
        return []

def tickers_a_procesar():
    portafolio = cargar_portafolio()
    combinados = list(TICKERS_CORE)
    for t in portafolio:
        t = t.strip().upper()
        if t and t not in combinados:
            combinados.append(t)
    return combinados[:30]

def extraer_ticker(entrada):
    m = re.search(r'\(([A-Z]{1,5})\)', entrada)
    if m:
        return m.group(1)
    m = re.search(r'\b([A-Z]{2,5})\b', entrada)
    if m:
        return m.group(1)
    return None

def generar_fallback(tickers):
    ahora = time.localtime()
    ano = ahora.tm_year
    mes = ahora.tm_mon
    dia = ahora.tm_mday
    hoy = (ano, mes, dia)

    ventanas = [
        (1, 15, 'Q4'),
        (4, 15, 'Q1'),
        (7, 15, 'Q2'),
        (10, 15, 'Q3')
    ]

    def fecha_a_num(m, d):
        return (ano if m >= mes else ano + 1, m, d)

    proximos = []
    recientes = []
    alertas = []

    for vm, vd, trimestre in ventanas:
        fn = fecha_a_num(vm, vd)
        diff = (fn[0] - ano) * 365 + (fn[1] - mes) * 30 + (fn[2] - dia)
        label = f"{trimestre} {fn[0]}"
        fecha_str = f"{fn[0]:04d}-{fn[1]:02d}-{fn[2]:02d}"
        if -30 <= diff <= 60:
            for t in tickers:
                proximos.append({
                    "ticker": t,
                    "tipo": "10-Q" if vm in (4, 7, 10) else "10-K",
                    "fecha": fecha_str,
                    "periodo": label,
                    "descripcion": f"Resultados {trimestre}"
                })
        if diff < -30:
            for t in tickers[:5]:
                recientes.append({
                    "ticker": t,
                    "tipo": "10-Q" if vm in (4, 7, 10) else "10-K",
                    "fecha": fecha_str,
                    "periodo": label,
                    "descripcion": f"Resultados {trimestre}"
                })

    return {
        "timestamp": time.strftime('%Y-%m-%dT%H:%M:%S'),
        "total_filings": len(proximos),
        "proximos": proximos,
        "recientes": recientes,
        "alertas": alertas
    }

def main():
    tickers = tickers_a_procesar()
    print(f"[!] Total tickers a monitorear: {len(tickers)}")

    try:
        url = 'https://www.sec.gov/cgi-bin/browse-edgar?action=getcurrent&type=10-K%2C10-Q&company=&dateb=&owner=include&start=0&count=100&output=atom'
        req = urllib.request.Request(url, headers={
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'application/atom+xml,application/xml'
        })
        with urllib.request.urlopen(req, timeout=30) as resp:
            xml_data = resp.read().decode('utf-8')

        ns = {'atom': 'http://www.w3.org/2005/Atom'}
        root = ET.fromstring(xml_data)

        filings_por_ticker = {t: [] for t in tickers}
        todas_entradas = []

        for entry in root.findall('.//atom:entry', ns):
            try:
                title_el = entry.find('atom:title', ns)
                updated_el = entry.find('atom:updated', ns)
                summary_el = entry.find('atom:summary', ns)
                category_el = entry.find('atom:category', ns)

                titulo = title_el.text if title_el is not None else ''
                fecha_str = updated_el.text[:10] if updated_el is not None and updated_el.text else ''
                descripcion = html.unescape(summary_el.text) if summary_el is not None and summary_el.text else ''
                form_type = category_el.get('term', '') if category_el is not None else ''

                texto_completo = f"{titulo} {descripcion}".upper()

                ticker_encontrado = None
                for t in tickers:
                    if t in texto_completo or f"({t})" in texto_completo:
                        ticker_encontrado = t
                        break

                if not ticker_encontrado:
                    extraido = extraer_ticker(titulo)
                    if extraido and extraido in tickers:
                        ticker_encontrado = extraido

                if ticker_encontrado and fecha_str:
                    tipo = '10-K' if '10-K' in form_type or '10-K' in texto_completo else '10-Q'
                    filing_entry = {
                        "ticker": ticker_encontrado,
                        "tipo": tipo,
                        "fecha": fecha_str,
                        "periodo": fecha_str[:7],
                        "descripcion": descripcion[:200]
                    }
                    filings_por_ticker[ticker_encontrado].append(filing_entry)
                    todas_entradas.append(filing_entry)
            except:
                continue

        ahora = time.localtime()
        hoy = time.mktime(ahora)
        un_dia = 86400

        proximos = []
        recientes = []
        alertas = []

        for fil in todas_entradas:
            try:
                fts = time.mktime(time.strptime(fil['fecha'], '%Y-%m-%d'))
                diff_dias = (fts - hoy) / un_dia
                if -7 <= diff_dias <= 60:
                    proximos.append(fil)
                elif diff_dias < -7:
                    recientes.append(fil)
            except:
                continue

        proximos.sort(key=lambda x: x['fecha'])
        recientes.sort(key=lambda x: x['fecha'], reverse=True)

        result = {
            "timestamp": time.strftime('%Y-%m-%dT%H:%M:%S'),
            "total_filings": len(todas_entradas),
            "proximos": proximos[:50],
            "recientes": recientes[:50],
            "alertas": alertas
        }

        with open(OUTPUT, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)

        print(f"[OK] SEC filings: {len(todas_entradas)} entradas encontradas, {len(proximos)} proximas")
        for p in proximos[:5]:
            print(f"  {p['ticker']} - {p['tipo']} - {p['fecha']}")

    except Exception as e:
        print(f"[!] Error fetching SEC filings: {e}")
        result = generar_fallback(tickers)
        with open(OUTPUT, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        print(f"[OK] Fallback usado: {len(result['proximos'])} eventos generados")

if __name__ == '__main__':
    main()
