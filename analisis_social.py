#!/usr/bin/env python3
import json, os, sys, urllib.request, re, time, html

DATA_DIR = os.path.dirname(os.path.abspath(__file__))
TICKERS_CORE = ['NVDA','MU','DELL','AVGO','DDOG','SMCI','SNOW','CRWD','NOW','TSM','ARM','OKTA','HPE','NTAP','CLS','AAPL','AMZN','GOOGL','META','MSFT','LLY','AMAT','LRCX','PANW','ORCL','HON','UBER','GE','COST','NEE']

PORTAFOLIO_PATH = os.path.join(DATA_DIR, 'Datos', 'portafolio_usuario.json')
portfolio = []
if os.path.exists(PORTAFOLIO_PATH):
    try:
        pf = json.load(open(PORTAFOLIO_PATH))
        if isinstance(pf, list):
            portfolio = [t.upper().strip() for t in pf if isinstance(t, str) and t.strip()]
    except:
        pass

seen = set()
tickers = []
for t in TICKERS_CORE + portfolio:
    if t not in seen:
        seen.add(t)
        tickers.append(t)
tickers = tickers[:60]

KW_POS = re.compile(r'\b(moon|rocket|boom|buy|bullish|rip)\b', re.I)
KW_NEG = re.compile(r'\b(dump|sell|bearish|crash|shit|down|drop)\b', re.I)

def parse_rss_titles(xml_text):
    titles = []
    for m in re.finditer(r'<title[^>]*>(.*?)</title>', xml_text, re.DOTALL | re.I):
        t = html.unescape(re.sub(r'<[^>]+>', '', m.group(1))).strip()
        if t:
            titles.append(t)
    return titles

def keyword_score(titles, ticker):
    pos = 0
    neg = 0
    for t in titles:
        if ticker.upper() not in t.upper():
            continue
        pos += len(KW_POS.findall(t))
        neg += len(KW_NEG.findall(t))
    return pos, neg

def fetch_titles(url):
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'})
        with urllib.request.urlopen(req, timeout=10) as resp:
            return parse_rss_titles(resp.read().decode('utf-8', errors='replace'))
    except:
        return []

resultados = {}
pos_count = 0
neg_count = 0
neu_count = 0

for i, t in enumerate(tickers):
    print(f"[{i+1}/{len(tickers)}] {t}...", end=' ')
    sys.stdout.flush()

    url1 = f'https://news.google.com/rss/search?q={t}+stock+reddit&hl=en-US&gl=US'
    url2 = f'https://news.google.com/rss/search?q={t}+stock+twits+wallstreetbets&hl=en-US&gl=US'

    t1 = fetch_titles(url1)
    time.sleep(0.3)
    t2 = fetch_titles(url2)

    all_titles = t1 + t2
    vol = min(len(all_titles), 20)
    pos, neg = keyword_score(all_titles, t)
    denom = pos + neg + 1
    score = (pos - neg) / denom

    if score > 0.1:
        sent = "positivo"
        pos_count += 1
    elif score < -0.1:
        sent = "negativo"
        neg_count += 1
    else:
        sent = "neutral"
        neu_count += 1

    fuentes = []
    if t1: fuentes.append("reddit")
    if t2: fuentes.append("google_news")
    if not fuentes: fuentes = ["n/a"]

    resultados[t] = {
        "score": round(score, 4),
        "menciones": vol,
        "sentimiento": sent,
        "fuentes": fuentes
    }

    print(f"score={score:.3f} menciones={vol} sentimiento={sent}")

output = {
    "timestamp": time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
    "tickers": resultados,
    "resumen": f"{pos_count} positivos, {neg_count} negativos, {neu_count} neutrales"
}

OUT_PATH = os.path.join(DATA_DIR, 'Datos', 'analisis_social.json')
with open(OUT_PATH, 'w', encoding='utf-8') as f:
    json.dump(output, f, indent=2, ensure_ascii=False)

print(f"\n[OK] Analisis social guardado en {OUT_PATH}")
print(f"Resumen: {output['resumen']}")
