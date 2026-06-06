import json, os, re, html, urllib.request, time
from html.parser import HTMLParser

DATA_DIR = os.path.dirname(os.path.abspath(__file__))

VADER_LEXICON = {
    'positive': ['surge','soar','jump','rally','gain','bullish','upgrade','outperform','beat','growth',
                 'strong','record','profit','boom','bull','moon','rocket','buy','rip','green','up',
                 'positive','breakthrough','innovation','partnership','expansion','upside','upward',
                 'break out','all-time high','momentum','bull market','overweight','buyback','dividend',
                 'raised','upgrade','outlook','optimistic','confidence','recovery','opportunity'],
    'negative': ['dump','sell','bearish','crash','down','drop','decline','fall','slump','loss',
                 'debt','downgrade','underperform','bear','red','negative','risk','warning',
                 'volatile','uncertainty','cut','layoff','firing','investigation','lawsuit',
                 'regulatory','fine','penalty','downgraded','lowered','deficit','recession',
                 'slowdown','inflation','rate hike','sell-off','correction','bear market',
                 'underweight','shortsell','short interest','delist','bankrupt','fraud']
}

class MLStripper(HTMLParser):
    def __init__(self):
        super().__init__()
        self.reset()
        self.strict = False
        self.convert_charrefs = True
        self.text = []
    def handle_data(self, d):
        self.text.append(d)
    def get_data(self):
        return ''.join(self.text)

def strip_html(s):
    s = re.sub(r'<[^>]+>', ' ', s)
    s = html.unescape(s)
    return re.sub(r'\s+', ' ', s).strip()

def vader_score(text):
    text_lower = text.lower()
    words = re.findall(r'\w+', text_lower)
    pos_score = 0
    neg_score = 0
    pos_words = [w for w in words if w in VADER_LEXICON['positive']]
    neg_words = [w for w in words if w in VADER_LEXICON['negative']]
    pos_score = len(pos_words) * 0.3
    neg_score = len(neg_words) * 0.3
    for w in pos_words:
        pos_score += 0.1
    for w in neg_words:
        neg_score += 0.1
    negators = ['not', 'no', 'never', 'neither', 'nor', 'hardly', 'barely', 'unlikely']
    for w in words:
        if w in negators:
            pos_score, neg_score = neg_score, pos_score
            break
    amplifiers = ['very', 'extremely', 'highly', 'strongly', 'significantly', 'remarkably']
    for w in words:
        if w in amplifiers:
            pos_score *= 1.5
            neg_score *= 1.5
            break
    compound = (pos_score - neg_score) / (pos_score + neg_score + 1)
    compound = max(-1.0, min(1.0, compound))
    if compound >= 0.15:
        label = 'positivo'
    elif compound <= -0.15:
        label = 'negativo'
    else:
        label = 'neutral'
    return label, round(compound, 4)

def analyze_headlines(ticker, headlines):
    if not headlines:
        return None
    all_text = ' '.join(strip_html(h.get('titulo', '')) for h in headlines[:5])
    label, compound = vader_score(all_text)
    _, max_compound = vader_score(strip_html(headlines[0].get('titulo', '')))
    for h in headlines[1:3]:
        _, sc = vader_score(strip_html(h.get('titulo', '')))
        if abs(sc) > abs(max_compound):
            max_compound = sc
    return {
        'sentimiento': label,
        'score': compound,
        'max_score': max_compound,
        'impacto': 'alto' if abs(compound) > 0.5 else 'medio' if abs(compound) > 0.2 else 'bajo',
        'resumen': f'Sentimiento {label} (score:{compound:.2f}) basado en {len(headlines)} noticias'
    }

def batch_analyze(noticias_por_ticker):
    resultados = {}
    for ticker, data in noticias_por_ticker.items():
        if isinstance(data, dict):
            notis = data.get('noticias', [])
        elif isinstance(data, list):
            notis = data
        else:
            notis = []
        sent = analyze_headlines(ticker, notis)
        if sent:
            resultados[ticker] = sent
    return resultados

if __name__ == '__main__':
    NEWS_PATH = os.path.join(DATA_DIR, 'Datos', 'noticias_recientes.json')
    if os.path.exists(NEWS_PATH):
        news = json.load(open(NEWS_PATH))
        pt = news.get('por_ticker', {})
        result = batch_analyze(pt)
        print(f'[Sentimiento] Analizados {len(result)} tickers con VADER')
    else:
        print('[!] No news data found')
