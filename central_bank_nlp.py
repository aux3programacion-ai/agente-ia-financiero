import json, os, sys, time, re, urllib.request, html
from collections import defaultdict

DATA_DIR = os.environ.get('GITHUB_WORKSPACE', '.')
OUTPUT_SPEECH = os.path.join(DATA_DIR, 'Datos', 'central_bank_nlp.json')
OUTPUT_EARNINGS = os.path.join(DATA_DIR, 'Datos', 'earnings_call_nlp.json')
os.makedirs(os.path.join(DATA_DIR, 'Datos'), exist_ok=True)

FINANCIAL_SENTIMENT_LEXICON = {
    'hawkish': -0.4, 'dovish': 0.4, 'tighten': -0.3, 'ease': 0.3,
    'inflation': -0.2, 'growth': 0.3, 'uncertainty': -0.3, 'confidence': 0.3,
    'slowdown': -0.4, 'expansion': 0.4, 'recession': -0.5, 'recovery': 0.4,
    'rate hike': -0.4, 'rate cut': 0.4, 'taper': -0.3, 'stimulus': 0.3,
    'overhang': -0.2, 'headwind': -0.3, 'tailwind': 0.3, 'momentum': 0.2,
    'beat': 0.3, 'miss': -0.3, 'outperform': 0.3, 'underperform': -0.3,
    'guidance': 0.1, 'outlook': 0.1, 'cautious': -0.2, 'optimistic': 0.3,
    'strong': 0.2, 'weak': -0.2, 'record': 0.2, 'decline': -0.2,
    'positive': 0.2, 'negative': -0.2, 'challenging': -0.2, 'favorable': 0.2
}

def simple_vader(text):
    words = re.findall(r'\w+', text.lower())
    score = 0.0
    matches = 0
    negators = {'not', 'no', 'never', 'neither', 'nor', 'hardly', 'barely'}
    amplifiers = {'very', 'extremely', 'highly', 'strongly', 'significantly', 'markedly', 'substantially'}
    for i, w in enumerate(words):
        mult = 1.0
        # Check for negators in previous 3 words
        for j in range(max(0, i-3), i):
            if words[j] in negators:
                mult = -1.0
                break
        # Check amplifiers
        for j in range(max(0, i-2), i):
            if words[j] in amplifiers:
                mult *= 1.5
                break
        if w in FINANCIAL_SENTIMENT_LEXICON:
            score += FINANCIAL_SENTIMENT_LEXICON[w] * mult
            matches += 1
    if matches > 0:
        score = score / matches
    return max(-1, min(1, score))

def fetch_fed_speeches():
    """Get recent Fed speeches from Richmond Fed or FRED."""
    texts = []
    try:
        url = 'https://www.federalreserve.gov/newsevents/pressreleases.htm'
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=15) as r:
            html_content = r.read().decode('utf-8', errors='replace')
        # Extract speech titles and teasers
        items = re.findall(r'<a[^>]*href="([^"]*press[^"]*)"[^>]*>([^<]+)</a>', html_content, re.I)
        for href, title in items[:5]:
            full_url = 'https://www.federalreserve.gov' + href if href.startswith('/') else href
            try:
                req2 = urllib.request.Request(full_url, headers={'User-Agent': 'Mozilla/5.0'})
                with urllib.request.urlopen(req2, timeout=10) as r2:
                    content = r2.read().decode('utf-8', errors='replace')
                # Extract paragraph text
                paras = re.findall(r'<p>([^<]+)</p>', content, re.I)
                text = ' '.join(html.unescape(p) for p in paras)
                if len(text) > 100:
                    texts.append({'title': html.unescape(title), 'text': text[:3000]})
            except:
                pass
    except:
        pass
    return texts

def fetch_earnings_calls():
    """Get recent earnings call transcripts from SeekingAlpha or Motley Fool."""
    calls = []
    tickers = ['NVDA','AAPL','MSFT','AMZN','GOOGL','META','TSLA']
    for t in tickers:
        try:
            url = f'https://seekingalpha.com/symbol/{t}/earnings/transcripts'
            req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
            with urllib.request.urlopen(req, timeout=10) as r:
                html_content = r.read().decode('utf-8', errors='replace')
            # Extract transcript snippets
            snippets = re.findall(r'<p[^>]*>([^<]+)</p>', html_content, re.I)
            text = ' '.join(html.unescape(s) for s in snippets[:50])
            if len(text) > 200:
                calls.append({'ticker': t, 'text': text[:3000]})
        except:
            pass
    return calls

def main():
    print('[Central Bank NLP] Analizando discursos de la Fed...')
    speeches = fetch_fed_speeches()
    speech_results = []
    for s in speeches:
        score = simple_vader(s['text'])
        speech_results.append({
            'title': s['title'],
            'sentiment_score': round(score, 4),
            'sentiment': 'dovish' if score > 0.1 else 'hawkish' if score < -0.1 else 'neutral',
            'text_snippet': s['text'][:200]
        })
    
    fed_output = {
        'timestamp': time.strftime('%Y-%m-%dT%H:%M:%S'),
        'speeches': speech_results,
        'aggregate_sentiment': round(np.mean([s['sentiment_score'] for s in speech_results]), 4) if speech_results else 0,
        'n_speeches': len(speech_results)
    }
    with open(OUTPUT_SPEECH, 'w', encoding='utf-8') as f:
        json.dump(fed_output, f, indent=2)
    print(f'  {len(speech_results)} speeches analizados, sentimiento={fed_output.get("aggregate_sentiment",0):.2f}')
    
    print('[Earnings Call NLP] Analizando transcripciones...')
    calls = fetch_earnings_calls()
    call_results = []
    for c in calls:
        score = simple_vader(c['text'])
        call_results.append({
            'ticker': c['ticker'],
            'sentiment_score': round(score, 4),
            'sentiment': 'positive' if score > 0.1 else 'negative' if score < -0.1 else 'neutral'
        })
    
    earnings_output = {
        'timestamp': time.strftime('%Y-%m-%dT%H:%M:%S'),
        'calls': call_results,
        'n_calls': len(call_results)
    }
    with open(OUTPUT_EARNINGS, 'w', encoding='utf-8') as f:
        json.dump(earnings_output, f, indent=2)
    print(f'  {len(call_results)} earnings calls analizados')
    if call_results:
        for cr in call_results:
            print(f'    {cr["ticker"]}: {cr["sentiment"]} ({cr["sentiment_score"]:.2f})')

if __name__ == '__main__':
    import numpy as np
    main()
