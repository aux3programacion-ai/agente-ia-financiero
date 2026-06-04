#!/usr/bin/env python3
"""
google_finance_scraper.py - Extrae datos fundamentales, estadisticas clave,
acciones relacionadas, perfil corporativo y datos financieros desde Google Finance.
Fuente gratuita, sin API key. Usa requests + BeautifulSoup.
Salida: Datos/google_finance.json
"""
import json, os, sys, time, re
from datetime import datetime, timezone

try:
    import requests
    from bs4 import BeautifulSoup
except ImportError:
    print('[!] Se necesita requests y beautifulsoup4: pip install requests beautifulsoup4 lxml')
    sys.exit(1)

TICKERS_CORE = ['NVDA','MU','DELL','AVGO','DDOG','SMCI','SNOW','CRWD','NOW','TSM',
                'ARM','OKTA','HPE','NTAP','CLS','AAPL','AMZN','GOOGL','META','MSFT',
                'LLY','AMAT','LRCX','PANW','ORCL','HON','UBER','GE','COST','NEE']

DATA_DIR = os.environ.get('GITHUB_WORKSPACE', '.')
OUTPUT = os.path.join(DATA_DIR, 'Datos', 'google_finance.json')
os.makedirs(os.path.join(DATA_DIR, 'Datos'), exist_ok=True)

HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
    'Accept-Language': 'en-US,en;q=0.9',
}

from portafolio_utils import cargar_portafolio

def tickers_a_procesar():
    portafolio = cargar_portafolio(DATA_DIR)
    combinados = list(TICKERS_CORE)
    for t in portafolio:
        t = t.strip().upper()
        if t and t not in combinados:
            combinados.append(t)
    return combinados[:50]

def parse_number(text):
    text = text.strip().replace(',', '').replace('$', '').replace('%', '').replace('(', '').replace(')', '')
    if not text:
        return None
    if text.endswith('T'):
        try: return round(float(text[:-1]) * 1e12, 2)
        except: return text
    elif text.endswith('B'):
        try: return round(float(text[:-1]) * 1e9, 2)
        except: return text
    elif text.endswith('M'):
        try: return round(float(text[:-1]) * 1e6, 2)
        except: return text
    elif text.endswith('K'):
        try: return round(float(text[:-1]) * 1e3, 2)
        except: return text
    try: return float(text)
    except: return text

EXCHANGES = ['NASDAQ', 'NYSE']

def scrape_ticker(ticker):
    html = None
    used_url = None
    for exchange in EXCHANGES:
        try:
            url = f'https://www.google.com/finance/quote/{ticker}:{exchange}'
            resp = requests.get(url, headers=HEADERS, timeout=15)
            if resp.status_code == 200:
                html = resp.text
                used_url = url
                break
        except:
            continue

    if not html:
        return None

    result = {'ticker': ticker, 'url': used_url}
    soup = BeautifulSoup(html, 'lxml')

    # --- Key Statistics ---
    stat_pairs = soup.find_all('div', class_='KxsRFb')
    stats = {}
    seen_labels = set()
    for pair in stat_pairs:
        label_el = pair.find('div', class_='SwQK7')
        value_el = pair.find('div', class_='dO6ijd')
        if label_el and value_el:
            label = label_el.get_text(strip=True)
            if label in seen_labels:
                continue
            seen_labels.add(label)
            raw_value = value_el.get_text(strip=True)
            if label == 'Mkt. cap':
                stats['market_cap_str'] = raw_value
                stats['market_cap'] = parse_number(raw_value)
            elif label == 'P/E ratio': stats['pe_ratio'] = parse_number(raw_value)
            elif label == 'EPS': stats['eps'] = parse_number(raw_value.replace('$', ''))
            elif label == 'Beta': stats['beta'] = parse_number(raw_value)
            elif label == 'Dividend': stats['dividend_yield'] = parse_number(raw_value.replace('%', ''))
            elif label == 'Quarterly dividend': stats['quarterly_dividend'] = parse_number(raw_value.replace('$', ''))
            elif label == 'Ex-dividend date': stats['ex_dividend_date'] = raw_value
            elif label == '52-wk high': stats['high_52w'] = parse_number(raw_value.replace('$', ''))
            elif label == '52-wk low': stats['low_52w'] = parse_number(raw_value.replace('$', ''))
            elif label == 'Avg. vol.': stats['avg_volume'] = parse_number(raw_value)
            elif label == 'Volume': stats['volume'] = parse_number(raw_value)
            elif label == 'Open': stats['open'] = parse_number(raw_value.replace('$', ''))
            elif label == 'High': stats['high_day'] = parse_number(raw_value.replace('$', ''))
            elif label == 'Low': stats['low_day'] = parse_number(raw_value.replace('$', ''))
            elif label == 'Shares outstanding': stats['shares_outstanding'] = parse_number(raw_value)
            elif label == 'No. of employees':
                v = parse_number(raw_value)
                stats['employees'] = int(v) if isinstance(v, (int, float)) else raw_value

    result['stats'] = stats if stats else None

    # --- Price & Change (from stock section, NOT sector table) ---
    stock_section = soup.find('div', class_='zhtAvb')
    if stock_section:
        price_el = stock_section.find('span', {'jsname': 'Pdsbrc'})
        if price_el:
            try:
                result['price'] = float(price_el.get_text(strip=True).replace('$', '').replace(',', ''))
            except: pass

        pct_el = stock_section.find('span', {'jsname': 'vY9t3b'})
        if pct_el:
            result['change_pct'] = pct_el.get_text(strip=True)

        change_el = stock_section.find('span', {'jsname': 'xnruHf'})
        if change_el:
            try:
                result['change_abs'] = float(change_el.get_text(strip=True))
            except: pass

    # --- Related Stocks ---
    related_stocks = []
    for link in soup.find_all('a', class_='HPIOqe'):
        href = link.get('href', '')
        if '/quote/' not in href:
            continue
        ticker_match = re.search(r'/quote/([A-Z]{1,5}):', href)
        if not ticker_match:
            continue
        name_el = link.find('bdi', class_='KzWCNc')
        name = name_el.get_text(strip=True) if name_el else ''
        price_el = link.find('span', class_='SpkPOc')
        price_text = price_el.get_text(strip=True).replace('$', '') if price_el else ''
        change_el = link.find('span', class_='ymyBi')
        change_text = change_el.get_text(strip=True) if change_el else ''
        rel = {'ticker': ticker_match.group(1)}
        if name: rel['name'] = name
        if price_text: rel['price'] = price_text
        if change_text: rel['change'] = change_text
        related_stocks.append(rel)

    result['related_stocks'] = related_stocks if related_stocks else None

    # --- Earnings Data ---
    earnings = {}
    earn_sections = soup.find_all('div', class_='BlP9gf')
    for es in earn_sections:
        label_el = es.find('div', class_='ySQujc')
        value_el = es.find('div', class_='h1iDC')
        if label_el and value_el:
            label = label_el.get_text(strip=True)
            value = value_el.get_text(strip=True)
            if 'EPS / Est' in label:
                parts = value.split('/')
                if len(parts) >= 2:
                    earnings['eps_actual'] = parts[0].strip().replace('$', '')
                    earnings['eps_estimado'] = parts[1].strip().replace('$', '')
            elif 'Revenue / Est' in label:
                parts = value.split('/')
                if len(parts) >= 2:
                    earnings['revenue_actual'] = parts[0].strip().replace('$', '')
                    earnings['revenue_estimado'] = parts[1].strip().replace('$', '')
            elif 'Fiscal period' in label or 'Fiscal' in label:
                earnings['fiscal_period'] = value

    for be in soup.find_all('span', class_='ougHge'):
        text = be.get_text(strip=True)
        if '%' in text:
            earnings['surprise_pct'] = text
        elif 'beat' in text.lower() or 'miss' in text.lower():
            earnings['result'] = text

    # Find beat/miss label (next to surprise pct)
    for div in soup.find_all('div', class_='L1O8gf'):
        txt = div.get_text(strip=True)
        if 'beat' in txt.lower() or 'miss' in txt.lower():
            earnings['result'] = txt

    result['earnings'] = earnings if earnings else None

    # --- Company Info ---
    company_info = {}

    # Description from Profile section
    profile_div = soup.find('div', class_='PB7fm')
    if profile_div and 'Profile' in profile_div.get_text():
        desc_container = profile_div.find_next('div', class_='u3xNFb')
        if desc_container:
            desc_span = desc_container.find('span')
            if desc_span:
                company_info['description'] = desc_span.get_text(strip=True)[:500]

    # Details from "About" section
    about_div = soup.find(string=lambda s: s and 'About' in s and ticker in s if s else False)
    if not about_div:
        about_div = soup.find(string=lambda s: s and 'About' in s and 'Corp' in s if s else False)

    if about_div:
        parent = about_div.find_parent('div', class_='PB7fm')
        if parent:
            details_div = parent.find_next('div', class_='WdJZLd')
            if details_div:
                detail_pairs = details_div.find_all('span', class_='OspXqd')
                for label_el in detail_pairs:
                    label = label_el.get_text(strip=True)
                    value_el = label_el.find_next('span', class_='oJCxTc')
                    if value_el:
                        value = value_el.get_text(strip=True)
                        if label == 'CEO': company_info['ceo'] = value
                        elif label == 'Employees': company_info['employees'] = value
                        elif label == 'Founded': company_info['founded'] = value
                        elif label == 'Headquarters': company_info['headquarters'] = value
                        elif label == 'Sector': company_info['sector'] = value
                        elif label == 'Website': company_info['website'] = value

    result['company'] = company_info if company_info else None

    return result

def main():
    tickers = tickers_a_procesar()
    print(f'[Google Finance] Extrayendo datos de {len(tickers)} tickers...')

    resultados = {}
    errores = 0
    for i, t in enumerate(tickers):
        print(f'  [{i+1}/{len(tickers)}] {t}...', end=' ')
        data = scrape_ticker(t)
        if data:
            resultados[t] = data
            stats_count = len(data.get('stats', {})) if data.get('stats') else 0
            related_count = len(data.get('related_stocks', [])) if data.get('related_stocks') else 0
            print(f'OK (stats: {stats_count}, relacionados: {related_count})')
        else:
            print('ERROR')
            errores += 1
        time.sleep(0.8)

    salida = {
        'timestamp': datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ'),
        'total_tickers': len(tickers),
        'exitosos': len(resultados),
        'errores': errores,
        'tickers': resultados
    }

    with open(OUTPUT, 'w', encoding='utf-8') as f:
        json.dump(salida, f, indent=2, ensure_ascii=False)

    print(f'\n[OK] Google Finance: {len(resultados)}/{len(tickers)} tickers extraidos -> {OUTPUT}')
    stats_count = sum(1 for t in resultados if resultados[t].get('stats'))
    related_count = sum(1 for t in resultados if resultados[t].get('related_stocks'))
    earnings_count = sum(1 for t in resultados if resultados[t].get('earnings'))
    print(f'  Estadisticas: {stats_count} | Relacionados: {related_count} | Earnings: {earnings_count}')

if __name__ == '__main__':
    main()
