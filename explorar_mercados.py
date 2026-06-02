import requests, json, re, sys, time
from bs4 import BeautifulSoup

UNIV_FILE = "Datos/univ_global.json"
TICKER_30_FILE = "Datos/tickers_30.txt"

WIKI_SP500 = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
WIKI_IPC = "https://en.wikipedia.org/wiki/Indice_de_Precios_y_Cotizaciones"
WIKI_DAX = "https://en.wikipedia.org/wiki/DAX"
WIKI_NIKKEI = "https://en.wikipedia.org/wiki/Nikkei_225"
WIKI_FTSE = "https://en.wikipedia.org/wiki/FTSE_100_Index"
WIKI_HANGSENG = "https://en.wikipedia.org/wiki/Hang_Seng_Index"

def scrape_wikipedia_table(url, col_ticker=0, suffix="", remove_colons=True):
    try:
        r = requests.get(url, timeout=15)
        soup = BeautifulSoup(r.text, 'html.parser')
        table = soup.find('table', {'class': 'wikitable'}) or soup.find('table', {'class': 'wikitable sortable'})
        if not table:
            table = soup.find_all('table', {'class': 'wikitable'})
            if table: table = table[0]
        if not table:
            tables = soup.find_all('table')
            for t in tables:
                if t.get('class') and ('wikitable' in ' '.join(t.get('class', []))):
                    table = t; break
                if t.find('th') and ('Ticker' in t.get_text() or 'Symbol' in t.get_text()):
                    table = t; break
        if not table: return []
        rows = table.find_all('tr')[1:]
        tickers = []
        header_cells = table.find_all('tr')[0].find_all('th')
        header_texts = [h.get_text(strip=True).lower() for h in header_cells]
        ticker_col = col_ticker
        for idx, ht in enumerate(header_texts):
            if any(k in ht for k in ['ticker', 'symbol', 'code', 'empresa', 'company']):
                ticker_col = idx; break
        for row in rows:
            cells = row.find_all('td')
            if len(cells) > ticker_col:
                t = cells[ticker_col].get_text(strip=True)
                t = re.sub(r'\s+', '', t)
                if remove_colons: t = t.replace(':', '')
                if suffix and not t.endswith(suffix): t = t + suffix
                if t and len(t) <= 12 and not re.match(r'^\d', t[:1]):
                    tickers.append(t)
        return tickers
    except Exception as e:
        print(f'  [!] Error scraping: {e}')
        return []

def get_sp500():
    print("[SP500] Obteniendo componentes...")
    tickers = scrape_wikipedia_table(WIKI_SP500, col_ticker=0)
    if not tickers or len(tickers) < 100:
        # Try alternate approach: read from the table by finding "Symbol" column
        try:
            r = requests.get(WIKI_SP500, timeout=15)
            soup = BeautifulSoup(r.text, 'html.parser')
            tbl = None
            for t in soup.find_all('table'):
                if 'wikitable' in str(t.get('class', [])):
                    tbl = t; break
                if t.find('th') and 'Symbol' in t.get_text():
                    tbl = t; break
            if tbl:
                rows = tbl.find_all('tr')[1:]
                tickers = []
                for row in rows:
                    cells = row.find_all('td')
                    for c in cells:
                        a = c.find('a')
                        if a and a.get('href') and len(a.text.strip()) <= 5:
                            t = a.text.strip()
                            if t and re.match(r'^[A-Z]', t):
                                tickers.append(t)
                                break
                tickers = list(dict.fromkeys(tickers))
        except:
            pass
    if tickers and len(tickers) >= 100:
        print(f"[OK] SP500: {len(tickers)} tickers")
    else:
        fallback = ["AAPL","MSFT","AMZN","GOOGL","META","NVDA","BRK.B","JPM","V","UNH",
                     "XOM","MA","PG","JNJ","HD","MRK","COST","AVGO","CVX","ABBV",
                     "CRM","NFLX","KO","PEP","WMT","BAC","TMO","ACN","DIS","MCD",
                     "CSCO","ABT","TXN","VZ","ADBE","LIN","CMCSA","NKE","NEE","PM",
                     "AMD","IBM","GE","QCOM","MDT","HON","AMT","RTX","LOW","SPGI",
                     "INTU","UPS","CAT","BLK","PLD","T","ELV","CI","GS","MS",
                     "DE","SYK","LMT","MDLZ","BA","SCHW","AMAT","CB","AXP","ADI",
                     "DUK","GILD","TMUS","SO","ISRG","MMM","MU","BKNG","LRCX","CL",
                     "APD","REGN","ATVI","FISV","EBAY","GM","EW","F","IQV","STZ",
                     "C","FCX","HUM"]
        tickers = fallback
        print(f"[FALLBACK] SP500: {len(tickers)} tickers")
    return tickers

def get_ipc_mexico():
    print("[IPC] Obteniendo componentes...")
    tickers = scrape_wikipedia_table(WIKI_IPC, col_ticker=0, suffix=".MX")
    if not tickers or len(tickers) < 10:
        fallback = ["WMT.MX","FEMSAUB.MX","GFNORTEO.MX","AMX.MX","GMEXICOB.MX",
                     "CEMEXCPO.MX","BBAJIO.MX","KIMBERA.MX",
                     "ASURB.MX","AC.MX","MEGACPO.MX","ALFA.MX",
                     "TLEVISACPO.MX","CUERVO.MX","PASAB.MX",
                     "GENTERA.MX","ORBIA.MX","PINFRA.MX","BIMBOA.MX","GCARSOA1.MX"]
        tickers = fallback
    print(f"[OK] IPC Mexico: {len(tickers)} tickers")
    return tickers

def get_dax():
    print("[DAX] Obteniendo componentes...")
    tickers = scrape_wikipedia_table(WIKI_DAX, col_ticker=0, suffix=".DE")
    if tickers: print(f"[OK] DAX: {len(tickers)} tickers")
    else:
        fallback = ["SAP.DE","SIEM.DE","ALV.DE","DBK.DE","DTE.DE","BMW.DE","VOW3.DE","BAYN.DE","ADS.DE","MRK.DE"]
        tickers = fallback
        print(f"[FALLBACK] DAX: {len(tickers)} tickers")
    return tickers

def get_nikkei():
    print("[Nikkei] Obteniendo componentes...")
    tickers = scrape_wikipedia_table(WIKI_NIKKEI, col_ticker=0, suffix=".T")
    if tickers: print(f"[OK] Nikkei: {len(tickers)} tickers")
    else:
        fallback = ["9984.T","9432.T","7203.T","6758.T","6861.T","9983.T","8035.T","9433.T","6098.T","4502.T"]
        tickers = fallback
        print(f"[FALLBACK] Nikkei: {len(tickers)} tickers")
    return tickers

def get_ftse():
    print("[FTSE] Obteniendo componentes...")
    tickers = scrape_wikipedia_table(WIKI_FTSE, col_ticker=0, suffix=".L")
    if tickers: print(f"[OK] FTSE: {len(tickers)} tickers")
    else:
        fallback = ["HSBA.L","SHEL.L","AZN.L","GSK.L","RIO.L","BP.L","ULVR.L","REL.L","LSEG.L","LLOY.L"]
        tickers = fallback
        print(f"[FALLBACK] FTSE: {len(tickers)} tickers")
    return tickers

def get_hangseng():
    print("[HangSeng] Obteniendo componentes...")
    tickers = scrape_wikipedia_table(WIKI_HANGSENG, col_ticker=0, suffix=".HK")
    if tickers: print(f"[OK] HangSeng: {len(tickers)} tickers")
    else:
        fallback = ["0700.HK","9988.HK","0941.HK","1398.HK","2318.HK","1299.HK","0005.HK","0883.HK","2388.HK","0016.HK"]
        tickers = fallback
        print(f"[FALLBACK] HangSeng: {len(tickers)} tickers")
    return tickers

def get_core_30():
    with open(TICKER_30_FILE) as f:
        return [l.strip() for l in f if l.strip() and not l.startswith('#')]

def main():
    import os
    os.makedirs("Datos", exist_ok=True)
    try:
        core = get_core_30()
        print(f"[Core] 30 tickers base")
    except:
        core = []
        print("[Core] No file, fresh universe")

    univ = {
        "sp500": get_sp500(),
        "ipc_mexico": get_ipc_mexico(),
        "dax": get_dax(),
        "nikkei": get_nikkei(),
        "ftse": get_ftse(),
        "hangseng": get_hangseng(),
        "core_30": core
    }
    all_tickers = []
    for mercado in univ:
        all_tickers.extend(univ[mercado])
    all_tickers = list(dict.fromkeys([t.upper() for t in all_tickers if t]))
    univ["todos"] = all_tickers
    univ["total"] = len(all_tickers)
    with open(UNIV_FILE, "w") as f:
        json.dump(univ, f, indent=2)
    print(f"\n[UNIVERSO GLOBAL] {len(all_tickers)} tickers unicos de {len([k for k in univ if k != 'todos' and k != 'total'])} mercados")

if __name__ == "__main__":
    main()
