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
        if not table: return []
        rows = table.find_all('tr')[1:]
        tickers = []
        for row in rows:
            cells = row.find_all('td')
            if len(cells) > col_ticker:
                t = cells[col_ticker].get_text(strip=True)
                t = re.sub(r'\s+', '', t)
                if remove_colons: t = t.replace(':', '')
                if suffix and not t.endswith(suffix): t = t + suffix
                if t and len(t) <= 10 and not re.match(r'^\d', t):
                    tickers.append(t)
        return tickers
    except:
        return []

def get_sp500():
    print("[SP500] Obteniendo componentes...")
    tickers = scrape_wikipedia_table(WIKI_SP500, col_ticker=0)
    if tickers: print(f"[OK] SP500: {len(tickers)} tickers")
    return tickers

def get_ipc_mexico():
    print("[IPC] Obteniendo componentes...")
    tickers = scrape_wikipedia_table(WIKI_IPC, col_ticker=0, suffix=".MX")
    if not tickers or len(tickers) < 10:
        fallback = ["AMXL.MX","WMT.MX","FEMSAUBD.MX","GFNORTEO.MX","GMEXICOB.MX",
                     "PE&OLES.MX","CEMEXCPO.MX","BBAJIOO.MX","ELEKTRA.MX","KIMBERA.MX",
                     "ASURB.MX","LAB.MX","AC.MX","MEGACPO.MX","ALFAA.MX",
                     "GCARSOA1.MX","TLEVISACPO.MX","OHLMEX.MX","CUERVO.MX","PASAB.MX"]
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
