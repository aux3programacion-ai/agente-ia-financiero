#!/usr/bin/env python3
"""
tikr_scraper.py - Extrae datos financieros, ratios, estimaciones,
ownership, noticias y mas desde TIKR Terminal via Playwright.
Requiere: playwright (pip install playwright && playwright install chromium)
Credenciales: archivo .env (TIKR_EMAIL, TIKR_PASSWORD)
Salida: Datos/tikr_data.json
"""
import json, os, sys, re, time
from datetime import datetime, timezone

try:
    import requests as http_req
except ImportError:
    print('[!] Necesitas requests: pip install requests')
    sys.exit(1)

try:
    from playwright.sync_api import sync_playwright
except ImportError:
    print('[!] Necesitas playwright: pip install playwright && playwright install chromium')
    sys.exit(1)

TICKERS_CORE = ['NVDA','MU','DELL','AVGO','DDOG','SMCI','SNOW','CRWD','NOW','TSM',
                'ARM','OKTA','HPE','NTAP','CLS','AAPL','AMZN','GOOGL','META','MSFT',
                'LLY','AMAT','LRCX','PANW','ORCL','HON','UBER','GE','COST','NEE']

DATA_DIR = os.environ.get('GITHUB_WORKSPACE', '.')
OUTPUT = os.path.join(DATA_DIR, 'Datos', 'tikr_data.json')
os.makedirs(os.path.join(DATA_DIR, 'Datos'), exist_ok=True)
STATE_FILE = os.path.join(DATA_DIR, 'Datos', 'tikr_state.json')

NAV_ITEMS = {'Terminal TIKR','apps','Noticias del mercado','Generacion de ideas','playlist_add',
    'Mis listas de seguimiento','Seguimiento de gurus de la inversion','Cribador global',
    'search','Busqueda de transcripciones','Valuation Model Hub','Calendario de eventos',
    'Analisis fundamental','business','Resumen de la empresa','radio','Noticias de la compania',
    'account_balance','Informacion financiera detallada','attach_money','Valoracion',
    'Company Model','timeline','Estimaciones de analistas','record_voice_over',
    'Transcripciones de conferencias','Presentaciones publicas','Accionariado',
    'Desbloquea datos premium y acceso global','settings','Add to Watchlist','USD',
    'Mostrar tabla de precios','Ocultar grafico','Descripcion General','Noticias','Finanzas',
    'Model','Estimaciones','Transcripciones','Presentaciones','Show Chart','keyboard_arrow_down',
    'Dataset','Period','Currency','Display Units','LTM','Anual','Trimestral','Semestral',
    'Ano Del Calendario','Export','Descargar','Consejo de TIKR','Powered by TIKR.com',
    'Price','Download','Lista de seguimiento activa: First Watchlist',
    '(c) 2019 - 2026, TIKR Inc.','Condiciones','Intimidad','Filter By Investor Type',
    'Transacciones De Insiders','Accionistas','Precio utilizado para valorar las acciones en posesion',
    'Ultimo cierre','Fecha de tenencia','Ticker','Ultimo','Cambio','% Cambio',
    'Nombre del inversor','Value (MM)','% De acciones en circulacion en posesion',
    '# De acciones en posesion','Chg in Shares Held','% Chg of Shares Held',
    "% of Firm's Portfolio",'Tipo de inversor',
    'Fecha de tenencia.1','% de tenencia en cartera',
    'Cartera total reportada','Posiciones largas','Inversores',
    'Nuevo','Aumentaron','Redujeron','Vendieron','Bienvenido a TIKR',
    'Mas Noticias','Filter News By Topic','Dataset','Main TIKR Data',
    'Periodo','Unidades de visualizacion','Ano del calendario'}

def load_credentials():
    env_path = os.path.join(DATA_DIR, '.env')
    email = os.environ.get('TIKR_EMAIL', '')
    password = os.environ.get('TIKR_PASSWORD', '')
    if not email and os.path.exists(env_path):
        for line in open(env_path, encoding='utf-8'):
            if line.startswith('TIKR_EMAIL='):
                email = line.strip().split('=', 1)[1]
            elif line.startswith('TIKR_PASSWORD='):
                password = line.strip().split('=', 1)[1]
    return email, password

def clean_text(body):
    lines = [l.strip() for l in body.split('\n') if l.strip()]
    seen = set()
    result = []
    for l in lines:
        nl = re.sub(r'\s+', ' ', l).strip()
        if nl and nl not in seen and len(nl) > 1:
            nl_clean = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f]', '', nl)
            if nl_clean and nl_clean not in NAV_ITEMS:
                seen.add(nl_clean)
                result.append(nl_clean)
    return result

class TIKRScraper:
    BASE = 'https://app.tikr.com'

    def __init__(self, email, password, headless=True):
        self.email = email
        self.password = password
        self.headless = headless
        self.browser = None
        self.context = None
        self.page = None
        self.playwright = None
        self.data = {}
        self._cid_cache = {}
        self.jwt_token = None

    def start(self):
        self.playwright = sync_playwright().start()
        self.browser = self.playwright.chromium.launch(headless=self.headless, args=[
            '--no-sandbox', '--disable-setuid-sandbox',
            '--disable-dev-shm-usage', '--disable-web-security'
        ])
        if os.path.exists(STATE_FILE):
            self.context = self.browser.new_context(storage_state=STATE_FILE, viewport={'width': 1920, 'height': 1080})
        else:
            self.context = self.browser.new_context(viewport={'width': 1920, 'height': 1080})
        self.page = self.context.new_page()
        self.page.set_default_timeout(30000)

    def close(self):
        try:
            if self.context:
                self.context.storage_state(path=STATE_FILE)
        except: pass
        try:
            if self.browser: self.browser.close()
        except: pass
        try: self.playwright.stop()
        except: pass

    def login(self):
        if not self.email or not self.password:
            print('[!] TIKR_EMAIL y TIKR_PASSWORD requeridos en .env')
            return False
        try:
            if os.path.exists(STATE_FILE):
                self.page.goto(f'{self.BASE}/markets', wait_until='domcontentloaded')
                time.sleep(2)
                body = self.page.inner_text('body')
                if 'Terminal TIKR' in body or 'Analisis fundamental' in body:
                    print('[OK] Sesion TIKR recuperada (state)')
                    self._extract_jwt()
                    return True
            self.page.goto(f'{self.BASE}/login', wait_until='domcontentloaded')
            time.sleep(2)
            if 'markets' in self.page.url:
                print('[OK] Sesion TIKR activa')
                return True
            try:
                self.page.wait_for_selector('#input-13', timeout=8000)
            except:
                if 'markets' in self.page.url or 'Terminal' in self.page.inner_text('body'):
                    print('[OK] Sesion TIKR activa')
                    return True
                raise
            self.page.fill('#input-13', self.email)
            self.page.fill('#input-16', self.password)
            self.page.click('button:has-text("Iniciar")')
            time.sleep(2)
            self.page.goto(f'{self.BASE}/markets', wait_until='domcontentloaded')
            time.sleep(2)
            for btn_text in ['Maybe Later']:
                try:
                    btn = self.page.locator(f'text={btn_text}')
                    if btn.count() > 0:
                        btn.click()
                        time.sleep(0.5)
                except: pass
            print('[OK] Login TIKR exitoso')
            self._extract_jwt()
            self.context.storage_state(path=STATE_FILE)
            return True
        except Exception as e:
            print(f'[!] Error en login TIKR: {e}')
            return False

    def search_ticker(self, ticker):
        """Busca ticker via v-select + Enter, retorna (cid, tid)."""
        try:
            self.page.goto(f'{self.BASE}/markets', wait_until='domcontentloaded')
            time.sleep(1)
            sel = self.page.locator('.v-select__selections').first
            sel.click()
            time.sleep(0.3)
            inp = self.page.locator('.v-select__selections input').first
            inp.fill('')
            inp.type(ticker, delay=20)
            time.sleep(1)
            texts = self.page.locator('.v-list-item').all_inner_texts()
            for text in texts:
                first_line = text.split('\n')[0].strip()
                if first_line.upper() == ticker.upper():
                    # Click the matching item by text
                    self.page.locator(f'.v-list-item:has-text("{first_line}")').first.click()
                    inp.press('Enter')
                    time.sleep(2)
                    url = self.page.url
                    cid_m = re.search(r'cid[=:](\d+)', url)
                    tid_m = re.search(r'tid[=:](\d+)', url)
                    if cid_m and tid_m:
                        self._cid_cache[ticker] = (cid_m.group(1), tid_m.group(1))
                        return cid_m.group(1), tid_m.group(1)
            return None, None
        except Exception as e:
            print(f'[!] Error buscando {ticker}: {e}')
            return None, None

    def _extract_jwt(self):
        """Extrae el JWT token desde localStorage (Cognito accessToken) para API calls."""
        try:
            # Buscar keys de Cognito que contengan 'accessToken'
            keys = self.page.evaluate('() => Object.keys(localStorage)')
            for k in keys:
                if 'accessToken' in k and 'Cognito' in k:
                    raw = self.page.evaluate(f'() => localStorage.getItem("{k}")')
                    if raw and raw.startswith('eyJ'):
                        self.jwt_token = raw
                        print(f'[OK] JWT extraido via {k[:40]}...')
                        return
            # Fallback: probar keys comunes
            for fallback_key in ['auth', 'persist:auth', 'token', 'idToken']:
                raw = self.page.evaluate(f'() => localStorage.getItem("{fallback_key}")')
                if raw:
                    try:
                        parsed = json.loads(raw)
                        if isinstance(parsed, dict):
                            self.jwt_token = (parsed.get('token') or parsed.get('accessToken')
                                              or parsed.get('jwt') or str(parsed))
                        else:
                            self.jwt_token = str(parsed).strip('"')
                    except json.JSONDecodeError:
                        self.jwt_token = raw.strip('"')
                    if self.jwt_token and self.jwt_token.startswith('eyJ'):
                        break
            if self.jwt_token:
                print('[OK] JWT extraido')
            else:
                print('[!] No se pudo extraer JWT')
        except Exception as e:
            print(f'[!] Error extrayendo JWT: {e}')

    def _extract_page(self, url, key, result):
        try:
            self.page.goto(url, wait_until='domcontentloaded')
            time.sleep(1.5)
            body = self.page.inner_text('body')
            lines = clean_text(body)
            html = self.page.content()
            result[f'{key}_text'] = lines[:500]
            result[f'{key}_html'] = html[:2000]
        except Exception as e:
            print(f'[!] Error en {key}: {e}')
            result[f'{key}_text'] = []

    def extract_ticker_data(self, ticker, cid, tid):
        result = {'ticker': ticker, 'cid': cid, 'tid': tid,
                  'timestamp': datetime.now(timezone.utc).isoformat()}
        pages = {
            'about': f'/stock/about?cid={cid}&tid={tid}',
            'financials_is': f'/stock/financials?cid={cid}&tid={tid}&tab=is',
            'financials_bs': f'/stock/financials?cid={cid}&tid={tid}&tab=bs',
            'financials_cf': f'/stock/financials?cid={cid}&tid={tid}&tab=cf',
            'financials_ratios': f'/stock/financials?cid={cid}&tid={tid}&tab=ratios',
            'multiples': f'/stock/multiples?cid={cid}&tid={tid}&tab=multi',
            'estimates': f'/stock/estimates?cid={cid}&tid={tid}&tab=est',
            'ownership': f'/stock/ownership?cid={cid}&tid={tid}',
            'news': f'/stock/news?cid={cid}&tid={tid}',
        }
        for key, path in pages.items():
            self._extract_page(f'{self.BASE}{path}', key, result)
        return result

    def scrape_all(self, tickers=None):
        if tickers is None:
            tickers = TICKERS_CORE
        print(f'[.] TIKR scraper: {len(tickers)} tickers...')
        if not self.login():
            print('[!] Login fallo')
            return self.data
        self.data = {'meta': {'source': 'TIKR Terminal',
                    'timestamp': datetime.now(timezone.utc).isoformat(),
                    'tickers': tickers}, 'tickers': {}}
        # Intentar resolver via API primero (watchlist)
        cid_map = {}
        if self.jwt_token:
            try:
                r = http_req.post('https://api.tikr.com/overview_it',
                                  json={'auth': self.jwt_token}, timeout=10)
                if r.status_code == 200:
                    for item in (r.json().get('active') or []):
                        if isinstance(item, dict):
                            tk = (item.get('symbol') or '').upper()
                            cid = str(item.get('cid') or '')
                            tid = str(item.get('tid') or '')
                            if tk and cid and tid and cid != 'None':
                                cid_map[tk] = (cid, tid)
            except: pass
        for tk in tickers:
            print(f'[.] {tk}...', end=' ')
            sys.stdout.flush()
            cid, tid = cid_map.get(tk, (None, None))
            if not cid:
                cid, tid = self.search_ticker(tk)
            if cid and tid:
                print(f'cid={cid}')
                td = self.extract_ticker_data(tk, cid, tid)
                self.data['tickers'][tk] = td
                with open(OUTPUT, 'w', encoding='utf-8') as f:
                    json.dump(self.data, f, indent=2, ensure_ascii=False)
            else:
                print('no encontrado')
                self.data['tickers'][tk] = {'ticker': tk, 'error': 'not_found'}
        return self.data

def main():
    email, password = load_credentials()
    scraper = TIKRScraper(email, password, headless=True)
    try:
        scraper.start()
        data = scraper.scrape_all()
        n = len(data.get('tickers', {}))
        print(f'\n[OK] TIKR completo: {n} tickers en {OUTPUT}')
    except Exception as e:
        print(f'[!] Error: {e}')
        import traceback; traceback.print_exc()
    finally:
        scraper.close()

if __name__ == '__main__':
    main()
