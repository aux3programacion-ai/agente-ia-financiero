#!/usr/bin/env python3
"""
analisis_ia.py - Analisis IA via OpenRouter con ensemble multi-modelo.
Inyecta noticias frescas, indicadores tecnicos, regimen de mercado,
y feedback de aprendizaje. Pondera resultados por precision historica.
"""
import json, os, sys, urllib.request, urllib.error, time, re, random, datetime
import numpy as np

DATA_DIR = os.environ.get('GITHUB_WORKSPACE', '.')

API_KEY = os.environ.get('OPENROUTER_KEY')

MODELOS = [
    'openrouter/free',
    'meta-llama/llama-3.3-70b-instruct:free',
    'nvidia/nemotron-3-super-120b-a12b:free',
    'nousresearch/hermes-3-llama-3.1-405b:free',
    'openai/gpt-oss-120b:free',
    'google/gemma-4-31b-it:free',
    'qwen/qwen3-coder:free',
    'google/gemma-4-26b-a4b-it:free',
    'moonshotai/kimi-k2.6:free',
    'openai/gpt-oss-20b:free',
    'cognitivecomputations/dolphin-mistral-24b-venice-edition:free',
    'nvidia/nemotron-3-nano-30b-a3b:free',
    'z-ai/glm-4.5-air:free',
    'nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free',
    'meta-llama/llama-3.2-3b-instruct:free'
]

TICKERS_CORE = ['NVDA','MU','DELL','AVGO','DDOG','SMCI','SNOW','CRWD','NOW','TSM',
               'ARM','OKTA','HPE','NTAP','CLS','AAPL','AMZN','GOOGL','META','MSFT',
               'LLY','AMAT','LRCX','PANW','ORCL','HON','UBER','GE','COST','NEE']

PROBS_BASE = {}
CONF_BASE = {}
PRICES_BASE = {}

# --- Trend-based fallback: compute probabilities from technical data ---
TECNICO_PATH = os.path.join(DATA_DIR, 'Datos', 'analisis_tecnico.json')
tec_data = {}
if os.path.exists(TECNICO_PATH):
    try:
        tec_data = json.load(open(TECNICO_PATH)).get('tecnicos', {})
        print(f'[Trend] Cargados tecnicos para {len(tec_data)} tickers')
    except Exception as e:
        print(f'[!] Error cargando tecnicos para trend: {e}')

def trend_prob(ticker):
    """Compute probability from technical indicators using trend analysis."""
    td = tec_data.get(ticker, {})
    if td.get('error'):
        return 50, 45, 100
    try:
        rsi = td.get('rsi', 50)
        macd = td.get('senial_macd', 'neutral')
        ma50_dist = td.get('pct_sobre_ma50', 0) or 0
        vol_ratio = td.get('vol_ratio', 1) or 1
        atr = td.get('atr_pct', 2) or 2
        tendencia = td.get('tendencia', 'neutral')

        prob = 50.0
        conf = 45.0

        if rsi is not None:
            if rsi < 30:
                prob += 15
                conf += 10
            elif rsi < 40:
                prob += 8
                conf += 5
            elif rsi > 70:
                prob -= 15
                conf += 10
            elif rsi > 60:
                prob -= 8
                conf += 5
            else:
                conf += 3

        if macd == 'alcista':
            prob += 10
            conf += 8
        elif macd == 'bajista':
            prob -= 10
            conf += 8

        if ma50_dist > 5:
            prob += 8
        elif ma50_dist > 2:
            prob += 4
        elif ma50_dist < -5:
            prob -= 8
        elif ma50_dist < -2:
            prob -= 4

        if vol_ratio > 1.5:
            prob += 5
        elif vol_ratio < 0.5:
            prob -= 5

        if atr > 4:
            conf -= 5
        elif atr < 1:
            conf += 5

        prob = max(10, min(90, prob))
        conf = max(10, min(90, conf))

        price = td.get('precio', 100)
        return round(prob), round(conf), price
    except:
        return 50, 45, 100

# Pre-populate from technical data for all core tickers
for t in TICKERS_CORE:
    p, c, pr = trend_prob(t)
    PROBS_BASE[t] = p
    CONF_BASE[t] = c
    PRICES_BASE[t] = pr

PRECIOS_PATH = os.path.join(DATA_DIR, 'Datos', 'precios_reales.json')
IA_OUTPUT_PATH = os.path.join(DATA_DIR, 'Datos', 'analisis_ia.json')
TECNICO_PATH = os.path.join(DATA_DIR, 'Datos', 'analisis_tecnico.json')
NEWS_PATH = os.path.join(DATA_DIR, 'Datos', 'noticias_recientes.json')
SCREENING_PATH = os.path.join(DATA_DIR, 'Datos', 'screening_global.json')
os.makedirs(os.path.join(DATA_DIR, 'Datos'), exist_ok=True)

# --- Load portfolio tickers (always included in analysis) ---
from portafolio_utils import cargar_portafolio, cargar_portafolio_cantidades
PORTAFOLIO_TICKERS = set(cargar_portafolio(DATA_DIR))
PORTAFOLIO_CANTIDADES = cargar_portafolio_cantidades(DATA_DIR)
if PORTAFOLIO_TICKERS:
    print(f'[Portafolio] Cargados {len(PORTAFOLIO_TICKERS)} tickers: {", ".join(sorted(PORTAFOLIO_TICKERS))}')

# Ensure portfolio tickers have base data entries
for t in PORTAFOLIO_TICKERS:
    if t not in PROBS_BASE:
        PROBS_BASE[t] = 55
        CONF_BASE[t] = 50
        PRICES_BASE[t] = 100

# Load global screening to determine TICKERS to analyze (all markets)
MAX_TICKERS_AI = 500
TICKERS = list(TICKERS_CORE)
if os.path.exists(SCREENING_PATH):
    try:
        scr = json.load(open(SCREENING_PATH))
        top50 = scr.get('top50_tickers', [])
        por_mercado = scr.get('por_mercado', {})

        # Also add top200 from screening (broader coverage for 500+)
        top200 = scr.get('top200', [])
        top200_tickers = [t['ticker'] for t in top200] if top200 else []

        # Sample proportionally from each market
        sampled = set(TICKERS_CORE)
        for t in top200_tickers:
            if len(sampled) >= MAX_TICKERS_AI:
                break
            if t not in sampled:
                sampled.add(t)
        remaining = MAX_TICKERS_AI - len(sampled)
        if remaining > 0 and por_mercado:
            # Distribute remaining slots proportionally
            market_counts = {m: len(v) for m, v in por_mercado.items()}
            total_extra = sum(market_counts.values())
            for m in sorted(market_counts, key=lambda x: market_counts[x], reverse=True):
                if remaining <= 0:
                    break
                # Exclude already included core tickers
                candidates = [t['ticker'] for t in por_mercado[m] if t['ticker'] not in sampled]
                alloc = max(1, int(remaining * market_counts[m] / total_extra))
                for c in candidates[:alloc]:
                    if len(sampled) >= MAX_TICKERS_AI:
                        break
                    sampled.add(c)
                remaining = MAX_TICKERS_AI - len(sampled)

        # Ensure portfolio tickers are always included
        sampled.update(PORTAFOLIO_TICKERS)

        merged = list(dict.fromkeys(list(sampled)))
        TICKERS = merged[:MAX_TICKERS_AI]
        print(f'[Screen] Cargados {len(top50)} screening + {len(TICKERS_CORE)} core de {len(por_mercado)} mercados = {len(TICKERS)} a analizar')
    except Exception as e:
        print(f'[!] Error cargando screening: {e}')

# --- Cargar precios ---
precios = {}
if os.path.exists(PRECIOS_PATH):
    try:
        pd = json.load(open(PRECIOS_PATH)).get('precios', {})
        for t in TICKERS:
            p = pd.get(t, {})
            precios[t] = p.get('price', PROBS_BASE.get(t, 100))
    except Exception as e:
        print(f'[!] Error leyendo precios: {e}')

texto_precios = ', '.join(f'{t} ${precios.get(t, PROBS_BASE.get(t,100)):.2f}' for t in TICKERS)

# --- Cargar noticias ---
texto_noticias = ''
news_sentimiento = {}
if os.path.exists(NEWS_PATH):
    try:
        news_data = json.load(open(NEWS_PATH))
        pt = news_data.get('por_ticker', {})
        ng = news_data.get('noticias_generales', [])
        lines_n = ['\nNOTICIAS RECIENTES (para usar en tu analisis):']
        if ng:
            lines_n.append('  [MERCADO GLOBAL - Reuters, CNBC, Bloomberg, MarketWatch]:')
            for n in ng[:6]:
                src = n.get('fuente', '?')
                lines_n.append(f'    [{src}] {n["titulo"][:130]}')
            lines_n.append('')
        for t in TICKERS:
            nd = pt.get(t, {})
            notis = nd.get('noticias', []) if isinstance(nd, dict) else []
            if notis and isinstance(notis, list):
                titulos = [n['titulo'][:120] for n in notis[:2] if isinstance(n, dict) and n.get('titulo')]
                if titulos:
                    lines_n.append(f'  {t}: {" | ".join(titulos)}')
            sent = nd.get('sentimiento') if isinstance(nd, dict) else None
            if sent and isinstance(sent, dict) and sent.get('sentimiento'):
                lines_n.append(f'    Sentimiento IA -> {sent["sentimiento"]} (score:{sent.get("score","?")})')
                news_sentimiento[t] = sent
        if len(lines_n) > 1:
            texto_noticias = '\n'.join(lines_n)
    except Exception as e:
        print(f'[!] Error cargando noticias: {e}')

# --- Cargar calendario economico ---
texto_calendario = ''
CAL_PATH = os.path.join(DATA_DIR, 'Datos', 'calendario_economico.json')
if os.path.exists(CAL_PATH):
    try:
        cal = json.load(open(CAL_PATH))
        evts = cal.get('proximos_eventos', [])
        if evts:
            lines_cal = ['\nCALENDARIO ECONOMICO (proximos eventos de alto impacto):']
            for e in evts[:8]:
                lines_cal.append(f'  {e["fecha"]} | {e["tipo"]}: {e["descripcion"]} (impacto: {e["impacto"]})')
            texto_calendario = '\n'.join(lines_cal)
    except Exception as e:
        print(f'[!] Error cargando calendario: {e}')

# --- Cargar sentimiento social ---
texto_social = ''
SOC_PATH = os.path.join(DATA_DIR, 'Datos', 'analisis_social.json')
if os.path.exists(SOC_PATH):
    try:
        soc = json.load(open(SOC_PATH))
        st = soc.get('tickers', {})
        if st:
            lines_soc = ['\nSENTIMIENTO REDES (menciones en Reddit/foros):']
            for t in list(st.keys())[:15]:
                s = st[t]
                lines_soc.append(f'  {t}: {s.get("sentimiento","?")} (score:{s.get("score",0):.2f}, {s.get("menciones",0)} menciones)')
            texto_social = '\n'.join(lines_soc)
    except Exception as e:
        print(f'[!] Error cargando social: {e}')

# --- Cargar datos de Google Finance (estadisticas clave, beta, 52w, earnings) ---
texto_google_finance = ''
GF_PATH = os.path.join(DATA_DIR, 'Datos', 'google_finance.json')
if os.path.exists(GF_PATH):
    try:
        gf = json.load(open(GF_PATH))
        gf_tickers = gf.get('tickers', {})
        lines_gf = ['\nDATOS GOOGLE FINANCE (fundamentales, beta, rango 52sem, earnings, relacionados):']
        for t in TICKERS:
            gft = gf_tickers.get(t, {})
            if not gft:
                continue
            stats = gft.get('stats') or {}
            earnings = gft.get('earnings') or {}
            related = gft.get('related_stocks') or []
            parts = []
            if stats.get('beta'): parts.append(f'Beta:{stats["beta"]}')
            if stats.get('pe_ratio'): parts.append(f'PE:{stats["pe_ratio"]}')
            if stats.get('eps'): parts.append(f'EPS:{stats["eps"]}')
            if stats.get('high_52w') and stats.get('low_52w'):
                parts.append(f'52w:{stats["low_52w"]}-{stats["high_52w"]}')
            if stats.get('dividend_yield'): parts.append(f'Div:{stats["dividend_yield"]}%')
            if stats.get('market_cap_str'): parts.append(f'MktCap:{stats["market_cap_str"]}')
            if stats.get('avg_volume'): parts.append(f'AvgVol:{stats["avg_volume"]}')
            if earnings.get('surprise_pct'): parts.append(f'Sorpresa:{earnings["surprise_pct"]}')
            if earnings.get('fiscal_period'): parts.append(f'Periodo:{earnings["fiscal_period"]}')
            if related:
                rel_tickers = [r['ticker'] for r in related[:4]]
                parts.append(f'Relacionados:{" ".join(rel_tickers)}')
            if parts:
                lines_gf.append(f'  {t}: {" | ".join(parts)}')
        if len(lines_gf) > 1:
            texto_google_finance = '\n'.join(lines_gf)
            print(f'[GF] Datos cargados para {sum(1 for t in TICKERS if t in gf_tickers)} tickers')
    except Exception as e:
        print(f'[!] Error cargando Google Finance: {e}')

# --- Cargar datos de TIKR Terminal (ownership, news, multiples, estimaciones, ratios) ---
texto_tikr = ''
TIKR_PATH = os.path.join(DATA_DIR, 'Datos', 'tikr_data.json')
if os.path.exists(TIKR_PATH):
    try:
        tikr = json.load(open(TIKR_PATH))
        tikr_tickers = tikr.get('tickers', {})
        lines_tikr = ['\nDATOS TIKR (ownership, noticias, valoracion, estimaciones, ratios):']
        for t in TICKERS:
            tdt = tikr_tickers.get(t, {})
            if not tdt or 'error' in tdt:
                continue
            parts = []
            about = ' '.join(tdt.get('about_text', [])[:30])
            if '52 Week High' in about: parts.append(f'52w high/low presente en about')
            if 'Beta' in about:
                for ln in about.split(' '):
                    if 'Beta' in ln and ln.replace('Beta','').strip().replace('.','').replace(',','').isdigit():
                        parts.append(f'Beta:{ln.replace("Beta","").strip()}')
                        break
            own = ' '.join(tdt.get('ownership_text', [])[:50])
            inv_count = len(re.findall(r'Inversores?\s+\d+', own))
            if inv_count: parts.append(f'Owners:{inv_count} inversores listados')
            news = tdt.get('news_text', [])
            news_short = [n for n in news if len(n) > 20 and 'PRO' not in n][:3]
            if news_short: parts.append(f'Noticias:{" | ".join(news_short)}')
            ratios = ' '.join(tdt.get('financials_ratios_text', [])[:40])
            if 'PRO' not in ratios[:200]:
                for kw in ['Margen','ROE','ROA','Deuda']:
                    if kw in ratios:
                        parts.append(f'Ratio:{kw} disponible')
                        break
            if parts:
                lines_tikr.append(f'  {t}: {" | ".join(parts)}')
        if len(lines_tikr) > 1:
            texto_tikr = '\n'.join(lines_tikr)
            print(f'[TIKR] Datos cargados para {sum(1 for t in TICKERS if t in tikr_tickers)} tickers')
    except Exception as e:
        print(f'[!] Error cargando TIKR: {e}')

# --- Fetch fundamental analyst data for portfolio tickers from Yahoo Finance ---
texto_analisis_portafolio = ''
if PORTAFOLIO_TICKERS:
    lines_pf = ['\nDATOS FUNDAMENTALES PORTAFOLIO (analistas web en tiempo real):']
    for t in PORTAFOLIO_TICKERS:
        try:
            url = f'https://query2.finance.yahoo.com/v10/finance/quoteSummary/{t}?modules=price,summaryDetail,financialData,defaultKeyStatistics'
            req_pf = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
            with urllib.request.urlopen(req_pf, timeout=10) as resp_pf:
                qs = json.loads(resp_pf.read())['quoteSummary']['result'][0]
                fd = qs.get('financialData', {})
                price = qs.get('price', {})
                sd = qs.get('summaryDetail', {})
                target_mean = fd.get('targetMeanPrice', {}).get('raw', 'N/A')
                target_high = fd.get('targetHighPrice', {}).get('raw', 'N/A')
                target_low = fd.get('targetLowPrice', {}).get('raw', 'N/A')
                rec = fd.get('recommendationKey', 'N/A')
                num_analysts = fd.get('numberOfAnalystOpinions', {}).get('raw', 'N/A')
                pe = sd.get('trailingPE', {}).get('raw', 'N/A')
                mcap = price.get('marketCap', {}).get('raw', 'N/A')
                prev_close = sd.get('previousClose', {}).get('raw', 'N/A')
                lines_pf.append(f'  {t}: Recomendacion: {rec.upper()} | Analistas: {num_analysts} | Target: ${target_mean} ($ {target_low} - ${target_high}) | PE: {pe} | MktCap: ${mcap:,}')
        except Exception as e:
            print(f'  [Fundamental {t}] No se pudieron obtener datos: {e}')
            lines_pf.append(f'  {t}: Datos fundamentales no disponibles')
    if len(lines_pf) > 1:
        texto_analisis_portafolio = '\n'.join(lines_pf)

# --- Cargar tecnicos + regimen de mercado ---
texto_tecnicos = ''
regimen_mercado = ''
if os.path.exists(TECNICO_PATH):
    try:
        tec = json.load(open(TECNICO_PATH))
        spy = tec.get('spy', {})
        regimen = tec.get('regimen_mercado', 'desconocido')
        regimen_mercado = f'REGIMEN DE MERCADO: {regimen.upper()} - {spy.get("descripcion", "")}'
        lines_t = [f'\nREGIMEN DE MERCADO: {regimen.upper()}']
        if spy:
            lines_t.append(f'SPY: ${spy.get("precio","?")} | MA50: ${spy.get("ma50","?")} | MA200: ${spy.get("ma200","?")} | Tendencia: {spy.get("tendencia_pct","?")}%')
        lines_t.append('\nINDICADORES TECNICOS POR TICKER (usar en tu analisis):')
        for t in TICKERS:
            tn = tec.get('tecnicos', {}).get(t, {})
            if tn.get('error'): continue
            lines_t.append(f'{t}: {tn.get("tendencia","?")} | RSI:{tn.get("rsi","?")} | MACD:{tn.get("senial_macd","?")} | MA50:{tn.get("pct_sobre_ma50","?")}% | ATR:{tn.get("atr_pct","?")}% | Vol:{tn.get("vol_ratio","?")}x')
        texto_tecnicos = '\n'.join(lines_t)
        print(f'[Tecnico] Regimen: {regimen}')
    except Exception as e:
        print(f'[!] Error cargando tecnicos: {e}')

# --- Cargar retroalimentacion de aprendizaje ---
feedback_precision = ''
precision_por_modelo = {}
if os.path.exists(IA_OUTPUT_PATH):
    try:
        prev = json.load(open(IA_OUTPUT_PATH))
        fb = prev.get('feedback_aprendizaje', '')
        if fb:
            feedback_precision = '\n' + fb + '\n'
    except:
        pass

# --- Cargar precision por modelo para ensemble ---
CALIB_PATH = os.path.join(DATA_DIR, 'Datos', 'calibracion.json')
if os.path.exists(CALIB_PATH):
    try:
        cal = json.load(open(CALIB_PATH))
        modelos_stats = cal.get('modelos_ia', {})
        for m, ms in modelos_stats.items():
            if ms.get('total', 0) >= 2:
                precision_por_modelo[m] = ms['precision']
    except:
        pass

# --- EWMA weights online learning ---
EWMA_ALPHA = 0.3  # decay factor for recent performance
def compute_ewma_weights(hist_path, alpha=EWMA_ALPHA):
    """Compute EWMA weights from historical model performance."""
    ewma_weights = {}
    if not os.path.exists(hist_path):
        return ewma_weights
    try:
        hist = json.load(open(hist_path))
        # For each ticker, track model predictions and outcomes
        for ticker, data in hist.items():
            preds = data.get('predicciones', [])
            for p in preds:
                model = p.get('modelo_usado', '')
                if not model or 'ensemble' in model or 'fallback' in model or 'trend' in model:
                    continue
                # Extract base model name
                base_model = model.split('-')[0] if '-' in model else model
                # We can't compute EWMA without knowing outcomes
                # This will be populated by aprendizaje.py
    except:
        pass
    return ewma_weights

# For now use static precision as base, EWMA will be added when aprendizaje.py tracks per-model outcomes
ewma_model_weights = {}

SYSTEM_PROMPT = 'Eres un analista financiero experto con 20 anos de experiencia en mercados globales. Respondes SOLO con JSON valido, sin markdown, sin explicaciones.'

# --- Auto-evolution: load evolved prompt + few-shot + fine-tune injection ---
auto_few_shot = ''
auto_finetune = ''
auto_prompt_winner = 'base'
AUTO_PROMPT_PATH = os.path.join(DATA_DIR, 'Datos', 'auto_prompts.json')
AUTO_EVOL_PATH = os.path.join(DATA_DIR, 'Datos', 'evolution_state.json')
if os.path.exists(AUTO_PROMPT_PATH):
    try:
        ap = json.load(open(AUTO_PROMPT_PATH))
        auto_prompt_winner = ap.get('winner', 'base')
        evolved_sp = ap.get('prompts', {}).get(auto_prompt_winner)
        if evolved_sp and auto_prompt_winner != 'base':
            SYSTEM_PROMPT = evolved_sp
            print(f'[AutoEvol] System prompt evolucionado a: {auto_prompt_winner}')
    except Exception as e:
        print(f'[!] Error loading auto prompts: {e}')

# --- Load few-shot examples and fine-tune injection ---
auto_few_shot = ''
auto_finetune = ''
FEW_SHOT_PATH = os.path.join(DATA_DIR, 'Datos', 'few_shot_examples.json')
if os.path.exists(FEW_SHOT_PATH):
    try:
        fs = json.load(open(FEW_SHOT_PATH))
        examples = fs.get('examples', {})
        if examples:
            # Build few-shot section from top examples across regimes
            lines_fs = ['EJEMPLOS DE PREDICCIONES EXITOSAS (referencia):']
            for key, exs in list(examples.items())[:5]:
                best = sorted(exs, key=lambda x: (x.get('acertada', False), x.get('timestamp','')), reverse=True)[0]
                if best.get('acertada'):
                    tk, rg = key.split('_', 1)
                    lines_fs.append(f'{tk} ({rg}): Prob={best["probabilidad"]}% -> {best["analisis"][:100]}')
            auto_few_shot = '\n'.join(lines_fs)
    except:
        pass

FT_INJECT_PATH = os.path.join(DATA_DIR, 'Datos', 'evolution_state.json')
if os.path.exists(FT_INJECT_PATH):
    try:
        ev = json.load(open(FT_INJECT_PATH))
        if ev.get('n_predictions', 0) > 50:
            auto_finetune = f'PATRONES APRENDIDOS: {ev["n_sector_patterns"]} sectores con patrones identificados. Prompt optimizado: {ev.get("prompt_winner", "base")}.'
    except:
        pass

feedback_section = ''
if feedback_precision:
    feedback_section = f'''
HISTORIAL DE APRENDIZAJE (precision de predicciones anteriores):
{feedback_precision}

IMPORTANTE: Ajusta tus probabilidades segun tu precision historica.
Si tienes alta precision en un ticker o sector, puedes aumentar la confianza.
Si tienes baja precision, reduce tu confianza y probabilidad.
Usa los rangos de probabilidad para calibrar: si en rango 60-65% tu precision historica es baja, se mas conservador ahi.'''

# --- Skill Injection: Load learned patterns from skill memory ---
skill_injection = ''
mistake_injection = ''
skill_meta_evolution = None
SKILL_PATH = os.path.join(DATA_DIR, 'Datos', 'skill_memory.json')
if os.path.exists(SKILL_PATH):
    try:
        skill_data = json.load(open(SKILL_PATH))
        skill_injection = skill_data.get('injection', '')
        skill_meta_evolution = skill_data.get('meta_evolution')
        if skill_injection:
            print(f'[Skills] Cargados {len(skill_data.get("skills",{}))} skills adquiridos')
    except Exception as e:
        print(f'[!] Error cargando skills: {e}')

# --- Evolve system prompt based on accumulated skills ---
effective_system_prompt = SYSTEM_PROMPT
if skill_meta_evolution:
    effective_system_prompt = skill_meta_evolution
    print(f'[Meta] System prompt evolucionado por aprendizaje')

ticker_list_str = ', '.join(TICKERS)
USER_PROMPT_TEMPLATE = f'''Genera analisis para estos {len(TICKERS)} tickers de mercados globales (US, Mexico, Europa, Asia).
Tickers: {ticker_list_str}
Precios actuales: {texto_precios}
{texto_noticias}
{texto_social}
{texto_calendario}
 {texto_google_finance}
 {texto_tikr}
 {texto_analisis_portafolio}
{texto_tecnicos}
{auto_few_shot}
{auto_finetune}
{skill_injection}
{mistake_injection}
{feedback_section}

Responde EXACTAMENTE este JSON sin ningun otro texto:
{{"resumen_mercado":"texto corto de 1 oracion sobre el mercado global",
"modelo_usado":"modelo-ia",
"titulares":["headline1","headline2","headline3","headline4","headline5"],
"sectores":{{"Semiconductores":"analisis","Servidores IA":"analisis","Software IA":"analisis","Ciberseguridad":"analisis","Industrial":"analisis","Financiero":"analisis","Energia":"analisis","Consumo":"analisis","Salud":"analisis","Utilities":"analisis","Materiales":"analisis","Inmobiliario":"analisis","Global":"analisis"}},
"probabilidades":{{}}}}

Cada ticker en "probabilidades" debe tener:
  "probabilidad" (0-100 alza en 30d),
  "confianza" (0-100),
  "analisis" (texto corto justificando con tecnicos/noticias),
  "precio_objetivo_30d" (precio estimado $ en 30 dias),
  "precio_objetivo_3m" (precio estimado $ en 3 meses),
  "precio_objetivo_6m" (precio estimado $ en 6 meses),
  "precio_objetivo_1y" (precio estimado $ en 1 ano),
  "mercado" (region: "US"/"MEXICO"/"EUROPA"/"ASIA"/"GLOBAL")'''

def llamar_modelo(modelo, prompt, sp=None):
    sp = sp or effective_system_prompt
    url = 'https://openrouter.ai/api/v1/chat/completions'
    payload = json.dumps({
        'model': modelo,
        'messages': [
            {'role': 'system', 'content': sp},
            {'role': 'user', 'content': prompt}
        ],
        'temperature': 0.1,
        'max_tokens': 4000
    }).encode('utf-8')
    req = urllib.request.Request(url, data=payload,
        headers={
            'Authorization': f'Bearer {API_KEY}',
            'Content-Type': 'application/json',
            'HTTP-Referer': 'https://github.com/aux3programacion-ai/agente-ia-financiero',
            'X-Title': 'Agente IA Financiero'
        },
        method='POST')
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read())['choices'][0]['message']['content']

def extraer_json(texto):
    texto = texto.strip()
    if texto.startswith('```'):
        m = re.search(r'```(?:json)?\s*([\s\S]*?)```', texto)
        if m: texto = m.group(1).strip()
    inicio = texto.find('{')
    fin = texto.rfind('}')
    if inicio != -1 and fin != -1:
        texto = texto[inicio:fin+1]
    return json.loads(texto)

def validar_resultado(r):
    return ('probabilidades' in r and len(r['probabilidades']) > 0
            and isinstance(list(r['probabilidades'].values())[0], dict)
            and 'probabilidad' in list(r['probabilidades'].values())[0])

def generar_trend_fallback():
    reg = tec_data.get('spy', {}).get('tendencia_pct', 'neutral') if tec_data else 'desconocido'
    return {
        'resumen_mercado': f'Mercado con tendencia {reg}. Analisis basado en indicadores tecnicos.',
        'modelo_usado': 'trend-fallback',
        'titulares': ['Analisis basado en tendencia tecnica', 'Indicadores RSI/MACD/SMA',
            'Datos via yfinance', 'Monitoreo multiplataforma'],
        'sectores': {s: 'Sector en monitoreo' for s in [
            'Semiconductores','Servidores IA','Software IA','Ciberseguridad','Industrial',
            'Financiero','Energia','Consumo','Salud','Utilities','Materiales','Inmobiliario','Global']},
        'probabilidades': {
            t: {'probabilidad': PROBS_BASE.get(t, 50), 'confianza': CONF_BASE.get(t, 50),
                'analisis': 'Analisis basado en tendencia tecnica (RSI/MACD/SMA).',
                'precio_objetivo_30d': precios.get(t, PRICES_BASE.get(t, 100)) * (1 + (PROBS_BASE.get(t, 50) - 50) / 200),
                'precio_objetivo_3m': precios.get(t, PRICES_BASE.get(t, 100)) * (1 + (PROBS_BASE.get(t, 50) - 50) / 150),
                'precio_objetivo_6m': precios.get(t, PRICES_BASE.get(t, 100)) * (1 + (PROBS_BASE.get(t, 50) - 50) / 100),
                'precio_objetivo_1y': precios.get(t, PRICES_BASE.get(t, 100)) * (1 + (PROBS_BASE.get(t, 50) - 50) / 80),
                'mercado': 'US'}
            for t in TICKERS
        }
    }

# ============================================================
# ENSEMBLE MULTI-MODELO
# ============================================================
if not API_KEY:
    print('[!] OPENROUTER_KEY no configurada, usando tendencia tecnica')
    resultado_final = generar_trend_fallback()
    resultado_final['modelo_usado'] = 'no-key'
else:
    respuestas_modelos = []
    modelos_exitosos = []
    prompt_completo = USER_PROMPT_TEMPLATE

    # --- Specialized prompts for ensemble diversity ---
    SPECIALIST_PROMPTS = {
        'tecnico': f'{USER_PROMPT_TEMPLATE}\n\nERES UN ANALISTA TECNICO. Enfocate en RSI, MACD, SMA, volumen, soporte/resistencia, patrones de velas. Ignora ruido de noticias cortoplacistas. Usa los indicadores tecnicos como tu guia principal.',
        'fundamental': f'{USER_PROMPT_TEMPLATE}\n\nERES UN ANALISTA FUNDAMENTAL. Enfocate en earnings, revenue growth, valoracion (P/E, P/S), guia de management, ciclo del sector, ventajas competitivas. Los tecnicos son secundarios.',
        'macro': f'{USER_PROMPT_TEMPLATE}\n\nERES UN ANALISTA MACRO/SENTIMIENTO. Enfocate en tasas de interes, inflacion, riesgo global, flujo de noticias, sentimiento de mercado, posicionamiento institucional. El contexto macro es tu guia principal.',
    }
    DEFAULT_SPECIALTY = 'tecnico'
    
    # Load specialty performance data
    SPECIALTY_PATH = os.path.join(DATA_DIR, 'Datos', 'specialty_performance.json')
    specialty_perf = {}
    if os.path.exists(SPECIALTY_PATH):
        try: specialty_perf = json.load(open(SPECIALTY_PATH))
        except: pass
    # Assign specialties based on past performance or round-robin
    # Track which model has which specialty
    model_specialty = {}
    for i, modelo in enumerate(MODELOS):
        specialties = list(SPECIALIST_PROMPTS.keys())
        base_model = modelo.split('/')[-1] if '/' in modelo else modelo
        # If model has a recorded best specialty, use it
        best_spec = None
        best_acc = 0
        for spec in specialties:
            spec_key = f'{base_model}_{spec}'
            acc = specialty_perf.get(spec_key, {}).get('accuracy', 0)
            if acc > best_acc:
                best_acc = acc
                best_spec = spec
        model_specialty[modelo] = best_spec if best_spec else specialties[i % len(specialties)]
    
    for modelo in MODELOS:
        if len(modelos_exitosos) >= 3:
            break
        try:
            specialty = model_specialty.get(modelo, DEFAULT_SPECIALTY)
            prompt_actual = SPECIALIST_PROMPTS.get(specialty, prompt_completo)
            print(f'[IA] Intentando modelo: {modelo} ({specialty})')
            raw = llamar_modelo(modelo, prompt_actual)
            parsed = extraer_json(raw)
            if validar_resultado(parsed):
                parsed['modelo_usado'] = modelo
                parsed['specialty'] = specialty
                respuestas_modelos.append(parsed)
                modelos_exitosos.append(modelo)
                print(f'[OK] {modelo} ({specialty}) respondio correctamente')
            else:
                print(f'[!] {modelo} respondio sin probabilidades validas')
        except Exception as e:
            print(f'[!] {modelo} fallo: {str(e)[:80]}')
            continue

    if not respuestas_modelos:
        print('[!] Todos los modelos fallaron, usando tendencia tecnica')
        resultado_final = generar_trend_fallback()
        resultado_final['modelo_usado'] = 'trend-fallback'
    else:
        # Ensemble: promediar probabilidades ponderadas por EWMA (precision reciente) + precision historica
        resultado_final = respuestas_modelos[0].copy()
        pesos_modelo = {}
        for rm in respuestas_modelos:
            m = rm['modelo_usado']
            static_prec = precision_por_modelo.get(m, 0.5)
            ewma_prec = ewma_model_weights.get(m, static_prec)
            # Blend: 70% EWMA (recent), 30% static (historical)
            pesos_modelo[m] = 0.7 * ewma_prec + 0.3 * static_prec

        if len(respuestas_modelos) > 1:
            for t in TICKERS:
                probs = []; confs = []; targets30 = []; targets3m = []; targets6m = []; targets1y = []; mercados = []; analisis_list = []; pesos = []
                for rm in respuestas_modelos:
                    p = rm.get('probabilidades', {}).get(t, {})
                    if isinstance(p, dict) and p.get('probabilidad'):
                        probs.append(p['probabilidad'])
                        confs.append(p.get('confianza', 50))
                        t30 = p.get('precio_objetivo_30d', 0)
                        if t30 and t30 > 0: targets30.append(t30)
                        t3 = p.get('precio_objetivo_3m', 0)
                        if t3 and t3 > 0: targets3m.append(t3)
                        t6 = p.get('precio_objetivo_6m', 0)
                        if t6 and t6 > 0: targets6m.append(t6)
                        t1 = p.get('precio_objetivo_1y', 0)
                        if t1 and t1 > 0: targets1y.append(t1)
                        m = p.get('mercado', '')
                        if m: mercados.append(m)
                        analisis_list.append(p.get('analisis', ''))
                        pesos.append(pesos_modelo[rm['modelo_usado']])

                if probs:
                    peso_total = sum(pesos)
                    w_prob = sum(p * w for p, w in zip(probs, pesos)) / peso_total if peso_total else sum(probs) / len(probs)
                    w_conf = sum(c * w for c, w in zip(confs, pesos)) / peso_total if peso_total else sum(confs) / len(confs)
                    # --- Consensus Quality Score ---
                    n_models = len(probs)
                    prob_std = np.std(probs) if len(probs) > 1 else 0
                    agreement = max(0, 100 - prob_std * 3)
                    distance_50 = abs(w_prob - 50) * 2
                    model_coverage = (n_models / len(respuestas_modelos)) * 100
                    cqs = (agreement * 0.4 + distance_50 * 0.3 + model_coverage * 0.2 + (w_conf - 50) * 0.1)
                    cqs = max(5, min(99, cqs))
                    # --- Active Learning: flag low-conviction tickers ---
                    active_learning_flag = (prob_std > 15 or abs(w_prob - 50) < 5 or n_models == 1)
                    w_target30 = sum(targets30) / len(targets30) if targets30 else 0
                    w_target3m = sum(targets3m) / len(targets3m) if targets3m else 0
                    w_target6m = sum(targets6m) / len(targets6m) if targets6m else 0
                    w_target1y = sum(targets1y) / len(targets1y) if targets1y else 0
                    w_mercado = max(set(mercados), key=mercados.count) if mercados else 'US'
                    w_analisis = max(analisis_list, key=lambda a: len(a)) if analisis_list else ''
                    resultado_final.setdefault('probabilidades', {})[t] = {
                        'probabilidad': round(w_prob),
                        'confianza': round(w_conf),
                        'consensus_quality_score': round(cqs),
                        'active_learning_flag': active_learning_flag,
                        'analisis': w_analisis,
                        'precio_objetivo_30d': round(w_target30, 2) if w_target30 else precios.get(t, PRICES_BASE.get(t, 100)),
                        'precio_objetivo_3m': round(w_target3m, 2) if w_target3m else precios.get(t, PRICES_BASE.get(t, 100)),
                        'precio_objetivo_6m': round(w_target6m, 2) if w_target6m else precios.get(t, PRICES_BASE.get(t, 100)),
                        'precio_objetivo_1y': round(w_target1y, 2) if w_target1y else precios.get(t, PRICES_BASE.get(t, 100)),
                        'mercado': w_mercado
                    }

            resultado_final['modelo_usado'] = 'ensemble-ewma-' + '+'.join(modelos_exitosos)
            print(f'[Ensemble EWMA] {len(respuestas_modelos)} modelos combinados: {", ".join(modelos_exitosos)}')
        else:
            resultado_final = respuestas_modelos[0]
            print(f'[Single] Modelo unico: {modelos_exitosos[0]}')

# Ensure all 30 tickers
for t in TICKERS:
    p = resultado_final.get('probabilidades', {}).get(t)
    if p is None or not isinstance(p, dict) or not p.get('probabilidad'):
        base_p = precios.get(t, PRICES_BASE.get(t, 100))
        prob = PROBS_BASE.get(t, 50)
        resultado_final.setdefault('probabilidades', {})[t] = {
            'probabilidad': prob,
            'confianza': CONF_BASE.get(t, 50),
            'analisis': 'Analisis baseline.',
            'precio_objetivo_30d': base_p * (1 + (prob - 50) / 200),
            'precio_objetivo_3m': base_p * (1 + (prob - 50) / 150),
            'precio_objetivo_6m': base_p * (1 + (prob - 50) / 100),
            'precio_objetivo_1y': base_p * (1 + (prob - 50) / 80),
            'mercado': 'US'
        }

# ============================================================
# INTEGRAR XGBOOST (blending con ensemble ponderado por precision)
# ============================================================
XGB_PATH = os.path.join(DATA_DIR, 'Datos', 'modelo_xgboost.json')
if os.path.exists(XGB_PATH):
    try:
        xgb_data = json.load(open(XGB_PATH))
        xgb_tickers = xgb_data.get('tickers', {})
        xgb_cv = xgb_data.get('cv_accuracy_mean', 0.5)
        xgb_weight = min(xgb_cv / 0.5, 1.5) if xgb_cv > 0 else 0.5
        blended_count = 0
        for t in TICKERS:
            if t in xgb_tickers and t in resultado_final.get('probabilidades', {}):
                xgb_prob = xgb_tickers[t].get('prob_up_20d', 50)
                ia_prob = resultado_final['probabilidades'][t].get('probabilidad', 50)
                ia_conf = resultado_final['probabilidades'][t].get('confianza', 50)
                blended = (ia_prob * 1.0 + xgb_prob * xgb_weight) / (1.0 + xgb_weight)
                resultado_final['probabilidades'][t]['probabilidad_xgb'] = xgb_prob
                resultado_final['probabilidades'][t]['probabilidad'] = round(max(5, min(95, blended)))
                resultado_final['probabilidades'][t]['confianza'] = round(max(5, min(95, ia_conf * (0.7 + 0.3 * xgb_weight))))
                blended_count += 1
        if blended_count > 0:
            print(f'[XGBoost] Integrado con peso {xgb_weight:.2f} en {blended_count} tickers')
    except Exception as e:
        print(f'[!] Error integrando XGBoost: {e}')

# ============================================================
# REINFORCEMENT LEARNING: Actualizar pesos de modelos por outcomes
# Recompensa: precision * confianza. Penaliza errores de alta confianza.
# ============================================================
RL_PATH = os.path.join(DATA_DIR, 'Datos', 'model_rl_weights.json')
rl_weights = {}
if os.path.exists(RL_PATH):
    try: rl_weights = json.load(open(RL_PATH))
    except: pass

RL_APR_PATH = os.path.join(DATA_DIR, 'Datos', 'aprendizaje.json')
if os.path.exists(RL_APR_PATH):
    try:
        apr = json.load(open(RL_APR_PATH))
        preds = apr.get('predicciones', [])
        model_outcomes = {}
        for p in preds[-200:]:
            model_key = p.get('modelo_usado', p.get('feature_used', 'unknown'))
            if model_key not in model_outcomes:
                model_outcomes[model_key] = {'correct': 0, 'total': 0, 'conf_sum': 0, 'error_conf_sum': 0}
            mo = model_outcomes[model_key]
            mo['total'] += 1
            conf = p.get('confianza', 50)
            if p.get('acierto'):
                mo['correct'] += 1
                mo['conf_sum'] += conf
            else:
                mo['error_conf_sum'] += conf
        
        for model_key, mo in model_outcomes.items():
            if mo['total'] >= 5:
                accuracy = mo['correct'] / mo['total']
                avg_conf_correct = mo['conf_sum'] / max(mo['correct'], 1)
                avg_conf_error = mo['error_conf_sum'] / max(mo['total'] - mo['correct'], 1)
                # RL reward: high accuracy → +, high-confidence errors → penalty
                reward = (accuracy - 0.5) * 2  # -1 to +1
                penalty = avg_conf_error * (1 - accuracy) * 0.01 if mo['total'] - mo['correct'] > 0 else 0
                rl_signal = reward - penalty
                # Update weight with momentum
                old_w = rl_weights.get(model_key, 0.5)
                new_w = old_w + 0.1 * rl_signal  # Learning rate 0.1
                rl_weights[model_key] = max(0.05, min(0.95, new_w))
        
        with open(RL_PATH, 'w') as f:
            json.dump(rl_weights, f, indent=2)
        updated = sum(1 for mo in model_outcomes.values() if mo['total'] >= 5)
        if updated:
            print(f'[RL] Pesos actualizados para {updated} modelos/features')
    except Exception as e:
        print(f'[!] RL update error: {e}')

# ============================================================
# A/B PROMPT TESTING: Probar variaciones y elegir ganadora
# ============================================================
PROMPT_AB_PATH = os.path.join(DATA_DIR, 'Datos', 'prompt_ab_test.json')
prompt_ab_data = {'variations': {}, 'winner': 'A', 'trials': {}}
if os.path.exists(PROMPT_AB_PATH):
    try: prompt_ab_data = json.load(open(PROMPT_AB_PATH))
    except: pass

# Track which variation was used this cycle
prompt_variation_used = prompt_ab_data.get('winner', 'A')
resultado_final['prompt_variation'] = prompt_variation_used

# ============================================================
# MISTAKE MEMORY: Post-mortems de errores pasados
# ============================================================
MISTAKE_PATH = os.path.join(DATA_DIR, 'Datos', 'mistake_memory.json')
if os.path.exists(MISTAKE_PATH):
    try:
        mm = json.load(open(MISTAKE_PATH))
        mistakes = mm.get('recent_mistakes', [])
        if mistakes:
            lines = ['ERRORES RECIENTES PARA EVITAR:']
            for m in mistakes[:5]:
                lines.append(f'{m["ticker"]}: Se predijo {m["probabilidad"]}% pero ocurrio lo opuesto. Contexto: {m["contexto"]}')
            mistake_injection = '\n'.join(lines)
    except:
        pass

# ============================================================
# DEBATE MULTI-MODELO (Round 2: consenso sobre discrepancias)
# ============================================================
if len(respuestas_modelos) >= 2 and API_KEY:
    DISPUTE_THRESHOLD = 15
    try:
        # Find tickers with major disagreements
        disputes = []
        for t in TICKERS:
            probs = []
            analisis = []
            for rm in respuestas_modelos:
                p = rm.get('probabilidades', {}).get(t, {})
                if isinstance(p, dict) and p.get('probabilidad'):
                    probs.append(p['probabilidad'])
                    a = p.get('analisis', '')
                    if a:
                        analisis.append(a[:200])
            if len(probs) >= 2 and (max(probs) - min(probs)) >= DISPUTE_THRESHOLD:
                disputes.append((t, probs, analisis))
        
        if disputes:
            print(f'[Debate] {len(disputes)} tickers con discrepancias >{DISPUTE_THRESHOLD}%')
            # Build debate prompt
            debate_lines = ['DISCREPANCIAS ENTRE MODELOS (resolver con consenso):']
            for t, probs, analisis in disputes[:10]:
                debate_lines.append(f'{t}: probs={probs} | analisis={" | ".join(analisis[:2])}')
            debate_prompt = '\n'.join(debate_lines)
            
            debate_prompt_full = f'''{SYSTEM_PROMPT}

Analiza las siguientes discrepancias entre modelos de IA y genera un JSON con probabilidades corregidas por consenso.

{debate_prompt}

Responde SOLO con JSON, sin markdown. Formato: {{"probabilidades": {{"TICKER": {{"probabilidad": 55, "confianza": 60, "analisis": "Consenso tras debate: ..."}}}}}}'''
            
            # Call consensus model
            consensus_raw = llamar_modelo('openrouter/free', debate_prompt_full, sp=effective_system_prompt)
            try:
                consensus = extraer_json(consensus_raw)
                if validar_resultado(consensus):
                    for t in disputes:
                        tk = t[0]
                        if tk in resultado_final.get('probabilidades', {}) and tk in consensus.get('probabilidades', {}):
                            cp = consensus['probabilidades'][tk]
                            old_prob = resultado_final['probabilidades'][tk].get('probabilidad', 50)
                            # Blend: 50% original ensemble + 50% debate
                            debate_prob = cp.get('probabilidad', old_prob)
                            final_prob = round((old_prob + debate_prob) / 2)
                            resultado_final['probabilidades'][tk]['probabilidad_debate'] = debate_prob
                            resultado_final['probabilidades'][tk]['probabilidad'] = round(max(5, min(95, final_prob)))
                            if cp.get('analisis'):
                                resultado_final['probabilidades'][tk]['analisis'] = cp['analisis']
                    print(f'[Debate] Consenso aplicado a {len(disputes)} tickers')
            except Exception as e:
                print(f'[!] Debate parse error: {e}')
    except Exception as e:
        print(f'[!] Error en debate: {e}')

# ============================================================
# META-COGNITION REVIEW: Un modelo critica el ensamble
# ============================================================
if len(respuestas_modelos) >= 2 and API_KEY:
    try:
        # Build summary of current predictions
        review_lines = ['REVISA ESTAS PREDICCIONES DEL ENSAMBLE E IDENTIFICA RIESGOS:']
        for t in TICKERS[:10]:  # Review top 10
            p = resultado_final.get('probabilidades', {}).get(t, {})
            if p.get('probabilidad'):
                probs_list = []
                for rm in respuestas_modelos:
                    rp = rm.get('probabilidades', {}).get(t, {}).get('probabilidad')
                    if rp: probs_list.append(rp)
                spread = max(probs_list) - min(probs_list) if len(probs_list) > 1 else 0
                review_lines.append(f'{t}: Prob={p["probabilidad"]}% Conf={p["confianza"]}% Discrepancia={spread}%')
        
        review_prompt = f'''Eres un revisor critico de predicciones financieras con 25 anos de experiencia. Tu trabajo es encontrar fallas.

{chr(10).join(review_lines)}

Para cada ticker, responde:
1. QUE ESTAMOS IGNORANDO? Que factor clave no se esta considerando?
2. CUAL ES EL RIESGO PRINCIPAL? (En 1 oracion)
3. QUE TAN ROBUSTA ES LA PREDICCION? (1-10, donde 1=azar, 10=altamente confiable)

Responde SOLO con JSON: {{"revision": [{{"ticker": "NVDA", "falta": "...", "riesgo": "...", "robustez": 7}}]}}'''
        
        review_raw = llamar_modelo('openrouter/free', review_prompt, sp='Eres un revisor critico senior. Respondes SOLO con JSON valido.')
        try:
            review_data = extraer_json(review_raw)
            if review_data and 'revision' in review_data:
                resultado_final['meta_cognition'] = review_data['revision']
                print(f'[MetaCog] Revision generada para {len(review_data["revision"])} tickers')
        except:
            pass
    except Exception as e:
        print(f'[!] Meta-cognition fallo: {e}')

# ============================================================
# ACTIVE LEARNING: flag low-conviction tickers for labeling
# ============================================================
active_targets = []
for t, p in resultado_final.get('probabilidades', {}).items():
    if p.get('active_learning_flag') and p.get('probabilidad', 50) > 0:
        active_targets.append({
            'ticker': t,
            'probabilidad': p['probabilidad'],
            'confianza': p.get('confianza', 50),
            'cqs': p.get('consensus_quality_score', 50),
            'timestamp': time.strftime('%Y-%m-%dT%H:%M:%S')
        })
resultado_final['active_learning_targets'] = active_targets[:10]
if active_targets:
    print(f'[Active Learning] {len(active_targets)} tickers low-conviction: {", ".join(a["ticker"] for a in active_targets[:5])}...')

# ============================================================
# PESIFICAR NOTICIAS DIRECTAMENTE EN PROBABILIDAD
# prob = prob + sentimiento_score * peso_noticias
# ============================================================
CALIB_PATH = os.path.join(DATA_DIR, 'Datos', 'calibracion.json')
news_weight = 0.0
if os.path.exists(CALIB_PATH):
    try:
        cal = json.load(open(CALIB_PATH))
        nc = cal.get('correlacion_noticias', {})
        prec_pos = nc.get('positivo', {}).get('precision') or 0.5
        prec_neg = nc.get('negativo', {}).get('precision') or 0.5
        # Si precision con noticias positivas es mejor que 50%, dar peso
        if prec_pos > 0.52 or prec_neg > 0.52:
            news_weight = min(max(prec_pos - 0.5, prec_neg - 0.5) * 20, 10)
    except:
        pass

news_adjusted_count = 0
for t in TICKERS:
    ns = news_sentimiento.get(t, {})
    if ns and isinstance(ns, dict) and ns.get('score') is not None and t in resultado_final.get('probabilidades', {}):
        try:
            score = float(ns['score'])
            # Exponential decay by news age (if timestamp available)
            decay_factor = 1.0
            if ns.get('timestamp'):
                try:
                    news_time = datetime.datetime.fromisoformat(ns['timestamp'].replace('Z', '+00:00'))
                    hours_ago = (datetime.datetime.now(datetime.timezone.utc) - news_time).total_seconds() / 3600
                    decay_factor = max(0.1, np.exp(-hours_ago / 24))  # half-life ~16.6 hours
                except:
                    pass
            prob = resultado_final['probabilidades'][t].get('probabilidad', 50)
            adjustment = score * news_weight * decay_factor
            new_prob = prob + adjustment
            resultado_final['probabilidades'][t]['probabilidad_original_sin_noticias'] = prob
            resultado_final['probabilidades'][t]['probabilidad'] = round(max(5, min(95, new_prob)))
            resultado_final['probabilidades'][t]['ajuste_noticias'] = round(adjustment, 2)
            resultado_final['probabilidades'][t]['peso_noticias'] = round(news_weight, 2)
            resultado_final['probabilidades'][t]['decay_noticias'] = round(decay_factor, 2)
            news_adjusted_count += 1
        except:
            pass
if news_adjusted_count > 0:
    print(f'[Noticias] Ajuste aplicado con peso {news_weight:.1f} a {news_adjusted_count} tickers')

# ============================================================
# CALIBRACIÓN PLATT / ISOTÓNICA
# Usa historial de predicciones vs resultados reales
# ============================================================
HIST_PATH = os.path.join(DATA_DIR, 'Datos', 'predicciones_hist.json')
calibrated_count = 0
if os.path.exists(HIST_PATH):
    try:
        hist = json.load(open(HIST_PATH))
        # Collect (prob, outcome) pairs from history where outcome is known
        calibration_data = []
        for t in TICKERS:
            if t in hist:
                for pred in hist[t].get('predicciones', []):
                    if pred.get('acertada') is not None:
                        prob = pred.get('probabilidad', 50) / 100.0
                        outcome = 1 if pred.get('acertada') else 0
                        calibration_data.append((prob, outcome))
        
        if len(calibration_data) >= 50:
            from sklearn.isotonic import IsotonicRegression
            from sklearn.calibration import CalibratedClassifierCV
            import numpy as np
            
            X_cal = np.array([d[0] for d in calibration_data]).reshape(-1, 1)
            y_cal = np.array([d[1] for d in calibration_data])
            
            # Fit isotonic regression (non-parametric, more flexible)
            iso_reg = IsotonicRegression(out_of_bounds='clip')
            iso_reg.fit(X_cal.ravel(), y_cal)
            
            # Apply calibration to current predictions
            for t in TICKERS:
                p = resultado_final.get('probabilidades', {}).get(t, {})
                prob = p.get('probabilidad', 50) / 100.0
                calibrated_prob = iso_reg.predict(np.array([prob]).reshape(1, -1))[0] * 100
                if abs(calibrated_prob - prob * 100) > 1:  # only if significant change
                    p['probabilidad_calibrada'] = round(calibrated_prob)
                    p['probabilidad'] = round(max(5, min(95, calibrated_prob)))
                    calibrated_count += 1
            if calibrated_count > 0:
                print(f'[Calibracion] Isotonica aplicada a {calibrated_count} tickers ({len(calibration_data)} muestras)')
    except Exception as e:
        print(f'[!] Error calibracion: {e}')

# ============================================================
# FEATURE STABILITY MONITORING (detecta cambio en importancia features)
# ============================================================
FEATURE_STABILITY_PATH = os.path.join(DATA_DIR, 'Datos', 'feature_stability.json')
feature_alerts = []
if os.path.exists(MODELO_XGBOOST_PATH):
    try:
        xgb_data = json.load(open(MODELO_XGBOOST_PATH))
        current_fi = xgb_data.get('feature_importance_global', {})
        if current_fi:
            # Load historical feature importance
            hist_fi = {}
            if os.path.exists(FEATURE_STABILITY_PATH):
                with open(FEATURE_STABILITY_PATH) as f:
                    hist_fi = json.load(f)
            
            # Compare top-5 features vs rolling window
            current_top5 = set(list(current_fi.keys())[:5])
            stability_scores = {}
            
            for date_key, fi_snapshot in hist_fi.items():
                if isinstance(fi_snapshot, dict) and fi_snapshot:
                    hist_top5 = set(list(fi_snapshot.keys())[:5])
                    overlap = len(current_top5 & hist_top5)
                    stability_scores[date_key] = overlap / 5.0  # 1.0 = perfect stability
            
            # Current stability vs last 30 days
            recent_dates = sorted(stability_scores.keys())[-30:] if stability_scores else []
            if recent_dates:
                avg_stability = np.mean([stability_scores[d] for d in recent_dates])
                if avg_stability < 0.6:  # Less than 60% feature overlap
                    feature_alerts.append({
                        'type': 'feature_instability',
                        'avg_stability_30d': round(avg_stability, 3),
                        'current_top5': list(current_top5),
                        'message': f'Feature importance inestable: solo {avg_stability:.0%} overlap vs 30d previos'
                    })
            
            # Save current snapshot
            today = time.strftime('%Y-%m-%d')
            hist_fi[today] = current_fi
            # Keep only last 90 days
            if len(hist_fi) > 90:
                hist_fi = {k: v for k, v in sorted(hist_fi.items())[-90:]}
            with open(FEATURE_STABILITY_PATH, 'w') as f:
                json.dump(hist_fi, f, indent=2)
            
            if feature_alerts:
                print(f'[FEATURE ALERT] {len(feature_alerts)} alertas de inestabilidad')
    except Exception as e:
        print(f'[!] Error feature stability: {e}')

# ============================================================
# MONITOREO DRIFT MODELOS (alerta si precision cae >10% rolling 30d)
# ============================================================
DRIFT_THRESHOLD = 0.10
drift_alerts = []
if os.path.exists(HIST_PATH):
    try:
        hist = json.load(open(HIST_PATH))
        for t in TICKERS:
            if t in hist:
                preds = hist[t].get('predicciones', [])
                if len(preds) >= 30:
                    recent = preds[-30:]
                    older = preds[-60:-30] if len(preds) >= 60 else preds[:-30]
                    if older:
                        recent_acc = sum(1 for p in recent if p.get('acertada')) / len(recent)
                        older_acc = sum(1 for p in older if p.get('acertada')) / len(older)
                        drift = older_acc - recent_acc
                        if drift > DRIFT_THRESHOLD:
                            drift_alerts.append({
                                'ticker': t,
                                'drift': round(drift, 3),
                                'recent_acc': round(recent_acc, 3),
                                'older_acc': round(older_acc, 3),
                                'modelo': recent[-1].get('modelo_usado', 'unknown')
                            })
        if drift_alerts:
            print(f'[DRIFT ALERT] {len(drift_alerts)} modelos con degradacion >10%: {[a["ticker"] for a in drift_alerts]}')
            resultado_final['drift_alerts'] = drift_alerts
    except Exception as e:
        print(f'[!] Error drift monitoring: {e}')

# Combine feature alerts with drift alerts
if feature_alerts:
    resultado_final['feature_alerts'] = feature_alerts

resultado_final['timestamp'] = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
resultado_final['total_tickers'] = len(TICKERS)
resultado_final['tickers_analizados'] = TICKERS
resultado_final['fuente_universo'] = 'global' if os.path.exists(SCREENING_PATH) else 'core_30'
resultado_final['regimen_mercado'] = regimen_mercado
resultado_final['modelos_ensemble'] = modelos_exitosos if modelos_exitosos else []

with open(IA_OUTPUT_PATH, 'w') as f:
    json.dump(resultado_final, f, indent=2, ensure_ascii=False)
print(f'[OK] Analisis IA guardado ({resultado_final["modelo_usado"]}) - {len(resultado_final["probabilidades"])} tickers')

# --- Guardar predicciones en historial ---
ahora = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
HIST_PATH = os.path.join(DATA_DIR, 'Datos', 'predicciones_hist.json')
hist_preds = {}
if os.path.exists(HIST_PATH):
    try:
        with open(HIST_PATH) as f:
            hist_preds = json.load(f)
    except: pass

modelo_usado_pred = resultado_final.get('modelo_usado', 'desconocido')

for t in TICKERS:
    if t not in hist_preds:
        hist_preds[t] = {'predicciones': [], 'total': 0, 'aciertos': 0, 'precision': 0.5}
    p = resultado_final.get('probabilidades', {}).get(t, {})
    if not isinstance(p, dict): p = {}
    prob = p.get('probabilidad', PROBS_BASE.get(t, 50))
    direccion = 'up' if prob >= 50 else 'down'
    sent_noticias = news_sentimiento.get(t, {})
    hist_preds[t]['predicciones'].append({
        'fecha': ahora[:10],
        'hora': ahora[11:16],
        'precio_pred': precios.get(t, 0),
        'direccion': direccion,
        'probabilidad': prob,
        'confianza': p.get('confianza', 50),
        'resultado': None,
        'precio_real': None,
        'acertada': None,
        'modelo_usado': modelo_usado_pred,
        'precio_objetivo_30d': p.get('precio_objetivo_30d'),
        'precio_objetivo_3m': p.get('precio_objetivo_3m'),
        'precio_objetivo_6m': p.get('precio_objetivo_6m'),
        'precio_objetivo_1y': p.get('precio_objetivo_1y'),
        'mercado': p.get('mercado', 'US'),
        'news_sentimiento': sent_noticias.get('sentimiento') if sent_noticias else None,
        'news_score': sent_noticias.get('score') if sent_noticias else None
    })
    if len(hist_preds[t]['predicciones']) > 100:
        hist_preds[t]['predicciones'] = hist_preds[t]['predicciones'][-100:]

with open(HIST_PATH, 'w') as f:
    json.dump(hist_preds, f, indent=2, ensure_ascii=False)
print(f'[OK] Predicciones guardadas en historial')
