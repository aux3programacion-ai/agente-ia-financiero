#!/usr/bin/env python3
"""
analisis_ia.py - Analisis IA via OpenRouter con ensemble multi-modelo.
Inyecta noticias frescas, indicadores tecnicos, regimen de mercado,
y feedback de aprendizaje. Pondera resultados por precision historica.
"""
import json, os, sys, urllib.request, urllib.error, time, re, random

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

PROBS_BASE = {'NVDA':72,'MU':68,'DELL':70,'AVGO':65,'DDOG':63,'SMCI':60,'SNOW':62,
    'CRWD':58,'NOW':55,'TSM':67,'ARM':52,'OKTA':64,'HPE':60,'NTAP':52,'CLS':61,
    'AAPL':58,'AMZN':62,'GOOGL':64,'META':60,'MSFT':63,'LLY':57,'AMAT':59,
    'LRCX':58,'PANW':56,'ORCL':57,'HON':54,'UBER':56,'GE':55,'COST':53,'NEE':55}

CONF_BASE = {'NVDA':60,'MU':58,'DELL':62,'AVGO':60,'DDOG':56,'SMCI':54,'SNOW':57,
    'CRWD':55,'NOW':53,'TSM':65,'ARM':52,'OKTA':58,'HPE':55,'NTAP':52,'CLS':56,
    'AAPL':55,'AMZN':57,'GOOGL':58,'META':55,'MSFT':58,'LLY':53,'AMAT':55,
    'LRCX':54,'PANW':52,'ORCL':53,'HON':50,'UBER':52,'GE':51,'COST':50,'NEE':51}

PRICES_BASE = {'NVDA':218,'MU':970,'DELL':425,'AVGO':420,'DDOG':195,'SMCI':985,'SNOW':254,
    'CRWD':349,'NOW':124,'TSM':197,'ARM':157,'OKTA':122,'HPE':60,'NTAP':209,'CLS':388,
    'AAPL':245,'AMZN':215,'GOOGL':490,'META':620,'MSFT':510,'LLY':890,'AMAT':245,
    'LRCX':290,'PANW':380,'ORCL':175,'HON':235,'UBER':82,'GE':200,'COST':950,'NEE':78}

DATA_DIR = os.environ.get('GITHUB_WORKSPACE', '.')
PRECIOS_PATH = os.path.join(DATA_DIR, 'Datos', 'precios_reales.json')
IA_OUTPUT_PATH = os.path.join(DATA_DIR, 'Datos', 'analisis_ia.json')
TECNICO_PATH = os.path.join(DATA_DIR, 'Datos', 'analisis_tecnico.json')
NEWS_PATH = os.path.join(DATA_DIR, 'Datos', 'noticias_recientes.json')
SCREENING_PATH = os.path.join(DATA_DIR, 'Datos', 'screening_global.json')
os.makedirs(os.path.join(DATA_DIR, 'Datos'), exist_ok=True)

# --- Load portfolio tickers (always included in analysis) ---
PORTAFOLIO_TICKERS = set()
PORTAFOLIO_PATH = os.path.join(DATA_DIR, 'Datos', 'portafolio_usuario.json')
if os.path.exists(PORTAFOLIO_PATH):
    try:
        pf = json.load(open(PORTAFOLIO_PATH))
        if isinstance(pf, list):
            PORTAFOLIO_TICKERS = set(t.upper().strip() for t in pf if isinstance(t, str) and t.strip())
            print(f'[Portafolio] Cargados {len(PORTAFOLIO_TICKERS)} tickers: {", ".join(sorted(PORTAFOLIO_TICKERS))}')
    except Exception as e:
        print(f'[!] Error cargando portafolio_usuario.json: {e}')

# Ensure portfolio tickers have base data entries
for t in PORTAFOLIO_TICKERS:
    if t not in PROBS_BASE:
        PROBS_BASE[t] = 55
        CONF_BASE[t] = 50
        PRICES_BASE[t] = 100

# Load global screening to determine TICKERS to analyze (all markets)
MAX_TICKERS_AI = 120
TICKERS = list(TICKERS_CORE)
if os.path.exists(SCREENING_PATH):
    try:
        scr = json.load(open(SCREENING_PATH))
        top50 = scr.get('top50_tickers', [])
        por_mercado = scr.get('por_mercado', {})

        # Sample proportionally from each market
        sampled = set(TICKERS_CORE)
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
        lines_n = ['\nNOTICIAS RECIENTES (para usar en tu analisis):']
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

SYSTEM_PROMPT = 'Eres un analista financiero experto con 20 anos de experiencia en mercados globales. Respondes SOLO con JSON valido, sin markdown, sin explicaciones.'

feedback_section = ''
if feedback_precision:
    feedback_section = f'''
HISTORIAL DE APRENDIZAJE (precision de predicciones anteriores):
{feedback_precision}

IMPORTANTE: Ajusta tus probabilidades segun tu precision historica.
Si tienes alta precision en un ticker o sector, puedes aumentar la confianza.
Si tienes baja precision, reduce tu confianza y probabilidad.
Usa los rangos de probabilidad para calibrar: si en rango 60-65% tu precision historica es baja, se mas conservador ahi.'''

ticker_list_str = ', '.join(TICKERS)
USER_PROMPT_TEMPLATE = f'''Genera analisis para estos {len(TICKERS)} tickers de mercados globales (US, Mexico, Europa, Asia).
Tickers: {ticker_list_str}
Precios actuales: {texto_precios}
{texto_noticias}
{texto_analisis_portafolio}
{texto_tecnicos}
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

def llamar_modelo(modelo, prompt):
    url = 'https://openrouter.ai/api/v1/chat/completions'
    payload = json.dumps({
        'model': modelo,
        'messages': [
            {'role': 'system', 'content': SYSTEM_PROMPT},
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

def generar_defaults():
    random.seed()
    return {
        'resumen_mercado': 'Mercado global operando en sesion regular.',
        'modelo_usado': 'fallback-local',
        'titulares': ['Analisis global generado por IA', 'Mercados en sesion activa',
            'Datos via yfinance', 'Monitoreo multiplataforma'],
        'sectores': {s: 'Sector en monitoreo' for s in [
            'Semiconductores','Servidores IA','Software IA','Ciberseguridad','Industrial',
            'Financiero','Energia','Consumo','Salud','Utilities','Materiales','Inmobiliario','Global']},
        'probabilidades': {
            t: {'probabilidad': PROBS_BASE.get(t, 50), 'confianza': CONF_BASE.get(t, 50),
                'analisis': 'Analisis basado en datos de mercado.',
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
    print('[!] OPENROUTER_KEY no configurada, usando defaults locales')
    resultado_final = generar_defaults()
    resultado_final['modelo_usado'] = 'no-key'
else:
    respuestas_modelos = []
    modelos_exitosos = []
    prompt_completo = USER_PROMPT_TEMPLATE

    for modelo in MODELOS:
        if len(modelos_exitosos) >= 3:
            break
        try:
            print(f'[IA] Intentando modelo: {modelo}')
            raw = llamar_modelo(modelo, prompt_completo)
            parsed = extraer_json(raw)
            if validar_resultado(parsed):
                parsed['modelo_usado'] = modelo
                respuestas_modelos.append(parsed)
                modelos_exitosos.append(modelo)
                print(f'[OK] {modelo} respondio correctamente')
            else:
                print(f'[!] {modelo} respondio sin probabilidades validas')
        except Exception as e:
            print(f'[!] {modelo} fallo: {str(e)[:80]}')
            continue

    if not respuestas_modelos:
        print('[!] Todos los modelos fallaron, usando defaults locales')
        resultado_final = generar_defaults()
        resultado_final['modelo_usado'] = 'fallback-local'
    else:
        # Ensemble: promediar probabilidades ponderadas por precision historica del modelo
        resultado_final = respuestas_modelos[0].copy()
        pesos_modelo = {}
        for rm in respuestas_modelos:
            m = rm['modelo_usado']
            pesos_modelo[m] = precision_por_modelo.get(m, 0.5)

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
                    w_target30 = sum(targets30) / len(targets30) if targets30 else 0
                    w_target3m = sum(targets3m) / len(targets3m) if targets3m else 0
                    w_target6m = sum(targets6m) / len(targets6m) if targets6m else 0
                    w_target1y = sum(targets1y) / len(targets1y) if targets1y else 0
                    w_mercado = max(set(mercados), key=mercados.count) if mercados else 'US'
                    w_analisis = max(analisis_list, key=lambda a: len(a)) if analisis_list else ''
                    resultado_final.setdefault('probabilidades', {})[t] = {
                        'probabilidad': round(w_prob),
                        'confianza': round(w_conf),
                        'analisis': w_analisis,
                        'precio_objetivo_30d': round(w_target30, 2) if w_target30 else precios.get(t, PRICES_BASE.get(t, 100)),
                        'precio_objetivo_3m': round(w_target3m, 2) if w_target3m else precios.get(t, PRICES_BASE.get(t, 100)),
                        'precio_objetivo_6m': round(w_target6m, 2) if w_target6m else precios.get(t, PRICES_BASE.get(t, 100)),
                        'precio_objetivo_1y': round(w_target1y, 2) if w_target1y else precios.get(t, PRICES_BASE.get(t, 100)),
                        'mercado': w_mercado
                    }

            resultado_final['modelo_usado'] = 'ensemble-' + '+'.join(modelos_exitosos)
            print(f'[Ensemble] {len(respuestas_modelos)} modelos combinados: {", ".join(modelos_exitosos)}')
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
