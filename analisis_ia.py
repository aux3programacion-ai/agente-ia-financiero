#!/usr/bin/env python3
"""
analisis_ia.py - Analisis IA via OpenRouter con fallback multi-modelo.
Prueba modelos gratis en cadena hasta obtener JSON valido.
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

TICKERS = ['NVDA','MU','DELL','AVGO','DDOG','SMCI','SNOW','CRWD','NOW','TSM',
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

DATA_DIR = os.environ.get('GITHUB_WORKSPACE', '.')
PRECIOS_PATH = os.path.join(DATA_DIR, 'Datos', 'precios_reales.json')
OUTPUT_PATH = os.path.join(DATA_DIR, 'Datos', 'analisis_ia.json')
os.makedirs(os.path.join(DATA_DIR, 'Datos'), exist_ok=True)

precios = {}
if os.path.exists(PRECIOS_PATH):
    try:
        with open(PRECIOS_PATH) as f:
            pd = json.load(f).get('precios', {})
            for t in TICKERS:
                p = pd.get(t, {})
                precios[t] = p.get('price', PROBS_BASE.get(t, 100))
    except Exception as e:
        print(f'[!] Error leyendo precios: {e}')

texto_precios = ', '.join(f'{t} ${precios.get(t, PROBS_BASE.get(t,100)):.2f}' for t in TICKERS)

# Cargar precision historica para feedback de aprendizaje
feedback_precision = ''
CALIB_PATH = os.path.join(DATA_DIR, 'Datos', 'calibracion.json')
HIST_PATH = os.path.join(DATA_DIR, 'Datos', 'predicciones_hist.json')
if os.path.exists(CALIB_PATH):
    try:
        cal = json.load(open(CALIB_PATH))
        pg = cal.get('precision_global', 0.5)
        te = cal.get('total_evaluado', 0)
        if te > 0:
            lines = [f'Precision global historica: {pg:.1%} ({te} predicciones)']
            for t in TICKERS[:10]:
                f = cal['factores'].get(t, {})
                if f.get('total', 0) > 0:
                    lines.append(f'{t}: precision {f["precision"]:.0%} ({f["aciertos"]}/{f["total"]})')
            feedback_precision = '\n'.join(lines)
    except: pass

SYSTEM_PROMPT = 'Eres un analista financiero experto con 20 anos de experiencia en mercados globales. Respondes SOLO con JSON valido, sin markdown, sin explicaciones.'

feedback_section = ''
if feedback_precision:
    feedback_section = f'''
HISTORIAL DE PRECISION:
{feedback_precision}

IMPORTANTE: Ajusta tus probabilidades segun tu precision historica.
Si tienes alta precision en un ticker, puedes aumentar la confianza.
Si tienes baja precision, reduce tu confianza y probabilidad.'''

USER_PROMPT = f'''Genera analisis para estos 30 tickers. Precios actuales: {texto_precios}{feedback_section}

Responde EXACTAMENTE este JSON sin ningun otro texto:
{{"resumen_mercado":"texto corto de 1-2 oraciones sobre el mercado",
"modelo_usado":"modelo-ia",
"titulares":["headline1","headline2","headline3","headline4","headline5"],
"sectores":{{"Semiconductores":"analisis corto","Servidores IA":"analisis","Software IA":"analisis","Ciberseguridad":"analisis","Almacenamiento":"analisis","Manufactura":"analisis","Consumer Tech":"analisis","Cloud/Commerce":"analisis","Internet/Cloud":"analisis","Social/IA":"analisis","Enterprise/Cloud":"analisis","Farmaceutico":"analisis","Semicon Equip":"analisis","Cloud/Database":"analisis","Industrial":"analisis","Movilidad/Tech":"analisis","Aeroespacial":"analisis","Consumo Defensivo":"analisis","Utilities/Energy":"analisis"}},
"probabilidades":{{}}}}

Cada ticker debe tener: "probabilidad" (0-100 segun momentum y fundamentals), "confianza" (0-100), "analisis" (texto corto).'''

def llamar_modelo(modelo):
    url = 'https://openrouter.ai/api/v1/chat/completions'
    payload = json.dumps({
        'model': modelo,
        'messages': [
            {'role': 'system', 'content': SYSTEM_PROMPT},
            {'role': 'user', 'content': USER_PROMPT}
        ],
        'temperature': 0.1,
        'max_tokens': 2000
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

def generar_defaults():
    random.seed()
    return {
        'resumen_mercado': 'Mercado operando en sesion regular con tendencias mixtas.',
        'modelo_usado': 'fallback-local',
        'titulares': ['Analisis financiero generado por IA', 'Mercado en sesion activa',
            'Datos de precios via yfinance', 'Monitoreo intradia 30 tickers', 'Sectores tecnologicos lideran'],
        'sectores': {s: 'Sector en monitoreo regular' for s in [
            'Semiconductores','Servidores IA','Software IA','Ciberseguridad','Almacenamiento',
            'Manufactura','Consumer Tech','Cloud/Commerce','Internet/Cloud','Social/IA',
            'Enterprise/Cloud','Farmaceutico','Semicon Equip','Cloud/Database','Industrial',
            'Movilidad/Tech','Aeroespacial','Consumo Defensivo','Utilities/Energy']},
        'probabilidades': {
            t: {'probabilidad': PROBS_BASE.get(t, 50), 'confianza': CONF_BASE.get(t, 50),
                'analisis': 'Analisis basado en datos de mercado y tendencias del sector.'}
            for t in TICKERS
        }
    }

if not API_KEY:
    print('[!] OPENROUTER_KEY no configurada, usando defaults locales')
    resultado = generar_defaults()
    resultado['modelo_usado'] = 'no-key'
else:
    resultado = None

if not resultado:
    for modelo in MODELOS:
        try:
            print(f'[IA] Intentando modelo: {modelo}')
            raw = llamar_modelo(modelo)
            parsed = extraer_json(raw)
            if 'probabilidades' in parsed and len(parsed['probabilidades']) > 0 and isinstance(list(parsed['probabilidades'].values())[0], dict):
                resultado = parsed
                resultado['modelo_usado'] = modelo
                print(f'[OK] {modelo} respondio correctamente')
                break
            else:
                print(f'[!] {modelo} respondio sin probabilidades, probando siguiente')
        except Exception as e:
            print(f'[!] {modelo} fallo: {str(e)[:80]}')
            continue

if not resultado:
    print('[!] Todos los modelos fallaron, usando defaults locales')
    resultado = generar_defaults()
    resultado['modelo_usado'] = 'fallback-local'

# Ensure all 30 tickers have probabilities (as dict objects)
for t in TICKERS:
    p = resultado.get('probabilidades', {}).get(t)
    if p is None or not isinstance(p, dict):
        resultado.setdefault('probabilidades', {})[t] = {
            'probabilidad': PROBS_BASE.get(t, 50),
            'confianza': CONF_BASE.get(t, 50),
            'analisis': 'Analisis baseline.'
        }

resultado['timestamp'] = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
resultado['total_tickers'] = len(TICKERS)

with open(OUTPUT_PATH, 'w') as f:
    json.dump(resultado, f, indent=2, ensure_ascii=False)
print(f'[OK] Analisis IA guardado ({resultado["modelo_usado"]}) - {len(resultado["probabilidades"])} tickers')

# Guardar predicciones en base historica
ahora = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
hist_preds = {}
if os.path.exists(HIST_PATH):
    try:
        with open(HIST_PATH) as f:
            hist_preds = json.load(f)
    except: pass

for t in TICKERS:
    if t not in hist_preds:
        hist_preds[t] = {'predicciones': [], 'total': 0, 'aciertos': 0, 'precision': 0.5}
    p = resultado.get('probabilidades', {}).get(t, {})
    if not isinstance(p, dict): p = {}
    prob = p.get('probabilidad', PROBS_BASE.get(t, 50))
    direccion = 'up' if prob >= 50 else 'down'
    hist_preds[t]['predicciones'].append({
        'fecha': ahora[:10],
        'hora': ahora[11:16],
        'precio_pred': precios.get(t, 0),
        'direccion': direccion,
        'probabilidad': prob,
        'confianza': p.get('confianza', 50),
        'resultado': None,
        'precio_real': None,
        'acertada': None
    })
    # Limitar a 100 predicciones por ticker
    if len(hist_preds[t]['predicciones']) > 100:
        hist_preds[t]['predicciones'] = hist_preds[t]['predicciones'][-100:]

with open(HIST_PATH, 'w') as f:
    json.dump(hist_preds, f, indent=2, ensure_ascii=False)
print(f'[OK] Predicciones guardadas en historial')
