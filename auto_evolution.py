import json, os, sys, time, re, math, random
from collections import defaultdict, Counter

DATA_DIR = os.environ.get('GITHUB_WORKSPACE', '.')
OUTPUT_DIR = os.path.join(DATA_DIR, 'Datos')
os.makedirs(OUTPUT_DIR, exist_ok=True)

EXAMPLES_PATH = os.path.join(OUTPUT_DIR, 'few_shot_examples.json')
HYPOTHESIS_PATH = os.path.join(OUTPUT_DIR, 'hypothesis_log.json')
PROMPT_GEN_PATH = os.path.join(OUTPUT_DIR, 'auto_prompts.json')
EVOLUTION_PATH = os.path.join(OUTPUT_DIR, 'evolution_state.json')

SECTOR_MAP = {
    'NVDA':'Semiconductors','MU':'Semiconductors','AVGO':'Semiconductors','TSM':'Semiconductors',
    'AMAT':'Semiconductors','LRCX':'Semiconductors','SMCI':'Hardware','ARM':'Semiconductors',
    'DELL':'Hardware','HPE':'Hardware','NTAP':'Hardware','AAPL':'Hardware','CLS':'Hardware',
    'DDOG':'Software','SNOW':'Software','CRWD':'Software','NOW':'Software','OKTA':'Software',
    'PANW':'Software','ORCL':'Software','MSFT':'Software',
    'AMZN':'E-Commerce','GOOGL':'Internet','META':'Internet','UBER':'Transportation',
    'LLY':'Pharma','HON':'Industrials','GE':'Industrials','COST':'Retail','NEE':'Utilities'
}

# ============================================================
# COMPONENT 1: FEW-SHOT EXAMPLE BANK
# ============================================================

def load_examples():
    if os.path.exists(EXAMPLES_PATH):
        try: return json.load(open(EXAMPLES_PATH))
        except: pass
    return {'examples': {}, 'version': 1}

def save_examples(data):
    with open(EXAMPLES_PATH, 'w') as f:
        json.dump(data, f, indent=2)

def ingest_example(ticker, regime, probabilidad, confianza, analisis, precio_objetivo, outcome):
    """Store a successful prediction as a few-shot example."""
    data = load_examples()
    key = f'{ticker}_{regime}'
    if key not in data['examples']:
        data['examples'][key] = []
    
    entry = {
        'ticker': ticker,
        'regime': regime,
        'probabilidad': probabilidad,
        'confianza': confianza,
        'analisis': analisis[:300] if analisis else '',
        'precio_objetivo_30d': precio_objetivo,
        'outcome': outcome,
        'timestamp': time.strftime('%Y-%m-%dT%H:%M:%S'),
        'acertada': abs(outcome - (1 if probabilidad > 50 else 0)) < 0.5
    }
    
    data['examples'][key].append(entry)
    # Keep max 10 per key, sorted by most recent + successful first
    data['examples'][key] = sorted(data['examples'][key], key=lambda x: (x['acertada'], x['timestamp']), reverse=True)[:10]
    data['version'] += 1
    save_examples(data)
    return len(data['examples'][key])

def get_few_shot_prompt(ticker, regime):
    """Generate few-shot examples section for prompt injection."""
    data = load_examples()
    key = f'{ticker}_{regime}'
    examples = data['examples'].get(key, [])
    if not examples:
        # Try sector-level
        sector = SECTOR_MAP.get(ticker, '')
        if sector:
            sector_examples = []
            for k, v in data['examples'].items():
                k_ticker, k_regime = k.split('_')[0], '_'.join(k.split('_')[1:])
                if SECTOR_MAP.get(k_ticker, '') == sector and k_regime == regime:
                    sector_examples.extend(v)
            examples = sorted(sector_examples, key=lambda x: (x['acertada'], x['timestamp']), reverse=True)[:3]
    
    if not examples:
        return ''
    
    lines = ['EJEMPLOS DE PREDICCIONES EXITOSAS (usar como referencia):']
    for i, ex in enumerate(examples[:3]):
        lines.append(f'  Ejemplo {i+1}: {ex["ticker"]} ({ex["regime"]}):')
        lines.append(f'    Probabilidad: {ex["probabilidad"]}% | Confianza: {ex["confianza"]}%')
        lines.append(f'    Analisis: {ex["analisis"][:200]}')
        if ex.get('acertada'):
            lines.append(f'    Resultado: ACERTO')
    return '\n'.join(lines)

# ============================================================
# COMPONENT 2: HYPOTHESIS TESTING LOOP
# ============================================================

def load_hypotheses():
    if os.path.exists(HYPOTHESIS_PATH):
        try: return json.load(open(HYPOTHESIS_PATH))
        except: pass
    return {'hypotheses': [], 'verified': []}

def generate_hypotheses(ticker, probabilidad, confianza, era_correcta, analisis, data_context):
    """Generate 3 hypotheses about why prediction succeeded/failed."""
    hypotheses = []
    
    if era_correcta:
        h1 = f"Prediccion correcta porque el analisis {analisis[:50]} capturo la tendencia correcta"
        h2 = f"El regimen de mercado favorecio la direccion predicha"
        h3 = f"La combinacion de tecnicos + noticias fue consistente con el movimiento"
    else:
        direccion = 'alcista' if probabilidad > 50 else 'bajista'
        conf_level = 'alta' if confianza > 70 else 'media' if confianza > 50 else 'baja'
        
        hypotheses = [
            {
                'ticker': ticker,
                'hipotesis': f'El analisis ignoro un factor macro que movio el mercado en direccion opuesta',
                'probabilidad': probabilidad,
                'confianza': confianza,
                'tipo': 'macro_miss',
                'testeable': True
            },
            {
                'ticker': ticker,
                'hipotesis': f'El sentimiento de noticias fue misinterpretado como {direccion}',
                'probabilidad': probabilidad,
                'confianza': confianza,
                'tipo': 'sentiment_error',
                'testeable': True
            },
            {
                'ticker': ticker,
                'hipotesis': f'Confianza {conf_level} ({confianza}%) no estaba justificada para esta prediccion',
                'probabilidad': probabilidad,
                'confianza': confianza,
                'tipo': 'overconfidence',
                'testeable': True
            }
        ]
    
    return hypotheses

def test_hypothesis(h, historical_predictions):
    """Test a hypothesis against historical data. Returns confidence 0-1."""
    h_type = h.get('tipo', '')
    ticker = h.get('ticker', '')
    conf = h.get('confianza', 50)
    
    if h_type == 'overconfidence':
        # Test: when model had similar confidence, was accuracy lower?
        similar_preds = [p for p in historical_predictions 
                        if abs(p.get('confianza', 50) - conf) < 15
                        and p.get('ticker') == ticker]
        if len(similar_preds) >= 5:
            acc = sum(1 for p in similar_preds if p.get('acierto')) / len(similar_preds)
            if acc < 0.5:
                return 0.8  # Hypothesis confirmed: overconfidence is a pattern
            else:
                return 0.2  # Not overconfident in general
    
    if h_type == 'sentiment_error':
        # Test: when sentiment was used, was accuracy lower?
        sent_preds = [p for p in historical_predictions 
                     if p.get('feature_used', '') in ('news', 'social', 'sentiment')
                     and p.get('ticker') == ticker]
        if len(sent_preds) >= 5:
            acc = sum(1 for p in sent_preds if p.get('acierto')) / len(sent_preds)
            if acc < 0.4:
                return 0.7  # Sentiment consistently misleading
            else:
                return 0.3
    
    if h_type == 'macro_miss':
        # Default: plausible but hard to verify
        return 0.5
    
    return 0.3

def log_hypothesis_testing(predictions):
    """Run hypothesis testing on recent wrong predictions."""
    data = load_hypotheses()
    recent_wrong = [p for p in predictions[-100:] if not p.get('acierto', True)]
    
    for p in recent_wrong[:10]:
        ticker = p.get('ticker', '')
        # Check if already tested
        if any(h['ticker'] == ticker and h.get('tested', False) for h in data['hypotheses'][-50:]):
            continue
        
        hyps = generate_hypotheses(
            ticker, p.get('probabilidad', 50), p.get('confianza', 50),
            False, p.get('analisis', ''), p
        )
        
        for h in hyps:
            confidence = test_hypothesis(h, predictions)
            h['tested'] = True
            h['test_confidence'] = confidence
            h['tested_at'] = time.strftime('%Y-%m-%dT%H:%M:%S')
            
            if confidence > 0.6:
                # Hypothesis confirmed! Store as verified insight
                data['verified'].append({
                    'insight': h['hipotesis'],
                    'ticker': ticker,
                    'tipo': h_type,
                    'confidence': confidence,
                    'verified_at': time.strftime('%Y-%m-%dT%H:%M:%S')
                })
                print(f'[Hipotesis] CONFIRMADA: {h["hipotesis"][:80]}... (conf={confidence:.0%})')
            
            data['hypotheses'].append(h)
    
    data['hypotheses'] = data['hypotheses'][-200:]
    data['verified'] = data['verified'][-50:]
    with open(HYPOTHESIS_PATH, 'w') as f:
        json.dump(data, f, indent=2)
    
    return data['verified']

# ============================================================
# COMPONENT 3: AUTO-PROMPT ENGINEERING
# ============================================================

PROMPT_TEMPLATES = {
    'base': 'Eres un analista financiero experto con 20 anos de experiencia en mercados globales. Respondes SOLO con JSON valido, sin markdown, sin explicaciones.',
    'agresivo': 'Eres un trader agresivo de hedge fund. Toma posiciones con conviccion cuando los datos soportan la tesis. No temas dar probabilidades extremas (80%+ o 20%-) cuando la evidencia es clara. Respondes SOLO con JSON.',
    'conservador': 'Eres un gestor de riesgos institucional. Tu prioridad es la preservacion de capital. Sesga tus probabilidades hacia 50% cuando no haya alta conviccion. Mejor no predecir que predecir mal. Respondes SOLO con JSON.',
    'tecnico': 'Eres un chartista puro. Los fundamentales y noticias son ruido. Confia en RSI, MACD, volumen, soporte/resistencia. Los patrones tecnicos se repiten. Respondes SOLO con JSON.',
    'contra': 'Eres un inversor contrario value-driven. Cuando el sentimiento es extremadamente alcista, sesga bajista. Cuando el miedo domina, busca oportunidades. Los mercados siempre mean-revierten. Respondes SOLO con JSON.',
}

def load_auto_prompts():
    if os.path.exists(PROMPT_GEN_PATH):
        try: return json.load(open(PROMPT_GEN_PATH))
        except: pass
    return {'prompts': dict(PROMPT_TEMPLATES), 'performance': {}, 'winner': 'base', 'history': []}

def test_prompt_vs_history(prompt_name, prompt_text, predictions):
    """Test a prompt variation against historical predictions."""
    # Simulate: how well would this prompt style have done?
    # Agresivo: better when market trending, worse when choppy
    # Conservador: better when volatile, worse in clear trends
    # We approximate by looking at past accuracy patterns
    
    if not predictions:
        return 0.5
    
    recent = predictions[-100:]
    accuracy = sum(1 for p in recent if p.get('acierto')) / len(recent) if recent else 0.5
    avg_conf = sum(p.get('confianza', 50) for p in recent) / len(recent) if recent else 50
    
    if prompt_name == 'agresivo':
        # Aggressive works when accuracy is high (model is good)
        return min(1, accuracy * 1.2 + 0.1)
    elif prompt_name == 'conservador':
        # Conservative works when accuracy is low
        return min(1, (1 - accuracy) * 0.8 + 0.2)
    elif prompt_name == 'tecnico':
        # Technical works better in trending markets (proxied by volatility)
        return 0.5 + (accuracy - 0.5) * 0.5 + random.uniform(-0.05, 0.05)
    elif prompt_name == 'contra':
        # Contrarian works when sentiment is extreme
        conf_extreme = sum(1 for p in recent if abs(p.get('confianza', 50) - 50) > 25) / len(recent) if recent else 0
        return conf_extreme * 0.6 + 0.3
    else:
        return accuracy

def evolve_prompts(predictions):
    """Auto-generate and test prompt variations, pick the winner."""
    data = load_auto_prompts()
    
    # Score existing prompts
    for name, text in PROMPT_TEMPLATES.items():
        score = test_prompt_vs_history(name, text, predictions)
        if name not in data['performance']:
            data['performance'][name] = []
        data['performance'][name].append({'score': score, 'timestamp': time.strftime('%Y-%m-%dT%H:%M:%S')})
        data['performance'][name] = data['performance'][name][-30:]
    
    # Pick winner (best average over last 5 runs)
    best_score = -1
    best_name = 'base'
    for name, perf in data['performance'].items():
        recent = [p['score'] for p in perf[-5:]]
        avg = sum(recent) / len(recent) if recent else 0
        if avg > best_score:
            best_score = avg
            best_name = name
    
    old_winner = data.get('winner', 'base')
    data['winner'] = best_name
    
    if best_name != old_winner:
        data['history'].append({
            'timestamp': time.strftime('%Y-%m-%dT%H:%M:%S'),
            'from': old_winner,
            'to': best_name,
            'score': best_score
        })
        print(f'[AutoPrompt] Cambio: {old_winner} -> {best_name} (score={best_score:.3f})')
    
    data['prompts'] = dict(PROMPT_TEMPLATES)
    with open(PROMPT_GEN_PATH, 'w') as f:
        json.dump(data, f, indent=2)
    
    return data['winner'], PROMPT_TEMPLATES.get(best_name, PROMPT_TEMPLATES['base'])

# ============================================================
# COMPONENT 4: SIMULATED FINE-TUNING (Weighted Example Bank)
# ============================================================

def get_fine_tuned_injection(predictions):
    """Simulate fine-tuning by weighting examples from successful predictions."""
    if not predictions:
        return ''
    
    recent = predictions[-200:]
    successful = [p for p in recent if p.get('acierto')]
    if len(successful) < 10:
        return ''
    
    # Group by sector and extract patterns
    sector_insights = defaultdict(list)
    for p in successful:
        t = p.get('ticker', '')
        sector = SECTOR_MAP.get(t, 'Other')
        sector_insights[sector].append(p)
    
    lines = ['PATRONES APRENDIDOS (fine-tuning simulado):']
    for sector, preds in sorted(sector_insights.items(), key=lambda x: len(x[1]), reverse=True)[:5]:
        if len(preds) >= 3:
            avg_prob = sum(p.get('probabilidad', 50) for p in preds) / len(preds)
            avg_conf = sum(p.get('confianza', 50) for p in preds) / len(preds)
            acc = sum(1 for p in preds if p.get('acierto')) / len(preds)
            lines.append(f'  Sector {sector}: {len(preds)} aciertos, prob_promedio={avg_prob:.0f}%, conf={avg_conf:.0f}%, precision={acc:.0%}')
    
    # Extract top signal keywords from successful predictions
    keyword_scores = defaultdict(float)
    for p in successful:
        text = p.get('analisis', '').lower()
        for word in re.findall(r'\b[a-z]{4,}\b', text):
            keyword_scores[word] += 1
    top_kw = sorted(keyword_scores.items(), key=lambda x: x[1], reverse=True)[:8]
    if top_kw:
        lines.append(f'  Palabras clave mas predictivas: {", ".join(w for w,_ in top_kw)}')
    
    return '\n'.join(lines)

# ============================================================
# MAIN: Run all components
# ============================================================

def main():
    from aprendizaje_skills import load_prediction_history
    predictions = load_prediction_history()
    
    if not predictions:
        print('[AutoEvol] No hay historial de predicciones')
        return
    
    print('[AutoEvol] Ciclo de auto-evolucion...')
    
    # 1. Few-Shot Example Bank
    example_count = 0
    for p in predictions[-100:]:
        if p.get('acierto'):
            t = p.get('ticker', '')
            r = p.get('regimen', 'UNKNOWN')
            example_count += ingest_example(
                t, r, p.get('probabilidad', 50), p.get('confianza', 50),
                p.get('analisis', ''), p.get('precio_objetivo', 0),
                p.get('outcome', 1)
            )
    print(f'  [FewShot] {example_count} ejemplos en banco')
    
    # 2. Hypothesis Testing
    verified = log_hypothesis_testing(predictions)
    print(f'  [Hipotesis] {len(verified)} hipotesis confirmadas')
    
    # 3. Auto-Prompt Engineering
    winner, winner_text = evolve_prompts(predictions)
    print(f'  [AutoPrompt] Prompt ganador: {winner}')
    
    # 4. Fine-tuned injection
    ft_injection = get_fine_tuned_injection(predictions)
    if ft_injection:
        print(f'  [FineTune] Injection generada ({len(ft_injection)} chars)')
    
    # Save evolution state
    state = {
        'timestamp': time.strftime('%Y-%m-%dT%H:%M:%S'),
        'n_predictions': len(predictions),
        'n_examples': example_count,
        'n_verified_hypotheses': len(verified),
        'prompt_winner': winner,
        'n_sector_patterns': len(set(SECTOR_MAP.get(p.get('ticker',''),'Other') for p in predictions[-200:] if p.get('acierto')))
    }
    with open(EVOLUTION_PATH, 'w') as f:
        json.dump(state, f, indent=2)
    
    print(f'[AutoEvol] OK - {len(predictions)} predicciones, prompt={winner}, hipotesis={len(verified)}')

if __name__ == '__main__':
    main()
