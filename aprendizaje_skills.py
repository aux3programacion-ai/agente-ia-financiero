import json, os, sys, time, re, math
from collections import defaultdict, Counter

DATA_DIR = os.environ.get('GITHUB_WORKSPACE', '.')
OUTPUT = os.path.join(DATA_DIR, 'Datos', 'skill_memory.json')
SKILL_LOG = os.path.join(DATA_DIR, 'Datos', 'skill_performance.json')
os.makedirs(os.path.join(DATA_DIR, 'Datos'), exist_ok=True)

SECTOR_MAP = {
    'NVDA':'Semiconductors','MU':'Semiconductors','AVGO':'Semiconductors','TSM':'Semiconductors','AMAT':'Semiconductors','LRCX':'Semiconductors',
    'DELL':'Hardware','HPE':'Hardware','NTAP':'Hardware','AAPL':'Hardware',
    'DDOG':'Software','SNOW':'Software','CRWD':'Software','NOW':'Software','OKTA':'Software','PANW':'Software','ORCL':'Software','MSFT':'Software',
    'SMCI':'Hardware','ARM':'Semiconductors','CLS':'Hardware',
    'AMZN':'E-Commerce','GOOGL':'Internet','META':'Internet','UBER':'Transportation',
    'LLY':'Pharma','HON':'Industrials','GE':'Industrials','COST':'Retail','NEE':'Utilities'
}

def load_prediction_history():
    """Load prediction outcomes from aprendizaje.json."""
    path = os.path.join(DATA_DIR, 'Datos', 'aprendizaje.json')
    if not os.path.exists(path):
        return []
    try:
        data = json.load(open(path))
        return data.get('predicciones', [])
    except:
        return []

def extract_skills_from_history(predictions):
    """Mine skills from correct predictions: patterns that led to success."""
    skills = defaultdict(list)
    correct_by_ticker = defaultdict(list)
    incorrect_by_ticker = defaultdict(list)
    
    for p in predictions[-500:]:  # Last 500
        t = p.get('ticker', '')
        correct = p.get('acierto', False)
        prob = p.get('probabilidad', 50)
        outcome = p.get('outcome', 0)
        analisis = p.get('analisis', '')
        regime = p.get('regimen', 'UNKNOWN')
        
        entry = {
            'probabilidad': prob,
            'outcome': outcome,
            'analisis': analisis,
            'regimen': regime,
            'timestamp': p.get('timestamp', ''),
            'feature_used': p.get('feature_used', 'unknown')
        }
        
        if correct:
            correct_by_ticker[t].append(entry)
        else:
            incorrect_by_ticker[t].append(entry)
    
    for t in set(list(correct_by_ticker.keys()) + list(incorrect_by_ticker.keys())):
        sector = SECTOR_MAP.get(t, 'Other')
        correct = correct_by_ticker.get(t, [])
        incorrect = incorrect_by_ticker.get(t, [])
        
        if len(correct) < 3:
            continue
        
        # Success rate
        total = len(correct) + len(incorrect)
        success_rate = len(correct) / total if total > 0 else 0
        
        # Optimal probability range
        if correct:
            probs = [c['probabilidad'] for c in correct]
            avg_prob = sum(probs) / len(probs)
            prob_std = math.sqrt(sum((p - avg_prob)**2 for p in probs) / len(probs)) if len(probs) > 1 else 5
            optimal_low = max(1, avg_prob - prob_std)
            optimal_high = min(99, avg_prob + prob_std)
        else:
            optimal_low, optimal_high = 50, 50
        
        # Regime-specific performance
        regime_perf = defaultdict(lambda: {'correct': 0, 'total': 0})
        for c in correct:
            r = c['regimen']
            regime_perf[r]['correct'] += 1
            regime_perf[r]['total'] += 1
        for ic in incorrect:
            r = ic['regimen']
            regime_perf[r]['total'] += 1
        
        # Extract keywords from correct analyses
        keyword_scores = defaultdict(float)
        for c in correct:
            text = c.get('analisis', '').lower()
            for word in re.findall(r'\b[a-z]{4,}\b', text):
                keyword_scores[word] += 1
        for ic in incorrect:
            text = ic.get('analisis', '').lower()
            for word in re.findall(r'\b[a-z]{4,}\b', text):
                keyword_scores[word] -= 0.5
        
        top_keywords = sorted(keyword_scores.items(), key=lambda x: x[1], reverse=True)[:10]
        signal_words = [w for w, s in top_keywords if s > 1]
        noise_words = [w for w, s in top_keywords if s < -0.5]
        
        # Confidence calibration
        conf_calibration = {}
        for bucket in [(0,30), (30,40), (40,45), (45,55), (55,60), (60,70), (70,100)]:
            lo, hi = bucket
            bucket_correct = sum(1 for c in correct if lo <= c['probabilidad'] < hi)
            bucket_total = bucket_correct + sum(1 for ic in incorrect if lo <= ic['probabilidad'] < hi)
            if bucket_total >= 3:
                conf_calibration[f'{lo}-{hi}'] = round(bucket_correct / bucket_total, 3)
        
        if success_rate >= 0.55:
            skill = {
                'ticker': t,
                'sector': sector,
                'success_rate': round(success_rate, 3),
                'n_correct': len(correct),
                'n_incorrect': len(incorrect),
                'optimal_prob_range': [round(optimal_low, 0), round(optimal_high, 0)],
                'regime_performance': {r: round(v['correct']/v['total'], 3) if v['total'] > 0 else 0 for r, v in regime_perf.items()},
                'signal_keywords': signal_words[:5],
                'noise_keywords': noise_words[:5],
                'confidence_calibration': conf_calibration,
                'skill_level': 'expert' if success_rate > 0.7 else 'advanced' if success_rate > 0.6 else 'developing',
                'last_updated': time.strftime('%Y-%m-%d')
            }
            skills[t] = skill
    
    return dict(skills)

def generate_skill_injection(skills, regime, transfer_notes=None):
    """Generate natural language skill injection for the LLM prompt."""
    if transfer_notes is None:
        transfer_notes = []
    injections = []
    
    # Top-level market wisdom
    if skills:
        avg_success = sum(s['success_rate'] for s in skills.values()) / len(skills)
        if avg_success > 0.6:
            injections.append(f"Nota: Nuestros modelos han mejorado significativamente. La precision promedio es {avg_success:.0%}. Confia mas en tus analisis.")
    
    # Ticker-specific injections
    ticker_injections = []
    for t, s in sorted(skills.items(), key=lambda x: x[1]['success_rate'], reverse=True)[:10]:
        if s['success_rate'] > 0.6 and s['n_correct'] >= 5:
            optimal = s['optimal_prob_range']
            regime_adj = s['regime_performance'].get(regime, s['success_rate'])
            kw = ', '.join(s['signal_keywords'][:3]) if s['signal_keywords'] else ''
            
            note = f"{t}: Nuestra precision historica es {s['success_rate']:.0%} ({s['n_correct']} aciertos). "
            note += f"Rango optimo de probabilidad: {optimal[0]:.0f}-{optimal[1]:.0f}. "
            note += f"Bajo regimen {regime}: precision de {regime_adj:.0%}. "
            if kw:
                note += f"Seniales clave: {kw}."
            ticker_injections.append(note)
    
    if ticker_injections:
        injections.append("APRENDIZAJE HISTORICO (skills adquiridos):")
        injections.extend(ticker_injections)
        injections.append("Usa este aprendizaje para ajustar tus probabilidades.")
    
    # Add cross-ticker knowledge transfer notes
    if transfer_notes:
        injections.append("")
        injections.append("TRANSFERENCIA DE CONOCIMIENTO ENTRE TICKERS:")
        injections.extend(transfer_notes[:5])
    
    return '\n'.join(injections)

def extract_mistake_memory(predictions):
    """Extract detailed post-mortems from wrong predictions."""
    mistakes = []
    for p in predictions[-200:]:
        if not p.get('acierto', True):
            t = p.get('ticker', '')
            prob = p.get('probabilidad', 50)
            conf = p.get('confianza', 50)
            outcome = p.get('outcome', 0)
            analisis = p.get('analisis', '')[:200]
            regime = p.get('regimen', '?')
            timestamp = p.get('timestamp', '')
            
            # Determine how wrong
            direction = 'alcista' if prob > 55 else 'bajista' if prob < 45 else 'neutral'
            actual = 'alza' if outcome == 1 else 'baja'
            severity = abs(prob - 50) * conf / 100  # High conf + far from 50 = severe mistake
            
            if severity > 15:  # Only severe mistakes
                mistakes.append({
                    'ticker': t,
                    'probabilidad': prob,
                    'confianza': conf,
                    'severity': round(severity, 1),
                    'direction': direction,
                    'actual': actual,
                    'contexto': f"Se esperaba {direction} pero fue {actual}. Confianza: {conf}%. Regimen: {regime}. {analisis[:100]}",
                    'timestamp': timestamp,
                    'regimen': regime
                })
    
    mistakes.sort(key=lambda x: x['severity'], reverse=True)
    return mistakes[:15]  # Keep top 15 most severe

def update_meta_prompt(skills, regime):
    """Generate an evolved system prompt based on accumulated skills."""
    if not skills:
        return None
    
    # Count what works
    avg_success = sum(s['success_rate'] for s in skills.values()) / len(skills)
    high_conf_tickers = sum(1 for s in skills.values() if s['success_rate'] > 0.65)
    
    if avg_success > 0.65:
        return f'Eres un analista financiero experto con 20 anos de experiencia. Has demostrado alta precision ({avg_success:.0%}) en {len(skills)} tickers. Confia en tu criterio pero mantente disciplinado. Respondes SOLO con JSON valido, sin markdown, sin explicaciones.'
    elif avg_success > 0.55:
        return f'Eres un analista financiero senior. Tu precision historica es {avg_success:.0%} en {len(skills)} tickers. Eres bueno pero no infalible. Manten conservadurismo en tickers nuevos. Respondes SOLO con JSON valido, sin markdown.'
    else:
        # Still learning - no change
        return None

def cross_ticker_transfer(skills, predictions):
    """Transfer knowledge from high-skill tickers to same-sector peers."""
    if not skills:
        return {}, []
    
    sector_groups = defaultdict(list)
    for t, s in skills.items():
        sector = SECTOR_MAP.get(t, 'Other')
        sector_groups[sector].append((t, s))
    
    transfers = {}
    transfer_notes = []
    
    for sector, members in sector_groups.items():
        if len(members) < 2:
            continue
        # Find the best-performing ticker in the sector
        best = max(members, key=lambda x: x[1]['success_rate'])
        best_ticker, best_skill = best
        if best_skill['success_rate'] < 0.6 or best_skill['n_correct'] < 5:
            continue
        
        for t, s in members:
            if t == best_ticker:
                continue
            # If peer has lower success rate, transfer knowledge
            if s['success_rate'] < best_skill['success_rate'] - 0.1:
                transfers[t] = {
                    'source': best_ticker,
                    'sector': sector,
                    'peer_success_rate': s['success_rate'],
                    'source_success_rate': best_skill['success_rate'],
                    'shared_keywords': list(set(best_skill.get('signal_keywords', [])) - set(s.get('signal_keywords', []))),
                    'source_optimal_range': best_skill['optimal_prob_range']
                }
                transfer_notes.append(
                    f'{t} (sector {sector}): Transferido conocimiento desde {best_ticker}. '
                    f'Usar rango optimo {best_skill["optimal_prob_range"]}. '
                    f'Keywords compartidas: {", ".join(transfers[t]["shared_keywords"][:3])}'
                )
    
    return transfers, transfer_notes

def detect_learning_plateaus(predictions):
    """Detect tickers where accuracy stopped improving (plateau detection)."""
    from collections import defaultdict
    ticker_accuracy_by_week = defaultdict(list)
    
    for p in predictions:
        t = p.get('ticker', '')
        ts = p.get('timestamp', '')
        if not ts or len(ts) < 10:
            continue
        try:
            week = ts[:7]  # YYYY-MM
        except:
            continue
        correct = 1 if p.get('acierto') else 0
        ticker_accuracy_by_week[t].append((week, correct))
    
    plateaus = []
    for t, entries in ticker_accuracy_by_week.items():
        if len(entries) < 20:
            continue
        # Group by month
        monthly = defaultdict(list)
        for w, c in entries:
            monthly[w].append(c)
        
        months = sorted(monthly.keys())
        if len(months) < 3:
            continue
        
        # Compute accuracy per month
        monthly_acc = []
        for m in months[-6:]:  # Last 6 months
            acc = sum(monthly[m]) / len(monthly[m])
            monthly_acc.append(acc)
        
        if len(monthly_acc) >= 3:
            # Detect plateau: last 3 months accuracy is flat (std < 0.03) or declining
            recent = monthly_acc[-3:]
            mean_recent = sum(recent) / len(recent)
            std_recent = (sum((x - mean_recent)**2 for x in recent) / len(recent)) ** 0.5
            trend = recent[-1] - recent[0]
            
            if std_recent < 0.03 and len(monthly_acc) >= 4:
                direction = 'mejorando' if trend > 0 else 'empeorando' if trend < 0 else 'estancado'
                if direction in ('estancado', 'empeorando'):
                    plateaus.append({
                        'ticker': t,
                        'current_accuracy': round(mean_recent, 3),
                        'direction': direction,
                        'months_tracked': len(monthly_acc),
                        'recommendation': 'revisar features' if direction == 'estancado' else 'reentrenar urgente'
                    })
    
    return plateaus

def main():
    print('[Skill Memory] Minando patrones de predicciones exitosas...')
    
    # Load prediction history
    predictions = load_prediction_history()
    if not predictions:
        print('  [!] No hay historial de predicciones. Espera al menos 10 predicciones.')
        result = {'skills': {}, 'injection': '', 'n_predictions': 0, 'timestamp': time.strftime('%Y-%m-%dT%H:%M:%S')}
        with open(OUTPUT, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2)
        return
    
    print(f'  {len(predictions)} predicciones en historial')
    
    # Load regime
    regime = 'UNKNOWN'
    regime_path = os.path.join(DATA_DIR, 'Datos', 'regimen_mercado.json')
    if os.path.exists(regime_path):
        try:
            regime = json.load(open(regime_path)).get('regimen', 'UNKNOWN')
        except:
            pass
    
    # Extract skills
    skills = extract_skills_from_history(predictions)
    print(f'  {len(skills)} tickers con skills adquiridos')
    
    for t, s in sorted(skills.items(), key=lambda x: x[1]['success_rate'], reverse=True)[:5]:
        print(f'    {t}: rate={s["success_rate"]:.0%} ({s["n_correct"]}C/{s["n_incorrect"]}I) rango={s["optimal_prob_range"]} nivel={s["skill_level"]}')
    
    # Generate injection
    injection = generate_skill_injection(skills, regime, transfer_notes)
    if injection:
        print(f'  Injection generada ({len(injection)} chars)')
    
    # Evolve meta-prompt
    meta_evolution = update_meta_prompt(skills, regime)
    if meta_evolution:
        print(f'  Meta-prompt evolucionado ({len(meta_evolution)} chars)')
    
    # Extract mistake memory
    mistakes = extract_mistake_memory(predictions)
    if mistakes:
        print(f'  {len(mistakes)} errores severos registrados en mistake_memory')
    
    # Cross-ticker knowledge transfer
    transfers, transfer_notes = cross_ticker_transfer(skills, predictions)
    if transfer_notes:
        print(f'  {len(transfer_notes)} transferencias cross-ticker generadas')
        for n in transfer_notes[:3]:
            print(f'    {n}')
    
    # Learning curve plateau detection
    plateaus = detect_learning_plateaus(predictions)
    if plateaus:
        print(f'  [!] {len(plateaus)} tickers con aprendizaje estancado')
        for p in plateaus[:3]:
            print(f'    {p["ticker"]}: acc={p["current_accuracy"]:.1%} ({p["direction"]}) -> {p["recommendation"]}')
    
    result = {
        'timestamp': time.strftime('%Y-%m-%dT%H:%M:%S'),
        'skills': skills,
        'recent_mistakes': mistakes,
        'cross_ticker_transfers': transfers,
        'cross_ticker_notes': transfer_notes,
        'learning_plateaus': plateaus,
        'injection': injection,
        'meta_evolution': meta_evolution,
        'n_predictions': len(predictions),
        'regimen': regime,
        'skill_stats': {
            'total_skills': len(skills),
            'avg_success_rate': round(sum(s['success_rate'] for s in skills.values()) / max(len(skills), 1), 3),
            'expert_count': sum(1 for s in skills.values() if s['skill_level'] == 'expert'),
            'advanced_count': sum(1 for s in skills.values() if s['skill_level'] == 'advanced')
        }
    }
    
    with open(OUTPUT, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2)
    
    # Also save mistakes separately for prompt injection
    MISTAKE_OUTPUT = os.path.join(DATA_DIR, 'Datos', 'mistake_memory.json')
    with open(MISTAKE_OUTPUT, 'w', encoding='utf-8') as f:
        json.dump({'recent_mistakes': mistakes, 'timestamp': result['timestamp']}, f, indent=2)
    
    # Track skill performance over time
    perf_path = SKILL_LOG
    perf_history = []
    if os.path.exists(perf_path):
        try: perf_history = json.load(open(perf_path))
        except: pass
    perf_history.append({
        'timestamp': result['timestamp'],
        'n_skills': len(skills),
        'avg_success': result['skill_stats']['avg_success_rate'],
        'n_predictions': len(predictions)
    })
    perf_history = perf_history[-100:]
    with open(perf_path, 'w', encoding='utf-8') as f:
        json.dump(perf_history, f, indent=2)
    
    print(f'[OK] Skills guardados: {len(skills)} skills, promedio {result["skill_stats"]["avg_success_rate"]:.1%} precision')

if __name__ == '__main__':
    main()
