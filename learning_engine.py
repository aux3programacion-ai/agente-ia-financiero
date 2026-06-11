#!/usr/bin/env python3
"""
learning_engine.py - Motor de aprendizaje integrado.
Resuelve: evaluacion 30d, EWMA, bandit algorithm, RL sizing, feedback loop.
"""
import json, os, math, time, random
import numpy as np
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple
from collections import defaultdict

DATA_DIR = os.environ.get('GITHUB_WORKSPACE', '.')
DATOS = Path(DATA_DIR) / 'Datos'

HIST_PATH = DATOS / 'predicciones_hist.json'
CALIB_PATH = DATOS / 'calibracion.json'
IA_PATH = DATOS / 'analisis_ia.json'
RL_PATH = DATOS / 'model_rl_weights.json'
BANDIT_PATH = DATOS / 'bandit_model.json'
SKILL_PATH = DATOS / 'skill_memory.json'
XGB_PATH = DATOS / 'modelo_xgboost.json'

TICKERS = ['NVDA','MU','DELL','AVGO','DDOG','SMCI','SNOW','CRWD','NOW','TSM',
           'ARM','OKTA','HPE','NTAP','CLS','AAPL','AMZN','GOOGL','META','MSFT',
           'LLY','AMAT','LRCX','PANW','ORCL','HON','UBER','GE','COST','NEE']


# ============================================================
# PRIORIDAD 1: Evaluacion a 30 dias real
# ============================================================
class Evaluador30D:
    """Evalua predicciones contra el precio REAL 20-30 dias despues."""

    def __init__(self):
        self.hist = self._cargar_historial()
        self.precios_historicos = self._cargar_precios_historicos()

    def _cargar_historial(self) -> Dict:
        if HIST_PATH.exists():
            try: return json.loads(HIST_PATH.read_text(encoding='utf-8'))
            except: return {}
        return {}

    def _cargar_precios_historicos(self) -> Dict[str, Dict[str, float]]:
        """Carga historico de precios desde yfinance para evaluacion 30d."""
        precios = {}
        for t in TICKERS:
            try:
                import yfinance as yf
                h = yf.Ticker(t).history(period='6mo', progress=False)
                if not h.empty:
                    precios[t] = {str(k.date()): float(v) for k, v in h['Close'].items()}
            except:
                pass
        return precios

    def evaluar_pendientes(self) -> Dict:
        """Evalua predicciones que tienen >=20 dias de antiguedad."""
        hoy = datetime.now(timezone.utc)
        resultados = {
            'evaluados_30d': 0,
            'acertados_30d': 0,
            'evaluados_1d': 0,
            'acertados_1d': 0,
            'targets_verificados': 0,
            'targets_acertados': 0,
            'error_promedio_target': 0,
        }
        errores_target = []

        for t in TICKERS:
            if t not in self.hist:
                continue
            preds = self.hist[t].get('predicciones', [])

            for p in preds:
                # --- Evaluacion 1 dia (original) ---
                if p.get('resultado') is None and t in self.precios_historicos:
                    fechas = sorted(self.precios_historicos[t].keys())
                    if fechas:
                        precio_hoy = self.precios_historicos[t].get(fechas[-1])
                        if precio_hoy:
                            cambio = 0
                            if len(fechas) >= 2:
                                precio_ayer = self.precios_historicos[t].get(fechas[-2], precio_hoy)
                                cambio = (precio_hoy - precio_ayer) / precio_ayer * 100
                            p['resultado'] = 'up' if cambio >= 0 else 'down'
                            p['precio_real'] = precio_hoy
                            p['acertada'] = p.get('direccion') == p['resultado']
                            p['fecha_evaluacion'] = hoy.strftime('%Y-%m-%d')
                            if p['acertada']:
                                resultados['acertados_1d'] += 1
                            resultados['evaluados_1d'] += 1

                # --- Evaluacion 20-30 dias (NUEVA) ---
                if p.get('resultado_30d') is None and t in self.precios_historicos:
                    try:
                        fecha_pred = datetime.strptime(p['fecha'], '%Y-%m-%d').date()
                    except:
                        continue
                    dias_transcurridos = (hoy.date() - fecha_pred).days
                    if 20 <= dias_transcurridos <= 40:
                        fechas = sorted(self.precios_historicos[t].keys())
                        # Buscar precio ~20-30 dias despues
                        target_date = fecha_pred.isoformat()
                        idx_objetivo = None
                        for i, f in enumerate(fechas):
                            if f >= target_date:
                                idx_objetivo = min(i + 20, len(fechas) - 1)
                                break
                        if idx_objetivo is not None and idx_objetivo < len(fechas):
                            precio_30d = self.precios_historicos[t].get(fechas[idx_objetivo])
                            if precio_30d:
                                precio_inicio = self.precios_historicos[t].get(target_date, p.get('precio_real'))
                                if not precio_inicio:
                                    for f in reversed(fechas[:fechas.index(fechas[idx_objetivo])]):
                                        if f <= target_date:
                                            precio_inicio = self.precios_historicos[t].get(f)
                                            break
                                if precio_inicio and precio_inicio > 0:
                                    cambio_30d = (precio_30d - precio_inicio) / precio_inicio * 100
                                    p['resultado_30d'] = 'up' if cambio_30d >= 0 else 'down'
                                    p['precio_real_30d'] = precio_30d
                                    p['cambio_30d_pct'] = round(cambio_30d, 2)
                                    p['acertada_30d'] = p.get('direccion') == p['resultado_30d']
                                    p['fecha_eval_30d'] = hoy.strftime('%Y-%m-%d')
                                    if p['acertada_30d']:
                                        resultados['acertados_30d'] += 1
                                    resultados['evaluados_30d'] += 1

                # --- Verificar target 30d (precio_objetivo_30d) ---
                if (not p.get('target_verificado') and
                    p.get('precio_objetivo_30d') and
                    p.get('precio_real_30d')):
                    objetivo = p['precio_objetivo_30d']
                    real = p['precio_real_30d']
                    if real > 0 and objetivo > 0:
                        error_pct = abs(real - objetivo) / real
                        p['target_verificado'] = True
                        p['target_error_pct'] = round(error_pct * 100, 1)
                        p['target_acertado'] = error_pct <= 0.10
                        resultados['targets_verificados'] += 1
                        if error_pct <= 0.10:
                            resultados['targets_acertados'] += 1
                        errores_target.append(error_pct)

        if errores_target:
            resultados['error_promedio_target'] = round(np.mean(errores_target) * 100, 1)

        # Guardar historial actualizado
        HIST_PATH.write_text(json.dumps(self.hist, indent=2, ensure_ascii=False), encoding='utf-8')
        return resultados

    def precision_30d_por_ticker(self) -> Dict[str, float]:
        """Precision a 30 dias por ticker para calibracion."""
        prec = {}
        for t in TICKERS:
            preds = self.hist.get(t, {}).get('predicciones', [])
            total = sum(1 for p in preds if p.get('acertada_30d') is not None)
            aciertos = sum(1 for p in preds if p.get('acertada_30d'))
            prec[t] = round(aciertos / total, 4) if total > 0 else 0.5
        return prec


# ============================================================
# PRIORIDAD 2: EWMA weights - Pesos temporales REALES por modelo
# ============================================================
class EWMAWeightManager:
    """Computa pesos EWMA por modelo usando historial real de predicciones."""

    def __init__(self, alpha: float = 0.3):
        self.alpha = alpha
        self.hist = self._cargar_historial()

    def _cargar_historial(self) -> Dict:
        if HIST_PATH.exists():
            try: return json.loads(HIST_PATH.read_text(encoding='utf-8'))
            except: return {}
        return {}

    def compute_weights(self) -> Dict[str, float]:
        """Retorna {modelo: peso_ewma} basado en rendimiento reciente real."""
        outcomes = defaultdict(lambda: {'correct': 0, 'total': 0, 'ewma': 0.5})

        for ticker, data in self.hist.items():
            for p in data.get('predicciones', []):
                model = p.get('modelo_usado', '')
                if not model or 'ensemble' in model or 'trend' in model or 'fallback' in model:
                    continue
                acertada = p.get('acertada_30d')
                if acertada is None:
                    acertada = p.get('acertada')
                if acertada is not None:
                    mo = outcomes[model]
                    mo['total'] += 1
                    reward = 1 if acertada else 0
                    mo['ewma'] = self.alpha * reward + (1 - self.alpha) * mo['ewma']
                    if acertada:
                        mo['correct'] += 1

        # Normalizar y filtrar modelos con pocos datos
        weights = {}
        for model, mo in outcomes.items():
            if mo['total'] >= 3:
                static_prec = mo['correct'] / mo['total'] if mo['total'] > 0 else 0.5
                weights[model] = 0.7 * mo['ewma'] + 0.3 * static_prec
            elif mo['total'] > 0:
                weights[model] = 0.5

        return weights

    def get_ensemble_weights(self, modelos_exitosos: List[str]) -> Dict[str, float]:
        """Retorna pesos para los modelos que respondieron en el ensamble."""
        ewma = self.compute_weights()
        pesos = {}
        for m in modelos_exitosos:
            base = m.split('-')[0] if '-' in m else m
            # Buscar coincidencia parcial
            encontrado = False
            for key, w in ewma.items():
                if base in key or key in base:
                    pesos[m] = w
                    encontrado = True
                    break
            if not encontrado:
                pesos[m] = 0.5  # Default para modelos sin historial
        return pesos


# ============================================================
# PRIORIDAD 3: Thompson Sampling Bandit para seleccion de modelos
# ============================================================
class BanditThompson:
    """
    Thompson Sampling para seleccion de modelos.
    Cada modelo tiene una distribucion Beta(alpha, beta).
    Seleccionamos modelos sampling de su posterior y eligiendo los top-K.
    """

    def __init__(self, epsilon: float = 0.1):
        self.epsilon = epsilon
        self.params = self._cargar()

    def _cargar(self) -> Dict:
        if BANDIT_PATH.exists():
            try: return json.loads(BANDIT_PATH.read_text(encoding='utf-8'))
            except: return {}
        return {}

    def _guardar(self):
        BANDIT_PATH.write_text(json.dumps(self.params, indent=2), encoding='utf-8')

    def registrar_outcome(self, modelo: str, acertada: bool):
        """Actualiza distribucion Beta con el resultado."""
        if modelo not in self.params:
            self.params[modelo] = {'alpha': 1, 'beta': 1, 'total': 0, 'aciertos': 0}
        p = self.params[modelo]
        if acertada:
            p['alpha'] += 1
            p['aciertos'] += 1
        else:
            p['beta'] += 1
        p['total'] += 1
        self._guardar()

    def seleccionar_modelos(self, candidatos: List[str], k: int = 3) -> List[str]:
        """
        Selecciona k modelos usando Thompson Sampling con exploracion epsilon-greedy.
        - Con prob epsilon: exploracion (elige aleatorio)
        - Con prob 1-epsilon: explotacion (samplea de Beta y elige top-k)
        """
        if random.random() < self.epsilon:
            return random.sample(candidatos, min(k, len(candidatos)))

        scores = []
        for m in candidatos:
            if m in self.params:
                p = self.params[m]
                # Sample from Beta distribution
                muestra = np.random.beta(p['alpha'], p['beta'])
            else:
                # Modelo nuevo: prior uniforme
                muestra = np.random.beta(1, 1)
            scores.append((m, muestra))

        scores.sort(key=lambda x: x[1], reverse=True)
        return [s[0] for s in scores[:k]]

    def mejores_modelos(self, n: int = 5) -> List[Tuple[str, float]]:
        """Retorna los n mejores modelos por media de la Beta."""
        scores = []
        for m, p in self.params.items():
            if p['total'] >= 3:
                media = p['alpha'] / (p['alpha'] + p['beta'])
                scores.append((m, media))
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:n]


# ============================================================
# PRIORIDAD 4: RL Position Sizing Bridge
# ============================================================
class RLSizingBridge:
    """Puente para usar RL position sizing en produccion."""

    def __init__(self):
        self.model = None
        self.algo = 'PPO'

    def cargar_o_entrenar(self, tickers: List[str] = None):
        """Carga modelo existente o entrena uno nuevo."""
        rl_path = DATOS / 'rl_sizing' / f'rl_{self.algo.lower()}_sizing.zip'
        try:
            from stable_baselines3 import PPO
            if rl_path.exists():
                self.model = PPO.load(str(rl_path))
                print(f'[RLSizing] Modelo cargado: {rl_path}')
                return
        except ImportError:
            print('[RLSizing] stable-baselines3 no instalado, saltando')
            return

        # Entrenar nuevo modelo
        try:
            from rl_position_sizing import train_rl_sizer
            sizer = train_rl_sizer(algo=self.algo, timesteps=20000, tickers=tickers)
            self.model = sizer.impl.model
            print('[RLSizing] Nuevo modelo entrenado')
        except Exception as e:
            print(f'[RLSizing] Error entrenando: {e}')

    def predecir_pesos(self, senales: Dict[str, float],
                       precios: Dict[str, float],
                       capital: float) -> Dict[str, float]:
        """
        Dadas senales de compra/venta con probabilidad, retorna pesos
        ajustados por RL.
        """
        if self.model is None:
            return senales  # Sin RL, usar senales originales

        pesos_ajustados = {}
        for ticker, prob in senales.items():
            precio = precios.get(ticker, 100)
            if precio <= 0:
                continue
            # RL modula: si prob > 55% COMPRA, si prob < 45% VENTA
            # El peso base es proporcional a (prob - 50) * 2 / 100
            peso_base = (prob - 50) * 2 / 100
            if peso_base > 0.001:
                peso_max = 0.15
                pesos_ajustados[ticker] = min(peso_base, peso_max)
            elif peso_base < -0.001:
                pesos_ajustados[ticker] = max(peso_base, -0.15)
            # RL tambien decide si mantener (no incluir = mantener peso actual)

        return pesos_ajustados


# ============================================================
# PRIORIDAD 5: Feedback Loop Completo 30d
# ============================================================
class FeedbackLoop:
    """Cuando una prediccion se confirma a 30d, actualiza todo el sistema."""

    def __init__(self):
        self.hist = self._cargar_historial()

    def _cargar_historial(self) -> Dict:
        if HIST_PATH.exists():
            try: return json.loads(HIST_PATH.read_text(encoding='utf-8'))
            except: return {}
        return {}

    def actualizar_skills(self):
        """Alimenta skill_memory con predicciones acertadas a 30d."""
        skills = {}
        if SKILL_PATH.exists():
            try: skills = json.loads(SKILL_PATH.read_text(encoding='utf-8'))
            except: skills = {}

        for t in TICKERS:
            preds = self.hist.get(t, {}).get('predicciones', [])
            for p in preds:
                if p.get('acertada_30d') and not p.get('feedback_inyectado'):
                    skill_key = f'{t}_alcista_30d'
                    if skill_key not in skills:
                        skills[skill_key] = []
                    skills[skill_key].append({
                        'fecha': p['fecha'],
                        'probabilidad': p.get('probabilidad', 0),
                        'analisis': p.get('analisis', '')[:200],
                        'modelo': p.get('modelo_usado', ''),
                        'cambio_pct': p.get('cambio_30d_pct', 0),
                    })
                    p['feedback_inyectado'] = True

        # Limitar a 20 ejemplos por skill
        for k in skills:
            skills[k] = skills[k][-20:]

        SKILL_PATH.write_text(json.dumps(skills, indent=2, ensure_ascii=False), encoding='utf-8')
        HIST_PATH.write_text(json.dumps(self.hist, indent=2, ensure_ascii=False), encoding='utf-8')

    def actualizar_rl_weights(self):
        """Actualiza pesos RL con outcomes a 30d."""
        rl = {}
        if RL_PATH.exists():
            try: rl = json.loads(RL_PATH.read_text(encoding='utf-8'))
            except: rl = {}

        model_stats = defaultdict(lambda: {'correct': 0, 'total': 0, 'conf_sum': 0, 'error_conf_sum': 0})
        for t in TICKERS:
            for p in self.hist.get(t, {}).get('predicciones', []):
                acertada = p.get('acertada_30d')
                if acertada is not None:
                    model = p.get('modelo_usado', 'unknown')
                    ms = model_stats[model]
                    ms['total'] += 1
                    conf = p.get('confianza', 50)
                    if acertada:
                        ms['correct'] += 1
                        ms['conf_sum'] += conf
                    else:
                        ms['error_conf_sum'] += conf

        for model, ms in model_stats.items():
            if ms['total'] >= 5:
                accuracy = ms['correct'] / ms['total']
                avg_conf_error = ms['error_conf_sum'] / max(ms['total'] - ms['correct'], 1)
                reward = (accuracy - 0.5) * 2
                penalty = avg_conf_error * (1 - accuracy) * 0.01 if ms['total'] - ms['correct'] > 0 else 0
                old_w = rl.get(model, 0.5)
                new_w = old_w + 0.1 * (reward - penalty)
                rl[model] = max(0.05, min(0.95, new_w))

        RL_PATH.write_text(json.dumps(rl, indent=2), encoding='utf-8')

    def inyectar_feedback_en_prompt(self) -> str:
        """Genera texto de feedback para inyectar en el prompt del LLM."""
        feedback = ['[APRENDIZAJE 30d - RESULTADOS CONFIRMADOS]:']

        # Precision 30d global
        total_30d = 0
        aciertos_30d = 0
        for t in TICKERS:
            for p in self.hist.get(t, {}).get('predicciones', []):
                if p.get('acertada_30d') is not None:
                    total_30d += 1
                    if p['acertada_30d']:
                        aciertos_30d += 1
        if total_30d > 0:
            prec = aciertos_30d / total_30d
            feedback.append(f'  Precision 30d GLOBAL: {prec:.0%} ({aciertos_30d}/{total_30d})')
        else:
            feedback.append('  Aun sin suficientes predicciones evaluadas a 30d')

        # Mejores tickers a 30d
        ticker_prec = {}
        for t in TICKERS:
            total_t = 0
            aciertos_t = 0
            for p in self.hist.get(t, {}).get('predicciones', []):
                if p.get('acertada_30d') is not None:
                    total_t += 1
                    if p['acertada_30d']:
                        aciertos_t += 1
            if total_t >= 3:
                ticker_prec[t] = aciertos_t / total_t

        if ticker_prec:
            mejores = sorted(ticker_prec.items(), key=lambda x: x[1], reverse=True)[:5]
            peores = sorted(ticker_prec.items(), key=lambda x: x[1])[:3]
            feedback.append('  Mejores tickers 30d:')
            for t, p in mejores:
                feedback.append(f'    {t}: {p:.0%}')
            feedback.append('  Tickers a evitar 30d:')
            for t, p in peores:
                feedback.append(f'    {t}: {p:.0%}')

        # Modelos mas precisos
        bandit = BanditThompson()
        mejores_mod = bandit.mejores_modelos(5)
        if mejores_mod:
            feedback.append('  Modelos mas precisos:')
            for m, p in mejores_mod:
                feedback.append(f'    {m}: {p:.0%}')

        return '\n'.join(feedback)


# ============================================================
# ORQUESTADOR PRINCIPAL
# ============================================================
class LearningEngine:
    """Punto unico de entrada para todo el aprendizaje."""

    def __init__(self):
        self.evaluador = Evaluador30D()
        self.ewma = EWMAWeightManager()
        self.bandit = BanditThompson()
        self.rl_sizing = RLSizingBridge()
        self.feedback = FeedbackLoop()

    def ejecutar_ciclo(self) -> Dict:
        """Ejecuta el ciclo completo de aprendizaje."""
        resultados = {}

        # 1. Evaluar predicciones pendientes (1d y 30d)
        eval_result = self.evaluador.evaluar_pendientes()
        resultados['evaluacion'] = eval_result
        print(f'[Learn] Evaluados {eval_result["evaluados_1d"]} 1d, {eval_result["evaluados_30d"]} 30d')
        if eval_result['evaluados_30d'] > 0:
            print(f'[Learn] Precision 30d: {eval_result["acertados_30d"]}/{eval_result["evaluados_30d"]} = {eval_result["acertados_30d"]/max(eval_result["evaluados_30d"],1):.1%}')

        # 2. Si hay nuevos outcomes 30d, ejecutar feedback loop
        if eval_result['evaluados_30d'] > 0:
            self.feedback.actualizar_skills()
            print(f'[Learn] Skills actualizadas con outcomes 30d')
            self.feedback.actualizar_rl_weights()
            print(f'[Learn] RL weights actualizados')

        # 3. Inyectar feedback en analisis_ia.json para el LLM
        feedback_text = self.feedback.inyectar_feedback_en_prompt()
        if IA_PATH.exists():
            try:
                ia = json.loads(IA_PATH.read_text(encoding='utf-8'))
                ia['feedback_aprendizaje_30d'] = feedback_text
                IA_PATH.write_text(json.dumps(ia, indent=2, ensure_ascii=False), encoding='utf-8')
                print(f'[Learn] Feedback 30d inyectado en prompt')
            except Exception as e:
                print(f'[Learn] Error inyectando feedback: {e}')

        resultados['feedback_30d'] = feedback_text

        # 4. Registrar outcomes en bandit (para los modelos que participaron)
        for t in TICKERS:
            for p in self.evaluador.hist.get(t, {}).get('predicciones', []):
                acertada = p.get('acertada_30d')
                if acertada is not None and p.get('modelo_usado'):
                    self.bandit.registrar_outcome(p['modelo_usado'], acertada)

        resultados['bandit_params'] = len(BanditThompson().params)

        # 5. EWMA weights disponibles
        pesos_ewma = self.ewma.compute_weights()
        resultados['ewma_weights_activos'] = len(pesos_ewma)
        if pesos_ewma:
            top_ewma = sorted(pesos_ewma.items(), key=lambda x: x[1], reverse=True)[:3]
            resultados['top_ewma'] = dict(top_ewma)

        return resultados

    def get_ewma_weights(self) -> Dict[str, float]:
        return self.ewma.compute_weights()

    def get_bandit_selection(self, modelos: List[str], k: int = 3) -> List[str]:
        return self.bandit.seleccionar_modelos(modelos, k)

    def get_feedback_text(self) -> str:
        return self.feedback.inyectar_feedback_en_prompt()


# Singleton
_engine = None

def get_engine() -> LearningEngine:
    global _engine
    if _engine is None:
        _engine = LearningEngine()
    return _engine


if __name__ == '__main__':
    print('=== Learning Engine ===')
    engine = get_engine()
    resultados = engine.ejecutar_ciclo()
    print(f'Evaluacion: {resultados.get("evaluacion", {})}')
    print(f'EWMA activos: {resultados.get("ewma_weights_activos", 0)}')
    print(f'Bandit params: {resultados.get("bandit_params", 0)}')

    # Demo bandit
    modelos_demo = ['modelo-a', 'modelo-b', 'modelo-c', 'modelo-d', 'modelo-e']
    for m in modelos_demo:
        engine.bandit.registrar_outcome(m, random.random() > 0.4)
    seleccion = engine.get_bandit_selection(modelos_demo, 3)
    print(f'Bandit selecciono: {seleccion}')

    # Demo EWMA
    pesos = engine.get_ewma_weights()
    print(f'EWMA weights: {pesos}')
