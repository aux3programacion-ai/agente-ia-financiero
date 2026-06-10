#!/usr/bin/env python3
"""
multi_agent_system.py - Multi-Agent System for financial analysis.
Specialized agents (technical, fundamental, macro, sentiment, risk, meta-reviewer)
that debate and produce ensemble predictions with dynamic weighting.
"""
import json
import os
import time
import re
import numpy as np
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional, Tuple, Callable
from collections import defaultdict
from dataclasses import dataclass, asdict

from config.settings import get_setting

DATA_DIR = os.environ.get('GITHUB_WORKSPACE', '.')
OUTPUT_DIR = Path(DATA_DIR) / 'Datos' / 'multi_agent'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


@dataclass
class AgentVote:
    agent_name: str
    ticker: str
    direction: str  # 'up', 'down', 'neutral'
    probability: float
    confidence: float
    reasoning: str
    features_used: List[str]
    regime: str
    timestamp: str = ''
    
    def to_dict(self):
        d = asdict(self)
        return d


@dataclass
class DebateRound:
    ticker: str
    votes: List[AgentVote]
    consensus_direction: str
    consensus_probability: float
    consensus_confidence: float
    disagreement: float
    timestamp: str = ''
    
    def to_dict(self):
        return asdict(self)


class BaseAgent:
    name = 'base'
    description = ''
    
    def __init__(self, weight: float = 1.0):
        self.weight = weight
        self.performance = {'total': 0, 'correct': 0, 'accuracy': 0.5}
        self.accuracy_by_regime = defaultdict(lambda: {'total': 0, 'correct': 0})
        self.confidence_calibration = []
    
    def analyze(self, ticker: str, context: Dict) -> AgentVote:
        raise NotImplementedError
    
    def update_performance(self, vote: AgentVote, actual_direction: str):
        was_correct = vote.direction == actual_direction
        self.performance['total'] += 1
        if was_correct:
            self.performance['correct'] += 1
        self.performance['accuracy'] = self.performance['correct'] / max(self.performance['total'], 1)
        
        regime = vote.regime or 'UNKNOWN'
        self.accuracy_by_regime[regime]['total'] += 1
        if was_correct:
            self.accuracy_by_regime[regime]['correct'] += 1
        
        self.confidence_calibration.append({
            'confidence': vote.confidence,
            'correct': was_correct,
            'ticker': vote.ticker,
            'regime': regime
        })
        self.confidence_calibration = self.confidence_calibration[-200:]


class TechnicalAgent(BaseAgent):
    name = 'tecnico'
    description = 'Chartist: RSI, MACD, volumen, soporte/resistencia, patrones de velas'
    
    def analyze(self, ticker: str, context: Dict) -> AgentVote:
        tec = context.get('tecnico', {})
        if not tec or tec.get('error'):
            return AgentVote(self.name, ticker, 'neutral', 50, 0, 'Sin datos técnicos', [], context.get('regimen', ''))
        
        rsi = tec.get('rsi', 50)
        macd = tec.get('macd', 0)
        vol_ratio = tec.get('vol_ratio', 1)
        dist_soporte = tec.get('dist_soporte_pct', 0)
        dist_resistencia = tec.get('dist_resistencia_pct', 0)
        tendencia = tec.get('tendencia', 'lateral')
        
        signals = []
        prob = 50
        conf = 0
        
        if rsi < 30:
            signals.append(f'RSI={rsi} sobreventa')
            prob += 10
            conf += 15
        elif rsi > 70:
            signals.append(f'RSI={rsi} sobrecompra')
            prob -= 10
            conf += 15
        else:
            signals.append(f'RSI neutral={rsi}')
            conf += 5
        
        if macd > 0:
            signals.append(f'MACD+={macd:.2f}')
            prob += 8
            conf += 10
        else:
            signals.append(f'MACD-={macd:.2f}')
            prob -= 8
            conf += 10
        
        if vol_ratio > 1.5:
            signals.append(f'Volumen alto x{vol_ratio}')
            conf += 10
        elif vol_ratio < 0.5:
            signals.append(f'Volumen bajo x{vol_ratio}')
            conf -= 5
        
        if tendencia == 'uptrend':
            signals.append('Tendencia alcista')
            prob += 5
            conf += 10
        elif tendencia == 'downtrend':
            signals.append('Tendencia bajista')
            prob -= 5
            conf += 10
        
        if abs(dist_soporte) < 3:
            signals.append(f'Cerca de soporte ({dist_soporte}%)')
            prob += 5
        if abs(dist_resistencia) < 3:
            signals.append(f'Cerca de resistencia ({dist_resistencia}%)')
            prob -= 5
        
        direction = 'up' if prob > 55 else 'down' if prob < 45 else 'neutral'
        conf = min(conf, 90)
        prob = max(10, min(90, prob))
        
        return AgentVote(self.name, ticker, direction, prob, conf, 
                        ' | '.join(signals), ['rsi', 'macd', 'vol_ratio', 'tendencia'],
                        context.get('regimen', ''))


class FundamentalAgent(BaseAgent):
    name = 'fundamental'
    description = 'Valor intrínseco: earnings, P/E, crecimiento, márgenes, analyst ratings'
    
    def analyze(self, ticker: str, context: Dict) -> AgentVote:
        ratings = context.get('analyst_ratings', {})
        earnings = context.get('earnings', {})
        
        signals = []
        prob = 50
        conf = 20
        
        consensus_score = ratings.get('consensus_score', 50) if isinstance(ratings, dict) else 50
        if consensus_score > 65:
            signals.append(f'Consenso analistas: {consensus_score:.0f}/100')
            prob += 8
            conf += 15
        elif consensus_score < 35:
            signals.append(f'Consenso analistas bajo: {consensus_score:.0f}/100')
            prob -= 8
            conf += 15
        
        earnings_sent = earnings.get('sentiment', 0) if isinstance(earnings, dict) else 0
        if earnings_sent > 0.3:
            signals.append(f'Earnings positivos: {earnings_sent:.2f}')
            prob += 12
            conf += 20
        elif earnings_sent < -0.3:
            signals.append(f'Earnings negativos: {earnings_sent:.2f}')
            prob -= 12
            conf += 20
        
        direction = 'up' if prob > 55 else 'down' if prob < 45 else 'neutral'
        conf = min(conf, 85)
        
        return AgentVote(self.name, ticker, direction, prob, conf,
                        ' | '.join(signals) if signals else 'Sin datos fundamentales',
                        ['consensus', 'earnings_sentiment'], context.get('regimen', ''))


class MacroAgent(BaseAgent):
    name = 'macro'
    description = 'Contexto macro: tasas, yield curve, VIX, DXY, régimen de mercado'
    
    def analyze(self, ticker: str, context: Dict) -> AgentVote:
        regime = context.get('regimen', 'LATERAL')
        macro = context.get('macro', {})
        riesgo = context.get('riesgo', {})
        
        signals = []
        prob = 50
        conf = 15
        
        regime_map = {'ALCISTA': 8, 'ALCISTA-FUERTE': 15, 'BAJISTA': -8, 'BAJISTA-FUERTE': -15, 'LATERAL': 0}
        regime_adj = regime_map.get(regime.upper(), 0)
        prob += regime_adj
        conf += 10
        
        tec = context.get('tecnico', {})
        spy = tec.get('spy', {})
        spy_trend = spy.get('tendencia_pct', 0) if isinstance(spy, dict) else 0
        if spy_trend > 2:
            signals.append(f'SPY tendencia: +{spy_trend}%')
            prob += 5
            conf += 5
        elif spy_trend < -2:
            signals.append(f'SPY tendencia: {spy_trend}%')
            prob -= 5
            conf += 5
        
        vix = macro.get('VIXCLS', {}).get('value', 15) if isinstance(macro, dict) else 15
        if vix > 30:
            signals.append(f'VIX={vix} miedo extremo')
            prob -= 5
            conf += 10
        elif vix < 14:
            signals.append(f'VIX={vix} complacencia')
            conf += 5
        
        correlation_data = riesgo.get('correlacion', {}) if isinstance(riesgo, dict) else {}
        avg_corr = correlation_data.get('avg_pairwise_corr', 0.5) if isinstance(correlation_data, dict) else 0.5
        if avg_corr > 0.7:
            signals.append(f'Correlación alta={avg_corr:.2f}')
            prob -= 3
            conf += 5
        
        direction = 'up' if prob > 55 else 'down' if prob < 45 else 'neutral'
        conf = min(conf, 80)
        
        return AgentVote(self.name, ticker, direction, prob, conf,
                        ' | '.join(signals) if signals else f'Régimen: {regime}',
                        ['regime', 'vix', 'spy_trend', 'correlation'], regime)


class SentimentAgent(BaseAgent):
    name = 'sentimiento'
    description = 'Análisis de sentimiento: noticias, social media, insider trading'
    
    def analyze(self, ticker: str, context: Dict) -> AgentVote:
        social = context.get('social', {})
        alt_data = context.get('alternative_data', {})
        
        signals = []
        prob = 50
        conf = 10
        
        ticker_social = social.get(ticker, {}) if isinstance(social, dict) else {}
        if isinstance(ticker_social, dict):
            social_score = ticker_social.get('score', 0)
            if social_score > 0.1:
                signals.append(f'Sentimiento social: {social_score:.3f}')
                prob += 8
                conf += 12
            elif social_score < -0.1:
                signals.append(f'Sentimiento social: {social_score:.3f}')
                prob -= 8
                conf += 12
        
        insider = alt_data.get('insider', {}).get(ticker, {}) if isinstance(alt_data, dict) else {}
        if isinstance(insider, dict):
            net = insider.get('net_sentiment', 0)
            if net > 0.3:
                signals.append(f'Insider comprando (net={net:.2f})')
                prob += 10
                conf += 15
            elif net < -0.3:
                signals.append(f'Insider vendiendo (net={net:.2f})')
                prob -= 10
                conf += 15
        
        search = alt_data.get('search', {}).get(ticker, {}) if isinstance(alt_data, dict) else {}
        if isinstance(search, dict):
            trend = search.get('trend', 'stable')
            if trend == 'rising':
                signals.append('Búsquedas en aumento')
                prob += 3
            elif trend == 'falling':
                signals.append('Búsquedas en declive')
                prob -= 3
        
        direction = 'up' if prob > 55 else 'down' if prob < 45 else 'neutral'
        conf = min(conf, 75)
        
        return AgentVote(self.name, ticker, direction, prob, conf,
                        ' | '.join(signals) if signals else 'Sin datos de sentimiento',
                        ['social_score', 'insider_net', 'search_trend'], context.get('regimen', ''))


class RiskAgent(BaseAgent):
    name = 'riesgo'
    description = 'Risk Manager: VaR, volatilidad, drawdown, correlación, beta'
    
    def analyze(self, ticker: str, context: Dict) -> AgentVote:
        riesgo = context.get('riesgo', {})
        tickers_data = riesgo.get('tickers', {}) if isinstance(riesgo, dict) else {}
        ticker_risk = tickers_data.get(ticker, {}) if isinstance(tickers_data, dict) else {}
        
        signals = []
        prob = 50
        conf = 20
        risk_penalty = 0
        
        if isinstance(ticker_risk, dict):
            var_95 = ticker_risk.get('var_95', 0)
            if var_95 > 0.03:
                signals.append(f'VaR95={var_95:.1%} ALTO')
                risk_penalty += 10
            elif var_95 < 0.01:
                signals.append(f'VaR95={var_95:.1%} bajo')
                risk_penalty -= 5
            
            beta = ticker_risk.get('beta', 1)
            if beta > 1.3:
                signals.append(f'Beta={beta:.2f} alto')
                risk_penalty += 5
            elif beta < 0.7:
                signals.append(f'Beta={beta:.2f} bajo (defensivo)')
                risk_penalty -= 3
            
            vol = ticker_risk.get('vol_anual', 0)
            if vol > 0.4:
                signals.append(f'Vol anual={vol:.0%} ALTA')
                risk_penalty += 8
            elif vol < 0.15:
                signals.append(f'Vol anual={vol:.0%} baja')
                risk_penalty -= 5
        
        max_dd = context.get('max_drawdown', 0)
        if max_dd > 0.2:
            signals.append(f'DD reciente: {max_dd:.1%}')
            risk_penalty += 5
        
        prob -= risk_penalty * 1.5
        conf += min(risk_penalty, 30)
        
        direction = 'down' if risk_penalty > 15 else 'neutral'
        if risk_penalty < 5:
            direction = 'up'
        
        prob = max(10, min(90, prob))
        conf = min(conf, 90)
        
        return AgentVote(self.name, ticker, direction, prob, conf,
                        ' | '.join(signals) if signals else 'Riesgo normal',
                        ['var_95', 'beta', 'vol_anual'], context.get('regimen', ''))


class MetaReviewer:
    name = 'meta_reviewer'
    description = 'Evalúa y pondera votos de agentes según precisión histórica por régimen'
    
    def __init__(self):
        self.agent_weights = {}
        self.consensus_history = []
    
    def evaluate_and_weight(self, votes: List[AgentVote], context: Dict) -> DebateRound:
        regime = context.get('regimen', 'INCIERTO')
        weighted_probs = []
        weighted_confs = []
        total_weight = 0
        
        for vote in votes:
            agent_name = vote.agent_name
            base_weight = self.agent_weights.get(agent_name, {}).get('weight', 1.0)
            
            regime_accuracy = self.agent_weights.get(agent_name, {}).get(regime, 0.5)
            regime_factor = 0.5 + regime_accuracy
            dynamic_weight = base_weight * regime_factor
            
            weighted_probs.append((vote.probability, dynamic_weight, vote.direction, vote.confidence))
            total_weight += dynamic_weight
        
        if not weighted_probs:
            return DebateRound('', [], 'neutral', 50, 0, 0)
        
        consensus_prob = sum(p * w for p, w, _, _ in weighted_probs) / max(total_weight, 0.01)
        consensus_conf = sum(c * w for _, w, _, c in weighted_probs) / max(total_weight, 0.01)
        
        directions = [v.direction for v in votes]
        up_count = directions.count('up')
        down_count = directions.count('down')
        total_votes = len(votes)
        
        disagreement = 1 - (max(up_count, down_count) / max(total_votes, 1))
        
        if consensus_prob > 55:
            consensus_dir = 'up'
        elif consensus_prob < 45:
            consensus_dir = 'down'
        else:
            consensus_dir = 'neutral'
        
        timestamp = datetime.now(timezone.utc).isoformat()
        
        debate = DebateRound(
            ticker=votes[0].ticker if votes else '',
            votes=votes,
            consensus_direction=consensus_dir,
            consensus_probability=round(consensus_prob, 1),
            consensus_confidence=round(consensus_conf, 1),
            disagreement=round(disagreement, 3),
            timestamp=timestamp
        )
        
        self.consensus_history.append(debate)
        self.consensus_history = self.consensus_history[-500:]
        
        return debate
    
    def update_weights(self, debate: DebateRound, actual_direction: str):
        for vote in debate.votes:
            agent_name = vote.agent_name
            if agent_name not in self.agent_weights:
                self.agent_weights[agent_name] = {'weight': 1.0, 'total': 0, 'correct': 0}
            
            was_correct = vote.direction == actual_direction
            self.agent_weights[agent_name]['total'] += 1
            if was_correct:
                self.agent_weights[agent_name]['correct'] += 1
            
            accuracy = self.agent_weights[agent_name]['correct'] / max(self.agent_weights[agent_name]['total'], 1)
            self.agent_weights[agent_name]['weight'] = 0.5 + accuracy
            
            regime = vote.regime
            if regime not in self.agent_weights[agent_name]:
                self.agent_weights[agent_name][regime] = {'total': 0, 'correct': 0}
            self.agent_weights[agent_name][regime]['total'] += 1
            if was_correct:
                self.agent_weights[agent_name][regime]['correct'] += 1
        
        self._save_state()
    
    def _save_state(self):
        state = {
            'agent_weights': self.agent_weights,
            'timestamp': datetime.now(timezone.utc).isoformat()
        }
        path = OUTPUT_DIR / 'meta_reviewer_state.json'
        with open(path, 'w') as f:
            json.dump(state, f, indent=2)
    
    def load_state(self):
        path = OUTPUT_DIR / 'meta_reviewer_state.json'
        if path.exists():
            try:
                state = json.loads(path.read_text())
                self.agent_weights = state.get('agent_weights', {})
            except:
                pass


class MultiAgentSystem:
    def __init__(self):
        self.agents: Dict[str, BaseAgent] = {
            'tecnico': TechnicalAgent(),
            'fundamental': FundamentalAgent(),
            'macro': MacroAgent(),
            'sentimiento': SentimentAgent(),
            'riesgo': RiskAgent()
        }
        self.meta_reviewer = MetaReviewer()
        self.meta_reviewer.load_state()
        self.debate_history_path = OUTPUT_DIR / 'debate_history.json'
        self._load_history()
    
    def _load_history(self):
        if self.debate_history_path.exists():
            try:
                self.debate_history = json.loads(self.debate_history_path.read_text())
            except:
                self.debate_history = {'rounds': []}
        else:
            self.debate_history = {'rounds': []}
    
    def _save_history(self):
        self.debate_history['rounds'] = self.debate_history['rounds'][-1000:]
        self.debate_history_path.write_text(json.dumps(self.debate_history, indent=2))
    
    def analyze_ticker(self, ticker: str, context: Dict) -> DebateRound:
        votes = []
        for name, agent in self.agents.items():
            try:
                vote = agent.analyze(ticker, context)
                votes.append(vote)
            except Exception as e:
                print(f'[MAS] Error {name} en {ticker}: {e}')
        
        debate = self.meta_reviewer.evaluate_and_weight(votes, context)
        
        record = debate.to_dict()
        self.debate_history['rounds'].append(record)
        self._save_history()
        
        return debate
    
    def analyze_universe(self, tickers: List[str], context: Dict) -> Dict[str, DebateRound]:
        results = {}
        for ticker in tickers:
            ticker_ctx = {**context}
            for key in list(ticker_ctx.keys()):
                val = ticker_ctx[key]
                if isinstance(val, dict) and ticker in val:
                    ticker_ctx[key] = val[ticker]
            results[ticker] = self.analyze_ticker(ticker, ticker_ctx)
        return results
    
    def update_from_outcomes(self, outcomes: Dict[str, str]):
        """Update agent performance from actual outcomes. outcomes = {ticker: 'up'/'down'}"""
        for round_data in self.debate_history['rounds']:
            ticker = round_data.get('ticker', '')
            if ticker in outcomes:
                actual = outcomes[ticker]
                votes_data = round_data.get('votes', [])
                votes = [AgentVote(**v) for v in votes_data]
                debate = DebateRound(
                    ticker=ticker, votes=votes,
                    consensus_direction=round_data.get('consensus_direction', 'neutral'),
                    consensus_probability=round_data.get('consensus_probability', 50),
                    consensus_confidence=round_data.get('consensus_confidence', 50),
                    disagreement=round_data.get('disagreement', 0)
                )
                self.meta_reviewer.update_weights(debate, actual)
        
        for name, agent in self.agents.items():
            if hasattr(agent, 'performance'):
                perf = agent.performance
                self.debate_history.setdefault('agent_performance', {})[name] = {
                    'accuracy': perf.get('accuracy', 0.5),
                    'total': perf.get('total', 0),
                    'correct': perf.get('correct', 0)
                }
        self._save_history()
    
    def get_consensus_summary(self, ticker: str) -> Dict:
        rounds = [r for r in self.debate_history['rounds'] if r.get('ticker') == ticker]
        if not rounds:
            return {}
        latest = rounds[-1]
        return {
            'ticker': ticker,
            'direction': latest['consensus_direction'],
            'probability': latest['consensus_probability'],
            'confidence': latest['consensus_confidence'],
            'disagreement': latest['disagreement'],
            'n_votes': len(latest['votes']),
            'votes_detail': [{'agent': v['agent_name'], 'direction': v['direction'],
                             'prob': v['probability'], 'conf': v['confidence']}
                            for v in latest['votes']]
        }
    
    def agent_report(self) -> Dict:
        report = {}
        for name, agent in self.agents.items():
            report[name] = {
                'accuracy': agent.performance.get('accuracy', 0.5),
                'total_predictions': agent.performance.get('total', 0),
                'accuracy_by_regime': dict(agent.accuracy_by_regime)
            }
        for name, data in self.debate_history.get('agent_performance', {}).items():
            if name not in report:
                report[name] = data
        return report


_mas = None

def get_multi_agent_system() -> MultiAgentSystem:
    global _mas
    if _mas is None:
        _mas = MultiAgentSystem()
    return _mas


def analyze_with_agents(ticker: str, context: Dict = None) -> Dict:
    mas = get_multi_agent_system()
    if context is None:
        context = _load_default_context()
    debate = mas.analyze_ticker(ticker, context)
    return mas.get_consensus_summary(ticker)


def _load_default_context() -> Dict:
    import glob
    context = {}
    for fname, key in [
        ('analisis_tecnico.json', 'tecnico'),
        ('analisis_social.json', 'social'),
        ('analyst_ratings.json', 'analyst_ratings'),
        ('analisis_riesgo.json', 'riesgo'),
        ('alternative_data.json', 'alternative_data'),
        ('regimen_mercado.json', 'regimen'),
    ]:
        path = Path(DATA_DIR) / 'Datos' / fname
        if path.exists():
            try:
                context[key] = json.loads(path.read_text())
            except:
                pass
    return context


if __name__ == '__main__':
    mas = get_multi_agent_system()
    context = _load_default_context()
    
    for ticker in ['NVDA', 'AAPL', 'MSFT']:
        debate = mas.analyze_ticker(ticker, context)
        summary = mas.get_consensus_summary(ticker)
        print(f'\n[{ticker}] Consensus: {summary["direction"].upper()} '
              f'(prob={summary["probability"]:.0f}%, conf={summary["confidence"]:.0f}%, '
              f'disagreement={summary["disagreement"]:.2f})')
        for v in summary.get('votes_detail', []):
            print(f'  {v["agent"]:12s}: {v["direction"]:8s} prob={v["prob"]:.0f}% conf={v["conf"]:.0f}%')
    
    print(f'\nAgent performance:')
    for name, data in mas.agent_report().items():
        print(f'  {name}: acc={data["accuracy"]:.1%} ({data["total_predictions"]} preds)')