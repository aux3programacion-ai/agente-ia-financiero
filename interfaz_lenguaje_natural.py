#!/usr/bin/env python3
"""
interfaz_lenguaje_natural.py - Chat en espanol para el sistema.
Procesamiento de lenguaje natural para consultar el agente financiero.
"""
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import json, os, re
from pathlib import Path

from config.settings import get_setting

DATA_DIR = os.environ.get('GITHUB_WORKSPACE', '.')
OUTPUT_DIR = Path(DATA_DIR) / 'Datos' / 'nlu'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


@dataclass
class NLUQuery:
    raw: str; intent: str; entities: Dict[str, Any]
    tickers: List[str]; confidence: float; timestamp: str


@dataclass
class NLUResponse:
    query: str; intent: str; answer: str; data: Optional[Dict] = None
    chart_data: Optional[Dict] = None; confidence: float = 0.0


class IntentClassifier:
    INTENTS = {
        'consulta_portafolio': r'\b(portafolio|cartera|portfolio|inversion)\b',
        'consulta_senal': r'\b(senal|recomienda|comprar|vender|que.*(hacer|hago|opinas|signal))\b',
        'consulta_riesgo': r'\b(riesgo|var|drawdown|volatilidad|perdida)\b',
        'consulta_rendimiento': r'\b(rendimiento|retorno|ganancia|performance|sharpe)\b',
        'consulta_agentes': r'\b(agentes|equipo|analisis.*multi|votos)\b',
        'consulta_ticker': r'\b([A-Z]{1,5})\b',
        'consulta_tiempo': r'\b(hoy|semana|mes|anual|ultimo)\b',
        'consulta_general': r'\b(como.*esta|status|estado|que.*pasa)\b',
        'ayuda': r'\b(ayuda|help|comandos|que.*puedes)\b',
    }

    def classify(self, text: str) -> Tuple[str, float]:
        text_lower = text.lower()
        for intent, pattern in self.INTENTS.items():
            if intent == 'consulta_ticker':
                continue
            match = re.search(pattern, text_lower)
            if match:
                return intent, 0.7 + len(match.group()) * 0.02
        return 'consulta_general', 0.5

    def extract_tickers(self, text: str) -> List[str]:
        pattern = r'\b([A-Z]{1,5})\b'
        matches = re.findall(pattern, text.upper())
        known = ['NVDA', 'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'TSLA',
                 'SPY', 'QQQ', 'BTC', 'ETH']
        return [m for m in matches if m in known][:3]

    def extract_entities(self, text: str, intent: str) -> Dict:
        entities = {}
        date_map = {'hoy': '1d', 'semana': '5d', 'mes': '21d',
                    'anual': '252d', 'ultimo': '30d', 'year': '252d'}
        for word, period in date_map.items():
            if word in text.lower():
                entities['periodo'] = period
                break
        if 'maximo' in text.lower() or 'max' in text.lower():
            entities['tipo'] = 'max'
        if 'minimo' in text.lower() or 'min' in text.lower():
            entities['tipo'] = 'min'
        return entities


class ResponseGenerator:
    def __init__(self):
        self.data_providers: Dict[str, Callable] = {}

    def register_provider(self, intent: str, provider):
        self.data_providers[intent] = provider

    def generate(self, query: NLUQuery) -> NLUResponse:
        provider = self.data_providers.get(query.intent)
        if provider:
            data = provider(query)
        else:
            data = {'mensaje': f'No tengo datos para: {query.intent}'}
        answer = self._format_answer(query, data)
        return NLUResponse(query=query.raw, intent=query.intent,
            answer=answer, data=data, confidence=query.confidence)

    def _format_answer(self, query: NLUQuery, data: Dict) -> str:
        intent = query.intent
        if intent == 'consulta_portafolio':
            v = data.get('valor', 0)
            r = data.get('retorno', 0)
            return f'Tu portafolio vale ${v:,.0f} con un retorno de {r:.1%}'
        if intent == 'consulta_senal':
            tickers = query.tickers
            if tickers:
                parts = []
                for t in tickers:
                    s = data.get(t, {}).get('senal', 'NEUTRAL')
                    c = data.get(t, {}).get('confianza', 0)
                    parts.append(f'{t}: {s} ({c:.0%} confianza)')
                return 'Senales: ' + ', '.join(parts)
            return 'Para que ticker? Ej: "que hago con NVDA"'
        if intent == 'consulta_riesgo':
            var95 = data.get('var_95', 0)
            dd = data.get('max_drawdown', 0)
            vol = data.get('volatilidad', 0)
            return f'VaR95: {var95:.1%} | Drawdown: {dd:.1%} | Vol: {vol:.1%}'
        if intent == 'consulta_rendimiento':
            s = data.get('sharpe', 0)
            r = data.get('retorno', 0)
            return f'Retorno: {r:.1%} | Sharpe: {s:.2f}'
        if intent == 'ayuda':
            return ('Puedo ayudarte con:\n'
                    '- "como va mi portafolio"\n'
                    '- "que hago con NVDA"\n'
                    '- "cual es el riesgo"\n'
                    '- "que recomiendan los agentes"\n'
                    '- "cual fue el rendimiento"')
        return data.get('mensaje', 'No entendi tu consulta')


def _provider_portfolio(query: NLUQuery) -> Dict:
    return {'valor': 100000 + np.random.randint(-5000, 5000),
            'retorno': np.random.randn() * 0.01, 'cambio': np.random.randn() * 0.02}


def _provider_signals(query: NLUQuery) -> Dict:
    result = {}
    for t in query.tickers or ['NVDA']:
        result[t] = {'senal': np.random.choice(['COMPRA', 'VENTA', 'NEUTRAL']),
                     'confianza': round(0.5 + np.random.random() * 0.4, 2)}
    return result


def _provider_risk(query: NLUQuery) -> Dict:
    return {'var_95': -0.023, 'var_99': -0.041, 'volatilidad': 0.185,
            'max_drawdown': -0.087, 'sharpe': 1.42}


def _provider_performance(query: NLUQuery) -> Dict:
    return {'retorno': 0.187, 'sharpe': 1.42, 'sortino': 1.85, 'max_drawdown': -0.087}


class NaturalLanguageInterface:
    def __init__(self):
        self.classifier = IntentClassifier()
        self.response_gen = ResponseGenerator()
        self.conversation_history: List[Dict] = []
        self._register_providers()

    def _register_providers(self):
        self.response_gen.register_provider('consulta_portafolio', _provider_portfolio)
        self.response_gen.register_provider('consulta_senal', _provider_signals)
        self.response_gen.register_provider('consulta_riesgo', _provider_risk)
        self.response_gen.register_provider('consulta_rendimiento', _provider_performance)
        self.response_gen.register_provider('consulta_agentes', _provider_performance)
        self.response_gen.register_provider('consulta_general', _provider_portfolio)
        self.response_gen.register_provider('ayuda', lambda q: {})

    def ask(self, text: str) -> NLUResponse:
        intent, confidence = self.classifier.classify(text)
        tickers = self.classifier.extract_tickers(text)
        entities = self.classifier.extract_entities(text, intent)
        query = NLUQuery(raw=text, intent=intent, entities=entities,
            tickers=tickers, confidence=confidence,
            timestamp=datetime.now().isoformat())
        response = self.response_gen.generate(query)
        self.conversation_history.append({'query': text, 'response': response.answer,
            'intent': intent, 'timestamp': datetime.now().isoformat()})
        self.conversation_history = self.conversation_history[-50:]
        return response

    def get_history(self) -> List[Dict]:
        return self.conversation_history

    def run_cli(self):
        print('=== Agente Financiero - Chat en Espanol ===')
        print('Escribe tu consulta (o "salir" para terminar)')
        while True:
            text = input('> ').strip()
            if text.lower() in ('salir', 'exit', 'quit'):
                break
            if not text:
                continue
            response = self.ask(text)
            print(f'[{response.intent}] {response.answer}')


if __name__ == '__main__':
    nlu = NaturalLanguageInterface()
    tests = [
        'como va mi portafolio hoy',
        'que hago con NVDA',
        'cual es el riesgo actual',
        'cual fue el rendimiento del mes',
        'que recomiendan los agentes',
        'ayuda',
    ]
    for t in tests:
        r = nlu.ask(t)
        print(f'Q: {t}')
        print(f'A: {r.answer}')
        print()
