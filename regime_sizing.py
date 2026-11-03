"""
regime_sizing.py - Dimensionamiento de posiciones segun el regimen de mercado.

Extraido de optimizacion_portafolio.py para que la logica sea importable y
testeable sin descargar datos de mercado: aquel modulo ejecuta el script
completo en el import.
"""
import json
import os

DATA_DIR = os.environ.get('GITHUB_WORKSPACE', '.')

REGIME_CONFIG = {
    'ALCISTA':  {'max_position': 0.15, 'leverage': 1.0,  'risk_per_trade': 0.02,  'max_sector': 0.30, 'cash_reserve': 0.05},
    'LATERAL':  {'max_position': 0.08, 'leverage': 0.5,  'risk_per_trade': 0.01,  'max_sector': 0.25, 'cash_reserve': 0.15},
    'BAJISTA':  {'max_position': 0.05, 'leverage': 0.0,  'risk_per_trade': 0.005, 'max_sector': 0.15, 'cash_reserve': 0.30},
    'INCIERTO': {'max_position': 0.06, 'leverage': 0.3,  'risk_per_trade': 0.008, 'max_sector': 0.20, 'cash_reserve': 0.20},
}


def get_regime_params(data_dir: str = DATA_DIR):
    """Lee el regimen de regimen_mercado.json y devuelve sus parametros de sizing."""
    regime_path = os.path.join(data_dir, 'Datos', 'regimen_mercado.json')
    if os.path.exists(regime_path):
        try:
            with open(regime_path) as f:
                regime_data = json.load(f)
            regime = regime_data.get('regimen', 'INCIERTO')
            confianza = regime_data.get('confianza', 0.5)
            target = REGIME_CONFIG.get(regime, REGIME_CONFIG['INCIERTO'])
            # Con confianza baja se mezcla el regimen detectado con INCIERTO.
            if confianza < 0.5:
                base = REGIME_CONFIG['INCIERTO']
                return {k: base[k] * (1 - confianza) + target[k] * confianza for k in base}
            return target
        except (OSError, ValueError, TypeError):
            pass
    return REGIME_CONFIG['INCIERTO']


def apply_regime_sizing(weights, regime_params, portfolio_value: float = 100000):
    """
    Limita cada posicion a max_position, renormaliza y aplica apalancamiento
    y reserva de caja. Devuelve los pesos efectivamente invertibles.
    """
    max_pos = regime_params['max_position']
    leverage = regime_params['leverage']
    cash_reserve = regime_params['cash_reserve']

    capped = {k: min(v, max_pos) for k, v in weights.items()}
    total = sum(capped.values())
    if total > 0:
        invested = (1 - cash_reserve) * leverage
        return {k: v / total * invested for k, v in capped.items()}
    return capped


def cap_weights(weights: dict, max_peso: float) -> dict:
    """Limita cada peso a max_peso sin renormalizar."""
    return {k: min(v, max_peso) for k, v in weights.items()}