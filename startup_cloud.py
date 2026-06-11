#!/usr/bin/env python3
"""startup_cloud.py - Inicializa datos semilla y lanza dashboard para Railway/Render."""
import json, os, sys
from pathlib import Path

DATOS = Path(__file__).parent / 'Datos'
DATOS.mkdir(parents=True, exist_ok=True)

# Generar analisis_ia.json semilla si no existe
IA_PATH = DATOS / 'analisis_ia.json'
if not IA_PATH.exists():
    prob_template = {t: {'probabilidad': 50, 'confianza': 50,
        'analisis': 'Semilla inicial. Ejecuta analisis_ia.py con OPENROUTER_KEY para predicciones reales.',
        'precio_objetivo_30d': 100.0, 'precio_objetivo_3m': 100.0,
        'precio_objetivo_6m': 100.0, 'precio_objetivo_1y': 100.0, 'mercado': 'US'}
        for t in ['NVDA','MU','DELL','AVGO','DDOG','SMCI','SNOW','CRWD','NOW','TSM',
                  'ARM','OKTA','HPE','NTAP','CLS','AAPL','AMZN','GOOGL','META','MSFT',
                  'LLY','AMAT','LRCX','PANW','ORCL','HON','UBER','GE','COST','NEE']}
    seed = {'resumen_mercado': 'Semilla - mercado en monitoreo',
            'modelo_usado': 'semilla-inicial', 'titulares': ['Sistema inicializado'],
            'sectores': {}, 'probabilidades': prob_template,
            'total_tickers': 30, 'timestamp': '2026-01-01T00:00:00Z'}
    IA_PATH.write_text(json.dumps(seed, indent=2), encoding='utf-8')
    print(f'[Seed] Creado {IA_PATH.name} con 30 tickers')

# Generar precios_reales.json semilla si no existe
PRECIOS_PATH = DATOS / 'precios_reales.json'
if not PRECIOS_PATH.exists():
    import yfinance as yf
    tickers = ['NVDA','MU','DELL','AVGO','DDOG','SMCI','SNOW','CRWD','NOW','TSM',
               'ARM','OKTA','HPE','NTAP','CLS','AAPL','AMZN','GOOGL','META','MSFT',
               'LLY','AMAT','LRCX','PANW','ORCL','HON','UBER','GE','COST','NEE',
               'SPY','QQQ','DIA','^VIX','DX-Y.NYB','TLT']
    precios = {}
    for t in tickers:
        try:
            h = yf.Ticker(t).history(period='5d', progress=False)
            if not h.empty:
                prices = h['Close'].tolist()
                precios[t] = {'price': round(float(prices[-1]), 2),
                              'change': round(float(prices[-1] - prices[-2]), 2) if len(prices) > 1 else 0,
                              'pct': round((prices[-1]/prices[-2] - 1)*100, 2) if len(prices) > 1 else 0}
        except:
            precios[t] = {'price': 100.0, 'change': 0, 'pct': 0}
    seed_p = {'fuente': 'yfinance', 'timestamp': '2026-01-01T00:00:00Z', 'precios': precios}
    PRECIOS_PATH.write_text(json.dumps(seed_p, indent=2), encoding='utf-8')
    print(f'[Seed] Creado {PRECIOS_PATH.name} con {len(precios)} tickers')

# Lanzar dashboard
port = os.environ.get('PORT', '8501')
print(f'[Startup] Lanzando dashboard en puerto {port}')
os.execvp(sys.executable, [sys.executable, '-m', 'streamlit', 'run',
    str(Path(__file__).parent / 'mercado_tiempo_real.py'),
    '--server.port', port, '--server.address', '0.0.0.0'])
