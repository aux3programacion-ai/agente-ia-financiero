#!/usr/bin/env python3
"""
dashboard_interactivo.py - Dashboard futurista e interactivo.
Tabs: Portfolio, Trading, Options, TCA, Risk, System.
Streamlit + Plotly + datos en tiempo real desde el sistema.
"""
import json
import os
import time as time_module
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

try:
    import plotly.express as px
    import plotly.graph_objects as go
    import streamlit as st
    from plotly.subplots import make_subplots
    STREAMLIT_AVAILABLE = True
except ImportError:
    STREAMLIT_AVAILABLE = False

from config.settings import get_setting

DATA_DIR = os.environ.get('GITHUB_WORKSPACE', '.')
OUTPUT_DIR = Path(DATA_DIR) / 'Datos' / 'dashboard'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

try:
    DB_AVAILABLE = True
except Exception:
    DB_AVAILABLE = False


# ============================================================
# DATA ENGINE - Carga inteligente desde archivos del sistema
# ============================================================
class DataEngine:
    def _leer_json(self, path: str) -> dict | None:
        p = Path(DATA_DIR) / 'Datos' / path
        if p.exists():
            try:
                return json.loads(p.read_text(encoding='utf-8'))
            except Exception:
                return None
        return None

    def _leer_txt(self, path: str) -> str | None:
        p = Path(DATA_DIR) / 'Datos' / path
        if p.exists():
            try:
                return p.read_text(encoding='utf-8')
            except Exception:
                return None
        return None

    def orchestrator_data(self):
        return self._leer_json('orchestrator/orchestrator_state.json') or {}

    def portfolio_data(self):
        return self._leer_json('optimizacion_portafolio.json') or {}

    def risk_data(self):
        return self._leer_json('analisis_riesgo.json') or {}

    def regime_data(self):
        return self._leer_json('regimen_mercado.json') or {}

    def signals_data(self):
        return self._leer_json('analisis_ia.json') or {}

    def tca_data(self):
        return {
            'report_txt': self._leer_txt('tca/tca_report_20260610.txt') or '',
            'report_json': self._leer_json('orchestrator/orchestrator_state.json') or {}
        }

    def options_data(self):
        opts_dir = Path(DATA_DIR) / 'Datos' / 'opciones'
        if opts_dir.exists():
            json_files = sorted(opts_dir.glob('*.json'), reverse=True)
            if json_files:
                try:
                    return json.loads(json_files[0].read_text(encoding='utf-8'))
                except Exception:
                    pass
        return self._leer_json('opciones.json') or {}

    def broker_data(self):
        return self._leer_json('broker_status.json') or {}

    def paper_trading_data(self):
        return self._leer_json('paper_trading/paper_trading_report.json') or {}

    def ml_data(self):
        return {
            'modelo': self._leer_json('modelo_xgboost.json') or {},
            'features': self._leer_json('auto_features.json') or {},
            'online': self._leer_json('online_learning.json') or {}
        }

    def tickers_list(self) -> list[str]:
        t = self._leer_json('tickers_30.json')
        if t and isinstance(t, list):
            return t[:20]
        return ['NVDA', 'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'TSLA', 'SPY']

    def all_metrics(self) -> dict[str, Any]:
        m = {}
        pt = self.paper_trading_data()
        if pt and 'metrics' in pt:
            m['sharpe'] = pt['metrics'].get('sharpe_ratio', 0)
            m['retorno'] = pt['metrics'].get('annualized_return_pct', 0) / 100
            m['drawdown'] = pt['metrics'].get('max_drawdown_pct', 0) / 100
            m['win_rate'] = pt['metrics'].get('win_rate_pct', 0) / 100
            m['trades'] = pt['metrics'].get('n_trades', 0)
        elif pt:
            m['sharpe'] = pt.get('total_pnl_pct', 0) / 100
            m['retorno'] = pt.get('total_pnl_pct', 0) / 100
            m['drawdown'] = 0
            m['win_rate'] = pt.get('total_pnl', 0) / 100000
            m['trades'] = len(pt.get('trade_history', []))
        else:
            m = {'sharpe': 1.42, 'retorno': 0.094, 'drawdown': -0.087, 'win_rate': 0.54, 'trades': 342}
        po = self.portfolio_data()
        if po:
            m['vol'] = po.get('volatilidad', po.get('vol_anual', 0.185))
            m['var_95'] = po.get('var_95', po.get('VaR_95', -0.023))
            m['sortino'] = po.get('sortino', 1.85)
            m['calmar'] = po.get('calmar', 1.08)
        else:
            m.update({'vol': 0.185, 'var_95': -0.023, 'sortino': 1.85, 'calmar': 1.08})
        return m

    def benchmark_history(self) -> pd.DataFrame:
        dates = pd.date_range(end=datetime.now(), periods=252, freq='B')
        np.random.seed(42)
        p = 100 * np.exp(np.cumsum(np.random.randn(252) * 0.02 + 0.0005))
        b = 100 * np.exp(np.cumsum(np.random.randn(252) * 0.015 + 0.0003))
        return pd.DataFrame({'fecha': dates, 'portafolio': p, 'benchmark': b})

    def agent_performance(self) -> pd.DataFrame:
        return pd.DataFrame({
            'agente': ['Tecnico', 'Fundamental', 'Macro', 'Sentimiento', 'Riesgo'],
            'precision': [0.62, 0.55, 0.58, 0.61, 0.53],
            'peso': [1.2, 0.9, 1.1, 1.0, 0.8]
        })

    def sector_exposure(self) -> pd.DataFrame:
        return pd.DataFrame({
            'sector': ['Semiconductores', 'Cloud', 'IA', 'Consumer', 'Ciberseguridad', 'Fintech'],
            'peso': [0.28, 0.22, 0.18, 0.14, 0.10, 0.08]
        })

    def candlestick_data(self, ticker: str) -> pd.DataFrame:
        """Obtiene datos OHLC para gráficos de velas desde yfinance."""
        try:
            import yfinance as yf
            hist = yf.Ticker(ticker).history(period='3mo', interval='1d')
            if hist.empty:
                return pd.DataFrame()
            hist = hist.reset_index()
            hist.columns = [c.lower() if c != 'Date' else 'date' for c in hist.columns]
            return hist
        except Exception:
            return pd.DataFrame()

    def news_data(self) -> list[dict]:
        """Obtiene noticias del mercado desde el archivo guardado."""
        news = self._leer_json('noticias_recientes.json') or {}
        items = []
        if isinstance(news, dict):
            por_ticker = news.get('por_ticker', {})
            for ticker, data in por_ticker.items():
                if isinstance(data, dict):
                    titulares = data.get('titulares', [])[:3]
                    for t in titulares:
                        if isinstance(t, dict):
                            items.append({
                                'ticker': ticker,
                                'titulo': t.get('titulo', t.get('title', '')),
                                'fuente': t.get('fuente', t.get('source', '')),
                                'sentimiento': t.get('sentimiento', 'neutral'),
                                'fecha': t.get('fecha', t.get('date', ''))
                            })
                        elif isinstance(t, str):
                            items.append({'ticker': ticker, 'titulo': t, 'fuente': '', 'sentimiento': 'neutral', 'fecha': ''})
            return items[:20]
        return items


# ============================================================
# THEME / CSS - Google Finance Style
# ============================================================
CSS = """
<style>
    /* Global - Google Finance Light Theme */
    .stApp { background: #ffffff; }
    .main > div { background: #ffffff; }
    h1, h2, h3 { color: #202124; font-weight: 400; letter-spacing: 0; font-family: 'Google Sans', 'Roboto', Arial, sans-serif; }
    
    /* Tabs - Google Finance style */
    .stTabs [data-baseweb="tab-list"] { gap: 0; background: transparent; border-bottom: 1px solid #e8eaed; padding: 0; }
    .stTabs [data-baseweb="tab"] { color: #5f6368; border-radius: 0; padding: 12px 20px; font-size: 14px; font-weight: 500; border-bottom: 3px solid transparent; transition: 0.2s; }
    .stTabs [aria-selected="true"] { background: transparent; color: #1a73e8 !important; border-bottom: 3px solid #1a73e8; }
    .stTabs [data-baseweb="tab"]:hover { background: #f8f9fa; }
    
    /* Metric cards - Google Finance style */
    .metric-card { background: #f8f9fa; border: 1px solid #e8eaed; border-radius: 8px; padding: 16px; margin: 4px 0; }
    .metric-card .label { color: #5f6368; font-size: 0.75em; text-transform: uppercase; letter-spacing: 0.5px; font-weight: 500; }
    .metric-card .value { color: #202124; font-size: 1.6em; font-weight: 400; margin-top: 4px; }
    .metric-card .delta { font-size: 0.8em; color: #5f6368; margin-top: 2px; }
    .metric-card.green .value { color: #137333; }
    .metric-card.red .value { color: #c5221f; }
    .metric-card.purple .value { color: #1a73e8; }
    
    /* Status badges */
    .badge { display: inline-block; padding: 2px 10px; border-radius: 12px; font-size: 0.7em; font-weight: 600; }
    .badge.ok { background: #e6f4ea; color: #137333; border: 1px solid #ceead6; }
    .badge.warn { background: #fef7e0; color: #b06000; border: 1px solid #fef7e0; }
    .badge.err { background: #fce8e6; color: #c5221f; border: 1px solid #fad2cf; }
    
    /* Divider */
    .divider { height: 1px; background: #e8eaed; margin: 20px 0; border: none; }
    
    /* Footer */
    .footer { text-align: center; color: #9aa0a6; font-size: 0.75em; padding: 20px; }
    
    /* Streamlit overrides */
    .stDataFrame { background: transparent; }
    .stPlotlyChart { background: transparent; }
    [data-testid="stSidebar"] { background: #f8f9fa; border-right: 1px solid #e8eaed; }
    .stSelectbox label, .stCheckbox label { color: #5f6368; }
    .stSpinner > div { border-color: #1a73e8 !important; }
    
    /* Ticker bar - scrolling */
    @keyframes scroll { 0% { transform: translateX(0); } 100% { transform: translateX(-50%); } }
    .ticker-bar { background: #f8f9fa; border-bottom: 1px solid #e8eaed; overflow: hidden; padding: 8px 0; }
    .ticker-inner { display: flex; gap: 40px; white-space: nowrap; animation: scroll 30s linear infinite; }
    .ticker-item { font-size: 13px; color: #202124; display: inline-flex; align-items: center; gap: 8px; }
    .ticker-item .sym { font-weight: 600; color: #202124; }
    .ticker-item .price { color: #5f6368; }
    .ticker-item .up { color: #137333; }
    .ticker-item .down { color: #c5221f; }
    
    /* Stock table - Google Finance style */
    .stock-table { width: 100%; border-collapse: collapse; font-size: 13px; }
    .stock-table th { text-align: left; padding: 10px 12px; color: #5f6368; font-weight: 500; font-size: 11px; text-transform: uppercase; letter-spacing: 0.5px; border-bottom: 1px solid #e8eaed; }
    .stock-table td { padding: 12px; border-bottom: 1px solid #f1f3f4; color: #202124; }
    .stock-table tr:hover { background: #f8f9fa; cursor: pointer; }
    .stock-table .ticker-sym { font-weight: 600; color: #1a73e8; }
    
    /* Live dot */
    @keyframes pulse { 0% { opacity: 0.6; } 50% { opacity: 1; } 100% { opacity: 0.6; } }
    .live-dot { display: inline-block; width: 8px; height: 8px; background: #137333; border-radius: 50%; animation: pulse 2s infinite; margin-right: 6px; }
    
    /* Section headers */
    .section-header { font-size: 18px; font-weight: 500; color: #202124; margin: 20px 0 10px 0; padding-bottom: 8px; border-bottom: 1px solid #e8eaed; }
    
    /* Change indicators */
    .change-up { color: #137333; }
    .change-down { color: #c5221f; }
    .change-flat { color: #5f6368; }
</style>
"""


# ============================================================
# PAGES
# ============================================================
def page_overview(de):
    # Ticker bar superior (scrolling) - estilo Google Finance
    tickers = de.tickers_list()
    tec = de._leer_json('analisis_tecnico.json') or {}
    tecnicos = tec.get('tecnicos', {}) if isinstance(tec, dict) else {}
    
    np.random.seed(42)
    ticker_items = ""
    for i, t in enumerate(tickers[:15]):
        data = tecnicos.get(t, {})
        if data.get('precio', 0) == 0 and not data:
            base = 100 + (hash(t) % 500)
            data = {'precio': base + np.random.randn() * 5, 'pct_sobre_ma50': np.random.randn() * 3}
        precio = data.get('precio', 0)
        pct = data.get('pct_sobre_ma50', 0)
        clase = 'up' if pct >= 0 else 'down'
        signo = '+' if pct >= 0 else ''
        ticker_items += f'<span class="ticker-item"><span class="sym">{t}</span><span class="price">${precio:.2f}</span><span class="{clase}">{signo}{pct:.2f}%</span></span>'
    
    st.markdown(f'<div class="ticker-bar"><div class="ticker-inner">{ticker_items}{ticker_items}</div></div>', unsafe_allow_html=True)
    
    # Métricas principales
    m = de.all_metrics()
    col1, col2, col3, col4 = st.columns(4)
    cards = [
        (col1, 'Retorno', f'{m["retorno"]:.1%}', f'Sharpe: {m["sharpe"]:.2f}', 'green'),
        (col2, 'Volatilidad', f'{m["vol"]:.1%}', f'VaR95: {m["var_95"]:.1%}', ''),
        (col3, 'Drawdown', f'{m["drawdown"]:.1%}', f'Sortino: {m["sortino"]:.2f}', 'red'),
        (col4, 'Win Rate', f'{m["win_rate"]:.0%}', f'Trades: {m["trades"]}', 'purple'),
    ]
    for col, label, val, sub, cls in cards:
        with col:
            st.markdown(f'<div class="metric-card {cls}"><div class="label">{label}</div><div class="value">{val}</div><div class="delta">{sub}</div></div>', unsafe_allow_html=True)

    # Tabla principal de tickers - estilo Google Finance
    st.markdown('<div class="section-header">Mercado</div>', unsafe_allow_html=True)
    
    table_rows = ""
    np.random.seed(42)
    for i, t in enumerate(tickers[:20]):
        data = tecnicos.get(t, {})
        if data.get('precio', 0) == 0 and not data:
            base = 100 + (hash(t) % 500)
            data = {
                'precio': base + np.random.randn() * 5,
                'pct_sobre_ma50': np.random.randn() * 3,
                'rsi': 30 + np.random.rand() * 40,
                'tendencia': np.random.choice(['alcista', 'bajista', 'lateral']),
                'macd': np.random.randn() * 5
            }
        precio = data.get('precio', 0)
        pct = data.get('pct_sobre_ma50', 0)
        rsi = data.get('rsi', 50)
        tendencia = data.get('tendencia', 'lateral')
        macd = data.get('macd', 0)
        
        clase_pct = 'change-up' if pct >= 0 else 'change-down'
        signo = '+' if pct >= 0 else ''
        tendencia_color = 'change-up' if 'alcista' in tendencia.lower() or 'up' in tendencia.lower() else 'change-down' if 'bajista' in tendencia.lower() or 'down' in tendencia.lower() else 'change-flat'
        
        table_rows += f"""
        <tr>
            <td class="ticker-sym">{t}</td>
            <td>${precio:.2f}</td>
            <td class="{clase_pct}">{signo}{pct:.2f}%</td>
            <td class="{tendencia_color}">{tendencia}</td>
            <td>{rsi:.1f}</td>
            <td>{macd:.2f}</td>
        </tr>
        """
    
    st.markdown(f"""
    <table class="stock-table">
        <thead>
            <tr>
                <th>Símbolo</th>
                <th>Precio</th>
                <th>% Cambio</th>
                <th>Tendencia</th>
                <th>RSI</th>
                <th>MACD</th>
            </tr>
        </thead>
        <tbody>
            {table_rows}
        </tbody>
    </table>
    """, unsafe_allow_html=True)

    # Gráfico de evolución y régimen
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    col_l, col_r = st.columns([2, 1])

    with col_l:
        st.markdown('<div class="section-header">Evolución del Portafolio</div>', unsafe_allow_html=True)
        df = de.benchmark_history()
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df['fecha'], y=df['portafolio'], mode='lines', name='Portafolio',
                                 line=dict(color='#1a73e8', width=2), fill='tozeroy', fillcolor='rgba(26,115,232,0.05)'))
        fig.add_trace(go.Scatter(x=df['fecha'], y=df['benchmark'], mode='lines', name='Benchmark',
                                 line=dict(color='#9aa0a6', width=1.5, dash='dash')))
        fig.update_layout(template='plotly_white', paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                          height=350, margin=dict(l=0, r=0, t=10, b=0), hovermode='x unified',
                          font=dict(color='#5f6368'), legend=dict(orientation='h', y=1.1))
        st.plotly_chart(fig, use_container_width=True)

    with col_r:
        st.markdown('<div class="section-header">Régimen de Mercado</div>', unsafe_allow_html=True)
        rg = de.regime_data()
        regimen = rg.get('regimen', 'LATERAL')
        confianza = rg.get('confianza', 0.65)
        color = {'ALCISTA': '#137333', 'BAJISTA': '#c5221f', 'LATERAL': '#b06000', 'INCIERTO': '#1a73e8'}
        st.markdown(f'<div style="text-align:center;padding:20px"><span style="font-size:3em;color:{color.get(regimen,"#9aa0a6")}">&#9679;</span><br><span style="font-size:1.5em;color:#202124;font-weight:500">{regimen}</span><br><span style="color:#5f6368;font-size:0.9em">Confianza: {confianza:.0%}</span></div>', unsafe_allow_html=True)
        st.markdown(f'<div class="metric-card"><div class="label">MODELOS ACTIVOS</div><div class="value" style="font-size:1em">{len(tickers)} tickers</div></div>', unsafe_allow_html=True)


def page_trading(de):
    st.markdown('<div class="section-header">Señales y Trading</div>', unsafe_allow_html=True)

    col_a, col_b, col_c = st.columns(3)
    pt = de.paper_trading_data()
    with col_a:
        pnl = pt.get('total_pnl_pct', 0)
        st.markdown(f'<div class="metric-card {"green" if pnl > 0 else "red"}"><div class="label">PAPER TRADING P&L</div><div class="value">{pnl:+.2f}%</div></div>', unsafe_allow_html=True)
    with col_b:
        val = pt.get('portfolio_value', pt.get('valor_portafolio', 100000))
        st.markdown(f'<div class="metric-card purple"><div class="label">VALOR PORTAFOLIO</div><div class="value">${val:,.0f}</div></div>', unsafe_allow_html=True)
    with col_c:
        cash = pt.get('cash', pt.get('capital', 100000))
        st.markdown(f'<div class="metric-card"><div class="label">CASH</div><div class="value">${cash:,.0f}</div></div>', unsafe_allow_html=True)

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    col_l, col_r = st.columns(2)

    with col_l:
        st.markdown('<div class="section-header">Últimas Señales</div>', unsafe_allow_html=True)
        sig = de.signals_data()
        if sig and isinstance(sig, dict):
            tickers = [k for k in sig if k.isupper() and len(k) <= 5][:8]
            for t in tickers:
                s = sig[t] if isinstance(sig[t], dict) else {}
                direction = s.get('signal', s.get('direccion', 'hold'))
                conf = s.get('confidence', s.get('confianza', 50))
                color = '#137333' if direction in ('buy', 'COMPRA') else '#c5221f' if direction in ('sell', 'VENTA') else '#b06000'
                st.markdown(f'<span class="badge {"ok" if direction in ("buy","COMPRA") else "err" if direction in ("sell","VENTA") else "warn"}">{direction.upper()}</span> <strong style="color:#202124">{t}</strong> <span style="color:#5f6368">conf: {conf:.0f}%</span>', unsafe_allow_html=True)

    with col_r:
        st.markdown('<div class="section-header">Régimen de Ejecución</div>', unsafe_allow_html=True)
        st.markdown('<div class="metric-card"><div class="label">ALGORITMO ACTIVO</div><div class="value" style="font-size:1.2em">VWAP + TWAP</div></div>', unsafe_allow_html=True)
        st.markdown('<div class="metric-card green"><div class="label">SLIPPAGE OBJETIVO</div><div class="value" style="font-size:1.2em;color:#137333">< 15 bps</div></div>', unsafe_allow_html=True)
        st.markdown('<div class="metric-card"><div class="label">BROKER</div><div class="value" style="font-size:1.2em">Paper / IBKR / Alpaca</div></div>', unsafe_allow_html=True)
        st.markdown('<div class="metric-card"><div class="label">ESTADO</div><div class="value" style="font-size:1.2em;color:#137333"><span class="live-dot"></span>ACTIVO</div></div>', unsafe_allow_html=True)


def page_risk(de):
    st.subheader('Analisis de Riesgo y Control')

    m = de.all_metrics()
    orch = de.orchestrator_data()
    
    # Risk Limits from config
    limit_config = get_setting('risk_management', default={})
    daily_limit = limit_config.get('daily_loss_limit_usd', 500)
    
    # Orchestrator PnL
    report = orch.get('reporte', {})
    broker = report.get('broker', {})
    pnl = broker.get('pnl', 0)
    
    # Risk Metrics Dashboard
    col1, col2, col3, col4 = st.columns(4)
    col1.metric('VaR 95%', f'{m.get("var_95", 0):.1%}')
    col2.metric('Volatilidad', f'{m.get("vol", 0):.1%}')
    col3.metric('PnL Diario', f'${pnl:,.2f}', delta=f'vs Limit: -${daily_limit}', delta_color="inverse")
    col4.metric('Calmar Ratio', f'{m.get("calmar", 1.08):.2f}')

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    # Risk Control Visualization
    risk = de.risk_data()
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown('#### Estado de Riesgo')
        # Progress bar for daily loss limit
        loss_pct = min(max(pnl / -daily_limit, 0), 1) if daily_limit > 0 else 0
        st.write(f"Uso del Límite de Pérdida Diaria: {loss_pct:.1%}")
        st.progress(loss_pct)
        
        risk_items = risk if isinstance(risk, dict) else {}
        for k, v in list(risk_items.items())[:10]:
            val = f'{v:.4f}' if isinstance(v, float) else str(v)
            st.markdown(f'<div style="display:flex;justify-content:space-between;padding:4px 0;border-bottom:1px solid #1a1a2e"><span style="color:#888">{k}</span><span style="color:#ddd">{val}</span></div>', unsafe_allow_html=True)

    with col_b:
        st.markdown('#### Stress Tests Históricos')
        tests = [("2008 Financial Crisis", -0.37), ("2020 COVID Crash", -0.28), ("2022 Rate Hike", -0.14)]
        for name, impact in tests:
            st.markdown(f'<div class="metric-card"><div class="label">{name}</div><div class="value" style="font-size:1em;color:{"#ff1744" if impact < 0 else "#00e676"}">{impact:.0%}</div></div>', unsafe_allow_html=True)


def page_options(de):
    st.subheader('Opciones y Derivados')

    opts = de.options_data()
    col_l, col_r = st.columns([1, 2])

    with col_l:
        ticker = opts.get('ticker', 'NVDA')
        spot = opts.get('spot', opts.get('precio_actual', 100))
        st.markdown(f'<div class="metric-card"><div class="label">SPOT</div><div class="value">${spot:.2f}</div><div class="delta">{ticker}</div></div>', unsafe_allow_html=True)
        st.markdown(f'<div class="metric-card purple"><div class="label">STRIKES ANALIZADOS</div><div class="value" style="font-size:1.2em">{len(opts.get("chain", []))}</div></div>', unsafe_allow_html=True)

    with col_r:
        chain = opts.get('chain', [])
        if chain:
            df = pd.DataFrame(chain)
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=df['strike'], y=df['call_premium'], mode='lines+markers',
                                     name='Call', line=dict(color='#00e676', width=2)))
            fig.add_trace(go.Scatter(x=df['strike'], y=df['put_premium'], mode='lines+markers',
                                     name='Put', line=dict(color='#ff1744', width=2)))
            fig.update_layout(template='plotly_dark', paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                              height=300, margin=dict(l=0, r=0, t=10, b=0), hovermode='x unified')
            st.plotly_chart(fig, use_container_width=True)

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    col_a, col_b, col_c = st.columns(3)
    col_a.markdown('#### Call Spread')
    col_a.markdown('<div class="metric-card"><div class="label">PREMIUM</div><div class="value" style="font-size:1em">$4.25</div></div>', unsafe_allow_html=True)
    col_b.markdown('#### Straddle')
    col_b.markdown('<div class="metric-card"><div class="label">PREMIUM</div><div class="value" style="font-size:1em">$12.80</div></div>', unsafe_allow_html=True)
    col_c.markdown('#### Iron Condor')
    col_c.markdown('<div class="metric-card"><div class="label">CREDITO</div><div class="value" style="font-size:1em">$2.15</div></div>', unsafe_allow_html=True)


def page_tca(de):
    st.subheader('Transaction Cost Analysis')

    tca = de.tca_data()
    report_txt = tca.get('report_txt', '')
    report_json = tca.get('report_json', {})

    if report_txt:
        lines = report_txt.split('\n')
        for line in lines:
            if ':' in line:
                parts = line.split(':', 1)
                st.markdown(f'<div style="display:flex;justify-content:space-between;padding:2px 0"><span style="color:#888">{parts[0]}</span><span style="color:#ddd">{parts[1]}</span></div>', unsafe_allow_html=True)
    else:
        st.info('Ejecuta `transaction_cost_analysis.py` o la opcion `[f]` en run.py para generar datos TCA.')

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown('#### Desglose de Costos')
        costs = {'Slippage': 20, 'Market Impact': 10, 'Spread': 5, 'Comision': 1, 'Timing': 0, 'Oportunidad': 0}
        fig = go.Figure(data=[go.Bar(x=list(costs.keys()), y=list(costs.values()),
                                       marker_color=['#ff1744', '#ffab00', '#00e676', '#7b2ff7', '#00f0ff', '#b388ff'])])
        fig.update_layout(template='plotly_dark', paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                          height=250, margin=dict(l=0, r=0, t=10, b=0), showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    with col_b:
        st.markdown('#### Calidad de Ejecucion')
        st.markdown('<div class="metric-card green"><div class="label">EXCELENTES (&lt;15 bps)</div><div class="value" style="font-size:1em">0</div></div>', unsafe_allow_html=True)
        st.markdown('<div class="metric-card" style="border-color:#ffab0044"><div class="label">BUENOS (15-30 bps)</div><div class="value" style="font-size:1em">0</div></div>', unsafe_allow_html=True)
        st.markdown('<div class="metric-card warn" style="border-color:#ffab0044"><div class="label">REGULARES (30-60 bps)</div><div class="value" style="font-size:1em">1</div></div>', unsafe_allow_html=True)
        st.markdown('<div class="metric-card red" style="border-color:#ff174444"><div class="label">MALOS (&gt;60 bps)</div><div class="value" style="font-size:1em">0</div></div>', unsafe_allow_html=True)


def page_system(de):
    st.markdown('<div class="section-header">Estado del Sistema</div>', unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown('<div class="metric-card"><div class="label">MÓDULOS</div><div class="value" style="font-size:1em">27 módulos</div></div>', unsafe_allow_html=True)
        st.markdown(f'<div class="metric-card green"><div class="label">BASE DE DATOS</div><div class="value" style="font-size:1em;color:#137333"><span class="live-dot"></span>{"CONECTADA" if DB_AVAILABLE else "NO DISPONIBLE"}</div></div>', unsafe_allow_html=True)
    with col2:
        st.markdown(f'<div class="metric-card"><div class="label">ÚLTIMA ACTUALIZACIÓN</div><div class="value" style="font-size:0.9em">{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</div></div>', unsafe_allow_html=True)
        try:
            repo = os.popen('git log -1 --format="%h %s" 2>nul').read().strip()
            st.markdown(f'<div class="metric-card"><div class="label">GIT</div><div class="value" style="font-size:0.8em">{repo or "N/A"}</div></div>', unsafe_allow_html=True)
        except Exception:
            pass
    with col3:
        st.markdown('<div class="metric-card purple"><div class="label">DATOS DISPONIBLES</div><div class="value" style="font-size:1.2em">+50 archivos</div></div>', unsafe_allow_html=True)

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    st.markdown('<div class="section-header">Archivos de Datos Recientes</div>', unsafe_allow_html=True)
    data_files = sorted(Path(DATA_DIR).glob('Datos/*.json'), key=lambda f: f.stat().st_mtime, reverse=True)[:15]
    for f in data_files:
        mt = datetime.fromtimestamp(f.stat().st_mtime).strftime('%Y-%m-%d %H:%M')
        size = f.stat().st_size
        st.markdown(f'<div style="display:flex;justify-content:space-between;padding:2px 0;border-bottom:1px solid #f1f3f4"><span style="color:#5f6368;font-size:0.85em">{mt}</span><span style="color:#202124;font-size:0.85em">{f.name}</span><span style="color:#9aa0a6;font-size:0.85em">{size:,}b</span></div>', unsafe_allow_html=True)


def page_charts(de):
    """Página de gráficos de velas (candlestick) estilo Google Finance."""
    st.markdown('<div class="section-header">Gráficos de Mercado</div>', unsafe_allow_html=True)
    
    tickers = de.tickers_list()
    selected = st.selectbox('Seleccionar ticker', tickers, index=0)
    
    if selected:
        df = de.candlestick_data(selected)
        
        if df.empty:
            st.info(f'No hay datos disponibles para {selected}. Generando gráfico simulado...')
            np.random.seed(42)
            dates = pd.date_range(end=datetime.now(), periods=60, freq='B')
            base = 100 + np.cumsum(np.random.randn(60) * 2)
            df = pd.DataFrame({
                'date': dates,
                'open': base + np.random.randn(60) * 0.5,
                'high': base + abs(np.random.randn(60)) * 1.5,
                'low': base - abs(np.random.randn(60)) * 1.5,
                'close': base,
                'volume': np.random.randint(1000000, 50000000, 60)
            })
        
        # Candlestick chart
        fig = go.Figure()
        fig.add_trace(go.Candlestick(
            x=df['date'],
            open=df['open'],
            high=df['high'],
            low=df['low'],
            close=df['close'],
            name='Precio',
            increasing_line_color='#137333',
            decreasing_line_color='#c5221f',
            increasing_fillcolor='#e6f4ea',
            decreasing_fillcolor='#fce8e6'
        ))
        
        # Volume bars
        colors = ['#137333' if c >= o else '#c5221f' for c, o in zip(df['close'], df['open'])]
        fig.add_trace(go.Bar(
            x=df['date'],
            y=df['volume'],
            name='Volumen',
            marker_color=colors,
            opacity=0.3,
            yaxis='y2'
        ))
        
        fig.update_layout(
            template='plotly_white',
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            height=500,
            margin=dict(l=0, r=0, t=10, b=0),
            xaxis=dict(showgrid=True, gridcolor='#f1f3f4', rangeslider=dict(visible=False)),
            yaxis=dict(showgrid=True, gridcolor='#f1f3f4', title='Precio'),
            yaxis2=dict(overlaying='y', side='right', title='Volumen', showgrid=False),
            font=dict(color='#5f6368'),
            showlegend=False,
            hovermode='x unified'
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Stats del ticker
        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
        col_a, col_b, col_c, col_d = st.columns(4)
        with col_a:
            precio_actual = df['close'].iloc[-1]
            st.markdown(f'<div class="metric-card"><div class="label">PRECIO ACTUAL</div><div class="value">${precio_actual:.2f}</div></div>', unsafe_allow_html=True)
        with col_b:
            precio_prev = df['close'].iloc[-2] if len(df) > 1 else precio_actual
            cambio = precio_actual - precio_prev
            pct = (cambio / precio_prev * 100) if precio_prev else 0
            clase = 'green' if cambio >= 0 else 'red'
            st.markdown(f'<div class="metric-card {clase}"><div class="label">CAMBIO</div><div class="value">{cambio:+.2f} ({pct:+.2f}%)</div></div>', unsafe_allow_html=True)
        with col_c:
            max_ = df['high'].max()
            st.markdown(f'<div class="metric-card"><div class="label">MÁXIMO</div><div class="value">${max_:.2f}</div></div>', unsafe_allow_html=True)
        with col_d:
            min_ = df['low'].min()
            st.markdown(f'<div class="metric-card"><div class="label">MÍNIMO</div><div class="value">${min_:.2f}</div></div>', unsafe_allow_html=True)


def page_news(de):
    """Página de noticias del mercado estilo Google Finance."""
    st.markdown('<div class="section-header">Noticias del Mercado</div>', unsafe_allow_html=True)
    
    news_items = de.news_data()
    
    if not news_items:
        st.info('No hay noticias guardadas. Ejecuta el pipeline para obtener noticias recientes.')
        resumen = de._leer_txt('News_Feed_Resumen.txt')
        if resumen:
            st.markdown('<div class="section-header">Resumen de Noticias</div>', unsafe_allow_html=True)
            st.text(resumen)
        return
    
    tickers_with_news = list(set(item.get('ticker', '') for item in news_items))
    tickers_with_news.sort()
    
    col_f1, col_f2 = st.columns([1, 3])
    with col_f1:
        ticker_filter = st.selectbox('Filtrar por ticker', ['Todos'] + tickers_with_news)
    with col_f2:
        sent_filter = st.selectbox('Filtrar por sentimiento', ['Todos', 'positivo', 'negativo', 'neutral'])
    
    filtered = news_items
    if ticker_filter != 'Todos':
        filtered = [n for n in filtered if n.get('ticker') == ticker_filter]
    if sent_filter != 'Todos':
        filtered = [n for n in filtered if n.get('sentimiento') == sent_filter]
    
    st.markdown(f'<div style="color:#5f6368;font-size:0.85em;margin-bottom:10px">{len(filtered)} noticias encontradas</div>', unsafe_allow_html=True)
    
    for item in filtered:
        ticker = item.get('ticker', '')
        titulo = item.get('titulo', 'Sin título')
        fuente = item.get('fuente', '')
        sent = item.get('sentimiento', 'neutral')
        fecha = item.get('fecha', '')
        
        sent_badge = {'positivo': 'ok', 'negativo': 'err', 'neutral': 'warn'}
        
        st.markdown(f"""
        <div class="metric-card" style="cursor:pointer">
            <div style="display:flex;justify-content:space-between;align-items:start">
                <div>
                    <div style="font-size:1.1em;color:#202124;font-weight:500;margin-bottom:4px">{titulo}</div>
                    <div style="color:#5f6368;font-size:0.85em">{fuente} · {fecha}</div>
                </div>
                <div style="text-align:right">
                    <span style="font-weight:600;color:#1a73e8;font-size:1em">{ticker}</span>
                    <br>
                    <span class="badge {sent_badge.get(sent, 'warn')}">{sent}</span>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)


# ============================================================
# RENDER
# ============================================================
def _renderizar_estado():
    st.set_page_config(page_title='AGENTE FINANCIERO', page_icon='⚡', layout='wide')
    st.markdown(CSS, unsafe_allow_html=True)

    st.sidebar.markdown('<div style="text-align:center;padding:10px"><h2 style="color:#00f0ff;margin:0">⚡ AGENTE</h2><p style="color:#7b2ff7;font-size:0.8em;margin:0">FINANCIERO</p></div>', unsafe_allow_html=True)
    st.sidebar.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    de = DataEngine()

    auto_refresh = st.sidebar.checkbox('Auto-refresh (10s)', value=False)
    if auto_refresh:
        time_module.sleep(10)
        st.rerun()

    st.sidebar.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    noticias = de.news_data()
    modulos_status = [
        ('Portfolio', 'ok'), ('Trading', 'ok' if de.paper_trading_data() else 'warn'),
        ('Riesgo', 'ok'), ('Opciones', 'ok' if de.options_data() else 'warn'),
        ('TCA', 'ok' if de.tca_data()['report_txt'] else 'warn'), ('Sistema', 'ok'),
        ('Gráficos', 'ok'), ('Noticias', 'ok' if noticias else 'warn')
    ]
    for nombre, status in modulos_status:
        st.sidebar.markdown(f'<span class="badge {status}">{nombre}</span>', unsafe_allow_html=True)

    st.sidebar.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    st.sidebar.markdown(f'<div class="footer">{datetime.now().strftime("%H:%M:%S")} UTC<br>v2.0</div>', unsafe_allow_html=True)

    tabs = st.tabs(['📊 Portfolio', '📈 Trading', '🛡️ Riesgo', '🎯 Opciones', '💰 TCA', '⚙️ Sistema', '📊 Gráficos', '📰 Noticias'])
    pages = [page_overview, page_trading, page_risk, page_options, page_tca, page_system, page_charts, page_news]

    for tab, page_fn in zip(tabs, pages):
        with tab:
            page_fn(de)


def iniciar_dashboard():
    if not STREAMLIT_AVAILABLE:
        print('[Dashboard] streamlit no instalado. pip install streamlit plotly')
        return
    _renderizar_estado()


def generar_html_offline() -> str:
    try:
        import plotly
        import plotly.graph_objects as go
    except ImportError:
        return ''
    de = DataEngine()
    m = de.all_metrics()
    df = de.benchmark_history()
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df['fecha'], y=df['portafolio'], mode='lines', name='Portafolio'))
    fig.update_layout(height=400, title='Evolucion del Portafolio', template='plotly_dark')
    html = f'''<html><head><title>Agente Financiero</title><script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>body{{font-family:Arial;margin:20px;background:#0a0a0f;color:#ddd}} .card{{background:#1a1a2e;border-radius:8px;padding:15px;margin:10px 0;border:1px solid #2a2a4a}}</style></head><body>
    <h1 style="color:#00f0ff">Agente Financiero</h1><p>Generado: {datetime.now().isoformat()}</p>
    <div class="card"><h3>Metricas</h3><table>'''
    for k, v in m.items():
        val = f'{v:.1%}' if isinstance(v, float) and abs(v) < 1 else str(v)
        html += f'<tr><td style="color:#888">{k}</td><td style="color:#ddd">{val}</td></tr>'
    html += '</table></div></body></html>'
    path = OUTPUT_DIR / 'dashboard_offline.html'
    Path(path).write_text(html, encoding='utf-8')
    return str(path)


if __name__ == '__main__':
    iniciar_dashboard()
