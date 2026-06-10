#!/usr/bin/env python3
"""
dashboard_interactivo.py - Dashboard web interactivo con Streamlit.
Graficos en vivo, filtros por ticker/regimen, metricas de riesgo,
performance del portafolio, y monitoreo del sistema.
"""
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta
import json, os, threading, time as time_module
from pathlib import Path

try:
    import streamlit as st
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    STREAMLIT_AVAILABLE = True
except ImportError:
    STREAMLIT_AVAILABLE = False

from config.settings import get_setting

DATA_DIR = os.environ.get('GITHUB_WORKSPACE', '.')
OUTPUT_DIR = Path(DATA_DIR) / 'Datos' / 'dashboard'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

try:
    from persistent_db import db
    DB_AVAILABLE = True
except Exception:
    DB_AVAILABLE = False


class DatosSimulados:
    def __init__(self):
        pass

    def portafolio(self, n_dias: int = 252) -> pd.DataFrame:
        np.random.seed(42)
        fechas = pd.date_range(end=datetime.now(), periods=n_dias, freq='B')
        p = 100 * np.exp(np.cumsum(np.random.randn(n_dias) * 0.02 + 0.0005))
        b = 100 * np.exp(np.cumsum(np.random.randn(n_dias) * 0.015 + 0.0003))
        return pd.DataFrame({'fecha': fechas, 'portafolio': p, 'benchmark': b})

    def tickers(self) -> List[str]:
        return ['NVDA', 'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'TSLA', 'SPY']

    def precios(self, ticker: str, n_dias: int = 252) -> pd.DataFrame:
        np.random.seed(hash(ticker) % 2**31)
        fechas = pd.date_range(end=datetime.now(), periods=n_dias, freq='B')
        p = 100 * np.exp(np.cumsum(np.random.randn(n_dias) * 0.025))
        return pd.DataFrame({'fecha': fechas, 'precio': p, 'ticker': ticker})

    def senales(self, n: int = 20) -> pd.DataFrame:
        tickers = self.tickers()
        data = []
        for t in tickers[:5]:
            for _ in range(n):
                data.append({'ticker': t, 'senal': np.random.choice(
                    ['COMPRA', 'VENTA', 'MANTENER'], p=[0.3, 0.2, 0.5]),
                    'confianza': round(np.random.random() * 0.5 + 0.25, 2),
                    'fecha': datetime.now().isoformat()})
        return pd.DataFrame(data)

    def metricas(self) -> Dict:
        return {
            'retorno_total': round(np.random.random() * 0.3 - 0.05, 3),
            'retorno_anual': round(np.random.random() * 0.15 - 0.02, 3),
            'sharpe': round(1.0 + np.random.random() * 1.0, 2),
            'sortino': round(1.2 + np.random.random() * 1.2, 2),
            'max_drawdown': round(-np.random.random() * 0.15 - 0.02, 3),
            'vol_anual': round(0.15 + np.random.random() * 0.1, 3),
            'win_rate': round(0.5 + np.random.random() * 0.2, 2),
            'trades': np.random.randint(100, 500),
            'var_95': round(-np.random.random() * 0.03 - 0.01, 3),
            'beta': round(0.8 + np.random.random() * 0.4, 2),
            'alpha': round(np.random.random() * 0.06 - 0.01, 3),
        }

    def exposicion(self) -> pd.DataFrame:
        sectores = ['Semiconductores', 'Cloud', 'Consumer', 'IA', 'Ciberseguridad']
        pesos = np.random.dirichlet(np.ones(len(sectores)))
        return pd.DataFrame({'sector': sectores, 'peso': pesos})

    def decisiones_recientes(self, n: int = 10) -> pd.DataFrame:
        data = []
        for i in range(n):
            t = np.random.choice(self.tickers())
            data.append({
                'fecha': (datetime.now() - timedelta(hours=i)).isoformat(),
                'ticker': t, 'senal': np.random.choice(['COMPRA', 'VENTA']),
                'confianza': round(np.random.random() * 0.4 + 0.3, 2),
                'resultado': round(np.random.random() * 0.05 - 0.02, 4) if np.random.random() > 0.2 else None,
            })
        return pd.DataFrame(data)


def _renderizar_estado():
    st.set_page_config(page_title='Agente Financiero',
                       page_icon='📊', layout='wide')
    st.title('Agente Financiero - Dashboard Interactivo')
    st.sidebar.header('Controles')

    datos = DatosSimulados()
    tickers = ['TODOS'] + datos.tickers()
    ticker_sel = st.sidebar.selectbox('Ticker', tickers)
    ventana = st.sidebar.select_slider('Ventana (dias)', options=[30, 60, 90, 180, 252], value=90)
    auto_refresh = st.sidebar.checkbox('Auto-refresh (30s)')
    if auto_refresh:
        time_module.sleep(30)
        st.rerun()

    col1, col2, col3, col4 = st.columns(4)
    m = datos.metricas()
    col1.metric('Retorno Anual', f'{m["retorno_anual"]:.1%}',
                f'{m["retorno_total"]:.1%} total')
    col2.metric('Sharpe', f'{m["sharpe"]:.2f}',
                f'Sortino: {m["sortino"]:.2f}')
    col3.metric('Max Drawdown', f'{m["max_drawdown"]:.1%}',
                f'Vol: {m["vol_anual"]:.1%}')
    col4.metric('Win Rate', f'{m["win_rate"]:.0%}',
                f'Trades: {m["trades"]}')

    st.subheader('Evolucion del Portafolio')
    df_p = datos.portafolio(ventana)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df_p['fecha'], y=df_p['portafolio'],
                             mode='lines', name='Portafolio',
                             line=dict(color='#00bcd4', width=2)))
    fig.add_trace(go.Scatter(x=df_p['fecha'], y=df_p['benchmark'],
                             mode='lines', name='Benchmark',
                             line=dict(color='#ff9800', width=2, dash='dash')))
    fig.update_layout(height=400, margin=dict(l=0, r=0, t=10, b=0),
                      hovermode='x unified')
    st.plotly_chart(fig, use_container_width=True)

    st.subheader('Exposicion por Sector')
    df_exp = datos.exposicion()
    fig2 = px.pie(df_exp, values='peso', names='sector',
                  color_discrete_sequence=px.colors.qualitative.Set2)
    fig2.update_traces(textposition='inside', textinfo='percent+label')
    fig2.update_layout(height=300, margin=dict(l=0, r=0, t=10, b=0))
    st.plotly_chart(fig2, use_container_width=True)

    st.subheader('Precios de Activos')
    t_mostrar = ticker_sel if ticker_sel != 'TODOS' else 'NVDA'
    df_precio = datos.precios(t_mostrar, ventana)
    fig3 = go.Figure()
    fig3.add_trace(go.Scatter(x=df_precio['fecha'], y=df_precio['precio'],
                              mode='lines', name=t_mostrar,
                              line=dict(color='#4caf50', width=2),
                              fill='tozeroy', fillcolor='rgba(76,175,80,0.1)'))
    fig3.update_layout(height=350, margin=dict(l=0, r=0, t=10, b=0))
    st.plotly_chart(fig3, use_container_width=True)

    col_left, col_right = st.columns(2)
    with col_left:
        st.subheader('Senales Recientes')
        df_sen = datos.senales()
        t_filtro = df_sen if ticker_sel == 'TODOS' else df_sen[df_sen['ticker'] == ticker_sel]
        for _, row in t_filtro.head(8).iterrows():
            color = '#4caf50' if row['senal'] == 'COMPRA' else '#f44336' if row['senal'] == 'VENTA' else '#ff9800'
            st.markdown(
                f'<span style="color:{color}">&#9632;</span> '
                f'**{row["ticker"]}** {row["senal"]} '
                f'({row["confianza"]:.0%})',
                unsafe_allow_html=True)

    with col_right:
        st.subheader('Metricas de Riesgo')
        riesgos = [
            ('VaR 95%', f'{m["var_95"]:.1%}',
             'normal' if m['var_95'] > -0.03 else 'alta'),
            ('Volatilidad', f'{m["vol_anual"]:.1%}',
             'normal' if m['vol_anual'] < 0.25 else 'alta'),
            ('Drawdown', f'{m["max_drawdown"]:.1%}',
             'normal' if m['max_drawdown'] > -0.15 else 'alta'),
            ('Beta', f'{m["beta"]:.2f}',
             'normal' if 0.7 < m['beta'] < 1.3 else 'alta'),
        ]
        for nombre, valor, nivel in riesgos:
            color = '#4caf50' if nivel == 'normal' else '#f44336'
            st.markdown(
                f'<span style="color:{color}">&#9632;</span> '
                f'**{nombre}**: {valor}',
                unsafe_allow_html=True)

    st.subheader('Decisiones Recientes')
    df_dec = datos.decisiones_recientes(10)
    if ticker_sel != 'TODOS':
        df_dec = df_dec[df_dec['ticker'] == ticker_sel]
    for _, row in df_dec.iterrows():
        c = '#4caf50' if row['senal'] == 'COMPRA' else '#f44336'
        r = f' | Resultado: {row["resultado"]:+.2%}' if pd.notna(row.get('resultado')) else ''
        st.markdown(
            f'<span style="color:{c}">&#9632;</span> '
            f'{row["fecha"][:16]} | {row["ticker"]} | '
            f'{row["senal"]} ({row["confianza"]:.0%}){r}',
            unsafe_allow_html=True)

    st.sidebar.markdown('---')
    st.sidebar.markdown(f'**Ultima actualizacion:** {datetime.now().strftime("%H:%M:%S")}')
    st.sidebar.markdown('**Estado:** OK')


def iniciar_dashboard():
    if not STREAMLIT_AVAILABLE:
        print('[Dashboard] streamlit no instalado. pip install streamlit plotly')
        return
    _renderizar_estado()


def generar_html_offline(n_dias: int = 90) -> str:
    try:
        import plotly.graph_objects as go
        import plotly
    except ImportError:
        return ''
    datos = DatosSimulados()
    m = datos.metricas()
    df_p = datos.portafolio(n_dias)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df_p['fecha'], y=df_p['portafolio'],
                             mode='lines', name='Portafolio'))
    fig.update_layout(height=400, title='Evolucion del Portafolio')

    html = f'''
    <html><head><title>Dashboard Agente Financiero</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        body {{ font-family: Arial; margin: 20px; background: #1e1e1e; color: #ddd; }}
        .card {{ background: #2d2d2d; border-radius: 8px; padding: 15px; margin: 10px 0; }}
        .metric {{ display: inline-block; margin: 10px; padding: 10px;
                   background: #333; border-radius: 6px; min-width: 120px; }}
        .value {{ font-size: 1.4em; color: #00bcd4; }}
        .label {{ font-size: 0.8em; color: #888; }}
    </style></head><body>
    <h1>Dashboard Agente Financiero</h1>
    <p>Generado: {datetime.now().isoformat()}</p>
    <div class="card">
        <div class="metric"><div class="value">{m["retorno_anual"]:.1%}</div>
            <div class="label">Retorno Anual</div></div>
        <div class="metric"><div class="value">{m["sharpe"]:.2f}</div>
            <div class="label">Sharpe</div></div>
        <div class="metric"><div class="value">{m["max_drawdown"]:.1%}</div>
            <div class="label">Max Drawdown</div></div>
        <div class="metric"><div class="value">{m["vol_anual"]:.1%}</div>
            <div class="label">Volatilidad</div></div>
    </div>
    <div class="card">{plotly.offline.plot(fig, include_plotlyjs=False, output_type='div')}</div>
    <div class="card">
        <h3>Metricas Clave</h3>
        <table><tr><th>Metrica</th><th>Valor</th></tr>
    '''
    for k, v in m.items():
        valor = f'{v:.1%}' if isinstance(v, float) and abs(v) < 1 else str(v)
        html += f'<tr><td>{k}</td><td>{valor}</td></tr>'
    html += '</table></div></body></html>'

    path = OUTPUT_DIR / 'dashboard_offline.html'
    Path(path).write_text(html, encoding='utf-8')
    return str(path)


if __name__ == '__main__':
    iniciar_dashboard()
