#!/usr/bin/env python3
"""mercado_tiempo_real.py - Dashboard de mercado en vivo. Precios, noticias, top picks 30d."""
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
import json, os, time as time_module
from pathlib import Path

try:
    import streamlit as st
    import plotly.graph_objects as go
    import plotly.express as px
    STREAMLIT_AVAILABLE = True
except ImportError:
    STREAMLIT_AVAILABLE = False

try:
    import yfinance as yf
    YF_AVAILABLE = True
except ImportError:
    YF_AVAILABLE = False

DATA_DIR = os.environ.get('GITHUB_WORKSPACE', '.')
DATOS = Path(DATA_DIR) / 'Datos'

CSS = """
<style>
    .stApp { background: #050510; }
    h1, h2, h3 { color: #00f0ff; font-weight: 300; }
    .metric-card { background: linear-gradient(135deg, #0d0d1a, #1a1a30); border: 1px solid #2a2a4a; border-radius: 12px; padding: 14px; margin: 4px 0; }
    .metric-card .label { color: #667; font-size: 0.7em; text-transform: uppercase; letter-spacing: 1px; }
    .metric-card .value { font-size: 1.5em; font-weight: 600; }
    .green { color: #00e676; } .red { color: #ff1744; } .cyan { color: #00f0ff; } .purple { color: #b388ff; } .yellow { color: #ffab00; }
    .ticker-row { display: flex; justify-content: space-between; padding: 6px 0; border-bottom: 1px solid #1a1a2e; }
    .news-item { background: #0d0d1a; border-left: 3px solid #00f0ff; border-radius: 0 8px 8px 0; padding: 10px 14px; margin: 6px 0; }
    .news-item .title { color: #ddd; font-size: 0.9em; }
    .news-item .meta { color: #667; font-size: 0.75em; }
    .footer { text-align: center; color: #445; font-size: 0.7em; padding: 20px; }
</style>
"""

class MercadoData:
    def _read_json(self, path):
        p = DATOS / path
        if p.exists():
            try: return json.loads(p.read_text(encoding='utf-8'))
            except: return None
        return None

    def get_probabilidades(self):
        ia = self._read_json('analisis_ia.json') or {}
        return ia.get('probabilidades', ia)

    def get_screening(self):
        s = self._read_json('screening_global.json') or {}
        return s.get('top50', [])

    def get_noticias(self):
        return self._read_json('noticias_recientes.json') or {}

    def get_xgboost(self):
        xgb = self._read_json('modelo_xgboost.json') or {}
        return xgb.get('tickers', xgb)

    def get_indices_vivos(self):
        defaults = {'SPY': 545.0, 'QQQ': 475.0, 'DIA': 390.0, '^VIX': 15.2, 'DX-Y.NYB': 104.5, 'TLT': 92.0}
        if not YF_AVAILABLE:
            return defaults
        result = {}
        for ticker in ['SPY', 'QQQ', 'DIA', '^VIX', 'DX-Y.NYB', 'TLT']:
            try:
                t = yf.Ticker(ticker)
                hist = t.history(period='1d', interval='1m', progress=False)
                if not hist.empty:
                    result[ticker] = float(hist['Close'].iloc[-1])
                else:
                    hist = t.history(period='5d', progress=False)
                    result[ticker] = float(hist['Close'].iloc[-1]) if not hist.empty else defaults[ticker]
            except:
                result[ticker] = defaults[ticker]
        return result

    def get_top_picks(self, n=10):
        probs = self.get_probabilidades()
        if not probs: return []
        picks = []
        for ticker, data in probs.items():
            if not isinstance(data, dict) or not ticker.isupper() or len(ticker) > 5: continue
            picks.append({'ticker': ticker, 'probabilidad': data.get('probabilidad', 0),
                          'confianza': data.get('confianza', 0),
                          'precio_objetivo': data.get('precio_objetivo_30d', 0),
                          'analisis': data.get('analisis', '')[:120]})
        picks.sort(key=lambda x: x['probabilidad'], reverse=True)
        return picks[:n]

    def get_noticias_por_ticker(self):
        n = self._read_json('noticias_recientes.json') or {}
        pt = n.get('por_ticker', {})
        result = {}
        for t, data in pt.items():
            if isinstance(data, dict) and 'sentimiento' in data and isinstance(data['sentimiento'], dict):
                result[t] = {
                    'sentimiento': data['sentimiento'].get('sentimiento', 'neutral'),
                    'score': data['sentimiento'].get('score', 0),
                    'impacto': data['sentimiento'].get('impacto', 'bajo'),
                }
        return result

    def get_calibracion(self):
        return self._read_json('calibracion.json') or {}

    def get_precio_vivo(self, ticker, fallback=100.0):
        if not YF_AVAILABLE:
            return fallback
        try:
            t = yf.Ticker(ticker)
            hist = t.history(period='1d', interval='1m', progress=False)
            if not hist.empty:
                return float(hist['Close'].iloc[-1])
            hist = t.history(period='5d', progress=False)
            if not hist.empty:
                return float(hist['Close'].iloc[-1])
        except:
            pass
        return fallback

    def get_spy_direction(self):
        indices = self.get_indices_vivos()
        spy = indices.get('SPY', 545.0)
        vix = indices.get('^VIX', 15.2)
        trend = 'alcista' if spy >= 540 else 'bajista' if spy <= 520 else 'neutral'
        riesgo = 'alto' if vix > 25 else 'bajo' if vix < 15 else 'moderado'
        return trend, riesgo, spy, vix

    def get_acciones_a_comprar(self, n=20):
        probs = self.get_probabilidades()
        if not probs:
            return []
        notis = self.get_noticias_por_ticker()
        trend, riesgo, spy, vix = self.get_spy_direction()
        cal = self.get_calibracion()
        precision_global = cal.get('precision_global', 0.5)
        sector_stats = cal.get('sectores', {})

        # Mapa de sector por ticker
        TICKERS_SECTOR = {t: s for s, tkrs in {
            'Semiconductores': ['NVDA','MU','AVGO','TSM','ARM'],
            'Servidores IA': ['DELL','SMCI','HPE'],
            'Software IA': ['DDOG','SNOW','NOW'],
            'Ciberseguridad': ['CRWD','PANW','OKTA'],
            'Almacenamiento': ['NTAP','CLS'],
            'Consumer Tech': ['AAPL','AMZN','GOOGL','META','MSFT'],
            'Farmaceutico': ['LLY'],
            'Semicon Equip': ['AMAT','LRCX'],
            'Cloud/Database': ['ORCL'],
            'Industrial': ['HON','GE'],
            'Movilidad/Tech': ['UBER'],
            'Consumo Defensivo': ['COST'],
            'Utilities/Energy': ['NEE']
        }.items() for t in tkrs}

        # Precios de precios_reales.json como fallback rapido
        precios_fallback = {}
        precios_json = self._read_json('precios_reales.json') or {}
        for t, p in precios_json.get('precios', {}).items():
            if isinstance(p, dict):
                precios_fallback[t] = p.get('price', 100)

        # Calcular mercado alcista = SPY > 540
        mercado_alcista = trend == 'alcista'

        acciones = []
        for ticker, data in probs.items():
            if not isinstance(data, dict) or not ticker.isupper() or len(ticker) > 5:
                continue
            prob = data.get('probabilidad', 0)
            conf = data.get('confianza', 0)
            target = data.get('precio_objetivo_30d', 0)
            if not target or target <= 0:
                continue

            # Precio actual: yfinance vivo con fallback
            precio_actual = self.get_precio_vivo(ticker, precios_fallback.get(ticker, 100))
            if precio_actual <= 0:
                precio_actual = 100

            # Retorno potencial
            retorno_pct = ((target - precio_actual) / precio_actual) * 100

            # Noticias
            ns = notis.get(ticker, {})
            sent = ns.get('sentimiento', 'neutral')
            score_noti = ns.get('score', 0)

            # Precision sector
            sector = TICKERS_SECTOR.get(ticker, 'Unknown')
            prec_sector = sector_stats.get(sector, {}).get('precision', 0.5)

            # Score compuesto (0-100)
            score_base = prob * 0.4
            score_news = 0
            if sent == 'positivo':
                score_news = score_noti * 25
            elif sent == 'negativo':
                score_news = -(score_noti * 15)

            score_mercado = 10 if mercado_alcista else 0
            score_sector = (prec_sector - 0.5) * 40
            score_confianza = (conf - 50) * 0.3
            score_total = score_base + score_news + score_mercado + score_sector + score_confianza
            score_total = max(0, min(100, score_total))

            # Determinar nivel de recomendacion
            if prob >= 65 or (prob >= 55 and sent == 'positivo' and mercado_alcista):
                nivel = 'ALTA'
            elif prob >= 55 or (prob >= 50 and sent == 'positivo'):
                nivel = 'MEDIA'
            else:
                nivel = 'BAJA'

            # Recomendacion de venta
            if nivel == 'ALTA' and retorno_pct > 0:
                vender_cuando = f"Vender si alcanza ${target:.2f}"
            elif nivel == 'MEDIA' and retorno_pct > 0:
                vender_cuando = f"Vender en 15-20d o si toca ${target:.2f}"
            else:
                vender_cuando = "Esperar senal mas clara"

            acciones.append({
                'ticker': ticker,
                'precio_compra': precio_actual,
                'probabilidad': prob,
                'confianza': conf,
                'noticia_sentimiento': sent,
                'noticia_score': score_noti,
                'precio_venta': target,
                'retorno_pct': retorno_pct,
                'nivel': nivel,
                'vender_cuando': vender_cuando,
                'sector': sector,
                'score_total': score_total,
            })

        acciones.sort(key=lambda x: x['score_total'], reverse=True)
        return acciones[:n]

def mostrar_tabs():
    de = MercadoData()

    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
        "Resumen", "Top Picks 30d", "Noticias", "Screening", "XGBoost", "Sistema", "Acciones a Comprar"])

    with tab1:
        st.subheader("Indices en Vivo")
        indices = de.get_indices_vivos()
        names = {'SPY': 'S&P 500', 'QQQ': 'Nasdaq', 'DIA': 'Dow Jones',
                 '^VIX': 'VIX', 'DX-Y.NYB': 'DXY', 'TLT': 'Bonos 20y'}
        cols = st.columns(6)
        for i, (t, name) in enumerate(names.items()):
            with cols[i]:
                st.markdown(f'<div class="metric-card"><div class="label">{name}</div><div class="value cyan">{indices.get(t,0):.2f}</div></div>', unsafe_allow_html=True)

        st.markdown('<hr style="border-color:#2a2a4a;margin:20px 0">', unsafe_allow_html=True)
        col_l, col_r = st.columns([3, 2])
        with col_l:
            st.subheader("Top Picks 30 Dias")
            for p in de.get_top_picks(8):
                prob = p['probabilidad']
                c = 'green' if prob >= 60 else 'yellow' if prob >= 50 else 'red'
                st.markdown(f'<div class="ticker-row"><span><strong>{p["ticker"]}</strong> <span class="{c}">{prob:.0f}%</span> <span style="color:#667">conf:{p["confianza"]:.0f}%</span></span><span style="color:#888">{p["analisis"][:60]}</span></div>', unsafe_allow_html=True)
        with col_r:
            st.subheader("Screening Top Score")
            for s in de.get_screening()[:8]:
                st.markdown(f'<div class="ticker-row"><span><strong>{s["ticker"]}</strong> ${s.get("precio",0):.1f}</span><span style="color:#00e676">Score: {s.get("score",0)}</span></div>', unsafe_allow_html=True)

    with tab2:
        st.subheader("Ranking - Probabilidad 30 Dias")
        top = de.get_top_picks(30)
        if not top:
            st.warning("Ejecuta analisis_ia.py primero")
        else:
            rows = [{'Ticker': p['ticker'], 'Prob 30d': f'{p["probabilidad"]:.0f}%',
                     'Confianza': f'{p["confianza"]:.0f}%',
                     'Target': f'${p["precio_objetivo"]:.0f}' if p['precio_objetivo'] else '-',
                     'Analisis': p['analisis'][:80]} for p in top]
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

            st.markdown('<hr style="border-color:#2a2a4a;margin:20px 0">', unsafe_allow_html=True)
            ticks = [p['ticker'] for p in top]
            vals = [p['probabilidad'] for p in top]
            colors = ['#00e676' if v >= 60 else '#ffab00' if v >= 50 else '#ff1744' for v in vals]
            fig = go.Figure(data=[go.Bar(x=ticks, y=vals, marker_color=colors, text=[f'{v:.0f}%' for v in vals], textposition='outside')])
            fig.update_layout(template='plotly_dark', paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                              height=350, margin=dict(l=0, r=0, t=10, b=0), showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

    with tab3:
        st.subheader("Noticias")
        n = de.get_noticias()
        generales = n.get('noticias_generales', [])
        if generales and isinstance(generales, list):
            for item in generales[:20]:
                titulo = item.get('titulo', item.get('title', '?'))
                sent = item.get('sentimiento', item.get('sentiment', 'neutral'))
                fuente = item.get('fuente', item.get('source', '?'))
                col_s = '#00e676' if sent in ('positivo','positive') else '#ff1744' if sent in ('negativo','negative') else '#ffab00'
                st.markdown(f'<div class="news-item"><div class="title">{titulo}</div><div class="meta">{fuente} | <span style="color:{col_s}">{sent}</span></div></div>', unsafe_allow_html=True)
        else:
            st.info("Ejecuta noticias_mercado.py para ver noticias")

    with tab4:
        st.subheader("Screening Global")
        screen = de.get_screening()
        if screen:
            mercado = st.selectbox("Mercado", ["TODOS","US","MEXICO","EUROPA","ASIA"])
            min_s = st.slider("Score minimo", 0, 100, 50)
            rows = []
            for s in screen:
                if mercado != "TODOS" and s.get("mercado","") != mercado: continue
                if s.get("score",0) < min_s: continue
                rows.append({'Ticker': s['ticker'], 'Precio': f'${s.get("precio",0):.1f}',
                             'Score': s.get('score',0), 'RSI': s.get('rsi',0),
                             'MACD': s.get('macd',0), '30d %': f'{s.get("pct_30d",0):.1f}%'})
            if rows:
                st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
                st.caption(f"{len(rows)} tickers")
        else:
            st.info("Ejecuta cobertura_global.py primero")

    with tab5:
        st.subheader("Predicciones XGBoost")
        xgb = de.get_xgboost()
        if isinstance(xgb, dict) and xgb:
            rows = []
            for t, d in xgb.items():
                if not isinstance(d, dict) or not t.isupper() or len(t) > 5: continue
                rows.append({'Ticker': t, 'Prob 20d': f'{d.get("prob_up_20d",0):.1f}%',
                             'Pred': d.get('prediccion','?'),
                             'Acc': f'{d.get("wf_accuracy",0)*100:.1f}%'})
            if rows:
                st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        else:
            st.info("Ejecuta modelo_xgboost.py primero")

    with tab6:
        st.subheader("Estado del Sistema")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f'<div class="metric-card"><div class="label">YFINANCE</div><div class="value" style="font-size:1em;color={"#00e676" if YF_AVAILABLE else "#ff1744"}">{"DISPONIBLE" if YF_AVAILABLE else "NO INSTALADO"}</div></div>', unsafe_allow_html=True)
            st.markdown(f'<div class="metric-card"><div class="label">STREAMLIT</div><div class="value" style="font-size:1em;color={"#00e676" if STREAMLIT_AVAILABLE else "#ff1744"}">{"DISPONIBLE" if STREAMLIT_AVAILABLE else "NO INSTALADO"}</div></div>', unsafe_allow_html=True)
        with col2:
            st.markdown(f'<div class="metric-card"><div class="label">ACTUALIZACION</div><div class="value" style="font-size:0.9em">{datetime.now().strftime("%H:%M:%S")}</div></div>', unsafe_allow_html=True)
            nf = len(list(DATOS.glob("*.json"))) if DATOS.exists() else 0
            st.markdown(f'<div class="metric-card"><div class="label">DATOS</div><div class="value" style="font-size:0.9em">{nf} JSONs</div></div>', unsafe_allow_html=True)

    with tab7:
        st.subheader("Acciones a Comprar")
        trend, riesgo, spy, vix = de.get_spy_direction()
        col_trend, col_riesgo, col_spy, col_vix = st.columns(4)
        tc = '#00e676' if trend == 'alcista' else '#ff1744' if trend == 'bajista' else '#ffab00'
        col_trend.markdown(f'<div class="metric-card"><div class="label">TENDENCIA MERCADO</div><div class="value" style="color:{tc}">{trend.upper()}</div></div>', unsafe_allow_html=True)
        rc = '#00e676' if riesgo == 'bajo' else '#ffab00' if riesgo == 'moderado' else '#ff1744'
        col_riesgo.markdown(f'<div class="metric-card"><div class="label">RIESGO</div><div class="value" style="color:{rc}">{riesgo.upper()}</div></div>', unsafe_allow_html=True)
        col_spy.markdown(f'<div class="metric-card"><div class="label">S&P 500</div><div class="value cyan">{spy:.1f}</div></div>', unsafe_allow_html=True)
        col_vix.markdown(f'<div class="metric-card"><div class="label">VIX</div><div class="value" style="color:{rc}">{vix:.1f}</div></div>', unsafe_allow_html=True)

        acciones = de.get_acciones_a_comprar(20)
        acciones.sort(key=lambda x: x['retorno_pct'], reverse=True)
        if not acciones:
            st.warning("Ejecuta analisis_ia.py primero")
        else:
            altas = sum(1 for a in acciones if a['nivel'] == 'ALTA')
            medias = sum(1 for a in acciones if a['nivel'] == 'MEDIA')
            bajas = sum(1 for a in acciones if a['nivel'] == 'BAJA')
            cols = st.columns(3)
            cols[0].markdown(f'<div class="metric-card"><div class="label">COMPRA ALTA</div><div class="value green">{altas}</div></div>', unsafe_allow_html=True)
            cols[1].markdown(f'<div class="metric-card"><div class="label">COMPRA MEDIA</div><div class="value yellow">{medias}</div></div>', unsafe_allow_html=True)
            cols[2].markdown(f'<div class="metric-card"><div class="label">EVITAR</div><div class="value red">{bajas}</div></div>', unsafe_allow_html=True)

            st.markdown('<hr style="border-color:#2a2a4a;margin:16px 0">', unsafe_allow_html=True)

            rows = []
            for a in acciones:
                nv = a['nivel']
                color_nivel = '#00e676' if nv == 'ALTA' else '#ffab00' if nv == 'MEDIA' else '#ff1744'
                color_sent = '#00e676' if a['noticia_sentimiento'] == 'positivo' else '#ff1744' if a['noticia_sentimiento'] == 'negativo' else '#888'
                ret = a['retorno_pct']
                color_ret = '#00e676' if ret >= 10 else '#ffab00' if ret >= 3 else '#ff1744'
                rows.append(f'''<tr>
                    <td style="padding:6px 8px;border-bottom:1px solid #1a1a2e"><strong style="color:#ddd">{a["ticker"]}</strong></td>
                    <td style="padding:6px 8px;border-bottom:1px solid #1a1a2e;color:#00f0ff">${a["precio_compra"]:.2f}</td>
                    <td style="padding:6px 8px;border-bottom:1px solid #1a1a2e;color:#ddd">{a["probabilidad"]:.0f}%</td>
                    <td style="padding:6px 8px;border-bottom:1px solid #1a1a2e;color:#888">{a["confianza"]:.0f}%</td>
                    <td style="padding:6px 8px;border-bottom:1px solid #1a1a2e;color:{color_sent}">{a["noticia_sentimiento"].upper()}</td>
                    <td style="padding:6px 8px;border-bottom:1px solid #1a1a2e;color:#b388ff">${a["precio_venta"]:.2f}</td>
                    <td style="padding:6px 8px;border-bottom:1px solid #1a1a2e;color:{color_ret}">{f"+{ret:.1f}%" if ret > 0 else f"{ret:.1f}%"}</td>
                    <td style="padding:6px 8px;border-bottom:1px solid #1a1a2e;color:{color_nivel};font-weight:bold">{nv}</td>
                    <td style="padding:6px 8px;border-bottom:1px solid #1a1a2e;color:#667;font-size:0.85em">{a["vender_cuando"]}</td>
                </tr>''')

            header = '''<thead><tr style="color:#667;font-size:0.75em;text-transform:uppercase;letter-spacing:1px">
                <th style="padding:6px 8px">Ticker</th><th style="padding:6px 8px">Precio Compra</th>
                <th style="padding:6px 8px">Prob 30d</th><th style="padding:6px 8px">Conf</th>
                <th style="padding:6px 8px">Noticia</th><th style="padding:6px 8px">Precio Venta</th>
                <th style="padding:6px 8px">Retorno</th><th style="padding:6px 8px">Nivel</th>
                <th style="padding:6px 8px">Vender Cuando</th>
            </tr></thead>'''
            st.markdown(f'<table style="width:100%;border-collapse:collapse">{header}{"".join(rows)}</table>', unsafe_allow_html=True)

            st.caption(f"Precios en vivo via yfinance. {len(acciones)} acciones analizadas. Precio venta = objetivo 30d del sistema. Ordenado por retorno potencial descendente.")


def main():
    if not STREAMLIT_AVAILABLE:
        print("pip install streamlit plotly")
        return
    st.set_page_config(page_title="Mercado en Vivo", layout="wide")
    st.markdown(CSS, unsafe_allow_html=True)
    st.sidebar.markdown('<div style="text-align:center;padding:10px"><h2 style="color:#00f0ff;margin:0">MERCADO</h2><p style="color:#7b2ff7;font-size:0.8em;margin:0">EN VIVO</p></div>', unsafe_allow_html=True)
    st.sidebar.markdown('<hr style="border-color:#2a2a4a">', unsafe_allow_html=True)

    auto = st.sidebar.checkbox("Auto-refresh", value=False)
    if auto:
        t = st.sidebar.slider("Segundos", 5, 60, 15)
        time_module.sleep(t)
        st.rerun()

    st.sidebar.markdown(f'<div style="color:#667;font-size:0.75em;text-align:center">{datetime.now().strftime("%H:%M:%S")}</div>', unsafe_allow_html=True)
    mostrar_tabs()


if __name__ == "__main__":
    main()
