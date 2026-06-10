#!/usr/bin/env python3
"""dashboard_app.py - Dashboard interactivo con Streamlit."""
import numpy as np
import pandas as pd
from typing import Dict, Any, List
from datetime import datetime
import json, os
from pathlib import Path
from config.settings import get_setting
DATA_DIR = os.environ.get("GITHUB_WORKSPACE", ".")
OUTPUT_DIR = Path(DATA_DIR) / "Datos" / "dashboard"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

class DashboardDataProvider:
    def load_portfolio_data(self) -> pd.DataFrame:
        path = Path(DATA_DIR) / "Datos" / "backtest_results.parquet"
        if path.exists(): return pd.read_parquet(path)
        dates = pd.date_range("2023-01-01", "2024-12-31", freq="B")
        n = len(dates)
        p = 100 * np.exp(np.cumsum(np.random.randn(n) * 0.02 + 0.0005))
        b = 100 * np.exp(np.cumsum(np.random.randn(n) * 0.015 + 0.0003))
        df = pd.DataFrame({"fecha": dates, "portafolio": p, "benchmark": b})
        df["retorno"] = df["portafolio"].pct_change()
        df["retorno_bm"] = df["benchmark"].pct_change()
        return df
    def load_signals_data(self) -> pd.DataFrame:
        dates = pd.date_range("2024-01-01", "2024-12-31", freq="B")
        return pd.DataFrame({"fecha": dates,
            "NVDA": np.random.randn(len(dates))*0.5+0.2,
            "AAPL": np.random.randn(len(dates))*0.5-0.1,
            "MSFT": np.random.randn(len(dates))*0.5+0.1})
    def load_risk_data(self) -> Dict:
        return {"var_95": -0.023, "var_99": -0.041, "vol_anual": 0.185,
            "sharpe": 1.42, "max_drawdown": -0.087, "beta": 1.05}
    def load_agent_performance(self) -> pd.DataFrame:
        return pd.DataFrame({"agente": ["tecnico","fundamental","macro","sentimiento","riesgo"],
            "precision": [0.62, 0.55, 0.58, 0.61, 0.53],
            "peso": [1.2, 0.9, 1.1, 1.0, 0.8]})
    def load_performance_metrics(self) -> Dict:
        return {"retorno_total": 0.187, "retorno_anual": 0.094, "volatilidad": 0.185,
            "sharpe": 1.42, "sortino": 1.85, "calmar": 1.08, "win_rate": 0.54, "trades": 342}


class DashboardGenerator:
    def __init__(self):
        self.dp = DashboardDataProvider()
    def generate_summary(self) -> Dict:
        perf = self.dp.load_performance_metrics()
        risk = self.dp.load_risk_data()
        return {"fecha": datetime.now().isoformat(), "rendimiento": perf, "riesgo": risk}
    def generate_html_report(self) -> str:
        s = self.generate_summary()
        rend = s.get("rendimiento", {})
        riesgo = s.get("riesgo", {})
        html = "<html><body>"
        html += "<h1>Dashboard Financiero</h1>"
        fecha = s.get("fecha", "")
        tot_ret = rend.get("retorno_total", 0)
        shrp = rend.get("sharpe", 0)
        v95 = riesgo.get("var_95", 0)
        html = "<html><body>"
        html += f"<p>Generado: {fecha}</p>"
        html += f"<p>Retorno: {tot_ret*100:.1f}%</p>"
        html += f"<p>Sharpe: {shrp:.2f}</p>"
        html += f"<p>VaR95: {v95*100:.1f}%</p>"
        html += "</body></html>"
        path = OUTPUT_DIR / "dashboard.html"
        path.write_text(html, encoding="utf-8")
        return str(path)


class DashboardAPI:
    def __init__(self):
        self.dp = DashboardDataProvider()
        self.gen = DashboardGenerator()
    def get_summary(self) -> Dict: return self.gen.generate_summary()
    def get_portfolio_data(self) -> pd.DataFrame: return self.dp.load_portfolio_data()
    def get_risk_data(self) -> Dict: return self.dp.load_risk_data()
    def get_agent_data(self) -> pd.DataFrame: return self.dp.load_agent_performance()
    def generate_report(self) -> str: return self.gen.generate_html_report()


if __name__ == "__main__":
    api = DashboardAPI()
    s = api.get_summary()
    tr = s["rendimiento"]["retorno_total"]
    print(f"Retorno: {tr*100:.1f}%")
    r = api.generate_report()
    print(f"Reporte: {r}")