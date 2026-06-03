import json, sys, os, time
import yfinance as yf
import pandas as pd
import numpy as np

UNIV_FILE = "Datos/univ_global.json"
SCREENING_FILE = "Datos/screening_global.json"

def calc_rsi(series, period=14):
    delta = series.diff()
    gain = delta.where(delta > 0, 0).rolling(period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))

def calc_macd(series, fast=12, slow=26, signal=9):
    ema_fast = series.ewm(span=fast).mean()
    ema_slow = series.ewm(span=slow).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal).mean()
    return macd_line - signal_line

def screen_ticker(ticker):
    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(period="6mo")
        if hist.empty or len(hist) < 30:
            return None
        close = hist['Close']
        volume = hist['Volume']
        last_price = float(close.iloc[-1])
        last_vol = int(volume.iloc[-1]) if not volume.empty else 0
        ma50 = float(close.rolling(50).mean().iloc[-1]) if len(close) >= 50 else last_price
        ma200 = float(close.rolling(200).mean().iloc[-1]) if len(close) >= 200 else last_price
        rsi_val = float(calc_rsi(close).iloc[-1]) if len(close) >= 15 else 50
        macd_val = float(calc_macd(close).iloc[-1]) if len(close) >= 27 else 0
        vol_avg = int(volume.rolling(20).mean().iloc[-1]) if len(volume) >= 20 else last_vol
        vol_ratio = round(last_vol / vol_avg, 2) if vol_avg > 0 else 1
        pct_30d = float((close.iloc[-1] / close.iloc[-21] - 1) * 100) if len(close) >= 21 else 0
        score = 0
        if rsi_val and rsi_val < 70: score += 10
        if rsi_val and rsi_val > 30: score += 10
        if ma50 and ma200 and ma50 > ma200: score += 25
        if macd_val and macd_val > 0: score += 15
        if vol_ratio and vol_ratio > 1.2: score += 10
        if pct_30d and pct_30d > -5: score += 10
        if pct_30d and pct_30d > 2: score += 10
        if last_price > 5: score += 10
        return {
            "ticker": ticker,
            "precio": round(last_price, 2),
            "rsi": round(rsi_val, 1) if rsi_val else 50,
            "macd": round(macd_val, 2) if macd_val else 0,
            "ma50": round(ma50, 2),
            "ma200": round(ma200, 2),
            "vol_ratio": vol_ratio,
            "pct_30d": round(pct_30d, 1),
            "score": score
        }
    except:
        return None

def main():
    if not os.path.exists(UNIV_FILE):
        print("[!] No existe univ_global.json - ejecuta explorar_mercados.py primero")
        sys.exit(1)
    with open(UNIV_FILE) as f:
        univ = json.load(f)
    todos = univ.get("todos", [])

    # Build ticker-to-market mapping from market categories
    market_labels = {
        "sp500": "US",
        "ipc_mexico": "MEXICO",
        "dax": "EUROPA",
        "nikkei": "ASIA",
        "ftse": "EUROPA",
        "hangseng": "ASIA",
        "core_30": "US"
    }
    ticker_mercado = {}
    for mercado, label in market_labels.items():
        for t in univ.get(mercado, []):
            ticker_mercado[t.upper()] = label

    print(f"[SCREENING] Evaluando {len(todos)} tickers...")
    resultados = []
    errores = 0
    por_mercado = {}
    for i, t in enumerate(todos):
        if (i+1) % 50 == 0:
            print(f"  Progreso: {i+1}/{len(todos)} - OK: {len(resultados)} Err: {errores}")
        r = screen_ticker(t)
        if r:
            r["mercado"] = ticker_mercado.get(t.upper(), "GLOBAL")
            resultados.append(r)
            m = r["mercado"]
            if m not in por_mercado: por_mercado[m] = []
            por_mercado[m].append(r["ticker"])
        else:
            errores += 1
        time.sleep(0.05)
    resultados.sort(key=lambda x: x['score'], reverse=True)
    top200 = resultados[:200]

    output = {
        "timestamp": time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
        "total_analizados": len(resultados),
        "total_mercados": len(por_mercado),
        "top50": top200[:50],
        "top200": top200,
        "top50_tickers": [r['ticker'] for r in top200[:50]],
        "todos": resultados,
        "por_mercado": por_mercado,
        "total_por_mercado": {m: len(por_mercado[m]) for m in por_mercado}
    }
    with open(SCREENING_FILE, "w") as f:
        json.dump(output, f, indent=2)
    mercados_summary = ', '.join(f"{m}: {len(por_mercado[m])}" for m in sorted(por_mercado))
    print(f"[OK] Screening: {len(resultados)} con datos | {mercados_summary}")

if __name__ == "__main__":
    main()
