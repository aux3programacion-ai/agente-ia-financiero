#!/usr/bin/env python3
import json, os, sys, yfinance as yf, numpy as np, pandas as pd, time
from portafolio_utils import cargar_portafolio

DATA_DIR = os.path.dirname(os.path.abspath(__file__))
DATOS_DIR = os.path.join(DATA_DIR, 'Datos')

TICKERS_CORE = ['NVDA','MU','DELL','AVGO','DDOG','SMCI','SNOW','CRWD','NOW','TSM','ARM','OKTA','HPE','NTAP','CLS','AAPL','AMZN','GOOGL','META','MSFT','LLY','AMAT','LRCX','PANW','ORCL','HON','UBER','GE','COST','NEE']

def merge_tickers(portfolio):
    combined = list(TICKERS_CORE)
    for t in portfolio:
        if t not in combined:
            combined.append(t)
    return combined

def calc_ema(values, period):
    alpha = 2 / (period + 1)
    ema = [values[0]]
    for v in values[1:]:
        ema.append(v * alpha + ema[-1] * (1 - alpha))
    return np.array(ema)

def calc_rsi(values, period=14):
    if len(values) < period + 1:
        return np.array([50.0] * len(values))
    deltas = np.diff(values)
    gains = np.where(deltas > 0, deltas, 0)
    losses = np.where(deltas < 0, -deltas, 0)
    avg_gain = np.mean(gains[:period])
    avg_loss = np.mean(losses[:period])
    rsi = [50.0] * period
    for i in range(period, len(deltas)):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period
        if avg_loss == 0:
            rsi.append(100.0)
        else:
            rs = avg_gain / avg_loss
            rsi.append(100 - 100 / (1 + rs))
    rsi = [50.0] + rsi
    return np.array(rsi[:len(values)])

def calc_macd_line(values):
    if len(values) < 26:
        return np.array([0.0] * len(values))
    ema12 = calc_ema(values, 12)
    ema26 = calc_ema(values, 26)
    macd = ema12 - ema26
    signal = calc_ema(macd, 9)
    return macd - signal

def detect_candlestick_patterns(df, last_n=5):
    patterns = set()
    open_p = df['Open'].values.flatten()
    high_p = df['High'].values.flatten()
    low_p = df['Low'].values.flatten()
    close_p = df['Close'].values.flatten()
    for i in range(max(1, len(df) - last_n), len(df)):
        o, h, l, c = open_p[i], high_p[i], low_p[i], close_p[i]
        body = abs(c - o)
        lower_wick = min(o, c) - l
        upper_wick = h - max(o, c)
        total_range = h - l
        if total_range == 0:
            continue
        if body / (abs(c) if c != 0 else 1) < 0.001:
            patterns.add('doji')
        if i > 0:
            prev_o, prev_c = open_p[i-1], close_p[i-1]
            curr_body = body
            prev_body = abs(prev_c - prev_o)
            if c > o:
                if prev_c < prev_o and curr_body > prev_body and o < prev_c and c > prev_o:
                    patterns.add('engulfing_bullish')
            else:
                if prev_c > prev_o and curr_body > prev_body and o > prev_c and c < prev_o:
                    patterns.add('engulfing_bearish')
        if body < total_range * 0.3:
            if lower_wick > body * 2 and upper_wick < body * 0.3:
                patterns.add('hammer')
            if upper_wick > body * 2 and lower_wick < body * 0.3:
                patterns.add('estrella_fugaz')
                patterns.add('martillo_invertido')
    return list(patterns)

def find_support_resistance(df):
    high = df['High'].values.flatten()
    low = df['Low'].values.flatten()
    close = df['Close'].values.flatten()
    current_price = close[-1]
    pivot_highs = []
    pivot_lows = []
    for i in range(1, len(df) - 1):
        if high[i] > high[i-1] and high[i] > high[i+1]:
            pivot_highs.append(high[i])
        if low[i] < low[i-1] and low[i] < low[i+1]:
            pivot_lows.append(low[i])
    support = None
    resistance = None
    for pl in sorted(pivot_lows, reverse=True):
        if pl < current_price:
            support = pl
            break
    for ph in sorted(pivot_highs):
        if ph > current_price:
            resistance = ph
            break
    if support is None:
        support = round(current_price * 0.95, 2)
    if resistance is None:
        resistance = round(current_price * 1.05, 2)
    return round(support, 2), round(resistance, 2)

def detect_divergence(price, indicator, window=20):
    if len(price) < window or len(indicator) < window:
        return 'none'
    p_slice = price[-window:]
    i_slice = indicator[-window:]
    p_highs, p_lows = [], []
    i_highs, i_lows = [], []
    for j in range(1, len(p_slice) - 1):
        if p_slice[j] > p_slice[j-1] and p_slice[j] > p_slice[j+1]:
            p_highs.append((j, p_slice[j]))
            i_highs.append((j, i_slice[j]))
        if p_slice[j] < p_slice[j-1] and p_slice[j] < p_slice[j+1]:
            p_lows.append((j, p_slice[j]))
            i_lows.append((j, i_slice[j]))
    if len(p_highs) >= 2:
        if p_highs[-1][1] > p_highs[-2][1] and i_highs[-1][1] < i_highs[-2][1]:
            return 'bearish'
    if len(p_lows) >= 2:
        if p_lows[-1][1] < p_lows[-2][1] and i_lows[-1][1] > i_lows[-2][1]:
            return 'bullish'
    return 'none'

def analyze_ticker(ticker):
    print(f'[Patrones] Procesando {ticker}...')
    try:
        df = yf.download(ticker, period='3mo', interval='1d', progress=False)
        if df is None or df.empty or len(df) < 30:
            print(f'[!] {ticker}: datos insuficientes')
            return None
        close = df['Close'].values.flatten()
        velas = detect_candlestick_patterns(df)
        soporte, resistencia = find_support_resistance(df)
        macd_line = calc_macd_line(close)
        rsi_values = calc_rsi(close, 14)
        div_macd = detect_divergence(close, macd_line, 20)
        div_rsi = detect_divergence(close, rsi_values, 20)
        ultimo = round(float(close[-1]), 2)
        bull_pat = any(v in velas for v in ['hammer', 'engulfing_bullish', 'martillo_invertido'])
        bear_pat = any(v in velas for v in ['engulfing_bearish', 'estrella_fugaz'])
        bull_div = div_macd == 'bullish' or div_rsi == 'bullish'
        bear_div = div_macd == 'bearish' or div_rsi == 'bearish'
        if bull_pat and bull_div:
            senal = 'alcista'
        elif bear_pat and bear_div:
            senal = 'bajista'
        else:
            senal = 'neutral'
        return {
            'velas': velas,
            'soporte': soporte,
            'resistencia': resistencia,
            'divergencia_macd': div_macd,
            'divergencia_rsi': div_rsi,
            'ultimo_precio': ultimo,
            'senal': senal
        }
    except Exception as e:
        print(f'[!] {ticker}: error - {str(e)[:80]}')
        return None

def main():
    portfolio = load_portfolio()
    tickers = merge_tickers(portfolio)
    print(f'[Patrones] Tickers a analizar: {len(tickers)}')
    resultados = {'timestamp': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), 'tickers': {}}
    for t in tickers:
        res = analyze_ticker(t)
        if res:
            resultados['tickers'][t] = res
        else:
            resultados['tickers'][t] = {'velas': [], 'soporte': None, 'resistencia': None, 'divergencia_macd': 'none', 'divergencia_rsi': 'none', 'ultimo_precio': 0, 'senal': 'neutral'}
    os.makedirs(DATOS_DIR, exist_ok=True)
    with open(os.path.join(DATOS_DIR, 'analisis_patrones.json'), 'w') as f:
        json.dump(resultados, f, indent=2, ensure_ascii=False)
    print(f'[OK] analisis_patrones.json guardado con {len(resultados["tickers"])} tickers')

if __name__ == '__main__':
    main()
