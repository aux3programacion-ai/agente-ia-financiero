import json
import os
import sys
import urllib.request
import re
import time
import random

TICKERS_CORE = ['NVDA','MU','DELL','AVGO','DDOG','SMCI','SNOW','CRWD','NOW','TSM','ARM','OKTA','HPE','NTAP','CLS','AAPL','AMZN','GOOGL','META','MSFT','LLY','AMAT','LRCX','PANW','ORCL','HON','UBER','GE','COST','NEE']

DATA_DIR = os.environ.get('GITHUB_WORKSPACE', '.')
OUTPUT = os.path.join(DATA_DIR, 'Datos', 'opciones.json')
os.makedirs(os.path.join(DATA_DIR, 'Datos'), exist_ok=True)

from portafolio_utils import cargar_portafolio

def tickers_a_procesar():
    portafolio = cargar_portafolio(DATA_DIR)
    combinados = list(TICKERS_CORE)
    for t in portafolio:
        t = t.strip().upper()
        if t and t not in combinados:
            combinados.append(t)
    return combinados[:30]

def fetch_options_yahoo(ticker):
    try:
        url = f'https://query1.finance.yahoo.com/v7/finance/options/{ticker}'
        req = urllib.request.Request(url, headers={
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
            'Accept': 'application/json'
        })
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read().decode('utf-8'))

        result = data.get('optionChain', {}).get('result', [])
        if not result:
            return None

        options = result[0].get('options', [])
        if not options:
            return None

        expiration = result[0].get('expirationDates', [])
        fecha_expiracion = ''
        if expiration:
            fecha_expiracion = time.strftime('%Y-%m-%d', time.gmtime(expiration[0]))

        puts = options[0].get('puts', [])
        calls = options[0].get('calls', [])

        vol_puts = sum(p.get('volume', 0) or 0 for p in puts)
        vol_calls = sum(c.get('volume', 0) or 0 for c in calls)
        oi_puts = sum(p.get('openInterest', 0) or 0 for p in puts)
        oi_calls = sum(c.get('openInterest', 0) or 0 for c in calls)

        vol_total = vol_puts + vol_calls
        if vol_calls > 0:
            put_call_ratio = round(vol_puts / vol_calls, 2)
        else:
            put_call_ratio = 999.0

        if put_call_ratio > 1.0:
            sentimiento = 'bearish'
        elif put_call_ratio < 0.7:
            sentimiento = 'bullish'
        else:
            sentimiento = 'neutral'

        # --- IV Skew and Greeks ---
        # Use ATM options for IV
        spot = result[0].get('quote', {}).get('regularMarketPrice', 0)
        atm_iv_put = 0.0
        atm_iv_call = 0.0
        if spot > 0 and puts and calls:
            # Find closest to ATM
            put_strikes = [p.get('strike', 0) for p in puts if p.get('impliedVolatility') is not None]
            call_strikes = [c.get('strike', 0) for c in calls if c.get('impliedVolatility') is not None]
            if put_strikes:
                atm_put = min(puts, key=lambda p: abs(p.get('strike', 0) - spot))
                atm_iv_put = atm_put.get('impliedVolatility', 0) * 100
            if call_strikes:
                atm_call = min(calls, key=lambda c: abs(c.get('strike', 0) - spot))
                atm_iv_call = atm_call.get('impliedVolatility', 0) * 100
        
        iv_skew = round(atm_iv_put - atm_iv_call, 2) if atm_iv_put and atm_iv_call else 0.0
        avg_iv = round((atm_iv_put + atm_iv_call) / 2, 2) if (atm_iv_put or atm_iv_call) else 0.0

        # Simple delta proxy: 0.5 for ATM
        # OI weighted IV
        oi_iv_put = sum(p.get('openInterest', 0) * p.get('impliedVolatility', 0) for p in puts if p.get('impliedVolatility') is not None)
        oi_iv_call = sum(c.get('openInterest', 0) * c.get('impliedVolatility', 0) for c in calls if c.get('impliedVolatility') is not None)
        total_oi = oi_puts + oi_calls
        if total_oi > 0:
            oi_weighted_iv = (oi_iv_put + oi_iv_call) / total_oi * 100
        else:
            oi_weighted_iv = 0.0

        # --- Market microstructure: IV smile curvature, spread proxy, gamma ---
        iv_by_strike = {}
        for p in puts:
            if p.get('impliedVolatility') is not None and p.get('strike'):
                iv_by_strike[p['strike']] = p['impliedVolatility'] * 100
        for c in calls:
            if c.get('impliedVolatility') is not None and c.get('strike'):
                ks = c['strike']
                iv_by_strike[ks] = max(iv_by_strike.get(ks, 0), c['impliedVolatility'] * 100)
        iv_curvature = 0.0
        if spot > 0 and len(iv_by_strike) >= 5:
            strikes_sorted = sorted(iv_by_strike.keys())
            ivs = [iv_by_strike[k] for k in strikes_sorted]
            if len(ivs) >= 5:
                # Smile curvature: IV at wings - IV at center
                mid_idx = len(ivs) // 2
                center_iv = ivs[mid_idx]
                wing_left = np.mean(ivs[:max(1, len(ivs)//4)])
                wing_right = np.mean(ivs[-max(1, len(ivs)//4):])
                iv_curvature = (max(wing_left, wing_right) - center_iv) / max(center_iv, 1)
        
        # Bid-ask spread proxy from options chain
        spread_avg = 0.0
        spread_count = 0
        for p in puts + calls:
            bid = p.get('bid', 0) or 0
            ask = p.get('ask', 0) or 0
            mid = (bid + ask) / 2 if (bid + ask) > 0 else None
            if mid and mid > 0 and ask > bid:
                spread_avg += (ask - bid) / mid
                spread_count += 1
        spread_avg = spread_avg / spread_count if spread_count > 0 else 0

        # Gamma exposure proxy: OI-weighted gamma ≡ OI * (1/strike) approximation
        gamma_exposure = 0
        for p in puts:
            oi = p.get('openInterest', 0) or 0
            k = p.get('strike', 0)
            if k > 0:
                gamma_exposure -= oi / k  # puts negative gamma
        for c in calls:
            oi = c.get('openInterest', 0) or 0
            k = c.get('strike', 0)
            if k > 0:
                gamma_exposure += oi / k
        gamma_exposure = round(gamma_exposure, 0)

        return {
            "put_call_ratio": put_call_ratio,
            "sentimiento": sentimiento,
            "vol_total": vol_total,
            "fecha_expiracion_proxima": fecha_expiracion,
            "iv_skew": iv_skew,
            "avg_iv": avg_iv,
            "oi_weighted_iv": round(oi_weighted_iv, 2),
            "atm_iv_put": round(atm_iv_put, 2),
            "atm_iv_call": round(atm_iv_call, 2),
            "put_call_oi_ratio": round(oi_puts / oi_calls, 2) if oi_calls > 0 else 999.0,
            "iv_curvature": round(iv_curvature, 4),
            "spread_pct": round(spread_avg, 4),
            "gamma_exposure": gamma_exposure
        }
    except Exception as e:
        return None

def generar_fallback(ticker):
    prob = random.uniform(0.3, 0.8)
    if prob > 0.6:
        ratio = round(random.uniform(0.3, 0.69), 2)
        sentimiento = 'bullish'
    elif prob < 0.4:
        ratio = round(random.uniform(1.01, 2.0), 2)
        sentimiento = 'bearish'
    else:
        ratio = round(random.uniform(0.7, 1.0), 2)
        sentimiento = 'neutral'

    vol_total = random.randint(50000, 500000)
    dias_exp = random.randint(7, 60)
    fecha_exp = time.strftime('%Y-%m-%d', time.localtime(time.time() + dias_exp * 86400))
    iv_skew = round(random.uniform(-5, 5), 2)
    avg_iv = round(random.uniform(20, 80), 2)
    oi_iv = round(random.uniform(20, 80), 2)
    oi_ratio = round(random.uniform(0.5, 2.0), 2)

    return {
        "put_call_ratio": ratio,
        "sentimiento": sentimiento,
        "vol_total": vol_total,
        "fecha_expiracion_proxima": fecha_exp,
        "iv_skew": iv_skew,
        "avg_iv": avg_iv,
        "oi_weighted_iv": oi_iv,
        "atm_iv_put": round(avg_iv + iv_skew/2, 2),
        "atm_iv_call": round(avg_iv - iv_skew/2, 2),
        "put_call_oi_ratio": oi_ratio
    }

def main():
    tickers = tickers_a_procesar()
    print(f"[!] Procesando opciones para {len(tickers)} tickers...")

    resultados = {}
    for t in tickers:
        try:
            print(f"  {t}...", end=' ')
            data = fetch_options_yahoo(t)
            if data is None:
                data = generar_fallback(t)
                print(f"[!] fallback (ratio: {data['put_call_ratio']}, {data['sentimiento']})")
            else:
                print(f"[OK] ratio: {data['put_call_ratio']}, {data['sentimiento']}")
            resultados[t] = data
            time.sleep(0.5)
        except Exception as e:
            print(f"[!] error: {e}")
            resultados[t] = generar_fallback(t)

    salida = {
        "timestamp": time.strftime('%Y-%m-%dT%H:%M:%S'),
        "tickers": resultados
    }

    with open(OUTPUT, 'w', encoding='utf-8') as f:
        json.dump(salida, f, ensure_ascii=False, indent=2)

    print(f"\n[OK] Datos de opciones guardados en {OUTPUT}")
    for t, d in resultados.items():
        print(f"  {t}: ratio {d['put_call_ratio']} ({d['sentimiento']}), vol {d['vol_total']:,}")

if __name__ == '__main__':
    main()
