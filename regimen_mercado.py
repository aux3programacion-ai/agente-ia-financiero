import json, os, sys, yfinance as yf, numpy as np, pandas as pd, time, datetime, warnings
warnings.filterwarnings('ignore')

DATA_DIR = os.environ.get('GITHUB_WORKSPACE', '.')

print(f'\n=== Market Regime Detection ===\n')

try:
    print('  Fetching SPY 2y data...', end=' ')
    spy = yf.download('SPY', period='2y', interval='1d', auto_adjust=True, progress=False, multi_level_index=False)
    if spy is None or spy.empty:
        raise Exception('No data received')
    if isinstance(spy.columns, pd.MultiIndex):
        spy = spy.xs('SPY', level=0, axis=1) if 'SPY' in spy.columns.get_level_values(0) else spy
    spy = spy.dropna(subset=['Close'])
    print(f'[OK] {len(spy)} days')
except Exception as e:
    print(f'[!] Failed: {e}')
    spy = pd.DataFrame()

if spy.empty:
    fallback = {
        'timestamp': datetime.datetime.now().strftime('%Y-%m-%dT%H:%M:%SZ'),
        'regimen': 'INCIERTO',
        'confianza': 0.0,
        'spy_precio': 0.0,
        'sma50': 0.0,
        'sma200': 0.0,
        'dist_sma50_pct': 0.0,
        'dist_sma200_pct': 0.0,
        'volatilidad_20d': 0.0,
        'vol_regimen': 'normal',
        'fase': 'unknown',
        'tendencia_200d': 0.0,
        'dias_en_regimen': 0,
        'error': 'No SPY data available'
    }
    try:
        os.makedirs(f'{DATA_DIR}/Datos', exist_ok=True)
        with open(f'{DATA_DIR}/Datos/regimen_mercado.json', 'w') as f:
            json.dump(fallback, f, indent=2)
        print('[!] Fallback empty dict saved')
    except Exception as ex:
        print(f'[!] Save failed: {ex}')
    print(f'\n=== Regimen: INCIERTO (no data) ===')
    sys.exit(0)

returns = spy['Close'].pct_change()
vol_20d = returns.rolling(window=20).std() * np.sqrt(252) * 100
return_20d = spy['Close'].pct_change(20) * 100
return_50d = spy['Close'].pct_change(50) * 100
return_200d = spy['Close'].pct_change(200) * 100
sma50 = spy['Close'].rolling(window=50).mean()
sma200 = spy['Close'].rolling(window=200).mean()
dist_sma50 = (spy['Close'] - sma50) / sma50 * 100
dist_sma200 = (spy['Close'] - sma200) / sma200 * 100

latest_close = float(spy['Close'].iloc[-1])
latest_sma50 = float(sma50.iloc[-1]) if pd.notna(sma50.iloc[-1]) else latest_close
latest_sma200 = float(sma200.iloc[-1]) if pd.notna(sma200.iloc[-1]) else latest_close
latest_dist50 = float(dist_sma50.iloc[-1]) if pd.notna(dist_sma50.iloc[-1]) else 0.0
latest_dist200 = float(dist_sma200.iloc[-1]) if pd.notna(dist_sma200.iloc[-1]) else 0.0
latest_ret50d = float(return_50d.iloc[-1]) if pd.notna(return_50d.iloc[-1]) else 0.0
latest_ret200d = float(return_200d.iloc[-1]) if pd.notna(return_200d.iloc[-1]) else 0.0
latest_vol = float(vol_20d.iloc[-1]) if pd.notna(vol_20d.iloc[-1]) else 0.0

valid_vol = vol_20d.dropna()
if len(valid_vol) > 0:
    vol_thresh_low = np.percentile(valid_vol, 25)
    vol_thresh_high = np.percentile(valid_vol, 75)
    if latest_vol > vol_thresh_high:
        vol_regimen = 'alta'
    elif latest_vol < vol_thresh_low:
        vol_regimen = 'baja'
    else:
        vol_regimen = 'normal'
else:
    vol_regimen = 'normal'

if latest_close > latest_sma50 and latest_sma50 > latest_sma200 and latest_ret50d > 5:
    regimen = 'ALCISTA'
    confianza = min(0.5 + (latest_ret50d / 40), 0.95)
    if latest_ret200d > 20:
        fase = 'late_expansion'
    elif latest_ret200d > 10:
        fase = 'mid_expansion'
    else:
        fase = 'early_expansion'
elif latest_close < latest_sma50 and latest_sma50 < latest_sma200 and latest_ret50d < -5:
    regimen = 'BAJISTA'
    confianza = min(0.5 + (abs(latest_ret50d) / 40), 0.95)
    if latest_ret200d < -20:
        fase = 'late_contraction'
    elif latest_ret200d < -10:
        fase = 'mid_contraction'
    else:
        fase = 'early_contraction'
elif latest_close > latest_sma50 and latest_close < latest_sma200:
    regimen = 'LATERAL'
    confianza = 0.5
    fase = 'range_bound'
elif latest_close < latest_sma50 and latest_close > latest_sma200:
    regimen = 'LATERAL'
    confianza = 0.4
    fase = 'pullback'
else:
    regimen = 'INCIERTO'
    confianza = 0.3
    if latest_sma50 > latest_sma200:
        fase = 'bullish_uncertain'
    else:
        fase = 'bearish_uncertain'

tendencia_200d = round(latest_ret200d, 2)

closes_series = spy['Close']
if regimen == 'ALCISTA':
    mask = (closes_series > sma50) & (sma50 > sma200) & (closes_series.pct_change(50) * 100 > 5)
elif regimen == 'BAJISTA':
    mask = (closes_series < sma50) & (sma50 < sma200) & (closes_series.pct_change(50) * 100 < -5)
elif regimen == 'LATERAL':
    mask = ((closes_series > sma50) & (closes_series < sma200)) | ((closes_series < sma50) & (closes_series > sma200))
else:
    mask = pd.Series(False, index=closes_series.index)

mask = mask.fillna(False)
consecutive = 0
dias_en_regimen = 0
for i in range(len(mask) - 1, -1, -1):
    if mask.iloc[i]:
        consecutive += 1
    else:
        break
dias_en_regimen = consecutive

output = {
    'timestamp': datetime.datetime.now().strftime('%Y-%m-%dT%H:%M:%SZ'),
    'regimen': regimen,
    'confianza': round(confianza, 2),
    'spy_precio': round(latest_close, 2),
    'sma50': round(latest_sma50, 2),
    'sma200': round(latest_sma200, 2),
    'dist_sma50_pct': round(latest_dist50, 2),
    'dist_sma200_pct': round(latest_dist200, 2),
    'volatilidad_20d': round(latest_vol, 2),
    'vol_regimen': vol_regimen,
    'fase': fase,
    'tendencia_200d': tendencia_200d,
    'dias_en_regimen': dias_en_regimen
}

try:
    os.makedirs(f'{DATA_DIR}/Datos', exist_ok=True)
    with open(f'{DATA_DIR}/Datos/regimen_mercado.json', 'w') as f:
        json.dump(output, f, indent=2)
    print(f'[OK] regimen_mercado.json saved')
except Exception as e:
    print(f'[!] Save failed: {e}')

print(f'\n=== Market Regime Summary ===')
print(f'  Regimen: {regimen}')
print(f'  Confianza: {confianza:.0%}')
print(f'  SPY: ${latest_close:.2f} (SMA50: ${latest_sma50:.2f}, SMA200: ${latest_sma200:.2f})')
print(f'  Dist SMA50: {latest_dist50:.2f}% | Dist SMA200: {latest_dist200:.2f}%')
print(f'  Vol 20d: {latest_vol:.2f}% ({vol_regimen})')
print(f'  Fase: {fase}')
print(f'  Tendencia 200d: {tendencia_200d:.2f}%')
print(f'  Dias en regimen: {dias_en_regimen}')
