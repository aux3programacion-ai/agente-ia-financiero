import json, os, sys, datetime, numpy as np, pandas as pd, warnings
warnings.filterwarnings('ignore')

DATA_DIR = os.environ.get('GITHUB_WORKSPACE', '.')

modelo_xgboost = {}
predicciones_hist = {}
analisis_ia = {}

try:
    with open(f'{DATA_DIR}/Datos/modelo_xgboost.json', 'r') as f:
        modelo_xgboost = json.load(f)
    print(f'[OK] modelo_xgboost.json loaded')
except Exception as e:
    print(f'[!] Could not load modelo_xgboost.json: {e}')

try:
    with open(f'{DATA_DIR}/Datos/predicciones_hist.json', 'r') as f:
        predicciones_hist = json.load(f)
    print(f'[OK] predicciones_hist.json loaded')
except Exception as e:
    print(f'[!] Could not load predicciones_hist.json: {e}')

try:
    with open(f'{DATA_DIR}/Datos/analisis_ia.json', 'r') as f:
        analisis_ia = json.load(f)
    print(f'[OK] analisis_ia.json loaded')
except Exception as e:
    print(f'[!] Could not load analisis_ia.json: {e}')

ia_probs = analisis_ia.get('probabilidades', {})

tickers_list = []
if 'tickers' in modelo_xgboost and modelo_xgboost['tickers']:
    tickers_list = list(modelo_xgboost['tickers'].keys())

if not tickers_list:
    print('[!] No tickers in modelo_xgboost, trying predicciones_hist')
    tickers_list = list(predicciones_hist.keys())

print(f'\n=== Computing SHAP-like Explanations ===\n')

explicacion_output = {
    'timestamp': datetime.datetime.now().strftime('%Y-%m-%dT%H:%M:%SZ'),
    'tickers': {}
}

import datetime

for ticker in sorted(tickers_list):
    try:
        xgb_data = modelo_xgboost.get('tickers', {}).get(ticker, {})
        prob_xgboost = xgb_data.get('prob_up_20d', 50)
        features_top = xgb_data.get('features_top', ['return_5d', 'rsi_14', 'vol_ratio'])

        ia_info = ia_probs.get(ticker, {})
        prob_ia = ia_info.get('probabilidad', 50)
        precision_hist = ia_info.get('precision_historica', None)
        if precision_hist is None:
            precision_hist = modelo_xgboost.get('precision_test', 0.55)

        hist_preds = predicciones_hist.get(ticker, {}).get('predicciones', [])
        if hist_preds:
            historicos = [p for p in hist_preds if p.get('acertada') is not None]
            if historicos:
                precision_ticker = sum(1 for p in historicos if p['acertada']) / len(historicos)
            else:
                precision_ticker = precision_hist
        else:
            precision_ticker = precision_hist

        total_prec = precision_ticker + precision_hist
        if total_prec > 0:
            weight_ia = precision_hist / total_prec
            weight_xgb = precision_ticker / total_prec
        else:
            weight_ia = weight_xgb = 0.5

        prob_consenso = (prob_ia * weight_ia + prob_xgboost * weight_xgb)

        feature_names_map = {
            'rsi_14': 'RSI(14)',
            'return_1d': 'Return 1d',
            'return_5d': 'Return 5d',
            'return_20d': 'Return 20d',
            'macd_hist': 'MACD Hist',
            'vol_ratio': 'Vol ratio',
            'volatility_20d': 'Volatility 20d',
            'sma50_dist_pct': 'Dist SMA50',
            'sma200_dist_pct': 'Dist SMA200'
        }

        fi_global = modelo_xgboost.get('feature_importance_global', {})
        total_fi = sum(fi_global.get(f, 0) for f in features_top) or 1

        factores = []
        for feat in features_top:
            peso = fi_global.get(feat, 0.1) / total_fi
            feat_name = feature_names_map.get(feat, feat)
            if feat in ['rsi_14', 'macd_hist', 'return_1d', 'return_5d', 'return_20d']:
                contribuye = 'alcista' if prob_xgboost > 55 else 'bajista'
            elif feat == 'vol_ratio':
                contribuye = 'alcista' if prob_xgboost > 55 else 'bajista'
            else:
                contribuye = 'alcista' if prob_xgboost > 50 else 'bajista'
            factores.append({
                'nombre': feat_name,
                'contribuye': contribuye,
                'peso': round(peso, 4)
            })

        if prob_consenso > 60:
            if prob_xgboost > prob_ia:
                explicacion = 'XGBoost senial alcista domina, respaldada por el analisis de IA.'
            else:
                explicacion = 'IA sugiere presion alcista, confirmada por el modelo XGBoost.'
        elif prob_consenso < 40:
            if prob_xgboost < prob_ia:
                explicacion = 'XGBoost senial bajista domina, consistente con el panorama de IA.'
            else:
                explicacion = 'IA y XGBoost coinciden en senial bajista.'
        else:
            if abs(prob_xgboost - prob_ia) > 15:
                explicacion = 'Senales mixtas entre XGBoost e IA, se recomienda cautela.'
            else:
                explicacion = 'Mercado neutral sin direccion clara segun ambos modelos.'

        precision_hist_val = round(precision_ticker, 4)

        explicacion_output['tickers'][ticker] = {
            'prob_ia': round(prob_ia, 1),
            'prob_xgboost': round(prob_xgboost, 1),
            'prob_consenso': round(prob_consenso, 1),
            'factores': factores,
            'precision_hist': precision_hist_val,
            'explicacion': explicacion
        }

        print(f'  {ticker}: consenso={prob_consenso:.0f}% ia={prob_ia}% xgb={prob_xgboost}% prec={precision_hist_val:.2f}')

    except Exception as e:
        print(f'[!] Error processing {ticker}: {e}')
        continue

try:
    os.makedirs(f'{DATA_DIR}/Datos', exist_ok=True)
    with open(f'{DATA_DIR}/Datos/explicabilidad.json', 'w') as f:
        json.dump(explicacion_output, f, indent=2)
    print(f'\n[OK] explicabilidad.json saved ({len(explicacion_output["tickers"])} tickers)')
except Exception as e:
    print(f'[!] Save failed: {e}')

print(f'\n=== SHAP Explainability Summary ===')
tickers_done = explicacion_output['tickers']
print(f'  Tickers explained: {len(tickers_done)}')
for tk, tv in sorted(tickers_done.items()):
    consenso = tv['prob_consenso']
    icon = '[*]' if consenso > 60 else ('[v]' if consenso < 40 else '[-]')
    print(f'  {icon} {tk}: consenso={consenso:.0f}% (ia={tv["prob_ia"]}%, xgb={tv["prob_xgboost"]}%)')
