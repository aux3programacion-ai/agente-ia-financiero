#!/usr/bin/env python3
"""backtest_conviccion.py - Backtest honesto: proba del modelo -> sizing por conviccion.

Para cada ticker: predice proba OOS con CV cronologico (cross_val_predict),
posiciona long con peso = clip((proba-0.5)*2, 0, 1), retorna r[t+1][t+2]...
Compara vs buy&hold SPY y equal-weight naive.
"""
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

os.environ.setdefault('PYTHONIOENCODING', 'utf-8')

from modelo_xgboost import FeatureBuilder, _ChronologicalBlockCV  # noqa

TICKERS = ['NVDA','MU','AVGO','TSM','ARM','DELL','SMCI','HPE','DDOG','SNOW','NOW',
           'CRWD','PANW','OKTA','NTAP','CLS','AAPL','AMZN','GOOGL','META','MSFT',
           'LLY','AMAT','LRCX','ORCL','HON','GE','UBER','COST','NEE']


def main():
    import yfinance as yf
    from sklearn.model_selection import cross_val_predict
    import xgboost as xgb

    def _expanding_past_cv(n_splits=4):
        def _split(X, y=None, groups=None):
            n = len(X)
            fold = max(1, n // (n_splits + 1))
            for i in range(n_splits):
                test_start = (i + 1) * fold
                test_end = n if i == n_splits - 1 else (i + 2) * fold
                train_idx = np.arange(0, test_start)
                test_idx = np.arange(test_start, test_end)
                yield train_idx, test_idx
        class _CV:
            def get_n_splits(self, X=None, y=None, groups=None):
                return n_splits
            def split(self, X, y=None, groups=None):
                return _split(X, y, groups)
        return _CV()

    spy = yf.download('SPY', period='2y', interval='1d', auto_adjust=True,
                      progress=False, multi_level_index=False)['Close']
    spy_ret = spy.pct_change().dropna()
    strat_rets = []
    spy_rets_aligned = []
    naive_rets = []

    for tk in TICKERS:
        try:
            df = yf.download(tk, period='2y', interval='1d', auto_adjust=True,
                             progress=False, multi_level_index=False)
            if df is None or len(df) < 120:
                continue
            df, fcols = FeatureBuilder.build_features(df, spy_close=spy)
            dfm = df[fcols + ['target']].dropna()
            X = dfm[fcols].values
            y = dfm['target'].values.astype(int)
            n = len(X)
            n_splits = 4
            fold = max(1, n // (n_splits + 1))
            proba = np.full(n, np.nan)
            for i in range(n_splits):
                te_start = (i + 1) * fold
                te_end = n if i == n_splits - 1 else (i + 2) * fold
                tr_end = te_start - 20  # embargo = horizonte del label
                if tr_end < 50:
                    continue
                m = xgb.XGBClassifier(n_estimators=200, max_depth=5, learning_rate=0.05,
                                      random_state=42, verbosity=0, use_label_encoder=False)
                m.fit(X[:tr_end], y[:tr_end])
                proba[te_start:te_end] = m.predict_proba(X[te_start:te_end])[:, 1]
            valid_p = ~np.isnan(proba)
            proba = proba[valid_p]
            weight = np.clip((proba - 0.5) * 2, 0, 1)
            r1 = df['Close'].pct_change().shift(-1).reindex(dfm.index).values[valid_p]  # r[t+1]
            keep = ~np.isnan(r1)
            w = weight[keep]
            gross = w * r1[keep]
            # costos: 10 bps por cada cambio unitario de peso
            dw = np.abs(np.diff(np.concatenate([[0.0], w])))
            net = gross - 0.001 * dw
            idx = dfm.index[valid_p][keep]
            ret = pd.Series(net, index=idx)
            naive_rets.append(df['Close'].pct_change().shift(-1).reindex(dfm.index).values[valid_p][keep] if False else pd.Series(r1[keep], index=idx))
            strat_rets.append(ret)
            spy_rets_aligned.append(spy_ret.reindex(ret.index).dropna())
        except Exception as e:
            print(f'{tk}: ERROR {e}')

    if not strat_rets:
        print('Sin datos')
        return
    port = pd.concat(strat_rets, axis=1).mean(axis=1)
    spyx = pd.concat(spy_rets_aligned, axis=1).mean(axis=1).reindex(port.index).fillna(0)

    def stats(r):
        ann = (1 + r).prod() ** (252 / len(r)) - 1
        vol = r.std() * np.sqrt(252)
        sharpe = ann / vol if vol > 0 else 0
        dd = ((1 + r).cumprod() / (1 + r).cumprod().cummax() - 1).min()
        return ann, vol, sharpe, dd

    naive = pd.concat(naive_rets, axis=1).mean(axis=1)
    ns = (1 + naive).prod() ** (252 / len(naive)) - 1
    nv = naive.std() * np.sqrt(252)
    ns_sharpe = ns / nv if nv > 0 else 0
    print(f'EqWeight 30 tickers B&H: ret_anual={ns*100:.1f}%  vol={nv*100:.1f}%  sharpe={ns_sharpe:.2f}')

    a, v, s, d = stats(port)
    sa, sv, ss, sd = stats(spyx)
    print(f'Estrategia: ret_anual={a*100:.1f}%  vol={v*100:.1f}%  sharpe={s:.2f}  maxDD={d*100:.1f}%')
    print(f'SPY B&H:    ret_anual={sa*100:.1f}%  vol={sv*100:.1f}%  sharpe={ss:.2f}  maxDD={sd*100:.1f}%')

    out = {'estrategia': {'ret_anual': a, 'vol': v, 'sharpe': s, 'maxdd': d},
           'spy': {'ret_anual': sa, 'vol': sv, 'sharpe': ss, 'maxdd': sd}}
    Path('Datos/backtest_conviccion.json').write_text(json.dumps(out, indent=2))
    print('Guardado en Datos/backtest_conviccion.json')


if __name__ == '__main__':
    main()
