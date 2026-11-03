#!/usr/bin/env python3
"""seguimiento_senales.py - Registro y evaluacion de senales en vivo.

Cada corrida:
  1. Lee Datos/modelo_xgboost.json y registra (fecha, ticker, proba, precio).
  2. Evalua las senales de hace >= 5 dias contra el retorno realizado de 5 dias
     vs SPY (exceso): acierto direccional y Brier score.
Salida: Datos/senales_log.jsonl (append), Datos/senales_evaluacion.json.
"""
import json
from datetime import datetime, timezone, timedelta
from pathlib import Path

import numpy as np

DATOS = Path('Datos')
LOG = DATOS / 'senales_log.jsonl'
EVAL = DATOS / 'senales_evaluacion.json'
HORIZON_DAYS = 5


def _now():
    return datetime.now(timezone.utc)


def log_signals():
    xgb = json.loads((DATOS / 'modelo_xgboost.json').read_text(encoding='utf-8'))
    tickers = xgb.get('tickers', {})
    entry_ts = _now().isoformat()
    with open(LOG, 'a', encoding='utf-8') as f:
        for tk, tr in tickers.items():
            f.write(json.dumps({
                'ts': entry_ts, 'ticker': tk,
                'prob_up_20d': tr.get('prob_up_20d'),
                'wf_accuracy': tr.get('wf_accuracy'),
                'fwd_vol': tr.get('fwd_vol_20d_annual'),
            }) + '\n')
    print(f'[log] {len(tickers)} senales registradas en {entry_ts}')


def eval_past():
    if not LOG.exists():
        return
    import yfinance as yf
    lines = [json.loads(l) for l in LOG.read_text(encoding='utf-8').splitlines() if l.strip()]
    cutoff = _now() - timedelta(days=HORIZON_DAYS)
    old = [e for e in lines if datetime.fromisoformat(e['ts']) <= cutoff]
    if len(old) < 5:
        print('[eval] aun no hay senales con horizonte cumplido')
        return
    hits, briers = [], []
    by_ts = {}
    for e in old:
        by_ts.setdefault(e['ts'], []).append(e)
    for ts, entries in by_ts.items():
        start = datetime.fromisoformat(ts)
        try:
            spy = yf.Ticker('SPY').history(start=start, period='10d')
            if len(spy) < 2:
                continue
            spy_ret = float(spy['Close'].iloc[-1] / spy['Close'].iloc[0] - 1)
            for e in entries:
                h = yf.Ticker(e['ticker']).history(start=start, period='10d')
                if len(h) < 2 or e['prob_up_20d'] is None:
                    continue
                realized_excess = float(h['Close'].iloc[-1] / h['Close'].iloc[0] - 1) - spy_ret
                y = 1 if realized_excess > 0 else 0
                p = e['prob_up_20d'] / 100.0
                hits.append(1 if (p > 0.5) == bool(y) else 0)
                briers.append((p - y) ** 2)
        except Exception as ex:
            print(f'[eval] skip {ts}: {ex}')
    if hits:
        out = {
            'n_evaluadas': len(hits),
            'accuracy_direccional_5d': round(float(np.mean(hits)), 4),
            'brier': round(float(np.mean(briers)), 4),
            'timestamp': _now().isoformat(),
        }
        EVAL.write_text(json.dumps(out, indent=2), encoding='utf-8')
        print(f'[eval] {out}')
    else:
        print('[eval] sin datos suficientes')


if __name__ == '__main__':
    eval_past()
    log_signals()
