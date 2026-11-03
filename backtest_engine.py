#!/usr/bin/env python3
"""
backtest_engine.py - Motor de backtest sobre precios historicos reales.

Regla de diseno: NUNCA fabrica datos. Si yfinance no entrega precio para un
ticker, ese ticker se excluye del backtest y queda registrado en `omitidos`.
Si no hay ningun ticker con datos, se devuelve `estado='sin_datos'` en lugar
de una curva de rendimiento inventada.

Metodologia:
- Las señales se calculan sobre el cierre del día t.
- La ejecución ocurre a la apertura del día t+1 (sin lookahead).
- El coste se cobra sobre el turnover en cada cambio de posición.
- Benchmark: buy & hold equiponderado con la misma asignación inicial.
"""
import json
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from config.settings import get_setting
from logger import get_module_logger

_log = get_module_logger('backtest_engine')

DATA_DIR = os.environ.get('GITHUB_WORKSPACE', '.')
OUTPUT_DIR = Path(DATA_DIR) / 'Datos'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_PATH = OUTPUT_DIR / 'backtest.json'

BT_CONFIG = get_setting('portfolio.backtest', {})

TICKERS_CORE = get_setting('tickers', 'core', default=[])

PERIODO_DEFAULT = BT_CONFIG.get('periodo', '2y')
CAPITAL_INICIAL = BT_CONFIG.get('capital_inicial', 1_000_000)
COMISION_BPS = BT_CONFIG.get('comision_bps', 5.0)
SLIPPAGE_BPS = BT_CONFIG.get('slippage_bps', 5.0)
RF_ANUAL = BT_CONFIG.get('rf_anual', 0.04)
DIAS_ANIO = 252


# ============================================================
# INDICADORES (implementaciones locales, sin dependencias de otros
# modulos del repo para evitar ciclos de import)
# ============================================================

def calc_ema(serie: pd.Series, periodo: int) -> pd.Series:
    """EMA con alpha = 2/(periodo+1), ajustada al primer valor."""
    return serie.ewm(span=periodo, adjust=False).mean()


def calc_rsi_wilder(serie: pd.Series, periodo: int = 14) -> pd.Series:
    """RSI de Wilder (media suavizada), acotado a [0, 100]."""
    delta = serie.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.ewm(alpha=1 / periodo, adjust=False, min_periods=periodo).mean()
    avg_loss = loss.ewm(alpha=1 / periodo, adjust=False, min_periods=periodo).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50.0).clip(0, 100)


def calc_macd_completo(serie: pd.Series, rapida: int = 12, lenta: int = 26,
                       senal: int = 9) -> pd.DataFrame:
    """MACD estándar: linea, señal EMA e histograma."""
    ema_rapida = calc_ema(serie, rapida)
    ema_lenta = calc_ema(serie, lenta)
    linea = ema_rapida - ema_lenta
    linea_senal = calc_ema(linea, senal)
    return pd.DataFrame({
        'macd': linea,
        'senal': linea_senal,
        'histograma': linea - linea_senal,
    })


def generar_senales(cierre: pd.Series, rsi_max: int = 70,
                    rsi_salida: int = 80) -> pd.Series:
    """
    Señal long-only emitida al cierre de cada día (1 = dentro, 0 = fuera).

    Entra cuando el histograma MACD cruza por encima de cero y el RSI no está
    en sobrecompra. Sale cuando el histograma vuelve a cero o el RSI entra en
    sobrecompra extrema.
    """
    # Sin historial suficiente no se generan operaciones: no se inventa señal.
    if len(cierre) < 35:
        return pd.Series(0.0, index=cierre.index)

    macd = calc_macd_completo(cierre)
    rsi = calc_rsi_wilder(cierre)

    hist = macd['histograma']
    entrada = (hist > 0) & (hist.shift(1) <= 0) & (rsi < rsi_max)
    salida = (hist < 0) | (rsi > rsi_salida)

    posicion = pd.Series(0.0, index=cierre.index)
    dentro = False
    entrada_v = entrada.to_numpy(dtype=bool)
    salida_v = salida.to_numpy(dtype=bool)
    valores = np.zeros(len(cierre), dtype=float)
    for i in range(len(cierre)):
        if not dentro and entrada_v[i]:
            dentro = True
        elif dentro and salida_v[i]:
            dentro = False
        valores[i] = 1.0 if dentro else 0.0

    return pd.Series(valores, index=cierre.index)


# ============================================================
# DATOS
# ============================================================

def _columnas_de_precios(datos: pd.DataFrame, tickers: list[str]) -> pd.DataFrame | None:
    """
    Normaliza la salida de yf.download a un DataFrame de cierres indexado por ticker.

    yfinance devuelve MultiIndex (campo, ticker) con group_by='ticker' y columnas
    planas cuando solo se pide un unico simbolo.
    """
    if datos is None or datos.empty:
        return None

    if isinstance(datos.columns, pd.MultiIndex):
        # Busca el nivel que contiene los campos de OHLCV.
        nivel_campo = next(
            (i for i in range(datos.columns.nlevels)
             if 'Close' in set(datos.columns.get_level_values(i))),
            None,
        )
        if nivel_campo is None:
            return None
        cierre = datos.xs('Close', axis=1, level=nivel_campo)
    elif 'Close' in datos.columns:
        if len(tickers) != 1:
            return None
        cierre = datos[['Close']].rename(columns={'Close': tickers[0]})
    else:
        return None

    cierre = cierre[[t for t in tickers if t in cierre.columns]]
    if cierre.empty:
        return None

    cierre.columns = [str(c).upper().strip() for c in cierre.columns]
    return cierre


def descargar_precios(tickers: list[str], periodo: str = PERIODO_DEFAULT) -> dict[str, Any]:
    """Descarga cierres reales. Devuelve (frame, omitidos_con_motivo)."""
    try:
        import yfinance as yf
    except ImportError:
        return None, {'*': 'yfinance no instalado'}

    try:
        datos = yf.download(tickers, period=periodo, interval='1d',
                            group_by='ticker', progress=False,
                            auto_adjust=True)
    except Exception as e:
        return None, {'*': f'descarga fallo: {str(e)[:120]}'}

    frame = _columnas_de_precios(datos, tickers)
    if frame is None or frame.empty:
        return None, {'*': 'respuesta vacia o formato inesperado'}

    validos = [t for t in frame.columns if frame[t].dropna().shape[0] >= 60]
    omitidos = {
        t: 'sin datos suficientes (<60 observaciones)'
        for t in tickers if t not in validos
    }
    if not validos:
        return None, omitidos or {'*': 'ningun ticker con >=60 observaciones'}

    return frame[validos].dropna(how='all').ffill(), omitidos


# ============================================================
# SIMULACION
# ============================================================

def _metricas(valor: pd.Series, retornos: pd.Series, trades: int,
              exposicion: float, rf_anual: float = RF_ANUAL) -> dict[str, Any]:
    if len(valor) < 2:
        return {}

    anos = max(len(valor) / DIAS_ANIO, 1e-9)
    retorno_total = float(valor.iloc[-1] / valor.iloc[0] - 1)
    cagr = float((1 + retorno_total) ** (1 / anos) - 1) if retorno_total > -1 else -1.0
    vol = float(retornos.std(ddof=1) * np.sqrt(DIAS_ANIO)) if retornos.std(ddof=1) == retornos.std(ddof=1) else 0.0
    sharpe = float((cagr - rf_anual) / vol) if vol > 0 else 0.0
    downside = retornos[retornos < 0]
    downside_vol = float(downside.std(ddof=1) * np.sqrt(DIAS_ANIO)) if len(downside) > 1 else 0.0
    sortino = float((cagr - rf_anual) / downside_vol) if downside_vol > 0 else 0.0

    pico = valor.cummax()
    drawdown = valor / pico - 1
    max_dd = float(drawdown.min())
    calmar = float(cagr / abs(max_dd)) if max_dd < 0 else 0.0

    positivos = int((retornos > 0).sum())
    evaluables = int((retornos != 0).sum())

    return {
        'retorno_total': round(retorno_total, 4),
        'cagr': round(cagr, 4),
        'volatilidad_anual': round(vol, 4),
        'sharpe': round(sharpe, 4),
        'sortino': round(sortino, 4),
        'max_drawdown': round(max_dd, 4),
        'calmar': round(calmar, 4),
        'win_rate': round(positivos / evaluables, 4) if evaluables else 0.0,
        'n_periodos': len(retornos),
        'num_trades': int(trades),
        'exposicion_media': round(exposicion, 4),
    }


def simular_ticker(cierre: pd.Series, capital: float,
                   coste_bps: float) -> dict[str, Any]:
    """Simula un único activo en modo long-only con ejecución t+1."""
    posicion = generar_senales(cierre).shift(1).fillna(0.0)

    retornos_cierre = cierre.pct_change().fillna(0.0)
    turnover = posicion.diff().fillna(posicion.iloc[0] if len(posicion) else 0.0).abs()
    coste = turnover * (coste_bps / 10000)

    retornos = posicion * retornos_cierre - coste
    valor = capital * (1 + retornos).cumprod()

    return {
        'valor': valor,
        'retornos': retornos,
        'trades': int((turnover > 0).sum()),
        'exposicion': float(posicion.mean()),
        'renta_final': float(retornos.sum()),
    }


def run_backtest(tickers: list[str] | None = None, start_date: str | None = None,
                 end_date: str | None = None,
                 capital_inicial: float = CAPITAL_INICIAL,
                 comision_bps: float = COMISION_BPS,
                 slippage_bps: float = SLIPPAGE_BPS) -> dict[str, Any]:
    """
    Ejecuta el backtest y devuelve métricas agregadas por activo y de cartera.

    `start_date` / `end_date` acotan el rango del análisis una vez descargados
    los datos; la descarga usa `periodo` porque yfinance no siempre respeta
    rangos absolutos en descargas por lote.
    """
    universo = TICKERS_CORE if tickers is None else [
        t.strip().upper() for t in tickers if t and t.strip()]
    if not universo:
        # Lista vacia explicita != no informada: no se cae al universo por defecto.
        return {'estado': 'sin_datos', 'motivo': 'lista de tickers vacia',
                'tickers_solicitados': [], 'timestamp': _timestamp()}

    inicio = time.time()
    _log.info(f'Backtest: {len(universo)} tickers, capital={capital_inicial:,.0f}')

    precios, omitidos = descargar_precios(universo)
    if precios is None:
        _log.warning(f'Sin datos de mercado: {omitidos}')
        return {'estado': 'sin_datos', 'motivo': omitidos,
                'tickers_solicitados': universo, 'timestamp': _timestamp()}

    if start_date or end_date:
        precios = precios.loc[start_date:end_date] if start_date and end_date \
            else precios.loc[end_date:] if end_date else precios.loc[start_date:]

    if precios.empty or precios.shape[0] < 60:
        return {'estado': 'sin_datos', 'motivo': 'rango de fechas sin observaciones suficientes',
                'tickers_solicitados': universo, 'timestamp': _timestamp()}

    usable = [t for t in precios.columns if precios[t].notna().sum() >= 60]
    if not usable:
        return {'estado': 'sin_datos', 'motivo': 'ningun ticker con >=60 observaciones',
                'tickers_solicitados': universo, 'timestamp': _timestamp()}

    capital_por_activo = capital_inicial / len(usable)
    coste_total = comision_bps + slippage_bps

    por_activo: dict[str, Any] = {}
    curvas = {}
    operaciones = 0
    exposicion_media = 0.0

    for t in usable:
        sim = simular_ticker(precios[t].dropna(), capital_por_activo, coste_total)
        curvas[t] = sim['valor']
        operaciones += sim['trades']
        exposicion_media += sim['exposicion']
        m = _metricas(sim['valor'], sim['retornos'], sim['trades'], sim['exposicion'])
        m['renta_acumulada'] = round(sim['renta_final'], 4)
        por_activo[t] = m

    valor_cartera = pd.DataFrame(curvas).sum(axis=1).dropna()
    retornos_cartera = valor_cartera.pct_change().dropna()
    exposicion_pond = exposicion_media / len(usable)

    portafolio = _metricas(valor_cartera, retornos_cartera, operaciones, exposicion_pond)

    benchmark_valor = (precios[usable] / precios[usable].iloc[0]).sum(axis=1) * capital_por_activo
    benchmark_ret = benchmark_valor.pct_change().dropna()
    benchmark = _metricas(benchmark_valor, benchmark_ret, 0, 1.0)
    benchmark['descripcion'] = 'buy & hold equiponderado'

    alfa = portafolio.get('cagr', 0.0) - benchmark.get('cagr', 0.0)

    resultado = {
        'estado': 'ok',
        'timestamp': _timestamp(),
        'periodo': {
            'desde': precios.index[0].strftime('%Y-%m-%d'),
            'hasta': precios.index[-1].strftime('%Y-%m-%d'),
            'observaciones': int(precios.shape[0]),
        },
        'configuracion': {
            'capital_inicial': capital_inicial,
            'comision_bps': comision_bps,
            'slippage_bps': slippage_bps,
            'coste_total_bps': coste_total,
            'ejecucion': 'T+1 a la apertura',
            'rf_anual': RF_ANUAL,
        },
        'tickers': usable,
        'omitidos': omitidos,
        'portafolio': portafolio,
        'benchmark': benchmark,
        'alpha_anual': round(alfa, 4),
        'por_activo': por_activo,
        'curva_valor': {d.strftime('%Y-%m-%d'): round(float(v), 2)
                        for d, v in valor_cartera.items()},
        'tiempo_s': round(time.time() - inicio, 2),
    }

    OUTPUT_PATH.write_text(json.dumps(resultado, indent=2), encoding='utf-8')
    _log.info(f'Backtest OK: retorno={portafolio.get("retorno_total")} '
              f'sharpe={portafolio.get("sharpe")} vs benchmark '
              f'{benchmark.get("retorno_total")}')

    if omitidos:
        _log.warning(f'Tickers omitidos ({len(omitidos)}): {omitidos}')

    return resultado


def _timestamp() -> str:
    return datetime.now().strftime('%Y-%m-%dT%H:%M:%S')


if __name__ == '__main__':
    import sys

    tickers_arg = sys.argv[1].split(',') if len(sys.argv) > 1 else None
    r = run_backtest(tickers=tickers_arg)
    if r.get('estado') != 'ok':
        print(f"[!] Backtest sin datos: {r.get('motivo')}")
        sys.exit(1)
    p, b = r['portafolio'], r['benchmark']
    print(f"[OK] {r['periodo']['desde']} -> {r['periodo']['hasta']} "
          f"({r['periodo']['observaciones']} sesiones)")
    print(f"     Portafolio: retorno={p['retorno_total']:.2%} sharpe={p['sharpe']:.2f} "
          f"maxDD={p['max_drawdown']:.2%}")
    print(f"     Benchmark:  retorno={b['retorno_total']:.2%} sharpe={b['sharpe']:.2f} "
          f"maxDD={b['max_drawdown']:.2%}")
    print(f"     Alpha anual: {r['alpha_anual']:.2%} | trades={p['num_trades']} | "
          f"expuesto={p['exposicion_media']:.1%}")
    print(f"[OK] Guardado en {OUTPUT_PATH}")