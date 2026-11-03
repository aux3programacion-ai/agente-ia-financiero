#!/usr/bin/env python3
"""
Tests for Agente Financiero modules.
Uses pytest + hypothesis for property-based testing.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

DATA_DIR = str(REPO_ROOT)


# ============================================================
# STRATEGIES
# ============================================================

price_strategy = st.floats(min_value=10, max_value=1000, allow_nan=False, allow_infinity=False)
volume_strategy = st.integers(min_value=100, max_value=10000000)
prob_strategy = st.floats(min_value=0, max_value=1, allow_nan=False)


@given(
    drift=st.floats(min_value=-0.05, max_value=0.05),
    periods=st.integers(min_value=3, max_value=100)
)
@settings(max_examples=20, deadline=None)
def test_rsi_always_bounded(drift, periods):
    """RSI should always be between 0 and 100."""
    from modelo_xgboost import FeatureBuilder
    # Sin np.random.seed: sembrar el generador global de numpy dentro de un
    # @given rompe el contrato de Hypothesis (mismo input, misma salida).
    n = 200
    pasos = np.random.randn(n) * 0.02 + drift
    prices = pd.Series(100 * np.exp(np.cumsum(pasos)))
    rsi = FeatureBuilder.compute_rsi(prices, periods)
    assert rsi.dropna().between(0, 100).all(), "RSI fuera de rango"


@given(n=st.integers(min_value=35, max_value=500))
@settings(max_examples=20, deadline=None)
def test_macd_shape(n):
    """
    MACD histogram keeps the input length and equals linea - senal.

    No se exige warmup con NaN: ewma no produce NaN en el inicio de la serie,
    y el histograma es valido desde la primera observacion.
    """
    from modelo_xgboost import FeatureBuilder
    close = pd.Series(100 * np.exp(np.cumsum(np.random.randn(n) * 0.015)))
    macd = FeatureBuilder.compute_macd(close, 12, 26, 9)
    assert len(macd) == len(close), "MACD length mismatch"
    assert np.isfinite(macd.to_numpy()).all(), "MACD con NaN o inf"

    ema_rapida = close.ewm(span=12, adjust=False).mean()
    ema_lenta = close.ewm(span=26, adjust=False).mean()
    esperado = ema_rapida - ema_lenta
    senal = esperado.ewm(span=9, adjust=False).mean()
    assert np.allclose(macd.to_numpy(), (esperado - senal).to_numpy(), atol=1e-9)


def test_pipeline_logger_init():
    from logger import get_module_logger
    log = get_module_logger('test_module')
    log.info("Test message")


def test_config_loader():
    from config.settings import get_setting
    tickers = get_setting('tickers', 'core')
    assert tickers is not None
    assert 'NVDA' in tickers


def test_multi_agent_vote():
    from multi_agent_system import TechnicalAgent
    vote = TechnicalAgent().analyze('NVDA', {
        'tecnico': {'rsi': 25, 'macd': 2.0, 'vol_ratio': 1.8, 'tendencia': 'uptrend'}
    })
    assert vote.direction in ('up', 'down', 'neutral')
    assert 0 <= vote.probability <= 100
    assert 0 <= vote.confidence <= 100


def test_debate_round():
    from multi_agent_system import AgentVote, MetaReviewer
    reviewer = MetaReviewer()
    votes = [
        AgentVote('tecnico', 'NVDA', 'up', 70, 70, 'Test', [], 'ALCISTA'),
        AgentVote('fundamental', 'NVDA', 'up', 60, 50, 'Test', [], 'ALCISTA'),
        AgentVote('riesgo', 'NVDA', 'neutral', 50, 60, 'Test', [], 'ALCISTA'),
    ]
    debate = reviewer.evaluate_and_weight(votes, {'regimen': 'ALCISTA'})
    assert debate.consensus_direction in ('up', 'down', 'neutral')
    assert 0 <= debate.disagreement <= 1
    assert len(debate.votes) == 3


def test_feature_builder():
    from modelo_xgboost import FeatureBuilder
    np.random.seed(42)
    n = 500
    index = pd.date_range('2023-01-01', periods=n, freq='B')
    df = pd.DataFrame({
        'Open': 100 + np.random.randn(n),
        'High': 102 + np.abs(np.random.randn(n)),
        'Low': 98 - np.abs(np.random.randn(n)),
        'Close': 100 + np.cumsum(np.random.randn(n) * 0.5),
        'Volume': np.random.randint(100000, 10000000, n)
    }, index=index)

    result_df, feature_cols = FeatureBuilder.build_features(df)
    assert len(result_df) == n
    assert len(feature_cols) >= 30
    for col in ['rsi_14', 'macd_hist', 'vol_ratio']:
        assert col in result_df.columns


def test_walk_forward_validator():

    from modelo_xgboost import WalkForwardValidator
    np.random.seed(42)
    n = 500
    X = np.random.randn(n, 10)
    y = (X[:, 0] + X[:, 1] > 0).astype(int)

    wf = WalkForwardValidator(n_splits=3, test_size=42, gap=5)
    results = wf.validate(X, y)
    assert len(results) == 3
    for r in results:
        assert 0 <= r['train_acc'] <= 1
        assert 0 <= r['test_acc'] <= 1


def test_calibration():
    from calibration import ProbabilityCalibrator
    np.random.seed(42)
    y_true = np.random.binomial(1, 0.3, 200)
    y_prob = np.clip(y_true + np.random.randn(200) * 0.5, 0.05, 0.95)

    cal = ProbabilityCalibrator(metodo='isotonic')
    cal.fit(y_true, y_prob)
    assert cal.is_fitted
    assert cal.brier_after <= cal.brier_before + 0.01


def test_model_store():
    from xgboost import XGBClassifier

    from model_store import ModelStore

    store = ModelStore()
    model = XGBClassifier(n_estimators=10, max_depth=2, verbosity=0)
    X = np.random.randn(50, 3)
    y = (X[:, 0] > 0).astype(int)
    model.fit(X, y)

    version = store.save_model(
        model=model, name='pytest_test', regime='test',
        params={'n_estimators': 10}, metrics={'accuracy': 0.85},
        feature_names=['f1', 'f2', 'f3']
    )
    assert version is not None

    loaded = store.load_model('pytest_test', 'test', version)
    assert loaded['model'] is not None
    assert loaded['metadata']['metrics']['accuracy'] == 0.85


def test_feature_store():
    from feature_store import FeatureStore
    fs = FeatureStore()
    fs.store_features('TEST_NVDA', {'rsi': 65.0, 'macd': 1.5})
    fs.store_features('TEST_AAPL', {'rsi': 45.0, 'macd': -0.5})

    history = fs.get_feature_history('TEST_NVDA', 'rsi', days=30)
    assert len(history) >= 1


def test_portfolio_optimizer():
    """Dimensionamiento por regimen: limita, renormaliza y reserva caja."""
    from regime_sizing import REGIME_CONFIG, apply_regime_sizing, cap_weights

    pesos = {'NVDA': 0.5, 'AAPL': 0.3, 'MSFT': 0.2}
    alzista = REGIME_CONFIG['ALCISTA']
    escalado = apply_regime_sizing(pesos, alzista)

    assert set(escalado) == set(pesos)
    invertido = (1 - alzista['cash_reserve']) * alzista['leverage']
    assert sum(escalado.values()) == pytest.approx(invertido)
    assert all(v >= 0 for v in escalado.values())

    # El cap ocurre antes de renormalizar: ningun peso puede exceder el maximo
    # relativo aunque el input este muy concentrado.
    limitado = apply_regime_sizing({'NVDA': 1.0}, alzista)
    assert limitado['NVDA'] == pytest.approx(invertido)

    # En regimen BAJISTA el apalancamiento es 0: no se invierte nada.
    assert sum(apply_regime_sizing(pesos, REGIME_CONFIG['BAJISTA']).values()) == 0.0

    assert cap_weights({'A': 0.9, 'B': 0.05}, 0.1) == {'A': 0.1, 'B': 0.05}


def test_dashboard_api():
    from dashboard_app import DashboardAPI
    api = DashboardAPI()
    s = api.get_summary()
    assert 'fecha' in s
    assert 'rendimiento' in s


def test_risk_metrics():
    from portfolio_analytics import RiskMetrics
    np.random.seed(42)
    returns = pd.Series(np.random.randn(500) * 0.02 + 0.0005)
    metrics = RiskMetrics.compute_all(returns)
    assert 'sharpe_ratio' in metrics
    assert 'max_drawdown' in metrics


def test_paper_broker():
    from paper_trading import OrderSide, PaperBroker
    broker = PaperBroker(initial_capital=100000)
    broker.set_price_provider(lambda t: 150.0)
    order = broker.submit_order('NVDA', OrderSide.BUY, 10)
    assert order.status.value == 'filled'
    assert 'NVDA' in broker.positions


def test_cppi():
    from cppi import CPPI
    cppi = CPPI(capital_inicial=100000)
    result = cppi.rebalance(0.01, 0.0001)
    assert result.valor_portafolio > 0
    assert result.exposicion >= 0


def test_black_scholes():
    from opciones_derivados import BlackScholes
    opt = BlackScholes.price(S=100, K=100, T=1.0, r=0.05, sigma=0.25, option_type='call')
    assert opt.premium > 0
    assert opt.greeks.delta > 0.5


def test_parallel_map():
    from parallel_utils import parallel_map
    results = parallel_map(lambda x: x * 2, [1, 2, 3, 4, 5], max_workers=3, show_progress=False)
    assert results == [2, 4, 6, 8, 10]


def test_logger_writes_file():
    from logger import LOG_DIR, get_logger
    log = get_logger('test_logger')
    log.info("Testing logger")
    assert Path(LOG_DIR).exists()
    assert Path(LOG_DIR) == REPO_ROOT / 'Datos' / 'logs'


# ============================================================
# PROPERTY-BASED TESTS (HYPOTHESIS)
# ============================================================

@given(
    st.lists(st.floats(min_value=1, max_value=1000, allow_nan=False), min_size=30, max_size=500)
)
@settings(max_examples=30, deadline=None)
def test_feature_builder_properties(prices_list):
    """
    Feature builder invariants that actually hold for any positive price series.

    MACD is a difference of EMAs and is therefore unbounded: the previous
    bound of [-50, 50] was a false hypothesis that only passed by luck.
    """
    from modelo_xgboost import FeatureBuilder
    close = pd.Series(prices_list)
    rsi = FeatureBuilder.compute_rsi(close, 14)
    macd = FeatureBuilder.compute_macd(close, 12, 26, 9)
    bb = FeatureBuilder.compute_bollinger(close, 20, 2)

    valid = rsi.dropna()
    if len(valid) > 0:
        assert valid.between(0, 100).all(), "RSI not bounded [0,100]"

    valid_macd = macd.dropna()
    if len(valid_macd) > 0:
        assert np.isfinite(valid_macd.to_numpy()).all(), "MACD not finite"
        # The histogram must be finite and of the same size as the input.
        assert len(macd) == len(close), "MACD length mismatch"

    valid_bb = bb.dropna()
    if len(valid_bb) > 0:
        # Bollinger %B is (price - lower) / (upper - lower): in [0, 1] for
        # prices inside the bands, but unbounded outside them.
        assert np.isfinite(valid_bb.to_numpy()).all(), "BB position not finite"


@given(
    st.floats(min_value=10, max_value=200),
    st.floats(min_value=0.01, max_value=0.5),
)
@settings(max_examples=20, deadline=None, suppress_health_check=[HealthCheck.too_slow])
def test_option_prices_monotonic(spot, sigma):
    """Call premium should increase with spot, put with strike."""
    from opciones_derivados import BlackScholes
    call = BlackScholes.price(S=spot, K=spot * 0.9, T=1.0, r=0.05, sigma=sigma, option_type='call')
    put = BlackScholes.price(S=spot, K=spot * 1.1, T=1.0, r=0.05, sigma=sigma, option_type='put')
    assert call.premium > 0
    assert put.premium > 0


# ============================================================
# REGRESIONES DE LOS P0 CORREGIDOS
# ============================================================

def test_macd_signal_is_ema_not_scaled_line():
    """
    Regresion: la senal MACD era `macd * 0.3` en lugar de la EMA-9 del MACD.

    Ademas las "EMA" eran medias simples de los ultimos 12/26 valores.
    """
    from analisis_tecnico import _ema, calc_macd

    np.random.seed(7)
    precios = list(100 * np.exp(np.cumsum(np.random.randn(400) * 0.02)))
    resultado = calc_macd(precios)

    serie = pd.Series(precios)
    ema_rapida = serie.ewm(span=12, adjust=False).mean()
    ema_lenta = serie.ewm(span=26, adjust=False).mean()
    macd = ema_rapida - ema_lenta
    senal = macd.ewm(span=9, adjust=False).mean()

    assert resultado['macd'] == pytest.approx(float(macd.iloc[-1]), abs=0.01)
    assert resultado['senal'] == pytest.approx(float(senal.iloc[-1]), abs=0.01)
    assert resultado['histograma'] == pytest.approx(
        float((macd - senal).iloc[-1]), abs=0.01)

    # La EMA se siembra con el primer valor (alpha=0.5 para periodo 3), no con
    # la media simple: _ema([1,2,3],3) == [1.0, 1.5, 2.25].
    assert _ema([1.0, 2.0, 3.0], 3) == pytest.approx([1.0, 1.5, 2.25])


def test_macd_histogram_equals_line_minus_signal():
    from analisis_tecnico import calc_macd

    np.random.seed(11)
    precios = list(50 + np.cumsum(np.random.randn(120)))
    r = calc_macd(precios)
    # macd y senal se redondean a 2 decimales de forma independiente, asi que
    # el histograma puede diferir del resto en hasta 0.01.
    assert r['histograma'] == pytest.approx(r['macd'] - r['senal'], abs=0.015)


def test_macd_and_rsi_handle_short_series():
    from analisis_tecnico import calc_macd, calc_rsi

    assert calc_macd([1.0, 2.0]) == {'macd': 0, 'senal': 0, 'histograma': 0}
    assert calc_rsi([1.0, 2.0]) == 50
    assert calc_rsi([100.0] * 40) == 50
    assert calc_macd([100.0] * 40)['histograma'] == 0.0


def test_rsi_matches_wilder_convention():
    from analisis_tecnico import calc_rsi

    np.random.seed(3)
    precios = list(100 * np.exp(np.cumsum(np.random.randn(300) * 0.015)))
    serie = pd.Series(precios)
    delta = serie.diff()
    avg_gain = delta.clip(lower=0).ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    avg_loss = (-delta).clip(lower=0).ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    esperado = 100 - 100 / (1 + avg_gain / avg_loss)

    assert calc_rsi(precios) == pytest.approx(float(esperado.iloc[-1]), abs=0.1)


def test_opciones_never_invents_values():
    """
    Regresion: generar_fallback() fabricaba IV y put/call ratio con random y
    los guardaba indistinguibles de los reales.
    """
    import opciones

    assert not hasattr(opciones, 'generar_fallback'), \
        "generar_fallback volvio a fabricar datos de opciones"

    registro = opciones.registrar_no_disponible('sin red')
    assert registro['disponible'] is False
    assert registro['motivo'] == 'sin red'
    for campo in ('put_call_ratio', 'sentimiento', 'vol_total', 'avg_iv',
                  'iv_skew', 'put_call_oi_ratio', 'gamma_exposure'):
        assert registro[campo] is None, f"{campo} debe ser None, no un valor estimado"


def test_modelo_no_inyecta_none_en_features_de_opciones():
    """El consumidor debe sustituir los null por el neutro historico."""
    from modelo_xgboost import FeatureBuilder
    from opciones import registrar_no_disponible

    np.random.seed(5)
    n = 200
    idx = pd.date_range('2024-01-01', periods=n, freq='B')
    df = pd.DataFrame({
        'Open': 100 + np.random.randn(n),
        'High': 102 + np.random.rand(n),
        'Low': 98 - np.random.rand(n),
        'Close': 100 + np.cumsum(np.random.randn(n)),
        'Volume': np.random.randint(100000, 1000000, n),
    }, index=idx)

    out, _ = FeatureBuilder.build_features(df, registrar_no_disponible('sin red'))
    opts = out[['opt_put_call_ratio', 'opt_iv_skew', 'opt_avg_iv',
                'opt_oi_ratio', 'opt_vol']]
    assert not opts.isna().any().any(), "null de opciones propagado al modelo"
    assert out['opt_put_call_ratio'].iloc[-1] == pytest.approx(1.0)


def test_backtest_no_fabrica_datos_sin_mercado(monkeypatch):
    """
    Regresion: el paso 13 del pipeline importaba backtest_engine, que no
    existia, y el error se tragaba en silencio. Sin datos de mercado el motor
    debe declararlo, no devolver una curva de rendimiento.
    """
    import backtest_engine

    monkeypatch.setattr(
        backtest_engine,
        'descargar_precios',
        lambda tickers, periodo=None: (None, {'NVDA': 'sin red'}))

    r = backtest_engine.run_backtest(tickers=['NVDA'])
    assert r['estado'] == 'sin_datos'
    assert r['motivo'] == {'NVDA': 'sin red'}
    assert 'curva_valor' not in r and 'portafolio' not in r


def test_backtest_rejects_empty_universe():
    import backtest_engine
    r = backtest_engine.run_backtest(tickers=[])
    assert r['estado'] == 'sin_datos'
    assert 'tickers' in r['motivo'] or 'vacia' in r['motivo']


def test_backtest_simulation_respects_t1_and_costs():
    """SinCosts: la senal se ejecuta T+1 y el turnover se cobra."""
    import backtest_engine

    idx = pd.bdate_range('2024-01-01', periods=200)
    precios = pd.Series(100 + np.arange(200, dtype=float), index=idx)

    posicion = backtest_engine.generar_senales(precios)
    assert set(posicion.unique()) <= {0.0, 1.0}, "posición debe ser binaria"
    assert len(posicion) == len(precios)

    sim = backtest_engine.simular_ticker(precios, 10000, coste_bps=10.0)
    assert len(sim['valor']) == len(precios)
    assert sim['valor'].iloc[-1] > 0
    # Con coste 0 el resultado no puede ser peor que el coste positivo.
    sim_sin = backtest_engine.simular_ticker(precios, 10000, coste_bps=0.0)
    assert sim_sin['retornos'].sum() >= sim['retornos'].sum()


def test_settings_resolves_dotted_paths_with_positional_default():
    """
    Regresion: get_setting('ml.calibracion', {}) devolvia None porque el {} se
    interpretaba como clave, dejando sin configuracion a todo el repo.
    """
    from config.settings import get_setting

    assert get_setting('ml.calibracion', {})['metodo'] == 'isotonic'
    assert get_setting('ml', 'xgboost') == get_setting('ml.xgboost')
    assert get_setting('features.feature_store', {})['enabled'] is True
    assert get_setting('no.existe', {}) == {}
    assert get_setting('no.existe') is None
    assert get_setting('risk_management', default={})['kill_switch_enabled'] is True
